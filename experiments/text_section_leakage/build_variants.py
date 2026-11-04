"""Phase 2: build the text variants for every cohort admission.

  T0_COPIV          verbatim COP-IV admission-section extraction
  T1_ROBUST         robust admission-related section extraction (reference)
  T2_SCREENED       T1 minus sentences flagged by the frozen screening rules
  T3_RETROSPECTIVE  T1 + later sections of the same note (positive control)
  T4_OTHERPATIENT   T1 + the same later sections from another patient of the
                    same split (exploratory negative control, outcome-blind)

All variants derive from the same discharge summary and the same admission.
Private outputs: variants.parquet (texts), screening_log.parquet (offsets only).
Public outputs:  results/variant_summary.json, results/section_comparison.csv
"""
from __future__ import annotations

import re
from multiprocessing import Pool

import numpy as np
import pandas as pd

from build_cohort import read_discharge_notes
from common import ADMISSION_SECTIONS, INJECTION_LABELS, load_config, results_dir, write_json
from copiv_original import filter_notes
from extract_sections import admission_spans, clean, later_sections, parse_note, render
from screen_retrospective_content import RULESET_VERSION, detect, screen_text

SECTION_NAMES = [n for n, _ in ADMISSION_SECTIONS]
CORE_SECTIONS = ["chief_complaint", "history_of_present_illness", "past_medical_history"]
T0_SPLIT_RE = re.compile(r"(?:^|\n\n)(" + "|".join(re.escape(l) for _, l in ADMISSION_SECTIONS) + r"): ")
LABEL_TO_NAME = {label: name for name, label in ADMISSION_SECTIONS}
INJ_NAMES: list[str] = load_config(create_dirs=False)["injection_sections"]


def split_t0(text: str) -> dict[str, str]:
    parts = T0_SPLIT_RE.split(text)
    return {LABEL_TO_NAME[parts[i]]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def compare(t0: str, t1: str) -> str:
    a, b = clean(t0).lower(), clean(t1).lower()
    if not a and not b:
        return "both_empty"
    if a == b:
        return "identical"
    if not a:
        return "only_t1"
    if not b:
        return "only_t0"
    if b in a:
        return "t0_superset"
    if a in b:
        return "t0_subset"
    return "different"


def process_chunk(rows: list[tuple[str, int, str]]):
    ids, subjects, texts = zip(*rows)
    t0_series = filter_notes(pd.DataFrame({"subject_id": subjects, "text": texts}),
                             admission_text_only=True)
    out, log = [], []
    for k, (sid, text) in enumerate(zip(ids, texts)):
        rec = {"study_id": sid}
        t0 = t0_series.get(k)
        rec["t0_retained"] = t0 is not None
        rec["t0_text"] = t0
        t0_secs = split_t0(t0) if t0 is not None else {}

        parsed = parse_note(text)
        spans, meta = admission_spans(text, parsed)
        rec.update(meta)
        t1_secs, t2_secs = {}, {}
        for name in SECTION_NAMES:
            raw_parts, kept_parts = [], []
            for a, b in spans[name]:
                raw_parts.append(text[a:b])
                kept, matches = screen_text(name, text[a:b])
                kept_parts.append(kept)
                log.extend((sid, name, a + m.start, a + m.end, m.rule_id, m.category, m.removed, m.reason)
                           for m in matches)
            t1_secs[name] = clean(" ".join(raw_parts))
            t2_secs[name] = clean(" ".join(kept_parts))
            rec[f"t1_chars_{name}"] = len(t1_secs[name])
            rec[f"t2_chars_{name}"] = len(t2_secs[name])
            rec[f"t0_chars_{name}"] = len(t0_secs.get(name, ""))
            rec[f"cmp_{name}"] = compare(t0_secs.get(name, ""), t1_secs[name]) if t0 is not None else "t0_dropped"
            # automated candidates in T0 (detection only, section-aware exemptions)
            rec[f"t0_cand_{name}"] = len(detect(t0_secs.get(name, ""), name)) if t0 is not None else 0
        rec["t1_text"] = render(t1_secs)
        rec["t2_text"] = render(t2_secs)
        rec["t1_core_nonempty"] = any(t1_secs[n] for n in CORE_SECTIONS)

        inj = later_sections(text, INJ_NAMES, parsed)
        rec["inj_text"] = render(inj, [(n, INJECTION_LABELS[n]) for n in INJ_NAMES if inj[n]])
        for n in INJ_NAMES:
            rec[f"inj_chars_{n}"] = len(inj[n])
            rec[f"inj_cand_{n}"] = len(detect(inj[n], n))
        out.append(rec)
    return out, log


def assign_negative_control(df: pd.DataFrame, seed: int) -> pd.Series:
    """Donor admission for T4: next admission in a seeded within-split permutation,
    skipping donors from the same patient. Labels are never consulted."""
    donor = pd.Series(index=df.index, dtype=object)
    rng = np.random.default_rng(seed)
    for _, g in df.groupby("split"):
        idx = rng.permutation(g.index.values)
        subj = df.loc[idx, "subject_id"].values
        n = len(idx)
        for i in range(n):
            j = (i + 1) % n
            while subj[j] == subj[i]:
                j = (j + 1) % n
            donor[idx[i]] = df.at[idx[j], "study_id"]
    return donor


def summarize(df: pd.DataFrame, log: pd.DataFrame, cfg: dict) -> None:
    res = results_dir(cfg)
    rows = []
    for name in SECTION_NAMES:
        t0_ok = df[df.t0_retained]
        cmp_counts = t0_ok[f"cmp_{name}"].value_counts(normalize=True).mul(100).round(2).to_dict()
        rows.append({
            "section": name,
            "t0_present_pct": round(100 * (t0_ok[f"t0_chars_{name}"] > 0).mean(), 2),
            "t1_present_pct": round(100 * (df[f"t1_chars_{name}"] > 0).mean(), 2),
            "t0_median_chars": float(t0_ok[f"t0_chars_{name}"].median()),
            "t1_median_chars": float(df[f"t1_chars_{name}"].median()),
            "t2_median_chars": float(df[f"t2_chars_{name}"].median()),
            **{f"t0_vs_t1_{k}_pct": v for k, v in cmp_counts.items()},
        })
    pd.DataFrame(rows).to_csv(res / "section_comparison.csv", index=False)

    removed = log[log.removed]
    by_rule = removed.groupby(["rule_id", "category"]).study_id.nunique().rename("n_admissions").reset_index()
    excluded = log[~log.removed].groupby(["rule_id", "reason"]).size().rename("n_matches").reset_index()
    summary = {
        "ruleset_version": RULESET_VERSION,
        "n_admissions": int(len(df)),
        "t0_retained": int(df.t0_retained.sum()),
        "t1_core_nonempty": int(df.t1_core_nonempty.sum()),
        "analysis_cohort": int(df.in_analysis.sum()),
        "pe_status_pct": df.pe_status.value_counts(normalize=True).mul(100).round(2).to_dict(),
        "notes_with_out_of_order_admission_section": int((df.n_out_of_order > 0).sum()),
        "notes_with_repeated_admission_section": int((df.n_repeated > 0).sum()),
        "injection_section_present_pct": {
            n: round(100 * (df[f"inj_chars_{n}"] > 0).mean(), 2) for n in INJ_NAMES},
        "t2_admissions_with_any_removal": int(removed.study_id.nunique()),
        "t2_removed_sentences_by_rule_admissions": by_rule.to_dict("records"),
        "screening_excluded_matches": excluded.to_dict("records"),
    }
    write_json(summary, res / "variant_summary.json")
    print({k: v for k, v in summary.items() if not isinstance(v, list)})


def main():
    cfg = load_config()
    priv = cfg["paths"]["private_dir"]
    cohort = pd.read_parquet(priv / "cohort.parquet")
    notes = read_discharge_notes(cfg, ["note_id", "text"])
    notes = cohort[["study_id", "subject_id", "note_id"]].merge(notes, on="note_id", how="left")
    assert notes.text.notna().all()
    rows = list(zip(notes.study_id, notes.subject_id, notes.text))
    del notes
    chunks = [rows[i:i + 2000] for i in range(0, len(rows), 2000)]

    recs, logs = [], []
    with Pool(10) as pool:
        for k, (r, l) in enumerate(pool.imap(process_chunk, chunks)):
            recs.extend(r)
            logs.extend(l)
            if k % 20 == 0:
                print(f"chunk {k}/{len(chunks)}", flush=True)
    df = cohort[["study_id", "subject_id", "split", "label"]].merge(pd.DataFrame(recs), on="study_id")
    df["in_analysis"] = df.t0_retained & df.t1_core_nonempty
    df["t4_donor"] = assign_negative_control(df[df.in_analysis], cfg["seed"])
    df["ruleset_version"] = RULESET_VERSION
    df.drop(columns=["subject_id", "label"]).to_parquet(priv / "variants.parquet", index=False, compression="zstd")
    log = pd.DataFrame(logs, columns=["study_id", "section", "note_start", "note_end", "rule_id", "category",
                                      "removed", "reason"])
    log.to_parquet(priv / "screening_log.parquet", index=False)
    summarize(df, log, cfg)


if __name__ == "__main__":
    main()
