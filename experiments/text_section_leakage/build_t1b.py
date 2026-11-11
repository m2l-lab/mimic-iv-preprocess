"""Parser diagnostics over the cohort, and the T1b / T2b text variants.

Per analysis-cohort admission:
  * T1b (extract_sections_t1b) and T2b (T1b minus sentences flagged by the frozen screening rules);
  * which T1b fixes fire and how the kept text differs from T1 (per section; death wording in the kept exam);
  * the T0 (COP-IV) failure mechanism of every admission section, relative to the T1b spans:
      missing_heading   the COP-IV heading string does not occur (de-identified or other wording);
      missing_unmatched the heading occurs but the regex finds no terminator;
      wrong_anchor      the match starts at another occurrence (e.g. inside "DISCHARGE PHYSICAL EXAM:");
      stop:<type>       content lost: the section ends at a blank line followed by a colon line
                        (type of that line: repeated_heading, ed, ros, transfer, workup, history_block,
                        exam_subsection, list_continuation, other);
      later_exam        the exam includes a discharge, death or later exam (COP-IV has no phase split);
      next_section      the match runs across a top-level heading into the next section;
      extra_other       other content beyond the T1b span.
The T0 offsets come from re-running the verbatim COP-IV regex; agreement with the stored T0 text is reported.

Private: variants_t1b.parquet (texts), parser_rows.parquet (per admission, no text).
Public:  results/parser_t0_mechanisms.csv, results/parser_t1b_fixes.csv, results/parser_cohort_summary.json
"""
from __future__ import annotations

import re
from multiprocessing import Pool

import numpy as np
import pandas as pd

from build_cohort import read_discharge_notes
from build_variants import split_t0
from common import ADMISSION_SECTIONS, load_config, results_dir, write_json
from extract_sections import admission_spans, clean, render
from extract_sections_t1b import admission_spans_t1b, parse_note_t1b, phase_of_line
from section_patterns import DEATH_RE
from screen_retrospective_content import screen_text

NAMES = [n for n, _ in ADMISSION_SECTIONS]
# verbatim from copiv_original.filter_admission_text
T0_HEADINGS = {"chief_complaint": "chief complaint:", "history_of_present_illness": "present illness:",
               "past_medical_history": "medical history:", "medications_on_admission": "medications on admission:",
               "allergies": "allergies:", "physical_exam": "physical exam:", "family_history": "family history:",
               "social_history": "social history:"}
T0_PATTERN = '{}([\\s\\S]+?)\n\\s*?\n[^(\\\\|\\d|\\.)]+?:'
_FH_FIX = re.compile(r"___\nFamily History:", re.IGNORECASE)
STOP_TYPES = [("repeated_heading", re.compile(r"^\W*(?:(?:micu|icu|ccu)\s+)?(?:hpi|history\s+of\s+present(?:ing)?\s+"
                                              r"illness|physical\s+exam(?:ination)?|past\s+medical\s+history)\b",
                                              re.IGNORECASE)),
              ("ed", re.compile(r"\b(?:ED|ER|emergency|triage)\b", re.IGNORECASE)),
              ("ros", re.compile(r"\b(?:ROS|review\s+of\s+systems)\b", re.IGNORECASE)),
              ("transfer", re.compile(r"\b(?:transfer|on\s+arrival|upon\s+arrival|arrival)\b", re.IGNORECASE)),
              ("workup", re.compile(r"\b(?:labs?|ABG|VBG|notable\s+for|vital\s+signs|VS\s+were|imaging|u/s|CT|CXR|"
                                    r"EKG|ECG|MRI)\b", re.IGNORECASE)),
              ("history_block", re.compile(r"^\W*(?:past|prior|other|surg\w*|psh\w*|pmh\w*|ob\b|gyn\w*|pob\w*|"
                                           r"pgyn\w*|onc\w*|psych\w*|cardiac|medical|treatment)", re.IGNORECASE)),
              ("exam_subsection", re.compile(r"^\W*(?:vs|vitals?|v/s|gen\w*|heent|neck|cv|cardiac|cardio\w*|lungs?|"
                                             r"pulm\w*|resp\w*|chest|abd\w*|ext\w*|skin|neuro\w*|psych\w*|gu|msk|"
                                             r"back|rectal|pulses?|mental|motor|sensory|reflexes|admission|on\s+adm|"
                                             r"opens|orientation|carotid|cn)\b", re.IGNORECASE)),
              ("list_continuation", re.compile(r"^\s*(?:[-#*•]|\d+[.)]|[a-z])"))]


def t0_spans(text: str) -> dict[str, tuple[tuple[int, int] | None, bool]]:
    """Original-text offsets of each COP-IV section match, and whether the heading string occurs at all."""
    lead = len(text) - len(text.lstrip())
    s = text.strip()
    ins = [m.start() + 4 for m in _FH_FIX.finditer(s)]          # an extra "\n" is inserted after "___\n"
    proc = _FH_FIX.sub("___\n\nFamily History:", s)

    def back(p: int) -> int:
        return p - sum(1 for k, q in enumerate(ins) if q + k < p) + lead

    out = {}
    for name, head in T0_HEADINGS.items():
        m = re.search(T0_PATTERN.format(head), proc, flags=re.IGNORECASE)
        present = re.search(re.escape(head), proc, flags=re.IGNORECASE) is not None
        if m and m.group(1).replace("\n", " ").strip().startswith("[]"):
            m = None
        out[name] = ((back(m.start(1)), back(m.end(1))) if m else None, present)
    return out


def _mask(n: int, spans) -> np.ndarray:
    m = np.zeros(n, bool)
    for a, b in spans:
        m[a:b] = True
    return m


def stop_type(text: str, end: int) -> str:
    line = next((ln for ln in text[end:end + 400].split("\n") if ln.strip()), "")
    return next((name for name, rx in STOP_TYPES if rx.search(line)), "other")


MIN_CHARS = 3       # alphanumeric characters; reviewers coded a lost or added last line as a boundary error


def later_content(text: str, a: int, b: int, extra: np.ndarray) -> bool:
    """True if the T0 exam span carries findings after a discharge/death/later/transfer heading line that
    T1b does not keep (a bare heading at the end of the span is not a later exam)."""
    for pos, ln in _line_offsets(text[a:b]):
        if extra[a + pos:a + pos + len(ln)].any() and (phase_of_line(ln) or ("",))[0] in ("discharge", "transfer"):
            after = a + pos + len(ln)
            heading_only = not phase_of_line(ln)[1].startswith(("death_line", "discharge_prose")) and \
                not re.search(r"[:\-,]\s*\S", ln)
            return extra[after if heading_only else a + pos:b].sum() >= MIN_CHARS
    return False


def t0_mechanisms(text: str, t0: dict, ref: dict, parsed, alnum: np.ndarray) -> dict[str, str]:
    """COP-IV failure mechanism per section; empty string = no failure detected. 't1b_missing' marks a T0
    section without any T1b counterpart (a failure of the robust parser, not of T0)."""
    n = len(text)
    heads = sorted(s.header_start for s in parsed.sections)
    out = {}
    for name in NAMES:
        span, present = t0[name]
        secs = parsed.by_name(name)
        ref_m = _mask(n, ref[name]) & alnum
        if span is None:
            out[name] = "" if not ref_m.any() else ("missing_unmatched" if present else "missing_heading")
            continue
        a, b = span
        t0_m = _mask(n, [span]) & alnum
        if not secs and not ref[name]:
            out[name] = "t1b_missing"
            continue
        stop_ref = ref_m if ref_m.any() else _mask(n, [(s.start, s.end) for s in secs]) & alnum
        anchor_ok = any(s.header_start <= a <= s.start for s in secs) or any(
            sa <= a <= text.find("\n", sa) for sa, _ in ref[name])
        tags = []
        if not anchor_ok:
            tags.append("wrong_anchor")
        elif (stop_ref & ~t0_m).sum() >= MIN_CHARS:
            tags.append("stop:" + stop_type(text, b))
        extra = t0_m & ~ref_m
        if extra.sum() >= MIN_CHARS:
            if name == "physical_exam" and later_content(text, a, b, extra):
                tags.append("later_exam")
            elif any(a < h < b for h in heads):
                tags.append("next_section")
            elif "wrong_anchor" not in tags and ref_m.any():
                tags.append("extra_other")
        out[name] = ";".join(tags)
    return out


def _line_offsets(s: str):
    pos = 0
    for ln in s.split("\n"):
        yield pos, ln
        pos += len(ln) + 1


def process(rows):
    out, texts = [], []
    for sid, text, t0_text in rows:
        parsed, _ = parse_note_t1b(text)
        t1, _ = admission_spans(text)
        t1b, meta = admission_spans_t1b(text)
        alnum = np.fromiter((c.isalnum() for c in text), bool, len(text))
        rec = {"study_id": sid, "pe_status_t1b": meta["pe_status"],
               **{k: meta[k] for k in ["F3_exam_alias", "F4_upper_heading", "F5_pmh_blocks", "F6_hpi_course"]},
               "pe_rules": ";".join(meta["pe_rules"])}
        t1b_secs, t2b_secs = {}, {}
        for name in NAMES:
            m1, m2 = _mask(len(text), t1[name]) & alnum, _mask(len(text), t1b[name]) & alnum
            rec[f"t1_only_{name}"] = int((m1 & ~m2).sum())
            rec[f"t1b_only_{name}"] = int((m2 & ~m1).sum())
            kept = [screen_text(name, text[a:b])[0] for a, b in t1b[name]]
            t1b_secs[name] = clean(" ".join(text[a:b] for a, b in t1b[name]))
            t2b_secs[name] = clean(" ".join(kept))
        rec["death_in_kept_pe_t1"] = sum(len(DEATH_RE.findall(text[a:b])) for a, b in t1["physical_exam"])
        rec["death_in_kept_pe_t1b"] = sum(len(DEATH_RE.findall(text[a:b])) for a, b in t1b["physical_exam"])
        if isinstance(t0_text, str):
            t0 = t0_spans(text)
            stored = split_t0(t0_text)
            rec["t0_regex_agrees"] = all(
                clean(text[sp[0]:sp[1]] if sp else "") == clean(stored.get(name, "")) for name, (sp, _) in t0.items())
            for name, mech in t0_mechanisms(text, t0, t1b, parsed, alnum).items():
                rec[f"t0_mech_{name}"] = mech
        out.append(rec)
        texts.append({"study_id": sid, "t1b_text": render(t1b_secs), "t2b_text": render(t2b_secs)})
    return out, texts


def main():
    cfg = load_config()
    priv, res = cfg["paths"]["private_dir"], results_dir(cfg)
    v = pd.read_parquet(priv / "variants.parquet", columns=["study_id", "in_analysis", "t0_text", "pe_status"])
    v = v[v.in_analysis]
    cohort = pd.read_parquet(priv / "cohort.parquet", columns=["study_id", "note_id", "label", "split"])
    cohort = cohort.merge(v, on="study_id")
    notes = read_discharge_notes(cfg, ["note_id", "text"]).merge(cohort[["study_id", "note_id", "t0_text"]], on="note_id")
    rows = list(zip(notes.study_id, notes.text, notes.t0_text))
    del notes
    chunks = [rows[i:i + 2000] for i in range(0, len(rows), 2000)]
    recs, texts = [], []
    with Pool(10) as pool:
        for k, (r, t) in enumerate(pool.imap(process, chunks)):
            recs.extend(r)
            texts.extend(t)
            if k % 20 == 0:
                print(f"chunk {k}/{len(chunks)}", flush=True)
    pd.DataFrame(texts).to_parquet(priv / "variants_t1b.parquet", index=False, compression="zstd")
    df = cohort.drop(columns=["t0_text", "note_id"]).merge(pd.DataFrame(recs), on="study_id")
    df.to_parquet(priv / "parser_rows.parquet", index=False)
    summarize(df, res)


def summarize(df: pd.DataFrame, res) -> None:
    t0 = df[df.t0_regex_agrees.notna()]
    rows = []
    for scope, g in [("all", t0), ("died", t0[t0.label == 1]), ("survived", t0[t0.label == 0])]:
        for name in NAMES:
            tags = g[f"t0_mech_{name}"].fillna("").str.split(";").explode()
            for tag, c in tags[tags != ""].value_counts().items():
                rows.append({"outcome": scope, "section": name, "mechanism": tag, "n": int(c),
                             "pct_admissions": round(100 * c / len(g), 3)})
        any_tags = g[[f"t0_mech_{n}" for n in NAMES]].fillna("").agg(";".join, axis=1)
        for tag in ["missing", "wrong_anchor", "stop:", "later_exam", "next_section", "extra_other"]:
            rows.append({"outcome": scope, "section": "any", "mechanism": tag.rstrip(":"),
                         "n": int(any_tags.str.contains(tag).sum()),
                         "pct_admissions": round(100 * any_tags.str.contains(tag).mean(), 3)})
    pd.DataFrame(rows).to_csv(res / "parser_t0_mechanisms.csv", index=False)

    fix_rows = []
    rules = df.pe_rules.fillna("").str.split(";")
    for scope, idx in [("all", df.index), ("died", df.index[df.label == 1]), ("survived", df.index[df.label == 0])]:
        g = df.loc[idx]
        row = {"outcome": scope, "n": int(len(g))}
        for f in ["F3_exam_alias", "F4_upper_heading", "F5_pmh_blocks", "F6_hpi_course"]:
            row[f"pct_{f}"] = round(100 * (g[f] > 0).mean(), 3)
        for r in ["discharge_heading", "discharge_prose", "death_heading", "death_line", "later_exam",
                  "transfer_exam", "admission_heading", "decoration_dropped"]:
            row[f"pct_pe_rule_{r}"] = round(100 * rules.loc[idx].apply(lambda x: r in x).mean(), 3)
        for name in NAMES:
            row[f"pct_changed_{name}"] = round(100 * ((g[f"t1_only_{name}"] > 0) | (g[f"t1b_only_{name}"] > 0)).mean(), 3)
        row["pct_pe_text_removed"] = round(100 * (g.t1_only_physical_exam > 0).mean(), 3)
        row["median_pe_chars_removed_if_any"] = float(g.loc[g.t1_only_physical_exam > 0, "t1_only_physical_exam"].median()) \
            if (g.t1_only_physical_exam > 0).any() else 0.0
        row["pct_death_wording_in_kept_pe_t1"] = round(100 * (g.death_in_kept_pe_t1 > 0).mean(), 3)
        row["pct_death_wording_in_kept_pe_t1b"] = round(100 * (g.death_in_kept_pe_t1b > 0).mean(), 3)
        fix_rows.append(row)
    pd.DataFrame(fix_rows).to_csv(res / "parser_t1b_fixes.csv", index=False)
    summary = {"n_admissions": int(len(df)), "n_t0_retained": int(len(t0)),
               "t0_regex_agreement_with_stored_t0": round(float(t0.t0_regex_agrees.mean()), 5),
               "pe_status_t1_vs_t1b": pd.crosstab(df.pe_status, df.pe_status_t1b).to_dict(),
               "note": "T0 mechanisms are relative to T1b spans; T1b fixes in parser_t1b_fixes.csv"}
    write_json(summary, res / "parser_cohort_summary.json")
    print({k: v for k, v in summary.items() if k != "pe_status_t1_vs_t1b"})
    print(pd.DataFrame(fix_rows).T.to_string())


if __name__ == "__main__":
    main()
