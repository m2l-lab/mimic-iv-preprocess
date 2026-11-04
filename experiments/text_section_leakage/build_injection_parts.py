"""Per-section injection texts for the section-specific positive controls (restricted output).

For every analysis admission:
  dcond, ddiag, bhc   Discharge Condition, Discharge Diagnosis, Brief Hospital Course (as in T3)
  bhc_noout           Brief Hospital Course without sentences matched by the *outcome* rules (O1-O5;
                      relative-death exemption as in T2). Post-admission rules are not applied: the
                      whole course is post-admission by definition.
  t4m_donor           donor for the outcome-matched shuffled control: another patient of the same split
                      *and the same in-hospital outcome* (diagnostic oracle; uses labels by design)

Writes (private): injection_parts.parquet; (public): injection_parts_summary.json
"""
from __future__ import annotations

from multiprocessing import Pool

import pandas as pd

from build_cohort import read_discharge_notes
from build_variants import assign_negative_control
from common import load_config, results_dir, write_json
from extract_sections import clean, later_sections, parse_note
from screen_retrospective_content import screen_text

NAMES = ["discharge_condition", "discharge_diagnosis", "brief_hospital_course"]
OUTCOME = frozenset({"outcome"})


def process_chunk(rows):
    out = []
    for sid, text in rows:
        parsed = parse_note(text)
        inj = later_sections(text, NAMES, parsed)
        kept, matches = screen_text("brief_hospital_course", inj["brief_hospital_course"], OUTCOME)
        out.append({"study_id": sid, "dcond": inj["discharge_condition"], "ddiag": inj["discharge_diagnosis"],
                    "bhc": inj["brief_hospital_course"], "bhc_noout": clean(kept),
                    "bhc_outcome_sentences_removed": len({(m.start, m.end) for m in matches if m.removed})})
    return out


def main():
    cfg = load_config()
    priv = cfg["paths"]["private_dir"]
    v = pd.read_parquet(priv / "variants.parquet", columns=["study_id", "in_analysis"])
    cohort = pd.read_parquet(priv / "cohort.parquet", columns=["study_id", "subject_id", "note_id", "split", "label"])
    cohort = cohort[cohort.study_id.isin(v.study_id[v.in_analysis])].reset_index(drop=True)
    notes = cohort[["study_id", "note_id"]].merge(read_discharge_notes(cfg, ["note_id", "text"]), on="note_id")
    rows = list(zip(notes.study_id, notes.text))
    del notes
    recs = []
    with Pool(10) as pool:
        for r in pool.imap(process_chunk, [rows[i:i + 2000] for i in range(0, len(rows), 2000)]):
            recs.extend(r)
    df = cohort.merge(pd.DataFrame(recs), on="study_id")
    strata = df[["study_id", "subject_id"]].assign(split=df.split + "_" + df.label.astype(str))
    df["t4m_donor"] = assign_negative_control(strata, cfg["seed"] + 1)
    df.drop(columns=["subject_id", "note_id", "split", "label"]).to_parquet(
        priv / "injection_parts.parquet", index=False, compression="zstd")
    s = {"n": len(df)}
    for lab, name in [(1, "died"), (0, "survived")]:
        d = df[df.label == lab]
        s[name] = {"pct_bhc_present": round(100 * (d.bhc.str.len() > 0).mean(), 2),
                   "pct_bhc_with_outcome_sentence_removed": round(100 * (d.bhc_outcome_sentences_removed > 0).mean(), 2),
                   "median_bhc_chars": int(d.bhc.str.len().median()),
                   "median_bhc_noout_chars": int(d.bhc_noout.str.len().median())}
    write_json(s, results_dir(cfg) / "injection_parts_summary.json")
    print(s)


if __name__ == "__main__":
    main()
