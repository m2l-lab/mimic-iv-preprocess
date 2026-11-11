"""Phase 1: fixed in-hospital mortality cohort with patient-disjoint splits.

Population: MIMIC-IV hospital admissions with a usable retrospective discharge
summary (MIMIC-IV-Note ``discharge``) and a non-missing ``hospital_expire_flag``.

Outputs
  private: cohort.parquet  (study_id, subject_id, hadm_id, note_id, label, split, ...)
           split_manifest.csv (identifier-only)
  public:  results/cohort_summary.json (counts only)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.csv as pacsv

from common import load_config, mimic_path, results_dir, write_json


def read_discharge_notes(cfg, columns=None) -> pd.DataFrame:
    opts = pacsv.ParseOptions(newlines_in_values=True)
    conv = pacsv.ConvertOptions(include_columns=columns) if columns else None
    return pacsv.read_csv(mimic_path(cfg, "discharge_csv"), parse_options=opts, convert_options=conv).to_pandas()


def patient_split(subject_ids: np.ndarray, fractions: dict, seed: int) -> dict:
    subjects = np.sort(np.unique(subject_ids))
    perm = np.random.default_rng(seed).permutation(subjects)
    n_train = int(round(fractions["train"] * len(perm)))
    n_val = int(round(fractions["val"] * len(perm)))
    split = {s: "train" for s in perm[:n_train]}
    split.update({s: "val" for s in perm[n_train:n_train + n_val]})
    split.update({s: "test" for s in perm[n_train + n_val:]})
    return split


def main():
    cfg = load_config()
    priv = cfg["paths"]["private_dir"]
    steps = []

    notes = read_discharge_notes(cfg, ["note_id", "subject_id", "hadm_id", "note_type", "note_seq",
                                       "charttime", "storetime", "text"])
    steps.append(("discharge notes (MIMIC-IV-Note discharge)", len(notes), notes.hadm_id.nunique()))
    notes = notes[notes.note_type == "DS"]
    notes = notes[notes.text.notna() & (notes.text.str.strip() != "")]
    steps.append(("note_type DS with non-empty text", len(notes), notes.hadm_id.nunique()))

    # Note-selection policy: one discharge summary per admission. If several exist,
    # keep the latest storetime (then highest note_seq). Independent of the label.
    n_multi = int((notes.groupby("hadm_id").size() > 1).sum())
    notes = (notes.sort_values(["hadm_id", "storetime", "note_seq"])
                  .drop_duplicates("hadm_id", keep="last"))

    adm = pd.read_csv(mimic_path(cfg, "admissions_csv"),
                      usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
                               "hospital_expire_flag", "admission_type"])
    pat = pd.read_csv(mimic_path(cfg, "patients_csv"), usecols=["subject_id", "anchor_year_group"])
    df = notes.drop(columns=["text"]).merge(adm, on="hadm_id", how="left", suffixes=("", "_adm"))
    unmatched = int(df.hospital_expire_flag.isna().sum())
    subj_mismatch = int((df.subject_id_adm.notna() & (df.subject_id != df.subject_id_adm)).sum())
    df = df[df.hospital_expire_flag.notna() & (df.subject_id == df.subject_id_adm)].drop(columns="subject_id_adm")
    steps.append(("linked to admissions with valid hospital_expire_flag", len(df), df.hadm_id.nunique()))
    df = df.merge(pat, on="subject_id", how="left")

    df["label"] = df.hospital_expire_flag.astype(int)
    split = patient_split(df.subject_id.values, cfg["split"], cfg["seed"])
    df["split"] = df.subject_id.map(split)
    df = df.sort_values("hadm_id").reset_index(drop=True)
    df["study_id"] = [f"A{i:06d}" for i in range(len(df))]

    assert df.groupby("subject_id").split.nunique().max() == 1, "patient leakage across splits"
    assert df.hadm_id.is_unique

    cols = ["study_id", "subject_id", "hadm_id", "note_id", "label", "split", "admittime", "dischtime",
            "deathtime", "charttime", "storetime", "admission_type", "anchor_year_group"]
    df[cols].to_parquet(priv / "cohort.parquet", index=False)
    df[["study_id", "subject_id", "hadm_id", "note_id", "split"]].to_csv(priv / "split_manifest.csv", index=False)

    summary = {
        "population": "MIMIC-IV hospital admissions with a usable retrospective discharge summary",
        "label": "admissions.hospital_expire_flag (in-hospital death; agrees with deathtime)",
        "versions": [cfg["data"]["mimic_hosp_version"], cfg["data"]["mimic_note_version"]],
        "attrition": [{"step": s, "n_notes": int(n), "n_admissions": int(h)} for s, n, h in steps],
        "admissions_with_multiple_discharge_notes": n_multi,
        "notes_unmatched_to_admissions": unmatched,
        "notes_subject_mismatch": subj_mismatch,
        "split_seed": cfg["seed"],
        "by_split": {
            s: {"admissions": int(len(g)), "patients": int(g.subject_id.nunique()),
                "deaths": int(g.label.sum()), "mortality_pct": round(100 * g.label.mean(), 3)}
            for s, g in df.groupby("split")
        },
        "total": {"admissions": int(len(df)), "patients": int(df.subject_id.nunique()),
                  "deaths": int(df.label.sum()), "mortality_pct": round(100 * df.label.mean(), 3)},
        "anchor_year_group": df.anchor_year_group.value_counts().sort_index().to_dict(),
    }
    write_json(summary, results_dir(cfg) / "cohort_summary.json")
    print({k: summary[k] for k in ["total", "by_split", "admissions_with_multiple_discharge_notes"]})


if __name__ == "__main__":
    main()
