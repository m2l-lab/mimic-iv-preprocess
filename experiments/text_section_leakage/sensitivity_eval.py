"""Compare mortality metrics using original and discharge-location-adjusted labels.

Existing TF-IDF and frozen CORe predictions are evaluated using both labels.
The TF-IDF T1/T2_LABEL_DIEDLOC conditions also compare retraining with the
adjusted labels. Requires those conditions to have been trained first.

Writes aggregate sensitivity_label_eval.csv, sensitivity_label_train.csv,
and sensitivity_summary.json files.
"""
from __future__ import annotations

import pandas as pd

from bootstrap_evaluation import load_wide, paired_bootstrap
from common import load_config, mimic_path, results_dir, write_json

MAIN = ["T0_COPIV", "T1_ROBUST", "T2_SCREENED", "T3_RETROSPECTIVE", "T4_OTHERPATIENT"]
KEEP = ["condition", "n_deaths", "auroc", "auroc_ci_low", "auroc_ci_high", "delta_auroc_vs_ref",
        "delta_auroc_ci_low", "delta_auroc_ci_high", "auprc", "auprc_ci_low", "auprc_ci_high",
        "delta_auprc_vs_ref", "delta_auprc_ci_low", "delta_auprc_ci_high", "brier"]


def corrected_label(cfg, wide: pd.DataFrame) -> pd.Series:
    priv = cfg["paths"]["private_dir"]
    c = pd.read_parquet(priv / "cohort.parquet", columns=["study_id", "hadm_id"])
    a = pd.read_csv(mimic_path(cfg, "admissions_csv"), usecols=["hadm_id", "discharge_location"])
    died = wide[["study_id"]].merge(c, on="study_id", how="left").merge(a, on="hadm_id", how="left")
    return (wide.label.astype(bool).values | died.discharge_location.eq("DIED").values).astype(int)


def labels(cfg, bc) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    priv = cfg["paths"]["private_dir"]
    parts = []
    for model in ["tfidf", "core"]:
        wide = load_wide(cfg, model)
        wide["label_diedloc"] = corrected_label(cfg, wide)
        conds = [c for c in MAIN if c in wide]
        for name, col in [("original", "label"), ("corrected", "label_diedloc")]:
            t = paired_bootstrap(wide, conds, "T1_ROBUST", bc["n_replicates"], bc["seed"], label_col=col)[KEEP]
            t.insert(0, "label", name)
            t.insert(0, "model", model)
            parts.append(t)
    evaluation = pd.concat(parts, ignore_index=True)

    fix = pd.read_parquet(priv / "predictions_tfidf_labelfix.parquet")
    wide = load_wide(cfg, "tfidf")
    wide["label_diedloc"] = corrected_label(cfg, wide)
    lf = fix.pivot(index="study_id", columns="condition", values="p_mortality").reset_index()
    w = wide[["study_id", "subject_id", "label", "label_diedloc", "T1_ROBUST", "T2_SCREENED"]].merge(lf, on="study_id")
    check = fix[fix.condition == "T1_LABEL_DIEDLOC"].set_index("study_id").label_used
    assert (w.set_index("study_id").label_diedloc == check.reindex(w.study_id).values).all()
    rows = []
    for base, fixed in [("T1_ROBUST", "T1_LABEL_DIEDLOC"), ("T2_SCREENED", "T2_LABEL_DIEDLOC")]:
        t = paired_bootstrap(w, [base, fixed], base, bc["n_replicates"], bc["seed"], label_col="label_diedloc")
        rows.append(t[KEEP])
    training = pd.concat(rows, ignore_index=True)
    summary = {"test_relabelled": int((wide.label_diedloc != wide.label).sum()),
               "test_deaths_original": int(wide.label.sum()), "test_deaths_corrected": int(wide.label_diedloc.sum())}
    return evaluation, training, summary


def main():
    cfg = load_config()
    bc = cfg["bootstrap"]
    out = results_dir(cfg)
    ev, tr, summary = labels(cfg, bc)
    ev.to_csv(out / "sensitivity_label_eval.csv", index=False)
    tr.to_csv(out / "sensitivity_label_train.csv", index=False)
    write_json({"labels": summary, "n_replicates": bc["n_replicates"], "seed": bc["seed"],
                "resampling_unit": "subject_id"}, out / "sensitivity_summary.json")
    pd.set_option("display.width", 200)
    print(summary)
    print(ev[["model", "label", "condition", "n_deaths", "auroc", "auprc"]].round(4).to_string(index=False))
    print(tr[["condition", "auroc", "delta_auroc_vs_ref", "auprc", "delta_auprc_vs_ref"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
