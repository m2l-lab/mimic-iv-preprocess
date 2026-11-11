"""Paired T1 -> T2 prediction changes stratified by actual exposure to screened text (test split).

Aggregate AUROC can hide changes concentrated in the few admissions whose input actually changed.
Strata (from the screening log, not from labels):
  none              no sentence removed by T2
  post_anchor_only  only post-admission rules removed sentences
  outcome           at least one outcome-rule sentence removed
For the frozen CORe model a further split: whether the removal changed its 382-token input window,
approximated by |Δp| > 1e-5 (identical inputs differ by < 1e-6 under bucket padding; checked on 'none').
Also by the section(s) with removals. Per stratum and outcome: mean / median Δp, share decreased, and
the mean paired change in log loss with a patient-clustered bootstrap CI. Subgroup AUROC is not
reported (too few deaths in exposed strata).

Writes (public): exposure_stratified.csv, exposure_summary.json
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bootstrap_evaluation import patient_weights
from common import load_config, results_dir, write_json

EPS = 1e-5


def logloss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def summarise(d: pd.DataFrame, n_rep: int, seed: int) -> dict:
    dll = d.dll.values
    rng = np.random.default_rng(seed)
    reps = [np.sum(w * dll) / w.sum() for w in (patient_weights(d.subject_id.values, rng) for _ in range(n_rep))]
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"n": len(d), "mean_delta_p": d.dp.mean(), "median_delta_p": d.dp.median(),
            "mean_abs_delta_p": d.dp.abs().mean(), "pct_decreased": 100 * (d.dp < 0).mean(),
            "mean_delta_logloss": dll.mean(), "delta_logloss_ci_low": lo, "delta_logloss_ci_high": hi}


def main():
    cfg = load_config()
    priv, bc = cfg["paths"]["private_dir"], cfg["bootstrap"]
    cohort = pd.read_parquet(priv / "cohort.parquet", columns=["study_id", "subject_id", "label"])
    log = pd.read_parquet(priv / "screening_log.parquet")
    log = log[log.removed]
    per = log.groupby("study_id").agg(cats=("category", lambda c: set(c)), secs=("section", lambda s: set(s)))
    rows, summary = [], {}
    for model in ["core", "tfidf"]:
        pred = pd.read_parquet(priv / f"predictions_{model}.parquet")
        w = pred[pred.condition.isin(["T1_ROBUST", "T2_SCREENED"])].pivot(
            index="study_id", columns="condition", values="p_mortality").reset_index()
        w = w.merge(cohort, on="study_id").merge(per, on="study_id", how="left")
        w["exposure"] = np.where(w.cats.isna(), "none",
                                 np.where(w.cats.map(lambda c: isinstance(c, set) and "outcome" in c), "outcome",
                                          "post_anchor_only"))
        w["section_group"] = w.secs.map(lambda s: "none" if not isinstance(s, set) else
                                        "multiple" if len(s) > 1 else
                                        {"physical_exam": "physical_exam",
                                         "history_of_present_illness": "hpi"}.get(next(iter(s)), "other"))
        w["dp"] = w.T2_SCREENED - w.T1_ROBUST
        w["dll"] = logloss(w.label.values, w.T2_SCREENED.values) - logloss(w.label.values, w.T1_ROBUST.values)
        if model == "core":
            w["in_window"] = w.dp.abs() > EPS
            summary["core_pct_none_with_abs_dp_gt_eps"] = float(100 * w.loc[w.exposure == "none", "in_window"].mean())
            summary["core_n_exposed_in_window"] = int((w.in_window & (w.exposure != "none")).sum())
        strata = [("exposure", w.exposure), ("section", w.section_group)]
        if model == "core":
            strata.append(("exposure_in_core_window",
                           w.exposure.where(w.exposure == "none",
                                            w.exposure + np.where(w.in_window, "_in_window", "_truncated_away"))))
        for kind, key in strata:
            for (stratum, lab), d in w.groupby([key, w.label]):
                rows.append({"model": model, "stratification": kind, "stratum": stratum,
                             "outcome": "died" if lab == 1 else "survived",
                             **summarise(d, bc["n_replicates"], bc["seed"])})
    out = pd.DataFrame(rows)
    res = results_dir(cfg)
    out.to_csv(res / "exposure_stratified.csv", index=False)
    write_json(summary, res / "exposure_summary.json")
    cols = ["model", "stratification", "stratum", "outcome", "n", "mean_delta_p", "pct_decreased",
            "mean_delta_logloss", "delta_logloss_ci_low", "delta_logloss_ci_high"]
    print(out[cols].round(5).to_string(index=False))
    print(summary)


if __name__ == "__main__":
    main()
