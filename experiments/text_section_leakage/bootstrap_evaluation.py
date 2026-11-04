"""Metrics and patient-clustered paired bootstrap.

Each replicate resamples *patients* with replacement; every admission of a drawn
patient enters with that patient's multiplicity. The multiplicities are used as
integer weights in exact weighted AUROC / AUPRC / Brier implementations, which
equal the metrics on the explicitly replicated data. All conditions are
evaluated on the same draws in the same replicate (paired differences).

CLI:  python bootstrap_evaluation.py core|tfidf
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import load_config, results_dir, write_json


class RankedScores:
    """Pre-sorted scores so that weighted metrics cost O(n) per replicate."""

    def __init__(self, scores: np.ndarray, y: np.ndarray):
        order = np.argsort(-scores, kind="mergesort")
        s = scores[order]
        self.order = order
        self.y = y[order].astype(float)
        self.p = scores[order]
        self.group = np.concatenate([[0], np.cumsum(s[1:] != s[:-1])])   # tie groups, descending score
        self.n_groups = int(self.group[-1]) + 1

    def metrics(self, w: np.ndarray) -> tuple[float, float, float]:
        w = w[self.order]
        tp = np.bincount(self.group, weights=w * self.y, minlength=self.n_groups)
        fp = np.bincount(self.group, weights=w * (1 - self.y), minlength=self.n_groups)
        P, N = tp.sum(), fp.sum()
        brier = float(np.sum(w * (self.p - self.y) ** 2) / w.sum())
        if P == 0 or N == 0:
            return np.nan, np.nan, brier
        # AUROC: positives in group g beat all negatives in later (lower-score) groups, tie = 1/2
        neg_below = N - np.cumsum(fp)
        auroc = float(np.sum(tp * (neg_below + 0.5 * fp)) / (P * N))
        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        precision = ctp / np.maximum(ctp + cfp, 1e-12)
        recall = ctp / P
        auprc = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
        return auroc, auprc, brier


def point_metrics(y, p):
    return dict(zip(["auroc", "auprc", "brier"], RankedScores(np.asarray(p, float), np.asarray(y)).metrics(
        np.ones(len(y)))))


def patient_weights(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    uniq, inv = np.unique(groups, return_inverse=True)
    counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
    return counts[inv].astype(float)


def paired_bootstrap(wide: pd.DataFrame, conditions: list[str], reference: str, n_rep: int, seed: int,
                     label_col="label", group_col="subject_id") -> pd.DataFrame:
    """wide: one row per admission with a probability column per condition."""
    y = wide[label_col].values
    ranked = {c: RankedScores(wide[c].values.astype(float), y) for c in conditions}
    rng = np.random.default_rng(seed)
    reps = np.full((n_rep, len(conditions), 3), np.nan)
    for r in range(n_rep):
        w = patient_weights(wide[group_col].values, rng)
        for k, c in enumerate(conditions):
            reps[r, k] = ranked[c].metrics(w)
    ref = conditions.index(reference)
    ref_point = ranked[reference].metrics(np.ones(len(y)))
    rows = []
    for k, c in enumerate(conditions):
        point = ranked[c].metrics(np.ones(len(y)))
        row = {"condition": c, "n_admissions": len(y), "n_patients": int(wide[group_col].nunique()),
               "n_deaths": int(y.sum())}
        for m, name in enumerate(["auroc", "auprc", "brier"]):
            vals = reps[:, k, m]
            d = vals - reps[:, ref, m]
            ok = ~np.isnan(d)
            row[name] = point[m]
            row[f"{name}_ci_low"], row[f"{name}_ci_high"] = np.nanpercentile(vals, [2.5, 97.5])
            row[f"delta_{name}_vs_ref"] = point[m] - ref_point[m]
            if k != ref:
                row[f"delta_{name}_ci_low"], row[f"delta_{name}_ci_high"] = np.percentile(d[ok], [2.5, 97.5])
                row[f"delta_{name}_p_boot"] = min(1.0, 2 * min(np.mean(d[ok] <= 0), np.mean(d[ok] >= 0)))
        row["n_valid_replicates"] = int((~np.isnan(reps[:, k, 0])).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_change(wide: pd.DataFrame, conditions: list[str], reference: str) -> pd.DataFrame:
    rows = []
    ref = wide[reference].values
    hi = np.quantile(ref, 0.9)
    for c in conditions:
        d = wide[c].values - ref
        a = np.abs(d)
        died = wide.label.values == 1
        rows.append({
            "condition": c, "mean_p": wide[c].mean(), "mean_p_died": wide.loc[died, c].mean(),
            "mean_p_survived": wide.loc[~died, c].mean(),
            "median_abs_change_vs_ref": np.median(a), "p90_abs_change": np.quantile(a, 0.9),
            "p95_abs_change": np.quantile(a, 0.95), "p99_abs_change": np.quantile(a, 0.99),
            "pct_abs_change_gt_0.05": 100 * np.mean(a > 0.05), "pct_abs_change_gt_0.10": 100 * np.mean(a > 0.10),
            "median_change_died": np.median(d[died]), "median_change_survived": np.median(d[~died]),
            "pct_crossing_ref_top_decile_threshold": 100 * np.mean((wide[c].values >= hi) != (ref >= hi)),
        })
    return pd.DataFrame(rows)


def load_wide(cfg, model: str) -> pd.DataFrame:
    priv = cfg["paths"]["private_dir"]
    pred = pd.read_parquet(priv / f"predictions_{model}.parquet")
    wide = pred.pivot(index="study_id", columns="condition", values="p_mortality").reset_index()
    cohort = pd.read_parquet(priv / "cohort.parquet", columns=["study_id", "subject_id", "label",
                                                                  "anchor_year_group"])
    return wide.merge(cohort, on="study_id", how="left")


def random_split_comparison(cfg, model: str = "tfidf") -> pd.DataFrame:
    """Patient-level vs random admission-level split (different test sets -> unpaired).

    Each test set is resampled by patient independently; the CI of the difference is
    the percentile interval of replicate differences. The random-split test set is
    also split by whether the admission's patient has admissions in its training split."""
    priv, bc = cfg["paths"]["private_dir"], cfg["bootstrap"]
    cohort = pd.read_parquet(priv / "cohort.parquet", columns=["study_id", "subject_id", "label"])
    pat = pd.read_parquet(priv / f"predictions_{model}.parquet")
    pat = pat[pat.condition == "T1_ROBUST"].merge(cohort, on="study_id")
    rnd = pd.read_parquet(priv / f"predictions_{model}_randsplit.parquet").merge(cohort, on="study_id")
    sets = {"patient_level_split": pat, "random_admission_split": rnd,
            "random_split_patient_seen_in_train": rnd[rnd.patient_in_train],
            "random_split_patient_unseen": rnd[~rnd.patient_in_train]}
    reps, rows = {}, []
    for k, (name, d) in enumerate(sets.items()):
        rs = RankedScores(d.p_mortality.values, d.label.values)
        rng = np.random.default_rng(bc["seed"] + k)
        reps[name] = np.array([rs.metrics(patient_weights(d.subject_id.values, rng))
                               for _ in range(bc["n_replicates"])])
        point = rs.metrics(np.ones(len(d)))
        row = {"test_set": name, "n_admissions": len(d), "n_patients": int(d.subject_id.nunique()),
               "n_deaths": int(d.label.sum())}
        for m, metric in enumerate(["auroc", "auprc", "brier"]):
            row[metric] = point[m]
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = np.nanpercentile(reps[name][:, m], [2.5, 97.5])
            if name != "patient_level_split":
                diff = reps[name][:, m] - reps["patient_level_split"][:, m]
                row[f"delta_{metric}_vs_patient_split"] = point[m] - rs_point(pat, m)
                row[f"delta_{metric}_ci_low"], row[f"delta_{metric}_ci_high"] = np.nanpercentile(diff, [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows)


def overlap_comparison(cfg, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixed evaluation set (patient_overlap.py): arm B (evaluation patients' other admissions in training)
    vs arm A (none), paired per training seed, overall and by whether the evaluation admission is the
    patient's last (all siblings earlier) or not (some siblings later)."""
    from patient_overlap import overlap_splits
    priv, bc = cfg["paths"]["private_dir"], cfg["bootstrap"]
    pred = pd.read_parquet(priv / f"predictions_{model}_overlap.parquet")
    wide = pred.pivot(index="study_id", columns="condition", values="p_mortality").reset_index()
    v = pd.read_parquet(priv / "variants.parquet", columns=["study_id", "split", "in_analysis"])
    v = v[v.in_analysis].reset_index(drop=True)
    c = pd.read_parquet(priv / "cohort.parquet", columns=["study_id", "subject_id", "label", "admittime"])
    o = overlap_splits(v.merge(c, on="study_id")[["study_id", "subject_id", "split", "label", "admittime"]],
                       cfg["seed"])
    wide = wide.merge(o[["study_id", "eval_is_last"]], on="study_id").merge(c[["study_id", "subject_id", "label"]],
                                                                          on="study_id")
    from exposure_analysis import logloss, summarise
    rows, shifts = [], []
    for seed, suffix in [(42, ""), (43, "_s43"), (44, "_s44")]:
        a, b = "T1_OVL_A" + suffix, "T1_OVL_B" + suffix
        if a not in wide or b not in wide:
            continue
        # Deaths are always a patient's last admission, so AUROC exists only for 'all' and 'eval_is_last'.
        for stratum, sub in [("all", wide), ("eval_is_last", wide[wide.eval_is_last])]:
            t = paired_bootstrap(sub, [a, b], a, bc["n_replicates"], bc["seed"])
            t.insert(0, "stratum", stratum)
            t.insert(0, "training_seed", seed)
            rows.append(t)
        w = wide.assign(dp=wide[b] - wide[a], dll=logloss(wide.label.values, wide[b].values)
                        - logloss(wide.label.values, wide[a].values))
        for (last, lab), sub in w.groupby(["eval_is_last", "label"]):
            shifts.append({"training_seed": seed, "eval_is_last": bool(last), "outcome": "died" if lab else "survived",
                           **summarise(sub, bc["n_replicates"], bc["seed"])})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(shifts)


def rs_point(d: pd.DataFrame, m: int) -> float:
    return RankedScores(d.p_mortality.values, d.label.values).metrics(np.ones(len(d)))[m]




def main(model: str):
    if model not in {"core", "tfidf"}:
        raise ValueError("Supported models: core, tfidf")
    cfg = load_config()
    bc = cfg["bootstrap"]
    wide = load_wide(cfg, model)
    conds = [c for c in ["T0_COPIV", "T1_ROBUST", "T2_SCREENED", "T3_RETROSPECTIVE", "T4_OTHERPATIENT",
                         "T1_NO_PE", "T1_NO_HPI", "T1_PE_LABELLED", "T1_ICD", "T3A_DCOND", "T3B_DDIAG",
                         "T3C_BHC", "T3D_BHC_NOOUTCOME", "T4M_OUTCOMEMATCHED"]
             if c in wide]
    out = results_dir(cfg)
    table = paired_bootstrap(wide, conds, "T1_ROBUST", bc["n_replicates"], bc["seed"])
    table.to_csv(out / f"{model}_metrics.csv", index=False)
    prediction_change(wide, conds, "T1_ROBUST").to_csv(out / f"{model}_prediction_change.csv", index=False)
    extra = {}
    sens = [c for c in ["T0_COPIV@510", "T1_ROBUST@510", "T2_SCREENED@510"] if c in wide]
    if sens:
        paired_bootstrap(wide, sens, "T1_ROBUST@510", bc["n_replicates"], bc["seed"]).to_csv(
            out / f"{model}_metrics_full_budget.csv", index=False)
    if "T2_SCREENED" in wide:
        # Admissions whose prediction changed at all under screening: direction by outcome.
        d = wide.T2_SCREENED - wide.T1_ROBUST
        rows = []
        for lab, name in [(1, "died"), (0, "survived")]:
            sel = (d != 0) & (wide.label == lab)
            rows.append({"outcome": name, "n_changed": int(sel.sum()),
                         "pct_of_outcome_group": 100 * sel.sum() / (wide.label == lab).sum(),
                         "mean_delta_p": d[sel].mean(), "median_delta_p": d[sel].median(),
                         "pct_decreased": 100 * (d[sel] < 0).mean() if sel.any() else np.nan})
        pd.DataFrame(rows).to_csv(out / f"{model}_screening_effect_changed.csv", index=False)
    if model == "core":
        # Possible overlap with the MIMIC-III training period (2001-2012): stratify by anchor_year_group.
        late = wide.anchor_year_group.isin(["2014 - 2016", "2017 - 2019", "2020 - 2022"])
        for name, sub in [("anchor_2014_or_later", wide[late]), ("anchor_2008_2013", wide[~late])]:
            t = paired_bootstrap(sub, conds, "T1_ROBUST", bc["n_replicates"], bc["seed"])
            t.insert(0, "stratum", name)
            extra[name] = t
        pd.concat(extra.values()).to_csv(out / "core_metrics_by_anchor_year.csv", index=False)
    t1b_conds = [c for c in ["T1_ROBUST", "T1B_ROBUST", "T0_COPIV", "T2_SCREENED", "T2B_SCREENED"]
                 if c in wide]
    if "T1B_ROBUST" in t1b_conds:
        # Improved parser: against T1, and its own screening effect against T1b
        t = paired_bootstrap(wide, t1b_conds, "T1_ROBUST", bc["n_replicates"], bc["seed"])
        t.insert(0, "reference", "T1_ROBUST")
        parts = [t]
        if "T2B_SCREENED" in wide:
            t = paired_bootstrap(wide, ["T1B_ROBUST", "T2B_SCREENED"], "T1B_ROBUST", bc["n_replicates"], bc["seed"])
            t.insert(0, "reference", "T1B_ROBUST")
            parts.append(t)
        full = [c for c in ["T1_ROBUST@510", "T1B_ROBUST@510", "T2_SCREENED@510", "T2B_SCREENED@510"] if c in wide]
        if "T1B_ROBUST@510" in full:      # CORe at the full 510-token budget (Physical Exam mostly visible)
            t = paired_bootstrap(wide, full, "T1_ROBUST@510", bc["n_replicates"], bc["seed"])
            t.insert(0, "reference", "T1_ROBUST@510")
            parts.append(t)
        pd.concat(parts).to_csv(out / f"{model}_metrics_t1b.csv", index=False)
    pe_conds = [c for c in ["T1_ROBUST", "T1_NO_PE", "T1_PE_LABELLED"] if c in wide]
    if model == "tfidf" and len(pe_conds) > 1:
        # Where does the Physical Exam signal come from? Paired effects within each PE documentation type.
        pe = pd.read_parquet(cfg["paths"]["private_dir"] / "variants.parquet", columns=["study_id", "pe_status"])
        w = wide.merge(pe, on="study_id", how="left")
        parts = []
        for status, sub in w.groupby("pe_status"):
            t = paired_bootstrap(sub, pe_conds, "T1_ROBUST", bc["n_replicates"], bc["seed"])
            t.insert(0, "pe_status", status)
            parts.append(t)
        pd.concat(parts).to_csv(out / "tfidf_metrics_by_pe_status.csv", index=False)
    if (cfg["paths"]["private_dir"] / f"predictions_{model}_overlap.parquet").exists():
        ovl, shift = overlap_comparison(cfg, model)
        ovl.to_csv(out / f"{model}_patient_overlap.csv", index=False)
        shift.to_csv(out / f"{model}_patient_overlap_shift.csv", index=False)
    if (cfg["paths"]["private_dir"] / f"predictions_{model}_randsplit.parquet").exists():
        random_split_comparison(cfg, model).to_csv(out / f"{model}_random_split.csv", index=False)
    write_json({"model": model, "reference": "T1_ROBUST", "n_replicates": bc["n_replicates"],
                "seed": bc["seed"], "resampling_unit": "subject_id"}, out / f"{model}_bootstrap_settings.json")
    cols = ["condition", "auroc", "auroc_ci_low", "auroc_ci_high", "delta_auroc_vs_ref", "delta_auroc_ci_low",
            "delta_auroc_ci_high", "auprc", "delta_auprc_vs_ref", "brier", "n_valid_replicates"]
    print(table[[c for c in cols if c in table]].round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "core")
