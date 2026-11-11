"""Leakage-injection row: microbiology results windowed by charttime instead of storetime.

Anchor = admittime + 48 h. Cohort: analysis admissions still in hospital and alive at the anchor
(same patient-level split). Specimens collected in [admittime, anchor] contribute tokens:
  - known at collection: specimen type and test name;
  - known only when the result is stored: organism, susceptibility (antibiotic x interpretation),
    or a negative result (no organism).
MICRO_STORETIME uses result tokens only if storetime <= anchor (correct); MICRO_CHARTTIME uses them
for every specimen collected before the anchor (the pitfall). T1_* adds the T1 admission text
(TF-IDF, as in train_tfidf_baseline.py). Free-text comments are not used.

Writes: private predictions_micro.parquet; public micro_summary.json, micro_metrics.csv.
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from bootstrap_evaluation import paired_bootstrap
from common import load_config, mimic_path, results_dir, write_json
from train_tfidf_baseline import Featurizer

ANCHOR_HOURS = 48
FAMILIES = {"MICRO_STORETIME": ["MICRO_STORETIME", "MICRO_CHARTTIME"],
            "T1_MICRO_STORETIME": ["T1_48H", "T1_MICRO_STORETIME", "T1_MICRO_CHARTTIME"]}


def norm(prefix: str, s: pd.Series) -> pd.Series:
    return prefix + s.fillna("na").astype(str).str.lower().map(lambda x: re.sub(r"[^a-z0-9]+", "_", x).strip("_"))


def micro_docs(cfg, cohort: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per study_id: token documents for both windowing rules, plus aggregate timing statistics."""
    m = pd.read_csv(mimic_path(cfg, "admissions_csv").parent / "microbiologyevents.csv",
                    usecols=["subject_id", "chartdate", "charttime", "spec_type_desc", "storetime", "test_name",
                             "org_name", "ab_name", "interpretation"],
                    parse_dates=["chartdate", "charttime", "storetime"])
    m["collected"] = m.charttime.fillna(m.chartdate)
    m = m.merge(cohort[["study_id", "subject_id", "admittime", "anchor"]], on="subject_id")
    m = m[(m.collected >= m.admittime) & (m.collected <= m.anchor)].copy()
    known = norm("spec_", m.spec_type_desc) + " " + norm("test_", m.test_name)
    result = np.where(m.org_name.notna(), norm("org_", m.org_name), norm("neg_", m.test_name))
    result = pd.Series(result, index=m.index)
    has_ab = m.ab_name.notna()
    result[has_ab] = result[has_ab] + " " + norm("ab_", m.ab_name[has_ab]) + "_" + norm("", m.interpretation[has_ab])
    stored_by_anchor = m.storetime.notna() & (m.storetime <= m.anchor)
    m["doc_store"] = known + np.where(stored_by_anchor, " " + result, "")
    m["doc_chart"] = known + " " + result
    docs = m.groupby("study_id").agg(doc_store=("doc_store", " ".join), doc_chart=("doc_chart", " ".join))
    delay_h = (m.storetime - m.collected).dt.total_seconds() / 3600
    stats = {"n_event_rows_in_window": int(len(m)),
             "pct_rows_result_stored_after_anchor": round(100 * float((~stored_by_anchor).mean()), 2),
             "pct_rows_storetime_missing": round(100 * float(m.storetime.isna().mean()), 2),
             "median_store_delay_hours": round(float(delay_h.median()), 1),
             "p90_store_delay_hours": round(float(delay_h.quantile(0.9)), 1),
             "pct_admissions_with_micro_in_window": round(100 * len(docs) / len(cohort), 2),
             "pct_admissions_docs_differ": round(100 * float((docs.doc_store != docs.doc_chart).sum()) / len(cohort), 2)}
    return docs, stats


def fit_predict(Xtr, ytr, Xva, yva, Xte, cfg) -> tuple[np.ndarray, dict]:
    tc = cfg["tfidf"]
    best = None
    for C in tc["C_grid"]:
        clf = LogisticRegression(C=C, solver="liblinear", class_weight=tc["class_weight"], max_iter=tc["max_iter"],
                                 random_state=cfg["seed"]).fit(Xtr, ytr)
        auc = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])
        if best is None or auc > best[0]:
            best = (auc, C, clf)
    return best[2].predict_proba(Xte)[:, 1], {"best_C": best[1], "val_auroc": round(best[0], 4)}


def main():
    cfg = load_config()
    priv = cfg["paths"]["private_dir"]
    cohort = pd.read_parquet(priv / "cohort.parquet",
                             columns=["study_id", "subject_id", "label", "split", "admittime", "dischtime", "deathtime"])
    keep = pd.read_parquet(priv / "variants.parquet", columns=["study_id", "in_analysis"])
    cohort = cohort[cohort.study_id.isin(keep.study_id[keep.in_analysis])]
    for c in ["admittime", "dischtime", "deathtime"]:
        cohort[c] = pd.to_datetime(cohort[c])
    cohort["anchor"] = cohort.admittime + pd.Timedelta(hours=ANCHOR_HOURS)
    cohort = cohort[(cohort.dischtime > cohort.anchor) & (cohort.deathtime.isna() | (cohort.deathtime > cohort.anchor))]
    cohort = cohort.reset_index(drop=True)
    docs, stats = micro_docs(cfg, cohort)
    d = cohort[["study_id", "subject_id", "label", "split"]].merge(docs, on="study_id", how="left").fillna(
        {"doc_store": "", "doc_chart": ""})
    text = pd.read_parquet(priv / "variants.parquet", columns=["study_id", "t1_text"])
    d = d.merge(text, on="study_id", how="left")
    y = d.label.values
    idx = {s: np.flatnonzero(d.split.values == s) for s in ["train", "val", "test"]}

    cv = CountVectorizer(token_pattern=r"\S+", lowercase=False, binary=True, min_df=5)
    cv.fit(pd.concat([d.doc_store, d.doc_chart]).values[np.r_[idx["train"], idx["train"] + len(d)]])
    micro = {k: cv.transform(d[col].values) for k, col in [("STORETIME", "doc_store"), ("CHARTTIME", "doc_chart")]}
    feat = Featurizer(cfg["tfidf"])
    t = time.time()
    Xt = {"train": feat.fit_transform(d.t1_text.values[idx["train"]])}
    Xt["val"], Xt["test"] = feat.transform(d.t1_text.values[idx["val"]]), feat.transform(d.t1_text.values[idx["test"]])
    print(f"features ready {time.time() - t:.0f}s", flush=True)

    def mats(cond):
        if cond == "T1_48H":
            return [Xt[s] for s in ["train", "val", "test"]]
        rule = cond.rsplit("_", 1)[1]
        mi = [micro[rule][idx[s]] for s in ["train", "val", "test"]]
        if cond.startswith("T1_"):
            return [sparse.hstack([Xt[s], m]).tocsr() for s, m in zip(["train", "val", "test"], mi)]
        return mi

    preds, selection = [], {}
    for cond in [c for fam in FAMILIES.values() for c in fam]:
        Xtr, Xva, Xte = mats(cond)
        p, selection[cond] = fit_predict(Xtr, y[idx["train"]], Xva, y[idx["val"]], Xte, cfg)
        preds.append(pd.DataFrame({"study_id": d.study_id.values[idx["test"]], "condition": cond, "p_mortality": p}))
        print(cond, selection[cond], flush=True)
    pred = pd.concat(preds, ignore_index=True)
    pred.to_parquet(priv / "predictions_micro.parquet", index=False)

    wide = pred.pivot(index="study_id", columns="condition", values="p_mortality").reset_index().merge(
        d[["study_id", "subject_id", "label"]], on="study_id")
    bc = cfg["bootstrap"]
    tables = []
    for ref, conds in FAMILIES.items():
        tab = paired_bootstrap(wide, conds, ref, bc["n_replicates"], bc["seed"])
        tab.insert(0, "reference", ref)
        tables.append(tab)
    res = results_dir(cfg)
    pd.concat(tables).to_csv(res / "micro_metrics.csv", index=False)
    stats.update({"anchor_hours": ANCHOR_HOURS, "n_admissions": len(d), "n_test": int(len(idx["test"])),
                  "deaths_test": int(y[idx["test"]].sum()), "n_micro_features": len(cv.vocabulary_),
                  "model_selection": selection})
    write_json(stats, res / "micro_summary.json")
    print(stats)


if __name__ == "__main__":
    main()
