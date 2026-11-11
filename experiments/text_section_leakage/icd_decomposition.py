"""Decompose the T1 + ICD gain into admission knowledge, in-hospital course and coding intensity.

All conditions add a block to the T1 TF-IDF text features, as in icd_feature_weights.py (same cohort, patient-level
split, C chosen on validation, test used once):
  T1_ICD_SEP       T1 + all codes of the current admission (reference for the decomposition)
  T1_ICD_PRIOR     T1 + all codes of the patient's earlier admissions discharged before this admission's admittime
                   (known at admission: a legitimate counterpart)
  T1_ICD_NO_ACUTE  T1 + current codes without tier A/B (icd_feature_weights.py) and without acute organ dysfunction,
                   sepsis and shock (ACUTE prefixes, after the Angus organ-dysfunction code set)
  T1_NCODES        T1 + number of current codes only (one-hot over training-split deciles)

Writes: private predictions_icd_decomposition.parquet; public icd_decomposition_metrics.csv (vs T1_ROBUST and vs
T1_ICD_SEP), icd_decomposition_summary.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

from bootstrap_evaluation import paired_bootstrap
from common import load_config, mimic_path, results_dir, write_json
from icd_feature_weights import code_table, fit_select
from train_tfidf_baseline import Featurizer

ACUTE = {10: ("A40", "A41", "R652", "R57", "J96", "J80", "N17", "K72", "D65", "G931", "G934", "G935", "G936"),
         9: ("038", "99591", "99592", "7855", "5188", "5185", "584", "570", "5722", "2866", "3481", "3483", "3484",
             "3485")}


def is_acute(codes: pd.DataFrame) -> pd.Series:
    out = pd.Series(False, index=codes.index)
    for version, prefixes in ACUTE.items():
        out |= (codes.icd_version == version) & codes.icd_code.str.startswith(prefixes)
    return out


def main():
    cfg = load_config()
    tc, priv = cfg["tfidf"], cfg["paths"]["private_dir"]
    v = pd.read_parquet(priv / "variants.parquet", columns=["study_id", "split", "in_analysis", "t1_text"])
    v = v[v.in_analysis].reset_index(drop=True)
    v = v.merge(pd.read_parquet(priv / "cohort.parquet", columns=["study_id", "hadm_id", "subject_id", "label"]),
                on="study_id")
    y = v.label.values
    idx = {s: np.flatnonzero(v.split.values == s) for s in ["train", "val", "test"]}

    adm = pd.read_csv(mimic_path(cfg, "admissions_csv"), usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
                      parse_dates=["admittime", "dischtime"])
    adm = adm[adm.subject_id.isin(v.subject_id)]
    codes, titles = code_table(cfg, adm.hadm_id)
    tier = dict(zip(titles.token, titles.tier))
    current = codes[codes.hadm_id.isin(v.hadm_id)]

    cur = v[["hadm_id", "subject_id"]].merge(adm[["hadm_id", "admittime"]], on="hadm_id")
    pairs = cur.merge(adm.rename(columns={"hadm_id": "prior_hadm", "admittime": "prior_admit"}), on="subject_id")
    pairs = pairs[pairs.dischtime <= pairs.admittime]
    prior = pairs[["hadm_id", "prior_hadm"]].merge(codes.rename(columns={"hadm_id": "prior_hadm"}), on="prior_hadm")
    prior = prior.drop_duplicates(["hadm_id", "token"])

    blocks = {"T1_ICD_SEP": current,
              "T1_ICD_PRIOR": prior,
              "T1_ICD_NO_ACUTE": current[~current.token.map(tier).fillna("").isin({"A", "B"}) & ~is_acute(current)]}

    feat = Featurizer(tc)
    Xt = {"train": feat.fit_transform(v.t1_text.values[idx["train"]])}
    Xt["val"], Xt["test"] = feat.transform(v.t1_text.values[idx["val"]]), feat.transform(v.t1_text.values[idx["test"]])

    def run(cond, Xb):
        X = {s: sparse.hstack([Xt[s], Xb[s]]).tocsr() for s in Xt}
        auc, C, clf = fit_select(X["train"], y[idx["train"]], X["val"], y[idx["val"]], tc, cfg["seed"])
        selection[cond] = {"best_C": C, "val_auroc": round(auc, 4), "n_block_features": int(Xb["train"].shape[1])}
        print(cond, selection[cond], flush=True)
        return clf.predict_proba(X["test"])[:, 1]

    preds, selection = {}, {}
    for cond, block in blocks.items():
        docs = v.hadm_id.map(block.groupby("hadm_id").token.agg(" ".join)).fillna("").values
        vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, min_df=tc["min_df"],
                              sublinear_tf=tc["sublinear_tf"])
        Xb = {"train": vec.fit_transform(docs[idx["train"]])}
        Xb["val"], Xb["test"] = vec.transform(docs[idx["val"]]), vec.transform(docs[idx["test"]])
        preds[cond] = run(cond, Xb)

    n_codes = v.hadm_id.map(current.groupby("hadm_id").size()).fillna(0).values
    edges = np.unique(np.quantile(n_codes[idx["train"]], np.linspace(0.1, 0.9, 9)))
    bins = np.digitize(n_codes, edges).reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown="ignore").fit(bins[idx["train"]])
    preds["T1_NCODES"] = run("T1_NCODES", {s: enc.transform(bins[idx[s]]) for s in idx})

    test = v.iloc[idx["test"]][["study_id", "subject_id", "label"]].reset_index(drop=True)
    pd.concat([test[["study_id"]].assign(condition=c, p_mortality=p) for c, p in preds.items()]).to_parquet(
        priv / "predictions_icd_decomposition.parquet", index=False)
    ref = pd.read_parquet(priv / "predictions_tfidf.parquet")
    ref = ref[ref.condition == "T1_ROBUST"].pivot(index="study_id", columns="condition", values="p_mortality")
    wide = test.assign(**preds).merge(ref, on="study_id")
    bc, conds = cfg["bootstrap"], ["T1_ROBUST", *preds]
    tabs = []
    for reference in ["T1_ROBUST", "T1_ICD_SEP"]:
        tab = paired_bootstrap(wide, conds, reference, bc["n_replicates"], bc["seed"])
        tab.insert(0, "reference", reference)
        tabs.append(tab)
    res = results_dir(cfg)
    pd.concat(tabs).to_csv(res / "icd_decomposition_metrics.csv", index=False)

    has_prior = v.hadm_id.isin(prior.hadm_id).values
    test_deaths = y[idx["test"]] == 1
    summary = {"model_selection": selection,
               "pct_test_with_prior_codes": round(100 * has_prior[idx["test"]].mean(), 1),
               "pct_test_deaths_with_prior_codes": round(100 * has_prior[idx["test"]][test_deaths].mean(), 1),
               "pct_current_code_rows_acute_or_tier_ab": round(100 * float(
                   (current.token.map(tier).fillna("").isin({"A", "B"}) | is_acute(current)).mean()), 1),
               "n_codes_decile_edges": edges.tolist(),
               "median_n_codes_test_deaths": float(np.median(n_codes[idx["test"]][test_deaths])),
               "median_n_codes_test_survivors": float(np.median(n_codes[idx["test"]][~test_deaths])),
               "acute_prefixes": {str(k): list(p) for k, p in ACUTE.items()}}
    write_json(summary, res / "icd_decomposition_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
