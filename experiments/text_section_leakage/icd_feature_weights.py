"""Which ICD codes drive the T1_ICD leakage row, and how much of it do death-encoding codes explain?

T1_ICD (train_tfidf_baseline.py) hashes the code tokens together with the text, and almost all 2**20 buckets are
occupied, so its weights cannot be attributed to single codes. Here the T1 text uses the same Featurizer and the
codes form a separate, unhashed TF-IDF block (one token per code, l2-normalised), so every code has its own weight.

Conditions (same cohort, patient-level split, C chosen on validation, test used once):
  T1_ICD_SEP          T1 + all codes (separate block); should reproduce T1_ICD
  T1_ICD_NO_DEATH     T1 + codes, without codes that encode death or end-of-life care (tier A)
  T1_ICD_NO_TERMINAL  T1 + codes, without tier A and terminal events / resuscitation (tier B)
Tiers are assigned from the code titles (TIER_A, TIER_B, EXCLUDE) and listed in the output.

Writes: public icd_feature_weights.csv (codes with >= MIN_CELL carriers among training admissions),
icd_death_codes.csv (all tier A/B codes, counts suppressed below MIN_CELL), icd_ablation_metrics.csv, icd_summary.json.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from bootstrap_evaluation import paired_bootstrap
from common import load_config, mimic_path, results_dir, write_json
from train_tfidf_baseline import Featurizer

MIN_CELL = 10   # suppress counts of fewer admissions (MIMIC small-cell rule)
TIER_A = re.compile(r"death|mortality|palliative|do not resuscitate|comfort care", re.IGNORECASE)
TIER_B = re.compile(r"cardiac arrest|cardiorespiratory arrest|respiratory arrest", re.IGNORECASE)
EXCLUDE = re.compile(r"family history|personal history|history of|family member|intrauterine|fetal|newborn|neonatal",
                     re.IGNORECASE)
CONDITIONS = {"T1_ICD_SEP": set(), "T1_ICD_NO_DEATH": {"A"}, "T1_ICD_NO_TERMINAL": {"A", "B"}}


def code_table(cfg, hadm_ids: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    hosp = mimic_path(cfg, "admissions_csv").parent
    d = pd.read_csv(hosp / "diagnoses_icd.csv", usecols=["hadm_id", "icd_code", "icd_version"], dtype={"icd_code": str})
    d = d[d.hadm_id.isin(hadm_ids)].copy()
    d["icd_code"] = d.icd_code.str.strip()
    d["token"] = "icd" + d.icd_version.astype(str) + "_" + d.icd_code.str.lower()
    titles = pd.read_csv(hosp / "d_icd_diagnoses.csv", dtype={"icd_code": str})
    titles["icd_code"] = titles.icd_code.str.strip()
    titles["token"] = "icd" + titles.icd_version.astype(str) + "_" + titles.icd_code.str.lower()
    t = titles.long_title.fillna("")
    titles["tier"] = ""
    titles.loc[t.str.contains(TIER_B), "tier"] = "B"
    titles.loc[t.str.contains(TIER_A), "tier"] = "A"
    titles.loc[t.str.contains(EXCLUDE), "tier"] = ""
    return d.drop_duplicates(["hadm_id", "token"]), titles


def fit_select(Xtr, ytr, Xva, yva, tc, seed):
    best = None
    for C in tc["C_grid"]:
        clf = LogisticRegression(C=C, solver="liblinear", class_weight=tc["class_weight"], max_iter=tc["max_iter"],
                                 random_state=seed).fit(Xtr, ytr)
        auc = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])
        if best is None or auc > best[0]:
            best = (auc, C, clf)
    return best


def main():
    cfg = load_config()
    tc, priv = cfg["tfidf"], cfg["paths"]["private_dir"]
    v = pd.read_parquet(priv / "variants.parquet", columns=["study_id", "split", "in_analysis", "t1_text"])
    v = v[v.in_analysis].reset_index(drop=True)
    v = v.merge(pd.read_parquet(priv / "cohort.parquet", columns=["study_id", "hadm_id", "subject_id", "label"]),
                on="study_id")
    y = v.label.values
    idx = {s: np.flatnonzero(v.split.values == s) for s in ["train", "val", "test"]}
    codes, titles = code_table(cfg, v.hadm_id)
    tier = dict(zip(titles.token, titles.tier))

    feat = Featurizer(tc)
    Xt = {"train": feat.fit_transform(v.t1_text.values[idx["train"]])}
    Xt["val"], Xt["test"] = feat.transform(v.t1_text.values[idx["val"]]), feat.transform(v.t1_text.values[idx["test"]])

    preds, selection, weights = {}, {}, None
    for cond, drop in CONDITIONS.items():
        kept = codes[~codes.token.map(tier).fillna("").isin(drop)]
        docs = v.hadm_id.map(kept.groupby("hadm_id").token.agg(" ".join)).fillna("").values
        vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, min_df=tc["min_df"],
                              sublinear_tf=tc["sublinear_tf"])
        Xc = {"train": vec.fit_transform(docs[idx["train"]])}
        Xc["val"], Xc["test"] = vec.transform(docs[idx["val"]]), vec.transform(docs[idx["test"]])
        X = {s: sparse.hstack([Xt[s], Xc[s]]).tocsr() for s in Xt}
        auc, C, clf = fit_select(X["train"], y[idx["train"]], X["val"], y[idx["val"]], tc, cfg["seed"])
        preds[cond] = clf.predict_proba(X["test"])[:, 1]
        selection[cond] = {"best_C": C, "val_auroc": round(auc, 4), "n_codes": len(vec.vocabulary_)}
        print(cond, selection[cond], flush=True)
        if cond == "T1_ICD_SEP":
            vocab = np.array(vec.get_feature_names_out())
            weights = pd.DataFrame({"token": vocab, "coef": clf.coef_[0, Xt["train"].shape[1]:]})
            text_coef = clf.coef_[0, :Xt["train"].shape[1]]
            weights["text_coef_max_abs"] = float(np.abs(text_coef).max())

    # carrier counts on the training split only
    tr = codes[codes.hadm_id.isin(v.hadm_id.values[idx["train"]])].merge(v[["hadm_id", "label"]], on="hadm_id")
    n = tr.groupby("token").agg(n_train=("label", "size"), deaths_train=("label", "sum")).reset_index()
    base_rate = float(y[idx["train"]].mean())
    w = weights.merge(n, on="token", how="left").merge(titles[["token", "icd_version", "icd_code", "long_title", "tier"]],
                                                       on="token", how="left")
    w["death_rate_train"] = w.deaths_train / w.n_train
    w["rank_coef"] = w.coef.rank(ascending=False).astype(int)
    out = w[w.n_train >= MIN_CELL].sort_values("coef", ascending=False)
    out.loc[(out.deaths_train < MIN_CELL) | (out.n_train - out.deaths_train < MIN_CELL),
            ["deaths_train", "death_rate_train"]] = np.nan   # also hides survivor counts below MIN_CELL
    cols = ["rank_coef", "icd_version", "icd_code", "long_title", "tier", "coef", "n_train", "deaths_train",
            "death_rate_train"]
    res = results_dir(cfg)
    out[cols].to_csv(res / "icd_feature_weights.csv", index=False)

    dc = titles[titles.tier != ""].merge(n, on="token", how="left").merge(weights[["token", "coef"]], on="token",
                                                                         how="left")
    dc = dc[dc.n_train.fillna(0) > 0]
    small = dc.n_train < MIN_CELL
    dc.loc[small, ["n_train", "deaths_train"]] = np.nan
    dc.loc[(dc.deaths_train < MIN_CELL) | (dc.n_train - dc.deaths_train < MIN_CELL), "deaths_train"] = np.nan
    dc["n_train_suppressed"] = small
    dc["death_rate_train"] = dc.deaths_train / dc.n_train
    dc.sort_values(["tier", "coef"], ascending=[True, False])[
        ["tier", "icd_version", "icd_code", "long_title", "coef", "n_train", "deaths_train", "death_rate_train",
         "n_train_suppressed"]].to_csv(res / "icd_death_codes.csv", index=False)

    ref = pd.read_parquet(priv / "predictions_tfidf.parquet")
    ref = ref[ref.condition.isin(["T1_ROBUST", "T1_ICD"])].pivot(index="study_id", columns="condition",
                                                                  values="p_mortality")
    wide = v.iloc[idx["test"]][["study_id", "subject_id", "label"]].assign(**preds).merge(ref, on="study_id")
    bc = cfg["bootstrap"]
    tab = paired_bootstrap(wide, ["T1_ROBUST", "T1_ICD", *CONDITIONS], "T1_ROBUST", bc["n_replicates"], bc["seed"])
    tab.to_csv(res / "icd_ablation_metrics.csv", index=False)

    carriers = codes.assign(t=codes.token.map(tier).fillna(""))
    test_ids = set(v.hadm_id.values[idx["test"]])
    test_deaths = set(v.hadm_id.values[idx["test"]][y[idx["test"]] == 1])
    summary = {"base_death_rate_train": round(base_rate, 4), "model_selection": selection,
               "n_codes_tier_A": int((titles.tier == "A").sum()), "n_codes_tier_B": int((titles.tier == "B").sum()),
               "pct_test_deaths_with_tier_A": round(100 * len(set(carriers.hadm_id[carriers.t == "A"]) & test_deaths)
                                                    / len(test_deaths), 1),
               "pct_test_deaths_with_tier_A_or_B": round(100 * len(set(carriers.hadm_id[carriers.t != ""])
                                                                   & test_deaths) / len(test_deaths), 1),
               "pct_test_survivors_with_tier_A": round(100 * len(set(carriers.hadm_id[carriers.t == "A"])
                                                                 & (test_ids - test_deaths))
                                                       / len(test_ids - test_deaths), 2),
               "tier_rules": {"A": TIER_A.pattern, "B": TIER_B.pattern, "exclude": EXCLUDE.pattern},
               "min_cell": MIN_CELL}
    write_json(summary, res / "icd_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
