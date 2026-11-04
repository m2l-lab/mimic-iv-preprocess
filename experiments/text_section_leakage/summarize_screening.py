"""Summarize automated retrospective-content rule hits by condition and section.

Counts describe screening-rule matches, rather than verified temporal content.
Writes aggregate CSV/JSON files under the configured report directory.

CLI: python summarize_screening.py
"""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np
import pandas as pd

from build_variants import split_t0
from common import load_config, results_dir, write_json
from screen_retrospective_content import detect

CATS = ["outcome", "post_anchor"]


def _detect_row(args):
    t0_text, inj_text = args
    out = {}
    secs = split_t0(t0_text) if isinstance(t0_text, str) else {}
    for c in CATS:
        out[f"T0_{c}"] = 0
    for name, txt in secs.items():
        for m in detect(txt, name):
            out[f"T0_{m.category}"] += 1
    for c in CATS:
        out[f"INJ_{c}"] = 0
    for m in detect(inj_text or "", "_injection_"):
        out[f"INJ_{m.category}"] += 1
    return out


def candidates(cfg):
    priv, res = cfg["paths"]["private_dir"], results_dir(cfg)
    v = pd.read_parquet(priv / "variants.parquet", columns=[
        "study_id", "split", "in_analysis", "pe_status", "t0_text", "inj_text", "t4_donor", "t1_text", "t2_text"]
    )
    v = v[v.in_analysis].reset_index(drop=True)
    v = v.merge(pd.read_parquet(priv / "cohort.parquet", columns=["study_id", "label"]), on="study_id")
    with Pool(6) as pool:
        det = pd.DataFrame(pool.map(_detect_row, zip(v.t0_text, v.inj_text), chunksize=2000))
    v = pd.concat([v, det], axis=1)
    log = pd.read_parquet(priv / "screening_log.parquet")
    rem = log[log.removed]
    for c in CATS:
        ids = set(rem.loc[rem.category == c, "study_id"])
        v[f"T1_{c}"] = v.study_id.isin(ids).astype(int)
    v["T1_any"] = (v.T1_outcome + v.T1_post_anchor > 0).astype(int)
    inj_flag = {c: dict(zip(v.study_id, v[f"INJ_{c}"] > 0)) for c in CATS}
    for c in CATS:
        v[f"T3_{c}"] = ((v[f"T1_{c}"] > 0) | (v[f"INJ_{c}"] > 0)).astype(int)
        v[f"T4_{c}"] = ((v[f"T1_{c}"] > 0) | v.t4_donor.map(inj_flag[c]).fillna(False)).astype(int)
    for k in ["T0", "T1", "T3", "T4"]:
        v[f"{k}_any"] = ((v[f"{k}_outcome"] > 0) | (v[f"{k}_post_anchor"] > 0)).astype(int)

    wc = {c: v[c].fillna("").str.count(r"\S+") for c in ["t0_text", "t1_text", "t2_text", "inj_text"]}
    words = {"T0_COPIV": wc["t0_text"], "T1_ROBUST": wc["t1_text"], "T2_SCREENED": wc["t2_text"],
             "T3_RETROSPECTIVE": wc["t1_text"] + wc["inj_text"]}
    rows = []
    for cond, key in [("T0_COPIV", "T0"), ("T1_ROBUST", "T1"), ("T2_SCREENED", None),
                      ("T3_RETROSPECTIVE", "T3"), ("T4_OTHERPATIENT", "T4")]:
        row = {"condition": cond, "admissions": len(v),
               "median_words": float(words[cond].median()) if cond in words else np.nan}
        for scope, sub in [("all", v), ("died", v[v.label == 1]), ("survived", v[v.label == 0])]:
            for c in CATS + ["any"]:
                row[f"cand_{c}_pct_{scope}"] = (round(100 * (sub[f"{key}_{c}"] > 0).mean(), 3) if key else 0.0)
        rows.append(row)
    table_a = pd.DataFrame(rows)
    table_a.to_csv(res / "table_a_candidates.csv", index=False)

    by_section = (rem.merge(v[["study_id", "label"]], on="study_id")
                  .groupby(["section", "category"])
                  .agg(n_admissions=("study_id", "nunique")).reset_index())
    n_died, n_surv = int((v.label == 1).sum()), int((v.label == 0).sum())
    sec_lab = (rem.merge(v[["study_id", "label"]], on="study_id")
               .groupby(["section", "category", "label"]).study_id.nunique().unstack(fill_value=0))
    by_section["pct_of_died"] = by_section.apply(lambda r: 100 * sec_lab.loc[(r.section, r.category)].get(1, 0) / n_died, axis=1)
    by_section["pct_of_survived"] = by_section.apply(lambda r: 100 * sec_lab.loc[(r.section, r.category)].get(0, 0) / n_surv, axis=1)
    by_section["pct_of_all"] = 100 * by_section.n_admissions / len(v)
    by_section.round(4).to_csv(res / "t1_candidates_by_section.csv", index=False)

    by_rule = (rem.merge(v[["study_id", "label"]], on="study_id").groupby(["rule_id", "label"]).study_id.nunique()
               .unstack(fill_value=0).rename(columns={0: "survived", 1: "died"}).reset_index())
    by_rule["pct_died"] = 100 * by_rule.died / n_died
    by_rule["pct_survived"] = 100 * by_rule.survived / n_surv
    by_rule.round(4).to_csv(res / "t1_candidates_by_rule.csv", index=False)

    pe = v.groupby("pe_status").agg(n=("study_id", "size"), t1_post_anchor_pct=("T1_post_anchor", "mean"),
                                    t1_outcome_pct=("T1_outcome", "mean"), mortality_pct=("label", "mean"))
    (pe.assign(t1_post_anchor_pct=100 * pe.t1_post_anchor_pct, t1_outcome_pct=100 * pe.t1_outcome_pct,
               mortality_pct=100 * pe.mortality_pct).round(3).to_csv(res / "candidates_by_pe_status.csv"))
    # outcome-disclosure among deaths: how often does each representation reveal death?
    write_json({"n_analysis": len(v), "n_died": n_died, "n_survived": n_surv,
                "note": "automated rule-based candidates; not independently verified"}, res / "candidates_meta.json")
    print(table_a[["condition", "median_words", "cand_any_pct_all", "cand_outcome_pct_all",
                   "cand_outcome_pct_died", "cand_outcome_pct_survived"]].to_string(index=False))


if __name__ == "__main__":
    candidates(load_config())
