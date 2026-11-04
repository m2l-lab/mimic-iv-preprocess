"""Time-available reference representation from MIMIC-IV-ED (experiment #8).

For analysis admissions linked to an ED stay (edstays.hadm_id), build ED_REF from information that
existed at the hospital admission time (anchor = admissions.admittime):
  - triage: chief complaint (free text), acuity, vitals, pain; arrival transport
  - ED vital signs charted <= anchor (last value and extremes, binned)
  - medication reconciliation (home medications) and Pyxis dispensing charted <= anchor
  - laboratory results (hosp labevents) collected after ED arrival and *stored* <= anchor:
    tokens for measured tests and abnormal flags
ED diagnoses are excluded (billing codes without a time stamp). Everything is rendered as tokens /
text for the same TF-IDF + LR protocol, retrained within the ED-linked subset on the same patient split.

Conditions (paired on the ED-linked test admissions):
  ED_REF, T1, T1_NO_PE, ED_T1 (ED_REF + T1), ED_T1_NO_PE, ED_T1_PE_LABELLED
The ED_T1 vs ED_T1_NO_PE contrast asks what the Physical Exam adds beyond ED-documented vitals and labs,
also by exam documentation type.

Writes (private): ed_reference.parquet, predictions_ed.parquet
       (public):  ed_reference_summary.json, ed_metrics.csv, ed_metrics_by_pe_status.csv
"""
from __future__ import annotations

import glob
import re
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from bootstrap_evaluation import paired_bootstrap
from common import load_config, results_dir, write_json
from train_tfidf_baseline import Featurizer, labelled_pe_only, without_section

CONDITIONS = ["ED_REF", "T1", "T1_NO_PE", "ED_T1", "ED_T1_NO_PE", "ED_T1_PE_LABELLED"]
# (column, cut points, labels): clinically conventional bands
BINS = {"temperature": ([96.8, 100.4, 102.2], ["low", "nl", "fever", "hifever"]),
        "heartrate": ([50, 100, 120], ["brady", "nl", "tachy", "tachy120"]),
        "resprate": ([12, 20, 30], ["low", "nl", "high", "high30"]),
        "o2sat": ([88, 92, 95], ["lt88", "lt92", "lt95", "nl"]),
        "sbp": ([90, 100, 140, 180], ["lt90", "lt100", "nl", "high", "gt180"]),
        "dbp": ([60, 90, 110], ["low", "nl", "high", "gt110"]),
        "pain": ([1, 4, 7], ["none", "mild", "mod", "severe"])}


def slug(s: pd.Series) -> pd.Series:
    return s.fillna("na").astype(str).str.lower().map(lambda x: re.sub(r"[^a-z0-9]+", "_", x).strip("_")[:40])


def binned(df: pd.DataFrame, prefix: str) -> pd.Series:
    parts = []
    for col, (cuts, labels) in BINS.items():
        if col not in df:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        b = pd.Series(pd.cut(v, [-np.inf, *cuts, np.inf], labels=labels, right=False).astype(object), index=df.index)
        parts.append((prefix + col + "_" + b).where(b.notna(), ""))
    return pd.concat(parts, axis=1).agg(" ".join, axis=1).str.split().str.join(" ")


def read_cache(root, table, columns, filters=None) -> pd.DataFrame:
    files = sorted(glob.glob(f"{root}/parquet_cache/{table}/*.parquet"))
    return pd.concat([pd.read_parquet(f, columns=columns, filters=filters) for f in files], ignore_index=True)


def build_reference(cfg, cohort: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    root = cfg["data"]["mimic_root"]
    stays = read_cache(root, "ed/edstays", ["subject_id", "hadm_id", "stay_id", "intime", "arrival_transport"])
    stays = stays.merge(cohort[["study_id", "hadm_id", "admittime"]], on="hadm_id")
    stays = stays[stays.intime <= stays.admittime].sort_values("intime").groupby("study_id").tail(1)
    sid = stays.set_index("stay_id")

    tri = read_cache(root, "ed/triage", None)
    tri = tri[tri.stay_id.isin(sid.index)].set_index("stay_id")
    triage_tok = ("acuity_" + tri.acuity.astype("Int64").astype(str)).str.replace("acuity_<NA>", "") + " " + \
        binned(tri, "tri_")
    complaint = tri.chiefcomplaint.fillna("").str.lower()

    def timed(table, cols):
        d = read_cache(root, table, ["stay_id", "charttime", *cols])
        d = d[d.stay_id.isin(sid.index)]
        return d[d.charttime <= d.stay_id.map(sid.admittime)]

    vit = timed("ed/vitalsign", list(BINS) + ["rhythm"])
    vit = vit.sort_values("charttime")
    last = vit.groupby("stay_id").tail(1).set_index("stay_id")
    worst = vit.groupby("stay_id").agg(heartrate=("heartrate", "max"), resprate=("resprate", "max"),
                                       o2sat=("o2sat", "min"), sbp=("sbp", "min"), temperature=("temperature", "max"))
    vit_tok = binned(last, "last_").reindex(sid.index).fillna("") + " " + \
        binned(worst, "worst_").reindex(sid.index).fillna("") + " " + \
        ("rhythm_" + slug(last.rhythm)).where(last.rhythm.notna(), "").reindex(sid.index).fillna("")
    med = timed("ed/medrecon", ["etcdescription"])
    home_tok = ("home_" + slug(med.etcdescription)).groupby(med.stay_id).agg(lambda x: " ".join(sorted(set(x))))
    pyx = timed("ed/pyxis", ["name"])
    ed_med_tok = ("edmed_" + slug(pyx.name)).groupby(pyx.stay_id).agg(lambda x: " ".join(sorted(set(x))))

    win = stays[["subject_id", "stay_id", "intime", "admittime"]]
    labs = []
    for f in sorted(glob.glob(f"{root}/parquet_cache/hosp/labevents/*.parquet")):
        d = pd.read_parquet(f, columns=["subject_id", "itemid", "charttime", "storetime", "flag"],
                            filters=[("subject_id", "in", list(set(win.subject_id)))])
        d = d.merge(win, on="subject_id")
        d = d[(d.charttime >= d.intime) & (d.storetime <= d.admittime)]
        labs.append(d[["stay_id", "itemid", "flag"]])
    labs = pd.concat(labs, ignore_index=True)
    lab_tok = ("lab" + labs.itemid.astype(str) + np.where(labs.flag.eq("abnormal"), "_abn", "")).groupby(
        labs.stay_id).agg(lambda x: " ".join(sorted(set(x))))

    out = pd.DataFrame(index=sid.index)
    out["study_id"] = sid.study_id
    out["ed_text"] = ("TRIAGE COMPLAINT: " + complaint.reindex(sid.index).fillna("") +
                      "\n\nED FINDINGS: arrival_" + slug(sid.arrival_transport) + " " +
                      triage_tok.reindex(sid.index).fillna("") + " " + vit_tok +
                      "\n\nHOME MEDICATIONS: " + home_tok.reindex(sid.index).fillna("") +
                      "\n\nED MEDICATIONS: " + ed_med_tok.reindex(sid.index).fillna("") +
                      "\n\nED LABS: " + lab_tok.reindex(sid.index).fillna(""))
    hours = (sid.admittime - sid.intime).dt.total_seconds() / 3600
    stats = {"n_ed_linked": len(out), "pct_of_analysis_cohort": round(100 * len(out) / len(cohort), 2),
             "median_hours_ed_arrival_to_admission": round(float(hours.median()), 1),
             "pct_with_triage_complaint": round(100 * float((complaint.reindex(sid.index).fillna("") != "").mean()), 1),
             "pct_with_ed_vitals_before_anchor": round(100 * float(sid.index.isin(last.index).mean()), 1),
             "pct_with_home_meds": round(100 * float(sid.index.isin(home_tok.index).mean()), 1),
             "pct_with_ed_meds_before_anchor": round(100 * float(sid.index.isin(ed_med_tok.index).mean()), 1),
             "pct_with_labs_stored_before_anchor": round(100 * float(sid.index.isin(lab_tok.index).mean()), 1)}
    return out.reset_index(drop=True), stats


def main():
    cfg = load_config()
    priv, tc, bc = cfg["paths"]["private_dir"], cfg["tfidf"], cfg["bootstrap"]
    v = pd.read_parquet(priv / "variants.parquet", columns=["study_id", "split", "in_analysis", "pe_status", "t1_text"])
    v = v[v.in_analysis].drop(columns="in_analysis")
    cohort = pd.read_parquet(priv / "cohort.parquet", columns=["study_id", "subject_id", "hadm_id", "label", "admittime"])
    cohort = cohort[cohort.study_id.isin(v.study_id)].assign(admittime=lambda c: pd.to_datetime(c.admittime))
    t = time.time()
    ref, stats = build_reference(cfg, cohort)
    ref.to_parquet(priv / "ed_reference.parquet", index=False)
    print(f"reference built in {time.time() - t:.0f}s", stats, flush=True)

    d = v.merge(ref, on="study_id").merge(cohort[["study_id", "subject_id", "label"]], on="study_id")
    texts = {"ED_REF": d.ed_text, "T1": d.t1_text, "T1_NO_PE": without_section(d.t1_text, "physical_exam")}
    texts["ED_T1"] = d.ed_text + "\n\n" + d.t1_text
    texts["ED_T1_NO_PE"] = d.ed_text + "\n\n" + texts["T1_NO_PE"]
    texts["ED_T1_PE_LABELLED"] = d.ed_text + "\n\n" + labelled_pe_only(d.t1_text, d.pe_status)
    y = d.label.values
    idx = {s: np.flatnonzero(d.split.values == s) for s in ["train", "val", "test"]}
    preds, selection = [], {}
    for cond in CONDITIONS:
        feat = Featurizer(tc)
        Xtr = feat.fit_transform(texts[cond].values[idx["train"]])
        Xva, Xte = feat.transform(texts[cond].values[idx["val"]]), feat.transform(texts[cond].values[idx["test"]])
        best = None
        for C in tc["C_grid"]:
            clf = LogisticRegression(C=C, solver="liblinear", class_weight=tc["class_weight"], max_iter=tc["max_iter"],
                                     random_state=cfg["seed"]).fit(Xtr, y[idx["train"]])
            auc = roc_auc_score(y[idx["val"]], clf.predict_proba(Xva)[:, 1])
            if best is None or auc > best[0]:
                best = (auc, C, clf)
        selection[cond] = {"best_C": best[1], "val_auroc": round(best[0], 4)}
        preds.append(pd.DataFrame({"study_id": d.study_id.values[idx["test"]], "condition": cond,
                                   "p_mortality": best[2].predict_proba(Xte)[:, 1]}))
        print(cond, selection[cond], flush=True)
        del Xtr, Xva, Xte, feat
    pred = pd.concat(preds, ignore_index=True)
    pred.to_parquet(priv / "predictions_ed.parquet", index=False)

    wide = pred.pivot(index="study_id", columns="condition", values="p_mortality").reset_index().merge(
        d[["study_id", "subject_id", "label", "pe_status"]], on="study_id")
    res = results_dir(cfg)
    tabs = []
    for ref_cond, conds in [("ED_REF", ["ED_REF", "T1", "ED_T1"]), ("ED_T1", ["ED_T1", "ED_T1_NO_PE", "ED_T1_PE_LABELLED"]),
                            ("T1", ["T1", "T1_NO_PE"])]:
        tab = paired_bootstrap(wide, conds, ref_cond, bc["n_replicates"], bc["seed"])
        tab.insert(0, "reference", ref_cond)
        tabs.append(tab)
    pd.concat(tabs).to_csv(res / "ed_metrics.csv", index=False)
    strata = []
    for status, sub in wide.groupby("pe_status"):
        tab = paired_bootstrap(sub, ["ED_T1", "ED_T1_NO_PE"], "ED_T1", bc["n_replicates"], bc["seed"])
        tab.insert(0, "pe_status", status)
        strata.append(tab)
    pd.concat(strata).to_csv(res / "ed_metrics_by_pe_status.csv", index=False)
    stats.update({"n_test": int(len(idx["test"])), "deaths_test": int(y[idx["test"]].sum()),
                  "mortality_pct_ed_linked": round(100 * float(y.mean()), 2), "model_selection": selection})
    write_json(stats, res / "ed_reference_summary.json")
    print(stats)


if __name__ == "__main__":
    main()
