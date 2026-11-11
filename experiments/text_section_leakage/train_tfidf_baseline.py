"""Phase 6: TF-IDF + logistic regression trained separately per text condition.

Extension conditions (same cohort, model and test set unless stated):
  T1_NO_PE, T1_NO_HPI  section ablations of T1 (apportion the T1 baseline)
  T1_PE_LABELLED       T1 with the Physical Exam kept only where the note labels its admission/discharge
                       sub-exams (pe_status == labelled); otherwise left empty (heading kept)
  T1_ICD               T1 + all ICD codes of the current admission (stay-level codes as features)
  T1_RANDOM_SPLIT      T1, but with a random *admission-level* 70/15/15 split (same seed) instead of
                       the patient-level split; different test set -> separate prediction file
  T3A_DCOND, T3B_DDIAG, T3C_BHC
                       T1 + one discharge-phase section of the same note (T3 = all three)
  T3D_BHC_NOOUTCOME    T1 + Brief Hospital Course without outcome-rule sentences (build_injection_parts.py)
  T4M_OUTCOMEMATCHED   T1 + the three sections of another patient with the same outcome (diagnostic
                       oracle control: donor selection uses labels by design)
  T1_OVL_A, T1_OVL_B   T1 on the fixed-evaluation-set patient-overlap design (patient_overlap.py):
                       training without (A) or with (B) the evaluation patients' other admissions
  T1_LABEL_DIEDLOC, T2_LABEL_DIEDLOC
                       T1 / T2 trained and evaluated with label = hospital_expire_flag OR
                       discharge_location == DIED (140 admissions with DIED but flag 0); own prediction
                       file with the label used, so the main tables are not scored against it
  T1B_ROBUST, T2B_SCREENED
                       improved robust parser (extract_sections_t1b; build_t1b.py) and its screened version
                       (same frozen screening rules as T2)

Identical cohort, labels, patient-disjoint splits, preprocessing and
hyper-parameter search for every condition. Transforms are fitted on the
training split only; C is chosen on the validation split; the test split is
used once. T3/T4 use the full (untruncated) injected sections.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from build_variants import split_t0
from common import ADMISSION_SECTIONS, INJECTION_LABELS, load_config, mimic_path, results_dir, write_json
from extract_sections import render

RANDOM_SPLIT = "T1_RANDOM_SPLIT"
LABEL_DIEDLOC = {"T1_LABEL_DIEDLOC": "T1_ROBUST", "T2_LABEL_DIEDLOC": "T2_SCREENED"}
OVERLAP = {"T1_OVL_A": "ovl_a", "T1_OVL_B": "ovl_b"}
T1B = {"T1B_ROBUST": "t1b_text", "T2B_SCREENED": "t2b_text"}
PART_CONDITIONS = {"T3A_DCOND": ("dcond", "discharge_condition"), "T3B_DDIAG": ("ddiag", "discharge_diagnosis"),
                   "T3C_BHC": ("bhc", "brief_hospital_course"), "T3D_BHC_NOOUTCOME": ("bhc_noout", "brief_hospital_course")}


def without_section(texts: pd.Series, drop: str) -> pd.Series:
    labels = [(n, l) for n, l in ADMISSION_SECTIONS if n != drop]
    return texts.map(lambda t: render(split_t0(t), labels))


def labelled_pe_only(texts: pd.Series, pe_status: pd.Series) -> pd.Series:
    def one(t, status):
        sec = split_t0(t)
        if status != "labelled":
            sec["physical_exam"] = ""
        return render(sec)
    return pd.Series([one(t, s) for t, s in zip(texts, pe_status)], index=texts.index)


def icd_text(cfg, cohort: pd.DataFrame) -> pd.Series:
    """'ICD CODES: icd10_r570 icd10_n179 ...' per study_id (all codes of the admission, seq_num order)."""
    d = pd.read_csv(mimic_path(cfg, "admissions_csv").parent / "diagnoses_icd.csv",
                    usecols=["hadm_id", "seq_num", "icd_code", "icd_version"])
    d = d.merge(cohort[["study_id", "hadm_id"]], on="hadm_id").sort_values(["study_id", "seq_num"])
    tok = "icd" + d.icd_version.astype(str) + "_" + d.icd_code.astype(str).str.strip()
    return "ICD CODES: " + tok.groupby(d.study_id).agg(" ".join)


def random_admission_split(study_ids: np.ndarray, fractions: dict, seed: int) -> np.ndarray:
    perm = np.random.default_rng(seed).permutation(len(study_ids))
    n_tr, n_va = int(round(fractions["train"] * len(perm))), int(round(fractions["val"] * len(perm)))
    split = np.empty(len(perm), dtype=object)
    split[perm[:n_tr]], split[perm[n_tr:n_tr + n_va]], split[perm[n_tr + n_va:]] = "train", "val", "test"
    return split




def label_diedloc(cfg, cohort: pd.DataFrame) -> pd.Series:
    """hospital_expire_flag OR discharge_location == DIED, per study_id."""
    a = pd.read_csv(mimic_path(cfg, "admissions_csv"), usecols=["hadm_id", "discharge_location"])
    died = cohort.merge(a, on="hadm_id", how="left").set_index("study_id").discharge_location.eq("DIED")
    return (cohort.set_index("study_id").label.astype(bool) | died).astype(int)


def one_section(texts: pd.Series, name: str) -> pd.Series:
    """Render one injected section like T3 does (omitted when empty)."""
    return texts.fillna("").map(lambda t: render({name: t}, [(name, INJECTION_LABELS[name])]) if t else "")


def condition_text(v: pd.DataFrame, condition: str, inj_by_id: dict, icd_by_id=None) -> pd.Series:
    if condition in LABEL_DIEDLOC:
        condition = LABEL_DIEDLOC[condition]
    if condition in ("T1_ROBUST", RANDOM_SPLIT, *OVERLAP):
        return v.t1_text
    if condition == "T1_NO_PE":
        return without_section(v.t1_text, "physical_exam")
    if condition == "T1_NO_HPI":
        return without_section(v.t1_text, "history_of_present_illness")
    if condition == "T1_PE_LABELLED":
        return labelled_pe_only(v.t1_text, v.pe_status)
    if condition == "T1_ICD":
        return v.t1_text + "\n\n" + v.study_id.map(icd_by_id).fillna("ICD CODES: ")
    if condition == "T0_COPIV":
        return v.t0_text
    if condition == "T2_SCREENED":
        return v.t2_text
    if condition in T1B:
        return v[T1B[condition]]
    if condition == "T3_RETROSPECTIVE":
        return v.t1_text + "\n\n" + v.inj_text.fillna("")
    if condition == "T4_OTHERPATIENT":
        return v.t1_text + "\n\n" + v.t4_donor.map(inj_by_id).fillna("")
    if condition in PART_CONDITIONS:
        col, name = PART_CONDITIONS[condition]
        return v.t1_text + "\n\n" + one_section(v[col], name)
    if condition == "T4M_OUTCOMEMATCHED":
        return v.t1_text + "\n\n" + v.t4m_donor.map(inj_by_id).fillna("")
    raise ValueError(condition)


class Featurizer:
    def __init__(self, tc: dict):
        self.hv = HashingVectorizer(n_features=tc["n_features"], ngram_range=tuple(tc["ngram_range"]),
                                    alternate_sign=False, norm=None, lowercase=True, dtype=np.float32)
        self.tc = tc

    def fit_transform(self, texts):
        X = self.hv.transform(texts)
        df = np.asarray((X > 0).sum(axis=0)).ravel()
        self.keep = np.flatnonzero((df >= self.tc["min_df"]) & (df <= self.tc["max_df"] * X.shape[0]))
        self.tfidf = TfidfTransformer(sublinear_tf=self.tc["sublinear_tf"])
        return self.tfidf.fit_transform(X[:, self.keep])

    def transform(self, texts):
        return self.tfidf.transform(self.hv.transform(texts)[:, self.keep])


def main(conditions=None):
    cfg = load_config()
    tc = cfg["tfidf"]
    priv = cfg["paths"]["private_dir"]
    conditions = conditions or tc["conditions"]
    v = pd.read_parquet(priv / "variants.parquet",
                        columns=["study_id", "split", "in_analysis", "pe_status", "t0_text", "t1_text",
                                 "t2_text", "inj_text", "t4_donor"])
    inj_by_id = dict(zip(v.study_id, v.inj_text))
    v = v[v.in_analysis].reset_index(drop=True)
    cohort = pd.read_parquet(priv / "cohort.parquet",
                             columns=["study_id", "hadm_id", "subject_id", "label", "admittime"])
    v = v.merge(cohort, on="study_id")
    if set(conditions) & set(OVERLAP):
        from patient_overlap import overlap_splits
        v = v.merge(overlap_splits(v[["study_id", "subject_id", "split", "label", "admittime"]], cfg["seed"]),
                    on="study_id")
    if set(conditions) & set(T1B):
        v = v.merge(pd.read_parquet(priv / "variants_t1b.parquet"), on="study_id", how="left")
    if set(conditions) & (set(PART_CONDITIONS) | {"T4M_OUTCOMEMATCHED"}):
        v = v.merge(pd.read_parquet(priv / "injection_parts.parquet"), on="study_id", how="left")
    y = v.label.values
    y_diedloc = v.study_id.map(label_diedloc(cfg, cohort)).values if set(conditions) & set(LABEL_DIEDLOC) else None
    icd_by_id = icd_text(cfg, cohort) if "T1_ICD" in conditions else None
    rand_split = random_admission_split(v.study_id.values, cfg["split"], cfg["seed"])

    group = {RANDOM_SPLIT: "randsplit", **{c: "overlap" for c in OVERLAP},   # conditions with their own test set
             **{c: "labelfix" for c in LABEL_DIEDLOC}}                       # or their own labels
    out_paths = {None: priv / "predictions_tfidf.parquet", "randsplit": priv / "predictions_tfidf_randsplit.parquet",
                 "overlap": priv / "predictions_tfidf_overlap.parquet",
                 "labelfix": priv / "predictions_tfidf_labelfix.parquet"}
    done_all = {k: pd.read_parquet(p) if p.exists() else pd.DataFrame() for k, p in out_paths.items()}
    selection = {}
    sel_path = results_dir(cfg) / "tfidf_model_selection.json"
    if sel_path.exists():
        import json
        selection = json.load(open(sel_path))
    for cond in conditions:
        is_rand, grp = cond == RANDOM_SPLIT, group.get(cond)
        done, out_path = done_all[grp], out_paths[grp]
        if len(done) and cond in set(done.condition):
            continue
        split = rand_split if is_rand else v[OVERLAP[cond]].values if cond in OVERLAP else v.split.values
        idx = {s: np.flatnonzero(split == s) for s in ["train", "val", "test"]}
        t = time.time()
        text = condition_text(v, cond, inj_by_id, icd_by_id).fillna("")
        y = y_diedloc if cond in LABEL_DIEDLOC else v.label.values
        feat = Featurizer(tc)
        Xtr = feat.fit_transform(text.values[idx["train"]])
        Xva, Xte = feat.transform(text.values[idx["val"]]), feat.transform(text.values[idx["test"]])
        grid = []
        for C in tc["C_grid"]:
            clf = LogisticRegression(C=C, solver="liblinear", class_weight=tc["class_weight"],
                                     max_iter=tc["max_iter"], random_state=cfg["seed"])
            clf.fit(Xtr, y[idx["train"]])
            pv = clf.predict_proba(Xva)[:, 1]
            grid.append({"C": C, "val_auroc": roc_auc_score(y[idx["val"]], pv),
                         "val_auprc": average_precision_score(y[idx["val"]], pv), "clf": clf})
        best = max(grid, key=lambda g: g[f"val_{tc['selection_metric']}"])
        p = best["clf"].predict_proba(Xte)[:, 1]
        part = pd.DataFrame({"study_id": v.study_id.values[idx["test"]], "condition": cond, "p_mortality": p})
        if cond in LABEL_DIEDLOC:
            part["label_used"] = y[idx["test"]]
        if is_rand:   # does the test admission's patient also have admissions in the training split?
            train_subjects = set(v.subject_id.values[idx["train"]])
            part["patient_in_train"] = [s in train_subjects for s in v.subject_id.values[idx["test"]]]
        done = pd.concat([done, part], ignore_index=True)
        done_all[grp] = done
        done.to_parquet(out_path, index=False)
        selection[cond] = {"n_features_kept": int(len(feat.keep)), "nnz_per_train_doc": float(Xtr.nnz / Xtr.shape[0]),
                           "best_C": best["C"],
                           "grid": [{k: g[k] for k in ["C", "val_auroc", "val_auprc"]} for g in grid],
                           "seconds": round(time.time() - t)}
        write_json(selection, sel_path)
        print(cond, {k: selection[cond][k] for k in ["best_C", "n_features_kept", "seconds"]},
              "val_auroc", round(best["val_auroc"], 4), flush=True)
        del Xtr, Xva, Xte, feat
    idx = {s: np.flatnonzero(v.split.values == s) for s in ["train", "val", "test"]}
    write_json({"n_train": int(len(idx["train"])), "n_val": int(len(idx["val"])), "n_test": int(len(idx["test"])),
                "deaths_train": int(v.label.values[idx["train"]].sum()), "settings": {k: v_ for k, v_ in tc.items()}},
               results_dir(cfg) / "tfidf_settings.json")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
