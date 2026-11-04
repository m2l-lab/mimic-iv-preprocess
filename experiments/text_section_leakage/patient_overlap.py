"""Patient-overlap experiment with a fixed evaluation set (redesign of the random-split comparison).

Evaluation set E: for every test-split patient with >= 2 analysis admissions, one admission drawn at
random (seeded). Two training sets of identical size and identical death count:
  A (ovl_a): the patient-level training split. The other admissions of the evaluation patients
             ("siblings") are excluded.
  B (ovl_b): the same training split with the siblings added and an equal number of training admissions
             removed. The removed admissions are drawn from training patients who also have >= 2
             admissions, stratified by outcome, so that the two sets match in size, deaths and
             repeat-patient status.
Validation (model selection) is the unchanged patient-level validation split in both arms. A and B are
evaluated on exactly the same admissions, so the comparison is paired.

Columns: ovl_a / ovl_b in {train, val, test, excluded}; eval_is_last (the evaluation admission is the
patient's last analysis admission, i.e. all siblings precede it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def overlap_splits(d: pd.DataFrame, seed: int) -> pd.DataFrame:
    """d: one row per admission with study_id, subject_id, split, label, admittime."""
    rng = np.random.default_rng(seed)
    d = d.reset_index(drop=True)
    n_adm = d.groupby("subject_id").study_id.transform("size")
    test_multi = d[(d.split == "test") & (n_adm >= 2)]
    order = rng.permutation(len(test_multi))
    eval_idx = test_multi.iloc[order].groupby("subject_id").head(1).index        # one random admission each
    is_eval = d.index.isin(eval_idx)
    sibling = (d.split == "test") & (n_adm >= 2) & ~is_eval
    pool = d[(d.split == "train") & (n_adm >= 2)]
    removed = []
    for lab, k in d[sibling].label.value_counts().items():
        cand = pool.index[pool.label.values == lab]
        removed.append(rng.choice(cand, k, replace=False))
    removed = np.concatenate(removed) if removed else np.array([], dtype=int)

    out = pd.DataFrame({"study_id": d.study_id})
    base = np.where(d.split.isin(["train", "val"]), d.split, "excluded")
    out["ovl_a"] = np.where(is_eval, "test", base)
    b = out.ovl_a.values.copy()
    b[sibling.values] = "train"
    b[removed] = "excluded"
    out["ovl_b"] = b
    last = d.groupby("subject_id").admittime.transform("max")
    out["eval_is_last"] = is_eval & (d.admittime == last).values
    return out
