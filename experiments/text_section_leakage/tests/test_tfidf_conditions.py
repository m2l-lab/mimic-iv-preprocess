import numpy as np
import pandas as pd

from extract_sections import render
from train_tfidf_baseline import labelled_pe_only, random_admission_split, without_section


def test_without_section_removes_only_that_section():
    t = render({"chief_complaint": "cough", "history_of_present_illness": "hpi words",
                "physical_exam": "exam words MARK", "social_history": "smoker"})
    out = without_section(pd.Series([t]), "physical_exam").iloc[0]
    assert "MARK" not in out and "PHYSICAL EXAM" not in out
    assert "PRESENT ILLNESS: hpi words" in out and "SOCIAL HISTORY: smoker" in out


def test_random_admission_split_proportions():
    s = random_admission_split(np.arange(1000), {"train": 0.7, "val": 0.15, "test": 0.15}, 42)
    assert (s == "train").sum() == 700 and (s == "val").sum() == 150 and (s == "test").sum() == 150


def test_labelled_pe_only_blanks_other_statuses_keeps_heading():
    t = render({"chief_complaint": "cough", "physical_exam": "exam MARK"})
    out = labelled_pe_only(pd.Series([t, t]), pd.Series(["labelled", "unlabelled"]))
    assert "PHYSICAL EXAM: exam MARK" in out.iloc[0]
    assert "MARK" not in out.iloc[1] and "PHYSICAL EXAM: " in out.iloc[1] and "cough" in out.iloc[1]


def test_overlap_splits_match_size_deaths_and_keep_eval_fixed():
    from patient_overlap import overlap_splits
    rng = np.random.default_rng(0)
    rows = []
    for s in range(400):                      # patients with 1-3 admissions
        split = ["train", "train", "train", "val", "test"][s % 5]
        for k in range(1 + s % 3):
            rows.append({"study_id": f"{s}_{k}", "subject_id": s, "split": split,
                         "label": int(rng.random() < 0.2), "admittime": pd.Timestamp("2020-01-01") + pd.Timedelta(days=k)})
    d = pd.DataFrame(rows)
    o = overlap_splits(d, 1).merge(d, on="study_id")
    assert (o.ovl_a == "test").equals(o.ovl_b == "test")                         # same evaluation set
    ev = o[o.ovl_a == "test"]
    assert ev.subject_id.is_unique and (ev.split == "test").all()
    ta, tb = o[o.ovl_a == "train"], o[o.ovl_b == "train"]
    assert len(ta) == len(tb) and ta.label.sum() == tb.label.sum()
    assert not set(ta.subject_id) & set(ev.subject_id)                         # A: no overlap
    assert set(ev.subject_id) <= set(tb.subject_id)                            # B: every eval patient seen
    assert (o.ovl_a == "val").equals(o.split == "val")
