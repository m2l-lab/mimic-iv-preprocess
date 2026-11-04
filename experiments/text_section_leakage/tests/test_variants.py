import pandas as pd
from conftest import STANDARD, note

from build_variants import assign_negative_control, process_chunk, split_t0


def test_process_chunk_builds_paired_variants():
    dying = STANDARD.replace("Patient was treated", "Patient expired on day 3 after being treated")
    hpi_leak = STANDARD.replace("In the ED he was given nebs.", "In the ED he was given nebs. The patient expired on ___.")
    recs, log = process_chunk([("A1", 1, STANDARD), ("A2", 2, dying), ("A3", 3, hpi_leak)])
    r1, r2, r3 = recs
    assert r1["t0_retained"] and r1["t1_core_nonempty"]
    assert r1["t1_text"] == r1["t2_text"]                       # nothing to screen
    assert "MARKER_BHC" in r1["inj_text"] and "MARKER_BHC" not in r1["t1_text"]
    assert r2["inj_cand_brief_hospital_course"] >= 1            # retrospective info lives in the injection
    assert "expired" in r3["t1_text"] and "expired" not in r3["t2_text"]
    assert any(row[0] == "A3" and row[6] for row in log)
    assert set(split_t0(r1["t0_text"])) >= {"chief_complaint", "history_of_present_illness"}


def test_negative_control_never_uses_same_patient():
    df = pd.DataFrame({"study_id": [f"A{i}" for i in range(8)], "subject_id": [1, 1, 2, 2, 3, 4, 5, 5],
                       "split": ["train"] * 8})
    donor = assign_negative_control(df, seed=0)
    subj = dict(zip(df.study_id, df.subject_id))
    for i, d in donor.items():
        assert subj[d] != df.at[i, "subject_id"]
