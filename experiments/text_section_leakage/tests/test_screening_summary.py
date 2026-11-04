import json
from multiprocessing.pool import ThreadPool

import pandas as pd

from conftest import STANDARD
from build_variants import process_chunk
import summarize_screening


def test_automated_summary_uses_only_generated_variants_and_screening_log(tmp_path, monkeypatch):
    private = tmp_path / "private"
    private.mkdir()
    report = tmp_path / "report"
    report.mkdir()
    death = STANDARD.replace("In the ED he was given nebs.", "The patient expired on ___.")
    course = STANDARD.replace("In the ED he was given nebs.", "During this admission developed AKI.")
    records, log = process_chunk([("A1", 1, STANDARD), ("A2", 2, death), ("A3", 3, course)])
    variants = pd.DataFrame(records).assign(
        split="test", in_analysis=True, t4_donor=["A2", "A3", "A1"])
    variants.to_parquet(private / "variants.parquet", index=False)
    pd.DataFrame({"study_id": ["A1", "A2", "A3"], "label": [0, 1, 0]}).to_parquet(
        private / "cohort.parquet", index=False)
    pd.DataFrame(log, columns=["study_id", "section", "note_start", "note_end", "rule_id",
                              "category", "removed", "reason"]).to_parquet(
        private / "screening_log.parquet", index=False)
    monkeypatch.setattr(summarize_screening, "Pool", ThreadPool)

    summarize_screening.candidates({"paths": {"private_dir": private, "report_dir": report}})

    results = report / "results"
    summary = pd.read_csv(results / "table_a_candidates.csv").set_index("condition")
    assert set(summary.index) == {
        "T0_COPIV", "T1_ROBUST", "T2_SCREENED", "T3_RETROSPECTIVE", "T4_OTHERPATIENT"}
    assert summary.admissions.eq(3).all()
    assert summary.loc["T1_ROBUST", "cand_outcome_pct_died"] == 100
    assert summary.loc["T2_SCREENED", "cand_any_pct_all"] == 0
    assert json.loads((results / "candidates_meta.json").read_text())["n_analysis"] == 3
    for name in ["t1_candidates_by_section.csv", "t1_candidates_by_rule.csv", "candidates_by_pe_status.csv"]:
        assert (results / name).is_file()
    assert {p.name for p in private.iterdir()} == {
        "cohort.parquet", "variants.parquet", "screening_log.parquet"}
