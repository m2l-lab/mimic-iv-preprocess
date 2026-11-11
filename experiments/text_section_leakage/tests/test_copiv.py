"""Characterisation tests for the verbatim COP-IV T0 extraction."""
import pandas as pd
from conftest import STANDARD, note

from copiv_original import filter_notes


def run_t0(texts):
    df = pd.DataFrame({"subject_id": range(len(texts)), "text": texts})
    return filter_notes(df.copy(), admission_text_only=True)


def test_t0_template_and_sections():
    out = run_t0([STANDARD])
    assert len(out) == 1
    text = out.iloc[0]
    assert text.startswith("CHIEF COMPLAINT: shortness of breath\n\nPRESENT ILLNESS: ")
    assert "MEDICAL HISTORY: COPD HTN" in text
    # blank line between the admission and discharge exam ends the T0 physical exam
    assert "MARKER_DISCHARGE_EXAM" not in text


def test_t0_includes_discharge_exam_without_blank_line():
    text = note("Chief Complaint:\ncough", "History of Present Illness:\nhpi",
                "Physical Exam:\nADMISSION EXAM:\nok\nDISCHARGE EXAM:\nDCMARK", "Pertinent Results:\nlab")
    assert "DCMARK" in run_t0([text]).iloc[0]


def test_t0_misses_deidentified_chief_complaint_and_drops_note_without_core_sections():
    deid = note("___ Complaint:\ncough", "Social History:\n___")
    assert len(run_t0([deid])) == 0
