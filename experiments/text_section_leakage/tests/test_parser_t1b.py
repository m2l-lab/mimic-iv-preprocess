from conftest import STANDARD, note
from extract_sections import admission_spans, section_text
from extract_sections_t1b import admission_spans_t1b, phase_of_line


def t1b_text(text, name):
    spans, _ = admission_spans_t1b(text)
    return section_text(text, spans[name])


def with_exam(exam: str) -> str:
    return note("History of Present Illness:\nDyspnea.", "Past Medical History:\nCOPD",
                "Social History:\n___\nFamily History:\nnone", "Physical Exam:\n" + exam,
                "Pertinent Results:\nlab")


def test_standard_note_matches_t1():
    t1, _ = admission_spans(STANDARD)
    t1b, meta = admission_spans_t1b(STANDARD)
    for name in t1:
        assert section_text(STANDARD, t1[name]) == section_text(STANDARD, t1b[name]), name
    assert meta["pe_status"] == "labelled"


def test_discharge_heading_variants():
    for line in ["Physical Exam on Day of Discharge:", "Pertinent Examination at Discharge", "On the day of discharge",
                 "===PHYSICAL EXAM AT DISCHARGE===", "Dishcarge Physical Exam:", "DISCHARG EXAM",
                 "On discharge, SBP 120s, HR, O2 normal", "DISCHARGE  VS: 97.5, Tm 99.3, BP 133/84",
                 "On discharge (changes only):", "on discharge VITALS -   98.0 130/78", "DISCHARGE LABS:",
                 "On Discharge the is afebrile with stable VSS", "Weight at discharge: 48.5 kg"]:
        assert phase_of_line(line)[0] == "discharge", line


def test_death_and_later_exam_lines():
    for line in ["Death:", "Death Exam:", "TIME OF DEATH:", "Expired ___", "expired-", "Patient expired at 16:40",
                 "Pt expired", "patient was found to be expired", "On subsequent examination:", "Post-op exam:"]:
        assert phase_of_line(line)[0] == "discharge", line


def test_clinical_findings_are_not_headings():
    for line in ["HEENT: no exudates, no rhinorrhea/discharge", "Discharge: none", "Wound: no discharge",
                 "Discharge from the left ear noted", "GEN: NAD", "Neuro: no deficits", "Pupils equal"]:
        assert phase_of_line(line) is None, line


def test_missed_discharge_exam_and_death_exam_are_removed():
    text = with_exam("VS 98 120/80\nGEN: tachypneic\nPhysical Exam on Day of Discharge:\nGEN: comfortable MARK")
    assert "tachypneic" in t1b_text(text, "physical_exam") and "MARK" not in t1b_text(text, "physical_exam")
    text = with_exam("ADMISSION EXAM:\nGEN: intubated\nDeath Exam:\nno pulses MARK")
    assert "intubated" in t1b_text(text, "physical_exam") and "MARK" not in t1b_text(text, "physical_exam")


def test_decoration_fragment_dropped():
    text = with_exam("========================\nADMISSION EXAM\nGEN: NAD\n========\nDISCHARGE EXAM\nGEN: MARK")
    spans, meta = admission_spans_t1b(text)
    assert len(spans["physical_exam"]) == 1 and "decoration_dropped" in meta["pe_rules"]


def test_icu_admission_exam_after_first_exam_is_later():
    text = with_exam("ADMISSION EXAM:\nGEN: NAD\nMICU Admission Exam:\nGEN: intubated MARK")
    assert "MARK" not in t1b_text(text, "physical_exam")
    text = with_exam("MICU Admission Exam:\nGEN: intubated KEEP")
    assert "KEEP" in t1b_text(text, "physical_exam")


def test_deidentified_exam_heading():
    text = note("History of Present Illness:\nDyspnea.", "Social History:\n___\nFamily History:\nnone",
                "___ Exam:\nGEN: NAD\nDISCHARGE PHYSICAL EXAM:\nGEN: MARK", "Pertinent Results:\nlab")
    assert t1b_text(text, "family_history") == "none"
    assert t1b_text(text, "physical_exam") == "GEN: NAD"


def test_upper_case_heading_in_hpi_without_blank_line():
    text = note("History of Present Illness:\nDyspnea.\nPAST MEDICAL/SURGICAL HISTORY:\nCOPD\nSOCIAL HISTORY:\nsmoker",
                "Physical Exam:\nGEN: NAD")
    assert t1b_text(text, "history_of_present_illness") == "Dyspnea."
    assert t1b_text(text, "past_medical_history") == "COPD"
    assert t1b_text(text, "social_history") == "smoker"


def test_pmh_blocks_move_to_their_sections():
    text = note("History of Present Illness:\nDyspnea.",
                "Past Medical History:\nPMH: COPD\nPSH: appendectomy\nMeds: albuterol\nAll: shellfish\nSH: smoker",
                "Physical Exam:\nGEN: NAD")
    assert t1b_text(text, "past_medical_history") == "PMH: COPD PSH: appendectomy"
    assert t1b_text(text, "medications_on_admission") == "Meds: albuterol"
    assert t1b_text(text, "allergies") == "All: shellfish"
    assert t1b_text(text, "social_history") == "SH: smoker"


def test_hpi_ends_at_icu_course_subheading():
    text = note("History of Present Illness:\nDyspnea, intubated in the ED.\nMICU course:\nextubated MARK",
                "Past Medical History:\nCOPD")
    assert t1b_text(text, "history_of_present_illness") == "Dyspnea, intubated in the ED."


def test_inline_discharge_vitals_inside_admission_block_keep_admission_findings():
    text = with_exam("ADMISSION EXAM:\nDISCHARGE VS: 97.5, BP 133/84\nAdmit - BP 102/70, 102, 18\nGENERAL - laboured KEEP")
    kept = t1b_text(text, "physical_exam")
    assert "KEEP" in kept and "133/84" not in kept
    assert phase_of_line("Clinical Death exam performed by ___ team")[0] == "discharge"


def test_death_heading_variants_and_pronounced_findings():
    for line in ["Exam upon death:", "Physical exam on death: Time of death was 1410", "Physical exam at the time of death",
                 "Post death exam", "Discharge/Death exam:", "Discharge Physical Exam/Death exam:",
                 "Discharge/ Pronunciation of Death Exam", "Time of death 2:54AM", "Patient pronounced dead at 04:10"]:
        assert phase_of_line(line)[0] == "discharge", line
    for line in ["pronounced at LUQ", "pronounced when arms extended. Tangential.", "pronounced word finding difficulties",
                 "Pronounced JVD to the jaw"]:
        assert phase_of_line(line) is None, line


def test_v2_narrowing_from_held_out_check():
    # wrapped lines are not death lines or admission cues
    for line in ["deceased wife. HEENT: NC/AT", "died of MI at 60", "admission, no wheezing identified"]:
        assert phase_of_line(line) is None, line
    assert phase_of_line("DC Exam:")[0] == "discharge"
    # psychiatric-history items stay in the past history
    text = note("History of Present Illness:\nDepressed.",
                "Past Medical History:\nPast psych history: depression\nMedication and ECT trials: citalopram\n"
                "SUBSTANCE ABUSE HX: none\nHome medications: sertraline", "Physical Exam:\nGEN: NAD")
    assert "citalopram" in t1b_text(text, "past_medical_history")
    assert "SUBSTANCE" in t1b_text(text, "past_medical_history")
    assert t1b_text(text, "medications_on_admission") == "Home medications: sertraline"
    # arrival state stays in the HPI
    text = note("History of Present Illness:\nDyspnea.\nOn the floor: comfortable KEEP", "Past Medical History:\nCOPD")
    assert "KEEP" in t1b_text(text, "history_of_present_illness")
