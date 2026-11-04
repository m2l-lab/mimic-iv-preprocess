from conftest import B, STANDARD, note
from extract_sections import admission_spans, later_sections, parse_note, render, section_text


def t1(text):
    spans, meta = admission_spans(text)
    return {k: section_text(text, v) for k, v in spans.items()}, meta


def test_standard_sections_found():
    names = [s.name for s in parse_note(STANDARD).sections]
    for n in ["chief_complaint", "history_of_present_illness", "past_medical_history", "social_history",
              "family_history", "physical_exam", "brief_hospital_course", "medications_on_admission",
              "discharge_diagnosis", "discharge_condition", "followup_instructions"]:
        assert n in names


def test_no_discharge_content_leaks_into_admission_view():
    secs, meta = t1(STANDARD)
    joined = render(secs)
    for marker in ["MARKER_DISCHARGE_EXAM", "MARKER_RESULTS", "MARKER_BHC", "MARKER_DCMEDS",
                   "MARKER_DXDX", "MARKER_DCCOND", "MARKER_INSTR"]:
        assert marker not in joined, marker
    assert meta["pe_status"] == "labelled"
    assert "tachypneic" in secs["physical_exam"]


def test_heading_inside_hospital_course_is_not_a_section():
    secs, _ = t1(STANDARD)
    assert secs["chief_complaint"] == "shortness of breath"
    assert "inside the course" not in secs["chief_complaint"]


def test_family_history_after_placeholder_social_history():
    secs, _ = t1(STANDARD)
    assert secs["social_history"] == "___"
    assert secs["family_history"].startswith("Mother died")


def test_aliases_and_deidentified_headings():
    text = note("___ Complaint:\nchest pain", "History of the Present Illness:\nhpi text",
                "Physical ___:\nexam text", "Brief Hospital Course:\ncourse", "___ on Admission:\naspirin")
    secs, meta = t1(text)
    assert secs["chief_complaint"] == "chest pain"
    assert secs["history_of_present_illness"] == "hpi text"
    assert secs["physical_exam"] == "exam text"
    assert meta["pe_status"] == "unlabelled"
    assert secs["medications_on_admission"] == "aspirin"


def test_mixed_case_whitespace_and_inline_content():
    text = note("CHIEF COMPLAINT:   fever  ", "history of present illness :\n  multi\n  line\n  hpi")
    secs, _ = t1(text)
    assert secs["chief_complaint"] == "fever"
    assert secs["history_of_present_illness"] == "multi line hpi"


def test_missing_and_empty_sections():
    text = note("Chief Complaint:\n", "History of Present Illness:\nonly hpi")
    secs, _ = t1(text)
    assert secs["chief_complaint"] == ""
    assert secs["past_medical_history"] == ""
    assert secs["history_of_present_illness"] == "only hpi"


def test_last_section_without_following_heading():
    text = note("Chief Complaint:\ncough", "Past Medical History:\nasthma\nGERD")
    secs, _ = t1(text)
    assert secs["past_medical_history"] == "asthma GERD"


def test_out_of_order_section_is_kept_with_strict_evidence():
    text = note("History of Present Illness:\nhpi", "Brief Hospital Course:\ncourse", "Chief Complaint:\nlate cc")
    parsed = parse_note(text)
    cc = parsed.by_name("chief_complaint")
    assert len(cc) == 1 and cc[0].out_of_order
    secs, meta = t1(text)
    assert secs["chief_complaint"] == "late cc"
    assert "course" not in secs["chief_complaint"]
    assert meta["n_out_of_order"] == 1


def test_out_of_order_inline_heading_is_rejected():
    text = note("History of Present Illness:\nhpi", "Brief Hospital Course:\ncourse line\nAllergies: none known later")
    secs, _ = t1(text)
    assert secs["allergies"] == ""


def test_repeated_empty_heading_then_filled():
    text = note("Past Medical History:", "Past Medical History:\nCHF", "Social History:\nsmoker")
    parsed = parse_note(text)
    assert [s.repeated for s in parsed.by_name("past_medical_history")] == [False, True]
    secs, meta = t1(text)
    assert secs["past_medical_history"] == "CHF"
    assert meta["n_repeated"] == 1


def test_similar_non_heading_phrases_are_not_headings():
    text = note("History of Present Illness:\nShe has a family history: of diabetes in mother.\n"
                "Social history notable for: smoking", "Past Medical History:\nDM")
    secs, _ = t1(text)
    assert "smoking" in secs["history_of_present_illness"]
    assert secs["family_history"] == ""


def test_pe_variants():
    for sub in ["On discharge:", "DISCHARGE EXAM:", "Physical exam on discharge:", "Discharge PE:", "DISCHARGE"]:
        text = note(f"Physical Exam:\nAdmission exam:\nadm findings\n{sub}\nDCMARK")
        secs, meta = t1(text)
        assert "adm findings" in secs["physical_exam"] and "DCMARK" not in secs["physical_exam"], sub
        assert meta["pe_status"] == "labelled"
    text = note("Physical Exam:\nDischarge exam:\nDCMARK", "Pertinent Results:\nlab")
    secs, meta = t1(text)
    assert secs["physical_exam"] == "" and meta["pe_status"] == "discharge_only"


def test_addendum_text_after_last_section_does_not_enter_admission_view():
    text = STANDARD + f"{B}\nAddendum:\nMARKER_ADDENDUM patient later expired\n"
    secs, _ = t1(text)
    assert "MARKER_ADDENDUM" not in render(secs)


def test_later_sections_for_injection():
    inj = later_sections(STANDARD, ["discharge_condition", "discharge_diagnosis", "brief_hospital_course"])
    assert "MARKER_DCCOND" in inj["discharge_condition"]
    assert "MARKER_DXDX" in inj["discharge_diagnosis"]
    assert "MARKER_BHC" in inj["brief_hospital_course"]


def test_offsets_point_into_original_note():
    spans, _ = admission_spans(STANDARD)
    a, b = spans["chief_complaint"][0]
    assert STANDARD[a:b].strip() == "shortness of breath"
