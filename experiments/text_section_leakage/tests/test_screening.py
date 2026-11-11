from screen_retrospective_content import RULESET_VERSION, detect, screen_text, sentences


def removed_rules(section, text):
    return {m.rule_id for m in screen_text(section, text)[1] if m.removed}


def test_explicit_patient_death_is_removed():
    text = "Pt admitted with sepsis. The patient expired on ___ with family at bedside. Initially on pressors."
    out, matches = screen_text("history_of_present_illness", text)
    assert "expired" not in out
    assert "admitted with sepsis" in out and "Initially on pressors" in out
    assert any(m.category == "outcome" and m.removed for m in matches)


def test_pronounced_and_autopsy():
    assert "O2_pronounced" in removed_rules("physical_exam", "Pronounced dead at 04:10. No pulses.")
    assert "O4_autopsy" in removed_rules("history_of_present_illness", "Family declined autopsy.")


def test_relative_death_is_not_removed():
    text = "She reports her mother died of breast cancer. She has chest pain."
    out, matches = screen_text("history_of_present_illness", text)
    assert out == text
    assert matches and all(not m.removed and m.reason == "relative_death" for m in matches)


def test_patients_husband_died_is_not_patient_death():
    out, matches = screen_text("social_history", "Patient's husband died last year. Lives alone.")
    assert "husband died" in out
    assert not any(m.removed for m in matches)


def test_family_history_section_is_exempt_from_death_rules():
    text = "Father died of MI at 55. Brother passed away from colon cancer."
    out, matches = screen_text("family_history", text)
    assert out == text
    assert all(m.reason == "excluded_section" for m in matches)


def test_historical_events_are_kept():
    text = "Prior admission in ___ for CHF exacerbation. History of cardiac arrest in ___ with ROSC."
    assert removed_rules("past_medical_history", text) == set()


def test_post_anchor_references():
    assert "P1_discharge_reference" in removed_rules("physical_exam", "On discharge, afebrile and comfortable.")
    assert "P2_hospital_day" in removed_rules("history_of_present_illness", "On HD #3 she was intubated.")
    assert "P3_during_this_admission" in removed_rules(
        "history_of_present_illness", "During this admission she developed AKI.")
    assert "P4_icu_callout" in removed_rules("history_of_present_illness", "She was called out to the floor.")
    assert "O5_comfort_transition" in removed_rules(
        "history_of_present_illness", "After a family meeting she was transitioned to comfort measures only.")


def test_multiple_temporal_references_in_one_text():
    text = ("Presented with fever two days prior to admission.\nIn the ED she received antibiotics.\n"
            "On hospital day 2 she became hypotensive. Patient later died.")
    out, matches = screen_text("history_of_present_illness", text)
    assert "two days prior to admission" in out and "In the ED" in out
    assert "hospital day 2" not in out and "died" not in out
    assert {m.rule_id for m in matches if m.removed} >= {"P2_hospital_day", "O1_patient_death"}


def test_no_removal_marker_is_left():
    out, _ = screen_text("history_of_present_illness", "A. The patient expired. B.")
    assert out == "A. B."


def test_sentence_split_keeps_all_text():
    text = "One. Two!\nThree; four. ___ five"
    joined = "".join(text[a:b] for a, b in sentences(text))
    assert joined.replace(" ", "").replace("\n", "") == text.replace(" ", "").replace("\n", "")


def test_detect_and_version():
    assert detect("The patient passed away peacefully.")
    assert not detect("No acute distress.")
    assert len(RULESET_VERSION) == 12


def test_screen_text_category_filter_keeps_post_anchor_sentences():
    from screen_retrospective_content import screen_text
    text = "On hospital day 3 she was extubated. The patient expired on hospital day 5."
    kept, matches = screen_text("brief_hospital_course", text, frozenset({"outcome"}))
    assert "extubated" in kept and "expired" not in kept
    assert {m.category for m in matches} == {"outcome"}
