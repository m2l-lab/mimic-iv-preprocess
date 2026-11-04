import numpy as np

from run_core_inference import Condition, PART_RE, part_labels, section_reach


class Tok:
    cls_token_id, sep_token_id = 101, 102


def test_paired_conditions_share_admission_prefix_and_respect_budgets():
    base = [np.arange(1000, 1500, dtype=np.int32), np.arange(5, dtype=np.int32)]
    inj = [np.arange(9000, 9300, dtype=np.int32), np.array([], dtype=np.int32)]
    t1, t3 = Condition(Tok, base, 382), Condition(Tok, base, 382, inj, 128)
    assert len(t1[0]) == 384 and len(t3[0]) == 512
    assert np.array_equal(t3[0][:383], t1[0][:383])          # identical [CLS] + 382 admission tokens
    assert t3[0][383] == 9000 and t3[0][-1] == 102
    assert list(t3.lengths()) == [512, 7] and list(t1.lengths()) == [384, 7]


def test_part_labels_and_reach():
    text = "CHIEF COMPLAINT: a\n\nPRESENT ILLNESS: \n\nMEDICAL HISTORY: b c"
    assert len(PART_RE.split(text)) == 3
    names = part_labels([text])
    assert names[0][1] == ("PRESENT ILLNESS", False)
    reach = section_reach([[4, 3, 6]], 8, ["CHIEF COMPLAINT", "MEDICAL HISTORY"], names)
    assert reach["CHIEF COMPLAINT"]["full"] == 100 and reach["MEDICAL HISTORY"]["partial"] == 100
