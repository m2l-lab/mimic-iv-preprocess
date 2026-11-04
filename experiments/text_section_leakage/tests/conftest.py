"""Synthetic discharge summaries in the MIMIC-IV-Note template (no real data)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# MIMIC-IV notes use lines containing a single space as blank lines.
B = " "


def note(*blocks: str) -> str:
    return f"\n{B}\n".join(blocks) + "\n"


STANDARD = note(
    "Name:  ___                     Unit No:   ___",
    "Admission Date:  ___              Discharge Date:   ___",
    "Service: MEDICINE",
    "Allergies: \nPenicillins",
    "Attending: ___.",
    "Chief Complaint:\nshortness of breath",
    "Major Surgical or Invasive Procedure:\nintubation",
    "History of Present Illness:\nMr. ___ is a 70M with COPD who presents with dyspnea for 3 days.\nIn the ED he was given nebs.",
    "Past Medical History:\nCOPD\nHTN",
    "Social History:\n___\nFamily History:\nMother died of MI at 60.",
    "Physical Exam:\nADMISSION PHYSICAL EXAM:\nVS: 98.0 120/80\nGEN: tachypneic\n \nDISCHARGE PHYSICAL EXAM:\nGEN: comfortable, MARKER_DISCHARGE_EXAM",
    "Pertinent Results:\n___ 07:00AM WBC-12 MARKER_RESULTS",
    "Brief Hospital Course:\nMARKER_BHC Patient was treated for COPD exacerbation.\nChief Complaint: this line is inside the course",
    "Medications on Admission:\n1. Albuterol inh",
    "Discharge Medications:\n1. Prednisone MARKER_DCMEDS",
    "Discharge Disposition:\nHome",
    "Discharge Diagnosis:\nCOPD exacerbation MARKER_DXDX",
    "Discharge Condition:\nMental Status: Clear and coherent. MARKER_DCCOND",
    "Discharge Instructions:\nMARKER_INSTR",
    "Followup Instructions:\n___",
)


@pytest.fixture
def standard_note():
    return STANDARD
