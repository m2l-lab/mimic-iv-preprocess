"""Shared configuration and I/O helpers for the text-section leakage experiment.

Data-security rule for every script in this directory: never print or log note
text. Only counts, rates and identifiers-free aggregates go to stdout or to the
public report directory; anything row-level goes to ``private_dir``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

CONDITIONS = ["T0_COPIV", "T1_ROBUST", "T2_SCREENED", "T3_RETROSPECTIVE"]
NEG_CONTROL = "T4_OTHERPATIENT"

# Order and template labels of the admission-related sections. The labels match
# the COP-IV / CORe input format so that T0 and T1 differ in content only.
ADMISSION_SECTIONS = [
    ("chief_complaint", "CHIEF COMPLAINT"),
    ("history_of_present_illness", "PRESENT ILLNESS"),
    ("past_medical_history", "MEDICAL HISTORY"),
    ("medications_on_admission", "MEDICATION ON ADMISSION"),
    ("allergies", "ALLERGIES"),
    ("physical_exam", "PHYSICAL EXAM"),
    ("family_history", "FAMILY HISTORY"),
    ("social_history", "SOCIAL HISTORY"),
]

INJECTION_LABELS = {
    "discharge_condition": "DISCHARGE CONDITION",
    "discharge_diagnosis": "DISCHARGE DIAGNOSIS",
    "brief_hospital_course": "HOSPITAL COURSE",
}


def load_config(path: Path | None = None, *, create_dirs: bool = True) -> dict:
    path = Path(path or os.environ.get("TSL_CONFIG", HERE / "config.yaml")).expanduser().resolve()
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key, rel in cfg["paths"].items():
        p = (path.parent / rel).resolve()
        if create_dirs:
            p.mkdir(parents=True, exist_ok=True)
        cfg["paths"][key] = p
    if create_dirs and cfg["paths"]["private_dir"].exists():
        os.chmod(cfg["paths"]["private_dir"], 0o700)
    if "TSL_MIMIC_ROOT" in os.environ:
        cfg["data"]["mimic_root"] = os.environ["TSL_MIMIC_ROOT"]
    return cfg


def mimic_path(cfg: dict, key: str) -> Path:
    return Path(cfg["data"]["mimic_root"]) / cfg["data"][key]


def results_dir(cfg: dict) -> Path:
    p = cfg["paths"]["report_dir"] / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
