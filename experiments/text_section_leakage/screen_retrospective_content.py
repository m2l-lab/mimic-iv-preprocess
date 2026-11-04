"""T2 temporal screening: remove sentences that clearly refer to events after
the prediction anchor (hospital admission) or disclose the eventual outcome.

The rules are deliberately conservative (precision over recall) and were fixed
a priori from clinical-documentation knowledge, without inspecting note text
and without using mortality labels. ``RULESET_VERSION`` is a hash of the rule
definitions and is recorded with every output so that the frozen rule set can
be verified before and after test evaluation.

Unit of removal: one sentence. Removed sentences are deleted without a marker,
because a marker would itself become a label-correlated feature.

Screening categories:
  outcome       explicit disclosure of the patient's eventual outcome (death, CMO)
  post_anchor   clearly post-admission clinical information (discharge-time
                findings, hospital-day references, events "during this admission")
Matches that are excluded (relative's death, family-history section) are logged
with ``removed=False`` so their frequency can be audited.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

RELATIVE = (r"(?:mother|father|mom|dad|parents?|brother|sister|siblings?|son|daughter|children|child|"
            r"husband|wife|spouse|partner|boyfriend|girlfriend|fianc[eé]e?|aunt|uncle|cousin|niece|nephew|"
            r"grand(?:mother|father|parent|son|daughter|child)s?|friend|relatives?|family members?|"
            r"step(?:mother|father|son|daughter))")
DEATH_VERB = r"(?:expired|passed\s+away|died|deceased|demised)"

# (rule_id, category, pattern, sections where the rule is NOT applied)
RULES: list[tuple[str, str, str, frozenset[str]]] = [
    ("O1_patient_death", "outcome",
     rf"\b(?:the\s+)?(?:pt|patient|he|she|mr|mrs|ms|miss)\b\.?(?:\s+\S+){{0,6}}?\s+(?:was\s+|had\s+)?{DEATH_VERB}\b",
     frozenset({"family_history"})),
    ("O2_pronounced", "outcome",
     r"\b(?:pronounced\s+(?:dead|deceased|at\s+\S+)|time\s+of\s+death|death\s+was\s+pronounced|"
     r"declared\s+dead)\b",
     frozenset({"family_history"})),
    ("O3_death_circumstance", "outcome",
     rf"\b{DEATH_VERB}\s+(?:peacefully|comfortably|surrounded\s+by|with\s+(?:her|his|the)?\s*family|"
     r"in\s+the\s+(?:icu|micu|sicu|ccu|hospital)|on\s+hospital\s+day)",
     frozenset({"family_history"})),
    ("O4_autopsy", "outcome",
     r"\b(?:autopsy|post-?mortem)\b",
     frozenset({"family_history"})),
    ("O5_comfort_transition", "outcome",
     r"\b(?:made|transitioned\s+to|changed\s+to|converted\s+to|decision\s+(?:was\s+made\s+)?to\s+(?:pursue|transition\s+to))\s+"
     r"(?:cmo|comfort(?:\s+(?:measures|care|focused\s+care))?(?:\s+only)?|hospice)\b",
     frozenset()),
    ("P1_discharge_reference", "post_anchor",
     r"\b(?:on|at|upon|prior\s+to|before|at\s+the\s+time\s+of|at\s+time\s+of|day\s+of)\s+discharge\b|"
     r"\bdischarge\s+(?:exam|physical|vitals|vital\s+signs|weight|labs|pe)\b|"
     r"\b(?:will\s+be|is\s+being|was\s+subsequently|ultimately|eventually)\s+discharged\b",
     frozenset()),
    ("P2_hospital_day", "post_anchor",
     r"\b(?:hospital\s+day|HD|POD|post-?op(?:erative)?\s+day)\s*#?\s*\d+\b|"
     r"\bon\s+(?:the\s+)?(?:second|third|fourth|fifth|sixth|seventh|\d+(?:st|nd|rd|th))\s+(?:hospital\s+)?day\s+of\s+"
     r"(?:admission|hospitali[sz]ation|(?:her|his|the)\s+stay)\b",
     frozenset()),
    ("P3_during_this_admission", "post_anchor",
     r"\b(?:during|throughout|over\s+the\s+course\s+of|later\s+in|for\s+the\s+remainder\s+of)\s+"
     r"(?:this|the\s+current|her\s+current|his\s+current)\s+(?:admission|hospitali[sz]ation|hospital\s+(?:stay|course)|stay)\b|"
     r"\b(?:hospital|icu|micu|sicu|ccu)\s+course\s+(?:was|has\s+been)\s+(?:complicated|notable|significant)\b",
     frozenset()),
    ("P4_icu_callout", "post_anchor",
     r"\bcalled\s+out\s+(?:to|of)\b|\btransferred\s+(?:back\s+)?(?:out\s+of\s+the\s+(?:icu|micu|sicu|ccu|csru)|to\s+the\s+floor)\b",
     frozenset()),
]
COMPILED = [(rid, cat, re.compile(p, re.IGNORECASE), excl) for rid, cat, p, excl in RULES]
RELATIVE_DEATH_RE = re.compile(rf"\b{RELATIVE}\b(?:\s+\S+){{0,6}}?\s+(?:who\s+)?(?:has\s+|had\s+|was\s+)?{DEATH_VERB}\b",
                               re.IGNORECASE)
# A relative's death in the same sentence does not mask an explicit patient death.
PATIENT_DEATH_RE = re.compile(r"\b(?:pt|patient)\s+(?:\w+\s+){0,2}?(?:expired|passed\s+away|died)\b", re.IGNORECASE)
DEATH_RULES = {"O1_patient_death", "O3_death_circumstance"}
RULESET_VERSION = hashlib.sha256(
    repr(RULES).encode() + RELATIVE_DEATH_RE.pattern.encode() + PATIENT_DEATH_RE.pattern.encode()).hexdigest()[:12]

# Sentence boundaries: newlines, or [.!?;] followed by whitespace and an upper-case
# letter / digit / de-identification placeholder.
SENT_BOUNDARY_RE = re.compile(r"\n+|(?<=[.!?;])\s+(?=[A-Z0-9_\[(])")


@dataclass
class Match:
    section: str
    start: int          # offset of the sentence within the section text passed in
    end: int
    rule_id: str
    category: str
    removed: bool
    reason: str = ""


def sentences(text: str) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for m in SENT_BOUNDARY_RE.finditer(text):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return spans


def screen_text(section: str, text: str, categories: frozenset[str] | None = None) -> tuple[str, list[Match]]:
    """Return (screened text, matches). Offsets refer to ``text``. ``categories`` restricts the rules
    applied (e.g. {"outcome"}); default: all rules."""
    keep, matches = [], []
    for a, b in sentences(text):
        sent = text[a:b]
        hits = []
        for rid, cat, rx, excluded_sections in COMPILED:
            if categories is not None and cat not in categories:
                continue
            if not rx.search(sent):
                continue
            if section in excluded_sections:
                matches.append(Match(section, a, b, rid, cat, False, "excluded_section"))
                continue
            if rid in DEATH_RULES and RELATIVE_DEATH_RE.search(sent) and not PATIENT_DEATH_RE.search(sent):
                matches.append(Match(section, a, b, rid, cat, False, "relative_death"))
                continue
            hits.append((rid, cat))
        if hits:
            matches.extend(Match(section, a, b, rid, cat, True) for rid, cat in hits)
        else:
            keep.append(sent)
    return " ".join(s.strip() for s in keep if s.strip()), matches


def detect(text: str, section: str = "_any_") -> list[Match]:
    """Detection only (no removal); used to quantify candidates in any condition."""
    return [m for m in screen_text(section, text)[1] if m.removed]
