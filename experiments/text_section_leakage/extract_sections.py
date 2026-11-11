"""T1 robust section parser for MIMIC-IV discharge summaries.

Design (derived from an aggregate heading inventory of all 331,793 MIMIC-IV-Note
2.2 discharge summaries):

* An explicit inventory of the top-level template headings, including
  de-identified variants such as ``___ Complaint`` or ``Physical ___``.
* A heading is only accepted at a plausible position: at line start, followed
  by a colon, and preceded by a blank line, a de-identification placeholder
  line, or the start of the document. ``Family History`` is exempt from the
  blank-line requirement because it usually follows the Social History line
  directly (98% of notes).
  (It still needs Social History to have been seen, or a placeholder line.)
* Order awareness: a heading whose template position lies *before* a section
  that has already been accepted (an out-of-order or repeated heading) is only
  accepted with strict evidence (blank line before it, exact heading, nothing
  inline). Otherwise it is treated as content, so heading-like phrases inside
  the hospital course cannot open a new admission section.
* Every heading of the inventory, including discharge-phase headings, closes
  the preceding section. The final section runs to the end of the note.
* Physical Exam is split at admission / discharge sub-headings; only the
  admission part (or an unlabelled exam) is kept for the admission view.

All sections carry character offsets into the original note text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from common import ADMISSION_SECTIONS

# canonical name -> heading aliases (lower case, without the trailing colon).
# Order = MIMIC-IV discharge template order.
HEADING_INVENTORY: list[tuple[str, list[str]]] = [
    ("service", ["service"]),
    ("allergies", ["allergies", "allergy"]),
    ("attending", ["attending"]),
    ("chief_complaint", ["chief complaint", "___ complaint", "chief ___"]),
    ("major_procedure", ["major surgical or invasive procedure", "major surgical or invasive procedures"]),
    ("history_of_present_illness", ["history of present illness", "history of the present illness",
                                    "history of presenting illness", "___ of present illness"]),
    ("past_medical_history", ["past medical history", "past medical and surgical history",
                              "past medical/surgical history", "___ medical history"]),
    ("social_history", ["social history", "___ history"]),
    ("family_history", ["family history"]),
    ("physical_exam", ["physical exam", "physical examination", "physical ___"]),
    ("pertinent_results", ["pertinent results"]),
    ("brief_hospital_course", ["brief hospital course", "hospital course", "___ hospital course"]),
    ("medications_on_admission", ["medications on admission", "___ on admission", "medications on ___"]),
    ("discharge_medications", ["discharge medications", "___ medications"]),
    ("discharge_disposition", ["discharge disposition", "___ disposition"]),
    ("facility", ["facility"]),
    ("discharge_diagnosis", ["discharge diagnosis", "discharge diagnoses", "___ diagnosis", "___ diagnoses"]),
    ("discharge_condition", ["discharge condition", "___ condition"]),
    ("discharge_instructions", ["discharge instructions", "___ instructions"]),
    ("followup_instructions", ["followup instructions", "follow-up instructions", "follow up instructions"]),
]
ORDER = {name: i for i, (name, _) in enumerate(HEADING_INVENTORY)}
ALIAS_TO_CANONICAL = {alias: name for name, aliases in HEADING_INVENTORY for alias in aliases}
# "___ history" is ambiguous; it is only accepted as Social History in its template slot.
AMBIGUOUS_ALIASES = {"___ history", "___ medications", "___ condition", "___ instructions",
                     "___ disposition", "___ diagnoses", "chief ___", "medications on ___"}
NO_BLANK_LINE_REQUIRED = {"family_history"}

_alias_alt = "|".join(sorted((re.escape(a) for a in ALIAS_TO_CANONICAL), key=len, reverse=True))
HEADING_RE = re.compile(rf"^[ \t]*(?P<head>{_alias_alt})[ \t]*:[ \t]*(?P<inline>.*)$", re.IGNORECASE)
PLACEHOLDER_LINE_RE = re.compile(r"^[\s_\-\*\.]*$")

# Physical-exam sub-headings. Matched only at line start and only on short lines.
_PE = r"(?:physical\s+)?(?:exam(?:ination)?|pe|physical)"
DISCHARGE_SUB_RE = re.compile(
    rf"^[ \t\*=_\-]*(?:"
    rf"(?:discharge|d/c|dc)\s+{_PE}(?:\s+and\s+(?:vitals|labs))?"
    rf"|{_PE}\s+(?:on|at|upon|prior\s+to|at\s+time\s+of|at\s+the\s+time\s+of|day\s+of)\s+discharge"
    rf"|(?:on|at|upon|prior\s+to|at\s+time\s+of|at\s+the\s+time\s+of)\s+discharge"
    rf"|discharge\s+(?:vitals|vs|exam\s+and\s+vitals)"
    rf"|discharge|day\s+of\s+discharge"
    rf")[ \t]*(?::|-|$)",
    re.IGNORECASE,
)
ADMISSION_SUB_RE = re.compile(
    rf"^[ \t\*=_\-]*(?:"
    rf"(?:admission|admit|admitting|initial|presenting)\s+{_PE}"
    rf"|{_PE}\s+(?:on|at|upon)\s+(?:admission|presentation|arrival)"
    rf"|(?:on|at|upon)\s+(?:admission|presentation|arrival)"
    rf"|admission|___\s+on\s+admission"
    rf")[ \t]*(?::|-|$)",
    re.IGNORECASE,
)
SUBHEADING_MAX_LEN = 60


@dataclass
class Section:
    name: str
    start: int          # char offset of the content start in the note
    end: int            # char offset of the content end (exclusive)
    header_start: int
    out_of_order: bool = False
    repeated: bool = False


@dataclass
class ParsedNote:
    sections: list[Section] = field(default_factory=list)
    n_rejected_heading_candidates: int = 0

    def by_name(self, name: str) -> list[Section]:
        return [s for s in self.sections if s.name == name]


def _line_spans(text: str):
    pos = 0
    for line in text.split("\n"):
        yield pos, line
        pos += len(line) + 1


def parse_note(text: str) -> ParsedNote:
    """Split a discharge summary into top-level template sections with offsets."""
    lines = list(_line_spans(text))
    accepted: list[tuple[int, int, str, int]] = []   # (line_idx, content_start, name, header_start)
    flags: list[tuple[bool, bool]] = []
    seen: set[str] = set()
    max_order = -1
    rejected = 0
    for i, (pos, line) in enumerate(lines):
        if len(line) > 120 or ":" not in line:
            continue
        m = HEADING_RE.match(line)
        if not m:
            continue
        alias = re.sub(r"\s+", " ", m.group("head").lower())
        name = ALIAS_TO_CANONICAL[alias]
        prev = lines[i - 1][1] if i > 0 else ""
        prev_blank = i == 0 or prev.strip() == ""
        prev_placeholder = bool(PLACEHOLDER_LINE_RE.match(prev))
        plausible = prev_blank or prev_placeholder or (
            name in NO_BLANK_LINE_REQUIRED and max_order >= ORDER["social_history"])
        if alias in AMBIGUOUS_ALIASES:
            # ambiguous de-identified aliases only count in their template slot
            plausible = plausible and prev_blank and ORDER[name] > max_order
        in_order = ORDER[name] > max_order and name not in seen
        if not in_order:
            strict = prev_blank and alias not in AMBIGUOUS_ALIASES and not m.group("inline").strip()
            plausible = plausible and strict
        if not plausible:
            rejected += 1
            continue
        content_start = pos + m.start("inline")
        accepted.append((i, content_start, name, pos))
        flags.append((not in_order and name not in seen, name in seen))
        seen.add(name)
        max_order = max(max_order, ORDER[name])

    parsed = ParsedNote(n_rejected_heading_candidates=rejected)
    for k, (i, cstart, name, hstart) in enumerate(accepted):
        cend = accepted[k + 1][3] if k + 1 < len(accepted) else len(text)
        out_of_order, repeated = flags[k]
        parsed.sections.append(Section(name, cstart, cend, hstart, out_of_order, repeated))
    return parsed


def split_physical_exam(text: str, start: int, end: int) -> tuple[list[tuple[int, int]], str]:
    """Return the admission-phase spans of a Physical Exam section and a status.

    Status: ``labelled`` (admission/discharge sub-headings found), ``unlabelled``
    (no sub-headings: kept entirely, temporally ambiguous), or ``discharge_only``
    (only a discharge exam: nothing kept).
    """
    segments: list[tuple[int, str]] = [(start, "unlabelled")]
    for pos, line in _line_spans(text[start:end]):
        if len(line.strip()) > SUBHEADING_MAX_LEN or not line.strip():
            continue
        if DISCHARGE_SUB_RE.match(line):
            segments.append((start + pos, "discharge"))
        elif ADMISSION_SUB_RE.match(line):
            segments.append((start + pos, "admission"))
    labels = {lab for _, lab in segments[1:]}
    spans = []
    for k, (seg_start, lab) in enumerate(segments):
        seg_end = segments[k + 1][0] if k + 1 < len(segments) else end
        if lab != "discharge" and seg_end > seg_start:
            spans.append((seg_start, seg_end))
    if not labels:
        status = "unlabelled"
    elif labels == {"discharge"} and not text[start:segments[1][0]].strip():
        status = "discharge_only"
    else:
        status = "labelled"
    return spans, status


def clean(text: str) -> str:
    """Whitespace normalisation identical in spirit to COP-IV (newlines -> spaces)."""
    return re.sub(r"\s+", " ", text).strip()


def admission_spans(text: str, parsed: ParsedNote | None = None) -> tuple[dict[str, list[tuple[int, int]]], dict]:
    """Admission-related spans per section (T1) plus per-note parse metadata."""
    parsed = parsed or parse_note(text)
    out: dict[str, list[tuple[int, int]]] = {}
    meta = {"pe_status": "absent", "n_out_of_order": 0, "n_repeated": 0,
            "n_rejected_heading_candidates": parsed.n_rejected_heading_candidates}
    for name, _ in ADMISSION_SECTIONS:
        spans = []
        for s in parsed.by_name(name):
            meta["n_out_of_order"] += s.out_of_order
            meta["n_repeated"] += s.repeated
            if name == "physical_exam":
                pe_spans, meta["pe_status"] = split_physical_exam(text, s.start, s.end)
                spans.extend(pe_spans)
            else:
                spans.append((s.start, s.end))
        out[name] = [(a, b) for a, b in spans if text[a:b].strip()]
    return out, meta


def section_text(text: str, spans: list[tuple[int, int]]) -> str:
    return clean(" ".join(text[a:b] for a, b in spans))


def render(sections: dict[str, str], labels: list[tuple[str, str]] | None = None) -> str:
    """Render sections in the COP-IV/CORe template ("LABEL: text" joined by blank lines)."""
    labels = labels or ADMISSION_SECTIONS
    return "\n\n".join(f"{label}: {sections.get(name, '')}" for name, label in labels)


def later_sections(text: str, names: list[str], parsed: ParsedNote | None = None) -> dict[str, str]:
    """Full text of discharge-phase sections (used for T3/T4 injection)."""
    parsed = parsed or parse_note(text)
    return {n: clean(" ".join(text[s.start:s.end] for s in parsed.by_name(n))) for n in names}
