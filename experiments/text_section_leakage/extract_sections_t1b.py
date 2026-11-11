"""Improved section parser (T1b), kept as a separate condition from T1.

Recognizes additional section aliases and Physical Exam phase headings,
separates home-medication and family/social-history blocks from past medical
history, and ends the HPI at explicit ICU/floor course sub-headings.

Content without a textual phase cue remains temporally ambiguous and is kept.
"""
from __future__ import annotations

import re

from common import ADMISSION_SECTIONS
from extract_sections import (AMBIGUOUS_ALIASES, HEADING_INVENTORY, NO_BLANK_LINE_REQUIRED, ORDER,
                              PLACEHOLDER_LINE_RE, ParsedNote, Section, _line_spans)
from section_patterns import EXT_ADMISSION_RE

# ---- top level (F3, F4) ----
T1B_INVENTORY = [(n, a + (["___ exam", "___ examination"] if n == "physical_exam" else []))
                 for n, a in HEADING_INVENTORY]
T1B_ALIAS = {alias: name for name, aliases in T1B_INVENTORY for alias in aliases}
_alias_alt = "|".join(sorted((re.escape(a) for a in T1B_ALIAS), key=len, reverse=True))
T1B_HEADING_RE = re.compile(rf"^[ \t]*(?P<head>{_alias_alt})[ \t]*:[ \t]*(?P<inline>.*)$", re.IGNORECASE)
T1B_AMBIGUOUS = AMBIGUOUS_ALIASES | {"___ exam", "___ examination"}


def parse_note_t1b(text: str) -> tuple[ParsedNote, dict]:
    """parse_note with F3/F4; also returns which fixes changed the top-level split."""
    lines = list(_line_spans(text))
    accepted, flags = [], []
    seen: set[str] = set()
    max_order, rejected = -1, 0
    fired = {"F3_exam_alias": 0, "F4_upper_heading": 0}
    for i, (pos, line) in enumerate(lines):
        if len(line) > 120 or ":" not in line:
            continue
        m = T1B_HEADING_RE.match(line)
        if not m:
            continue
        alias = re.sub(r"\s+", " ", m.group("head").lower())
        name = T1B_ALIAS[alias]
        prev = lines[i - 1][1] if i > 0 else ""
        prev_blank = i == 0 or prev.strip() == ""
        prev_placeholder = bool(PLACEHOLDER_LINE_RE.match(prev))
        in_order = ORDER[name] > max_order and name not in seen
        upper = m.group("head").isupper() and len(m.group("head")) >= 8
        alone = not m.group("inline").strip()
        plausible = prev_blank or prev_placeholder or (
            name in NO_BLANK_LINE_REQUIRED and max_order >= ORDER["social_history"])
        via_upper = False
        if not plausible and (upper or alone) and in_order and alias not in T1B_AMBIGUOUS:
            plausible = via_upper = True                                   # F4
        if alias in T1B_AMBIGUOUS:
            plausible = plausible and prev_blank and ORDER[name] > max_order
        if not in_order:
            strict = prev_blank and alias not in T1B_AMBIGUOUS and not m.group("inline").strip()
            plausible = plausible and strict
        if not plausible:
            rejected += 1
            continue
        fired["F4_upper_heading"] += via_upper
        fired["F3_exam_alias"] += alias in ("___ exam", "___ examination")
        accepted.append((i, pos + m.start("inline"), name, pos))
        flags.append((not in_order and name not in seen, name in seen))
        seen.add(name)
        max_order = max(max_order, ORDER[name])
    parsed = ParsedNote(n_rejected_heading_candidates=rejected)
    for k, (i, cstart, name, hstart) in enumerate(accepted):
        cend = accepted[k + 1][3] if k + 1 < len(accepted) else len(text)
        parsed.sections.append(Section(name, cstart, cend, hstart, *flags[k]))
    return parsed, fired


# ---- Physical Exam phases (F1, F2) ----
_DECOR_LEAD = re.compile(r"^[ \t\*=_\-#~>\.•]*")
_DECOR_TRAIL = re.compile(r"[ \t=_\-\*~#]*$")
DISCHARGE_WORD = r"(?:discharge|dishcarge|discharg|dicharge|dischage|dischrage|discahrge|disharge|dsicharge|d/c|" \
                 r"dc(?=\s+(?:exam|examination|physical|pe)\b))"
_PRE = r"(?:pertinent|focused|brief|final|physical|exam|examination|pe|vitals?|vs|labs?|weight|on|at|upon|" \
       r"prior|to|the|time|of|day|in|by|repeat)"
_POST = r"(?:physical|exam|examination|pe|vitals?|vs|labs?|day|date|and|death|pronunciation|pronouncement|of)"
_FINDING = re.compile(r"^\s*(?:none|no|neg|negative|minimal|clear|white|yellow|green|purulent|bloody|serous|scant|"
                      r"small|mild|moderate|copious|from|noted)\b", re.IGNORECASE)
DISCHARGE_HEAD_RE = re.compile(
    rf"^(?P<pre>(?:{_PRE}\s+){{0,5}}){DISCHARGE_WORD}(?P<post>(?:\s+{_POST}){{0,5}})\s*(?P<end>[:\-,(]|$)",
    re.IGNORECASE)
DISCHARGE_PROSE_RE = re.compile(rf"^(?:on|at|upon|prior\s+to)\s+(?:the\s+)?(?:day\s+of\s+)?{DISCHARGE_WORD}\b",
                                re.IGNORECASE)
DEATH_HEAD_RE = re.compile(
    r"^(?:(?:clinical\s+)?death(?:\s+exam(?:ination)?)?|(?:physical\s+)?exam(?:ination)?\s+(?:upon|on|at|after)\s+"
    r"(?:the\s+time\s+of\s+)?death|post[\s-]?death\s+exam(?:ination)?|pronouncement(?:\s+exam(?:ination)?)?|"
    r"pronunciation\s+of\s+death(?:\s+exam(?:ination)?)?)\s*(?:[:\-,]|$)|^time\s+of\s+death\b", re.IGNORECASE)
# with "patient"/"pt" any continuation; a bare word only before punctuation, a time or the end of the line
# (a wrapped line such as "deceased wife." or "died of MI at 60" is not a death line)
_DEATH_WORD = r"(?:expired|expires|deceased|died|passed\s+away|pronounced\s+(?:dead|deceased|expired|at\s+\d|at\s+___|by|death))"
DEATH_LINE_RE = re.compile(rf"^(?:the\s+)?(?:patient|pt\.?)\s*(?:was\s+|is\s+)?(?:found\s+to\s+be\s+)?{_DEATH_WORD}\b"
                           rf"|^(?:was\s+)?(?:found\s+to\s+be\s+)?{_DEATH_WORD}(?=\s*(?:$|[.\-:,;]|(?:at|on|@|___|\d)))"
                           r"|^(?:a\s+)?(?:clinical\s+)?death\s+exam(?:ination)?\b", re.IGNORECASE)
# an admission cue with findings on the same line re-opens the admission phase ("Admit - BP 102/70, ...")
ADMIT_INLINE_RE = re.compile(r"^(?:[Aa]dmit(?:ting)?|ADMIT(?:TING)?|[Aa]dmission|ADMISSION|(?:On|At|Upon|ON|AT|UPON)\s+"
                             r"(?:admission|ADMISSION))\b\s*[:\-]")
LATER_HEAD_RE = re.compile(
    r"^(?:on\s+)?(?:subsequent|interval|repeat|follow-?up)\s+exam(?:ination)?|^post-?op(?:erative(?:ly)?)?"
    r"(?:\s+(?:exam(?:ination)?|day\s*#?\d*|check))?\s*(?:[:\-,]|$)", re.IGNORECASE)
_UNIT = r"(?:micu|sicu|ccu|csru|cvicu|tsicu|icu|floor)"
TRANSFER_HEAD_RE = re.compile(
    rf"^(?:{_UNIT}\s+(?:admission|admit|transfer|arrival)\s+(?:exam(?:ination)?|pe|physical(?:\s+exam)?)"
    rf"|(?:exam(?:ination)?|pe)\s+(?:on|upon|at)\s+(?:transfer|arrival)\s+to\s+(?:the\s+)?{_UNIT}"
    rf"|(?:admission|transfer)\s+to\s+(?:the\s+)?{_UNIT}|transfer\s+exam(?:ination)?)\s*(?:[:\-,]|$)", re.IGNORECASE)
SHORT = 60


def phase_of_line(line: str) -> tuple[str, str] | None:
    """(phase, rule) of a Physical Exam line: phase in discharge / transfer / admission."""
    norm = re.sub(r"\s*/\s*", " ", _DECOR_TRAIL.sub("", _DECOR_LEAD.sub("", line))).strip()
    if not norm:
        return None
    short = len(norm) <= SHORT
    m = DISCHARGE_HEAD_RE.match(norm)
    if m and (m.group("end") or short):
        bare = not m.group("pre").strip() and not m.group("post").strip()
        if not (bare and _FINDING.match(norm[m.end():])):
            return "discharge", "discharge_heading"
    if DISCHARGE_PROSE_RE.match(norm):
        return "discharge", "discharge_prose"
    if DEATH_HEAD_RE.match(norm) and (short or norm.rstrip().endswith(":") or re.match(r"^\S+(\s+\S+){0,5}\s*[:\-,]", norm)):
        return "discharge", "death_heading"
    if DEATH_LINE_RE.match(norm):
        return "discharge", "death_line"
    if LATER_HEAD_RE.match(norm) and short:
        return "discharge", "later_exam"
    if TRANSFER_HEAD_RE.match(norm) and short:
        return "transfer", "transfer_exam"
    wrapped = norm[0].islower() and re.search(r",\s*\S", norm)          # "admission, no wheezing ..." continues prose
    if (short and EXT_ADMISSION_RE.match(norm) and not wrapped) or ADMIT_INLINE_RE.match(norm):
        return "admission", "admission_heading"
    return None


def _has_content(s: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9]", s))


def split_physical_exam_t1b(text: str, start: int, end: int) -> tuple[list[tuple[int, int]], str, list[str]]:
    """Admission-phase spans of a Physical Exam section, its status and the rules that fired."""
    segments: list[tuple[int, str]] = [(start, "unlabelled")]
    rules = []
    for pos, line in _line_spans(text[start:end]):
        hit = phase_of_line(line)
        if hit:
            segments.append((start + pos, hit[0]))
            rules.append(hit[1])
    spans, kept_before = [], False
    for k, (seg_start, lab) in enumerate(segments):
        seg_end = segments[k + 1][0] if k + 1 < len(segments) else end
        keep = lab in ("unlabelled", "admission") or (lab == "transfer" and not kept_before)
        if keep and seg_end > seg_start:
            if _has_content(text[seg_start:seg_end]):
                spans.append((seg_start, seg_end))
                kept_before = True
            elif text[seg_start:seg_end].strip():
                rules.append("decoration_dropped")                         # F2
    labels = {lab for _, lab in segments[1:]}
    if not labels:
        status = "unlabelled"
    elif labels == {"discharge"} and not _has_content(text[start:segments[1][0]]):
        status = "discharge_only"
    else:
        status = "labelled"
    return spans, status, rules


# ---- blocks inside the Past Medical History (F5) and HPI course sub-headings (F6) ----
PMH_BLOCK_RE = re.compile(
    r"^[ \t\*\-]*(?:(?P<meds>(?:home|outpatient|current|admission|outside)\s+(?:medications?|meds)|medications?|meds)"
    r"|(?P<all>allerg(?:y|ies)|all(?=\s*:))|(?P<soc>social\s+(?:history|hx)|sh(?=\s*:)|shx|substance\s+(?:use|abuse)"
    r"(?:\s+(?:history|hx))?)|(?P<fam>family\s+(?:history|hx)|fh(?=\s*:)|fhx))\b[^:\n]{0,40}:", re.IGNORECASE)
# psychiatric-history templates list "Medication (and ECT) trials" as part of the past history
_PMH_NOT_A_BLOCK = re.compile(r"^[ \t\*\-]*(?:medications?|meds)\s+(?:and\s+\w+\s+)?trials|^[ \t\*\-]*substance",
                              re.IGNORECASE)
PMH_HEADING_RE = re.compile(r"^[ \t\*\-]*(?:past|pmh|psh|pmhx|pshx|pobhx|pgynhx|ob|gyn|onc|oncolog\w*|surg\w*|psych\w*|"
                            r"other|prior|cardiac|medical)\b[^:\n]{0,40}:", re.IGNORECASE)
BLOCK_TARGET = {"meds": "medications_on_admission", "all": "allergies", "soc": "social_history",
                "fam": "family_history"}
HPI_COURSE_RE = re.compile(rf"^[ \t\*\-]*(?:(?:{_UNIT}|hospital|ward)\s+course|{_UNIT}\s+(?:events|summary))\s*(?:[:\-]|$)",
                           re.IGNORECASE)


def pmh_blocks(text: str, start: int, end: int) -> tuple[list[tuple[int, int]], dict[str, list[tuple[int, int]]]]:
    """Split a PMH span into the PMH proper and embedded blocks for other admission sections."""
    keep, moved, cur, cur_start = [], {}, None, start
    for pos, line in _line_spans(text[start:end]):
        at = start + pos
        m = None if _PMH_NOT_A_BLOCK.match(line) else PMH_BLOCK_RE.match(line)
        target = next((BLOCK_TARGET[g] for g in BLOCK_TARGET if m and m.group(g)), None)
        if target is None and not (cur and PMH_HEADING_RE.match(line)):
            continue
        (moved.setdefault(cur, []) if cur else keep).append((cur_start, at))
        cur, cur_start = target, at
    (moved.setdefault(cur, []) if cur else keep).append((cur_start, end))
    return [s for s in keep if s[1] > s[0]], {k: [s for s in v if s[1] > s[0]] for k, v in moved.items()}


def hpi_end(text: str, start: int, end: int) -> int:
    for pos, line in _line_spans(text[start:end]):
        if len(line.strip()) <= 80 and HPI_COURSE_RE.match(line) and pos > 0:
            return start + pos
    return end


def admission_spans_t1b(text: str) -> tuple[dict[str, list[tuple[int, int]]], dict]:
    """Admission-related spans per section (T1b), with per-note metadata and the fixes that fired."""
    parsed, fired = parse_note_t1b(text)
    out: dict[str, list[tuple[int, int]]] = {n: [] for n, _ in ADMISSION_SECTIONS}
    meta = {"pe_status": "absent", "pe_rules": [], **fired, "F5_pmh_blocks": 0, "F6_hpi_course": 0}
    for name, _ in ADMISSION_SECTIONS:
        for s in parsed.by_name(name):
            if name == "physical_exam":
                pe_spans, meta["pe_status"], rules = split_physical_exam_t1b(text, s.start, s.end)
                out[name].extend(pe_spans)
                meta["pe_rules"].extend(rules)
            elif name == "past_medical_history":
                keep, moved = pmh_blocks(text, s.start, s.end)
                out[name].extend(keep)
                for target, spans in moved.items():
                    out[target].extend(spans)
                    meta["F5_pmh_blocks"] += len(spans)
            elif name == "history_of_present_illness":
                e = hpi_end(text, s.start, s.end)
                meta["F6_hpi_course"] += e < s.end
                out[name].append((s.start, e))
            else:
                out[name].append((s.start, s.end))
    out = {n: sorted((a, b) for a, b in v if text[a:b].strip()) for n, v in out.items()}
    return out, meta
