"""Shared admission-heading and death-wording patterns for text extraction."""

import re

from extract_sections import _PE

_LEAD = r"^[ \t\*=_\-#]*"
_END = r"[ \t]*(?:[:,\-=]|$)"
_EXAM = rf"(?:(?:pertinent|brief|focused|final)\s+)?(?:{_PE}|vitals|vs)"
EXT_ADMISSION_RE = re.compile(
    rf"{_LEAD}(?:(?:admission|admit|admitting|initial|presenting|arrival)\s+{_EXAM}"
    rf"|(?:{_EXAM}\s+)?(?:on|at|upon)\s+(?:admission|presentation|arrival)(?:\s+to\s+[\w_/]+(?:\s+\w+)?)?"
    rf"|admission){_END}",
    re.IGNORECASE,
)
DEATH_RE = re.compile(r"\b(?:expired|pronounced|time\s+of\s+death|death\s+exam|deceased|declared\s+dead)\b",
                      re.IGNORECASE)
