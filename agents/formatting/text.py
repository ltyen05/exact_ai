"""Text normalization helpers."""

import re
from typing import Any


def normalize_text(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = s.replace("\\sqrt", "sqrt")
    s = s.replace("−", "-").replace("–", "-")
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_answer(s: Any) -> str:
    s = normalize_text(s).lower()
    aliases = {
        "uncertain": "unknown",
        "cannot be determined": "unknown",
        "not enough information": "unknown",
        "true": "yes",
        "false": "no",
    }
    s = aliases.get(s, s)
    # keep MCQ letters clean
    if re.fullmatch(r"[abcd]", s):
        return s.upper()
    return s
