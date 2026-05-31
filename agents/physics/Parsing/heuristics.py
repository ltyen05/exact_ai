"""Heuristic fallbacks for physics semantic parsing."""

from __future__ import annotations

import re
from typing import Any

from agents.physics.domain.symbols import _normalize_text


def heuristic_parse(question: str) -> dict[str, Any] | None:
    """Return a lightweight parse for known easy cases when the LLM fails."""
    text = _normalize_text(question).lower()
    if "solenoid" in text and "magnetic field" in text:
        current_match = re.search(
            r"current[^0-9+-]*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*a\b",
            text,
        )
        turns_match = re.search(
            r"(?:turns per meter|turn per meter|n)\s*(?:is|=)?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
            text,
        )
        if current_match and turns_match:
            return {
                "question": question,
                "domain": "Sources of Magnetic Fields",
                "target": {"symbol": "B", "unit": "T"},
                "givens": [
                    {
                        "symbol": "I",
                        "si_value": float(current_match.group(1)),
                        "si_unit": "A",
                        "uncertainty": None,
                    },
                    {
                        "symbol": "n",
                        "si_value": float(turns_match.group(1)),
                        "si_unit": "1/m",
                        "uncertainty": None,
                    },
                ],
                "relations": ["long solenoid"],
                "question_kind": "computational",
            }
    return None
