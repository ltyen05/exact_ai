"""P1 answer-correctness evaluator for EXACT 2026 outputs.

P1 is the challenge's correctness-of-answers criterion. The evaluator returns
an automatic answer-level score:

    P1 accuracy = correct answers / total answers

Logic answers are compared after light text normalization. Numeric physics
answers are compared with configurable tolerance and common SI-prefix unit
conversion when units are available.
"""

from __future__ import annotations

import math
import re
from typing import Any


DEFAULT_RTOL = 1e-3
DEFAULT_ATOL = 1e-2

NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

UNIT_FACTORS: dict[str, tuple[str, float]] = {
    "j": ("energy", 1.0),
    "mj": ("energy", 1e-3),
    "uj": ("energy", 1e-6),
    "nj": ("energy", 1e-9),
    "v": ("voltage", 1.0),
    "mv": ("voltage", 1e-3),
    "a": ("current", 1.0),
    "ma": ("current", 1e-3),
    "ohm": ("resistance", 1.0),
    "hz": ("frequency", 1.0),
    "khz": ("frequency", 1e3),
    "rad/s": ("angular_frequency", 1.0),
    "f": ("capacitance", 1.0),
    "uf": ("capacitance", 1e-6),
    "pf": ("capacitance", 1e-12),
    "h": ("inductance", 1.0),
    "mh": ("inductance", 1e-3),
    "c": ("charge", 1.0),
    "mc": ("charge", 1e-3),
    "uc": ("charge", 1e-6),
    "nc": ("charge", 1e-9),
    "n": ("force", 1.0),
    "mn": ("force", 1e-3),
    "w": ("power", 1.0),
    "mw": ("power", 1e-3),
    "wb": ("flux", 1.0),
    "mwb": ("flux", 1e-3),
    "uwb": ("flux", 1e-6),
    "t": ("magnetic_field", 1.0),
    "mt": ("magnetic_field", 1e-3),
    "m": ("length", 1.0),
    "cm": ("length", 1e-2),
    "mm": ("length", 1e-3),
    "v/m": ("field", 1.0),
    "n/c": ("field", 1.0),
    "%": ("percent", 1.0),
    "degree": ("angle", 1.0),
    "turns": ("turns", 1.0),
    "turns/m": ("turn_density", 1.0),
}


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("\\sqrt", "sqrt")
    text = text.replace("\u2212", "-").replace("\u2013", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;")


def normalize_answer(value: Any) -> str:
    """Normalize answers exactly as the zip P1 scorer does for logic labels."""
    text = normalize_text(value)
    aliases = {
        "uncertain": "unknown",
        "cannot be determined": "unknown",
        "not enough information": "unknown",
        "true": "yes",
        "false": "no",
    }
    text = aliases.get(text, text)
    if re.fullmatch(r"[abcd]", text):
        return text.upper()
    return text


def normalize_unit(value: str) -> str:
    unit = normalize_text(value)
    replacements = {
        "\\omega": "ohm",
        "omega": "ohm",
        "ohms": "ohm",
        "newton": "n",
        "newtons": "n",
        "volt": "v",
        "volts": "v",
        "watt": "w",
        "watts": "w",
        "joule": "j",
        "joules": "j",
        "meter": "m",
        "meters": "m",
        "metre": "m",
        "metres": "m",
        "second": "s",
        "seconds": "s",
        "\u03c9": "ohm",
        "\u03a9": "ohm",
        "\u03bc": "u",
        "\u00b5": "u",
        "\\mu": "u",
        "micro": "u",
        "^{\\circ}": "degree",
        "^\\circ": "degree",
        "degrees": "degree",
    }
    for old, new in replacements.items():
        unit = unit.replace(old.lower(), new)
    unit = unit.replace(" ", "")
    if unit in {"", "-", "dimensionless"}:
        return ""
    return unit


def extract_number(text: str) -> tuple[float | None, str]:
    match = NUMBER_RE.search(str(text))
    if match is None:
        return None, ""
    try:
        value = float(match.group(0))
    except ValueError:
        return None, ""
    unit = str(text)[match.end() :].strip().split(",", 1)[0].strip()
    return value, unit


def _scaled(value: float, unit: str) -> tuple[str, float] | None:
    mapped = UNIT_FACTORS.get(normalize_unit(unit))
    if mapped is None:
        return None
    dimension, factor = mapped
    return dimension, value * factor


def numeric_match(
    predicted_answer: str,
    expected_answer: str,
    expected_unit: str = "",
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> dict[str, Any] | None:
    predicted_value, predicted_unit = extract_number(predicted_answer)
    expected_value, embedded_expected_unit = extract_number(expected_answer)
    if predicted_value is None or expected_value is None:
        return None

    expected_unit = expected_unit or embedded_expected_unit
    predicted_scaled = _scaled(predicted_value, predicted_unit)
    expected_scaled = _scaled(expected_value, expected_unit)
    if predicted_scaled and expected_scaled and predicted_scaled[0] == expected_scaled[0]:
        predicted_value = predicted_scaled[1]
        expected_value = expected_scaled[1]

    tolerance = max(atol, abs(expected_value) * rtol)
    correct = math.isclose(predicted_value, expected_value, rel_tol=rtol, abs_tol=tolerance)
    return {
        "correct": correct,
        "method": "numeric",
        "reason": f"numeric tolerance={tolerance:g}",
        "predicted_value": predicted_value,
        "expected_value": expected_value,
        "predicted_unit": normalize_unit(predicted_unit),
        "expected_unit": normalize_unit(expected_unit),
    }


def answer_match(
    predicted_answer: Any,
    expected_answer: Any,
    predicted_unit: Any = "",
    expected_unit: Any = "",
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> bool:
    """Return P1 correctness for both logic and physics answers.

    This mirrors the zip evaluator: first compare normalized answer labels, then
    fall back to numeric matching with unit normalization for physics.
    """
    predicted_normalized = normalize_answer(predicted_answer)
    expected_normalized = normalize_answer(expected_answer)
    expected_unit_normalized = normalize_unit(str(expected_unit or ""))
    if predicted_normalized == expected_normalized:
        return not expected_unit_normalized or normalize_unit(str(predicted_unit or "")) == expected_unit_normalized

    numeric = numeric_match(
        str(predicted_answer or ""),
        str(expected_answer or ""),
        str(expected_unit or ""),
        rtol=rtol,
        atol=atol,
    )
    if numeric is None:
        return False
    return bool(numeric["correct"]) and (
        not expected_unit_normalized
        or numeric["predicted_unit"] == expected_unit_normalized
    )


def evaluate_prediction(
    predicted_answer: Any,
    expected_answer: Any,
    expected_unit: str = "",
    *,
    predicted_unit: str = "",
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> dict[str, Any]:
    predicted_text = str(predicted_answer or "")
    expected_text = str(expected_answer or "")
    predicted_normalized = normalize_answer(predicted_text)
    expected_normalized = normalize_answer(expected_text)
    expected_unit_normalized = normalize_unit(expected_unit)
    predicted_unit_normalized = normalize_unit(predicted_unit)
    if predicted_normalized == expected_normalized:
        unit_ok = not expected_unit_normalized or predicted_unit_normalized == expected_unit_normalized
        return {
            "correct": unit_ok,
            "method": "normalized_answer_exact",
            "reason": "normalized answer exact match",
            "predicted_normalized": predicted_normalized,
            "expected_normalized": expected_normalized,
            "predicted_unit": predicted_unit_normalized,
            "expected_unit": expected_unit_normalized,
        }

    numeric = numeric_match(
        predicted_text,
        expected_text,
        expected_unit,
        rtol=rtol,
        atol=atol,
    )
    if numeric is not None:
        return numeric

    return {
        "correct": predicted_normalized == expected_normalized,
        "method": "normalized_answer_exact",
        "reason": "normalized answer exact match",
        "predicted_normalized": predicted_normalized,
        "expected_normalized": expected_normalized,
    }


def summarize_p1(results: list[dict[str, Any]], task_key: str = "task") -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for item in results if item.get("correct") is True)
    summary: dict[str, Any] = {
        "total": total,
        "correct": correct,
        "p1": correct / total if total else 0.0,
        "p1_accuracy": correct / total if total else 0.0,
    }

    tasks = sorted({str(item.get(task_key) or "unknown") for item in results})
    by_task: dict[str, dict[str, Any]] = {}
    for task in tasks:
        subset = [item for item in results if str(item.get(task_key) or "unknown") == task]
        task_correct = sum(1 for item in subset if item.get("correct") is True)
        by_task[task] = {
            "total": len(subset),
            "correct": task_correct,
            "p1": task_correct / len(subset) if subset else 0.0,
            "p1_accuracy": task_correct / len(subset) if subset else 0.0,
        }
    summary["by_task"] = by_task
    return summary
