"""P1 answer-correctness evaluator for EXACT 2026 outputs.

This module folds the physics-oriented Postman evaluator in
``p1_eval_physics.js`` into Python while keeping the public API used by the
demo runner:

    P1 accuracy = correct answers / total answers

Supported comparisons include boolean labels, numeric physics quantities with
SI-prefix/unit normalization, simple math expressions such as ``9sqrt(3)*10^-27``,
and a final normalized string-containment fallback.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any


DEFAULT_RTOL = 0.10
DEFAULT_ATOL = 1e-12

NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

PREFIX_WORD_TO_SYMBOL = {
    "pico": "p",
    "nano": "n",
    "micro": "u",
    "milli": "m",
    "centi": "c",
    "kilo": "k",
    "mega": "M",
    "giga": "G",
}

WORD_UNIT_PATTERNS = [
    (r"meters?|metres?", "m"),
    (r"grams?", "g"),
    (r"seconds?|secs?", "s"),
    (r"hertz", "Hz"),
    (r"volts?", "V"),
    (r"amperes?|amps?", "A"),
    (r"watts?", "W"),
    (r"joules?", "J"),
    (r"newtons?", "N"),
    (r"coulombs?", "C"),
    (r"farads?", "F"),
    (r"henrys?", "H"),
    (r"ohms?", "ohm"),
    (r"pascals?", "Pa"),
    (r"teslas?", "T"),
]

PREFIXES = [
    ("p", 1e-12),
    ("n", 1e-9),
    ("u", 1e-6),
    ("m", 1e-3),
    ("c", 1e-2),
    ("k", 1e3),
    ("M", 1e6),
    ("G", 1e9),
]

EXACT_UNIT_MAP: dict[str, tuple[str, float]] = {
    "m": ("m", 1.0),
    "cm": ("m", 1e-2),
    "km": ("m", 1e3),
    "g": ("kg", 1e-3),
    "kg": ("kg", 1.0),
    "s": ("s", 1.0),
    "min": ("s", 60.0),
    "hr": ("s", 3600.0),
    "Hz": ("Hz", 1.0),
    "hz": ("Hz", 1.0),
    "khz": ("Hz", 1e3),
    "mhz": ("Hz", 1e6),
    "ghz": ("Hz", 1e9),
    "V": ("V", 1.0),
    "v": ("V", 1.0),
    "A": ("A", 1.0),
    "a": ("A", 1.0),
    "W": ("W", 1.0),
    "w": ("W", 1.0),
    "J": ("J", 1.0),
    "j": ("J", 1.0),
    "N": ("N", 1.0),
    "n": ("N", 1.0),
    "C": ("C", 1.0),
    "c": ("C", 1.0),
    "F": ("F", 1.0),
    "f": ("F", 1.0),
    "H": ("H", 1.0),
    "h": ("H", 1.0),
    "ohm": ("ohm", 1.0),
    "Ohm": ("ohm", 1.0),
    "OHM": ("ohm", 1.0),
    "Pa": ("Pa", 1.0),
    "pa": ("Pa", 1.0),
    "T": ("T", 1.0),
    "t": ("T", 1.0),
    "eV": ("J", 1.602176634e-19),
    "ev": ("J", 1.602176634e-19),
    "deg": ("deg", 1.0),
    "degree": ("deg", 1.0),
    "degrees": ("deg", 1.0),
    "\u00b0": ("deg", 1.0),
    "rad": ("rad", 1.0),
    "%": ("ratio", 0.01),
}

PREFIXABLE_BASE_UNITS = {
    "m": ("m", 1.0),
    "g": ("kg", 1e-3),
    "s": ("s", 1.0),
    "Hz": ("Hz", 1.0),
    "hz": ("Hz", 1.0),
    "V": ("V", 1.0),
    "v": ("V", 1.0),
    "A": ("A", 1.0),
    "a": ("A", 1.0),
    "W": ("W", 1.0),
    "w": ("W", 1.0),
    "J": ("J", 1.0),
    "j": ("J", 1.0),
    "N": ("N", 1.0),
    "n": ("N", 1.0),
    "C": ("C", 1.0),
    "c": ("C", 1.0),
    "F": ("F", 1.0),
    "f": ("F", 1.0),
    "H": ("H", 1.0),
    "h": ("H", 1.0),
    "ohm": ("ohm", 1.0),
    "Ohm": ("ohm", 1.0),
    "OHM": ("ohm", 1.0),
    "Pa": ("Pa", 1.0),
    "pa": ("Pa", 1.0),
    "T": ("T", 1.0),
    "t": ("T", 1.0),
    "eV": ("J", 1.602176634e-19),
    "ev": ("J", 1.602176634e-19),
}

SUPERSCRIPT_MAP = str.maketrans(
    {
        "\u2070": "0",
        "\u00b9": "1",
        "\u00b2": "2",
        "\u00b3": "3",
        "\u2074": "4",
        "\u2075": "5",
        "\u2076": "6",
        "\u2077": "7",
        "\u2078": "8",
        "\u2079": "9",
        "\u207b": "-",
        "\u207a": "+",
    }
)


def value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(value_to_text(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def combine_answer_and_unit(answer: Any, unit: Any) -> str:
    answer_text = value_to_text(answer).strip()
    unit_text = value_to_text(unit).strip()
    if not answer_text or not unit_text or unit_text == "-":
        return answer_text
    if re.search(rf"(^|\s){re.escape(unit_text)}(\s|$)", answer_text, flags=re.I):
        return answer_text
    return f"{answer_text} {unit_text}"


def normalize_superscript(text: Any) -> str:
    return value_to_text(text).translate(SUPERSCRIPT_MAP)


def normalize_math_text(value: Any) -> str:
    text = normalize_superscript(value)
    replacements = {
        ",": "",
        "\\\\": "\\",
        r"\times": "*",
        "\u00d7": "*",
        "\u2219": "*",
        "\u00b7": "*",
        "\u2212": "-",
        r"\cdot": "*",
        r"\(": " ",
        r"\)": " ",
        "$": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_boolean(text: Any) -> str:
    normalized = value_to_text(text).lower()
    normalized = re.sub(r"[`*_~]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def extract_boolean(value: Any) -> bool | None:
    text = normalize_for_boolean(value)
    if not text:
        return None

    explicit = re.search(
        r"(?:answer|result|final answer|the answer is|therefore|thus)\s*(?:is|:)?\s*\b(yes|no|true|false)\b",
        text,
        flags=re.I,
    )
    if explicit:
        token = explicit.group(1).lower()
        return token in {"yes", "true"}

    if re.match(r"\s*(yes|true)\b", text):
        return True
    if re.match(r"\s*(no|false)\b", text):
        return False

    compact = re.sub(r"[^a-z\u00e0-\u1ef9\u0111]+", " ", text).strip()
    if compact in {"yes", "true", "c\u00f3", "co", "\u0111\u00fang", "dung"}:
        return True
    if compact in {"no", "false", "kh\u00f4ng", "khong", "sai"}:
        return False
    return None


def replace_prefixed_unit_words(text: str) -> str:
    output = text
    prefix_pattern = "|".join(PREFIX_WORD_TO_SYMBOL)
    for unit_pattern, symbol in WORD_UNIT_PATTERNS:
        regex = re.compile(rf"\b({prefix_pattern})\s*-?\s*({unit_pattern})\b", flags=re.I)

        def repl(match: re.Match[str]) -> str:
            return PREFIX_WORD_TO_SYMBOL[match.group(1).lower()] + symbol

        output = regex.sub(repl, output)
    return output


def replace_plain_unit_words(text: str) -> str:
    replacements = [
        (r"kilograms?", "kg"),
        (r"grams?", "g"),
        (r"meters?|metres?", "m"),
        (r"seconds?|secs?", "s"),
        (r"minutes?|mins?", "min"),
        (r"hours?|hrs?", "hr"),
        (r"hertz", "Hz"),
        (r"volts?", "V"),
        (r"amperes?|amps?", "A"),
        (r"watts?", "W"),
        (r"joules?", "J"),
        (r"newtons?", "N"),
        (r"coulombs?", "C"),
        (r"farads?", "F"),
        (r"henrys?", "H"),
        (r"ohms?", "ohm"),
        (r"pascals?", "Pa"),
        (r"teslas?", "T"),
        (r"degrees?", "deg"),
        (r"radians?", "rad"),
    ]
    output = text
    for pattern, replacement in replacements:
        output = re.sub(pattern, replacement, output, flags=re.I)
    return output


def normalize_unit(value: Any) -> str:
    if not value:
        return ""
    unit = value_to_text(value).strip()
    unit = re.sub(r"[.,;:!?]+$", "", unit)
    replacements = {
        r"\Omega": "ohm",
        "\u03a9": "ohm",
        "\u2126": "ohm",
        "\u03c9": "ohm",
        r"\mu": "u",
        "\u03bc": "u",
        "\u00b5": "u",
        "\u00b2": "^2",
        "\u00b3": "^3",
    }
    for old, new in replacements.items():
        unit = unit.replace(old, new)
    unit = re.sub(r"\bper\b", "/", unit, flags=re.I)
    unit = re.sub(r"[()\[\]{}]", "", unit)
    unit = replace_prefixed_unit_words(unit)
    unit = replace_plain_unit_words(unit)
    unit = re.sub(r"\s*(/|\*|\u00b7)\s*", r"\1", unit)
    unit = unit.replace("\u00b7", "*")
    unit = re.sub(r"\s+", "", unit).strip()
    if unit in {"", "-", "dimensionless"}:
        return ""
    return unit


def _add_exponent_to_base(base: str, exponent: int) -> str:
    return base if exponent == 1 else f"{base}^{exponent}"


def get_simple_unit_info(unit_token: str) -> dict[str, Any] | None:
    token = str(unit_token or "").strip()
    if not token:
        return None

    exponent_match = re.match(r"^(.+?)\^([-+]?\d+)$", token)
    if exponent_match:
        base_info = get_simple_unit_info(exponent_match.group(1))
        exponent = int(exponent_match.group(2))
        if base_info:
            return {
                "base": _add_exponent_to_base(base_info["base"], exponent),
                "factor": base_info["factor"] ** exponent,
                "recognized": True,
            }

    if token in EXACT_UNIT_MAP:
        base, factor = EXACT_UNIT_MAP[token]
        return {"base": base, "factor": factor, "recognized": True}

    for symbol, factor in PREFIXES:
        if not token.startswith(symbol) or len(token) <= len(symbol):
            continue
        rest = token[len(symbol) :]
        if rest in PREFIXABLE_BASE_UNITS:
            base, base_factor = PREFIXABLE_BASE_UNITS[rest]
            return {"base": base, "factor": factor * base_factor, "recognized": True}
    return None


def normalize_compound_base(numerator_bases: list[str], denominator_bases: list[str]) -> str:
    if len(numerator_bases) == 1 and len(denominator_bases) == 1:
        pair = (numerator_bases[0], denominator_bases[0])
        if pair in {("N", "C"), ("V", "m")}:
            return "electric_field"
    numerator = "*".join(numerator_bases) if numerator_bases else "1"
    denominator = "*".join(denominator_bases)
    return f"{numerator}/{denominator}" if denominator else numerator


def get_compound_unit_info(unit_original: str) -> dict[str, Any] | None:
    if "/" not in unit_original and "*" not in unit_original:
        return None
    parts = [part for part in re.split(r"([/*])", unit_original) if part]
    if len(parts) < 3:
        return None

    op = "*"
    factor = 1.0
    numerator_bases: list[str] = []
    denominator_bases: list[str] = []
    for part in parts:
        if part in {"*", "/"}:
            op = part
            continue
        info = get_simple_unit_info(part)
        if not info or not info["recognized"]:
            return None
        if op == "/":
            factor /= info["factor"]
            denominator_bases.append(info["base"])
        else:
            factor *= info["factor"]
            numerator_bases.append(info["base"])
    return {
        "base": normalize_compound_base(numerator_bases, denominator_bases),
        "factor": factor,
        "recognized": True,
    }


def get_unit_info(raw_unit: Any) -> dict[str, Any]:
    unit_original = normalize_unit(raw_unit)
    if not unit_original:
        return {"base": None, "factor": 1.0, "recognized": False, "unit": ""}

    simple_info = get_simple_unit_info(unit_original)
    if simple_info:
        return {**simple_info, "unit": unit_original}

    compound_info = get_compound_unit_info(unit_original)
    if compound_info:
        return {**compound_info, "unit": unit_original}

    return {"base": unit_original, "factor": 1.0, "recognized": False, "unit": unit_original}


def extract_unit_after(text: str, end_index: int) -> str:
    rest = text[end_index:].strip()
    rest = re.sub(r"^[\s*),.;:]+", "", rest).strip()
    unit_match = re.match(
        r"([a-zA-Z\u00b5\u03bc\u03a9\u2126\u00b0%\\/\^\*\u00b70-9+-]+(?:\s*(?:/|\*|\u00b7)\s*[a-zA-Z\u00b5\u03bc\u03a9\u2126\u00b0%\\\^0-9+-]+)*)",
        rest,
    )
    return unit_match.group(1) if unit_match else ""


def parse_math_number(text: str) -> dict[str, Any] | None:
    patterns = [
        rf"({NUMBER_PATTERN})\s*\\?sqrt\s*\{{?\s*({NUMBER_PATTERN})\s*\}}?\s*(?:\*\s*10\s*\^?\s*([-+]?\d+))?",
        rf"\\?sqrt\s*\{{?\s*({NUMBER_PATTERN})\s*\}}?\s*(?:\*\s*10\s*\^?\s*([-+]?\d+))?",
        rf"({NUMBER_PATTERN})\s*\*\s*10\s*\^?\s*([-+]?\d+)",
        rf"({NUMBER_PATTERN})",
    ]

    for index, pattern in enumerate(patterns):
        match = re.search(pattern, text, flags=re.I)
        if not match:
            continue
        try:
            if index == 0:
                coefficient = float(match.group(1))
                sqrt_value = float(match.group(2))
                exponent = float(match.group(3) if match.group(3) is not None else 0)
                value = coefficient * math.sqrt(sqrt_value) * (10**exponent)
            elif index == 1:
                sqrt_value = float(match.group(1))
                exponent = float(match.group(2) if match.group(2) is not None else 0)
                value = math.sqrt(sqrt_value) * (10**exponent)
            elif index == 2:
                coefficient = float(match.group(1))
                exponent = float(match.group(2))
                value = coefficient * (10**exponent)
            else:
                value = float(match.group(1))
        except ValueError:
            continue
        if math.isfinite(value):
            return {
                "value": value,
                "start": match.start(),
                "end": match.end(),
                "raw": match.group(0),
            }
    return None


def extract_quantity(value: Any) -> dict[str, Any] | None:
    text = normalize_math_text(value)
    parsed = parse_math_number(text)
    if parsed is None:
        return None

    raw_number = parsed["value"]
    raw_unit = extract_unit_after(text, parsed["end"])
    unit_info = get_unit_info(raw_unit)
    normalized_number = raw_number * unit_info["factor"]
    return {
        "raw_number": raw_number,
        "raw_unit": raw_unit,
        "normalized_number": normalized_number,
        "base_unit": unit_info["base"],
        "unit_recognized": unit_info["recognized"],
        "normalized_unit_label": unit_info["base"] or None,
        "parsed_raw_expression": parsed["raw"],
    }


def within_tolerance(gold_value: float, api_value: float, *, rtol: float, atol: float) -> dict[str, Any]:
    tolerance = max(abs(api_value) * rtol, atol)
    lower = api_value - tolerance
    upper = api_value + tolerance
    return {
        "isCorrect": lower <= gold_value <= upper,
        "lower": lower,
        "upper": upper,
        "tolerance": tolerance,
    }


def compare_quantities(
    gold_q: dict[str, Any] | None,
    api_q: dict[str, Any] | None,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> dict[str, Any]:
    if gold_q is None:
        return {
            "isValid": False,
            "isCorrect": False,
            "errorReason": "Cannot parse gold answer number",
            "lower": None,
            "upper": None,
            "tolerance": None,
            "unitMismatch": False,
        }
    if api_q is None:
        return {
            "isValid": False,
            "isCorrect": False,
            "errorReason": "Cannot parse API answer number",
            "lower": None,
            "upper": None,
            "tolerance": None,
            "unitMismatch": False,
        }

    both_recognized = gold_q["unit_recognized"] and api_q["unit_recognized"]
    both_have_base = gold_q["base_unit"] and api_q["base_unit"]
    if both_recognized and both_have_base and gold_q["base_unit"] != api_q["base_unit"]:
        return {
            "isValid": True,
            "isCorrect": False,
            "errorReason": f"Unit mismatch: gold unit {gold_q['base_unit']}, API unit {api_q['base_unit']}",
            "lower": None,
            "upper": None,
            "tolerance": None,
            "unitMismatch": True,
            "compared_gold_value": gold_q["normalized_number"],
            "compared_api_value": api_q["normalized_number"],
            "used_normalized_units": True,
            "compare_strategy": "unit_mismatch",
        }

    candidates: list[dict[str, Any]] = []
    if both_recognized and gold_q["base_unit"] == api_q["base_unit"]:
        candidates.append(
            {
                "strategy": "normalized_same_unit",
                "goldValue": gold_q["normalized_number"],
                "apiValue": api_q["normalized_number"],
                "usedNormalized": True,
            }
        )
    else:
        if gold_q["unit_recognized"]:
            candidates.append(
                {
                    "strategy": "gold_normalized_vs_api_raw",
                    "goldValue": gold_q["normalized_number"],
                    "apiValue": api_q["raw_number"],
                    "usedNormalized": True,
                }
            )
        if api_q["unit_recognized"]:
            candidates.append(
                {
                    "strategy": "gold_raw_vs_api_normalized",
                    "goldValue": gold_q["raw_number"],
                    "apiValue": api_q["normalized_number"],
                    "usedNormalized": True,
                }
            )
        candidates.append(
            {
                "strategy": "raw_numeric",
                "goldValue": gold_q["raw_number"],
                "apiValue": api_q["raw_number"],
                "usedNormalized": False,
            }
        )

    first_result: dict[str, Any] | None = None
    for candidate in candidates:
        result = within_tolerance(
            candidate["goldValue"],
            candidate["apiValue"],
            rtol=rtol,
            atol=atol,
        )
        enriched = {
            "isValid": True,
            "isCorrect": result["isCorrect"],
            "errorReason": None
            if result["isCorrect"]
            else f"Value outside tolerance: gold={candidate['goldValue']}, api={candidate['apiValue']}",
            "lower": result["lower"],
            "upper": result["upper"],
            "tolerance": result["tolerance"],
            "unitMismatch": False,
            "compared_gold_value": candidate["goldValue"],
            "compared_api_value": candidate["apiValue"],
            "used_normalized_units": candidate["usedNormalized"],
            "compare_strategy": candidate["strategy"],
        }
        if first_result is None:
            first_result = enriched
        if result["isCorrect"]:
            return enriched

    return first_result or {
        "isValid": False,
        "isCorrect": False,
        "errorReason": "No numeric comparison candidates",
    }


def normalize_for_string_match(value: Any) -> str:
    text = normalize_superscript(value)
    replacements = {
        r"\Omega": "ohm",
        "\u03a9": "ohm",
        "\u2126": "ohm",
        "\u03c9": "ohm",
        r"\mu": "u",
        "\u03bc": "u",
        "\u00b5": "u",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[`*_~]", " ", text)
    text = re.sub(r"[\"'\u201c\u201d\u2018\u2019]", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def compare_strings(gold_raw: Any, api_raw: Any) -> dict[str, Any]:
    gold_text = normalize_for_string_match(gold_raw)
    api_text = normalize_for_string_match(api_raw)
    if not gold_text:
        return {
            "mode": "string_contains",
            "isValid": False,
            "isCorrect": False,
            "errorReason": "Gold answer is empty, cannot compare as string",
            "normalizedGoldString": gold_text,
            "normalizedApiString": api_text,
        }
    if not api_text:
        return {
            "mode": "string_contains",
            "isValid": False,
            "isCorrect": False,
            "errorReason": "API answer is empty, cannot compare as string",
            "normalizedGoldString": gold_text,
            "normalizedApiString": api_text,
        }
    is_correct = api_text in gold_text
    return {
        "mode": "string_contains",
        "isValid": True,
        "isCorrect": is_correct,
        "errorReason": None if is_correct else "String mismatch: API answer is not contained in gold answer",
        "normalizedGoldString": gold_text,
        "normalizedApiString": api_text,
        "compare_strategy": "api_string_in_gold_string",
    }


def compare_answers(
    gold_raw: Any,
    api_raw: Any,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> dict[str, Any]:
    gold_bool = extract_boolean(gold_raw)
    api_bool = extract_boolean(api_raw)
    if gold_bool is not None:
        if api_bool is None:
            return {
                "mode": "boolean",
                "isValid": False,
                "isCorrect": False,
                "errorReason": "Gold answer is Yes/No, but API answer is not parseable as Yes/No",
                "goldBool": gold_bool,
                "apiBool": api_bool,
            }
        return {
            "mode": "boolean",
            "isValid": True,
            "isCorrect": gold_bool == api_bool,
            "errorReason": None if gold_bool == api_bool else f"Boolean mismatch: gold={gold_bool}, api={api_bool}",
            "goldBool": gold_bool,
            "apiBool": api_bool,
        }

    gold_q = extract_quantity(gold_raw)
    api_q = extract_quantity(api_raw)
    if gold_q is not None or api_q is not None:
        result = compare_quantities(gold_q, api_q, rtol=rtol, atol=atol)
        return {
            "mode": "numeric_unit",
            **result,
            "goldQuantity": gold_q,
            "apiQuantity": api_q,
        }

    return compare_strings(gold_raw, api_raw)


def normalize_text(value: Any) -> str:
    text = value_to_text(value).strip().lower()
    text = text.replace(r"\sqrt", "sqrt")
    text = text.replace("\u2212", "-").replace("\u2013", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;")


def normalize_answer(value: Any) -> str:
    """Normalize answer labels kept for compatibility with the older scorer."""
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


def extract_number(text: str) -> tuple[float | None, str]:
    quantity = extract_quantity(text)
    if quantity is None:
        return None, ""
    return float(quantity["raw_number"]), str(quantity["raw_unit"] or "")


def _scaled(value: float, unit: str) -> tuple[str, float] | None:
    info = get_unit_info(unit)
    if not info["recognized"] or not info["base"]:
        return None
    return str(info["base"]), value * float(info["factor"])


def numeric_match(
    predicted_answer: str,
    expected_answer: str,
    expected_unit: str = "",
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> dict[str, Any] | None:
    expected_text = combine_answer_and_unit(expected_answer, expected_unit)
    comparison = compare_answers(expected_text, predicted_answer, rtol=rtol, atol=atol)
    if comparison.get("mode") != "numeric_unit":
        return None
    gold_q = comparison.get("goldQuantity") or {}
    api_q = comparison.get("apiQuantity") or {}
    return {
        "correct": bool(comparison.get("isCorrect")),
        "method": "numeric_unit",
        "reason": comparison.get("errorReason") or f"numeric tolerance={comparison.get('tolerance'):g}",
        "predicted_value": comparison.get("compared_api_value", api_q.get("raw_number")),
        "expected_value": comparison.get("compared_gold_value", gold_q.get("raw_number")),
        "predicted_unit": normalize_unit(api_q.get("raw_unit", "")),
        "expected_unit": normalize_unit(gold_q.get("raw_unit", expected_unit)),
        "comparison": comparison,
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
    """Return P1 correctness for both logic and physics answers."""
    return bool(
        evaluate_prediction(
            predicted_answer,
            expected_answer,
            predicted_unit=predicted_unit,
            expected_unit=expected_unit,
            rtol=rtol,
            atol=atol,
        )["correct"]
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
    predicted_text = combine_answer_and_unit(predicted_answer, predicted_unit)
    expected_text = combine_answer_and_unit(expected_answer, expected_unit)

    predicted_normalized = normalize_answer(predicted_text)
    expected_normalized = normalize_answer(expected_text)
    if predicted_normalized == expected_normalized:
        return {
            "correct": True,
            "method": "normalized_answer_exact",
            "reason": "normalized answer exact match",
            "predicted_normalized": predicted_normalized,
            "expected_normalized": expected_normalized,
            "predicted_unit": normalize_unit(predicted_unit),
            "expected_unit": normalize_unit(expected_unit),
        }

    comparison = compare_answers(expected_text, predicted_text, rtol=rtol, atol=atol)
    method = str(comparison.get("mode") or "unknown")
    return {
        "correct": bool(comparison.get("isValid") and comparison.get("isCorrect")),
        "method": method,
        "reason": comparison.get("errorReason") or f"{method} match",
        "predicted_normalized": predicted_normalized,
        "expected_normalized": expected_normalized,
        "predicted_unit": normalize_unit(predicted_unit),
        "expected_unit": normalize_unit(expected_unit),
        "comparison": comparison,
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
