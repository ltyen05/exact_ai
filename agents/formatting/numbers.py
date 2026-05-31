"""Numeric parsing and formatting helpers."""

import math
import re
from typing import Any, Optional


def parse_float_like(s: Any) -> Optional[float]:
    if s is None:
        return None
    text = str(s).strip()
    if not text:
        return None
    text = text.replace("×", "*").replace("x", "*").replace("^", "**")
    text = text.replace("\\times", "*")
    text = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", text)
    text = text.replace("√", "sqrt")
    text = text.replace(",", "")
    # Extract a leading expression such as 24.45 * 10**-3
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:\s*\*\s*10\s*\*\*\s*[-+]?\d+)?", text)
    if not m:
        return None
    expr = m.group(0)
    try:
        import sympy as sp
        return float(sp.N(sp.sympify(expr)))
    except Exception:
        try:
            return float(expr)
        except Exception:
            return None


def as_number(value: Any) -> Optional[float]:
    """Coerce numeric parser/LLM values, including simple scientific/sqrt text."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        text = text.replace("×", "x").replace("√", "sqrt")
        text = re.sub(r"(\d)\s*sqrt", r"\1*sqrt", text)
        sci = re.fullmatch(
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:x|\*)\s*10\s*(?:\^|\*\*)?\s*([+-]?\d+)",
            text,
        )
        if sci:
            return float(sci.group(1)) * (10.0 ** int(sci.group(2)))
        sqrt_match = re.fullmatch(
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)?)\s*\*?\s*sqrt\s*\(?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)?",
            text,
        )
        if sqrt_match:
            coefficient = sqrt_match.group(1)
            factor = float(coefficient) if coefficient not in {"", "+", "-"} else (-1.0 if coefficient == "-" else 1.0)
            return factor * math.sqrt(float(sqrt_match.group(2)))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_number(x: float, digits: int = 6) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "Unknown"
    if x != 0 and abs(x) < 1e-12:
        # Keep tiny non-zero magnitudes visible instead of collapsing to 0.
        return f"{x:.{digits}e}"
    if abs(x) >= 1e-6 and abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.{digits}g}"


def normalize_unit(unit: str) -> str:
    normalized = str(unit or "").strip()
    normalized = normalized.replace("μ", "u").replace("µ", "u").replace("Ω", "Ohm")
    normalized = normalized.replace("^2", "2").replace("²", "2")
    normalized = re.sub(r"\s+", "", normalized)
    if len(normalized) > 1 and normalized[0] == "M" and normalized[1].isalpha():
        return "M" + normalized[1:].lower()
    return normalized.lower()


SI_TO_REQUESTED_UNIT_FACTORS = {
    "j": 1.0,
    "mj": 1e3,
    "uj": 1e6,
    "microj": 1e6,
    "f": 1.0,
    "uf": 1e6,
    "microf": 1e6,
    "nf": 1e9,
    "pf": 1e12,
    "c": 1.0,
    "uc": 1e6,
    "microc": 1e6,
    "nc": 1e9,
    "pc": 1e12,
    "mc": 1e3,
    "m": 1.0,
    "cm": 1e2,
    "mm": 1e3,
    "m2": 1.0,
    "cm2": 1e4,
    "h": 1.0,
    "mh": 1e3,
    "uh": 1e6,
    "microh": 1e6,
    "v": 1.0,
    "mv": 1e3,
    "kv": 1e-3,
    "a": 1.0,
    "ma": 1e3,
    "ua": 1e6,
    "microa": 1e6,
    "ohm": 1.0,
    "kohm": 1e-3,
    "mohm": 1e3,
    "Mohm": 1e-6,
    "w": 1.0,
    "wb": 1.0,
    "mwb": 1e3,
    "uwb": 1e6,
    "microwb": 1e6,
    "nwb": 1e9,
    "t": 1.0,
    "n": 1.0,
    "n/c": 1.0,
    "v/m": 1.0,
    "hz": 1.0,
    "khz": 1e-3,
    "Mhz": 1e-6,
    "rad/s": 1.0,
    "s": 1.0,
    "ms": 1e3,
    "us": 1e6,
    "micros": 1e6,
    "%": 1.0,
    "": 1.0,
    "dimensionless": 1.0,
}


def convert_si_to_requested(value: float, unit: str) -> float:
    return value * SI_TO_REQUESTED_UNIT_FACTORS.get(normalize_unit(unit), 1.0)
