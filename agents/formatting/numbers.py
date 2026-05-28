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


def format_number(x: float, digits: int = 6) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "Unknown"
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.{digits}g}"
