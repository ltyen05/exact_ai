import json
import math
import re
from typing import Any, Dict, Optional


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extractor for local LLM outputs."""
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


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


def standard_response(answer: Any, explanation: str, **extra: Any) -> Dict[str, Any]:
    out = {
        "answer": str(answer) if answer is not None else "Unknown",
        "explanation": explanation or "No explanation was generated.",
    }
    for k, v in extra.items():
        if v is not None:
            out[k] = v
    return out
