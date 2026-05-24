from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from agents.formatter import format_number


@dataclass
class CalcResult:
    answer: str
    unit: str
    cot: List[str]
    confidence: float


UNIT_FACTOR = {
    "c": 1.0,
    "mc": 1e-3,
    "uc": 1e-6,
    "μc": 1e-6,
    "µc": 1e-6,
    "nc": 1e-9,
    "f": 1.0,
    "mf": 1e-3,
    "uf": 1e-6,
    "μf": 1e-6,
    "µf": 1e-6,
    "nf": 1e-9,
    "pf": 1e-12,
    "v": 1.0,
    "a": 1.0,
    "ma": 1e-3,
    "cm": 1e-2,
    "mm": 1e-3,
    "m": 1.0,
    "ohm": 1.0,
    "ω": 1.0,
    "Ω": 1.0,
    "n": 1.0,
    "j": 1.0,
    "nj": 1e-9,
}


def _norm(q: str) -> str:
    return q.replace("µ", "μ").replace("×", "x").replace("^", "^")


def _num(s: str) -> float:
    s = s.strip().replace("−", "-")
    s = s.replace("\\times", "x").replace("×", "x")
    s = s.replace("^", "**")
    # 6 x 10**-8
    s = re.sub(r"\s*x\s*10", "*10", s, flags=re.I)
    return float(eval(s, {"__builtins__": {}}, {}))


def find_value(question: str, names: Tuple[str, ...], units: Tuple[str, ...]) -> Optional[Tuple[float, str, str]]:
    q = _norm(question)
    name_pat = "|".join(re.escape(n) for n in names)
    unit_pat = "|".join(re.escape(u) for u in units)
    patterns = [
        rf"(?:{name_pat})\s*=\s*([-+]?\d+(?:\.\d+)?(?:\s*(?:x|×|\\times)\s*10\s*\^?\s*[-+]?\d+)?)\s*({unit_pat})\b",
        rf"([-+]?\d+(?:\.\d+)?(?:\s*(?:x|×|\\times)\s*10\s*\^?\s*[-+]?\d+)?)\s*({unit_pat})\b\s*(?:for|as)?\s*(?:{name_pat})",
    ]
    for pat in patterns:
        m = re.search(pat, q, flags=re.I)
        if m:
            value = _num(m.group(1))
            unit = m.group(2)
            unit_key = unit.lower().replace("Ω", "ohm")
            return value * UNIT_FACTOR.get(unit_key, 1.0), unit, m.group(1)
    return None


def find_all_forces(question: str) -> List[float]:
    vals = []
    # Pattern: "each with a magnitude of 5 N" means two equal forces.
    m_each = re.search(r"each\s+with\s+(?:a\s+)?magnitude\s+of\s+([-+]?\d+(?:\.\d+)?)\s*N\b", question, flags=re.I)
    if m_each:
        v = float(m_each.group(1))
        return [v, v]
    for m in re.finditer(r"([-+]?\d+(?:\.\d+)?)\s*N\b", question, flags=re.I):
        vals.append(float(m.group(1)))
    return vals


def find_angle(question: str) -> Optional[float]:
    m = re.search(r"angle(?: of)?\s*(?:between)?\s*(?:the two forces)?\s*(?:is|of)?\s*([-+]?\d+(?:\.\d+)?)\s*(?:°|degree|degrees)", question, flags=re.I)
    if not m:
        m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(?:°|degree|degrees)", question, flags=re.I)
    return float(m.group(1)) if m else None


def solve_by_formula(question: str) -> Optional[CalcResult]:
    qlow = question.lower()

    # Resultant forces: same/opposite/perpendicular/angle.
    forces = find_all_forces(question)
    if "resultant" in qlow and len(forces) >= 2:
        f1, f2 = forces[0], forces[1]
        if "same direction" in qlow:
            r = f1 + f2
            return CalcResult(format_number(r), "N", [f"Forces are same-direction: R = {f1} + {f2} = {r} N."], 0.92)
        if "opposite direction" in qlow or "opposite directions" in qlow:
            r = abs(f1 - f2)
            return CalcResult(format_number(r), "N", [f"Forces are opposite: R = |{f1} - {f2}| = {r} N."], 0.92)
        if "perpendicular" in qlow or "90" in qlow:
            r = math.sqrt(f1 * f1 + f2 * f2)
            return CalcResult(format_number(r), "N", [f"Forces are perpendicular: R = sqrt({f1}^2 + {f2}^2) = {r} N."], 0.92)
        theta = find_angle(question)
        if theta is not None:
            r = math.sqrt(f1 * f1 + f2 * f2 + 2 * f1 * f2 * math.cos(math.radians(theta)))
            return CalcResult(format_number(r), "N", [f"Use vector addition: R = sqrt(F1^2 + F2^2 + 2F1F2cos(theta)) with theta={theta}° gives {r} N."], 0.90)

    # Find angle when F1=F2=R pattern.
    if "find the angle" in qlow and len(forces) >= 3:
        f1, f2, r = forces[0], forces[1], forces[2]
        cosv = (r * r - f1 * f1 - f2 * f2) / (2 * f1 * f2)
        cosv = max(-1.0, min(1.0, cosv))
        theta = math.degrees(math.acos(cosv))
        return CalcResult(format_number(theta), "Độ", [f"From R^2=F1^2+F2^2+2F1F2cos(theta), theta={theta} degrees."], 0.90)

    # Capacitor energy E = 1/2 C V^2.
    if ("energy" in qlow or "stored" in qlow) and "capacitor" in qlow:
        cv = find_value(question, ("C", "capacitance"), ("μF", "µF", "uF", "pF", "nF", "F"))
        vv = find_value(question, ("U", "V", "voltage", "potential difference"), ("V",))
        if cv and vv:
            C, cunit, craw = cv
            V, _, vraw = vv
            E = 0.5 * C * V * V
            # choose nJ for pF-scale questions, otherwise J. This is physical; retrieval fallback can override exact training quirks.
            if C < 1e-9:
                return CalcResult(format_number(E / 1e-9), "nJ", [f"Use E = 1/2 C U^2. Convert C={craw} {cunit} to {C} F, U={V} V, so E={E} J = {E/1e-9} nJ."], 0.87)
            return CalcResult(format_number(E), "J", [f"Use E = 1/2 C U^2. Convert C={craw} {cunit} to {C} F, U={V} V, so E={E} J."], 0.87)

    # Charge stored Q = C V.
    if "charge" in qlow and "capacitor" in qlow:
        cv = find_value(question, ("C", "capacitance"), ("μF", "µF", "uF", "pF", "nF", "F"))
        vv = find_value(question, ("U", "V", "voltage", "potential difference"), ("V",))
        if cv and vv:
            C, cunit, craw = cv
            V, _, _ = vv
            Q = C * V
            if Q < 1e-6:
                return CalcResult(format_number(Q / 1e-9), "nC", [f"Use Q = C U. C={C} F and U={V} V, so Q={Q} C = {Q/1e-9} nC."], 0.87)
            return CalcResult(format_number(Q / 1e-6), "μC", [f"Use Q = C U. C={C} F and U={V} V, so Q={Q} C = {Q/1e-6} μC."], 0.87)

    # Parallel-plate capacitance C = eps0 A/d.
    if "parallel-plate" in qlow and "capacitance" in qlow and "area" in qlow:
        av = find_value(question, ("A", "area", "plate area"), ("cm", "m", "mm"))
        dv = find_value(question, ("d", "distance", "separation"), ("mm", "cm", "m"))
        # area regex needs square units; handle explicitly
        am = re.search(r"(?:area|A)\s*(?:of)?\s*(?:=|is)?\s*([-+]?\d+(?:\.\d+)?)\s*cm\^?2", question, flags=re.I)
        dm = re.search(r"(?:distance|separation|d)\s*(?:between the plates)?\s*(?:=|is)?\s*([-+]?\d+(?:\.\d+)?)\s*mm", question, flags=re.I)
        if am and dm:
            A = float(am.group(1)) * 1e-4
            d = float(dm.group(1)) * 1e-3
            eps0 = 8.854e-12
            C = eps0 * A / d
            return CalcResult(format_number(C / 1e-12), "pF", [f"Use C = eps0*A/d. A={A} m^2, d={d} m, C={C} F = {C/1e-12} pF."], 0.87)

    # RLC resonance frequency scaling tasks.
    if ("rlc" in qlow or "series circuit" in qlow or "circuit" in qlow) and ("rms voltage across" in qlow or "voltage across r" in qlow):
        xl = find_value(question, ("XL", "X_L", "inductive reactance"), ("Ω", "ohm"))
        xc = find_value(question, ("XC", "X_C", "capacitive reactance"), ("Ω", "ohm"))
        u = find_value(question, ("U", "voltage", "source voltage", "total voltage"), ("V",))
        factor = None
        if "doubled" in qlow:
            factor = 2
        elif "tripled" in qlow:
            factor = 3
        elif "quadrupled" in qlow or "fourfold" in qlow or "4 times" in qlow:
            factor = 4
        else:
            m = re.search(r"(?:increased by|frequency is)\s*(\d+(?:\.\d+)?)\s*times", qlow)
            if m:
                factor = float(m.group(1))
        if xl and xc and u and factor:
            xl_new = xl[0] * factor
            xc_new = xc[0] / factor
            if abs(xl_new - xc_new) < 1e-9:
                return CalcResult(format_number(u[0]), "V", [f"After frequency scaling by {factor}, XL'={xl_new} Ω and XC'={xc_new} Ω, so the circuit is at resonance and U_R=U={u[0]} V."], 0.93)

    return None
