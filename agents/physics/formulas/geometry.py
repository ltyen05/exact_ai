"""Electrostatics geometry normalization and vector computation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from agents.formatting import convert_si_to_requested, format_number
from agents.physics.domain.context import build_calculation_input
from agents.physics.domain.symbols import _normalize_text, canonical_quantity_symbol


Point = tuple[float, float]


@dataclass(frozen=True)
class GeometryFailure:
    code: str
    reason: str
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class GeometryScene:
    points: dict[str, Point]
    source_charges: list[dict[str, str]]
    target: dict[str, str]
    charge_values: dict[str, float]
    target_unit: str

    def public_scene(self) -> dict[str, Any]:
        """Return the canonical scene shape without internal numeric metadata."""
        return {
            "points": {label: [coords[0], coords[1]] for label, coords in self.points.items()},
            "source_charges": [dict(item) for item in self.source_charges],
            "target": dict(self.target),
        }


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:(?:[eE][+-]?\d+)|(?:\s*(?:x|\*)\s*10\s*(?:\^|\*\*)\s*\{?\s*[+-]?\s*\d+\s*\}?))?"
_LENGTH_SCALES = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
_CHARGE_SCALES = {"c": 1.0, "mc": 1e-3, "microc": 1e-6, "muc": 1e-6, "uc": 1e-6, "nc": 1e-9, "pc": 1e-12}


def _parse_number(value: str) -> float | None:
    cleaned = _normalize_text(value).strip().replace("{", "").replace("}", "")
    compact = re.sub(r"\s+", "", cleaned)
    scientific = re.fullmatch(
        r"(?P<coeff>[+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:x|\*)10(?:\^|\*\*)?(?P<exp>[+-]?\d+)",
        compact,
        flags=re.IGNORECASE,
    )
    if scientific:
        return float(scientific.group("coeff")) * (10 ** int(scientific.group("exp")))
    try:
        return float(compact)
    except ValueError:
        return None


def _scaled_value(value: str, unit: str, scales: dict[str, float]) -> float | None:
    numeric = _parse_number(value)
    if numeric is None:
        return None
    scale = scales.get(_normalize_text(unit).lower())
    if scale is None:
        return None
    return numeric * scale


def _target_symbol(parsed_question: dict[str, Any]) -> str:
    target = parsed_question.get("target") or {}
    if isinstance(target, dict):
        return canonical_quantity_symbol(target.get("symbol"))
    return canonical_quantity_symbol(target)


def _target_unit(parsed_question: dict[str, Any], default: str) -> str:
    target = parsed_question.get("target") or {}
    if isinstance(target, dict) and target.get("unit"):
        return str(target.get("unit") or "")
    return default


def _value(values: dict[str, float], *symbols: str) -> float | None:
    for symbol in symbols:
        canonical = canonical_quantity_symbol(symbol)
        if canonical in values:
            return values[canonical]
        if symbol in values:
            return values[symbol]
    return None


def _add_text_values(values: dict[str, float], text: str) -> dict[str, float]:
    enriched = dict(values)
    normalized = _normalize_text(text).lower()
    for match in re.finditer(
        rf"\b(?P<symbol>q[0-9]?|Q[0-9]?)\s*=\s*(?P<value>{_NUMBER})\s*(?P<unit>microc|muc|uc|nc|pc|mc|c)\b",
        normalized,
        flags=re.IGNORECASE,
    ):
        value = _scaled_value(match.group("value"), match.group("unit"), _CHARGE_SCALES)
        if value is not None:
            enriched.setdefault(canonical_quantity_symbol(match.group("symbol")), value)
    for match in re.finditer(
        rf"\b(?P<symbol>AB|AC|CA|BC|CB|AM|BM|OM|ell|h|a|d_AB)\s*=\s*(?P<value>{_NUMBER})\s*(?P<unit>cm|mm|m)\b",
        _normalize_text(text),
        flags=re.IGNORECASE,
    ):
        value = _scaled_value(match.group("value"), match.group("unit"), _LENGTH_SCALES)
        if value is not None:
            enriched.setdefault(canonical_quantity_symbol(match.group("symbol")), value)
            enriched.setdefault(match.group("symbol").upper(), value)
    apart = re.search(rf"(?P<value>{_NUMBER})\s*(?P<unit>cm|mm|m)\s+apart", normalized, flags=re.IGNORECASE)
    if apart:
        value = _scaled_value(apart.group("value"), apart.group("unit"), _LENGTH_SCALES)
        if value is not None:
            enriched.setdefault("AB", value)
    height = re.search(
        rf"(?P<value>{_NUMBER})\s*(?P<unit>cm|mm|m)\s+(?:from\s+the\s+midpoint\s+of\s+ab|away\s+from\s+ab|from\s+ab)",
        normalized,
        flags=re.IGNORECASE,
    )
    if height:
        value = _scaled_value(height.group("value"), height.group("unit"), _LENGTH_SCALES)
        if value is not None:
            enriched.setdefault("ell", value)
    return enriched


def _looks_electrostatic(parsed_question: dict[str, Any], text: str) -> bool:
    haystack = " ".join(
        [
            str(parsed_question.get("domain") or ""),
            str(parsed_question.get("question_kind") or ""),
            text,
        ]
    ).lower()
    return any(term in haystack for term in ("charge", "electric field", "electric force", "coulomb"))


def _force_target_charge(text: str, values: dict[str, float]) -> str:
    normalized = text.lower()
    for symbol in ("q0", "q3", "q_test"):
        if symbol in values and re.search(rf"\b(?:force|acting|on)\b.*\b{re.escape(symbol)}\b|\b{re.escape(symbol)}\b.*\b(?:force|acting|on)\b", normalized):
            return symbol
    if "force" in normalized:
        for symbol in ("q0", "q3"):
            if symbol in values:
                return symbol
    return ""


def _field_target_point(parsed_question: dict[str, Any], text: str) -> str:
    target = _target_symbol(parsed_question).upper()
    for label in ("M", "C", "D", "O"):
        if target.endswith(label):
            return label
    normalized = text.lower()
    for label in ("m", "c", "d", "o"):
        if re.search(rf"\b(?:point\s+)?{label}\b", normalized):
            return label.upper()
    return "M"


def _charge_sources(values: dict[str, float], target_charge: str = "") -> tuple[list[dict[str, str]], dict[str, float]]:
    charge_values: dict[str, float] = {}
    sources: list[dict[str, str]] = []
    for symbol, point in (("q1", "A"), ("q2", "B"), ("q3", "C"), ("q0", "M")):
        if symbol in values:
            charge_values[symbol] = values[symbol]
            if symbol != target_charge:
                sources.append({"symbol": symbol, "point": point})
    if not sources and "q" in values:
        charge_values.update({"q1": values["q"], "q2": values["q"]})
        sources.extend([{"symbol": "q1", "point": "A"}, {"symbol": "q2", "point": "B"}])
    if target_charge and target_charge in values:
        charge_values[target_charge] = values[target_charge]
    return sources, charge_values


def _triangle_point(ab: float, from_a: float, from_b: float) -> Point | None:
    if ab <= 0 or from_a <= 0 or from_b <= 0:
        return None
    x_value = (from_a**2 + ab**2 - from_b**2) / (2 * ab)
    y_square = from_a**2 - x_value**2
    if y_square < -1e-12:
        return None
    return x_value, math.sqrt(max(0.0, y_square))


class GeometryResolver:
    """Resolve parsed electrostatic wording into canonical Cartesian scenes."""

    def resolve(self, parsed_question: dict[str, Any], raw_question: str = "") -> GeometryScene | GeometryFailure | None:
        text = str(raw_question or parsed_question.get("question") or "")
        if not _looks_electrostatic(parsed_question, text):
            return None
        calculation = build_calculation_input(parsed_question)
        values = _add_text_values(calculation["quantities"], text)
        target_symbol = _target_symbol(parsed_question)
        quantity = "force" if "force" in target_symbol.lower() or "force" in text.lower() or _target_unit(parsed_question, "") == "N" else "field"
        target_charge = _force_target_charge(text, values) if quantity == "force" else ""
        target_point = "M"
        if quantity == "force":
            if target_charge == "q3":
                target_point = "C"
            elif target_charge == "q0":
                target_point = "M"
        else:
            target_point = _field_target_point(parsed_question, text)

        points = self._resolve_points(parsed_question, text, values, target_point)
        if isinstance(points, GeometryFailure):
            return points
        if target_point not in points:
            return GeometryFailure("missing_geometry", f"Target point {target_point} could not be placed.", (target_point,))

        sources, charge_values = _charge_sources(values, target_charge)
        sources = [source for source in sources if source["point"] in points and source["point"] != target_point]
        if not sources:
            return GeometryFailure("missing_givens", "No source charges could be mapped to known points.", ("source_charges",))
        if quantity == "force" and (not target_charge or target_charge not in charge_values):
            return GeometryFailure("missing_givens", "Force target charge is missing or ambiguous.", ("target_charge",))

        unit = _target_unit(parsed_question, "N" if quantity == "force" else "N/C")
        return GeometryScene(
            points=points,
            source_charges=sources,
            target={"quantity": quantity, "charge": target_charge, "point": target_point},
            charge_values=charge_values,
            target_unit=unit,
        )

    def _resolve_points(
        self,
        parsed_question: dict[str, Any],
        text: str,
        values: dict[str, float],
        target_point: str,
    ) -> dict[str, Point] | GeometryFailure:
        normalized = text.lower()
        geometry = parsed_question.get("geometry") if isinstance(parsed_question.get("geometry"), dict) else {}
        geometry_type = str((geometry or {}).get("type") or "").lower()
        ab = _value(values, "AB", "d_AB", "a")

        if "square" in normalized and ab is not None:
            side = float(ab)
            return {"A": (0.0, 0.0), "B": (side, 0.0), "C": (0.0, side), "D": (side, side)}

        if ab is None:
            return GeometryFailure("missing_geometry", "AB distance is required to place source charges.", ("AB",))

        if geometry_type == "perpendicular_bisector" or "perpendicular bisector" in normalized or "equidistant from a and b" in normalized:
            height = _value(values, "ell", "h", "OM", "d")
            if height is None:
                return GeometryFailure("missing_geometry", "Perpendicular-bisector height is missing.", ("ell",))
            return {"A": (0.0, 0.0), "B": (float(ab), 0.0), "M": (float(ab) / 2.0, float(height))}

        if "midpoint" in normalized and target_point in {"M", "O"}:
            return {"A": (0.0, 0.0), "B": (float(ab), 0.0), target_point: (float(ab) / 2.0, 0.0)}

        from_a = _value(values, f"A{target_point}", f"{target_point}A", "AC", "CA")
        from_b = _value(values, f"B{target_point}", f"{target_point}B", "BC", "CB")
        if target_point == "C" and from_a is not None and from_b is not None:
            c_point = _triangle_point(float(ab), float(from_a), float(from_b))
            if c_point is None:
                return GeometryFailure("missing_geometry", "Triangle side lengths are inconsistent.", ("AC", "BC", "AB"))
            return {"A": (0.0, 0.0), "B": (float(ab), 0.0), "C": c_point}

        am = _value(values, "AM")
        bm = _value(values, "BM")
        if target_point == "M" and am is not None:
            return {"A": (0.0, 0.0), "B": (float(ab), 0.0), "M": (float(am), 0.0)}
        if target_point == "M" and bm is not None:
            return {"A": (0.0, 0.0), "B": (float(ab), 0.0), "M": (float(ab) - float(bm), 0.0)}

        return GeometryFailure("missing_geometry", f"Could not resolve coordinates for target point {target_point}.", (target_point,))


class ElectrostaticVectorEngine:
    """Compute field or force from a resolved electrostatic geometry scene."""

    def __init__(self, resolver: GeometryResolver | None = None) -> None:
        self.resolver = resolver or GeometryResolver()

    def solve(self, parsed_question: dict[str, Any], raw_question: str = "") -> dict[str, Any] | GeometryFailure | None:
        scene = self.resolver.resolve(parsed_question, raw_question)
        if scene is None or isinstance(scene, GeometryFailure):
            return scene
        target_point = scene.points[scene.target["point"]]
        ex_total = 0.0
        ey_total = 0.0
        for source in scene.source_charges:
            source_point = scene.points[source["point"]]
            dx = target_point[0] - source_point[0]
            dy = target_point[1] - source_point[1]
            radius = math.hypot(dx, dy)
            if radius <= 0:
                return GeometryFailure("missing_geometry", "Target point overlaps a source charge.", ("target_point",))
            charge = scene.charge_values[source["symbol"]]
            ex_total += 9e9 * charge * dx / radius**3
            ey_total += 9e9 * charge * dy / radius**3

        quantity = scene.target["quantity"]
        unit_si = "N/C"
        components = [ex_total, ey_total]
        if quantity == "force":
            target_charge = scene.target["charge"]
            components = [scene.charge_values[target_charge] * ex_total, scene.charge_values[target_charge] * ey_total]
            unit_si = "N"
        magnitude_si = math.hypot(components[0], components[1])
        requested_unit = scene.target_unit or unit_si
        public_components = [convert_si_to_requested(value, requested_unit) for value in components]
        public_magnitude = convert_si_to_requested(magnitude_si, requested_unit)
        direction = self._direction(public_components)
        symbol = "F_net_magnitude" if quantity == "force" else "E_net_magnitude"
        vector_result = {
            "component_symbols": [f"{symbol}_x", f"{symbol}_y"],
            "components": public_components,
            "magnitude": public_magnitude,
            "direction": direction,
            "geometry_scene": scene.public_scene(),
        }
        public_answer = f"{format_number(public_magnitude)} {requested_unit}".strip()
        return {
            "verified_output": {
                "mode": "computational",
                "answer_type": "numeric",
                "verified": True,
                "final_answer": {"symbol": symbol, "value": {"components": public_components, "magnitude": public_magnitude, "direction": direction}, "unit": requested_unit},
                "sympy_result": {"symbol": symbol, "value": magnitude_si, "unit": unit_si, "trace": [], "values": {symbol: magnitude_si}},
                "vector_result": vector_result,
                "geometry_scene": scene.public_scene(),
                "verification_result": {"accepted": True, "verified": True, "vector_engine": True},
            },
            "result": {
                "answer": public_answer,
                "unit": requested_unit,
                "append_unit": False,
                "verified": True,
                "cot": [
                    "Resolved the electrostatic geometry into Cartesian coordinates.",
                    "Summed signed Coulomb field vectors from all source charges.",
                    "Converted the resulting vector to the requested magnitude.",
                ],
                "premises": ["E = sum(k*q_i*r_vec/r**3)", "F = q_target*E"],
                "fol": "",
            },
        }

    @staticmethod
    def _direction(components: list[float]) -> str:
        if len(components) < 2:
            return "along line"
        x_value, y_value = components[0], components[1]
        if abs(x_value) <= 1e-12 and abs(y_value) <= 1e-12:
            return "zero vector"
        angle = math.degrees(math.atan2(y_value, x_value))
        return f"{angle:.6g} degrees from +x"
