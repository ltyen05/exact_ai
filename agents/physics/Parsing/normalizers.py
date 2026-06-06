"""Output normalization for the physics semantic parser."""

from __future__ import annotations

import math
import re
from typing import Any

from agents.physics.domain.symbols import _normalize_text
from agents.physics.domain.units import UNIT_PATTERN, UNIT_TO_SI, _canonical_unit_key


def normalize_value(value: Any) -> Any:
    """Normalize parser values recursively while preserving non-text scalars."""
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(normalize_value(key)): normalize_value(item)
            for key, item in value.items()
        }
    return value


class ParsingOutputNormalizer:
    """Enforce the compact internal parser contract and repair common LLM issues."""

    def compact_output(self, parsed: dict[str, Any], question: str) -> dict[str, Any]:
        """Drop empty sections and normalize known parser edge cases."""
        compact: dict[str, Any] = {
            "question": str(parsed.get("question") or question),
            "domain": str(parsed.get("domain") or "unknown"),
            "target": parsed.get("target") if isinstance(parsed.get("target"), dict) else {},
            "givens": parsed.get("givens") if isinstance(parsed.get("givens"), list) else [],
            "relations": parsed.get("relations") if isinstance(parsed.get("relations"), list) else [],
            "question_kind": str(parsed.get("question_kind") or "computational"),
        }
        for key in ("geometry", "comparison", "options", "answer_format", "warnings"):
            value = parsed.get(key)
            if self._present(value):
                if key == "geometry" and isinstance(value, dict) and not value.get("present"):
                    continue
                if key == "comparison" and isinstance(value, dict) and not value.get("present"):
                    continue
                compact[key] = value

        compact = self._normalize_compact(compact, question)
        self._apply_unit_corrections(compact, question)
        self._normalize_resonance_yes_no(compact, question)
        self._normalize_formula_only_question(compact, question)
        self._apply_requested_form(compact, question)
        self._normalize_electrostatic_geometry_from_text(compact, question)
        self._normalize_perpendicular_bisector_geometry(compact)
        return compact

    @staticmethod
    def _present(value: Any) -> bool:
        if value in (None, "", False):
            return False
        if isinstance(value, (list, dict)):
            return bool(value)
        return True

    @staticmethod
    def _normalize_compact(compact: dict[str, Any], question: str) -> dict[str, Any]:
        normalized = {
            key: (normalize_value(value) if key != "question" else str(value))
            for key, value in compact.items()
        }
        normalized["question"] = str(compact.get("question") or question)
        return normalized

    @staticmethod
    def _numeric_by_symbol(items: list[Any]) -> dict[str, float]:
        values: dict[str, float] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol") or "")
            value = item.get("si_value")
            if symbol and isinstance(value, (int, float)):
                values[symbol] = float(value)
        return values

    @staticmethod
    def _append_numeric_given(compact: dict[str, Any], symbol: str, value: float, unit: str) -> None:
        givens = compact.get("givens")
        if not isinstance(givens, list):
            givens = []
            compact["givens"] = givens
        for item in givens:
            if isinstance(item, dict) and str(item.get("symbol") or "") == symbol:
                return
        givens.append({"symbol": symbol, "si_value": value, "si_unit": unit, "uncertainty": None})

    @staticmethod
    def _remove_numeric_givens(compact: dict[str, Any], *symbols: str) -> None:
        givens = compact.get("givens")
        if not isinstance(givens, list):
            return
        blocked = set(symbols)
        compact["givens"] = [
            item
            for item in givens
            if not (isinstance(item, dict) and str(item.get("symbol") or "") in blocked)
        ]

    def _put_numeric_given(self, compact: dict[str, Any], symbol: str, value: float, unit: str) -> None:
        self._remove_numeric_givens(compact, symbol)
        self._append_numeric_given(compact, symbol, value, unit)

    @staticmethod
    def _length_from_match(value: str, unit: str) -> float:
        scale = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
        return float(value) * scale[unit.lower()]

    @staticmethod
    def _explicit_si_values(question: str) -> dict[str, tuple[float, str]]:
        values: dict[str, tuple[float, str]] = {}
        pattern = re.compile(
            rf"\b(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
            rf"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*"
            rf"(?P<unit>{UNIT_PATTERN})(?=\b|[^A-Za-z0-9_])",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(question):
            unit_key = _canonical_unit_key(match.group("unit"))
            conversion = UNIT_TO_SI.get(unit_key)
            if conversion is None:
                continue
            scale, si_unit = conversion
            values[match.group("symbol")] = (float(match.group("value")) * scale, si_unit)
        return values

    def _apply_unit_corrections(self, compact: dict[str, Any], question: str) -> None:
        explicit_values = self._explicit_si_values(question)
        if not explicit_values:
            return
        for item in compact.get("givens") or []:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol") or "")
            corrected = explicit_values.get(symbol)
            if corrected is None:
                continue
            item["si_value"], item["si_unit"] = corrected

    @staticmethod
    def _operating_frequency_from_text(question: str) -> float | None:
        number = r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
        patterns = (
            rf"\bfrequency\b(?:\s+is|\s*=|\s+of|\s+at)?\s*{number}\s*hz\b",
            rf"\b{number}\s*hz\b.*\b(?:resonant|resonance)\s+frequency\b",
            rf"\b(?:resonate|resonant|resonance)\b.*?\b{number}\s*hz\b",
        )
        for pattern in patterns:
            match = re.search(pattern, question, flags=re.IGNORECASE)
            if match:
                return float(match.group("value"))
        return None

    @staticmethod
    def _is_resonance_yes_no_question(question: str) -> bool:
        text = question.lower()
        if not any(term in text for term in ("resonance", "resonate", "resonant")):
            return False
        return (
            "does resonance occur" in text
            or "resonance occur" in text
            or "resonate at" in text
            or "resonates at" in text
            or bool(re.search(r"\b(does|do|is|are|will|can)\b.*\?", text))
        )

    def _normalize_resonance_yes_no(self, compact: dict[str, Any], question: str) -> None:
        if not self._is_resonance_yes_no_question(question):
            return
        givens = compact.get("givens")
        if not isinstance(givens, list):
            return
        values = self._numeric_by_symbol(givens)
        if "f" not in values and "frequency" not in values:
            frequency = self._operating_frequency_from_text(question)
            if frequency is not None:
                givens.append(
                    {
                        "symbol": "f",
                        "si_value": frequency,
                        "si_unit": "Hz",
                        "uncertainty": None,
                    }
                )
                values["f"] = frequency
        has_circuit_values = all(symbol in values for symbol in ("L", "C"))
        has_frequency = "f" in values or "frequency" in values
        if not has_circuit_values or not has_frequency:
            return
        compact["question_kind"] = "yes_no_computational"
        compact["target"] = {"symbol": "f_res", "unit": "Hz"}
        expected_symbol = "f" if "f" in values else "frequency"
        compact["comparison"] = {
            "present": True,
            "computed_quantity_symbol": "f_res",
            "given_quantity_symbol": expected_symbol,
            "given_si_value": values[expected_symbol],
            "given_si_unit": "Hz",
        }
        answer_format = compact.get("answer_format")
        if not isinstance(answer_format, dict):
            answer_format = {}
        answer_format["requested_form"] = "yes_no"
        compact["answer_format"] = answer_format

    @staticmethod
    def _normalize_formula_only_question(compact: dict[str, Any], question: str) -> None:
        text = question.lower()
        if compact.get("givens"):
            return
        if not any(term in text for term in ("formula", "expression", "what is", "define")):
            return
        if "resonant angular frequency" not in text and "resonance angular frequency" not in text:
            return
        compact["question_kind"] = "conceptual"
        compact["target"] = {"symbol": "answer", "unit": ""}
        answer_format = compact.get("answer_format")
        if not isinstance(answer_format, dict):
            answer_format = {}
        answer_format["requested_form"] = "conceptual"
        compact["answer_format"] = answer_format

    @staticmethod
    def _distance_item(
        symbol: str,
        expression: str,
        value: float,
        unit: str = "m",
    ) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "expression": expression,
            "si_value": value,
            "si_unit": unit,
        }

    def _normalize_perpendicular_bisector_geometry(self, compact: dict[str, Any]) -> None:
        """Repair general AB perpendicular-bisector geometry into computable distances."""
        geometry = compact.get("geometry")
        if not isinstance(geometry, dict):
            return
        text = " ".join(
            [
                str(compact.get("question") or ""),
                " ".join(str(relation) for relation in compact.get("relations") or []),
                str(geometry.get("type") or ""),
            ]
        ).lower()
        perpendicular_bisector_like = (
            "perpendicular bisector" in text
            or ("equidistant from a and b" in text and "perpendicular" in text and "ab" in text)
            or ("perpendicular to ab" in text and ("midpoint of ab" in text or "away from ab" in text))
        )
        if not perpendicular_bisector_like:
            return

        segment_values = self._numeric_by_symbol(geometry.get("segments") or [])
        given_values = self._numeric_by_symbol(compact.get("givens") or [])
        values = {**given_values, **segment_values}
        base = values.get("AB") or values.get("d_AB")
        height = values.get("ell") or values.get("h") or values.get("OM")
        if base is None:
            return

        d_mid = base / 2
        parsed_d_mid = values.get("d_mid")
        if (
            height is None
            and parsed_d_mid is not None
            and not math.isclose(float(parsed_d_mid), d_mid, rel_tol=1e-9, abs_tol=1e-12)
        ):
            height = float(parsed_d_mid)
        if height is None:
            return

        self._remove_numeric_givens(compact, "d_mid")
        self._put_numeric_given(compact, "AB", float(base), "m")
        self._put_numeric_given(compact, "ell", float(height), "m")

        source_distance = math.sqrt(d_mid**2 + height**2)
        geometry["type"] = "perpendicular_bisector"
        geometry.pop("line_order", None)
        geometry["segments"] = [
            {"symbol": "AB", "si_value": base, "si_unit": "m"},
            {"symbol": "d_mid", "si_value": d_mid, "si_unit": "m"},
            {"symbol": "ell", "si_value": height, "si_unit": "m"},
        ]
        geometry["derived_distances"] = [
            self._distance_item("AM", "sqrt(d_mid**2 + ell**2)", source_distance),
            self._distance_item("BM", "sqrt(d_mid**2 + ell**2)", source_distance),
        ]
        geometry["direction_convention"] = (
            "x-axis from A to B; y-axis from midpoint of AB toward target point"
        )

    def _normalize_electrostatic_geometry_from_text(
        self,
        compact: dict[str, Any],
        question: str,
    ) -> None:
        text = _normalize_text(question).lower()
        if not any(term in text for term in ("charge", "electric", "force")):
            return
        geometry = compact.get("geometry")
        if not isinstance(geometry, dict):
            geometry = {}
        geometry.setdefault("present", True)

        line_order: list[str] | None = None
        order_match = re.search(
            r"points?\s+([a-z])\s*,\s*([a-z])\s*,\s*([a-z])\s+are\s+collinear\s+in\s+that\s+order",
            text,
        )
        if order_match:
            line_order = [order_match.group(index).upper() for index in range(1, 4)]

        if (
            "perpendicular bisector" in text
            or ("equidistant from a and b" in text and "perpendicular" in text and "ab" in text)
            or ("perpendicular to ab" in text and ("midpoint of ab" in text or "away from ab" in text))
        ):
            geometry["type"] = "perpendicular_bisector"
        elif "equilateral triangle" in text:
            geometry["type"] = "equilateral_triangle"
        elif (
            "isosceles right triangle" in text
            or "right-angle vertex" in text
            or "right angle vertex" in text
        ):
            geometry["type"] = "right_triangle"
        elif line_order:
            geometry["type"] = "collinear"
            geometry["line_order"] = line_order
        elif "midpoint" in text:
            geometry["type"] = "midpoint_1d"
            geometry.setdefault("line_order", ["A", "M", "B"])
        elif "extension beyond a" in text or "beyond a" in text:
            geometry["type"] = "collinear"
            geometry["line_order"] = ["M", "A", "B"]
        elif "extension beyond b" in text or "beyond b" in text:
            geometry["type"] = "collinear"
            geometry["line_order"] = ["A", "B", "M"]
        elif "between them" in text or "lies between" in text:
            geometry["type"] = "collinear"
            geometry["line_order"] = ["A", "M", "B"]
        elif all(label in text for label in ("ac", "bc", "ab")):
            geometry["type"] = "triangle_by_sides"

        if geometry.get("type"):
            compact["geometry"] = geometry

        if geometry.get("type") == "perpendicular_bisector":
            apart_match = re.search(
                r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?P<unit>cm|mm|m)\s+apart",
                text,
            )
            side_match = re.search(
                r"\ba\s*=\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?P<unit>cm|mm|m)\b",
                text,
            )
            height_match = re.search(
                r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?P<unit>cm|mm|m)\s+(?:from\s+the\s+midpoint\s+of\s+ab|away\s+from\s+ab|from\s+ab)",
                text,
            )
            base_source = apart_match or side_match
            if base_source:
                self._put_numeric_given(compact, "AB", self._length_from_match(base_source.group("value"), base_source.group("unit")), "m")
            if height_match:
                self._put_numeric_given(compact, "ell", self._length_from_match(height_match.group("value"), height_match.group("unit")), "m")

    @staticmethod
    def _apply_requested_form(compact: dict[str, Any], question: str) -> None:
        text = question.lower()
        signed_terms = ("direction", "sign", "signed", "polarity", "lenz")
        magnitude_terms = (
            "magnitude",
            "strength",
            "intensity",
            "how large",
            "absolute value",
            "electromotive force",
            "emf",
        )
        if any(term in text for term in signed_terms):
            requested_form = "signed"
        elif any(term in text for term in magnitude_terms):
            requested_form = "magnitude"
        else:
            requested_form = "numeric"
        answer_format = compact.get("answer_format")
        if not isinstance(answer_format, dict):
            answer_format = {}
        answer_format.setdefault("requested_form", requested_form)
        compact["answer_format"] = answer_format
