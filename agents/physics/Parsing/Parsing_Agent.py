"""Physics semantic parsing agent."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from agents.formatting import extract_json
from agents.llm import LLMClientBase

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

SYMBOL_REPLACEMENTS = {
    "\\u03bc": "u",
    "\\u00b5": "u",
    "\\u00d7": "x",
    "\\u2212": "-",
    "\\u207b": "-",
    "\\times": "x",
    "\\cdot": "x",
    "Âµ": "u",
    "μ": "u",
    "µ": "u",
    "×": "x",
    "·": "x",
    "−": "-",
    "⁻": "-",
    "¹": "1",
    "²": "2",
    "³": "3",
    "ℓ": "ell",
    "φ": "phi",
    "Φ": "Phi",
    "θ": "theta",
    "ω": "omega",
    "Ω": "Ohm",
    "μ": "mu",
    "−": "-",
    "–": "-",
}
UNIT_TO_SI: dict[str, tuple[float, str]] = {
    "m": (1.0, "m"),
    "m2": (1.0, "m2"),
    "m^2": (1.0, "m2"),
    "m²": (1.0, "m2"),
    "cm2": (1e-4, "m2"),
    "cm^2": (1e-4, "m2"),
    "cm²": (1e-4, "m2"),
    "mm2": (1e-6, "m2"),
    "mm^2": (1e-6, "m2"),
    "mm²": (1e-6, "m2"),
    "cm": (1e-2, "m"),
    "mm": (1e-3, "m"),
    "km": (1e3, "m"),
    "pc": (1e-12, "C"),
    "microc": (1e-6, "C"),
    "muc": (1e-6, "C"),
    "μc": (1e-6, "C"),
    "µc": (1e-6, "C"),
    "uc": (1e-6, "C"),
    "nc": (1e-9, "C"),
    "mc": (1e-3, "C"),
    "c": (1.0, "C"),
    "pf": (1e-12, "F"),
    "microf": (1e-6, "F"),
    "muf": (1e-6, "F"),
    "μf": (1e-6, "F"),
    "µf": (1e-6, "F"),
    "uf": (1e-6, "F"),
    "nf": (1e-9, "F"),
    "mf": (1e-3, "F"),
    "f": (1.0, "F"),
    "kv": (1e3, "V"),
    "mv": (1e-3, "V"),
    "kohm": (1e3, "Ohm"),
    "kω": (1e3, "Ohm"),
    "ma": (1e-3, "A"),
    "microa": (1e-6, "A"),
    "mua": (1e-6, "A"),
    "μa": (1e-6, "A"),
    "µa": (1e-6, "A"),
    "ua": (1e-6, "A"),
    "a": (1.0, "A"),
    "v": (1.0, "V"),
    "mh": (1e-3, "H"),
    "uh": (1e-6, "H"),
    "h": (1.0, "H"),
    "khz": (1e3, "Hz"),
    "mhz": (1e6, "Hz"),
    "hz": (1.0, "Hz"),
    "kn": (1e3, "N"),
    "mn": (1e-3, "N"),
    "un": (1e-6, "N"),
    "n": (1.0, "N"),
    "deg": (1.0, "degree"),
    "degree": (1.0, "degree"),
    "degrees": (1.0, "degree"),
    "ohm": (1.0, "Ohm"),
    "ω": (1.0, "Ohm"),
}
UNIT_PATTERN = "|".join(
    re.escape(unit)
    for unit in sorted(UNIT_TO_SI, key=len, reverse=True)
)
NUMBER_PATTERN = (
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:(?:[eE][+-]?\d+)|(?:\s*(?:x|\*)\s*10\s*(?:\^|\*\*)\s*\{?\s*[+-]?\s*\d+\s*\}?))?"
)


def _parse_number(value: str) -> float:
    cleaned = _normalize_text(value).strip().replace("{", "").replace("}", "")
    compact = re.sub(r"\s+", "", cleaned)
    scientific = re.fullmatch(
        r"(?P<coeff>[+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:x|\*)10(?:\^|\*\*)?(?P<exp>[+-]?\d+)",
        compact,
        flags=re.IGNORECASE,
    )
    if scientific:
        return float(scientific.group("coeff")) * (10 ** int(scientific.group("exp")))
    return float(compact)


def _normalize_unit(unit: str) -> str:
    return _normalize_text(unit).strip().lower().replace(" ", "")


def _convert_to_si(value: float, unit: str) -> tuple[float, str] | None:
    conversion = UNIT_TO_SI.get(_normalize_unit(unit))
    if conversion is None:
        return None
    scale, si_unit = conversion
    return value * scale, si_unit


def _given(symbol: str, value: float, unit: str, si_value: float, si_unit: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "value": value,
        "unit": unit,
        "si_value": si_value,
        "si_unit": si_unit,
    }


def _unit_options(units: tuple[str, ...]) -> str:
    return "|".join(re.escape(unit) for unit in sorted(units, key=len, reverse=True))


def _normalize_text(value: str) -> str:
    normalized = value
    for source, replacement in SYMBOL_REPLACEMENTS.items():
        normalized = normalized.replace(source, replacement)
    return normalized


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(_normalize_value(key)): _normalize_value(item) for key, item in value.items()}
    return value


class ParsingAgent:
    """Parse physics questions into structured semantic JSON."""

    DEFAULT_PROMPT_PATH = _PROJECT_ROOT / "prompts" / "semantic_parser_type2.md"

    def __init__(
        self,
        prompt_path: str | None = None,
        llm_provider: LLMClientBase | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Configure the semantic-parser prompt and its LLM provider."""
        self.config = config or {}
        self.prompt_path = Path(prompt_path) if prompt_path else self.DEFAULT_PROMPT_PATH
        self.llm_provider = llm_provider
        self.prompt_template = self.prompt_path.read_text(encoding="utf-8")

    def _load_prompt(self) -> str:
        """Return the cached semantic-parser prompt template."""
        return self.prompt_template

    def _build_prompt(self, question: str) -> str:
        """Insert the input question into the semantic-parser prompt."""
        prompt_template = self._load_prompt()
        if "{{QUESTION}}" in prompt_template:
            return prompt_template.replace("{{QUESTION}}", question)
        return f"{prompt_template.rstrip()}\n\nInput:\n{question}\n\nOutput:"

    @staticmethod
    def _build_repair_prompt(question: str, response: str) -> str:
        """Ask the LLM to convert a malformed parser response into schema JSON."""
        return (
            "Convert the physics parser response into exactly one valid JSON object. "
            "Return JSON only, with no markdown or commentary. "
            "Required fields: question, domain, target, givens, relations, question_kind. "
            "Use ASCII symbol names only: ell, phi, theta, omega, mu. "
            "Use question_kind computational, yes_no_computational, yes_no_conceptual, "
            "multiple_choice, or conceptual. Do not solve the problem.\n\n"
            f"Question:\n{question}\n\nMalformed response:\n{response}\n\nJSON:"
        )

    @staticmethod
    def _build_retry_prompt(question: str, response_preview: str) -> str:
        """Ask the LLM for a fresh compact JSON parse after malformed output."""
        return (
            "Return exactly one complete valid JSON object parsing this physics question. "
            "No markdown, no prose, no calculation. Required fields: question, domain, target, "
            "givens, relations, question_kind. Use ASCII symbols such as mu_0, ell, omega; "
            "convert stated numeric quantities to SI floats where possible.\n\n"
            f"Question:\n{question}\n\n"
            f"Previous invalid response preview:\n{response_preview}\n\n"
            "JSON:"
        )

    @staticmethod
    def _make_given(symbol: str, value_text: str, unit_text: str) -> dict[str, Any] | None:
        try:
            value = _parse_number(value_text)
        except ValueError:
            return None
        converted = _convert_to_si(value, unit_text)
        if converted is None:
            return None
        si_value, si_unit = converted
        return _given(symbol, value, _normalize_unit(unit_text), si_value, si_unit)

    @classmethod
    def _find_symbol_quantity(
        cls,
        text: str,
        canonical_symbol: str,
        aliases: tuple[str, ...],
        units: tuple[str, ...],
    ) -> dict[str, Any] | None:
        alias_pattern = "|".join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True))
        unit_pattern = _unit_options(units)
        pattern = re.compile(
            rf"\b(?:{alias_pattern})\s*=\s*(?P<value>{NUMBER_PATTERN})\s*"
            rf"(?P<unit>{unit_pattern})(?=\b|[^A-Za-z0-9_])",
            flags=re.IGNORECASE,
        )
        match = pattern.search(text)
        if match is None:
            return None
        return cls._make_given(canonical_symbol, match.group("value"), match.group("unit"))

    @classmethod
    def _find_phrase_quantity(
        cls,
        text: str,
        canonical_symbol: str,
        phrases: tuple[str, ...],
        units: tuple[str, ...],
    ) -> dict[str, Any] | None:
        unit_pattern = _unit_options(units)
        for phrase in phrases:
            pattern = re.compile(
                rf"{phrase}\s*(?:of|is|=|:)?\s*(?:a\s*)?(?P<value>{NUMBER_PATTERN})\s*"
                rf"(?P<unit>{unit_pattern})(?=\b|[^A-Za-z0-9_])",
                flags=re.IGNORECASE,
            )
            match = pattern.search(text)
            if match is not None:
                return cls._make_given(canonical_symbol, match.group("value"), match.group("unit"))
        return None

    @staticmethod
    def _quantity_value(given: dict[str, Any] | None) -> float | None:
        if not isinstance(given, dict):
            return None
        value = given.get("si_value")
        return float(value) if isinstance(value, (int, float)) else None

    def _parse_capacitor_rule_based(self, question: str) -> dict[str, Any] | None:
        normalized = _normalize_text(question)
        text = normalized.lower()
        capacitor_hints = (
            "capacitor",
            "capacitance",
            "parallel plate",
            "dielectric",
            "tụ",
            "điện dung",
            "năng lượng",
        )
        if not any(hint in text for hint in capacitor_hints) and not re.search(r"\bc\s*=", text):
            return None

        capacitance_units = ("microf", "muf", "uf", "nf", "pf", "mf", "f")
        charge_units = ("microc", "muc", "uc", "nc", "pc", "mc", "c")
        voltage_units = ("kv", "mv", "v")
        area_units = ("m2", "cm2", "mm2")
        distance_units = ("km", "cm", "mm", "m")

        c_given = self._find_symbol_quantity(normalized, "C", ("C",), capacitance_units) or self._find_phrase_quantity(
            normalized,
            "C",
            (
                r"\bcapacitance(?:\s+of)?",
                r"\bhas\s+a\s+capacitance\s+of",
                r"\bđiện\s+dung",
            ),
            capacitance_units,
        )
        q_given = self._find_symbol_quantity(normalized, "Q", ("Q", "q"), charge_units)
        u_given = self._find_symbol_quantity(normalized, "U", ("U", "V"), voltage_units) or self._find_phrase_quantity(
            normalized,
            "U",
            (
                r"\bpotential\s+difference",
                r"\bvoltage",
                r"\bcharged\s+to",
                r"\bhiệu\s+điện\s+thế",
                r"\bđiện\s+áp",
            ),
            voltage_units,
        )
        c1_given = self._find_symbol_quantity(normalized, "C1", ("C1", "C_1"), capacitance_units)
        c2_given = self._find_symbol_quantity(normalized, "C2", ("C2", "C_2"), capacitance_units)
        s_given = self._find_symbol_quantity(normalized, "S", ("S", "A"), area_units) or self._find_phrase_quantity(
            normalized,
            "S",
            (r"\bplate\s+area", r"\barea", r"\bdiện\s+tích"),
            area_units,
        )
        d_given = self._find_symbol_quantity(normalized, "d", ("d",), distance_units) or self._find_phrase_quantity(
            normalized,
            "d",
            (r"\bplate\s+separation", r"\bseparation", r"\bseparated\s+by", r"\bkhoảng\s+cách", r"\bcách\s+nhau"),
            distance_units,
        )

        givens = [item for item in (c_given, q_given, u_given, c1_given, c2_given, s_given, d_given) if item]
        if not givens:
            return None

        target_symbol = ""
        target_unit = ""
        relations: list[str] = []
        if c1_given and c2_given and ("series" in text or "nối tiếp" in text):
            target_symbol, target_unit = "C", "F"
            relations.append("capacitors:series")
        elif c1_given and c2_given and ("parallel" in text or "song song" in text):
            target_symbol, target_unit = "C", "F"
            relations.append("capacitors:parallel")
        elif "energy" in text or "stored" in text or "năng lượng" in text:
            target_symbol, target_unit = "W", "J"
            relations.append("capacitor:energy")
        elif ("charge" in text or "điện tích" in text) and c_given and u_given:
            target_symbol, target_unit = "Q", "C"
            relations.append("capacitor:charge_from_capacitance_voltage")
        elif ("capacitance" in text or "điện dung" in text or "parallel plate" in text) and (q_given or s_given):
            target_symbol, target_unit = "C", "F"
            relations.append("capacitor:capacitance")
        elif ("potential" in text or "voltage" in text or "hiệu điện thế" in text or "điện áp" in text) and q_given and c_given:
            target_symbol, target_unit = "U", "V"
            relations.append("capacitor:voltage_from_charge_capacitance")

        enough = (
            (target_symbol == "W" and c_given and u_given)
            or (target_symbol == "Q" and c_given and u_given)
            or (target_symbol == "C" and ((q_given and u_given) or (s_given and d_given) or (c1_given and c2_given)))
            or (target_symbol == "U" and q_given and c_given)
        )
        if not enough:
            return None

        return {
            "question": question,
            "domain": "Capacitance",
            "target": {"symbol": target_symbol, "unit": target_unit},
            "givens": givens,
            "relations": relations,
            "question_kind": "computational",
            "answer_format": {"requested_form": "numeric"},
        }

    def _parse_vector_rule_based(self, question: str) -> dict[str, Any] | None:
        normalized = _normalize_text(question)
        text = normalized.lower()
        if "force" not in text and "lực" not in text:
            return None
        vector_hints = (
            "resultant",
            "net force",
            "same direction",
            "opposite",
            "perpendicular",
            "two forces",
            "two electric forces",
            "lực tổng hợp",
            "cùng chiều",
            "ngược chiều",
            "vuông góc",
        )
        if not any(hint in text for hint in vector_hints):
            return None

        force_units = ("kn", "mn", "un", "n")
        f_net_given = self._find_symbol_quantity(
            normalized,
            "F_net",
            ("F_net", "F_resultant", "R"),
            force_units,
        ) or self._find_phrase_quantity(
            normalized,
            "F_net",
            (r"\bresultant(?:\s+force)?", r"\bnet\s+force", r"\blực\s+tổng\s+hợp"),
            force_units,
        )
        force_values = [
            self._make_given("F", match.group("value"), match.group("unit"))
            for match in re.finditer(
                rf"(?P<value>{NUMBER_PATTERN})\s*(?P<unit>{_unit_options(force_units)})(?=\b|[^A-Za-z0-9_])",
                normalized,
                flags=re.IGNORECASE,
            )
        ]
        force_values = [item for item in force_values if item]
        solve_angle = bool(re.search(r"\bfind\s+(?:the\s+)?angle\b|tìm\s+góc", text)) and f_net_given is not None
        if solve_angle:
            if len(force_values) < 3 and not (
                self._find_symbol_quantity(normalized, "F1", ("F1",), force_units)
                and self._find_symbol_quantity(normalized, "F2", ("F2",), force_units)
            ):
                return None
        elif len(force_values) >= 2:
            pass
        elif len(force_values) == 1 and ("each" in text or "both" in text or "mỗi" in text):
            force_values = [force_values[0], dict(force_values[0])]
        else:
            return None

        f1_given = self._find_symbol_quantity(normalized, "F1", ("F1",), force_units)
        f2_given = self._find_symbol_quantity(normalized, "F2", ("F2",), force_units)
        if f1_given is None and force_values:
            f1_given = dict(force_values[0], symbol="F1")
        if f2_given is None and len(force_values) > 1:
            f2_given = dict(force_values[1], symbol="F2")
        if f1_given is None or f2_given is None:
            return None

        angle_match = re.search(rf"(?P<value>{NUMBER_PATTERN})\s*(?P<unit>°|deg|degree|degrees|độ)\b", normalized, re.IGNORECASE)
        alpha_given = self._make_given("alpha", angle_match.group("value"), "deg") if angle_match else None

        if solve_angle:
            if f_net_given is None and len(force_values) >= 3:
                f_net_given = dict(force_values[2], symbol="F_net")
            if f_net_given is None:
                return None
            return {
                "question": question,
                "domain": "Vector Mechanics",
                "target": {"symbol": "alpha", "unit": "degree"},
                "givens": [f1_given, f2_given, f_net_given],
                "relations": ["vector_resultant:solve_angle"],
                "question_kind": "computational",
                "answer_format": {"requested_form": "numeric"},
            }

        if "same direction" in text or "cùng chiều" in text or "cùng hướng" in text:
            mode = "same_direction"
        elif "opposite" in text or "ngược chiều" in text or "ngược hướng" in text:
            mode = "opposite"
        elif "perpendicular" in text or "vuông góc" in text or (alpha_given and math.isclose(float(alpha_given["si_value"]), 90.0)):
            mode = "perpendicular"
        elif alpha_given:
            mode = "angle"
        else:
            return None

        givens = [f1_given, f2_given] + ([alpha_given] if alpha_given else [])
        return {
            "question": question,
            "domain": "Vector Mechanics",
            "target": {"symbol": "F_net", "unit": "N"},
            "givens": givens,
            "relations": [f"vector_resultant:{mode}"],
            "question_kind": "computational",
            "answer_format": {"requested_form": "magnitude"},
        }

    def _parse_coulomb_rule_based(self, question: str) -> dict[str, Any] | None:
        normalized = _normalize_text(question)
        text = normalized.lower()
        if not any(hint in text for hint in ("charge", "coulomb", "electric field", "điện tích", "điện trường")):
            return None

        charge_units = ("microc", "muc", "uc", "nc", "pc", "mc", "c")
        force_units = ("kn", "mn", "un", "n")
        distance_units = ("km", "cm", "mm", "m")
        q1_given = self._find_symbol_quantity(normalized, "q1", ("q1", "q_1"), charge_units)
        q2_given = self._find_symbol_quantity(normalized, "q2", ("q2", "q_2"), charge_units)
        q_given = self._find_symbol_quantity(normalized, "q", ("q",), charge_units)
        r_given = self._find_symbol_quantity(normalized, "r", ("r", "d"), distance_units) or self._find_phrase_quantity(
            normalized,
            "r",
            (r"\bseparated\s+by", r"\bdistance\s+of", r"\bapart", r"\bkhoảng\s+cách", r"\bcách\s+nhau"),
            distance_units,
        )
        f_given = self._find_symbol_quantity(normalized, "F", ("F",), force_units) or self._find_phrase_quantity(
            normalized,
            "F",
            (r"\bforce", r"\blực"),
            force_units,
        )

        if q1_given and q2_given and r_given and ("force" in text or "lực" in text):
            return {
                "question": question,
                "domain": "Electric Charges and Fields",
                "target": {"symbol": "F", "unit": "N"},
                "givens": [q1_given, q2_given, r_given],
                "relations": ["coulomb:pair_force"],
                "question_kind": "computational",
                "answer_format": {"requested_form": "magnitude"},
            }

        if f_given and r_given and re.search(r"\bfind\s+(?:the\s+)?(?:charge\s+)?q\b|tìm\s+điện\s+tích\s+q", text):
            return {
                "question": question,
                "domain": "Electric Charges and Fields",
                "target": {"symbol": "q", "unit": "C"},
                "givens": [f_given, r_given],
                "relations": ["coulomb:equal_charges_from_force"],
                "question_kind": "computational",
                "answer_format": {"requested_form": "magnitude"},
            }

        if q_given and r_given and ("electric field" in text or "điện trường" in text):
            return {
                "question": question,
                "domain": "Electric Charges and Fields",
                "target": {"symbol": "E", "unit": "N/C"},
                "givens": [q_given, r_given],
                "relations": ["electric_field:point_charge"],
                "question_kind": "computational",
                "answer_format": {"requested_form": "magnitude"},
            }
        return None

    def _rule_based_parse(self, question: str) -> dict[str, Any] | None:
        for parser in (
            self._parse_capacitor_rule_based,
            self._parse_vector_rule_based,
            self._parse_coulomb_rule_based,
        ):
            parsed = parser(question)
            if parsed is not None:
                return self._compact_output(parsed, question)
        return None

    @staticmethod
    def _present(value: Any) -> bool:
        if value in (None, "", False):
            return False
        if isinstance(value, (list, dict)):
            return bool(value)
        return True

    def _compact_output(self, parsed: dict[str, Any], question: str) -> dict[str, Any]:
        """Enforce the compact internal parser contract and drop empty sections."""
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
        self._apply_requested_form(compact, question)
        self._normalize_perpendicular_bisector_geometry(compact)
        return compact

    def _normalize_compact(self, compact: dict[str, Any], question: str) -> dict[str, Any]:
        normalized = {
            key: (_normalize_value(value) if key != "question" else str(value))
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
    def _explicit_si_values(question: str) -> dict[str, tuple[float, str]]:
        question = _normalize_text(question)
        values: dict[str, tuple[float, str]] = {}
        pattern = re.compile(
            rf"\b(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
            rf"(?P<value>{NUMBER_PATTERN})\s*"
            rf"(?P<unit>{UNIT_PATTERN})(?=\b|[^A-Za-z0-9_])",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(question):
            unit_key = _normalize_unit(match.group("unit"))
            conversion = UNIT_TO_SI.get(unit_key)
            if conversion is None:
                continue
            scale, si_unit = conversion
            values[match.group("symbol")] = (_parse_number(match.group("value")) * scale, si_unit)
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
    def _distance_item(symbol: str, expression: str, value: float, unit: str = "m") -> dict[str, Any]:
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
        if "perpendicular bisector" not in text:
            return

        segment_values = self._numeric_by_symbol(geometry.get("segments") or [])
        given_values = self._numeric_by_symbol(compact.get("givens") or [])
        values = {**given_values, **segment_values}
        base = values.get("AB") or values.get("d_AB")
        height = values.get("ell") or values.get("h")
        if base is None or height is None:
            return

        d_mid = base / 2
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
        geometry["direction_convention"] = "x-axis from A to B; y-axis from midpoint of AB toward target point"

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

    @staticmethod
    def _build_example_repair_prompt(question: str, response: str) -> str:
        """Provide a concrete JSON example to guide the LLM repair on second attempt."""
        return (
            "The following physics question must be parsed into a JSON object. "
            "The previous attempts failed to produce valid JSON. "
            "Return ONLY a valid JSON object with NO markdown, NO commentary, NO extra text. "
            "Do not wrap in ```json``` blocks.\n\n"
            f"Question:\n{question}\n\n"
            "Required JSON schema (fill in the values, do not solve the problem):\n"
            '{"question": "...", "domain": "...", "target": {"symbol": "...", "unit": "..."}, '
            '"givens": [{"symbol": "...", "value": ..., "si_value": ..., "si_unit": "..."}], '
            '"relations": ["..."], "question_kind": "computational"}\n\n'
            f"Previous malformed response (for reference only):\n{response[:500]}\n\n"
            "Valid JSON:"
        )

    @staticmethod
    def _heuristic_parse(question: str) -> dict[str, Any] | None:
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
                        {"symbol": "I", "si_value": float(current_match.group(1)), "si_unit": "A", "uncertainty": None},
                        {"symbol": "n", "si_value": float(turns_match.group(1)), "si_unit": "1/m", "uncertainty": None},
                    ],
                    "relations": ["long solenoid"],
                    "question_kind": "computational",
                }
        return None

    def run(self, input_data: Any) -> dict[str, Any]:
        """Return semantic JSON extracted from one physics question."""
        question = str(input_data)
        rule_based = self._rule_based_parse(question)
        if rule_based is not None:
            return rule_based
        if self.llm_provider is None:
            raise ValueError("llm_provider is required for physics parsing.")
        response = self.llm_provider.chat(
            [{"role": "user", "content": self._build_prompt(question)}],
            temperature=0.0,
            max_tokens=self.config.get("max_tokens", 2048),
            response_format={"type": "json_object"},
            stage="physics.parsing",
        )
        parsed = extract_json(response)
        if not isinstance(parsed, dict):
            response_preview = response[:200] if response else "(empty)"
            retry_response = self.llm_provider.chat(
                [{"role": "user", "content": self._build_retry_prompt(question, response_preview)}],
                temperature=0.0,
                max_tokens=self.config.get("retry_max_tokens", self.config.get("repair_max_tokens", 2048)),
                response_format={"type": "json_object"},
                stage="physics.parsing.retry",
            )
            parsed = extract_json(retry_response)
        if not isinstance(parsed, dict):
            # Repair attempt 1: standard repair prompt
            repair_response = self.llm_provider.chat(
                [{"role": "user", "content": self._build_repair_prompt(question, response)}],
                temperature=0.0,
                max_tokens=self.config.get("repair_max_tokens", 2048),
                response_format={"type": "json_object"},
                stage="physics.parsing.repair",
            )
            parsed = extract_json(repair_response)
        if not isinstance(parsed, dict):
            # Repair attempt 2: stricter example-based prompt
            repair_response_2 = self.llm_provider.chat(
                [{"role": "user", "content": self._build_example_repair_prompt(question, response)}],
                temperature=0.0,
                max_tokens=self.config.get("repair_max_tokens", 2048),
                response_format={"type": "json_object"},
                stage="physics.parsing.repair2",
            )
            parsed = extract_json(repair_response_2)
        if not isinstance(parsed, dict):
            heuristic = self._heuristic_parse(question)
            if isinstance(heuristic, dict):
                return self._compact_output(heuristic, question)
            response_preview = response[:200] if response else "(empty)"
            raise ValueError(
                f"Physics parser response must be a JSON object. "
                f"Raw response preview: {response_preview}"
            )
        return self._compact_output(parsed, question)
