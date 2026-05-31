"""Physics semantic parsing agent."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from agents.formatting import as_number, extract_json
from agents.llm import LLMClientBase

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

SYMBOL_REPLACEMENTS = {
    "ℓ": "ell",
    "φ": "phi",
    "Φ": "Phi",
    "θ": "theta",
    "ω": "omega",
    "Ω": "Ohm",
    "ε": "epsilon_",
    "μ": "mu",
    "µ": "mu",
    "λ": "lambda_",
    "−": "-",
    "–": "-",
}
UNIT_TO_SI: dict[str, tuple[float, str]] = {
    "cm": (1e-2, "m"),
    "cm2": (1e-4, "m2"),
    "cm^2": (1e-4, "m2"),
    "cm²": (1e-4, "m2"),
    "mm": (1e-3, "m"),
    "mm2": (1e-6, "m2"),
    "mm^2": (1e-6, "m2"),
    "mm²": (1e-6, "m2"),
    "km": (1e3, "m"),
    "m2": (1.0, "m2"),
    "m^2": (1.0, "m2"),
    "m²": (1.0, "m2"),
    "microc": (1e-6, "C"),
    "muc": (1e-6, "C"),
    "μc": (1e-6, "C"),
    "µc": (1e-6, "C"),
    "uc": (1e-6, "C"),
    "pc": (1e-12, "C"),
    "nc": (1e-9, "C"),
    "mc": (1e-3, "C"),
    "c": (1.0, "C"),
    "microf": (1e-6, "F"),
    "muf": (1e-6, "F"),
    "μf": (1e-6, "F"),
    "µf": (1e-6, "F"),
    "uf": (1e-6, "F"),
    "pf": (1e-12, "F"),
    "nf": (1e-9, "F"),
    "mf": (1e-3, "F"),
    "f": (1.0, "F"),
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
    "mv": (1e-3, "V"),
    "kv": (1e3, "V"),
    "microh": (1e-6, "H"),
    "muh": (1e-6, "H"),
    "μh": (1e-6, "H"),
    "µh": (1e-6, "H"),
    "uh": (1e-6, "H"),
    "mh": (1e-3, "H"),
    "h": (1.0, "H"),
    "hz": (1.0, "Hz"),
    "khz": (1e3, "Hz"),
    "Mhz": (1e6, "Hz"),
    "rad/s": (1.0, "rad/s"),
    "ohm": (1.0, "Ohm"),
    "ω": (1.0, "Ohm"),
    "mohm": (1e-3, "Ohm"),
    "Mohm": (1e6, "Ohm"),
    "j": (1.0, "J"),
    "mj": (1e-3, "J"),
    "μj": (1e-6, "J"),
    "µj": (1e-6, "J"),
    "microj": (1e-6, "J"),
    "uj": (1e-6, "J"),
    "wb": (1.0, "Wb"),
    "mwb": (1e-3, "Wb"),
    "microwb": (1e-6, "Wb"),
    "muwb": (1e-6, "Wb"),
    "μwb": (1e-6, "Wb"),
    "µwb": (1e-6, "Wb"),
    "uwb": (1e-6, "Wb"),
    "nwb": (1e-9, "Wb"),
    "t": (1.0, "T"),
    "n": (1.0, "N"),
    "n/c": (1.0, "N/C"),
    "v/m": (1.0, "V/m"),
    "w": (1.0, "W"),
    "s": (1.0, "s"),
    "ms": (1e-3, "s"),
    "micros": (1e-6, "s"),
    "mus": (1e-6, "s"),
    "μs": (1e-6, "s"),
    "µs": (1e-6, "s"),
    "us": (1e-6, "s"),
    "ml": (1e-6, "m3"),
}
UNIT_PATTERN = "|".join(
    re.escape(unit)
    for unit in sorted(UNIT_TO_SI, key=len, reverse=True)
)
QUANTITY_ALIAS_GROUPS = (
    ("U", "V", "U_rms", "V_rms"),
    ("I", "I_rms", "I_effective"),
    ("f", "frequency"),
    ("XL", "X_L", "ZL", "Z_L"),
    ("XC", "X_C", "ZC", "Z_C"),
    ("lambda_", "lambda", "flux_linkage"),
    ("ell", "l", "length"),
    ("Q_max", "Qmax", "qmax", "q_max", "maximum_charge"),
    ("side", "side_length", "a", "AB", "triangle_side"),
    ("W_C", "electric_energy", "E_elec", "capacitor_energy"),
    ("W_L", "magnetic_energy", "E_magn", "inductor_energy"),
    ("epsilon_r", "er", "eps_r", "relative_permittivity"),
)


def _normalize_text(value: str) -> str:
    normalized = value
    for source, replacement in SYMBOL_REPLACEMENTS.items():
        normalized = normalized.replace(source, replacement)
    return normalized


def _canonical_unit_key(unit: str) -> str:
    """Normalize unit spelling while preserving uppercase mega prefix."""
    raw = str(unit or "").strip()
    normalized = raw.replace("Ω", "Ohm").replace("μ", "u").replace("µ", "u")
    normalized = normalized.replace("^2", "2").replace("²", "2")
    normalized = re.sub(r"\s+", "", normalized)
    if len(normalized) > 1 and normalized[0] == "M" and normalized[1].isalpha():
        return "M" + normalized[1:].lower()
    return normalized.lower()


def canonical_quantity_symbol(symbol: Any) -> str:
    """Map parser/LLM symbol aliases to the internal physics symbol name."""
    name = _normalize_text(str(symbol or "")).strip()
    if name in {"lambda", "flux_linkage"}:
        return "lambda_"
    if name in {"qmax", "q_max", "Qmax", "maximum_charge"}:
        return "Q_max"
    if name in {"electric_energy", "E_elec", "capacitor_energy"}:
        return "W_C"
    if name in {"magnetic_energy", "E_magn", "inductor_energy"}:
        return "W_L"
    if name in {"er", "eps_r", "relative_permittivity"}:
        return "epsilon_r"
    return name


def expand_quantity_aliases(quantities: dict[str, float]) -> dict[str, float]:
    expanded = dict(quantities)
    for group in QUANTITY_ALIAS_GROUPS:
        if any(re.fullmatch(rf"{re.escape(symbol)}_\d+", key) for symbol in group for key in expanded):
            continue
        present = [(symbol, expanded[symbol]) for symbol in group if symbol in expanded]
        if not present:
            continue
        first_value = present[0][1]
        if any(not math.isclose(first_value, value, rel_tol=1e-9, abs_tol=1e-12) for _, value in present[1:]):
            continue
        for alias in group:
            expanded.setdefault(alias, first_value)
    return expanded


def put_quantity(quantities: dict[str, float], symbol: Any, value: Any) -> None:
    name = canonical_quantity_symbol(symbol)
    numeric = as_number(value)
    if not name or numeric is None:
        return
    previous = quantities.get(name)
    if previous is not None and not math.isclose(previous, numeric, rel_tol=1e-9, abs_tol=1e-12):
        index = 2
        while f"{name}_{index}" in quantities:
            index += 1
        quantities[f"{name}_{index}"] = numeric
        return
    quantities[name] = numeric


def build_calculation_input(parsed_question: dict[str, Any]) -> dict[str, Any]:
    """Build trusted numeric context from parsed quantities, geometry, and comparison."""
    quantities: dict[str, float] = {}
    raw_quantities = parsed_question.get("quantities") or {}
    if isinstance(raw_quantities, dict):
        for symbol, value in raw_quantities.items():
            put_quantity(quantities, symbol, value)
    elif isinstance(raw_quantities, list):
        for quantity in raw_quantities:
            if isinstance(quantity, dict):
                value = quantity.get("si_value")
                put_quantity(quantities, quantity.get("symbol"), quantity.get("value") if value is None else value)

    for given in parsed_question.get("givens") or []:
        if not isinstance(given, dict):
            continue
        value = given.get("si_value")
        put_quantity(quantities, given.get("symbol"), given.get("value") if value is None else value)
        uncertainty = given.get("uncertainty") or {}
        if isinstance(uncertainty, dict):
            uncertainty_value = uncertainty.get("si_value")
            if uncertainty_value is None:
                uncertainty_value = uncertainty.get("value")
            symbol = str(given.get("symbol") or "").strip()
            put_quantity(quantities, f"delta_{symbol}", uncertainty_value)
            put_quantity(quantities, "uncertainty", uncertainty_value)
            put_quantity(quantities, "absolute_uncertainty", uncertainty_value)

    geometry = parsed_question.get("geometry") or {}
    if isinstance(geometry, dict):
        for field in ("segments", "derived_distances"):
            for item in geometry.get(field) or []:
                if isinstance(item, dict):
                    value = item.get("si_value")
                    put_quantity(quantities, item.get("symbol"), item.get("value") if value is None else value)

    comparison = parsed_question.get("comparison") or {}
    if isinstance(comparison, dict):
        comparison_value = comparison.get("given_si_value")
        if comparison_value is None:
            comparison_value = comparison.get("given_value")
        comparison_symbol = comparison.get("given_quantity_symbol")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(comparison_symbol or "").strip()):
            comparison_unit = str(comparison.get("given_si_unit") or comparison.get("unit") or "").lower()
            if "hz" in comparison_unit:
                comparison_symbol = "f"
            elif "rad" in comparison_unit:
                comparison_symbol = "omega"
            elif comparison_unit in {"v", "volt", "volts"}:
                comparison_symbol = "U"
        put_quantity(quantities, comparison_symbol, comparison_value)

    raw_target = parsed_question.get("target") or parsed_question.get("objective") or ""
    if isinstance(raw_target, dict):
        target = str(raw_target.get("symbol") or "")
        unit = str(raw_target.get("unit") or "")
    else:
        target = str(raw_target)
        unit = str(parsed_question.get("unit") or parsed_question.get("objective_unit") or "")
    return {"quantities": expand_quantity_aliases(quantities), "target": target, "unit": unit}


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
        self._normalize_resonance_yes_no(compact, question)
        self._normalize_formula_only_question(compact, question)
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
                givens.append({"symbol": "f", "si_value": frequency, "si_unit": "Hz", "uncertainty": None})
                values["f"] = frequency
        if not all(symbol in values for symbol in ("L", "C")) or ("f" not in values and "frequency" not in values):
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
        if self.llm_provider is None:
            raise ValueError("llm_provider is required for physics parsing.")
        question = str(input_data)
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
