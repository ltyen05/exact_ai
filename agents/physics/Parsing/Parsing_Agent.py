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
    "cm": (1e-2, "m"),
    "mm": (1e-3, "m"),
    "km": (1e3, "m"),
    "microc": (1e-6, "C"),
    "muc": (1e-6, "C"),
    "μc": (1e-6, "C"),
    "µc": (1e-6, "C"),
    "uc": (1e-6, "C"),
    "nc": (1e-9, "C"),
    "c": (1.0, "C"),
    "microf": (1e-6, "F"),
    "muf": (1e-6, "F"),
    "μf": (1e-6, "F"),
    "µf": (1e-6, "F"),
    "uf": (1e-6, "F"),
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
    "h": (1.0, "H"),
    "hz": (1.0, "Hz"),
    "ohm": (1.0, "Ohm"),
    "ω": (1.0, "Ohm"),
}
UNIT_PATTERN = "|".join(
    re.escape(unit)
    for unit in sorted(UNIT_TO_SI, key=len, reverse=True)
)


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
        values: dict[str, tuple[float, str]] = {}
        pattern = re.compile(
            rf"\b(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
            rf"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*"
            rf"(?P<unit>{UNIT_PATTERN})(?=\b|[^A-Za-z0-9_])",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(question):
            unit_key = match.group("unit").lower()
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
            response_preview = response[:200] if response else "(empty)"
            raise ValueError(
                f"Physics parser response must be a JSON object. "
                f"Raw response preview: {response_preview}"
            )
        return self._compact_output(parsed, question)
