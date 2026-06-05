"""Physics semantic parsing agent."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agents.formatting import extract_json
from agents.llm import LLMClientBase
from agents.physics.domain.symbols import _normalize_text
from agents.physics.domain.units import UNIT_TO_SI

from .heuristics import heuristic_parse
from .normalizers import ParsingOutputNormalizer
from .prompts import (
    ParserPromptBuilder,
    build_example_repair_prompt,
    build_repair_prompt,
    build_retry_prompt,
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

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


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
        self.prompt_builder = ParserPromptBuilder(self.prompt_path)
        self.normalizer = ParsingOutputNormalizer()

    @property
    def prompt_template(self) -> str:
        """Return the cached semantic-parser prompt template."""
        return self.prompt_builder.prompt_template

    def _load_prompt(self) -> str:
        """Return the cached semantic-parser prompt template."""
        return self.prompt_template

    def _build_prompt(self, question: str) -> str:
        """Insert the input question into the semantic-parser prompt."""
        return self.prompt_builder.build_prompt(question)

    _build_repair_prompt = staticmethod(build_repair_prompt)
    _build_retry_prompt = staticmethod(build_retry_prompt)
    _build_example_repair_prompt = staticmethod(build_example_repair_prompt)
    _heuristic_parse = staticmethod(heuristic_parse)

    def _compact_output(self, parsed: dict[str, Any], question: str) -> dict[str, Any]:
        """Enforce the compact internal parser contract and drop empty sections."""
        return self.normalizer.compact_output(parsed, question)

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
        rule_based = self._heuristic_parse(question)
        if rule_based is not None:
            return rule_based
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
                [
                    {
                        "role": "user",
                        "content": self._build_retry_prompt(question, response_preview),
                    }
                ],
                temperature=0.0,
                max_tokens=self.config.get(
                    "retry_max_tokens",
                    self.config.get("repair_max_tokens", 2048),
                ),
                response_format={"type": "json_object"},
                stage="physics.parsing.retry",
            )
            parsed = extract_json(retry_response)

        if not isinstance(parsed, dict):
            repair_response = self.llm_provider.chat(
                [{"role": "user", "content": self._build_repair_prompt(question, response)}],
                temperature=0.0,
                max_tokens=self.config.get("repair_max_tokens", 2048),
                response_format={"type": "json_object"},
                stage="physics.parsing.repair",
            )
            parsed = extract_json(repair_response)

        if not isinstance(parsed, dict):
            repair_response_2 = self.llm_provider.chat(
                [
                    {
                        "role": "user",
                        "content": self._build_example_repair_prompt(question, response),
                    }
                ],
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
