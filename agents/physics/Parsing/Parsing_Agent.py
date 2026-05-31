"""Physics semantic parsing agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.formatting import extract_json
from agents.llm import LLMClientBase
from agents.physics.domain.units import UNIT_TO_SI

from .heuristics import heuristic_parse
from .normalizers import ParsingOutputNormalizer
from .prompts import (
    ParserPromptBuilder,
    build_example_repair_prompt,
    build_repair_prompt,
    build_retry_prompt,
)

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
