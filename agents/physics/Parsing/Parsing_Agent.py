"""Physics semantic parsing agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.formatting import extract_json
from agents.llm import LLMClientBase

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class ParsingAgent:
    """Ask the LLM to parse physics questions into structured JSON."""

    DEFAULT_PROMPT_PATH = _PROJECT_ROOT / "prompts" / "physics" / "semantic_parser_type2.md"
    OPTIONAL_FIELDS = ("geometry", "comparison", "options", "answer_format", "warnings")

    def __init__(
        self,
        prompt_path: str | None = None,
        llm_provider: LLMClientBase | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or {}
        self.prompt_path = Path(prompt_path) if prompt_path else self.DEFAULT_PROMPT_PATH
        self.llm_provider = llm_provider
        self.prompt_template = self.prompt_path.read_text(encoding="utf-8")

    def _load_prompt(self) -> str:
        return self.prompt_template

    def _build_prompt(self, question: str) -> str:
        if "{{QUESTION}}" in self.prompt_template:
            return self.prompt_template.replace("{{QUESTION}}", question)
        return f"{self.prompt_template.rstrip()}\n\nInput:\n{question}\n\nOutput:"

    @classmethod
    def _minimal_output(cls, parsed: dict[str, Any], question: str) -> dict[str, Any]:
        """Keep the LLM output shape, filling only required missing fields."""
        output = dict(parsed)
        output["question"] = str(output.get("question") or question)
        output["domain"] = str(output.get("domain") or "unknown")
        if not isinstance(output.get("target"), dict):
            output["target"] = {}
        if not isinstance(output.get("givens"), list):
            output["givens"] = []
        if not isinstance(output.get("relations"), list):
            output["relations"] = []
        output["question_kind"] = str(output.get("question_kind") or "computational")
        for key in cls.OPTIONAL_FIELDS:
            value = output.get(key)
            if value in (None, "", False) or (isinstance(value, (list, dict)) and not value):
                output.pop(key, None)
        return output

    def run(self, input_data: Any) -> dict[str, Any]:
        """Return semantic JSON extracted by the LLM without rule-based physics normalization."""
        if self.llm_provider is None:
            raise ValueError("llm_provider is required for physics parsing.")

        question = str(input_data)
        response = self.llm_provider.chat(
            [{"role": "user", "content": self._build_prompt(question)}],
            temperature=0.0,
            max_tokens=self.config.get("max_tokens", 1024),
            response_format={"type": "json_object"},
            stage="physics.parsing",
        )
        parsed = extract_json(response)
        if not isinstance(parsed, dict):
            response_preview = response[:200] if response else "(empty)"
            raise ValueError(
                "Physics parser response must be a JSON object. "
                f"Raw response preview: {response_preview}"
            )
        return self._minimal_output(parsed, question)
