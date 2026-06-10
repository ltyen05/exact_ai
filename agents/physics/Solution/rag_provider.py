"""Retrieval-augmented provider for SymPy-compatible physics formulas."""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from agents.llm import LLMClientBase

from .llm_provider import LLMSolutionProvider


class RAGSolutionProvider(LLMSolutionProvider):
    """Generate formulas after retrieving similar solved physics examples."""

    DEFAULT_KB_PATH = Path(__file__).resolve().parents[3] / "data" / "Physics_Problems_Text_Only_removeQA.json"

    def __init__(
        self,
        llm_provider: LLMClientBase,
        kb_path: str | Path | None = None,
        top_k: int = 2,
    ) -> None:
        """Configure the LLM, knowledge-base file, and retrieval result count."""
        super().__init__(llm_provider)
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        self.kb_path = Path(kb_path) if kb_path else self.DEFAULT_KB_PATH
        self.top_k = top_k
        self._documents: list[dict[str, Any]] | None = None
        self._normalized_documents: list[tuple[dict[str, Any], str]] | None = None

    def _load_documents(self) -> list[dict[str, Any]]:
        """Load usable physics examples from the configured JSON knowledge base."""
        if self._documents is not None:
            return self._documents
        try:
            with self.kb_path.open("r", encoding="utf-8") as source:
                data = json.load(source)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not load physics knowledge base: {self.kb_path}") from exc
        if not isinstance(data, list):
            raise ValueError("Physics knowledge base must contain a JSON list.")
        self._documents = [
            item
            for item in data
            if isinstance(item, dict) and isinstance(item.get("question"), str)
        ]
        self._normalized_documents = [
            (item, item["question"].lower())
            for item in self._documents
        ]
        return self._documents

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        """Return the top-k examples with the closest question text."""
        normalized_question = question.lower()
        self._load_documents()
        documents = self._normalized_documents or []
        ranked = sorted(
            documents,
            key=lambda item_and_text: SequenceMatcher(
                None,
                normalized_question,
                item_and_text[1],
            ).ratio(),
            reverse=True,
        )[: self.top_k]
        return [item for item, _ in ranked]

    def _compact_example(self, item: dict[str, Any]) -> dict[str, Any]:
        """Convert a retrieved solved problem into a small formula/strategy hint."""
        return self._compact_rag_example(item)

    def _deterministic_solution(self, semantic_output: dict[str, Any]) -> dict[str, Any] | None:
        """Use shared deterministic formulas before falling back to retrieved prompts."""
        return super()._deterministic_solution(semantic_output)

    def get_solution(self, question: str, semantic_output: dict[str, Any]) -> dict[str, Any]:
        """Request a solution with parsed data, deterministic hints, retrieved examples, and SymPy conversion."""
        deterministic = self._deterministic_solution(semantic_output)
        examples = [self._compact_example(item) for item in self.retrieve(question)]
        prompt = self._build_prompt(
            semantic_output,
            retrieved_examples=examples,
            deterministic_solution=deterministic,
        )
        solution_draft = self._request_solution(
            prompt,
            semantic_output,
            deterministic_fallback=deterministic,
        )
        return self._convert_to_sympy(semantic_output, solution_draft)
