"""Physics solution agent that obtains formulas for later calculation."""

from __future__ import annotations

from typing import Any

from .solution_provider import SolutionProvider


class SolutionAgent:
    """Delegate structured physics solution selection to a configured provider."""

    def __init__(self, provider: SolutionProvider) -> None:
        """Store the provider used to generate physics formulas."""
        self.provider = provider

    def run(self, question: str, semantic_output: dict[str, Any]) -> dict[str, Any]:
        """Return a validated computational or direct solution specification."""
        return self.provider.get_solution(question, semantic_output)

    def repair(
        self,
        question: str,
        semantic_output: dict[str, Any],
        invalid_solution: dict[str, Any],
        validation_error: str,
    ) -> dict[str, Any]:
        """Return a repaired solution specification using downstream validation feedback."""
        return self.provider.repair_solution(question, semantic_output, invalid_solution, validation_error)
