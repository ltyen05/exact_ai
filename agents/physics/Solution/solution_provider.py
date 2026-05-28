"""Abstract interface for physics formula solution providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SolutionProvider(ABC):
    """Provide a validated solution specification for a parsed physics question."""

    @abstractmethod
    def get_solution(self, question: str, semantic_output: dict[str, Any]) -> dict[str, Any]:
        """Return a computational or direct solution specification."""
        raise NotImplementedError

    def repair_solution(
        self,
        question: str,
        semantic_output: dict[str, Any],
        invalid_solution: dict[str, Any],
        validation_error: str,
    ) -> dict[str, Any]:
        """Return a repaired solution specification after downstream validation failed."""
        raise NotImplementedError

