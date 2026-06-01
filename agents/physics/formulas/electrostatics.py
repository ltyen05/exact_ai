"""Electrostatics formula rule pack."""

from __future__ import annotations

from typing import Any

from .base import RuleFn
from .legacy import _electric_solution, _resultant_two_vectors_solution


def resultant_two_vectors_solution(
    semantic_output: dict[str, Any],
    values: dict[str, float],
    text: str,
) -> dict[str, Any] | None:
    return _resultant_two_vectors_solution(semantic_output, values, text)


def electric_solution(semantic_output: dict[str, Any], values: dict[str, float], text: str) -> dict[str, Any] | None:
    del values
    return _electric_solution(semantic_output, text)


RULES: tuple[RuleFn, ...] = (
    resultant_two_vectors_solution,
    electric_solution,
)
