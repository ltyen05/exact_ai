"""AC/DC circuit formula rule pack."""

from __future__ import annotations

from typing import Any

from .base import RuleFn
from .legacy import _ac_solution, _basic_circuit_solution


def ac_solution(semantic_output: dict[str, Any], values: dict[str, float], text: str) -> dict[str, Any] | None:
    return _ac_solution(semantic_output, values, text)


def basic_circuit_solution(semantic_output: dict[str, Any], values: dict[str, float], text: str) -> dict[str, Any] | None:
    return _basic_circuit_solution(semantic_output, values, text)


RULES: tuple[RuleFn, ...] = (
    ac_solution,
    basic_circuit_solution,
)
