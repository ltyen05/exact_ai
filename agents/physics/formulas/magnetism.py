"""Magnetism and inductance formula rule pack."""

from __future__ import annotations

from typing import Any

from .base import RuleFn
from .legacy import _faraday_induction_solution, _inductance_solution, _magnetism_solution


def magnetism_solution(semantic_output: dict[str, Any], values: dict[str, float], text: str) -> dict[str, Any] | None:
    return _magnetism_solution(semantic_output, values, text)


def inductance_solution(semantic_output: dict[str, Any], values: dict[str, float], text: str) -> dict[str, Any] | None:
    return _inductance_solution(semantic_output, values, text)


def faraday_induction_solution(
    semantic_output: dict[str, Any],
    values: dict[str, float],
    text: str,
) -> dict[str, Any] | None:
    return _faraday_induction_solution(semantic_output, values, text)


RULES: tuple[RuleFn, ...] = (
    magnetism_solution,
    inductance_solution,
    faraday_induction_solution,
)
