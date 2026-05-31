"""Capacitance formula rule pack."""

from __future__ import annotations

from typing import Any

from .base import RuleFn
from .legacy import _capacitance_solution


def capacitance_solution(semantic_output: dict[str, Any], values: dict[str, float], text: str) -> dict[str, Any] | None:
    return _capacitance_solution(semantic_output, values, text)


RULES: tuple[RuleFn, ...] = (capacitance_solution,)
