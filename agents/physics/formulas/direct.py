"""Direct/conceptual formula rule pack."""

from __future__ import annotations

from typing import Any

from .base import RuleFn
from .legacy import _formula_only_solution


def formula_only_solution(semantic_output: dict[str, Any], values: dict[str, float], text: str) -> dict[str, Any] | None:
    return _formula_only_solution(semantic_output, text, values)


RULES: tuple[RuleFn, ...] = (formula_only_solution,)
