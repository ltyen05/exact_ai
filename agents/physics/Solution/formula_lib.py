"""Compatibility exports for deterministic physics formulas."""

from __future__ import annotations

from typing import Any

from agents.physics.formulas.legacy import (
    COMPUTATIONAL_MODE_ALIASES,
    CONCEPTUAL_ANSWER_ALIASES,
    DIRECT_MODE_ALIASES,
    MULTIPLE_CHOICE_ANSWER_ALIASES,
    NUMERIC_ANSWER_ALIASES,
    PHYSICAL_CONSTANT_NAMES,
    SYMBOL_ALIAS_GROUPS,
    SYMBOL_REPLACEMENTS,
    YES_NO_ANSWER_ALIASES,
    build_solution_cot_steps,
)


def deterministic_solution(semantic_output: dict[str, Any]) -> dict[str, Any] | None:
    from agents.physics.formulas.registry import deterministic_solution as _deterministic_solution

    return _deterministic_solution(semantic_output)
