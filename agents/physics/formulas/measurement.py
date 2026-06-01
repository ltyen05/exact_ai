"""Measurement and uncertainty formula rule pack."""

from __future__ import annotations

from typing import Any

from .base import RuleFn
from .legacy import _measurement_statistics_solution


def measurement_statistics_solution(
    semantic_output: dict[str, Any],
    values: dict[str, float],
    text: str,
) -> dict[str, Any] | None:
    return _measurement_statistics_solution(semantic_output, values, text)


RULES: tuple[RuleFn, ...] = (measurement_statistics_solution,)
