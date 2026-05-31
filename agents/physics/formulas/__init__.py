"""Deterministic physics formula registry."""

from __future__ import annotations

from typing import Any


def deterministic_solution(semantic_output: dict[str, Any]) -> dict[str, Any] | None:
    from .registry import deterministic_solution as _deterministic_solution

    return _deterministic_solution(semantic_output)

__all__ = ["deterministic_solution"]
