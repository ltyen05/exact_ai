"""Base interfaces for deterministic formula rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class FormulaRuleResult:
    formula_id: str
    priority: int
    solution: dict[str, Any]


class FormulaRule(Protocol):
    id: str
    priority: int

    def match(self, semantic_output: dict[str, Any], values: dict[str, float], text: str) -> bool:
        ...

    def build(self, semantic_output: dict[str, Any], values: dict[str, float], text: str) -> dict[str, Any]:
        ...


RuleFn = Callable[[dict[str, Any], dict[str, float], str], dict[str, Any] | None]
