"""Validation wrapper for physics solve specs."""

from __future__ import annotations

from dataclasses import dataclass

from agents.physics.validation import select_effective_target, validated_context


@dataclass(frozen=True)
class ValidatedSolveContext:
    quantities: dict[str, float]
    target: str
    unit: str
    equations: list[str]


class PhysicsSolutionValidator:
    """Build a normalized computation context from parser and solution output."""

    def validate(
        self,
        parsed_question: dict[str, object],
        solution_output: dict[str, object],
    ) -> ValidatedSolveContext:
        quantities, target, unit, equations = validated_context(parsed_question, solution_output)
        target = select_effective_target(parsed_question, equations, target)
        return ValidatedSolveContext(
            quantities=quantities,
            target=target,
            unit=unit,
            equations=equations,
        )
