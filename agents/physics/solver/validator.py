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

    @staticmethod
    def _validate_answer_type_matches_question(
        parsed_question: dict[str, object],
        solution_output: dict[str, object],
    ) -> None:
        question_kind = str(parsed_question.get("question_kind") or "").lower()
        answer_format = parsed_question.get("answer_format") or {}
        requested_form = (
            str(answer_format.get("requested_form") or "").lower()
            if isinstance(answer_format, dict)
            else ""
        )
        answer_type = str(solution_output.get("answer_type") or "").lower()
        if answer_type == "yes_no" and question_kind != "yes_no_computational" and requested_form != "yes_no":
            raise ValueError(
                "Solution answer_type yes_no is inconsistent with a numeric parsed question; "
                "return a numeric computational solution for the requested target."
            )

    def validate(
        self,
        parsed_question: dict[str, object],
        solution_output: dict[str, object],
    ) -> ValidatedSolveContext:
        self._validate_answer_type_matches_question(parsed_question, solution_output)
        quantities, target, unit, equations = validated_context(parsed_question, solution_output)
        target = select_effective_target(parsed_question, equations, target)
        return ValidatedSolveContext(
            quantities=quantities,
            target=target,
            unit=unit,
            equations=equations,
        )
