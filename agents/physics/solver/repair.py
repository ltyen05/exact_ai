"""Repair controller for computational physics solution specs."""

from __future__ import annotations

import logging
from typing import Any

from agents.physics.Solution.formula_lib import build_solution_cot_steps
from agents.workflows.state import WorkflowExecutionError, WorkflowState

from .executor import SympyExecutor
from .validator import PhysicsSolutionValidator, ValidatedSolveContext

logger = logging.getLogger(__name__)


def _llm_available(llm: Any) -> bool:
    return llm is not None and bool(getattr(llm, "enabled", True))


class SolutionRepairController:
    """Prepare and repair computational solution specs around validation/SymPy failures."""

    def __init__(
        self,
        solution_agent: Any = None,
        llm: Any = None,
        validator: PhysicsSolutionValidator | None = None,
        executor: SympyExecutor | None = None,
    ) -> None:
        self.solution_agent = solution_agent
        self.llm = llm
        self.validator = validator or PhysicsSolutionValidator()
        self.executor = executor or SympyExecutor()

    def prepare_computational_solution(
        self,
        state: WorkflowState,
        parsed_question: dict[str, Any],
        solution_output: dict[str, Any],
    ) -> tuple[dict[str, Any], str, ValidatedSolveContext, list[str]]:
        answer_type = self._assert_computational_solution(solution_output)
        steps = [str(step) for step in solution_output.get("solution_steps") or []]
        try:
            context = self.validator.validate(parsed_question, solution_output)
        except ValueError as exc:
            logger.debug("physics.verification_fail_reason=%s", exc)
            solution_output = self._repair_solution_output(
                state,
                solution_output,
                f"Physics computation validation failed: {exc}",
            )
            answer_type = self._assert_computational_solution(solution_output)
            steps = [str(step) for step in solution_output.get("solution_steps") or []]
            context = self._validate_repaired(parsed_question, solution_output)

        steps = build_solution_cot_steps(steps, context.equations, context.target)
        return solution_output, answer_type, context, steps

    def repair_after_sympy_failure(
        self,
        state: WorkflowState,
        parsed_question: dict[str, Any],
        solution_output: dict[str, Any],
    ) -> tuple[dict[str, Any], str, ValidatedSolveContext, list[str], Any]:
        solution_output = self._repair_solution_output(
            state,
            solution_output,
            "Physics computation failed: equations could not resolve the target.",
        )
        answer_type = self._assert_computational_solution(solution_output)
        steps = [str(step) for step in solution_output.get("solution_steps") or []]
        context = self._validate_repaired(parsed_question, solution_output)
        steps = build_solution_cot_steps(steps, context.equations, context.target)
        computation = self.executor.solve(context)
        if computation is None:
            logger.debug("physics.verification_fail_reason=equations could not resolve the target after repair")
            raise WorkflowExecutionError("Physics computation failed after repair: equations could not resolve the target.")
        return solution_output, answer_type, context, steps, computation

    def _repair_solution_output(
        self,
        state: WorkflowState,
        solution_output: dict[str, Any],
        validation_error: str,
    ) -> dict[str, Any]:
        if self.solution_agent is None or not _llm_available(self.llm):
            raise WorkflowExecutionError(validation_error)
        try:
            return self.solution_agent.repair(
                state["question"],
                state.get("parsed_question", {}),
                solution_output,
                validation_error,
            )
        except Exception as exc:
            raise WorkflowExecutionError(f"Physics SolutionAgent failed to repair sympy_spec: {exc}") from exc

    @staticmethod
    def _assert_computational_solution(solution_output: dict[str, Any]) -> str:
        mode = solution_output.get("mode")
        answer_type = solution_output.get("answer_type")
        if mode != "computational":
            raise WorkflowExecutionError("Repaired physics solution must remain computational.")
        if answer_type not in {"numeric", "yes_no"}:
            raise WorkflowExecutionError("Computational physics solution has an unsupported answer_type.")
        return str(answer_type)

    def _validate_repaired(
        self,
        parsed_question: dict[str, Any],
        solution_output: dict[str, Any],
    ) -> ValidatedSolveContext:
        try:
            return self.validator.validate(parsed_question, solution_output)
        except ValueError as repair_exc:
            logger.debug("physics.verification_fail_reason=%s", repair_exc)
            raise WorkflowExecutionError(f"Physics computation validation failed after repair: {repair_exc}") from repair_exc
