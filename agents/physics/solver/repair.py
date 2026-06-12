"""Repair controller for computational physics solution specs."""

from __future__ import annotations

import logging
from typing import Any

from agents.physics.Solution.formula_lib import build_solution_cot_steps, deterministic_solution
from agents.physics.formulas.geometry import ElectrostaticVectorEngine, GeometryFailure
from agents.workflows.state import WorkflowExecutionError, WorkflowState

from .answer_builder import PhysicsAnswerBuilder
from .direct_answer import DirectAnswerHandler
from .executor import SympyExecutor
from .validator import PhysicsSolutionValidator, ValidatedSolveContext

logger = logging.getLogger(__name__)


def _llm_available(llm: Any) -> bool:
    return llm is not None and bool(getattr(llm, "enabled", True))


def _should_replan(validation_error: str) -> bool:
    text = validation_error.lower()
    return any(
        phrase in text
        for phrase in (
            "undefined symbols",
            "unresolved symbols",
            "could not resolve the target",
        )
    )


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
        if _should_replan(validation_error):
            fallback = deterministic_solution(state.get("parsed_question", {}))
            if isinstance(fallback, dict) and fallback.get("mode") == "computational":
                try:
                    self.validator.validate(state.get("parsed_question", {}), fallback)
                    logger.debug("physics.repair_used_deterministic_replan=%s", fallback.get("formula_ids"))
                    return fallback
                except ValueError as exc:
                    logger.debug("physics.deterministic_replan_rejected=%s", exc)
        if self.solution_agent is not None and _llm_available(self.llm):
            try:
                logger.debug("physics.repair_using_llm error=%s", validation_error)
                repaired = self.solution_agent.repair(
                    semantic_output=state.get("parsed_question", {}),
                    invalid_solution=solution_output,
                    validation_error=validation_error,
                )
                return repaired

            except Exception as exc:
                logger.debug("physics.llm_repair_failed=%s", exc)
        raise WorkflowExecutionError(validation_error)


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


class PhysicsRescueSolver:
    """Try last-chance verified computations before surfacing workflow errors."""

    def __init__(
        self,
        validator: PhysicsSolutionValidator | None = None,
        executor: SympyExecutor | None = None,
        answer_builder: PhysicsAnswerBuilder | None = None,
        direct_handler: DirectAnswerHandler | None = None,
        vector_engine: ElectrostaticVectorEngine | None = None,
    ) -> None:
        self.validator = validator or PhysicsSolutionValidator()
        self.executor = executor or SympyExecutor()
        self.answer_builder = answer_builder or PhysicsAnswerBuilder()
        self.direct_handler = direct_handler or DirectAnswerHandler()
        self.vector_engine = vector_engine or ElectrostaticVectorEngine()

    def rescue(self, state: dict[str, Any], reason: str) -> dict[str, Any]:
        """Try deterministic/vector rescue; otherwise raise the original failure."""
        parsed_question = state.get("parsed_question") if isinstance(state.get("parsed_question"), dict) else {}
        raw_question = str(state.get("question") or parsed_question.get("question") or "")

        vector_result = self._try_vector_engine(parsed_question, raw_question)
        if vector_result is not None:
            return vector_result

        deterministic_result = self._try_deterministic(parsed_question)
        if deterministic_result is not None:
            return deterministic_result

        raise WorkflowExecutionError(str(reason or "Physics computation failed."))

    def _try_deterministic(self, parsed_question: dict[str, Any]) -> dict[str, Any] | None:
        if not parsed_question:
            return None
        try:
            fallback_solution = deterministic_solution(parsed_question)
            if not isinstance(fallback_solution, dict):
                return None
            if fallback_solution.get("mode") == "direct":
                return self.direct_handler.handle(parsed_question, fallback_solution)
            if fallback_solution.get("mode") != "computational":
                return None
            context = self.validator.validate(parsed_question, fallback_solution)
            computation = self.executor.solve(context)
            if computation is None:
                return None
            answer_type = str(fallback_solution.get("answer_type") or "")
            steps = build_solution_cot_steps(
                [str(step) for step in fallback_solution.get("solution_steps") or []],
                context.equations,
                context.target,
            )
            return self.answer_builder.build(
                parsed_question=parsed_question,
                solution_output=fallback_solution,
                context=context,
                computation=computation,
                answer_type=answer_type,
                steps=steps,
            )
        except Exception as exc:  # pragma: no cover - logged to avoid hiding the original failure.
            logger.debug("physics.deterministic_rescue_failed=%s", exc)
            return None

    def _try_vector_engine(self, parsed_question: dict[str, Any], raw_question: str) -> dict[str, Any] | None:
        try:
            result = self.vector_engine.solve(parsed_question, raw_question)
            if isinstance(result, GeometryFailure) or result is None:
                return None
            return result
        except Exception as exc:  # pragma: no cover - rescue path must not create a new runtime failure.
            logger.debug("physics.vector_rescue_failed=%s", exc)
            return None
