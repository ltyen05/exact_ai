"""LangGraph physics subgraph wired to the structured physics agents."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.physics.Explain import ExplainAgent
from agents.physics.Parsing import ParsingAgent
from agents.physics.solver import (
    DirectAnswerHandler,
    PhysicsAnswerBuilder,
    PhysicsRescueSolver,
    SolutionRepairController,
    SympyExecutor,
)
from agents.physics.Solution import LLMSolutionProvider, RAGSolutionProvider, SolutionAgent
from .state import WorkflowExecutionError, WorkflowState

logger = logging.getLogger(__name__)


def _llm_available(llm: Any) -> bool:
    return llm is not None and bool(getattr(llm, "enabled", True))


def _with_error(state: WorkflowState, error: str) -> list[str]:
    return [*state.get("errors", []), error]


class PhysicsWorkflow:
    """Parse, specify, compute, and explain a physics answer."""

    def __init__(self, llm: Any = None, kb_path: str | None = None) -> None:
        self.llm = llm
        self.kb_path = Path(kb_path) if kb_path else None
        self.parser = ParsingAgent(llm_provider=llm)
        solution_provider = (
            RAGSolutionProvider(llm, kb_path=self.kb_path)
            if llm is not None and self.kb_path
            else LLMSolutionProvider(llm)
            if llm is not None
            else None
        )
        self.solution_agent = SolutionAgent(solution_provider) if solution_provider is not None else None
        self.explain_agent = ExplainAgent(llm_provider=llm)
        self.direct_answer_handler = DirectAnswerHandler()
        self.executor = SympyExecutor()
        self.repair_controller = SolutionRepairController(
            solution_agent=self.solution_agent,
            llm=self.llm,
            executor=self.executor,
        )
        self.answer_builder = PhysicsAnswerBuilder()
        self.rescue_solver = PhysicsRescueSolver(
            validator=self.repair_controller.validator,
            executor=self.executor,
            answer_builder=self.answer_builder,
            direct_handler=self.direct_answer_handler,
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        graph = StateGraph(WorkflowState)
        graph.add_node("parse_question", self.parse_question)
        graph.add_node("select_solution", self.select_solution)
        graph.add_node("compute_sympy", self.compute_sympy)
        graph.add_node("explain_answer", self.explain_answer)
        graph.add_edge(START, "parse_question")
        graph.add_edge("parse_question", "select_solution")
        graph.add_edge("select_solution", "compute_sympy")
        graph.add_edge("compute_sympy", "explain_answer")
        graph.add_edge("explain_answer", END)
        return graph.compile()

    def parse_question(self, state: WorkflowState) -> dict[str, Any]:
        """Parse one public physics question into compact structured input."""
        try:
            parsed_question = self.parser.run(state["question"])
        except Exception as exc:
            if not _llm_available(self.llm) and "llm_provider is required" in str(exc):
                raise WorkflowExecutionError("Physics ParsingAgent requires a configured LLM.") from exc
            raise WorkflowExecutionError(f"Physics ParsingAgent failed: {exc}") from exc
        logger.debug("physics.parsed_question=%s", parsed_question)
        return {"parsed_question": parsed_question}

    def select_solution(self, state: WorkflowState) -> dict[str, Any]:
        """Obtain one structured physics solution specification."""
        parsed_question = state.get("parsed_question", {})
        if self.solution_agent is None or not _llm_available(self.llm):
            try:
                deterministic = LLMSolutionProvider.deterministic_solution(parsed_question)
            except Exception as exc:
                raise WorkflowExecutionError(f"Physics SolutionAgent failed: {exc}") from exc
            if deterministic is not None:
                logger.debug("physics.selected_formula_ids=%s", deterministic.get("formula_ids"))
                logger.debug("physics.generated_solution_spec=%s", deterministic)
                return {"solution_output": deterministic}
            raise WorkflowExecutionError("Physics SolutionAgent requires a configured LLM.")
        try:
            solution_output = self.solution_agent.run(state["question"], parsed_question)
            logger.debug(
                "physics.selected_formula_ids=%s",
                solution_output.get("formula_ids") if isinstance(solution_output, dict) else None,
            )
            logger.debug("physics.generated_solution_spec=%s", solution_output)
            return {"solution_output": solution_output}
        except Exception as exc:
            try:
                deterministic = LLMSolutionProvider.deterministic_solution(parsed_question)
                if deterministic is not None:
                    logger.debug("physics.solution_agent_failed_used_deterministic=%s", deterministic.get("formula_ids"))
                    return {"solution_output": deterministic}
            except Exception as deterministic_exc:
                logger.debug("physics.solution_agent_deterministic_fallback_failed=%s", deterministic_exc)
            raise WorkflowExecutionError(f"Physics SolutionAgent failed: {exc}") from exc

    def compute_sympy(self, state: WorkflowState) -> dict[str, Any]:
        """Execute a structured direct or computational solution."""
        self.repair_controller.solution_agent = self.solution_agent
        self.repair_controller.llm = self.llm
        self.repair_controller.executor = self.executor
        parsed_question = state.get("parsed_question", {})
        solution_output = state.get("solution_output", {})
        mode = solution_output.get("mode")
        try:
            if mode == "direct":
                return self.direct_answer_handler.handle(parsed_question, solution_output)
            if mode != "computational":
                raise WorkflowExecutionError("No valid physics solution specification was produced.")

            solution_output, answer_type, context, steps = self.repair_controller.prepare_computational_solution(
                state=state,
                parsed_question=parsed_question,
                solution_output=solution_output,
            )
            logger.debug("physics.normalized_knowns=%s", context.quantities)
            computation = self.executor.solve(context)
            if computation is None:
                logger.debug("physics.verification_fail_reason=equations could not resolve the target")
                try:
                    solution_output, answer_type, context, steps, computation = self.repair_controller.repair_after_sympy_failure(
                        state=state,
                        parsed_question=parsed_question,
                        solution_output=solution_output,
                    )
                except WorkflowExecutionError as exc:
                    return self.rescue_solver.rescue(state, str(exc))
            return self.answer_builder.build(
                parsed_question=parsed_question,
                solution_output=solution_output,
                context=context,
                computation=computation,
                answer_type=answer_type,
                steps=steps,
            )
        except WorkflowExecutionError as exc:
            return self.rescue_solver.rescue(state, str(exc))
        except ValueError as exc:
            return self.rescue_solver.rescue(state, str(exc))

    def explain_answer(self, state: WorkflowState) -> dict[str, Any]:
        result = dict(state.get("result", {}))
        if result.get("answer") == "Unknown":
            return {"result": result}
        solution_output = state.get("solution_output", {})
        
        # Steps gets the formulas/equations
        equations = solution_output.get("sympy_spec", {}).get("equations", [])
        if not equations:
            # Fallback for direct/conceptual answers
            equations = solution_output.get("direct_answer", {}).get("rationale_steps", [])
            if not equations:
                equations = ["Computed the requested quantity from the selected equations."]
        result["cot"] = [str(eq) for eq in equations]
        
        # Explanation gets the solution_steps text
        solution_steps = solution_output.get("solution_steps", [])
        if not solution_steps:
            solution_steps = solution_output.get("direct_answer", {}).get("rationale_steps", [])
            if not solution_steps:
                solution_steps = ["Computed the requested quantity from the selected equations."]
                
        explanation_text = " ".join(str(step).strip().rstrip(".") + "." for step in solution_steps)
        
        unit = str(result.get("unit") or "")
        suffix = f" {unit}" if result.get("append_unit") and unit and unit.lower() != "dimensionless" else ""
        result["explanation"] = f"{explanation_text} Therefore, the answer is {result.get('answer')}{suffix}."
        return {"result": result, "errors": state.get("errors", [])}
