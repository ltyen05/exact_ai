"""LangGraph physics subgraph wired to the structured physics agents."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.physics.Explain import ExplainAgent
from agents.physics.Parsing import ParsingAgent
from agents.physics.solver import DirectAnswerHandler, PhysicsAnswerBuilder, SolutionRepairController, SympyExecutor
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
            raise WorkflowExecutionError(f"Physics SolutionAgent failed: {exc}") from exc

    def compute_sympy(self, state: WorkflowState) -> dict[str, Any]:
        """Execute a structured direct or computational solution."""
        self.repair_controller.solution_agent = self.solution_agent
        self.repair_controller.llm = self.llm
        self.repair_controller.executor = self.executor
        parsed_question = state.get("parsed_question", {})
        solution_output = state.get("solution_output", {})
        mode = solution_output.get("mode")
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
            solution_output, answer_type, context, steps, computation = self.repair_controller.repair_after_sympy_failure(
                state=state,
                parsed_question=parsed_question,
                solution_output=solution_output,
            )
        return self.answer_builder.build(
            parsed_question=parsed_question,
            solution_output=solution_output,
            context=context,
            computation=computation,
            answer_type=answer_type,
            steps=steps,
        )

    def explain_answer(self, state: WorkflowState) -> dict[str, Any]:
        result = dict(state.get("result", {}))
        if result.get("answer") == "Unknown":
            return {"result": result}
        parsed_question = state.get("parsed_question", {})
        solution_output = state.get("solution_output", {})
        verified_output = state.get("verified_output", {})
        cot = self.explain_agent.build_cot(parsed_question, solution_output, verified_output)
        if _llm_available(self.llm):
            try:
                explanation = self.explain_agent.run(
                    parsed_question,
                    solution_output,
                    verified_output,
                )
                result["explanation"] = str(explanation["explanation"])
                result["cot"] = [str(step) for step in explanation.get("cot") or cot]
                return {"result": result}
            except Exception as exc:
                errors = _with_error(state, f"Physics ExplainAgent failed: {exc}")
        else:
            errors = state.get("errors", [])
        unit = str(result.get("unit") or "")
        suffix = f" {unit}" if result.get("append_unit") and unit and unit.lower() != "dimensionless" else ""
        steps = cot or result.get("cot") or []
        reasoning = " ".join(str(step) for step in steps) or "Computed the requested quantity from the selected equations."
        result["cot"] = [str(step) for step in steps]
        result["explanation"] = f"{reasoning} Therefore, the answer is {result.get('answer')}{suffix}."
        return {"result": result, "errors": errors}
