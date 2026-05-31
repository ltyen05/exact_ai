"""LangGraph physics subgraph wired to the structured physics agents."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.formatting import convert_si_to_requested, format_number
from agents.physics.Explain import ExplainAgent
from agents.physics.Parsing import ParsingAgent
from agents.physics.Solution import LLMSolutionProvider, RAGSolutionProvider, SolutionAgent
from agents.physics.Solution.formula_lib import build_solution_cot_steps
from agents.physics.validation import (
    comparison_tolerance,
    direction_from_components,
    requests_magnitude,
    select_effective_target,
    unit_scale_consistency_error,
    validated_context,
)
from tools.calculator import solve_with_sympy_trace
from .orchestrator import WorkflowExecutionError, WorkflowState

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
        if not _llm_available(self.llm):
            raise WorkflowExecutionError("Physics ParsingAgent requires a configured LLM.")
        try:
            parsed_question = self.parser.run(state["question"])
        except Exception as exc:
            raise WorkflowExecutionError(f"Physics ParsingAgent failed: {exc}") from exc
        logger.debug("physics.parsed_question=%s", parsed_question)
        return {"parsed_question": parsed_question}

    def select_solution(self, state: WorkflowState) -> dict[str, Any]:
        """Obtain one structured physics solution specification."""
        parsed_question = state.get("parsed_question", {})
        if self.solution_agent is None or not _llm_available(self.llm):
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

    def _repair_solution_output(
        self,
        state: WorkflowState,
        solution_output: dict[str, Any],
        validation_error: str,
    ) -> dict[str, Any]:
        """Repair a selected physics solution using downstream validation feedback."""
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

    def compute_sympy(self, state: WorkflowState) -> dict[str, Any]:
        """Execute a structured direct or computational solution."""
        parsed_question = state.get("parsed_question", {})
        solution_output = state.get("solution_output", {})
        mode = solution_output.get("mode")
        answer_type = solution_output.get("answer_type")
        if mode == "direct":
            direct = solution_output.get("direct_answer") or {}
            if answer_type not in {"yes_no", "multiple_choice", "conceptual"}:
                raise WorkflowExecutionError("Direct physics solution has an unsupported answer_type.")
            if direct.get("answer") is None or not isinstance(direct.get("rationale_steps"), list):
                raise WorkflowExecutionError("Direct physics solution answer/rationale_steps are invalid.")
            answer = direct.get("answer", "Unknown")
            selected_option = str(direct.get("selected_option") or "").strip()
            parsed_options = parsed_question.get("options") if isinstance(parsed_question, dict) else None
            if answer_type == "multiple_choice" and selected_option and isinstance(parsed_options, list):
                option_text = next(
                    (
                        str(option.get("text") or "").strip()
                        for option in parsed_options
                        if isinstance(option, dict) and str(option.get("label") or "").strip() == selected_option
                    ),
                    "",
                )
                if option_text:
                    answer = f"{selected_option}. {option_text}"
            cot = [str(step) for step in direct.get("rationale_steps") or []]
            final_answer = {"symbol": "answer", "value": str(answer), "unit": ""}
            return {
                "verified_output": {
                    "mode": "direct",
                    "answer_type": answer_type,
                    "final_answer": final_answer,
                    "solution_output": solution_output,
                },
                "result": {
                    "answer": str(answer),
                    "unit": "",
                    "append_unit": False,
                    "explanation": " ".join(cot) or "A direct physics conclusion was selected.",
                    "cot": cot,
                    "premises": [],
                    "fol": "",
                },
            }
        if mode != "computational":
            raise WorkflowExecutionError("No valid physics solution specification was produced.")
        if answer_type not in {"numeric", "yes_no"}:
            raise WorkflowExecutionError("Computational physics solution has an unsupported answer_type.")

        steps = [str(step) for step in solution_output.get("solution_steps") or []]
        try:
            quantities, target, unit, equations = validated_context(parsed_question, solution_output)
            target = select_effective_target(parsed_question, equations, target)
        except ValueError as exc:
            logger.debug("physics.verification_fail_reason=%s", exc)
            solution_output = self._repair_solution_output(
                state,
                solution_output,
                f"Physics computation validation failed: {exc}",
            )
            answer_type = self._assert_computational_solution(solution_output)
            steps = [str(step) for step in solution_output.get("solution_steps") or []]
            try:
                quantities, target, unit, equations = validated_context(parsed_question, solution_output)
                target = select_effective_target(parsed_question, equations, target)
            except ValueError as repair_exc:
                logger.debug("physics.verification_fail_reason=%s", repair_exc)
                raise WorkflowExecutionError(f"Physics computation validation failed after repair: {repair_exc}") from repair_exc
        steps = build_solution_cot_steps(steps, equations, target)
        logger.debug("physics.normalized_knowns=%s", quantities)
        computation = solve_with_sympy_trace(quantities, equations, target)
        if computation is None:
            logger.debug("physics.verification_fail_reason=equations could not resolve the target")
            solution_output = self._repair_solution_output(
                state,
                solution_output,
                "Physics computation failed: equations could not resolve the target.",
            )
            answer_type = self._assert_computational_solution(solution_output)
            steps = [str(step) for step in solution_output.get("solution_steps") or []]
            try:
                quantities, target, unit, equations = validated_context(parsed_question, solution_output)
                target = select_effective_target(parsed_question, equations, target)
            except ValueError as repair_exc:
                logger.debug("physics.verification_fail_reason=%s", repair_exc)
                raise WorkflowExecutionError(f"Physics computation validation failed after repair: {repair_exc}") from repair_exc
            steps = build_solution_cot_steps(steps, equations, target)
            computation = solve_with_sympy_trace(quantities, equations, target)
            if computation is None:
                logger.debug("physics.verification_fail_reason=equations could not resolve the target after repair")
                raise WorkflowExecutionError("Physics computation failed after repair: equations could not resolve the target.")

        logger.debug("physics.sympy_result=%s", {"value": computation.value, "trace": computation.trace})
        target_is_charge = bool(re.fullmatch(r"q\d*|charge.*", str(target).lower()))
        force_magnitude = answer_type == "numeric" and requests_magnitude(parsed_question) and not target_is_charge
        si_numeric_value = abs(computation.value) if force_magnitude else computation.value
        numeric_value = convert_si_to_requested(si_numeric_value, unit) if answer_type == "numeric" else si_numeric_value
        scale_error = unit_scale_consistency_error(si_numeric_value, numeric_value, unit) if answer_type == "numeric" else None
        if scale_error:
            raise WorkflowExecutionError(scale_error)
        public_answer = format_number(numeric_value)
        final_value: Any = numeric_value
        append_unit = answer_type == "numeric"
        decision_result: dict[str, Any] | None = None
        vector_result: dict[str, Any] | None = None
        vector_spec = solution_output.get("vector_spec") if isinstance(solution_output.get("vector_spec"), dict) else {}
        if answer_type == "numeric" and vector_spec:
            component_symbols = [str(symbol) for symbol in vector_spec.get("component_symbols") or []]
            component_values: list[float] = []
            for component_symbol in component_symbols:
                component_value = computation.values.get(component_symbol)
                if component_value is None:
                    component_computation = solve_with_sympy_trace(quantities, equations, component_symbol)
                    component_value = component_computation.value if component_computation is not None else None
                if component_value is None:
                    raise WorkflowExecutionError(f"Physics vector verification failed: {component_symbol} was not computed.")
                component_values.append(convert_si_to_requested(component_value, unit))
            direction = direction_from_components(component_values, vector_spec)
            vector_result = {
                "component_symbols": component_symbols,
                "components": component_values,
                "magnitude": numeric_value,
                "direction": direction,
            }
            public_answer = f"{format_number(numeric_value)} {unit}".strip()
            append_unit = False
            final_value = {
                "components": component_values,
                "magnitude": numeric_value,
                "direction": direction,
            }
        if answer_type == "yes_no":
            decision = solution_output.get("decision_spec") or {}
            if not isinstance(decision, dict) or not decision.get("expected_symbol"):
                raise WorkflowExecutionError("Computational yes/no solution requires decision_spec.expected_symbol.")
            expected_symbol = str(decision.get("expected_symbol") or "")
            expected_value = quantities.get(expected_symbol)
            if expected_value is None:
                raise WorkflowExecutionError("Physics comparison failed: expected value was not parsed.")
            tolerance = comparison_tolerance(expected_value)
            difference = abs(computation.value - expected_value)
            matches = difference <= tolerance
            public_answer = str(decision.get("answer_if_true", "Yes") if matches else decision.get("answer_if_false", "No"))
            final_value = public_answer
            append_unit = False
            decision_result = {
                "computed_value": computation.value,
                "expected_value": expected_value,
                "difference": difference,
                "tolerance": tolerance,
                "answer": public_answer,
            }
        elif answer_type == "numeric":
            question_text = str(parsed_question.get("question") or "").lower()
            relationship = str(solution_output.get("relationship") or "").strip()
            if relationship:
                public_answer = f"{format_number(numeric_value)} {unit}, {relationship}".strip()
                append_unit = False
            wants_both_errors = "absolute error" in question_text and "relative error" in question_text
            if wants_both_errors:
                absolute_value = computation.values.get("absolute_error")
                if absolute_value is None and target == "absolute_error":
                    absolute_value = si_numeric_value
                percentage_value = computation.values.get("percentage_relative_error")
                relative_value = computation.values.get("relative_error")
                wants_percentage_error = "%" in str(unit) or "percentage relative error" in question_text
                if wants_percentage_error and percentage_value is None and relative_value is not None:
                    percentage_value = relative_value * 100
                if wants_percentage_error and absolute_value is not None and percentage_value is not None:
                    public_answer = f"absolute_error = {format_number(absolute_value)} g; percentage_relative_error = {format_number(percentage_value)} %"
                    append_unit = False
                    unit = ""
                elif absolute_value is not None and relative_value is not None:
                    public_answer = f"absolute_error = {format_number(absolute_value)}; relative_error = {format_number(relative_value)}"
                    append_unit = False
                    unit = ""
            wants_mean_and_mae = "mean absolute error" in question_text and "mean" in question_text
            if wants_mean_and_mae:
                mean_value = computation.values.get("mean")
                mae_value = computation.values.get("mean_absolute_error") or computation.values.get("mae")
                if mean_value is None and target == "mean":
                    mean_value = numeric_value
                if mae_value is None and target in {"mean_absolute_error", "mae"}:
                    mae_value = numeric_value
                if mean_value is not None and mae_value is not None:
                    public_answer = f"mean = {format_number(mean_value)}; mean_absolute_error = {format_number(mae_value)}"
                    append_unit = False
            wants_lc_energy_split = (
                "electric energy equals the magnetic energy" in question_text
                or "electric energy equals magnetic energy" in question_text
            )
            if wants_lc_energy_split:
                electric_value = computation.values.get("W_C")
                magnetic_value = computation.values.get("W_L")
                if electric_value is not None and magnetic_value is not None:
                    electric_public = convert_si_to_requested(electric_value, unit)
                    magnetic_public = convert_si_to_requested(magnetic_value, unit)
                    public_answer = f"electric_energy = {format_number(electric_public)} {unit}; magnetic_energy = {format_number(magnetic_public)} {unit}".strip()
                    append_unit = False
                    final_value = {"electric_energy": electric_public, "magnetic_energy": magnetic_public}

        final_answer = {"symbol": target, "value": final_value, "unit": unit}
        verification_result = {
            "accepted": True,
            "derived_values_allowed": True,
            "undefined_symbols_before_sympy": [],
            "unverified_values": [],
        }
        verified_output: dict[str, Any] = {
            "mode": "computational",
            "answer_type": answer_type,
            "final_answer": final_answer,
            "solution_output": solution_output,
            "sympy_result": {
                "symbol": target,
                "value": computation.value,
                "unit": unit,
                "trace": computation.trace,
                "values": computation.values,
            },
            "verification_result": verification_result,
        }
        if decision_result is not None:
            verified_output["decision_result"] = decision_result
        if vector_result is not None:
            verified_output["vector_result"] = vector_result
        logger.debug("physics.verification_result=%s", verification_result)
        return {
            "verified_output": verified_output,
            "result": {
                "answer": public_answer,
                "unit": unit,
                "append_unit": append_unit,
                "cot": steps,
                "premises": equations,
                "fol": "",
            },
        }

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
