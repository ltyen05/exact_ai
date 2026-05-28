"""LangGraph physics subgraph wired to the structured physics agents."""

from __future__ import annotations

import math
import logging
import re
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.formatting import format_number
from agents.physics.Explain import ExplainAgent
from agents.physics.Parsing import ParsingAgent
from agents.physics.Solution import LLMSolutionProvider, RAGSolutionProvider, SolutionAgent
from agents.physics.Solution.formula_lib import SYMBOL_ALIAS_GROUPS, build_solution_cot_steps
from tools.calculator import solve_with_sympy_trace, PHYSICAL_CONSTANTS
from .orchestrator import WorkflowExecutionError, WorkflowState

logger = logging.getLogger(__name__)


def _expand_quantity_aliases(quantities: dict[str, float]) -> dict[str, float]:
    expanded = dict(quantities)
    for group in SYMBOL_ALIAS_GROUPS:
        if any(re.fullmatch(rf"{re.escape(symbol)}_\d+", key) for symbol in group for key in expanded):
            continue
        present = [(symbol, expanded[symbol]) for symbol in group if symbol in expanded]
        if not present:
            continue
        first_value = present[0][1]
        if any(not math.isclose(first_value, value, rel_tol=1e-9, abs_tol=1e-12) for _, value in present[1:]):
            continue
        for alias in group:
            expanded.setdefault(alias, first_value)
    return expanded


def _llm_available(llm: Any) -> bool:
    return llm is not None and bool(getattr(llm, "enabled", True))


def _with_error(state: WorkflowState, error: str) -> list[str]:
    return [*state.get("errors", []), error]


def _as_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _comparison_tolerance(expected_value: float) -> float:
    """Treat integer-valued measurements as rounded to their displayed unit."""
    rounding_tolerance = 0.5 if float(expected_value).is_integer() else 1e-2
    return max(rounding_tolerance, abs(expected_value) * 1e-3)


def _requests_magnitude(parsed_question: dict[str, Any]) -> bool:
    answer_format = parsed_question.get("answer_format") or {}
    if isinstance(answer_format, dict):
        requested_form = str(answer_format.get("requested_form") or "").lower()
        if requested_form == "signed":
            return False
        if requested_form == "magnitude":
            return True

    question = str(parsed_question.get("question") or "").lower()
    if any(term in question for term in ("direction", "sign", "signed", "polarity", "lenz")):
        return False
    return any(
        term in question
        for term in (
            "magnitude",
            "strength",
            "intensity",
            "how large",
            "absolute value",
            "electromotive force",
            "emf",
        )
    )


def _requests_vector(parsed_question: dict[str, Any]) -> bool:
    answer_format = parsed_question.get("answer_format") or {}
    if isinstance(answer_format, dict) and str(answer_format.get("requested_form") or "").lower() == "vector":
        return True
    question = str(parsed_question.get("question") or "").lower()
    target = parsed_question.get("target") or {}
    target_symbol = str(target.get("symbol") or "").lower() if isinstance(target, dict) else str(target).lower()
    return "vector" in question or target_symbol.endswith("_vector")


def _select_effective_target(parsed_question: dict[str, Any], equations: list[str], default_target: str) -> str:
    question = str((parsed_question or {}).get("question") or "").lower()
    lhs_symbols = {
        equation.split("=", 1)[0].strip()
        for equation in equations
        if isinstance(equation, str) and equation.count("=") == 1
    }
    if "absolute error" in question:
        for candidate in ("delta_P", "absolute_error"):
            if candidate in lhs_symbols:
                return candidate
    if default_target in lhs_symbols:
        return default_target
    return default_target


def _direction_from_components(components: list[float], vector_spec: dict[str, Any]) -> str:
    if len(components) == 1:
        if abs(components[0]) <= 1e-9:
            return "zero field"
        return "along AB" if components[0] > 0 else "opposite AB"
    x_value = components[0]
    y_value = components[1]
    if abs(y_value) <= max(1e-9, abs(x_value) * 1e-9):
        if x_value > 0:
            return "parallel to AB, from A to B"
        if x_value < 0:
            return "parallel to AB, from B to A"
        return "zero field"
    return str(vector_spec.get("direction") or "direction determined by vector components")


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

    @staticmethod
    def _put_quantity(quantities: dict[str, float], symbol: Any, value: Any) -> None:
        name = str(symbol or "").strip()
        numeric = _as_number(value)
        if not name or numeric is None:
            return
        previous = quantities.get(name)
        if previous is not None and not math.isclose(previous, numeric, rel_tol=1e-9, abs_tol=1e-12):
            index = 2
            while f"{name}_{index}" in quantities:
                index += 1
            quantities[f"{name}_{index}"] = numeric
            return
        quantities[name] = numeric

    def _calculation_input(self, parsed_question: dict[str, Any]) -> dict[str, Any]:
        """Build trustworthy numeric context from parsed quantities and geometry."""
        quantities: dict[str, float] = {}
        raw_quantities = parsed_question.get("quantities") or {}
        if isinstance(raw_quantities, dict):
            for symbol, value in raw_quantities.items():
                self._put_quantity(quantities, symbol, value)
        elif isinstance(raw_quantities, list):
            for quantity in raw_quantities:
                if isinstance(quantity, dict):
                    value = quantity.get("si_value")
                    self._put_quantity(quantities, quantity.get("symbol"), quantity.get("value") if value is None else value)

        for given in parsed_question.get("givens") or []:
            if not isinstance(given, dict):
                continue
            value = given.get("si_value")
            self._put_quantity(quantities, given.get("symbol"), given.get("value") if value is None else value)
            uncertainty = given.get("uncertainty") or {}
            if isinstance(uncertainty, dict):
                uncertainty_value = uncertainty.get("si_value")
                if uncertainty_value is None:
                    uncertainty_value = uncertainty.get("value")
                self._put_quantity(quantities, f"delta_{given.get('symbol')}", uncertainty_value)

        geometry = parsed_question.get("geometry") or {}
        if isinstance(geometry, dict):
            for field in ("segments", "derived_distances"):
                for item in geometry.get(field) or []:
                    if isinstance(item, dict):
                        value = item.get("si_value")
                        self._put_quantity(quantities, item.get("symbol"), item.get("value") if value is None else value)

        comparison = parsed_question.get("comparison") or {}
        if isinstance(comparison, dict):
            comparison_value = comparison.get("given_si_value")
            if comparison_value is None:
                comparison_value = comparison.get("given_value")
            comparison_symbol = comparison.get("given_quantity_symbol")
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(comparison_symbol or "").strip()):
                comparison_unit = str(comparison.get("given_si_unit") or comparison.get("unit") or "").lower()
                if "hz" in comparison_unit:
                    comparison_symbol = "f"
                elif "rad" in comparison_unit:
                    comparison_symbol = "omega"
                elif comparison_unit in {"v", "volt", "volts"}:
                    comparison_symbol = "U"
            self._put_quantity(
                quantities,
                comparison_symbol,
                comparison_value,
            )

        raw_target = parsed_question.get("target") or parsed_question.get("objective") or ""
        if isinstance(raw_target, dict):
            target = str(raw_target.get("symbol") or "")
            unit = str(raw_target.get("unit") or "")
        else:
            target = str(raw_target)
            unit = str(parsed_question.get("unit") or parsed_question.get("objective_unit") or "")
        quantities = _expand_quantity_aliases(quantities)
        return {"quantities": quantities, "target": target, "unit": unit}

    def _validated_context(
        self,
        parsed_question: dict[str, Any],
        solution_output: dict[str, Any],
    ) -> tuple[dict[str, float], str, str, list[str]]:
        calculation = self._calculation_input(parsed_question)
        spec = solution_output.get("sympy_spec") or {}
        equations = [str(item) for item in spec.get("equations") or []]
        target = str(spec.get("target_symbol") or calculation["target"])
        unit = str(spec.get("target_unit") or calculation["unit"])
        quantities = dict(calculation["quantities"])
        if not target or not equations:
            raise ValueError("Computational solution is missing target symbol or equations.")

        known_values = spec.get("known_values") or {}
        if not isinstance(known_values, dict):
            raise ValueError("Computational solution known_values must be an object.")
        equation_text = " ".join(equations)
        for symbol, value in PHYSICAL_CONSTANTS.items():
            if re.search(rf"\b{re.escape(symbol)}\b", equation_text):
                if symbol in quantities and not math.isclose(quantities[symbol], value, rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError(f"Parsed value for fixed physical constant {symbol} is invalid.")
                quantities[symbol] = value

        trusted_quantities = dict(quantities)
        derived_candidates: dict[str, float] = {}
        for symbol, value in known_values.items():
            numeric = _as_number(value)
            if numeric is None:
                if symbol in PHYSICAL_CONSTANTS:
                    continue
                raise ValueError(f"Known value for {symbol} is not numeric.")
            if symbol in quantities:
                if not math.isclose(quantities[symbol], numeric, rel_tol=1e-9, abs_tol=1e-12):
                    # Trust the parser's SI-converted value over the LLM's raw value.
                    # Common case: LLM returns value in original unit (e.g. 23.8 cm²)
                    # while parser already converted to SI (e.g. 0.00238 m²).
                    logger.warning(
                        "physics.known_value_conflict symbol=%s parser=%s solution=%s — using parser value",
                        symbol, quantities[symbol], numeric,
                    )
            elif symbol in PHYSICAL_CONSTANTS:
                if not math.isclose(PHYSICAL_CONSTANTS[symbol], numeric, rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError(f"Solution value for physical constant {symbol} is invalid.")
                quantities[symbol] = PHYSICAL_CONSTANTS[symbol]
                trusted_quantities[symbol] = PHYSICAL_CONSTANTS[symbol]
            else:
                derived_candidates[str(symbol)] = numeric

        for symbol, numeric in derived_candidates.items():
            defining_equation = any(equation.split("=", 1)[0].strip() == symbol for equation in equations if "=" in equation)
            if not defining_equation:
                raise ValueError(f"Solution introduced untrusted numeric value {symbol}.")
            derived = solve_with_sympy_trace(trusted_quantities, equations, symbol)
            if derived is None or not math.isclose(derived.value, numeric, rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError(f"Derived value for {symbol} could not be verified from parsed givens.")
            quantities[symbol] = numeric
            trusted_quantities[symbol] = numeric

        undefined = self._undefined_symbols(equations, quantities)
        logger.debug("physics.undefined_symbols_before_sympy=%s", undefined)
        if undefined:
            raise ValueError(f"Undefined symbols before SymPy: {', '.join(undefined)}.")
        return quantities, target, unit, equations

    @staticmethod
    def _undefined_symbols(equations: list[str], quantities: dict[str, float]) -> list[str]:
        allowed = {
            "Abs",
            "abs",
            "Im",
            "im",
            "Re",
            "re",
            "conjugate",
            "sqrt",
            "acos",
            "sin",
            "cos",
            "tan",
            "atan",
            "exp",
            "log",
            "pi",
            *PHYSICAL_CONSTANTS,
        }
        known = {str(symbol) for symbol in quantities}
        defined: set[str] = set()
        unresolved: set[str] = set()
        for equation in equations:
            if equation.count("=") != 1:
                unresolved.add(equation)
                continue
            lhs, rhs = equation.split("=", 1)
            rhs_symbols = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", rhs))
            unresolved.update(rhs_symbols - known - defined - allowed)
            lhs_symbol = lhs.strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs_symbol):
                defined.add(lhs_symbol)
        return sorted(unresolved)

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
            quantities, target, unit, equations = self._validated_context(parsed_question, solution_output)
            target = _select_effective_target(parsed_question, equations, target)
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
                quantities, target, unit, equations = self._validated_context(parsed_question, solution_output)
                target = _select_effective_target(parsed_question, equations, target)
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
                quantities, target, unit, equations = self._validated_context(parsed_question, solution_output)
                target = _select_effective_target(parsed_question, equations, target)
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
        force_magnitude = answer_type == "numeric" and _requests_magnitude(parsed_question) and not target_is_charge
        numeric_value = abs(computation.value) if force_magnitude else computation.value
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
                component_values.append(component_value)
            direction = _direction_from_components(component_values, vector_spec)
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
            tolerance = _comparison_tolerance(expected_value)
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
            wants_both_errors = "absolute error" in question_text and "relative error" in question_text
            if wants_both_errors:
                absolute_value = computation.values.get("absolute_error")
                if absolute_value is None and target == "absolute_error":
                    absolute_value = numeric_value
                relative_value = computation.values.get("relative_error")
                if absolute_value is not None and relative_value is not None:
                    public_answer = f"absolute_error = {format_number(absolute_value)}; relative_error = {format_number(relative_value)}"
                    append_unit = False
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
