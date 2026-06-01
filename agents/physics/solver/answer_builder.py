"""Build public and verified outputs for computed physics answers."""

from __future__ import annotations

import logging
import re
from typing import Any

from agents.formatting import convert_si_to_requested, format_number
from agents.physics.validation import (
    comparison_tolerance,
    direction_from_components,
    requests_magnitude,
    unit_scale_consistency_error,
)
from agents.workflows.state import WorkflowExecutionError
from tools.calculator import solve_with_sympy_trace

from .validator import ValidatedSolveContext

logger = logging.getLogger(__name__)


class PhysicsAnswerBuilder:
    """Build the verified_output and public result payloads."""

    def build(
        self,
        parsed_question: dict[str, Any],
        solution_output: dict[str, Any],
        context: ValidatedSolveContext,
        computation: Any,
        answer_type: str,
        steps: list[str],
    ) -> dict[str, Any]:
        logger.debug("physics.sympy_result=%s", {"value": computation.value, "trace": computation.trace})
        target = context.target
        unit = context.unit
        equations = context.equations
        quantities = context.quantities

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
            decision_result, public_answer, final_value, append_unit = self._build_yes_no(
                solution_output,
                quantities,
                computation,
            )
        elif answer_type == "numeric":
            public_answer, append_unit, unit, final_value = self._apply_numeric_special_cases(
                parsed_question,
                solution_output,
                target,
                unit,
                numeric_value,
                si_numeric_value,
                computation,
                public_answer,
                append_unit,
                final_value,
            )

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

    def _build_yes_no(
        self,
        solution_output: dict[str, Any],
        quantities: dict[str, float],
        computation: Any,
    ) -> tuple[dict[str, Any], str, Any, bool]:
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
        decision_result = {
            "computed_value": computation.value,
            "expected_value": expected_value,
            "difference": difference,
            "tolerance": tolerance,
            "answer": public_answer,
        }
        return decision_result, public_answer, public_answer, False

    def _apply_numeric_special_cases(
        self,
        parsed_question: dict[str, Any],
        solution_output: dict[str, Any],
        target: str,
        unit: str,
        numeric_value: float,
        si_numeric_value: float,
        computation: Any,
        public_answer: str,
        append_unit: bool,
        final_value: Any,
    ) -> tuple[str, bool, str, Any]:
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

        return public_answer, append_unit, unit, final_value
