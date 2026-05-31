"""Validation helpers for physics computation specs."""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from agents.formatting import as_number, convert_si_to_requested, normalize_unit
from agents.physics.Parsing.Parsing_Agent import build_calculation_input
from tools.calculator import PHYSICAL_CONSTANTS, solve_with_sympy_trace

logger = logging.getLogger(__name__)


ALLOWED_SYMPY_NAMES = {
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
FORBIDDEN_UNIT_TOKENS = {
    "Wb",
    "mWb",
    "uWb",
    "microWb",
    "nWb",
    "Ohm",
    "kOhm",
    "Hz",
    "kHz",
    "MHz",
    "Tesla",
    "tesla",
    "weber",
    "volt",
    "ampere",
    "joule",
    "farad",
    "henry",
    "microF",
    "uF",
    "nF",
    "pF",
    "microH",
    "uH",
    "mH",
    "microC",
    "uC",
    "nC",
    "pC",
}


def comparison_tolerance(expected_value: float) -> float:
    """Treat integer-valued measurements as rounded to their displayed unit."""
    rounding_tolerance = 0.5 if float(expected_value).is_integer() else 1e-2
    return max(rounding_tolerance, abs(expected_value) * 1e-3)


def requests_magnitude(parsed_question: dict[str, object]) -> bool:
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


def requests_vector(parsed_question: dict[str, object]) -> bool:
    answer_format = parsed_question.get("answer_format") or {}
    if isinstance(answer_format, dict) and str(answer_format.get("requested_form") or "").lower() == "vector":
        return True
    question = str(parsed_question.get("question") or "").lower()
    target = parsed_question.get("target") or {}
    target_symbol = str(target.get("symbol") or "").lower() if isinstance(target, dict) else str(target).lower()
    return "vector" in question or target_symbol.endswith("_vector")


def select_effective_target(parsed_question: dict[str, object], equations: list[str], default_target: str) -> str:
    question = str((parsed_question or {}).get("question") or "").lower()
    lhs_symbols = {
        equation.split("=", 1)[0].strip()
        for equation in equations
        if isinstance(equation, str) and equation.count("=") == 1
    }
    if "absolute error" in question and "relative error" not in question:
        for candidate in ("delta_P", "absolute_error"):
            if candidate in lhs_symbols:
                return candidate
    if default_target in lhs_symbols:
        return default_target
    return default_target


def direction_from_components(components: list[float], vector_spec: dict[str, object]) -> str:
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


def dimension_label_for_unit(unit: str) -> str:
    key = normalize_unit(unit)
    if key in {"w"}:
        return "power"
    if key in {"j", "mj"}:
        return "energy"
    if key in {"c", "uc", "microc", "nc", "mc"}:
        return "charge"
    if key in {"v"}:
        return "voltage"
    return ""


def expression_has_bad_dimension(target: str, unit: str, equations: list[str]) -> str | None:
    target_dim = dimension_label_for_unit(unit)
    target_names = {str(target), "P", "P_active", "W", "E", "W_max", "Q", "Qmax", "Q_max"}
    for equation in equations:
        if equation.count("=") != 1:
            continue
        lhs, rhs = [part.strip() for part in equation.split("=", 1)]
        compact_rhs = re.sub(r"\s+", "", rhs)
        if lhs not in target_names and lhs != target:
            continue
        if target_dim == "power" and re.fullmatch(r"I(?:_rms)?\*R|R\*I(?:_rms)?", compact_rhs):
            return "Formula I*R has voltage dimension, not power."
        if target_dim == "energy" and re.fullmatch(r"C\*q(?:max)?/2|C\*Q(?:_max)?/2", compact_rhs, flags=re.IGNORECASE):
            return "Formula C*q/2 does not have energy dimension."
        if target_dim == "charge" and re.fullmatch(r"sqrt\(2\*W/C\)", compact_rhs, flags=re.IGNORECASE):
            return "Formula sqrt(2*W/C) has voltage dimension, not charge."
    return None


def unit_symbol_error(equations: list[str]) -> str | None:
    normalized_unit_tokens = {
        token.replace("μ", "u").replace("µ", "u").lower()
        for token in FORBIDDEN_UNIT_TOKENS
    }
    for equation in equations:
        tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", equation))
        unit_tokens = sorted(
            token
            for token in tokens
            if token in FORBIDDEN_UNIT_TOKENS
            or token.replace("μ", "u").replace("µ", "u").lower() in normalized_unit_tokens
        )
        if unit_tokens:
            return f"Equation contains unit symbols as variables: {', '.join(unit_tokens)}."
    return None


def unit_scale_consistency_error(computed_si_value: float, final_numeric: float, requested_unit: str) -> str | None:
    expected = convert_si_to_requested(computed_si_value, requested_unit)
    tolerance = max(1e-12, abs(expected) * 1e-9)
    if abs(final_numeric - expected) > tolerance:
        return (
            "Final answer unit scale is inconsistent with computed SI value: "
            f"expected {expected} {requested_unit}, got {final_numeric} {requested_unit}."
        )
    return None


def undefined_symbols(equations: list[str], quantities: dict[str, float], target: str = "") -> list[str]:
    known = {str(symbol) for symbol in quantities}
    allowed = {*ALLOWED_SYMPY_NAMES, target}
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


def validated_context(
    parsed_question: dict[str, Any],
    solution_output: dict[str, Any],
) -> tuple[dict[str, float], str, str, list[str]]:
    """Validate the solve graph and return quantities, target, unit, and equations."""
    calculation = build_calculation_input(parsed_question)
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
        numeric = as_number(value)
        if numeric is None:
            if symbol in PHYSICAL_CONSTANTS:
                continue
            raise ValueError(f"Known value for {symbol} is not numeric.")
        if symbol in quantities:
            if not math.isclose(quantities[symbol], numeric, rel_tol=1e-9, abs_tol=1e-12):
                logger.warning(
                    "physics.known_value_conflict symbol=%s parser=%s solution=%s - using parser value",
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

    unit_error = unit_symbol_error(equations)
    if unit_error:
        raise ValueError(unit_error)
    unresolved = undefined_symbols(equations, quantities, target)
    logger.debug("physics.undefined_symbols_before_sympy=%s", unresolved)
    if unresolved:
        raise ValueError(f"Undefined symbols before SymPy: {', '.join(unresolved)}.")
    dimension_error = expression_has_bad_dimension(target, unit, equations)
    if dimension_error:
        raise ValueError(dimension_error)

    vector_spec = solution_output.get("vector_spec") if isinstance(solution_output.get("vector_spec"), dict) else {}
    component_symbols = [str(symbol) for symbol in vector_spec.get("component_symbols") or []]
    if component_symbols:
        lhs_symbols = {
            equation.split("=", 1)[0].strip()
            for equation in equations
            if equation.count("=") == 1
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", equation.split("=", 1)[0].strip())
        }
        missing_components = sorted(
            symbol for symbol in component_symbols if symbol not in lhs_symbols and symbol not in quantities
        )
        if missing_components:
            raise ValueError(
                "vector_spec component symbols must be defined by equations or known values: "
                f"{', '.join(missing_components)}."
            )
    return quantities, target, unit, equations
