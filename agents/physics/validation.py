"""Validation helpers for physics computation specs."""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from agents.formatting import as_number, convert_si_to_requested, normalize_unit
from agents.physics.domain.context import build_calculation_input
from agents.physics.domain.symbols import canonical_quantity_symbol
from tools.calculator import PHYSICAL_CONSTANTS, sanitize_sympy_equation, solve_with_sympy_trace


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
    "asin",
    "sin",
    "cos",
    "diff",
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
    if key in {"j", "mj", "uj", "microj", "nj"}:
        return "energy"
    if key in {"c", "uc", "microc", "nc", "mc", "pc"}:
        return "charge"
    if key in {"v"}:
        return "voltage"
    if key in {"n/c", "v/m"}:
        return "electric_field"
    if key in {"h", "mh", "uh", "microh"}:
        return "inductance"
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


BAD_DEGREE_TRIG_RE = re.compile(
    r"\b(?:sin|cos|tan)\s*\(\s*(?:30|45|60|90|120|135|180)(?:\.0+)?\s*\)",
    flags=re.IGNORECASE,
)
BAD_COORDINATE_COMPONENT_RE = re.compile(
    r"\bF[A-Za-z0-9_]*\s*\*\s*C[xy]\b|\bC[xy]\s*\*\s*F[A-Za-z0-9_]*\b"
)


def solve_spec_structural_error(equations: list[str], known_symbols: set[str]) -> str | None:
    equation_symbols = set(known_symbols)
    for equation in equations:
        equation_symbols.update(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", equation))
    for equation in equations:
        compact = re.sub(r"\s+", "", equation)
        if BAD_DEGREE_TRIG_RE.search(compact):
            return "Equation applies sin/cos/tan to a degree literal; convert degrees to radians first."
        if BAD_COORDINATE_COMPONENT_RE.search(compact):
            return "Equation multiplies force by raw Cx/Cy coordinates instead of normalized unit-vector components."
        if (
            re.search(r"\bk(?:_e)?\b", equation)
            and {"L", "delta_t"} <= equation_symbols
            and equation_symbols & {"delta_I", "I_initial", "I_final"}
        ):
            return "Self-induction formulas must not use Coulomb constant k."

        if equation.count("=") != 1:
            continue
        lhs, rhs = [part.strip() for part in equation.split("=", 1)]
        test_charge_symbols = {"q0", "q3", "q_test"} & known_symbols
        if (
            test_charge_symbols
            and re.fullmatch(r"F[A-Za-z0-9_]*", lhs)
            and re.search(r"\bk\b|\bk_e\b", rhs)
            and re.search(r"\bq[12]\b", rhs)
            and not any(re.search(rf"\b{re.escape(symbol)}\b", rhs) for symbol in test_charge_symbols)
        ):
            return "Coulomb force on a test charge must include the signed product qi*q0."
    return None


def target_consistency_error(parsed_target: str, computed_target: str, unit: str) -> str | None:
    parsed = str(parsed_target or "").strip()
    computed = str(computed_target or "").strip()
    if not parsed or not computed or parsed in {"result", "answer", "each_energy"}:
        return None
    if parsed == computed:
        return None
    if canonical_quantity_symbol(parsed) == canonical_quantity_symbol(computed):
        return None
    target_dim = dimension_label_for_unit(unit)
    equivalent_groups = {
        "charge": {"q", "Q", "Q_source", "q_source", "charge"},
        "electric_field": {"E", "E_net", "E_total", "E_field", "E_M", "E_net_magnitude"},
        "inductance": {"L", "L_self", "L_ind", "inductance"},
        "energy": {"W", "E", "W_B", "W_C", "W_L", "W_max", "E_total", "W_total"},
        "voltage": {"U", "V", "U_new", "V_new", "epsilon", "emf", "E_ind"},
        "power": {"P", "P_active", "power"},
    }
    group = equivalent_groups.get(target_dim)
    if not group:
        return None
    if parsed in group and computed not in group:
        return f"Target mismatch: parsed target is {parsed}, but solution computes {computed}."
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
    def normalize_uncertainty_symbols(text: str) -> str:
        text = re.sub(r"\bd(?P<sym>[A-Z][A-Za-z0-9_]*)\b", r"delta_\g<sym>", text)
        text = re.sub(r"\bd(?P<sym>[x-z])\b", r"delta_\g<sym>", text)
        return text

    spec = solution_output.get("sympy_spec") or {}
    equations = [normalize_uncertainty_symbols(sanitize_sympy_equation(str(item))) for item in spec.get("equations") or []]
    target = normalize_uncertainty_symbols(str(spec.get("target_symbol") or calculation["target"]))
    unit = str(spec.get("target_unit") or calculation["unit"])
    quantities = dict(calculation["quantities"])
    if not target or not equations:
        raise ValueError("Computational solution is missing target symbol or equations.")

    raw_knowns = spec.get("known_values") or {}
    if not isinstance(raw_knowns, dict):
        raise ValueError("Computational solution known_values must be an object.")
    known_values = {normalize_uncertainty_symbols(k): v for k, v in raw_knowns.items()}
    equation_text = " ".join(equations)
    for symbol, value in PHYSICAL_CONSTANTS.items():
        if re.search(rf"\b{re.escape(symbol)}\b", equation_text):
            if symbol in quantities and not math.isclose(quantities[symbol], value, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"Parsed value for fixed physical constant {symbol} is invalid.")
            quantities[symbol] = value

    trusted_quantities = dict(quantities)
    derived_candidates: dict[str, float] = {}

    def find_matching_quantity_key(sym: str, quants: dict[str, float]) -> str | None:
        if sym in quants:
            return sym
        canon_sym = canonical_quantity_symbol(sym)
        for q_key in quants:
            if canonical_quantity_symbol(q_key) == canon_sym:
                return q_key
        return None

    def find_given_by_value(val: float, parsed_q: dict[str, Any]) -> dict[str, Any] | None:
        for given in parsed_q.get("givens") or []:
            given_val = given.get("si_value")
            if given_val is None:
                given_val = given.get("value")
            given_numeric = as_number(given_val)
            if given_numeric is not None and math.isclose(given_numeric, val, rel_tol=1e-9, abs_tol=1e-12):
                return given
        geometry = parsed_q.get("geometry") or {}
        if isinstance(geometry, dict):
            for field in ("segments", "derived_distances"):
                for item in geometry.get(field) or []:
                    if isinstance(item, dict):
                        item_val = item.get("si_value")
                        if item_val is None:
                            item_val = item.get("value")
                        item_numeric = as_number(item_val)
                        if item_numeric is not None and math.isclose(item_numeric, val, rel_tol=1e-9, abs_tol=1e-12):
                            return item
        return None

    for symbol, value in known_values.items():
        numeric = as_number(value)
        if numeric is None:
            if symbol in PHYSICAL_CONSTANTS:
                continue
            raise ValueError(f"Known value for {symbol} is not numeric.")
        
        matching_key = find_matching_quantity_key(symbol, quantities)
        if matching_key is not None:
            quantities[symbol] = quantities[matching_key]
            trusted_quantities[symbol] = quantities[matching_key]
            if not math.isclose(quantities[matching_key], numeric, rel_tol=1e-9, abs_tol=1e-12):
                logger.warning(
                    "physics.known_value_conflict symbol=%s matching_key=%s parser=%s solution=%s - using parser value",
                    symbol, matching_key, quantities[matching_key], numeric,
                )
        elif symbol in PHYSICAL_CONSTANTS:
            if not math.isclose(PHYSICAL_CONSTANTS[symbol], numeric, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"Solution value for physical constant {symbol} is invalid.")
            quantities[symbol] = PHYSICAL_CONSTANTS[symbol]
            trusted_quantities[symbol] = PHYSICAL_CONSTANTS[symbol]
        else:
            given_match = find_given_by_value(numeric, parsed_question)
            if given_match is not None:
                quantities[symbol] = numeric
                trusted_quantities[symbol] = numeric
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
    structural_error = solve_spec_structural_error(equations, set(quantities))
    if structural_error:
        raise ValueError(structural_error)
    unresolved = undefined_symbols(equations, quantities, target)
    logger.debug("physics.undefined_symbols_before_sympy=%s", unresolved)
    if unresolved:
        lhs_symbols = sorted(
            equation.split("=", 1)[0].strip()
            for equation in equations
            if isinstance(equation, str)
            and equation.count("=") == 1
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", equation.split("=", 1)[0].strip())
        )
        raise ValueError(
            "Undefined symbols before SymPy: "
            f"{', '.join(unresolved)}. "
            f"Known symbols: {', '.join(sorted(quantities)) or '(none)'}; "
            f"defined_by_equation: {', '.join(lhs_symbols) or '(none)'}; "
            f"target: {target or '(none)'}."
        )
    consistency_error = target_consistency_error(str(calculation["target"]), target, unit)
    if consistency_error:
        raise ValueError(consistency_error)
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
