from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import sympy as sp

PHYSICAL_CONSTANTS = {
    "k": 9_000_000_000.0,
    "k_e": 9_000_000_000.0,
    "epsilon_0": 8.8541878128e-12,
    "mu_0": 1.25663706212e-6,
    "c": 299_792_458.0,
}


SYMPY_VALUES = {
    "Abs": sp.Abs,
    "abs": sp.Abs,
    "acos": sp.acos,
    "atan": sp.atan,
    "cos": sp.cos,
    "diff": sp.diff,
    "exp": sp.exp,
    "log": sp.log,
    "pi": sp.pi,
    "sin": sp.sin,
    "sqrt": sp.sqrt,
    "tan": sp.tan,
}


@dataclass(frozen=True)
class SympyComputation:
    """Numeric answer plus useful resolved intermediate equations."""

    value: float
    trace: list[str]
    values: dict[str, float]


def _numeric_value(expression: Any) -> float | None:
    numeric = sp.N(expression)
    if getattr(numeric, "free_symbols", None):
        return None
    if numeric.is_real is not True:
        return None
    try:
        return float(numeric)
    except (TypeError, ValueError):
        return None


def _sympify_quantity(value: Any) -> sp.Expr:
    """Preserve user-facing decimal values instead of importing binary float noise."""
    if isinstance(value, bool):
        return sp.sympify(value, locals=SYMPY_VALUES)
    if isinstance(value, int):
        return sp.Integer(value)
    if isinstance(value, (float, Decimal)):
        return sp.Rational(str(value))
    return sp.sympify(value, locals=SYMPY_VALUES)


def solve_with_sympy_trace(
    quantities: dict[str, Any],
    formulas: list[str],
    target: str,
) -> SympyComputation | None:
    """Solve a SymPy-ready equation system and retain computed intermediates."""
    if not target or not formulas:
        return None

    try:
        symbol_names = set(quantities) | {target}
        for formula in formulas:
            symbol_names.update(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", formula))
        symbols = {
            name: sp.Symbol(name)
            for name in symbol_names
            if name not in SYMPY_VALUES
        }
        parsing_values = {**SYMPY_VALUES, **symbols}

        equations: list[sp.Equality] = []
        equation_pairs: list[tuple[sp.Expr, sp.Expr]] = []
        symbolic_definitions: dict[str, str] = {}

        def expand_diff_references(expression_text: str) -> str:
            def replace_reference(match: re.Match[str]) -> str:
                expression_name = match.group("expr")
                variable_name = match.group("var")
                if expression_name not in symbolic_definitions:
                    return match.group(0)
                return f"diff(({symbolic_definitions[expression_name]}), {variable_name})"

            return re.sub(
                r"\bdiff\s*\(\s*(?P<expr>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*\)",
                replace_reference,
                expression_text,
            )

        for formula in formulas:
            if formula.count("=") != 1:
                return None
            lhs, rhs = formula.split("=", 1)
            lhs_expr = sp.sympify(lhs.strip(), locals=parsing_values)
            rhs_source = expand_diff_references(rhs.strip())
            rhs_expr = sp.sympify(rhs_source, locals=parsing_values)
            equation_pairs.append((lhs_expr, rhs_expr))
            equations.append(
                sp.Eq(
                    lhs_expr,
                    rhs_expr,
                )
            )
            lhs_name = lhs.strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs_name):
                symbolic_definitions[lhs_name] = rhs_source

        target_symbol = symbols.get(target)
        if target_symbol is None:
            return None
        substitutions = {
            symbols[name]: _sympify_quantity(value)
            for name, value in quantities.items()
            if name in symbols and value is not None
        }
        target_defined_by_equation = any(
            isinstance(lhs_expr, sp.Symbol) and lhs_expr == target_symbol
            for lhs_expr, _ in equation_pairs
        )
        if target_defined_by_equation:
            substitutions.pop(target_symbol, None)
        elif target_symbol in substitutions:
            value = _numeric_value(substitutions[target_symbol])
            return SympyComputation(value, [], {target: value}) if value is not None else None

        sequential_substitutions = dict(substitutions)
        sequential_trace: list[str] = []
        sequential_values: dict[str, float] = {}
        progressed = True
        while progressed:
            progressed = False
            for lhs_expr, rhs_expr in equation_pairs:
                if not isinstance(lhs_expr, sp.Symbol) or lhs_expr in sequential_substitutions:
                    continue
                value = _numeric_value(rhs_expr.subs(sequential_substitutions))
                if value is None:
                    continue
                sequential_substitutions[lhs_expr] = value
                sequential_values[str(lhs_expr)] = value
                progressed = True
                if lhs_expr != target_symbol:
                    sequential_trace.append(f"{lhs_expr} = {value}")
        if target_symbol in sequential_substitutions:
            value = _numeric_value(sequential_substitutions[target_symbol])
            if value is not None:
                sequential_values[target] = value
                return SympyComputation(value, sequential_trace, sequential_values)

        reduced = [equation.subs(substitutions) for equation in equations]
        unknowns = sorted(
            (symbol for name, symbol in symbols.items() if name not in quantities),
            key=str,
        )
        solutions = sp.solve(reduced, unknowns, dict=True)

        candidates: list[SympyComputation] = []
        for solution in solutions:
            expression = solution.get(target_symbol)
            if expression is None:
                continue
            resolved_target = expression.subs(solution).subs(substitutions)
            value = _numeric_value(resolved_target)
            if value is None:
                continue
            trace: list[str] = []
            values: dict[str, float] = {}
            for symbol, intermediate in sorted(solution.items(), key=lambda item: str(item[0])):
                resolved = intermediate.subs(solution).subs(substitutions)
                numeric = _numeric_value(resolved)
                if numeric is not None:
                    values[str(symbol)] = numeric
                    if symbol != target_symbol:
                        trace.append(f"{symbol} = {numeric}")
            values[target] = value
            candidates.append(SympyComputation(value, trace, values))
        if not candidates:
            return None
        return next((result for result in candidates if result.value >= 0), candidates[0])
    except (KeyError, TypeError, ValueError, NotImplementedError, sp.SympifyError):
        return None


def solve_with_sympy(
    quantities: dict[str, Any],
    formulas: list[str],
    target: str,
) -> float | None:
    """Backward-compatible numeric-only wrapper around the traced solver."""
    computation = solve_with_sympy_trace(quantities, formulas, target)
    return computation.value if computation is not None else None
