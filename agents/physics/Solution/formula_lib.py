"""Deterministic physics formula library used before LLM solution generation."""

from __future__ import annotations

import math
import re
from typing import Any


SYMBOL_REPLACEMENTS = {
    "ℓ": "ell",
    "φ": "phi",
    "Φ": "Phi",
    "θ": "theta",
    "ω": "omega",
    "Ω": "Ohm",
    "μ": "mu",
    "µ": "mu",
    "λ": "lambda_",
    "√": "sqrt",
    "−": "-",
    "–": "-",
}
SYMBOL_ALIAS_GROUPS = (
    ("U", "V", "U_rms", "V_rms"),
    ("I", "I_rms", "I_effective"),
    ("f", "frequency"),
    ("XL", "X_L", "ZL", "Z_L"),
    ("XC", "X_C", "ZC", "Z_C"),
)
PHYSICAL_CONSTANT_NAMES = {"k", "k_e", "epsilon_0", "mu_0", "c"}
COMPUTATIONAL_MODE_ALIASES = {
    "computational",
    "compute",
    "calculation",
    "formula",
    "numeric",
    "numerical",
    "sympy",
    "computational_numeric",
    "yes_no_computational",
}
DIRECT_MODE_ALIASES = {
    "direct",
    "conceptual",
    "qualitative",
    "text",
    "direct_answer",
    "yes_no_conceptual",
    "multiple_choice",
}
NUMERIC_ANSWER_ALIASES = {
    "numeric",
    "number",
    "numerical",
    "scalar",
    "calculation",
    "computed",
    "computational",
    "computational_numeric",
}
YES_NO_ANSWER_ALIASES = {
    "yes_no",
    "yes/no",
    "boolean",
    "bool",
    "true_false",
    "true/false",
    "yes_no_computational",
    "yes_no_conceptual",
}
CONCEPTUAL_ANSWER_ALIASES = {"conceptual", "qualitative", "text", "direct"}
MULTIPLE_CHOICE_ANSWER_ALIASES = {"multiple_choice", "multiple-choice", "mcq", "choice"}
FARADAY_COT_STEPS = [
    "Step1: Use Faraday's law of electromagnetic induction to relate the induced electromotive force (EMF) to the rate of change of magnetic flux.",
    "Step 2:The induced EMF is given by the equation E_ind = -N * (phi_final - phi_initial) / t, where N is the number of turns, phi_final is the final magnetic flux, phi_initial is the initial magnetic flux, and t is the time interval.",
]


def _normalize_text(value: str) -> str:
    normalized = value
    for source, replacement in SYMBOL_REPLACEMENTS.items():
        normalized = normalized.replace(source, replacement)
    return normalized


def _clean_symbol_name(symbol: Any) -> str:
    name = _normalize_text(str(symbol or "")).strip()
    name = name.strip("`'\" .,;:")
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def _is_identifier(symbol: Any) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(symbol or "")))


def _alias_groups_for(symbol: str) -> list[tuple[str, ...]]:
    return [group for group in SYMBOL_ALIAS_GROUPS if symbol in group]


def _expand_symbol_aliases(values: dict[str, float]) -> dict[str, float]:
    expanded = dict(values)
    for group in SYMBOL_ALIAS_GROUPS:
        if any(re.fullmatch(rf"{re.escape(symbol)}_\d+", key) for symbol in group for key in expanded):
            continue
        present = [(symbol, expanded[symbol]) for symbol in group if symbol in expanded]
        if not present:
            continue
        group_value = present[0][1]
        if any(not math.isclose(float(group_value), float(value), rel_tol=1e-9, abs_tol=1e-12) for _, value in present[1:]):
            continue
        for alias in group:
            expanded.setdefault(alias, group_value)
    return expanded


def _lookup_value(values: dict[str, float], *symbols: str) -> float | None:
    for symbol in symbols:
        if symbol in values:
            return values[symbol]
        for group in _alias_groups_for(symbol):
            for alias in group:
                if alias in values:
                    return values[alias]
    return None


def _comparison_symbol(comparison: dict[str, Any]) -> str:
    symbol = _clean_symbol_name(comparison.get("given_quantity_symbol"))
    unit = str(comparison.get("given_si_unit") or comparison.get("unit") or "").lower()
    if _is_identifier(symbol):
        return symbol
    if "hz" in unit:
        return "f"
    if "rad" in unit:
        return "omega"
    if unit in {"v", "volt", "volts"}:
        return "U"
    return ""


def _numeric_values(semantic_output: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    conflicted_symbols: set[str] = set()

    def put_value(symbol: Any, value: Any) -> None:
        if not isinstance(value, (int, float)):
            return
        clean_symbol = _clean_symbol_name(symbol)
        if not clean_symbol or not _is_identifier(clean_symbol):
            return
        numeric = float(value)
        if clean_symbol in values and not math.isclose(values[clean_symbol], numeric, rel_tol=1e-9, abs_tol=1e-12):
            conflicted_symbols.add(clean_symbol)
        values[clean_symbol] = numeric

    for item in semantic_output.get("givens") or []:
        if isinstance(item, dict):
            symbol = item.get("symbol")
            put_value(symbol, item.get("si_value"))
            uncertainty = item.get("uncertainty") or {}
            if isinstance(uncertainty, dict):
                uncertainty_value = uncertainty.get("si_value")
                if isinstance(uncertainty_value, (int, float)):
                    clean_symbol = _clean_symbol_name(symbol)
                    put_value(f"delta_{clean_symbol}", uncertainty_value)
                    put_value("uncertainty", uncertainty_value)
                    put_value("absolute_uncertainty", uncertainty_value)
    geometry = semantic_output.get("geometry") or {}
    if isinstance(geometry, dict):
        for field in ("segments", "derived_distances"):
            for item in geometry.get(field) or []:
                if isinstance(item, dict):
                    put_value(item.get("symbol"), item.get("si_value"))
    comparison = semantic_output.get("comparison") or {}
    if isinstance(comparison, dict) and isinstance(comparison.get("given_si_value"), (int, float)):
        put_value(_comparison_symbol(comparison), comparison.get("given_si_value"))
    expanded = _expand_symbol_aliases(values)
    for symbol in conflicted_symbols:
        for group in _alias_groups_for(symbol):
            for alias in group:
                if alias != symbol and alias not in values:
                    expanded.pop(alias, None)
    return expanded


def _numeric_sequence_items(semantic_output: dict[str, Any], *symbols: str) -> list[tuple[str, float]]:
    accepted = set(symbols)
    for symbol in symbols:
        for group in _alias_groups_for(symbol):
            accepted.update(group)
    counts: dict[str, int] = {}
    values: list[tuple[str, float]] = []
    for item in semantic_output.get("givens") or []:
        if not isinstance(item, dict) or not isinstance(item.get("si_value"), (int, float)):
            continue
        symbol = _clean_symbol_name(item.get("symbol"))
        if symbol not in accepted:
            continue
        counts[symbol] = counts.get(symbol, 0) + 1
        output_symbol = symbol if counts[symbol] == 1 else f"{symbol}_{counts[symbol]}"
        values.append((output_symbol, float(item["si_value"])))
    return values


def _target_from_semantics(semantic_output: dict[str, Any]) -> str:
    target = semantic_output.get("target") or {}
    if isinstance(target, dict):
        return _clean_symbol_name(target.get("symbol"))
    return _clean_symbol_name(target)


def _target_unit_from_semantics(semantic_output: dict[str, Any], default: str = "") -> str:
    target = semantic_output.get("target") or {}
    if isinstance(target, dict):
        unit = str(target.get("unit") or "").strip()
        if unit:
            return _normalize_text(unit)
    return default


def _semantic_text(semantic_output: dict[str, Any]) -> str:
    parts = [
        str(semantic_output.get("question") or ""),
        str(semantic_output.get("domain") or ""),
        str(semantic_output.get("question_kind") or ""),
    ]
    parts.extend(str(relation) for relation in semantic_output.get("relations") or [])
    target = semantic_output.get("target") or {}
    if isinstance(target, dict):
        parts.append(str(target.get("symbol") or ""))
        parts.append(str(target.get("unit") or ""))
    parts.append(str(semantic_output.get("comparison") or ""))
    return _normalize_text(" ".join(parts)).lower()


def _parse_number_token(token: str) -> float | None:
    text = _normalize_text(str(token or "")).lower().strip()
    text = text.replace(",", "")
    text = text.replace("×", "x")
    text = text.replace("√", "sqrt")
    sci = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:x|\*)\s*10\s*(?:\^|\*\*)?\s*([+-]?\d+)",
        text,
    )
    if sci:
        return float(sci.group(1)) * (10.0 ** int(sci.group(2)))
    sqrt_match = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)?)\s*\*?\s*sqrt\s*\(?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)?",
        text,
    )
    if sqrt_match:
        coefficient = sqrt_match.group(1)
        factor = float(coefficient) if coefficient not in {"", "+", "-"} else (-1.0 if coefficient == "-" else 1.0)
        return factor * math.sqrt(float(sqrt_match.group(2)))
    try:
        return float(text)
    except ValueError:
        return None


def _extract_total_charge_from_text(question: str) -> float | None:
    match = re.search(
        r"q1\s*\+\s*q2\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*(?:x|\*)\s*10\s*(?:\^|\*\*)?\s*[+-]?\d+)?)",
        _normalize_text(question),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return _parse_number_token(match.group(1))


def _extract_q_distances_from_text(question: str) -> tuple[float, float] | None:
    match = re.search(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(cm|mm|m)\s*from\s*q1.*?"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(cm|mm|m)\s*from\s*q2",
        _normalize_text(question),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    unit_scale = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
    r1 = float(match.group(1)) * unit_scale[match.group(2).lower()]
    r2 = float(match.group(3)) * unit_scale[match.group(4).lower()]
    return r1, r2


def _measurement_value_and_delta(values: dict[str, float], *preferred_symbols: str) -> tuple[str, float, float] | None:
    for symbol in preferred_symbols:
        if values.get(symbol) is not None and values.get(f"delta_{symbol}") is not None:
            return symbol, float(values[symbol]), float(values[f"delta_{symbol}"])
    for symbol, value in values.items():
        if symbol.startswith("delta_") or symbol in {"uncertainty", "absolute_uncertainty"}:
            continue
        delta = values.get(f"delta_{symbol}")
        if delta is not None:
            return symbol, float(value), float(delta)
    generic_delta = values.get("absolute_uncertainty") or values.get("uncertainty")
    if generic_delta is None:
        return None
    candidates = [
        (symbol, value)
        for symbol, value in values.items()
        if not symbol.startswith("delta_")
        and symbol not in {"uncertainty", "absolute_uncertainty"}
        and isinstance(value, (int, float))
    ]
    if len(candidates) == 1:
        symbol, value = candidates[0]
        return symbol, float(value), float(generic_delta)
    return None


def _normalized_label(value: Any) -> str:
    return re.sub(r"[_\s\-/]+", "_", _normalize_text(str(value or "")).strip().lower()).strip("_")


def _target_is(target: str, *symbols: str) -> bool:
    normalized_target = _clean_symbol_name(target)
    return normalized_target in {_clean_symbol_name(symbol) for symbol in symbols}


def _target_from_terms(semantic_output: dict[str, Any], fallback: str) -> str:
    target = _target_from_semantics(semantic_output)
    return target if _is_identifier(target) else fallback


def _solution(
    formula_ids: list[str],
    target_symbol: str,
    target_unit: str,
    equations: list[str],
    known_values: dict[str, float],
    steps: list[str],
    **extra: Any,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "mode": "computational",
        "answer_type": "numeric",
        "formula_ids": formula_ids,
        "sympy_spec": {
            "target_symbol": target_symbol,
            "target_unit": target_unit,
            "equations": equations,
            "known_values": known_values,
        },
        "solution_steps": steps,
    }
    output.update(extra)
    return output


def _is_force_target(target: str, text: str) -> bool:
    return _target_is(target, "F", "F_net", "F_total", "force") or "force" in text


def _point_charge_vector_solution(
    *,
    formula_ids: list[str],
    target_symbol: str,
    target_unit: str,
    coordinate_equations: list[str],
    known_values: dict[str, float],
    source_charges: list[tuple[str, str, str]],
    point_x: str,
    point_y: str,
    target_kind: str,
    test_charge_symbol: str | None = None,
    steps: list[str] | None = None,
    assumptions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build reusable Coulomb vector equations from coordinates and charges."""
    equations = list(coordinate_equations)
    field_x_terms: list[str] = []
    field_y_terms: list[str] = []
    for index, (charge_symbol, charge_x, charge_y) in enumerate(source_charges, start=1):
        equations.extend(
            [
                f"dx{index} = {point_x} - {charge_x}",
                f"dy{index} = {point_y} - {charge_y}",
                f"r{index} = sqrt(dx{index}**2 + dy{index}**2)",
                f"E{index}x = k * {charge_symbol} * dx{index} / r{index}**3",
                f"E{index}y = k * {charge_symbol} * dy{index} / r{index}**3",
            ]
        )
        field_x_terms.append(f"E{index}x")
        field_y_terms.append(f"E{index}y")

    equations.append(f"E_net_x = {' + '.join(field_x_terms)}")
    equations.append(f"E_net_y = {' + '.join(field_y_terms)}")
    equations.append("E_net_magnitude = sqrt(E_net_x**2 + E_net_y**2)")

    component_symbols = ["E_net_x", "E_net_y"]
    magnitude_symbol = "E_net_magnitude"
    if target_kind == "force":
        if not test_charge_symbol:
            return {}
        equations.extend(
            [
                f"F_net_x = {test_charge_symbol} * E_net_x",
                f"F_net_y = {test_charge_symbol} * E_net_y",
                "F_net_magnitude = sqrt(F_net_x**2 + F_net_y**2)",
            ]
        )
        component_symbols = ["F_net_x", "F_net_y"]
        magnitude_symbol = "F_net_magnitude"

    if target_symbol != magnitude_symbol:
        equations.append(f"{target_symbol} = {magnitude_symbol}")

    formula_ids = [
        *formula_ids,
        "electrostatics.point_charge_field_vector",
        "electrostatics.net_vector_cartesian",
    ]
    if target_kind == "force":
        formula_ids.append("electrostatics.force_on_test_charge_from_field")
    extra: dict[str, Any] = {
        "vector_spec": {
            "component_symbols": component_symbols,
            "magnitude_symbol": target_symbol,
            "direction": "direction determined by signed Cartesian components",
        }
    }
    if assumptions:
        extra["assumptions"] = assumptions
    return _solution(
        list(dict.fromkeys(formula_ids)),
        target_symbol,
        target_unit,
        equations,
        known_values,
        steps
        or [
            "Assign coordinates to the source charges and the target point.",
            "Use the signed Coulomb vector field formula for each source charge.",
            "Add components before computing the final magnitude.",
        ],
        **extra,
    )


def _triangle_point_symbols(values: dict[str, float]) -> tuple[str, str, str] | None:
    for point, from_a, from_b in (
        ("C", "AC", "BC"),
        ("M", "AM", "BM"),
        ("N", "AN", "BN"),
        ("P", "AP", "BP"),
        ("Q0", "AQ0", "BQ0"),
    ):
        if values.get(from_a) is not None and values.get(from_b) is not None:
            return point, from_a, from_b
    return None


def _direct(answer: str, steps: list[str]) -> dict[str, Any]:
    return {
        "mode": "direct",
        "answer_type": "conceptual",
        "direct_answer": {
            "answer": answer,
            "selected_option": None,
            "rationale_steps": steps,
        },
    }


def _direct_multiple_choice(answer: str, selected_option: str, steps: list[str]) -> dict[str, Any]:
    return {
        "mode": "direct",
        "answer_type": "multiple_choice",
        "direct_answer": {
            "answer": answer,
            "selected_option": selected_option or None,
            "rationale_steps": steps,
        },
    }


def _formula_only_solution(semantic_output: dict[str, Any], text: str, values: dict[str, float]) -> dict[str, Any] | None:
    if values:
        return None
    if "lc" not in text and "rlc" not in text:
        return None
    if "angular" in text or "omega" in text:
        return _direct(
            "The resonant angular frequency is omega_0 = 1/sqrt(L*C).",
            [
                "Identify the LC resonance condition for the circuit.",
                "For an LC circuit, resonance occurs when omega_0**2 * L * C = 1 because the inductive and capacitive reactances are equal in magnitude.",
                "Solving that relation gives omega_0 = 1/sqrt(L*C).",
            ],
        )
    if "resonant frequency" in text or "resonance frequency" in text:
        return _direct(
            "The resonant frequency is f_res = 1/(2*pi*sqrt(L*C)).",
            [
                "Identify the LC resonance condition for the circuit.",
                "First find the angular resonance frequency from omega_0 = 1/sqrt(L*C).",
                "Then convert angular frequency to ordinary frequency with f_res = omega_0/(2*pi).",
            ],
        )
    return None


def _electric_equilateral_solution(semantic_output: dict[str, Any]) -> dict[str, Any] | None:
    values = _numeric_values(semantic_output)
    q1 = values.get("q1")
    q2 = values.get("q2")
    side = values.get("a") or values.get("AB")
    if q1 is None or q2 is None or side is None:
        return None
    target = _target_from_terms(semantic_output, "E_net_magnitude")
    target = target if _is_identifier(target) and target != "result" else "E_net_magnitude"
    return _point_charge_vector_solution(
        formula_ids=["electrostatics.equilateral_geometry"],
        target_symbol=target,
        target_unit=_target_unit_from_semantics(semantic_output, "N/C"),
        coordinate_equations=[
            "Ax = 0",
            "Ay = 0",
            "Bx = a",
            "By = 0",
            "Nx = a / 2",
            "Ny = a * sqrt(3) / 2",
        ],
        known_values={"q1": q1, "q2": q2, "a": side, "k": 9e9},
        source_charges=[("q1", "Ax", "Ay"), ("q2", "Bx", "By")],
        point_x="Nx",
        point_y="Ny",
        target_kind="field",
        steps=[
            "Place A at the origin, B on the positive x-axis, and N above AB.",
            "Use the signed vector field formula E = k*q*r_vector/|r_vector|**3 for each charge.",
            "Add x/y components before computing magnitude and direction.",
        ],
    )


def _electric_midpoint_solution(semantic_output: dict[str, Any]) -> dict[str, Any] | None:
    values = _numeric_values(semantic_output)
    q1 = values.get("q1")
    q2 = values.get("q2")
    ab = values.get("AB") or values.get("a")
    if q1 is None or q2 is None or ab is None:
        return None
    text = _semantic_text(semantic_output)
    target = _target_from_terms(semantic_output, "E_net_magnitude")
    target_kind = "force" if _is_force_target(target, text) and values.get("q3") is not None else "field"
    target = target if _is_identifier(target) and target != "result" else ("F_net_magnitude" if target_kind == "force" else "E_net_magnitude")
    unit = _target_unit_from_semantics(semantic_output, "N" if target_kind == "force" else "N/C")
    knowns = {"q1": q1, "q2": q2, "AB": ab, "k": 9e9}
    if target_kind == "force" and values.get("q3") is not None:
        knowns["q3"] = values["q3"]
    return _point_charge_vector_solution(
        formula_ids=["electrostatics.midpoint_geometry"],
        target_symbol=target,
        target_unit=unit,
        coordinate_equations=[
            "Ax = 0",
            "Ay = 0",
            "Bx = AB",
            "By = 0",
            "Mx = AB / 2",
            "My = 0",
        ],
        known_values=knowns,
        source_charges=[("q1", "Ax", "Ay"), ("q2", "Bx", "By")],
        point_x="Mx",
        point_y="My",
        target_kind=target_kind,
        test_charge_symbol="q3" if target_kind == "force" else None,
        steps=[
            "Use a signed coordinate model with A and B on the x-axis.",
            "Compute the net electric field by adding components before any magnitude.",
            "For force on the midpoint charge, multiply the net field by the signed test charge.",
        ],
    )


def _identical_midpoint_zero_field_solution(semantic_output: dict[str, Any], text: str) -> dict[str, Any] | None:
    if "midpoint" not in text or "identical" not in text:
        return None
    values = _numeric_values(semantic_output)
    q_value = values.get("q")
    ab = values.get("AB") or values.get("a")
    if q_value is None or ab is None:
        return None
    target = _target_from_terms(semantic_output, "E_net_magnitude")
    solved_target = target if _is_identifier(target) else "E_net_magnitude"
    return _solution(
        ["electrostatics.midpoint_identical_charges_cancel"],
        solved_target,
        _target_unit_from_semantics(semantic_output, "N/C"),
        [f"{solved_target} = 0"],
        {"q": q_value, "AB": ab},
        [
            "At the midpoint, each identical charge contributes the same field magnitude.",
            "Those two fields point in opposite directions, so the net electric field is zero.",
        ],
    )


def _zero_field_unknown_charge_solution(semantic_output: dict[str, Any], text: str) -> dict[str, Any] | None:
    if "e = 0" not in text and "field strength is e = 0" not in text and "net electric field strength is e = 0" not in text:
        return None
    target = _target_from_terms(semantic_output, "q2")
    if target not in {"q1", "q2"}:
        return None
    question = str(semantic_output.get("question") or "")
    values = _numeric_values(semantic_output)
    total_charge = values.get("q_total") or values.get("q_sum") or _extract_total_charge_from_text(question)
    distance_pair = _extract_q_distances_from_text(question)
    if total_charge is None or distance_pair is None:
        return None
    r1, r2 = distance_pair
    if r1 <= 0 or r2 <= 0:
        return None
    ab = values.get("AB") or values.get("d_AB") or values.get("a")
    between_segment = False
    outside_segment = False
    if isinstance(ab, (int, float)) and ab > 0:
        between_segment = math.isclose(r1 + r2, ab, rel_tol=1e-2, abs_tol=5e-4)
        outside_segment = math.isclose(abs(r1 - r2), ab, rel_tol=1e-2, abs_tol=5e-4)
    sign_factor = -1 if between_segment and not outside_segment else 1
    return _solution(
        ["electrostatics.charge_balance_from_zero_field"],
        target,
        _target_unit_from_semantics(semantic_output, "C"),
        [
            f"q_total = {total_charge}",
            f"r1 = {r1}",
            f"r2 = {r2}",
            "q1 + q2 = q_total",
            f"q1 / r1**2 {'-' if sign_factor < 0 else '+'} q2 / r2**2 = 0",
        ],
        {},
        [
            "Use the sum relation between the two charges.",
            "Apply the zero-field condition at M using the two distances.",
            "Solve the resulting linear system for the requested charge.",
        ],
    )


def _zero_field_solution(semantic_output: dict[str, Any]) -> dict[str, Any] | None:
    values = _numeric_values(semantic_output)
    q1 = values.get("q1")
    q2 = values.get("q2")
    ab = values.get("AB") or values.get("a")
    if q1 is None or q2 is None or ab is None or q1 == 0 or q2 == 0:
        return None
    if q1 * q2 > 0:
        requested_target = _target_from_terms(semantic_output, "")
        target = requested_target if requested_target in {"AM", "AN", "AP", "BM", "BN", "BP", "x_from_A"} else "x_from_A"
        equations = ["x_from_A = AB * sqrt(Abs(q1)) / (sqrt(Abs(q1)) + sqrt(Abs(q2)))"]
        if target in {"BM", "BN", "BP"}:
            equations.append(f"{target} = AB - x_from_A")
        elif target in {"AM", "AN", "AP"}:
            equations.append(f"{target} = x_from_A")
        return _solution(
            ["electrostatics.zero_field_point_two_charges_1d"],
            target,
            "m",
            equations,
            {"q1": q1, "q2": q2, "AB": ab},
            [
                "Set k*Abs(q1)/AM**2 = k*Abs(q2)/BM**2 for same-sign charges between A and B.",
                "Solve for the zero-field position x_from_A, then convert it to the requested distance.",
            ],
        )
    abs_q1 = abs(q1)
    abs_q2 = abs(q2)
    requested_target = _target_from_terms(semantic_output, "")
    if math.isclose(abs_q1, abs_q2, rel_tol=1e-12, abs_tol=0.0):
        return _direct(
            "No finite zero-field point exists for equal-magnitude opposite charges.",
            ["Outside the segment the fields never balance, and between the charges they add."],
        )
    if abs_q1 < abs_q2:
        if requested_target == "BM":
            return _solution(
                ["electrostatics.zero_field_point_two_charges_1d"],
                "BM",
                "m",
                ["BM = AB * sqrt(Abs(q2)) / (sqrt(Abs(q2)) - sqrt(Abs(q1)))"],
                {"q1": q1, "q2": q2, "AB": ab},
                ["For opposite-sign charges, the zero-field point is outside the segment on the side of the smaller charge."],
            )
        return _solution(
            ["electrostatics.zero_field_point_two_charges_1d"],
            "AM",
            "m",
            ["AM = AB * sqrt(Abs(q1)) / (sqrt(Abs(q2)) - sqrt(Abs(q1)))"],
            {"q1": q1, "q2": q2, "AB": ab},
            ["For opposite-sign charges, the zero-field point is outside the segment on the side of the smaller charge."],
        )
    if requested_target == "AM":
        return _solution(
            ["electrostatics.zero_field_point_two_charges_1d"],
            "AM",
            "m",
            ["AM = AB * sqrt(Abs(q1)) / (sqrt(Abs(q1)) - sqrt(Abs(q2)))"],
            {"q1": q1, "q2": q2, "AB": ab},
            ["For opposite-sign charges, the zero-field point is outside the segment on the side of the smaller charge."],
        )
    return _solution(
        ["electrostatics.zero_field_point_two_charges_1d"],
        "BM",
        "m",
        ["BM = AB * sqrt(Abs(q2)) / (sqrt(Abs(q1)) - sqrt(Abs(q2)))"],
        {"q1": q1, "q2": q2, "AB": ab},
        ["For opposite-sign charges, the zero-field point is outside the segment on the side of the smaller charge."],
    )


def _dielectric_point_charge_solution(semantic_output: dict[str, Any], text: str) -> dict[str, Any] | None:
    if "dielectric constant" not in text and "homogeneous" not in text:
        return None
    if "point charge" not in text or "electric field" not in text:
        return None
    values = _numeric_values(semantic_output)
    epsilon_r = _lookup_value(values, "epsilon_r", "epsilon", "er")
    e_value = _lookup_value(values, "E", "E_M", "E_field")
    r_value = _lookup_value(values, "r", "AM", "OM", "d")
    question = str(semantic_output.get("question") or "")
    if r_value is None:
        distance_match = re.search(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*m\s+away", _normalize_text(question), flags=re.IGNORECASE)
        if distance_match:
            r_value = float(distance_match.group(1))
    if epsilon_r is None:
        dielectric_match = re.search(
            r"dielectric constant(?:\s+of)?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
            _normalize_text(question),
            flags=re.IGNORECASE,
        )
        if dielectric_match:
            epsilon_r = float(dielectric_match.group(1))
    if e_value is None:
        field_match = re.search(
            r"(?:magnitude of|magnitude)\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*(?:x|\*)\s*10\s*(?:\^|\*\*)?\s*[+-]?\d+)?)\s*v\s*/\s*m",
            _normalize_text(question),
            flags=re.IGNORECASE,
        )
        if field_match:
            e_value = _parse_number_token(field_match.group(1))
    if epsilon_r is None or e_value is None or r_value is None:
        return None
    q_magnitude = abs(float(e_value) * float(epsilon_r) * float(r_value) ** 2 / 9e9)
    points_toward = any(
        phrase in text
        for phrase in ("towards the charge", "toward the charge", "towards q", "toward q", "points to q", "points toward q")
    )
    q_value = -q_magnitude if points_toward else q_magnitude

    options = semantic_output.get("options") or []
    if isinstance(options, list):
        for option in options:
            if not isinstance(option, dict):
                continue
            label = str(option.get("label") or "").strip()
            option_text = _normalize_text(str(option.get("text") or "")).lower()
            if not label or not option_text:
                continue
            if q_value < 0 and not any(term in option_text for term in ("negative", "q<0", "q < 0", "-")):
                continue
            if q_value > 0 and any(term in option_text for term in ("negative", "q<0", "q < 0")):
                continue
            numeric_match = re.search(
                r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*(?:x|\*)\s*10\s*(?:\^|\*\*)?\s*[+-]?\d+)?)",
                option_text,
            )
            if numeric_match:
                option_value = _parse_number_token(numeric_match.group(1))
                if option_value is not None and not math.isclose(abs(option_value), q_magnitude, rel_tol=0.12, abs_tol=max(1e-12, q_magnitude * 0.12)):
                    continue
            return _direct_multiple_choice(
                label,
                label,
                [
                    "The electric field points toward the source charge, so the charge must be negative.",
                    f"Use E = k*|q|/(epsilon_r*r**2), giving |q| = epsilon_r*E*r**2/k = {q_magnitude}.",
                ],
            )

    equations = []
    knowns: dict[str, float] = {"k": 9e9}
    if values.get("epsilon_r") is not None or values.get("epsilon") is not None or values.get("er") is not None:
        epsilon_symbol = "epsilon_r" if values.get("epsilon_r") is not None else "epsilon" if values.get("epsilon") is not None else "er"
        knowns[epsilon_symbol] = float(epsilon_r)
    else:
        epsilon_symbol = "epsilon_r_eff"
        equations.append(f"epsilon_r_eff = {float(epsilon_r)}")
    if _lookup_value(values, "E", "E_M", "E_field") is not None:
        field_symbol = "E" if values.get("E") is not None else "E_M" if values.get("E_M") is not None else "E_field"
        knowns[field_symbol] = float(e_value)
    else:
        field_symbol = "E_eff"
        equations.append(f"E_eff = {float(e_value)}")
    if _lookup_value(values, "r", "AM", "OM", "d") is not None:
        radius_symbol = "r" if values.get("r") is not None else "AM" if values.get("AM") is not None else "OM" if values.get("OM") is not None else "d"
        knowns[radius_symbol] = float(r_value)
    else:
        radius_symbol = "r_eff"
        equations.append(f"r_eff = {float(r_value)}")
    equations.append(f"q = -{epsilon_symbol} * {field_symbol} * {radius_symbol}**2 / k")
    return _solution(
        ["electrostatics.point_charge_from_field_in_dielectric"],
        "q",
        "C",
        equations,
        knowns,
        [
            "Use E = k*|q|/(epsilon_r*r**2) in a dielectric medium.",
            "Because the field points toward the charge, q is negative.",
        ],
    )


def _two_source_charges(values: dict[str, float]) -> tuple[dict[str, float], list[tuple[str, str, str]]] | None:
    q1 = values.get("q1")
    q2 = values.get("q2")
    if (q1 is None or q2 is None) and values.get("q") is not None:
        return {"q": float(values["q"])}, [("q", "Ax", "Ay"), ("q", "Bx", "By")]
    if q1 is None or q2 is None:
        return None
    return {"q1": float(q1), "q2": float(q2)}, [("q1", "Ax", "Ay"), ("q2", "Bx", "By")]


def _test_charge_symbol(values: dict[str, float]) -> tuple[str, float] | None:
    for symbol in ("q3", "q0", "q_test"):
        if values.get(symbol) is not None:
            return symbol, float(values[symbol])
    return None


def _perpendicular_bisector_vector_solution(semantic_output: dict[str, Any], values: dict[str, float], text: str) -> dict[str, Any] | None:
    charges = _two_source_charges(values)
    base_symbol = "AB" if values.get("AB") is not None else "d_AB" if values.get("d_AB") is not None else ""
    height_symbol = "ell" if values.get("ell") is not None else "h" if values.get("h") is not None else "d"
    if charges is None or not base_symbol or values.get(height_symbol) is None:
        return None
    if "perpendicular bisector" not in text and "trung truc" not in text and "trung trực" not in text:
        return None
    target = _target_from_terms(semantic_output, "E_net_magnitude")
    test_charge = _test_charge_symbol(values)
    target_kind = "force" if _is_force_target(target, text) and test_charge is not None else "field"
    target = target if _is_identifier(target) and target != "result" else ("F_net_magnitude" if target_kind == "force" else "E_net_magnitude")
    knowns = {**charges[0], base_symbol: float(values[base_symbol]), height_symbol: float(values[height_symbol]), "k": 9e9}
    if target_kind == "force" and test_charge is not None:
        knowns[test_charge[0]] = test_charge[1]
    return _point_charge_vector_solution(
        formula_ids=["electrostatics.perpendicular_bisector_geometry"],
        target_symbol=target,
        target_unit=_target_unit_from_semantics(semantic_output, "N" if target_kind == "force" else "N/C"),
        coordinate_equations=[
            f"Ax = -{base_symbol} / 2",
            "Ay = 0",
            f"Bx = {base_symbol} / 2",
            "By = 0",
            "Mx = 0",
            f"My = {height_symbol}",
        ],
        known_values=knowns,
        source_charges=charges[1],
        point_x="Mx",
        point_y="My",
        target_kind=target_kind,
        test_charge_symbol=test_charge[0] if target_kind == "force" and test_charge is not None else None,
        steps=[
            "Normalize the perpendicular-bisector geometry to Cartesian coordinates.",
            "Compute signed Coulomb field components from both source charges.",
            "Add components before computing the field or force magnitude.",
        ],
    )


def _triangle_ab_point_vector_solution(semantic_output: dict[str, Any], values: dict[str, float], text: str) -> dict[str, Any] | None:
    charges = _two_source_charges(values)
    point_symbols = _triangle_point_symbols(values)
    ab = values.get("AB") or values.get("a")
    if charges is None or point_symbols is None or ab is None or ab <= 0:
        return None
    point_name, from_a, from_b = point_symbols
    target = _target_from_terms(semantic_output, "E_net_magnitude")
    test_charge = _test_charge_symbol(values)
    target_kind = "force" if _is_force_target(target, text) and test_charge is not None else "field"
    target = target if _is_identifier(target) and target != "result" else ("F_net_magnitude" if target_kind == "force" else "E_net_magnitude")
    knowns = {**charges[0], "AB": float(ab), from_a: float(values[from_a]), from_b: float(values[from_b]), "k": 9e9}
    if target_kind == "force" and test_charge is not None:
        knowns[test_charge[0]] = test_charge[1]
    px = f"{point_name}x"
    py = f"{point_name}y"
    return _point_charge_vector_solution(
        formula_ids=["electrostatics.triangle_distance_geometry"],
        target_symbol=target,
        target_unit=_target_unit_from_semantics(semantic_output, "N" if target_kind == "force" else "N/C"),
        coordinate_equations=[
            "Ax = 0",
            "Ay = 0",
            "Bx = AB",
            "By = 0",
            f"{px} = ({from_a}**2 + AB**2 - {from_b}**2) / (2 * AB)",
            f"{py} = sqrt({from_a}**2 - {px}**2)",
        ],
        known_values=knowns,
        source_charges=charges[1],
        point_x=px,
        point_y=py,
        target_kind=target_kind,
        test_charge_symbol=test_charge[0] if target_kind == "force" and test_charge is not None else None,
        steps=[
            "Normalize the triangle from side lengths by placing A and B on the x-axis.",
            "Use the law of cosines to compute the target point coordinates.",
            "Apply the reusable Coulomb vector component equations.",
        ],
    )


def _right_angle_vertex_force_solution(semantic_output: dict[str, Any], values: dict[str, float], text: str) -> dict[str, Any] | None:
    if "right angle" not in text and "right-angle" not in text and "vuong" not in text and "vuông" not in text:
        return None
    if "force" not in text:
        return None
    q_value = values.get("q") or values.get("q1")
    leg = values.get("a") or values.get("AB") or values.get("BC") or values.get("AC")
    if q_value is None or leg is None or leg <= 0:
        return None
    target = _target_from_terms(semantic_output, "F_net_magnitude")
    target = target if _is_identifier(target) and target != "result" else "F_net_magnitude"
    return _point_charge_vector_solution(
        formula_ids=["electrostatics.right_angle_vertex_geometry"],
        target_symbol=target,
        target_unit=_target_unit_from_semantics(semantic_output, "N"),
        coordinate_equations=[
            "Px = 0",
            "Py = 0",
            "Ax = a",
            "Ay = 0",
            "Bx = 0",
            "By = a",
        ],
        known_values={"q": float(q_value), "a": float(leg), "k": 9e9},
        source_charges=[("q", "Ax", "Ay"), ("q", "Bx", "By")],
        point_x="Px",
        point_y="Py",
        target_kind="force",
        test_charge_symbol="q",
        steps=[
            "Normalize the right-angle vertex as the target point at the origin.",
            "Place the other two equal charges on perpendicular axes at the leg length.",
            "Add force components generated through the shared field-vector engine.",
        ],
    )


def _resultant_two_vectors_solution(semantic_output: dict[str, Any], values: dict[str, float], text: str) -> dict[str, Any] | None:
    f1 = values.get("F1")
    f2 = values.get("F2")
    theta = values.get("theta")
    target = _target_from_terms(semantic_output, "F_resultant")
    if f1 is None or f2 is None or theta is None:
        return None
    if not (_target_is(target, "F", "F_net", "F_total", "F_resultant", "R") or "resultant" in text):
        return None
    solved_target = target if _is_identifier(target) and target != "result" else "F_resultant"
    angle_equation = "theta_rad = theta * pi / 180" if abs(theta) > 2 * math.pi or "degree" in text or "°" in text else "theta_rad = theta"
    return _solution(
        ["vectors.resultant_two_vectors_law_of_cosines"],
        solved_target,
        _target_unit_from_semantics(semantic_output, "N"),
        [angle_equation, f"{solved_target} = sqrt(F1**2 + F2**2 + 2 * F1 * F2 * cos(theta_rad))"],
        {"F1": float(f1), "F2": float(f2), "theta": float(theta)},
        ["Use the law of cosines with the angle converted to radians before applying SymPy trigonometry."],
    )


def _electric_solution(semantic_output: dict[str, Any], text: str) -> dict[str, Any] | None:
    domain = str(semantic_output.get("domain") or "").lower()
    geometry = semantic_output.get("geometry") or {}
    geometry_type = str(geometry.get("type") or "").lower() if isinstance(geometry, dict) else ""
    if "electric" not in domain and "charge" not in domain:
        return None
    values = _numeric_values(semantic_output)
    right_angle = _right_angle_vertex_force_solution(semantic_output, values, text)
    if right_angle is not None:
        return right_angle
    perpendicular = _perpendicular_bisector_vector_solution(semantic_output, values, text)
    if perpendicular is not None:
        return perpendicular
    triangle = _triangle_ab_point_vector_solution(semantic_output, values, text)
    if triangle is not None:
        return triangle
    # Handle AC = BC isosceles geometry before any generic triangle heuristics.
    if "ac = bc" in text or "ac=bc" in text:
        target = _target_from_terms(semantic_output, "E")
        q_value = values.get("q") or (values.get("q1") if values.get("q1") == values.get("q2") else None)
        ab = values.get("AB") or values.get("a")
        ac = values.get("AC")
        if q_value is not None and ab is not None and ac is not None and ac > 0 and ab > 0:
            solved_target = target if _is_identifier(target) and target != "result" else "E_C"
            return _solution(
                ["electrostatics.isosceles_point_field"],
                solved_target,
                _target_unit_from_semantics(semantic_output, "N/C"),
                [
                    "h = sqrt(AC**2 - (AB/2)**2)",
                    "r = AC",
                    "E_single = k * Abs(q) / r**2",
                    "cos_alpha = h / r",
                    f"{solved_target} = 2 * E_single * cos_alpha",
                ],
                {"q": float(q_value), "AB": float(ab), "AC": float(ac), "k": 9e9},
                [
                    "By symmetry, horizontal components cancel and vertical components add.",
                    "Compute field from one charge, then multiply its projection by 2.",
                ],
            )
    # Only use equilateral branch when explicitly stated in text.
    if "equilateral triangle" in text:
        return _electric_equilateral_solution(semantic_output)
    dielectric = _dielectric_point_charge_solution(semantic_output, text)
    if dielectric is not None:
        return dielectric
    midpoint_identical = _identical_midpoint_zero_field_solution(semantic_output, text)
    if midpoint_identical is not None:
        return midpoint_identical
    if "midpoint" in geometry_type or "midpoint" in text:
        return _electric_midpoint_solution(semantic_output)
    unknown_charge_solution = _zero_field_unknown_charge_solution(semantic_output, text)
    if unknown_charge_solution is not None:
        return unknown_charge_solution
    if "field is zero" in text or "electric field is zero" in text or "zero-field" in text:
        return _zero_field_solution(semantic_output)
    target = _target_from_terms(semantic_output, "E")

    # Field magnitude from force on a charge: E = F/|q|
    target = _target_from_terms(semantic_output, "E")
    q_value = _lookup_value(values, "q", "q0")
    f_value = _lookup_value(values, "F", "F_e", "F_electric")
    if (
        q_value is not None
        and f_value is not None
        and (_target_is(target, "E", "E_net", "E_total") or "electric field" in text)
        and ("force" in text or "experiences" in text)
    ):
        solved_target = target if _is_identifier(target) and target != "result" else "E"
        return _solution(
            ["electrostatics.field_from_force_on_charge"],
            solved_target,
            _target_unit_from_semantics(semantic_output, "N/C"),
            [f"{solved_target} = F / Abs(q)"],
            {"F": float(f_value), "q": float(q_value)},
            [
                "Use the relation between electric force and field magnitude: F = |q|*E.",
                "Rearrange to E = F/|q| and substitute the known force and charge.",
            ],
        )
    # Point-charge magnitude from measured field in air/vacuum: |q| = E*r^2/k.
    r_value = _lookup_value(values, "r", "AM", "AC", "d")
    e_value = _lookup_value(values, "E", "E_M")
    if (
        q_value is None
        and r_value is not None
        and e_value is not None
        and (_target_is(_target_from_terms(semantic_output, "q"), "q", "Q") or "find the magnitude of the charge" in text)
        and ("point charge" in text or "in air" in text or "in vacuum" in text)
    ):
        solved_target = _target_from_terms(semantic_output, "q")
        solved_target = solved_target if _is_identifier(solved_target) and solved_target != "result" else "q"
        return _solution(
            ["electrostatics.point_charge_from_field_air"],
            solved_target,
            _target_unit_from_semantics(semantic_output, "C"),
            [f"{solved_target} = E * r**2 / k"],
            {"E": float(e_value), "r": float(r_value), "k": 9e9},
            ["Use E = k*|q|/r^2 and rearrange to |q| = E*r^2/k."],
        )
    return None


def _frequency_multiplier(text: str) -> float | None:
    multiplier_terms = {
        "doubled": 2.0,
        "double": 2.0,
        "twice": 2.0,
        "tripled": 3.0,
        "triple": 3.0,
        "quadrupled": 4.0,
        "quadruple": 4.0,
        "halved": 0.5,
    }
    for term, value in multiplier_terms.items():
        if re.search(rf"\b{term}\b", text):
            return value
    match = re.search(r"\bfrequency\b.*?\b(?:by|to)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:times|x)\b", text)
    if match:
        return float(match.group(1))
    return None


def _is_resonance_context(text: str) -> bool:
    return any(term in text for term in ("resonance", "resonant", "at resonance", "resonate"))


def _is_ac_context(text: str) -> bool:
    return any(term in text for term in ("alternating-current", "ac circuit", "rlc", "reson", "reactance", "impedance", "lc circuit"))


def _two_section_phase_power_solution(semantic_output: dict[str, Any], values: dict[str, float], text: str) -> dict[str, Any] | None:
    target = _target_from_terms(semantic_output, "P")
    if not ("am" in text and "mb" in text and ("90" in text or "quadrature" in text or "out of phase" in text)):
        return None
    if "power" not in text and not target.startswith("P"):
        return None
    u_value = _lookup_value(values, "U")
    if u_value is None:
        return None
    if _target_is(target, "R2") and "R1" in values:
        p_value = _lookup_value(values, "P")
        if p_value is None or p_value == 0:
            return None
        return _solution(
            ["ac.two_section_phase_balanced_power"],
            "R2",
            _target_unit_from_semantics(semantic_output, "Ohm"),
            ["R_total = U**2 / P", "R2 = R_total - R1"],
            {"U": u_value, "P": p_value, "R1": values["R1"]},
            [
                "With LC*omega**2 = 1 and the 90-degree segment-voltage condition, the total impedance is purely resistive.",
                "Use P = U**2 / R_total to find the equivalent resistance.",
                "Then subtract R1 to obtain R2.",
            ],
            assumptions={"circuit_type": "two_section_phase_balanced"},
        )
    if not all(symbol in values for symbol in ("R1", "R2")):
        return None
    target = target if _is_identifier(target) and target != "result" else "P"
    return _solution(
        ["ac.two_section_phase_balanced_power"],
        target,
        _target_unit_from_semantics(semantic_output, "W"),
        ["R_total = R1 + R2", f"{target} = U**2 / R_total"],
        {"U": u_value, "R1": values["R1"], "R2": values["R2"]},
        [
            "Step 1: Analyze the resonance and phase-balance conditions. Since LC*omega**2 = 1, the reactive parts cancel and the segment voltages are in quadrature.",
            "Step 2: Reduce the circuit to its real resistance. With the reactive contributions canceled, the total impedance becomes purely resistive, so R_total = R1 + R2.",
            "Step 3: Write the power relation for the RMS source voltage. The consumed power is P = U**2 / R_total.",
            "Step 4: Substitute the parsed values and solve for the unknown resistance R2.",
        ],
        assumptions={"circuit_type": "two_section_phase_balanced"},
    )


def _resonance_basic_solution(semantic_output: dict[str, Any], values: dict[str, float], text: str) -> dict[str, Any] | None:
    if not _is_resonance_context(text):
        return None
    target = _target_from_terms(semantic_output, "result")
    unit = _target_unit_from_semantics(semantic_output)
    r_value = values.get("R")
    u_value = _lookup_value(values, "U")
    i_value = _lookup_value(values, "I")
    z_value = values.get("Z")

    if ("power" in text or target.startswith("P")) and r_value is not None:
        if u_value is not None:
            target = target if target != "result" else "P"
            return _solution(
                ["ac.resonance.power_from_voltage_resistance"],
                target,
                unit or "W",
                [f"{target} = U**2 / R"],
                {"U": u_value, "R": r_value},
                [
                    "At resonance the series RLC impedance is purely resistive because the inductive and capacitive reactances cancel.",
                    "That leaves only the real resistance R in the power calculation, so P = U**2 / R.",
                    "Substitute the RMS source voltage and the resistance to obtain the power.",
                ],
            )
        if i_value is not None:
            target = target if target != "result" else "P"
            return _solution(
                ["ac.resonance.power_from_current_resistance"],
                target,
                unit or "W",
                [f"{target} = I**2 * R"],
                {"I": i_value, "R": r_value},
                [
                    "At resonance the series RLC impedance is purely resistive because the inductive and capacitive reactances cancel.",
                    "That leaves only the real resistance R in the power calculation, so P = I**2 * R.",
                    "Substitute the RMS current and resistance to obtain the power.",
                ],
            )
    if _target_is(target, "I", "I_rms", "I_effective") and u_value is not None and r_value is not None:
        return _solution(
            ["ac.resonance.current"],
            target,
            unit or "A",
            [f"{target} = U / R"],
            {"U": u_value, "R": r_value},
            [
                "At resonance the net reactance is zero, so the circuit behaves like a pure resistor.",
                "Use Ohm's law for the RMS values: I = U / R.",
                "Substitute the source voltage and resistance to find the current.",
            ],
        )
    if _target_is(target, "U", "U_rms", "V", "V_rms") and i_value is not None and r_value is not None:
        return _solution(
            ["ac.resonance.voltage"],
            target,
            unit or "V",
            [f"{target} = I * R"],
            {"I": i_value, "R": r_value},
            [
                "At resonance the total impedance is purely resistive, so the circuit reduces to a simple resistor.",
                "Use the RMS Ohm law U = I * R.",
                "Substitute the given current and resistance to get the voltage.",
            ],
        )
    if _target_is(target, "Z") and r_value is not None:
        return _solution(
            ["ac.resonance.impedance_equals_resistance"],
            target,
            unit or "Ohm",
            [f"{target} = R"],
            {"R": r_value},
            [
                "At resonance the inductive and capacitive reactances cancel each other.",
                "That leaves the circuit impedance equal to the real resistance.",
                "Therefore Z = R.",
            ],
        )
    if _target_is(target, "R") and z_value is not None:
        return _solution(
            ["ac.resonance.resistance_equals_impedance"],
            "R",
            unit or "Ohm",
            ["R = Z"],
            {"Z": z_value},
            [
                "At resonance the inductive and capacitive reactances cancel, so the circuit behaves as a pure resistor.",
                "In that case the impedance equals the resistance.",
                "Therefore R = Z.",
            ],
        )
    return None


def _resonance_frequency_solution(semantic_output: dict[str, Any], values: dict[str, float], text: str) -> dict[str, Any] | None:
    target = _target_from_terms(semantic_output, "f_res")
    unit = _target_unit_from_semantics(semantic_output)
    l_value = values.get("L")
    c_value = values.get("C")
    operating_f_value = values.get("f") or values.get("frequency")
    resonance_f_value = values.get("f_res") or values.get("f0")
    f_value = operating_f_value if operating_f_value is not None else resonance_f_value
    omega_symbol = "omega" if values.get("omega") is not None else "omega_res" if values.get("omega_res") is not None else "omega_L"
    omega_value = values.get(omega_symbol)
    wants_resonance_formula = _is_resonance_context(text) or any(term in text for term in ("lc circuit", "rlc"))
    semantic_kind = _normalized_label(semantic_output.get("question_kind"))
    wants_angular = "angular" in text or "rad/s" in unit.lower() or _target_is(target, "omega", "omega_res", "omega0", "omega_L")

    if not wants_resonance_formula:
        return None
    if wants_angular and l_value is not None and c_value is not None:
        angular_target = target if _target_is(target, "omega", "omega_res", "omega0", "omega_L") else "omega"
        return _solution(
            ["ac.resonance.angular_frequency"],
            angular_target,
            "rad/s",
            [f"{angular_target} = 1 / sqrt(L * C)"],
            {"L": l_value, "C": c_value},
            [
                "Recognize the LC resonance condition where the inductive and capacitive reactances cancel.",
                "Use the angular resonance relation omega = 1/sqrt(L*C).",
                "Substitute the inductance and capacitance to obtain the angular resonance frequency.",
            ],
        )
    if "yes_no" in semantic_kind and operating_f_value is not None and l_value is not None and c_value is not None:
        return _solution(
            ["ac.resonance.frequency"],
            "f_res",
            unit or "Hz",
            ["f_res = 1 / (2 * pi * sqrt(L * C))"],
            {"L": l_value, "C": c_value, "f": operating_f_value},
            ["Compute the LC resonance frequency and compare it with the applied frequency."],
            answer_type="yes_no",
            decision_spec={
                "computed_symbol": "f_res",
                "expected_symbol": "f",
                "operator": "approximately_equal",
                "tolerance_policy": "significant_figures",
                "answer_if_true": "Yes",
                "answer_if_false": "No",
            },
        )
    if _target_is(target, "f", "f_res", "f0") and l_value is not None and c_value is not None:
        return _solution(
            ["ac.resonance.frequency"],
            target,
            unit or "Hz",
            [f"{target} = 1 / (2 * pi * sqrt(L * C))"],
            {"L": l_value, "C": c_value},
            [
                "Recognize the LC resonance condition where the inductive and capacitive reactances cancel.",
                "Use the frequency relation f = 1/(2*pi*sqrt(L*C)).",
                "Substitute the inductance and capacitance to obtain the resonant frequency.",
            ],
        )
    if _target_is(target, "L") and c_value is not None:
        if f_value is not None:
            return _solution(["ac.resonance.inductance_from_frequency_capacitance"], "L", unit or "H", ["L = 1 / (4 * pi**2 * f**2 * C)"], {"f": f_value, "C": c_value}, ["Rearrange f = 1/(2*pi*sqrt(L*C)) to solve for L."])
        if omega_value is not None:
            return _solution(["ac.resonance.inductance_from_angular_frequency_capacitance"], "L", unit or "H", [f"L = 1 / ({omega_symbol}**2 * C)"], {omega_symbol: omega_value, "C": c_value}, ["Rearrange omega = 1/sqrt(L*C) to solve for L."])
    if _target_is(target, "C") and l_value is not None:
        if f_value is not None:
            return _solution(["ac.resonance.capacitance_from_frequency_inductance"], "C", unit or "F", ["C = 1 / (4 * pi**2 * f**2 * L)"], {"f": f_value, "L": l_value}, ["Rearrange f = 1/(2*pi*sqrt(L*C)) to solve for C."])
        if omega_value is not None:
            return _solution(["ac.resonance.capacitance_from_angular_frequency_inductance"], "C", unit or "F", [f"C = 1 / ({omega_symbol}**2 * L)"], {omega_symbol: omega_value, "L": l_value}, ["Rearrange omega = 1/sqrt(L*C) to solve for C."])
    return None


def _reactance_solution(semantic_output: dict[str, Any], values: dict[str, float], text: str) -> dict[str, Any] | None:
    target = _target_from_terms(semantic_output, "Z")
    unit = _target_unit_from_semantics(semantic_output)
    l_value = values.get("L")
    c_value = values.get("C")
    f_value = values.get("f")
    omega_symbol = "omega" if values.get("omega") is not None else "omega_L"
    omega_value = values.get(omega_symbol)
    xl_value = _lookup_value(values, "XL")
    xc_value = _lookup_value(values, "XC")
    r_value = values.get("R")
    if _target_is(target, "ZL", "Z_L", "XL", "X_L") and l_value is not None:
        if omega_value is not None:
            return _solution(["ac.inductive_reactance"], target, unit or "Ohm", [f"{target} = {omega_symbol} * L"], {omega_symbol: omega_value, "L": l_value}, ["Compute inductive reactance from angular frequency and inductance."])
        if f_value is not None:
            return _solution(["ac.inductive_reactance"], target, unit or "Ohm", [f"{target} = 2 * pi * f * L"], {"f": f_value, "L": l_value}, ["Compute inductive reactance from frequency and inductance."])
    if _target_is(target, "ZC", "Z_C", "XC", "X_C") and c_value is not None:
        if omega_value is not None:
            return _solution(["ac.capacitive_reactance"], target, unit or "Ohm", [f"{target} = 1 / ({omega_symbol} * C)"], {omega_symbol: omega_value, "C": c_value}, ["Compute capacitive reactance from angular frequency and capacitance."])
        if f_value is not None:
            return _solution(["ac.capacitive_reactance"], target, unit or "Ohm", [f"{target} = 1 / (2 * pi * f * C)"], {"f": f_value, "C": c_value}, ["Compute capacitive reactance from frequency and capacitance."])
    if _target_is(target, "Z") and r_value is not None and xl_value is not None and xc_value is not None:
        return _solution(["rlc.series.impedance_from_reactances"], "Z", unit or "Ohm", ["Z = sqrt(R**2 + (XL - XC)**2)"], {"R": r_value, "XL": xl_value, "XC": xc_value}, ["Use the series RLC impedance formula with the parsed reactances."], assumptions={"circuit_type": "series_assumed"})
    if _target_is(target, "factor", "frequency_factor", "omega_factor") and xl_value is not None and xc_value is not None:
        return _solution(["ac.resonance.frequency_factor_from_reactances"], target, unit or "dimensionless", [f"{target} = sqrt(XC / XL)"], {"XL": xl_value, "XC": xc_value}, ["Because XL scales with frequency and XC scales inversely, resonance requires multiplier sqrt(XC/XL)."])
    return None


def _frequency_scaled_ac_solution(semantic_output: dict[str, Any], values: dict[str, float], text: str) -> dict[str, Any] | None:
    ratio = _frequency_multiplier(text)
    xl_value = _lookup_value(values, "XL")
    xc_value = _lookup_value(values, "XC")
    u_value = _lookup_value(values, "U")
    r_value = values.get("R")
    if ratio is None or ratio <= 0 or xl_value is None or xc_value is None:
        return None
    target = _target_from_terms(semantic_output, "result")
    unit = _target_unit_from_semantics(semantic_output)
    resistor_voltage_target = target in {"U_R", "V_R", "UR", "VR", "U_R_rms"} or ("voltage" in text and "across r" in text)
    if resistor_voltage_target and target in {"", "U", "V", "result"}:
        target = "U_R"
    scaled_xl = ratio * xl_value
    scaled_xc = xc_value / ratio
    at_resonance_after_scaling = math.isclose(scaled_xl, scaled_xc, rel_tol=1e-9, abs_tol=1e-12)
    common_equations = [f"frequency_ratio = {ratio}", "XL_new = frequency_ratio * XL", "XC_new = XC / frequency_ratio"]
    common_knowns = {"XL": xl_value, "XC": xc_value}
    if _target_is(target, "Z") and r_value is not None:
        return _solution(["rlc.series.frequency_scaled_impedance"], "Z", unit or "Ohm", [*common_equations, "Z = sqrt(R**2 + (XL_new - XC_new)**2)"], {**common_knowns, "R": r_value}, ["Scale XL and XC by the frequency change, then compute the series impedance."], assumptions={"circuit_type": "series_assumed"})
    if _target_is(target, "I", "I_rms", "I_effective") and u_value is not None and r_value is not None:
        return _solution(["rlc.series.frequency_scaled_current"], target, unit or "A", [*common_equations, "Z_new = sqrt(R**2 + (XL_new - XC_new)**2)", f"{target} = U / Z_new"], {**common_knowns, "U": u_value, "R": r_value}, ["Scale the reactances by the frequency change and apply I = U/Z."], assumptions={"circuit_type": "series_assumed"})
    if resistor_voltage_target and u_value is not None:
        if r_value is not None:
            return _solution(["rlc.series.frequency_scaled_resistor_voltage"], target, unit or "V", [*common_equations, "Z_new = sqrt(R**2 + (XL_new - XC_new)**2)", f"{target} = U * R / Z_new"], {**common_knowns, "U": u_value, "R": r_value}, ["Scale the reactances by the frequency change, compute Z, then use U_R = U*R/Z."], assumptions={"circuit_type": "series_assumed"})
        if at_resonance_after_scaling:
            return _solution(["ac.resonance.resistor_voltage_equals_source_voltage"], target, unit or "V", [*common_equations, f"{target} = U"], {**common_knowns, "U": u_value}, ["After the frequency change XL equals XC, so all source RMS voltage is across R."], assumptions={"circuit_type": "series_assumed"})
    if ("power" in text or target.startswith("P")) and u_value is not None and r_value is not None:
        target = target if target != "result" else "P"
        return _solution(["rlc.series.frequency_scaled_power"], target, unit or "W", [*common_equations, "Z_new = sqrt(R**2 + (XL_new - XC_new)**2)", "U_R = U * R / Z_new", f"{target} = U_R**2 / R"], {**common_knowns, "U": u_value, "R": r_value}, ["Scale the reactances by the frequency change, find resistor voltage, then compute power."], assumptions={"circuit_type": "series_assumed"})
    return None


def _resonance_shifted_reactance_solution(semantic_output: dict[str, Any], values: dict[str, float], text: str) -> dict[str, Any] | None:
    target = _target_from_terms(semantic_output, "ZL")
    if not (_target_is(target, "ZL", "Z_L", "XL", "X_L", "ZC", "Z_C", "XC", "X_C") or "inductive reactance" in text or "capacitive reactance" in text):
        return None
    if not _is_resonance_context(text):
        return None
    r_value = values.get("R")
    frequency_items = _numeric_sequence_items(semantic_output, "f", "f_res", "f_new", "f0")
    current_items = _numeric_sequence_items(semantic_output, "I", "I_rms", "I_res", "I_new", "I0")
    if len(frequency_items) < 2 or len(current_items) < 2 or r_value is None:
        return None
    f_res_name, f_res = frequency_items[0]
    f_new_name, f_new = frequency_items[-1]
    i_res_name, i_res = current_items[0]
    i_new_name, i_new = current_items[-1]
    if f_new_name == f_res_name:
        f_new_name = f"{f_new_name}_{len(frequency_items)}"
    if i_new_name == i_res_name:
        i_new_name = f"{i_new_name}_{len(current_items)}"
    if f_res == 0:
        return None
    target = target if _is_identifier(target) and target != "result" else "ZL"
    target_equation = f"{target} = frequency_ratio * X_res"
    formula_id = "ac.resonance_shift.inductive_reactance"
    if _target_is(target, "ZC", "Z_C", "XC", "X_C") or "capacitive reactance" in text:
        target_equation = f"{target} = X_res / frequency_ratio"
        formula_id = "ac.resonance_shift.capacitive_reactance"
    knowns = {"R": r_value, f_res_name: f_res, f_new_name: f_new, i_res_name: i_res, i_new_name: i_new}
    return _solution(
        [formula_id],
        target,
        _target_unit_from_semantics(semantic_output, "Ohm"),
        [
            f"frequency_ratio = {f_new_name} / {f_res_name}",
            f"U = {i_res_name} * R",
            f"Z_new = U / {i_new_name}",
            "X_net_new = sqrt(Z_new**2 - R**2)",
            "X_res = X_net_new / Abs(frequency_ratio - 1 / frequency_ratio)",
            target_equation,
        ],
        knowns,
        [
            "At resonance XL equals XC and the source voltage is I_res*R.",
            "Use the changed-frequency current to find the new impedance and net reactance.",
            "Scale XL directly with frequency and XC inversely with frequency.",
        ],
        assumptions={"circuit_type": "series_assumed"},
    )


def _ac_solution(semantic_output: dict[str, Any], values: dict[str, float], text: str) -> dict[str, Any] | None:
    if not _is_ac_context(text):
        return None
    # Special case: given XL, XC at omega0 and asking resonance at k*omega0.
    if ("kω0" in text or "komega0" in text or "k*omega0" in text or "k omega0" in text) and values.get("XL") is not None and values.get("XC") is not None:
        target = _target_from_terms(semantic_output, "k")
        target = target if _is_identifier(target) and target != "result" else "k"
        return _solution(
            ["ac.resonance.frequency_factor_from_reactances"],
            target,
            _target_unit_from_semantics(semantic_output, "dimensionless"),
            [f"{target} = sqrt(XC / XL)"],
            {"XL": float(values["XL"]), "XC": float(values["XC"])},
            ["At scaled frequency, XL scales by k and XC scales by 1/k; resonance needs k*XL = XC/k."],
        )
    for builder in (
        _two_section_phase_power_solution,
        _resonance_basic_solution,
        _resonance_shifted_reactance_solution,
        _frequency_scaled_ac_solution,
        _resonance_frequency_solution,
        _reactance_solution,
    ):
        solution = builder(semantic_output, values, text)
        if solution is not None:
            return solution
    return None


def _extract_unit_value(text: str, pattern: str, scales: dict[str, float]) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    value = _parse_number_token(match.group("value"))
    if value is None:
        return None
    unit = _normalize_text(match.group("unit")).lower()
    scale = scales.get(unit)
    if scale is None:
        return None
    return value * scale


CAPACITANCE_UNIT_SCALES = {
    "f": 1.0,
    "mf": 1e-3,
    "microf": 1e-6,
    "muf": 1e-6,
    "uf": 1e-6,
    "nf": 1e-9,
    "pf": 1e-12,
}
ENERGY_UNIT_SCALES = {"j": 1.0, "mj": 1e-3, "microj": 1e-6, "muj": 1e-6, "uj": 1e-6}
AREA_UNIT_SCALES = {"m2": 1.0, "m^2": 1.0, "m²": 1.0, "cm2": 1e-4, "cm^2": 1e-4, "cm²": 1e-4, "mm2": 1e-6, "mm^2": 1e-6, "mm²": 1e-6}
LENGTH_UNIT_SCALES = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}


def _extract_capacitance_from_text(question_norm: str) -> float | None:
    unit_pattern = r"(?P<unit>microf|muf|uf|nf|pf|mf|f)\b"
    patterns = (
        rf"\bcapacitance(?:\s+of)?\s*(?:c\s*=\s*)?(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*{unit_pattern}",
        rf"\bc\s*=\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*{unit_pattern}",
    )
    for pattern in patterns:
        value = _extract_unit_value(question_norm, pattern, CAPACITANCE_UNIT_SCALES)
        if value is not None:
            return value
    return None


def _extract_energy_from_text(question_norm: str) -> float | None:
    unit_pattern = r"(?P<unit>microj|muj|uj|mj|j)\b"
    patterns = (
        rf"\b(?:energy|initial energy|stores)\D*?(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*{unit_pattern}",
        rf"\bw\s*=\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*{unit_pattern}",
    )
    for pattern in patterns:
        value = _extract_unit_value(question_norm, pattern, ENERGY_UNIT_SCALES)
        if value is not None:
            return value
    return None


def _extract_area_from_text(question_norm: str) -> float | None:
    unit_pattern = r"(?P<unit>cm2|cm\^2|cm²|mm2|mm\^2|mm²|m2|m\^2|m²)\b"
    patterns = (
        rf"\b(?:area|plate area|cross-sectional area)(?:\s+of)?\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*{unit_pattern}",
        rf"\b(?:a|s)\s*=\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*{unit_pattern}",
    )
    for pattern in patterns:
        value = _extract_unit_value(question_norm, pattern, AREA_UNIT_SCALES)
        if value is not None:
            return value
    return None


def _extract_length_from_text(question_norm: str, *names: str) -> float | None:
    unit_pattern = r"(?P<unit>mm|cm|m)\b"
    name_pattern = "|".join(re.escape(name) for name in names)
    patterns = (
        rf"\b(?:{name_pattern})(?:\s+of)?\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*{unit_pattern}",
        rf"\b(?:{name_pattern})\s*=\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*{unit_pattern}",
    )
    for pattern in patterns:
        value = _extract_unit_value(question_norm, pattern, LENGTH_UNIT_SCALES)
        if value is not None:
            return value
    return None


def _extract_voltage_from_text(question_norm: str) -> float | None:
    match = re.search(
        r"\b(?:voltage|potential difference|u|v)\b.*?(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*v\b",
        question_norm,
        flags=re.IGNORECASE,
    )
    return _parse_number_token(match.group("value")) if match else None


def _capacitance_solution(semantic_output: dict[str, Any], values: dict[str, float], text: str) -> dict[str, Any] | None:
    domain = str(semantic_output.get("domain") or "").lower()
    if "capacitance" not in domain and "capacitor" not in text and "electric potential" not in domain:
        return None
    target = _target_from_terms(semantic_output, "result")
    unit = _target_unit_from_semantics(semantic_output)
    c_value = values.get("C")
    q_value = _lookup_value(values, "Q", "Q1", "Q2")
    u_value = _lookup_value(values, "U")
    question = str(semantic_output.get("question") or "")
    question_norm = _normalize_text(question).lower()
    voltage_symbol = "U" if "U" in values or "U_rms" in values else "V"
    text_capacitance = _extract_capacitance_from_text(question_norm)
    if text_capacitance is not None:
        c_value = text_capacitance
    text_voltage = _extract_voltage_from_text(question_norm)
    if text_voltage is not None:
        u_value = text_voltage
        voltage_symbol = "U"
    text_energy = _extract_energy_from_text(question_norm)
    area_value = _extract_area_from_text(question_norm) or values.get("A") or values.get("S")
    separation_value = _extract_length_from_text(question_norm, "plate separation", "separation", "distance", "d") or values.get("d")

    factor_match = re.search(r"(?:factor of|by a factor of|increases by)\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))", question_norm)
    if (
        ("disconnected" in question_norm or "isolated" in question_norm)
        and ("permittivity" in question_norm or "dielectric" in question_norm)
        and text_energy is not None
        and factor_match
    ):
        factor = float(factor_match.group("value"))
        if factor != 0:
            solved_target = target if _is_identifier(target) and target != "result" else "U_new"
            return _solution(
                ["capacitance.isolated_capacitor_energy_dielectric_factor"],
                solved_target,
                unit or "J",
                [f"U_initial = {text_energy}", f"factor = {factor}", f"{solved_target} = U_initial / factor"],
                {},
                [
                    "For a disconnected capacitor, charge remains constant.",
                    "Since energy at fixed charge is inversely proportional to capacitance, increasing permittivity by a factor reduces energy by that factor.",
                ],
            )

    if ("breakdown" in question_norm or "emax" in question_norm or "maximum charge" in question_norm) and (_target_is(target, "Q", "Q_max", "q") or "charge" in question_norm):
        e_max = _lookup_value(values, "Emax", "E_max", "E")
        if e_max is None:
            e_match = re.search(
                r"(?:emax|e_max|electric field).*?(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*(?:x|\*)\s*10\s*(?:\^|\*\*)?\s*[+-]?\d+)?)\s*v\s*/\s*m",
                question_norm,
            )
            if e_match:
                e_max = _parse_number_token(e_match.group("value"))
        radius = values.get("r") or _extract_length_from_text(question_norm, "radius", "r")
        solved_target = target if _is_identifier(target) and target != "result" else "Q_max"
        if e_max is not None and radius is not None:
            return _solution(
                ["capacitance.parallel_plate_breakdown_charge"],
                solved_target,
                unit or "C",
                ["A = pi * r**2", f"{solved_target} = epsilon_0 * E_max * A"],
                {"r": float(radius), "E_max": float(e_max)},
                [
                    "At air breakdown, maximum surface charge density is sigma_max = epsilon_0*E_max.",
                    "The maximum charge is Q_max = sigma_max*A; plate separation does not multiply this expression.",
                ],
            )
        if e_max is not None and area_value is not None:
            return _solution(
                ["capacitance.parallel_plate_breakdown_charge"],
                solved_target,
                unit or "C",
                [f"{solved_target} = epsilon_0 * E_max * A"],
                {"A": float(area_value), "E_max": float(e_max)},
                ["Use Q_max = epsilon_0*E_max*A at air breakdown."],
            )

    if _target_is(target, "C") and text_energy is not None and u_value is not None and u_value != 0:
        requested_micro = "microf" in unit.lower() or "muf" in unit.lower() or "μf" in unit.lower() or "µf" in unit.lower()
        factor = 1e6 if requested_micro else 1.0
        return _solution(
            ["capacitance.from_stored_energy_voltage"],
            "C",
            unit or "F",
            [f"W_eff = {text_energy}", f"U_eff = {u_value}", f"C = 2 * W_eff / U_eff**2{f' * {factor}' if factor != 1.0 else ''}"],
            {},
            ["Use W = C*U**2/2 and solve for capacitance."],
        )

    if _target_is(target, "C") and area_value is not None and separation_value is not None:
        epsilon_r = _lookup_value(values, "epsilon_r", "epsilon", "eps_r") or 1.0
        target_unit = unit or "F"
        equations = []
        knowns: dict[str, float] = {}
        if values.get("A") is not None and math.isclose(float(values["A"]), float(area_value), rel_tol=1e-9, abs_tol=1e-12):
            area_symbol = "A"
            knowns["A"] = float(area_value)
        else:
            area_symbol = "A_eff"
            equations.append(f"A_eff = {area_value}")
        if values.get("d") is not None and math.isclose(float(values["d"]), float(separation_value), rel_tol=1e-9, abs_tol=1e-12):
            distance_symbol = "d"
            knowns["d"] = float(separation_value)
        else:
            distance_symbol = "d_eff"
            equations.append(f"d_eff = {separation_value}")
        if _lookup_value(values, "epsilon_r", "epsilon", "eps_r") is not None:
            epsilon_symbol = "epsilon_r" if values.get("epsilon_r") is not None else "epsilon" if values.get("epsilon") is not None else "eps_r"
            knowns[epsilon_symbol] = float(epsilon_r)
        else:
            epsilon_symbol = "epsilon_r_eff"
            equations.append("epsilon_r_eff = 1")
        equations.append(f"C = epsilon_0 * {epsilon_symbol} * {area_symbol} / {distance_symbol}")
        return _solution(
            ["capacitance.parallel_plate_capacitance"],
            "C",
            target_unit,
            equations,
            knowns,
            ["Use the parallel-plate capacitance relation C = epsilon_0*epsilon_r*A/d."],
        )

    if _target_is(target, "epsilon_r", "epsilon") and c_value is not None and area_value is not None and separation_value is not None:
        solved_target = target if _target_is(target, "epsilon_r", "epsilon") else "epsilon_r"
        equations = []
        knowns = {}
        if values.get("C") is not None and math.isclose(float(values["C"]), float(c_value), rel_tol=1e-9, abs_tol=1e-18):
            capacitance_symbol = "C"
            knowns["C"] = float(c_value)
        else:
            capacitance_symbol = "C_eff"
            equations.append(f"C_eff = {c_value}")
        if values.get("A") is not None and math.isclose(float(values["A"]), float(area_value), rel_tol=1e-9, abs_tol=1e-12):
            area_symbol = "A"
            knowns["A"] = float(area_value)
        else:
            area_symbol = "A_eff"
            equations.append(f"A_eff = {area_value}")
        if values.get("d") is not None and math.isclose(float(values["d"]), float(separation_value), rel_tol=1e-9, abs_tol=1e-12):
            distance_symbol = "d"
            knowns["d"] = float(separation_value)
        else:
            distance_symbol = "d_eff"
            equations.append(f"d_eff = {separation_value}")
        equations.append(f"{solved_target} = {capacitance_symbol} * {distance_symbol} / (epsilon_0 * {area_symbol})")
        return _solution(
            ["capacitance.dielectric_constant_from_geometry"],
            solved_target,
            unit or "",
            equations,
            knowns,
            ["Rearrange C = epsilon_0*epsilon_r*A/d to solve for the dielectric constant."],
        )

    if "parallel" in question_norm and q_value is not None and ("less than" in question_norm or "<" in question_norm):
        capacitances = [(symbol, value) for symbol, value in values.items() if re.fullmatch(r"C\d+", symbol)]
        limit_match = re.search(r"(?:u\s*<|less than)\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*v", question_norm)
        if capacitances and limit_match:
            limit = float(limit_match.group("value"))
            candidates = [(symbol, q_value / capacitance) for symbol, capacitance in capacitances if capacitance != 0]
            valid = [(symbol, voltage) for symbol, voltage in candidates if voltage < limit]
            selected = valid[0] if valid else min(candidates, key=lambda item: abs(item[1] - limit))
            solved_target = target if _target_is(target, "U", "V") else "U"
            return _solution(
                ["capacitance.parallel_capacitor_unknown_charged_branch_voltage"],
                solved_target,
                unit or "V",
                [f"{solved_target} = Q / {selected[0]}"],
                {"Q": q_value, selected[0]: dict(capacitances)[selected[0]]},
                ["In parallel, both capacitors share the same voltage; select the branch consistent with the stated voltage constraint."],
            )

    if "equally shared among two identical capacitors" in question_norm and c_value is not None and u_value is not None:
        solved_target = target if _is_identifier(target) and target != "result" else "E_total"
        return _solution(
            ["capacitance.charge_sharing_two_identical"],
            solved_target,
            unit or "J",
            [
                "Q_initial = C * V",
                "Q_each = Q_initial / 2",
                "V_final = Q_each / C",
                f"{solved_target} = 2 * (C * V_final**2 / 2)",
            ],
            {"C": c_value, "V": u_value},
            [
                "Total charge is conserved and shared equally between two identical capacitors.",
                "Compute the final common voltage from charge per capacitor.",
                "Add the energies of both capacitors to get the remaining total energy.",
            ],
        )

    if "parallel plate capacitor" in question_norm and (_target_is(target, "Q", "q") or "charge" in question_norm):
        s_value = values.get("S") or values.get("A")
        d_value = values.get("d")
        epsilon_r = _lookup_value(values, "epsilon_r", "epsilon", "eps_r")
        if s_value is None:
            area_match = re.search(
                r"\bS\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(cm2|cm\^2|cm²|mm2|mm\^2|mm²|m2|m\^2|m²)",
                question_norm,
                flags=re.IGNORECASE,
            )
            if area_match:
                area_value = float(area_match.group(1))
                area_unit = area_match.group(2).lower()
                area_scale = {"m2": 1.0, "m^2": 1.0, "m²": 1.0, "cm2": 1e-4, "cm^2": 1e-4, "cm²": 1e-4, "mm2": 1e-6, "mm^2": 1e-6, "mm²": 1e-6}
                s_value = area_value * area_scale[area_unit]
        if d_value is None:
            distance_match = re.search(
                r"\bd\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(mm|cm|m)\b",
                question_norm,
                flags=re.IGNORECASE,
            )
            if distance_match:
                distance_scale = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
                d_value = float(distance_match.group(1)) * distance_scale[distance_match.group(2).lower()]
        if epsilon_r is None:
            epsilon_match = re.search(
                r"(?:dielectric constant|epsilon|eps|ε)\s*=?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
                question_norm,
                flags=re.IGNORECASE,
            )
            if epsilon_match:
                epsilon_r = float(epsilon_match.group(1))
        if s_value is not None and d_value is not None and epsilon_r is not None and u_value is not None:
            epsilon_symbol = "epsilon_r" if "epsilon_r" in values else "epsilon" if "epsilon" in values else "epsilon_r"
            solved_target = "Q" if not _target_is(target, "Q", "q") else target
            return _solution(
                ["capacitance.parallel_plate_charge"],
                solved_target,
                unit or "C",
                [f"C_eq = epsilon_0 * {epsilon_symbol} * S / d", f"{solved_target} = C_eq * U"],
                {epsilon_symbol: epsilon_r, "S": s_value, "d": d_value, "U": u_value},
                [
                    "Compute the parallel-plate capacitance C = epsilon_0*epsilon_r*S/d.",
                    "Then use Q = C*U to obtain the plate charge magnitude.",
                ],
            )

    if (
        ("maximum" in question_norm or "max" in question_norm)
        and "sin(" in question_norm
        and c_value is not None
        and ("energy" in question_norm or _target_is(target, "W", "E", "W_max"))
    ):
        amplitude_match = re.search(
            r"(?:u|v)\s*\(t\)\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
            question_norm,
            flags=re.IGNORECASE,
        )
        u_max = float(amplitude_match.group(1)) if amplitude_match else u_value
        if u_max is not None:
            solved_target = target if _is_identifier(target) and target != "result" else "W_max"
            return _solution(
                ["capacitance.stored_energy_max_sinusoidal_voltage"],
                solved_target,
                unit or "J",
                [f"{solved_target} = C * U_max**2 / 2", "U_max = U_amp"],
                {"C": c_value, "U_amp": u_max},
                [
                    "For U(t) = U_max*sin(...), the maximum voltage magnitude is the amplitude U_max.",
                    "Use W_max = C*U_max**2/2 for the maximum stored electric energy.",
                ],
            )

    energy_context = "energy" in question_norm or _target_is(target, "W", "E", "W_max")
    if energy_context and c_value is not None and u_value is not None:
        solved_target = target if _target_is(target, "W", "E", "W_max") else "W"
        return _solution(
            ["capacitance.stored_energy"],
            solved_target,
            unit or "J",
            [f"{solved_target} = C * {voltage_symbol}**2 / 2"],
            {"C": c_value, voltage_symbol: u_value},
            ["Use the capacitor stored-energy relation W = C*U**2/2."],
        )
    if _target_is(target, "Q") and c_value is not None and u_value is not None:
        return _solution(["capacitance.charge"], "Q", unit or "C", [f"Q = C * {voltage_symbol}"], {"C": c_value, voltage_symbol: u_value}, ["Use the capacitor charge relation Q = C*U."])
    if _target_is(target, "C") and q_value is not None and u_value is not None:
        return _solution(["capacitance.capacitance"], "C", unit or "F", [f"C = Q / {voltage_symbol}"], {"Q": q_value, voltage_symbol: u_value}, ["Rearrange Q = C*U to solve for capacitance."])
    if _target_is(target, "U", "V") and q_value is not None and c_value is not None:
        return _solution(["capacitance.voltage"], target, unit or "V", [f"{target} = Q / C"], {"Q": q_value, "C": c_value}, ["Rearrange Q = C*U to solve for voltage."])
    return None


def _measurement_statistics_solution(semantic_output: dict[str, Any], values: dict[str, float], text: str) -> dict[str, Any] | None:
    target = _target_from_terms(semantic_output, "result")
    unit = _target_unit_from_semantics(semantic_output)
    measurement = _measurement_value_and_delta(values, target, "I", "L", "V", "x", "measured_value", "measurement")
    if "maximum possible" in text and measurement is not None:
        symbol, measured_value, delta_value = measurement
        solved_target = target if _is_identifier(target) and target != "result" else f"{symbol}_max"
        return _solution(
            ["measurement.maximum_possible_value"],
            solved_target,
            unit or "",
            [f"{solved_target} = {symbol} + delta_{symbol}"],
            {symbol: measured_value, f"delta_{symbol}": delta_value},
            ["The maximum possible value is the measured value plus its absolute uncertainty."],
        )

    if ("percentage relative uncertainty" in text or "percentage relative error" in text or "relative error" in text) and measurement is not None:
        symbol, measured_value, delta_value = measurement
        if measured_value != 0:
            solved_target = target if _is_identifier(target) and target != "result" else "error_percentage"
            return _solution(
                ["measurement.percentage_relative_uncertainty"],
                solved_target,
                unit or "%",
                [f"{solved_target} = Abs(delta_{symbol} / {symbol}) * 100"],
                {symbol: measured_value, f"delta_{symbol}": delta_value},
                ["Percentage relative uncertainty equals absolute uncertainty divided by the measured value, times 100."],
            )

    if ("least count" in text or "least_count" in values) and ("percentage" in text or "%" in unit):
        least_count = _lookup_value(values, "least_count", "LC")
        measured_value = _lookup_value(values, "measured_value", "measurement", "L")
        if least_count is not None and measured_value is not None and measured_value != 0:
            solved_target = target if _is_identifier(target) and target != "result" else "error_percentage"
            return _solution(
                ["measurement.percentage_error_from_least_count"],
                solved_target,
                unit or "%",
                [f"{solved_target} = Abs((least_count / 2) / measured_value) * 100"],
                {"least_count": float(least_count), "measured_value": float(measured_value)},
                ["For an analog scale, absolute uncertainty is half the least count; percentage error is delta/value*100."],
            )

    if "absolute error of the power" in text:
        v_value = _lookup_value(values, "V", "U")
        i_value = _lookup_value(values, "I")
        delta_v = _lookup_value(values, "delta_V", "delta_U")
        delta_i = _lookup_value(values, "delta_I")
        if all(value is not None for value in (v_value, i_value, delta_v, delta_i)):
            solved_target = "delta_P" if target in {"result", "P", "power"} else target
            return _solution(
                ["measurement.power_error_propagation"],
                solved_target,
                unit or "W",
                ["P = V * I", "delta_P = V * delta_I + I * delta_V"],
                {"V": float(v_value), "I": float(i_value), "delta_V": float(delta_v), "delta_I": float(delta_i)},
                [
                    "Compute nominal power with P = V*I.",
                    "Propagate absolute error for a product: delta_P = V*delta_I + I*delta_V.",
                ],
            )

    if "absolute error" in text and "relative error" in text:
        true_value = _lookup_value(values, "true_value", "x_true")
        measured_value = _lookup_value(values, "measured_result", "x_measured", "measurement")
        if true_value is not None and measured_value is not None and true_value != 0:
            solved_target = target if target in {"absolute_error", "relative_error"} else "relative_error"
            return _solution(
                ["measurement.absolute_and_relative_error"],
                solved_target,
                unit or "",
                ["absolute_error = Abs(true_value - measured_result)", "relative_error = absolute_error / true_value"],
                {"true_value": float(true_value), "measured_result": float(measured_value)},
                [
                    "Absolute error is the absolute difference between true and measured values.",
                    "Relative error equals absolute error divided by the true value.",
                ],
            )

    if "mean absolute error" in text and ("temperature" in text or "measurements" in text):
        measurement_items = _numeric_sequence_items(semantic_output, "x", "T", "temperature")
        if not measurement_items:
            matches = re.findall(
                r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:degc|°c|c)\b",
                _normalize_text(str(semantic_output.get("question") or "")).lower(),
            )
            measurement_items = [(f"x{i + 1}", float(value)) for i, value in enumerate(matches)]
        if len(measurement_items) >= 2:
            symbols = [name for name, _ in measurement_items]
            known_values = {name: float(value) for name, value in measurement_items}
            count = len(symbols)
            mean_expr = " + ".join(symbols)
            mae_terms = " + ".join(f"Abs({name} - mean)" for name in symbols)
            return _solution(
                ["statistics.mean_and_mean_absolute_error"],
                "mean_absolute_error" if target in {"result", "mean_and_mae", ""} else target,
                unit or "",
                [f"mean = ({mean_expr}) / {count}", f"mean_absolute_error = ({mae_terms}) / {count}"],
                known_values,
                [
                    "Compute the arithmetic mean of all measurements.",
                    "Compute mean absolute error as the average absolute deviation from the mean.",
                ],
            )
    return None


def _basic_circuit_solution(semantic_output: dict[str, Any], values: dict[str, float], text: str) -> dict[str, Any] | None:
    target = _target_from_terms(semantic_output, "result")
    unit = _target_unit_from_semantics(semantic_output)
    if "rlc" in text and "impedance" in text and all(symbol in values for symbol in ("R", "L", "C", "f")):
        return _solution(
            ["rlc.series.impedance"],
            "Z",
            "Ohm",
            ["X_L = 2 * pi * f * L", "X_C = 1 / (2 * pi * f * C)", "Z = sqrt(R**2 + (X_L - X_C)**2)"],
            {symbol: values[symbol] for symbol in ("R", "L", "C", "f")},
            ["Assume a series RLC circuit when topology is absent but the dataset convention uses series RLC.", "Compute inductive reactance, capacitive reactance, and series impedance."],
            assumptions={"circuit_type": "series_assumed"},
        )
    if "rms current" in text and all(symbol in values for symbol in ("U", "R1", "R2")):
        return _solution(["ac.rms_current_resistive_equivalent"], "I_rms", "A", ["R_total = R1 + R2", "I_rms = U / R_total"], {"U": values["U"], "R1": values["R1"], "R2": values["R2"]}, ["Use the equivalent resistance condition, then apply RMS Ohm law."])
    if (
        ("parallel" in text or "song song" in text)
        and ("total current" in text or "tong dong dien" in text or "current" in text)
        and all(symbol in values for symbol in ("U", "R1", "R2"))
    ):
        solved_target = target if _target_is(target, "I", "I_total", "I_rms") else "I_total"
        return _solution(
            ["dc.parallel_resistors.total_current"],
            solved_target,
            unit or "A",
            ["R_eq = 1 / (1 / R1 + 1 / R2)", f"{solved_target} = U / R_eq"],
            {"U": values["U"], "R1": values["R1"], "R2": values["R2"]},
            ["For two parallel resistors, compute equivalent resistance, then apply I = U/R_eq."],
        )
    if _target_is(target, "R2") and values.get("Z") is not None and values.get("X_net") is not None:
        return _solution(["ac.resistance_from_impedance_and_net_reactance"], "R2", _target_unit_from_semantics(semantic_output, "Ohm"), ["R2 = sqrt(Z**2 - X_net**2)"], {"Z": values["Z"], "X_net": values["X_net"]}, ["Use only parsed impedance and net reactance values to solve R2."])
    return None


def _magnetism_solution(semantic_output: dict[str, Any], values: dict[str, float], text: str) -> dict[str, Any] | None:
    if "magnetic flux" in text or "flux through" in text:
        b_value = _lookup_value(values, "B")
        area_value = _lookup_value(values, "A", "S") or _extract_area_from_text(_normalize_text(str(semantic_output.get("question") or "")).lower())
        if b_value is not None and area_value is not None:
            target = _target_from_terms(semantic_output, "Phi")
            target = target if _is_identifier(target) and target != "result" else "Phi"
            return _solution(
                ["magnetism.magnetic_flux_uniform_field"],
                target,
                _target_unit_from_semantics(semantic_output, "Wb"),
                [f"{target} = B * A"],
                {"B": float(b_value), "A": float(area_value)},
                ["For a uniform magnetic field perpendicular to the cross-section, magnetic flux is Phi = B*A."],
            )
    if "solenoid" in text and "magnetic field" in text and all(symbol in values for symbol in ("N", "I", "ell")):
        return _solution(["magnetism.solenoid_magnetic_field"], "B", "T", ["B = mu_0 * N * I / ell"], {symbol: values[symbol] for symbol in ("N", "I", "ell")}, ["Use the long-solenoid magnetic-field formula with mu_0 as a physical constant."])
    if "solenoid" in text and "magnetic field" in text:
        n_value = _lookup_value(values, "n", "N_per_m", "turns_per_meter")
        i_value = _lookup_value(values, "I")
        question = _normalize_text(str(semantic_output.get("question") or "")).lower()
        if n_value is None:
            n_match = re.search(r"(?:turns per meter|turn per meter|n)\s*(?:is|=)?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", question)
            if n_match:
                n_value = float(n_match.group(1))
        if i_value is None:
            i_match = re.search(r"current(?:\s+is|=)?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*a\b", question)
            if i_match:
                i_value = float(i_match.group(1))
        if n_value is not None and i_value is not None:
            return _solution(
                ["magnetism.solenoid_magnetic_field_density"],
                "B",
                "T",
                ["B = mu_0 * n * I"],
                {"n": float(n_value), "I": float(i_value)},
                ["For a long solenoid, B = mu_0 * n * I where n is turns per meter."],
            )
    if "flux linkage" in text and values.get("N") is not None:
        phi_value = _lookup_value(values, "Phi", "phi", "phi_initial", "phi_final")
        if phi_value is not None:
            target = _target_from_terms(semantic_output, "lambda_flux")
            target = target if _is_identifier(target) and target != "result" else "total_flux_linkage"
            return _solution(
                ["magnetism.flux_linkage"],
                target,
                _target_unit_from_semantics(semantic_output, "Wb"),
                [f"{target} = N * Phi"],
                {"N": float(values["N"]), "Phi": float(phi_value)},
                ["Flux linkage equals number of turns times flux per turn: lambda = N*Phi."],
            )
    return None


def _inductance_solution(semantic_output: dict[str, Any], values: dict[str, float], text: str) -> dict[str, Any] | None:
    domain = str(semantic_output.get("domain") or "").lower()
    target = _target_from_terms(semantic_output, "result")
    unit = _target_unit_from_semantics(semantic_output)
    if ("self-inductance" in text or "self inductance" in text or "inductance" in domain) and all(symbol in values for symbol in ("epsilon", "I_initial", "I_final", "delta_t")):
        return _solution(["inductance.self_inductance_from_emf_current_change"], "L_self", "H", ["delta_I = I_final - I_initial", "L_self = Abs(epsilon) * delta_t / Abs(delta_I)"], {symbol: values[symbol] for symbol in ("epsilon", "I_initial", "I_final", "delta_t")}, ["Compute the current change, then use Abs(epsilon) = L*Abs(delta_I/delta_t)."])
    energy_terms = ("magnetic field energy", "energy stored in the inductor", "inductor energy")
    if any(term in text for term in energy_terms) or "inductance" in text:
        l_value = values.get("L")
        i_value = _lookup_value(values, "I", "I_max")
        w_value = _lookup_value(values, "W_B", "W", "W_max", "E")
        question = _normalize_text(str(semantic_output.get("question") or "")).lower()
        if i_value is None:
            current_match = re.search(
                r"(?:current(?:\s+is|\s+reaches|\s+reaches\s+(?:its\s+)?maximum\s+value(?:\s+of)?|=)?\s*)([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*sqrt\s*\(?\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*\)?)?)\s*a\b",
                question,
            )
            if current_match:
                i_value = _parse_number_token(current_match.group(1))
        if w_value is None:
            energy_match = re.search(
                r"energy(?:\s+of)?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(mj|uj|microj|j)\b",
                question,
            )
            if energy_match:
                scale = {"j": 1.0, "mj": 1e-3, "uj": 1e-6, "microj": 1e-6}
                w_value = float(energy_match.group(1)) * scale[energy_match.group(2).lower()]
        if _target_is(target, "L", "L_self", "L_ind") and w_value is not None and i_value is not None and i_value != 0:
            return _solution(
                ["inductance.from_magnetic_energy"],
                "L",
                unit or "H",
                ["L = 2 * W_B / I**2"],
                {"W_B": float(w_value), "I": float(i_value)},
                ["Use the inductor energy relation W_B = (1/2)*L*I**2 and solve for L."],
            )
        if (_target_is(target, "W", "W_B", "W_max", "E") or "maximum magnetic field energy" in text) and l_value is not None and i_value is not None:
            solved_target = target if _target_is(target, "W", "W_B", "W_max", "E") else "W_max"
            if values.get("I") is not None or values.get("I_max") is not None:
                equations = [f"{solved_target} = L * I_max**2 / 2"]
                knowns = {"L": float(l_value), "I_max": float(i_value)}
            else:
                equations = [f"I_max = {float(i_value)}", f"{solved_target} = L * I_max**2 / 2"]
                knowns = {"L": float(l_value)}
            return _solution(
                ["inductance.magnetic_energy"],
                solved_target,
                unit or "J",
                equations,
                knowns,
                ["Use W = (1/2)*L*I^2 with the peak current value."],
            )
    return None


def _faraday_induction_solution(semantic_output: dict[str, Any], values: dict[str, float], text: str) -> dict[str, Any] | None:
    if "faraday" not in text and "magnetic flux" not in text and "induced electromotive force" not in text:
        return None
    if not all(symbol in values for symbol in ("N", "phi_initial", "phi_final", "t")):
        return None
    target = _target_from_terms(semantic_output, "E_ind")
    target = target if _is_identifier(target) else "E_ind"
    return _solution(
        ["magnetism.faraday_induction"],
        target,
        _target_unit_from_semantics(semantic_output, "V"),
        [f"{target} = -N * (phi_final - phi_initial) / t"],
        {symbol: values[symbol] for symbol in ("N", "phi_initial", "phi_final", "t")},
        FARADAY_COT_STEPS.copy(),
    )


def build_solution_cot_steps(steps: list[str], equations: list[str], target: str) -> list[str]:
    if steps:
        return steps
    if not equations:
        return steps
    equation_text = " ".join(equations)
    if any(symbol in equation_text for symbol in ("E_ind", "phi_final", "phi_initial")):
        return FARADAY_COT_STEPS.copy()
    target_equation = next(
        (
            equation
            for equation in equations
            if equation.count("=") == 1 and equation.split("=", 1)[0].strip() == target
        ),
        equations[-1],
    )
    return [
        f"Step 1: Use the verified equation {target_equation}.",
        f"Step 2: Substitute the parsed SI values into the equation and simplify the known terms.",
        f"Step 3: Solve the remaining expression for {target} and report the result with its unit.",
    ]


def deterministic_solution(semantic_output: dict[str, Any]) -> dict[str, Any] | None:
    """Return a trusted formula solution for common parsed physics patterns."""
    values = _numeric_values(semantic_output)
    text = _semantic_text(semantic_output)
    for builder in (
        lambda: _resultant_two_vectors_solution(semantic_output, values, text),
        lambda: _electric_solution(semantic_output, text),
        lambda: _ac_solution(semantic_output, values, text),
        lambda: _capacitance_solution(semantic_output, values, text),
        lambda: _measurement_statistics_solution(semantic_output, values, text),
        lambda: _basic_circuit_solution(semantic_output, values, text),
        lambda: _magnetism_solution(semantic_output, values, text),
        lambda: _inductance_solution(semantic_output, values, text),
        lambda: _faraday_induction_solution(semantic_output, values, text),
        # Keep this restricted fallback last so it only applies when
        # no numeric givens exist and no stronger rule matched.
        lambda: _formula_only_solution(semantic_output, text, values),
    ):
        solution = builder()
        if solution is not None:
            return solution
    return None
