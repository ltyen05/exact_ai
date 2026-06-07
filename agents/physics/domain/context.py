"""Build trusted numeric contexts from parsed physics questions."""

from __future__ import annotations

import re
from typing import Any

from .quantities import expand_quantity_aliases, put_quantity


def build_calculation_input(parsed_question: dict[str, Any]) -> dict[str, Any]:
    """Build trusted numeric context from parsed quantities, geometry, and comparison."""
    quantities: dict[str, float] = {}
    raw_quantities = parsed_question.get("quantities") or {}
    if isinstance(raw_quantities, dict):
        for symbol, value in raw_quantities.items():
            put_quantity(quantities, symbol, value)
    elif isinstance(raw_quantities, list):
        for quantity in raw_quantities:
            if isinstance(quantity, dict):
                value = quantity.get("si_value")
                put_quantity(quantities, quantity.get("symbol"), quantity.get("value") if value is None else value)

    generic_uncertainties: list[Any] = []
    for given in parsed_question.get("givens") or []:
        if not isinstance(given, dict):
            continue
        value = given.get("si_value")
        put_quantity(quantities, given.get("symbol"), given.get("value") if value is None else value)
        uncertainty = given.get("uncertainty") or {}
        if isinstance(uncertainty, dict):
            uncertainty_value = uncertainty.get("si_value")
            if uncertainty_value is None:
                uncertainty_value = uncertainty.get("value")
            symbol = str(given.get("symbol") or "").strip()
            put_quantity(quantities, f"delta_{symbol}", uncertainty_value)
            generic_uncertainties.append(uncertainty_value)
    if len(generic_uncertainties) == 1:
        put_quantity(quantities, "uncertainty", generic_uncertainties[0])
        put_quantity(quantities, "absolute_uncertainty", generic_uncertainties[0])

    geometry = parsed_question.get("geometry") or {}
    if isinstance(geometry, dict):
        for field in ("segments", "derived_distances"):
            for item in geometry.get(field) or []:
                if isinstance(item, dict):
                    value = item.get("si_value")
                    put_quantity(quantities, item.get("symbol"), item.get("value") if value is None else value)

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
        put_quantity(quantities, comparison_symbol, comparison_value)

    raw_target = parsed_question.get("target") or parsed_question.get("objective") or ""
    if isinstance(raw_target, dict):
        target = str(raw_target.get("symbol") or "")
        unit = str(raw_target.get("unit") or "")
    else:
        target = str(raw_target)
        unit = str(parsed_question.get("unit") or parsed_question.get("objective_unit") or "")
    return {"quantities": expand_quantity_aliases(quantities), "target": target, "unit": unit}
