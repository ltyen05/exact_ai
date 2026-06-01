"""Registry entrypoint for deterministic physics formulas."""

from __future__ import annotations

from typing import Any

from .capacitance import capacitance_solution
from .circuits import ac_solution, basic_circuit_solution
from .direct import formula_only_solution
from .electrostatics import electric_solution, resultant_two_vectors_solution
from .magnetism import faraday_induction_solution, inductance_solution, magnetism_solution
from .measurement import measurement_statistics_solution
from .base import RuleFn
from .shared import _numeric_values, _semantic_text

FORMULA_RULES: tuple[RuleFn, ...] = (
    resultant_two_vectors_solution,
    electric_solution,
    ac_solution,
    capacitance_solution,
    measurement_statistics_solution,
    basic_circuit_solution,
    magnetism_solution,
    inductance_solution,
    faraday_induction_solution,
    formula_only_solution,
)


def deterministic_solution(semantic_output: dict[str, Any]) -> dict[str, Any] | None:
    values = _numeric_values(semantic_output)
    text = _semantic_text(semantic_output)

    for rule in FORMULA_RULES:
        result = rule(semantic_output, values, text)
        if result is not None:
            return result

    return None
