"""Shared physics symbol normalization."""

from __future__ import annotations

import re
from typing import Any

SYMBOL_REPLACEMENTS = {
    "ℓ": "ell",
    "φ": "phi",
    "Φ": "Phi",
    "θ": "theta",
    "ω": "omega",
    "Ω": "Ohm",
    "ε": "epsilon_",
    "μ": "mu",
    "µ": "mu",
    "λ": "lambda_",
    "−": "-",
    "–": "-",
}
QUANTITY_ALIAS_GROUPS = (
    ("U", "V", "U0", "V0", "U_initial", "V_initial", "U_rms", "V_rms", "voltage"),
    ("I", "I_rms", "I_effective"),
    ("I_max", "I_peak", "I_amplitude", "maximum_current", "peak_current", "current_amplitude"),
    ("f", "frequency"),
    ("XL", "X_L", "ZL", "Z_L"),
    ("XC", "X_C", "ZC", "Z_C"),
    ("lambda_", "lambda", "flux_linkage"),
    ("ell", "l", "length"),
    ("Q_max", "Qmax", "qmax", "q_max", "maximum_charge"),
    ("side", "side_length", "a", "AB", "triangle_side"),
    ("W_C", "electric_energy", "E_elec", "capacitor_energy"),
    ("W_L", "W_B", "U_B", "magnetic_energy", "magnetic_field_energy", "E_magn", "inductor_energy"),
    ("W_total", "E_total", "total_energy"),
    ("epsilon_r", "er", "eps_r", "relative_permittivity"),
    ("C", "C0", "C_initial", "C_air", "capacitance"),
)


def _normalize_text(value: str) -> str:
    normalized = value
    for source, replacement in SYMBOL_REPLACEMENTS.items():
        normalized = normalized.replace(source, replacement)
    return normalized


def canonical_quantity_symbol(symbol: Any) -> str:
    """Map parser/LLM symbol aliases to the internal physics symbol name."""
    name = _normalize_text(str(symbol or "")).strip()
    if name.startswith("delta_"):
        inner = canonical_quantity_symbol(name[len("delta_"):])
        if inner in {"V", "V_rms"}:
            inner = "U"
        return f"delta_{inner}" if inner else name
    if re.fullmatch(r"[Qq]_?\d+", name):
        return "q" + re.sub(r"\D", "", name)
    if re.fullmatch(r"charge_[Qq]_?\d+", name):
        return "q" + re.sub(r"\D", "", name)
    if name in {"lambda", "flux_linkage"}:
        return "lambda_"
    if name in {"qmax", "q_max", "Qmax", "maximum_charge"}:
        return "Q_max"
    if name in {"electric_energy", "E_elec", "capacitor_energy"}:
        return "W_C"
    if name in {"W_B", "U_B", "magnetic_energy", "magnetic_field_energy", "E_magn", "inductor_energy"}:
        return "W_L"
    if name in {"total_energy"}:
        return "W_total"
    if name in {"er", "eps_r", "relative_permittivity"}:
        return "epsilon_r"
    if name in {"I_peak", "I_amplitude", "maximum_current", "peak_current", "current_amplitude"}:
        return "I_max"
    if name in {"U_peak", "U_amplitude", "V_peak", "V_amplitude", "maximum_voltage", "peak_voltage", "voltage_amplitude"}:
        return "U_max"
    if name in {"C0", "C_initial", "C_air", "capacitance"}:
        return "C"
    if name in {"U0", "V0", "U_initial", "V_initial", "voltage"}:
        return "U"
    return name
