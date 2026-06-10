"""Deterministic unit normalization for physics parser outputs.

The parser prompt asks the LLM to convert quantities to SI, but small/local
models can still mis-scale prefixed units such as ``25 μF`` as ``0.025 F``.
This module re-reads explicit assignments in the original question and uses
those values as the trusted source before the parsed JSON reaches the solver.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any

from agents.formatting import as_number
from agents.physics.domain.symbols import canonical_quantity_symbol


_SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SUPERSCRIPT = str.maketrans({"²": "2", "³": "3"})


@dataclass(frozen=True)
class Quantity:
    symbol: str
    si_value: float
    si_unit: str
    raw_value: float
    raw_unit: str


# Scale convention: si_value = raw_value * scale.
UNIT_TO_SI: dict[str, tuple[str, float]] = {
    "": ("", 1.0),
    "dimensionless": ("dimensionless", 1.0),
    "%": ("dimensionless", 0.01),
    "m": ("m", 1.0),
    "cm": ("m", 1e-2),
    "mm": ("m", 1e-3),
    "km": ("m", 1e3),
    "m2": ("m^2", 1.0),
    "cm2": ("m^2", 1e-4),
    "mm2": ("m^2", 1e-6),
    "s": ("s", 1.0),
    "ms": ("s", 1e-3),
    "us": ("s", 1e-6),
    "hz": ("Hz", 1.0),
    "khz": ("Hz", 1e3),
    "mhz": ("Hz", 1e6),
    "ghz": ("Hz", 1e9),
    "rad/s": ("rad/s", 1.0),
    "v": ("V", 1.0),
    "mv": ("V", 1e-3),
    "uv": ("V", 1e-6),
    "kv": ("V", 1e3),
    "a": ("A", 1.0),
    "ma": ("A", 1e-3),
    "ua": ("A", 1e-6),
    "c": ("C", 1.0),
    "mc": ("C", 1e-3),
    "uc": ("C", 1e-6),
    "nc": ("C", 1e-9),
    "pc": ("C", 1e-12),
    "f": ("F", 1.0),
    "mf": ("F", 1e-3),
    "uf": ("F", 1e-6),
    "microf": ("F", 1e-6),
    "nf": ("F", 1e-9),
    "pf": ("F", 1e-12),
    "h": ("H", 1.0),
    "mh": ("H", 1e-3),
    "uh": ("H", 1e-6),
    "microh": ("H", 1e-6),
    "Ohm": ("Ohm", 1.0),
    "kOhm": ("Ohm", 1e3),
    "MOhm": ("Ohm", 1e6),
    "mOhm": ("Ohm", 1e-3),
    "ohm": ("Ohm", 1.0),
    "kohm": ("Ohm", 1e3),
    "mohm": ("Ohm", 1e-3),
    "megohm": ("Ohm", 1e6),
    "n": ("N", 1.0),
    "mn": ("N", 1e-3),
    "j": ("J", 1.0),
    "mj": ("J", 1e-3),
    "uj": ("J", 1e-6),
    "w": ("W", 1.0),
    "mw": ("W", 1e-3),
    "kw": ("W", 1e3),
    "t": ("T", 1.0),
    "mt": ("T", 1e-3),
    "wb": ("Wb", 1.0),
    "mwb": ("Wb", 1e-3),
    "uwb": ("Wb", 1e-6),
    "n/c": ("N/C", 1.0),
    "v/m": ("V/m", 1.0),
}

_UNIT_WORDS = sorted(
    {
        "N/C", "V/m", "rad/s", "MΩ", "kΩ", "mΩ", "Ω",
        "MOhm", "kOhm", "mOhm", "Ohm",
        "cm²", "mm²", "m²", "cm^2", "mm^2", "m^2",
        "GHz", "MHz", "kHz", "Hz", "mWb", "uWb", "μWb", "Wb",
        "microF", "mF", "uF", "μF", "nF", "pF", "F",
        "microH", "mH", "uH", "μH", "H",
        "microC", "mC", "uC", "μC", "nC", "pC", "C",
        "mV", "uV", "μV", "kV", "V", "mA", "uA", "μA", "A",
        "mN", "N", "mJ", "uJ", "μJ", "J", "mW", "kW", "W",
        "mT", "T", "km", "cm", "mm", "m", "ms", "us", "μs", "s", "%",
    },
    key=len,
    reverse=True,
)
_UNIT_RE = "|".join(re.escape(unit) for unit in _UNIT_WORDS)
_NUMBER_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*(?:x|×|\*)\s*10\s*(?:\^|\*\*)?\s*[+-]?\d+)?"
_ASSIGNMENT_RE = re.compile(
    rf"(?<![A-Za-z0-9_])(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>{_NUMBER_RE})\s*(?P<unit>{_UNIT_RE})(?![A-Za-z0-9_/])",
    flags=re.I,
)
_DISTANCE_RE = re.compile(
    rf"distance\s+from\s+(?P<p1>[A-Z])\s+to\s+(?P<p2>[A-Z])\s+(?:is|=)\s*(?P<value>{_NUMBER_RE})\s*(?P<unit>{_UNIT_RE})(?![A-Za-z0-9_/])",
    flags=re.I,
)
_FREQUENCY_RE = re.compile(
    rf"(?P<value>{_NUMBER_RE})\s*(?P<unit>GHz|MHz|kHz|Hz|rad/s)(?![A-Za-z0-9_/])",
    flags=re.I,
)
_RESONANCE_YES_NO_RE = re.compile(
    r"\b(?:does|do|is|are|whether)\b.*\bresonan|\bresonance\s+occur|\bresonant\s+frequency",
    flags=re.I,
)


def clean_unit(unit: Any) -> str:
    text = str(unit or "").strip().translate(_SUPERSCRIPT)
    text = text.replace("μ", "u").replace("µ", "u")
    text = text.replace("Ω", "Ohm").replace("Ω", "Ohm")
    text = re.sub(r"ohms?", "Ohm", text, flags=re.I)
    return text.replace(" ", "").replace("^2", "2")


def lookup_unit(unit: Any) -> tuple[str, float] | None:
    cleaned = clean_unit(unit)
    return UNIT_TO_SI.get(cleaned) or UNIT_TO_SI.get(cleaned.lower())


def normalize_text(text: str) -> str:
    return (
        str(text or "")
        .translate(_SUBSCRIPT_DIGITS)
        .replace("μ", "u")
        .replace("µ", "u")
        .replace("Ω", "Ohm")
        .replace("Ω", "Ohm")
        .replace("−", "-")
        .replace("–", "-")
    )


def make_quantity(symbol: str, value_text: str, unit_text: str) -> Quantity | None:
    value = as_number(value_text.replace("×", "x"))
    unit = lookup_unit(unit_text)
    if value is None or unit is None:
        return None
    si_unit, scale = unit
    return Quantity(
        symbol=canonical_quantity_symbol(symbol),
        si_value=value * scale,
        si_unit=si_unit,
        raw_value=value,
        raw_unit=clean_unit(unit_text),
    )


def extract_quantities(question: str) -> dict[str, Quantity]:
    text = normalize_text(question)
    found: dict[str, Quantity] = {}
    for match in _ASSIGNMENT_RE.finditer(text):
        quantity = make_quantity(match.group("symbol"), match.group("value"), match.group("unit"))
        if quantity is not None:
            found.setdefault(quantity.symbol, quantity)
    for match in _DISTANCE_RE.finditer(text):
        symbol = f"{match.group('p1').upper()}{match.group('p2').upper()}"
        quantity = make_quantity(symbol, match.group("value"), match.group("unit"))
        if quantity is not None:
            found.setdefault(quantity.symbol, quantity)
            found.setdefault(symbol[::-1], Quantity(symbol[::-1], quantity.si_value, quantity.si_unit, quantity.raw_value, quantity.raw_unit))
    return found


def coerce_value_unit(value: Any, unit: Any) -> tuple[float, str] | None:
    numeric = as_number(value)
    unit_info = lookup_unit(unit)
    if numeric is None or unit_info is None:
        return None
    si_unit, scale = unit_info
    return numeric * scale, si_unit


def normalize_quantity_item(item: Any, extracted: dict[str, Quantity]) -> Any:
    if not isinstance(item, dict):
        return item
    output = dict(item)
    symbol = canonical_quantity_symbol(output.get("symbol"))
    if symbol:
        output["symbol"] = symbol
    matched = extracted.get(symbol)
    if matched is not None:
        output["si_value"] = matched.si_value
        output["si_unit"] = matched.si_unit
        output.setdefault("raw_value", matched.raw_value)
        output.setdefault("raw_unit", matched.raw_unit)
    else:
        source_value = output.get("si_value") if output.get("si_value") is not None else output.get("value")
        source_unit = output.get("unit") or output.get("si_unit")
        coerced = coerce_value_unit(source_value, source_unit)
        if coerced is not None:
            output["si_value"], output["si_unit"] = coerced
    uncertainty = output.get("uncertainty")
    if isinstance(uncertainty, dict):
        source_value = uncertainty.get("si_value") if uncertainty.get("si_value") is not None else uncertainty.get("value")
        source_unit = uncertainty.get("unit") or uncertainty.get("si_unit") or output.get("si_unit")
        coerced = coerce_value_unit(source_value, source_unit)
        if coerced is not None:
            output["uncertainty"] = {**uncertainty, "si_value": coerced[0], "si_unit": coerced[1]}
    return output


def _find_frequency_in_question(question: str) -> Quantity | None:
    text = normalize_text(question)
    matches = list(_FREQUENCY_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    unit = lookup_unit(match.group("unit"))
    value = as_number(match.group("value").replace("×", "x"))
    if unit is None or value is None:
        return None
    si_unit, scale = unit
    symbol = "omega" if si_unit == "rad/s" else "f"
    return Quantity(symbol, value * scale, si_unit, value, clean_unit(match.group("unit")))


def _upsert_given(givens: list[Any], quantity: Quantity) -> list[Any]:
    updated: list[Any] = []
    replaced = False
    for item in givens:
        if isinstance(item, dict) and canonical_quantity_symbol(item.get("symbol")) == quantity.symbol:
            output = dict(item)
            output.update({"symbol": quantity.symbol, "si_value": quantity.si_value, "si_unit": quantity.si_unit})
            output.setdefault("uncertainty", None)
            updated.append(output)
            replaced = True
        else:
            updated.append(item)
    if not replaced:
        updated.append({"symbol": quantity.symbol, "si_value": quantity.si_value, "si_unit": quantity.si_unit, "uncertainty": None})
    return updated


def _normalize_resonance_comparison(output: dict[str, Any], question: str) -> dict[str, Any]:
    domain = str(output.get("domain") or "")
    if domain not in {"Alternating-Current Circuits", "Alternating Current Circuits"}:
        return output
    text = str(question or "")
    if "resonan" not in text.lower():
        return output
    if not _RESONANCE_YES_NO_RE.search(text):
        return output
    frequency = _find_frequency_in_question(text)
    if frequency is None:
        return output

    target_symbol = "omega_res" if frequency.symbol == "omega" else "f_res"
    output["question_kind"] = "yes_no_computational"
    output["target"] = {"symbol": target_symbol, "unit": frequency.si_unit}
    if isinstance(output.get("givens"), list):
        output["givens"] = _upsert_given(output["givens"], frequency)
    else:
        output["givens"] = [
            {"symbol": frequency.symbol, "si_value": frequency.si_value, "si_unit": frequency.si_unit, "uncertainty": None}
        ]
    output["comparison"] = {
        "present": True,
        "computed_quantity_symbol": target_symbol,
        "given_quantity_symbol": frequency.symbol,
        "given_si_value": frequency.si_value,
        "given_si_unit": frequency.si_unit,
    }
    relations = output.get("relations")
    if isinstance(relations, list):
        marker = f"compare resonance with {frequency.symbol} = {frequency.si_value} {frequency.si_unit}"
        if marker not in relations:
            output["relations"] = [*relations, marker]
    return output


def normalize_parser_units(parsed: dict[str, Any], question: str) -> dict[str, Any]:
    output = copy.deepcopy(parsed)
    extracted = extract_quantities(question)

    target = output.get("target")
    if isinstance(target, dict):
        target = dict(target)
        symbol = canonical_quantity_symbol(target.get("symbol"))
        if symbol:
            target["symbol"] = symbol
        unit = lookup_unit(target.get("unit"))
        if unit is not None:
            target["unit"] = unit[0]
        output["target"] = target

    if isinstance(output.get("givens"), list):
        output["givens"] = [normalize_quantity_item(item, extracted) for item in output["givens"]]

    if isinstance(output.get("quantities"), list):
        output["quantities"] = [normalize_quantity_item(item, extracted) for item in output["quantities"]]

    geometry = output.get("geometry")
    if isinstance(geometry, dict):
        geometry = dict(geometry)
        for field in ("segments", "derived_distances"):
            if isinstance(geometry.get(field), list):
                geometry[field] = [normalize_quantity_item(item, extracted) for item in geometry[field]]
        output["geometry"] = geometry

    comparison = output.get("comparison")
    if isinstance(comparison, dict):
        comparison = dict(comparison)
        symbol = canonical_quantity_symbol(comparison.get("given_quantity_symbol"))
        if symbol:
            comparison["given_quantity_symbol"] = symbol
        matched = extracted.get(symbol)
        if matched is not None:
            comparison["given_si_value"] = matched.si_value
            comparison["given_si_unit"] = matched.si_unit
        else:
            source_value = comparison.get("given_si_value") if comparison.get("given_si_value") is not None else comparison.get("given_value")
            source_unit = comparison.get("given_unit") or comparison.get("unit") or comparison.get("given_si_unit")
            coerced = coerce_value_unit(source_value, source_unit)
            if coerced is not None:
                comparison["given_si_value"], comparison["given_si_unit"] = coerced
        output["comparison"] = comparison

    return _normalize_resonance_comparison(output, question)
