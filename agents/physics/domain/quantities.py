"""Shared helpers for parsed physics quantities."""

from __future__ import annotations

import math
import re
from typing import Any

from agents.formatting import as_number

from .symbols import QUANTITY_ALIAS_GROUPS, canonical_quantity_symbol


def expand_quantity_aliases(quantities: dict[str, float]) -> dict[str, float]:
    expanded = dict(quantities)
    for group in QUANTITY_ALIAS_GROUPS:
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


def put_quantity(quantities: dict[str, float], symbol: Any, value: Any) -> None:
    name = canonical_quantity_symbol(symbol)
    numeric = as_number(value)
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
