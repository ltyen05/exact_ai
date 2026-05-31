"""Shared helpers for deterministic formula rules.

This module re-exports the current helper implementation from the legacy
formula module so new domain rule packs can be introduced without changing
existing behavior.
"""

from __future__ import annotations

from agents.physics.formulas.legacy import (
    _clean_symbol_name,
    _direct,
    _direct_multiple_choice,
    _is_identifier,
    _lookup_value,
    _normalize_text,
    _numeric_sequence_items,
    _numeric_values,
    _parse_number_token,
    _semantic_text,
    _solution,
    _target_from_semantics,
    _target_from_terms,
    _target_is,
    _target_unit_from_semantics,
)

__all__ = [
    "_clean_symbol_name",
    "_direct",
    "_direct_multiple_choice",
    "_is_identifier",
    "_lookup_value",
    "_normalize_text",
    "_numeric_sequence_items",
    "_numeric_values",
    "_parse_number_token",
    "_semantic_text",
    "_solution",
    "_target_from_semantics",
    "_target_from_terms",
    "_target_is",
    "_target_unit_from_semantics",
]
