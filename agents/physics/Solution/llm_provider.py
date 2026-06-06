"""LLM-backed provider for structured physics solution specifications."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import sympy as sp

from agents.formatting import extract_json
from agents.llm import LLMClientBase
from agents.physics.domain.symbols import canonical_quantity_symbol
from agents.physics.validation import solve_spec_structural_error

from .formula_lib import (
    COMPUTATIONAL_MODE_ALIASES,
    CONCEPTUAL_ANSWER_ALIASES,
    DIRECT_MODE_ALIASES,
    MULTIPLE_CHOICE_ANSWER_ALIASES,
    NUMERIC_ANSWER_ALIASES,
    PHYSICAL_CONSTANT_NAMES,
    SYMBOL_ALIAS_GROUPS,
    SYMBOL_REPLACEMENTS,
    YES_NO_ANSWER_ALIASES,
    deterministic_solution as formula_lib_solution,
)
from .solution_provider import SolutionProvider

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
RAG_HINT_CHAR_LIMIT = 2100
RAG_HINT_ITEM_LIMIT = 3
DETERMINISTIC_HINT_CHAR_LIMIT = 2600


def _normalize_text(value: str) -> str:
    normalized = value
    for source, replacement in SYMBOL_REPLACEMENTS.items():
        normalized = normalized.replace(source, replacement)
    return normalized


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(_normalize_value(key)): _normalize_value(item) for key, item in value.items()}
    return value


def _clean_symbol_name(symbol: Any) -> str:
    name = _normalize_text(str(symbol or "")).strip()
    name = name.strip("`'\" .,;:")
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def _canonical_symbol_name(symbol: Any) -> str:
    name = canonical_quantity_symbol(_clean_symbol_name(symbol))
    if re.fullmatch(r"q_?\d+", name):
        return "q" + re.sub(r"\D", "", name)
    if re.fullmatch(r"charge_q_?\d+", name):
        return "q" + re.sub(r"\D", "", name)
    if name.startswith("charge_"):
        stripped = name[len("charge_") :]
        return stripped or name
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
        numeric_values = [value for _, value in present if isinstance(value, (int, float))]
        if len(numeric_values) == len(present) and any(
            not math.isclose(float(group_value), float(value), rel_tol=1e-9, abs_tol=1e-12)
            for value in numeric_values[1:]
        ):
            continue
        for alias in group:
            expanded.setdefault(alias, group_value)
    return expanded


class LLMSolutionProvider(SolutionProvider):
    """Generate and validate one computational or direct physics solution."""

    DEFAULT_PROMPT_PATH = _PROJECT_ROOT / "prompts" / "solution_type2.md"
    SYMPY_LOCALS = {
        "Abs": sp.Abs,
        "abs": sp.Abs,
        "Im": sp.im,
        "im": sp.im,
        "Re": sp.re,
        "re": sp.re,
        "acos": sp.acos,
        "conjugate": sp.conjugate,
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

    def __init__(
        self,
        llm_provider: LLMClientBase,
        prompt_path: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.llm_provider = llm_provider
        self.prompt_path = Path(prompt_path) if prompt_path else self.DEFAULT_PROMPT_PATH
        self.config = config or {}
        self.prompt_template = self.prompt_path.read_text(encoding="utf-8")
        self.last_prompt_diagnostics: dict[str, Any] = {}

    @staticmethod
    def _truncate_text(value: Any, max_chars: int) -> str:
        text = _normalize_text(str(value or "")).strip()
        if len(text) <= max_chars:
            return text
        return f"{text[: max_chars - 3].rstrip()}..."

    @staticmethod
    def _numeric_values(semantic_output: dict[str, Any]) -> dict[str, float]:
        values: dict[str, float] = {}
        conflicted_symbols: set[str] = set()
        generic_uncertainties: list[float] = []

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
                        generic_uncertainties.append(float(uncertainty_value))
        if len(generic_uncertainties) == 1:
            put_value("uncertainty", generic_uncertainties[0])
            put_value("absolute_uncertainty", generic_uncertainties[0])
        geometry = semantic_output.get("geometry") or {}
        if isinstance(geometry, dict):
            for field in ("segments", "derived_distances"):
                for item in geometry.get(field) or []:
                    if isinstance(item, dict):
                        put_value(item.get("symbol"), item.get("si_value"))
        comparison = semantic_output.get("comparison") or {}
        if isinstance(comparison, dict) and isinstance(comparison.get("given_si_value"), (int, float)):
            put_value(comparison.get("given_quantity_symbol"), comparison.get("given_si_value"))
        expanded = _expand_symbol_aliases(values)
        for symbol in conflicted_symbols:
            for group in _alias_groups_for(symbol):
                for alias in group:
                    if alias != symbol and alias not in values:
                        expanded.pop(alias, None)
        return expanded

    @classmethod
    def _target_from_semantics(cls, semantic_output: dict[str, Any]) -> str:
        target = semantic_output.get("target") or {}
        if isinstance(target, dict):
            return _clean_symbol_name(target.get("symbol"))
        return _clean_symbol_name(target)

    @staticmethod
    def _normalized_label(value: Any) -> str:
        return re.sub(r"[_\s\-/]+", "_", _normalize_text(str(value or "")).strip().lower()).strip("_")

    @classmethod
    def _normalize_solution_contract(
        cls,
        solution: dict[str, Any],
        semantic_output: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Coerce common LLM schema aliases into the public solution contract."""
        if not isinstance(solution, dict):
            return solution

        mode_label = cls._normalized_label(solution.get("mode"))
        answer_label = cls._normalized_label(solution.get("answer_type"))
        semantic_kind = cls._normalized_label((semantic_output or {}).get("question_kind"))
        answer_format = (semantic_output or {}).get("answer_format") or {}
        requested_form = cls._normalized_label(answer_format.get("requested_form")) if isinstance(answer_format, dict) else ""
        has_sympy = isinstance(solution.get("sympy_spec"), dict)
        has_direct = isinstance(solution.get("direct_answer"), dict)

        mode: str | None = None
        if (
            mode_label in COMPUTATIONAL_MODE_ALIASES
            or "comput" in mode_label
            or "numeric" in mode_label
            or "formula" in mode_label
        ):
            mode = "computational"
        elif (
            mode_label in DIRECT_MODE_ALIASES
            or "concept" in mode_label
            or "direct" in mode_label
            or "qualitative" in mode_label
        ):
            mode = "direct"
        elif has_sympy:
            mode = "computational"
        elif has_direct or "conceptual" in semantic_kind or semantic_kind == "multiple_choice":
            mode = "direct"

        answer_type: str | None = None
        if (
            answer_label in NUMERIC_ANSWER_ALIASES
            or "numeric" in answer_label
            or "number" in answer_label
            or "scalar" in answer_label
        ):
            answer_type = "numeric"
        elif (
            answer_label in YES_NO_ANSWER_ALIASES
            or ("yes" in answer_label and "no" in answer_label)
            or "boolean" in answer_label
        ):
            answer_type = "yes_no"
        elif answer_label in MULTIPLE_CHOICE_ANSWER_ALIASES or (
            "multiple" in answer_label and "choice" in answer_label
        ):
            answer_type = "multiple_choice"
        elif answer_label in CONCEPTUAL_ANSWER_ALIASES or "concept" in answer_label:
            answer_type = "conceptual"

        if answer_type is None:
            if "yes_no" in semantic_kind or requested_form == "yes_no":
                answer_type = "yes_no"
            elif semantic_kind == "multiple_choice" or requested_form == "multiple_choice":
                answer_type = "multiple_choice"
            elif "conceptual" in semantic_kind and mode != "computational":
                answer_type = "conceptual"
            elif has_direct and not has_sympy:
                answer_type = "conceptual"
            elif has_sympy or mode == "computational":
                answer_type = "numeric"

        if mode == "computational" and answer_type not in {"numeric", "yes_no"}:
            answer_type = "yes_no" if "yes_no" in semantic_kind or requested_form == "yes_no" else "numeric"
        if mode == "direct" and answer_type not in {"yes_no", "multiple_choice", "conceptual"}:
            if "yes_no" in semantic_kind or requested_form == "yes_no":
                answer_type = "yes_no"
            elif semantic_kind == "multiple_choice" or requested_form == "multiple_choice":
                answer_type = "multiple_choice"
            else:
                answer_type = "conceptual"

        # Final fallback: infer from structure when all heuristics fail
        if mode is None:
            if has_sympy:
                mode = "computational"
            elif has_direct:
                mode = "direct"
            else:
                # Default to computational for physics questions
                mode = "computational"
        if answer_type is None:
            if mode == "computational":
                answer_type = "numeric"
            elif mode == "direct":
                answer_type = "conceptual"

        if mode is not None:
            solution["mode"] = mode
        if answer_type is not None:
            solution["answer_type"] = answer_type
        return solution

    @classmethod
    def _normalize_target_symbol(
        cls,
        solution: dict[str, Any],
        semantic_output: dict[str, Any],
    ) -> None:
        spec = solution.get("sympy_spec")
        if not isinstance(spec, dict):
            return
        raw_target = str(spec.get("target_symbol") or "").strip()
        cleaned = _clean_symbol_name(raw_target)
        equations = spec.setdefault("equations", [])
        if not isinstance(equations, list):
            return
        semantic_target = cls._target_from_semantics(semantic_output)
        target_symbol = raw_target if _is_identifier(raw_target) else semantic_target if _is_identifier(semantic_target) else cleaned
        if not _is_identifier(target_symbol):
            target_symbol = "result"
        target_expr = _normalize_text(raw_target)
        if target_expr and target_expr != target_symbol:
            lhs_symbols = {
                lhs
                for equation in equations
                if isinstance(equation, str) and equation.count("=") == 1
                for lhs in [equation.split("=", 1)[0].strip()]
                if _is_identifier(lhs)
            }
            if target_symbol not in lhs_symbols:
                equations.append(f"{target_symbol} = {target_expr}")
        lhs_symbols = [
            lhs
            for equation in equations
            if isinstance(equation, str) and equation.count("=") == 1
            for lhs in [equation.split("=", 1)[0].strip()]
            if _is_identifier(lhs)
        ]
        known_symbols = {
            _canonical_symbol_name(symbol)
            for symbol in (spec.get("known_values") or {}).keys()
        }
        if target_symbol not in lhs_symbols and target_symbol not in known_symbols:
            question_text = _normalize_text(str(semantic_output.get("question") or "")).lower()
            if "mean absolute error" in question_text and "mean" in question_text:
                for candidate in ("mean_absolute_error", "mae", "absolute_error", "mean"):
                    if candidate in lhs_symbols or candidate in known_symbols:
                        target_symbol = candidate
                        break
            elif semantic_target in lhs_symbols or semantic_target in known_symbols:
                target_symbol = semantic_target
            elif lhs_symbols:
                target_symbol = lhs_symbols[-1]
        spec["target_symbol"] = target_symbol

    @classmethod
    def _merge_semantic_known_values(
        cls,
        solution: dict[str, Any],
        semantic_output: dict[str, Any],
    ) -> None:
        spec = solution.get("sympy_spec")
        if not isinstance(spec, dict):
            return
        known_values = spec.get("known_values")
        if not isinstance(known_values, dict):
            known_values = {}

        normalized_knowns: dict[str, Any] = {}
        for symbol, value in known_values.items():
            clean_symbol = _canonical_symbol_name(symbol)
            if clean_symbol:
                normalized_knowns[clean_symbol] = value
        for symbol, value in cls._numeric_values(semantic_output).items():
            normalized_knowns.setdefault(symbol, value)
        spec["known_values"] = _expand_symbol_aliases(normalized_knowns)

    @classmethod
    def _normalize_equations(cls, solution: dict[str, Any]) -> None:
        spec = solution.get("sympy_spec")
        if not isinstance(spec, dict):
            return
        equations = spec.get("equations")
        if not isinstance(equations, list):
            return

        normalized: list[str] = []
        for item in equations:
            equation = _normalize_text(str(item or "")).strip()
            if not equation:
                continue
            if equation.count("=") == 0:
                candidate = f"{equation} = 0"
                try:
                    cls._validate_equation(candidate)
                except ValueError:
                    continue
                equation = candidate
            if equation.count("=") != 1:
                continue
            lhs, rhs = equation.split("=", 1)
            lhs_str = lhs.strip()
            lhs_str = re.sub(r"\b[UuVv]\s*\(\s*t\s*\)", "U_inst", lhs_str)
            lhs_str = re.sub(r"\b[Ii]\s*\(\s*t\s*\)", "I_inst", lhs_str)
            lhs = _canonical_symbol_name(lhs_str) if _is_identifier(lhs_str) else lhs_str
            rhs_text = rhs.strip()
            rhs_text = re.sub(r"\b[UuVv]\s*\(\s*t\s*\)", "U_inst", rhs_text)
            rhs_text = re.sub(r"\b[Ii]\s*\(\s*t\s*\)", "I_inst", rhs_text)
            rhs_text = re.sub(r"\blambda\b", "lambda_", rhs_text)
            rhs_text = re.sub(r"\bcharge_q_?(\d+)\b", r"q\1", rhs_text)
            rhs_text = re.sub(r"\bq_(\d+)\b", r"q\1", rhs_text)
            normalized.append(f"{lhs} = {rhs_text}")
        if normalized:
            spec["equations"] = normalized

    @classmethod
    def _semantic_normalize_solution(
        cls,
        solution: dict[str, Any],
        semantic_output: dict[str, Any] | None,
    ) -> dict[str, Any]:
        normalized = _normalize_value(solution)
        if not isinstance(normalized, dict):
            return normalized
        cls._normalize_solution_contract(normalized, semantic_output)
        if semantic_output is None:
            return normalized
        cls._merge_semantic_known_values(normalized, semantic_output)
        cls._normalize_equations(normalized)
        cls._normalize_target_symbol(normalized, semantic_output)
        return normalized

    @classmethod
    def _extract_strategy_hints(cls, value: Any) -> list[str]:
        if isinstance(value, list):
            text = " ".join(str(item) for item in value)
        else:
            text = str(value or "")
        text = _normalize_text(text).replace("\r", "\n")
        parts = [
            part.strip(" -:\n\t")
            for part in re.split(r"(?:\n+|Step\s*\d+\s*[:.]|(?<=[.!?])\s+)", text, flags=re.IGNORECASE)
            if part.strip(" -:\n\t")
        ]
        formulaish = [
            part
            for part in parts
            if any(token in part for token in ("=", "**", "sqrt", "Abs", "sin", "cos", "tan"))
            or any(keyword in part.lower() for keyword in ("coulomb", "faraday", "resonance", "component", "magnitude"))
        ]
        hints = formulaish or parts
        return [cls._truncate_text(hint, 220) for hint in hints[:2]]

    @classmethod
    def _compact_rag_example(cls, item: dict[str, Any]) -> dict[str, Any]:
        compact: dict[str, Any] = {}
        question = item.get("question")
        if question:
            compact["question"] = cls._truncate_text(question, 240)
        answer = item.get("answer") or item.get("final_answer")
        if answer is not None:
            compact["answer"] = cls._truncate_text(answer, 120)
        unit = item.get("unit") or item.get("target_unit")
        if unit:
            compact["unit"] = cls._truncate_text(unit, 40)
        strategy = item.get("strategy")
        if isinstance(strategy, list):
            compact["strategy"] = [cls._truncate_text(step, 220) for step in strategy[:2]]
        else:
            compact["strategy"] = cls._extract_strategy_hints(
                item.get("cot") or item.get("solution") or item.get("explanation") or ""
            )
        return compact

    @classmethod
    def _format_rag_hints(cls, retrieved_examples: list[dict[str, Any]] | None) -> str:
        if not retrieved_examples:
            return "None."
        hints: list[dict[str, Any]] = []
        for item in retrieved_examples[:RAG_HINT_ITEM_LIMIT]:
            if not isinstance(item, dict):
                continue
            compact = cls._compact_rag_example(item)
            if not compact:
                continue
            candidate = hints + [compact]
            encoded = json.dumps(candidate, ensure_ascii=False, default=str)
            if len(encoded) > RAG_HINT_CHAR_LIMIT and hints:
                break
            if len(encoded) > RAG_HINT_CHAR_LIMIT:
                compact["question"] = cls._truncate_text(compact.get("question", ""), 120)
                compact["strategy"] = [
                    cls._truncate_text(step, 120)
                    for step in compact.get("strategy", [])[:1]
                ]
                candidate = [compact]
                encoded = json.dumps(candidate, ensure_ascii=False, default=str)
            hints = candidate
        if not hints:
            return "None."
        encoded = json.dumps(hints, ensure_ascii=False, default=str)
        if len(encoded) <= RAG_HINT_CHAR_LIMIT:
            return encoded
        first = dict(hints[0])
        first["question"] = cls._truncate_text(first.get("question", ""), 120)
        first["strategy"] = [
            cls._truncate_text(step, 120)
            for step in first.get("strategy", [])[:1]
        ]
        encoded = json.dumps([first], ensure_ascii=False, default=str)
        if len(encoded) <= RAG_HINT_CHAR_LIMIT:
            return encoded
        fallback = {"strategy": first.get("strategy", [])[:1]}
        return json.dumps([fallback], ensure_ascii=False, default=str)

    @classmethod
    def _format_deterministic_hints(cls, deterministic_solution: dict[str, Any] | None) -> str:
        if not isinstance(deterministic_solution, dict):
            return "None."
        compact: dict[str, Any] = {}
        formula_ids = deterministic_solution.get("formula_ids")
        if isinstance(formula_ids, list) and formula_ids:
            compact["formula_ids"] = [cls._truncate_text(item, 80) for item in formula_ids[:4]]
        mode = deterministic_solution.get("mode")
        answer_type = deterministic_solution.get("answer_type")
        if mode:
            compact["mode"] = mode
        if answer_type:
            compact["answer_type"] = answer_type
        spec = deterministic_solution.get("sympy_spec")
        if isinstance(spec, dict):
            target_symbol = spec.get("target_symbol")
            target_unit = spec.get("target_unit")
            equations = spec.get("equations")
            known_values = spec.get("known_values")
            compact_spec: dict[str, Any] = {}
            if target_symbol:
                compact_spec["target_symbol"] = cls._truncate_text(target_symbol, 60)
            if target_unit:
                compact_spec["target_unit"] = cls._truncate_text(target_unit, 24)
            if isinstance(equations, list) and equations:
                compact_spec["equations"] = [cls._truncate_text(eq, 180) for eq in equations[:6]]
            if isinstance(known_values, dict) and known_values:
                compact_spec["known_values"] = {
                    cls._truncate_text(symbol, 40): value
                    for symbol, value in list(known_values.items())[:8]
                }
            if compact_spec:
                compact["sympy_spec"] = compact_spec
        steps = deterministic_solution.get("solution_steps")
        if isinstance(steps, list) and steps:
            compact["solution_steps"] = [cls._truncate_text(step, 180) for step in steps[:3]]
        if not compact:
            return "None."
        encoded = json.dumps(compact, ensure_ascii=False, default=str)
        if len(encoded) <= DETERMINISTIC_HINT_CHAR_LIMIT:
            return encoded
        if "sympy_spec" in compact and isinstance(compact["sympy_spec"], dict):
            compact["sympy_spec"].pop("known_values", None)
            encoded = json.dumps(compact, ensure_ascii=False, default=str)
        if len(encoded) <= DETERMINISTIC_HINT_CHAR_LIMIT:
            return encoded
        minimal = {
            "formula_ids": compact.get("formula_ids", []),
            "sympy_spec": {
                "target_symbol": compact.get("sympy_spec", {}).get("target_symbol", ""),
                "equations": compact.get("sympy_spec", {}).get("equations", [])[:3],
            },
        }
        return json.dumps(minimal, ensure_ascii=False, default=str)

    @staticmethod
    def _deterministic_solution(semantic_output: dict[str, Any]) -> dict[str, Any] | None:
        return formula_lib_solution(semantic_output)

    @staticmethod
    def _merge_deterministic_metadata(
        solution: dict[str, Any],
        deterministic_solution: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(solution, dict) or not isinstance(deterministic_solution, dict):
            return solution
        merged = dict(solution)
        for field in ("formula_ids", "assumptions", "vector_spec"):
            if field not in merged and field in deterministic_solution:
                merged[field] = deterministic_solution[field]
        return merged

    def _build_prompt(
        self,
        semantic_output: dict[str, Any],
        retrieved_examples: list[dict[str, Any]] | None = None,
        deterministic_solution: dict[str, Any] | None = None,
    ) -> str:
        rag_hints = self._format_rag_hints(retrieved_examples)
        deterministic_hints = self._format_deterministic_hints(deterministic_solution)
        parsed_question = json.dumps(semantic_output, ensure_ascii=False, default=str)
        prompt = self.prompt_template
        prompt = (
            prompt.replace("{{RAG_HINTS}}", rag_hints)
            .replace("{{DETERMINISTIC_HINTS}}", deterministic_hints)
            .replace("{{PARSED_QUESTION}}", parsed_question)
        )
        if "{{DETERMINISTIC_HINTS}}" in prompt:
            prompt = prompt.replace("{{DETERMINISTIC_HINTS}}", deterministic_hints)
        self.last_prompt_diagnostics = {
            "prompt_chars": len(prompt),
            "rag_chars": 0 if rag_hints == "None." else len(rag_hints),
            "deterministic_chars": 0 if deterministic_hints == "None." else len(deterministic_hints),
            "selected_rule_pack": ["prompt"],
            "used_json_mode": True,
            "used_repair": False,
        }
        return prompt

    def _validated_deterministic_fallback(
        self,
        semantic_output: dict[str, Any] | None,
        deterministic_fallback: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if semantic_output is None or deterministic_fallback is None:
            return None
        return self._validate_solution(self._semantic_normalize_solution(deterministic_fallback, semantic_output))

    @staticmethod
    def _should_use_deterministic_direct(deterministic_solution: dict[str, Any] | None) -> bool:
        if not isinstance(deterministic_solution, dict):
            return False
        formula_ids = [str(item) for item in deterministic_solution.get("formula_ids") or []]
        direct_prefixes = (
            "electrostatics.collinear_test_charge_geometry",
            "electrostatics.equilateral_triangle_force",
            "electrostatics.right_angle_vertex_geometry",
            "electrostatics.perpendicular_bisector_geometry",
            "electrostatics.triangle_distance_geometry",
            "electrostatics.midpoint_geometry",
            "electrostatics.midpoint_identical_charges_cancel",
            "electrostatics.square_center_identical_charges_cancel",
            "electrostatics.identical_charge_from_force",
            "capacitance.battery_connected_dielectric_charge",
            "capacitance.isolated_dielectric_voltage",
            "capacitance.energy_ratio_",
            "capacitance.series_parallel_identical_energy_ratio",
            "lc.energy_equal_split",
            "measurement.power_percentage_relative_error",
            "measurement.power_relative_error",
            "measurement.relative_error_from_least_count",
            "ac.resonance.frequency_factor_from_reactances",
            "rlc.series.power_factor_from_net_reactance",
            "inductance.current_from_magnetic_energy",
            "vectors.resultant_two_vectors_inverse_angle",
            "vectors.resultant_collinear_",
            "vectors.resultant_perpendicular",
        )
        return any(any(formula_id.startswith(prefix) for prefix in direct_prefixes) for formula_id in formula_ids)

    @staticmethod
    def _build_correction_prompt(
        semantic_output: dict[str, Any],
        invalid_output: Any,
        validation_error: str,
        *,
        raw_response: bool,
    ) -> str:
        output_label = "Previous invalid response preview" if raw_response else "Invalid solution"
        output_text = (
            str(invalid_output)
            if raw_response
            else json.dumps(invalid_output, ensure_ascii=False, default=str)
        )
        return (
            "Correct the physics solution specification. Return exactly one complete valid JSON object, "
            "with no markdown, prose, or final numeric calculation. Keep the object compact but complete. "
            "Always include top-level mode and answer_type. For computational questions include "
            "sympy_spec.target_symbol, sympy_spec.target_unit, sympy_spec.equations, "
            "sympy_spec.known_values, and solution_steps. For direct questions include "
            "direct_answer.answer and direct_answer.rationale_steps. Use ASCII SymPy equations, "
            "explicit '*', and plain identifier symbols only. Allowed functions/constants in equations: "
            "Abs, abs, sqrt, sin, cos, tan, atan, diff, exp, log, pi, Im, Re, conjugate, k, k_e, "
            "epsilon_0, mu_0, c. Every RHS symbol must be a parsed known value, an allowed "
            "physical constant/function, or defined by another equation. C, L, f, R, and helper "
            "symbols are valid only when parsed as givens or defined by equations. The target_symbol "
            "must be defined by an equation. For magnitude/strength/intensity answers, make the final "
            "target nonnegative with Abs(...) or a magnitude formula.\n\n"
            f"Validation error:\n{validation_error}\n\n"
            f"Parsed question:\n{json.dumps(semantic_output, ensure_ascii=False, default=str)}\n\n"
            f"{output_label}:\n{output_text}\n\n"
            "JSON:"
        )

    def _repair_solution(
        self,
        semantic_output: dict[str, Any],
        invalid_solution: Any,
        validation_error: str,
    ) -> dict[str, Any]:
        self.last_prompt_diagnostics["used_repair"] = True
        repair_response = self.llm_provider.chat(
            [
                {
                    "role": "user",
                    "content": self._build_correction_prompt(
                        semantic_output,
                        invalid_solution,
                        validation_error,
                        raw_response=False,
                    ),
                }
            ],
            temperature=0.0,
            max_tokens=self.config.get("repair_max_tokens", 4096),
            response_format={"type": "json_object"},
            stage="physics.solution.repair",
        )
        repaired = extract_json(repair_response)
        if not isinstance(repaired, dict):
            raise ValueError(f"{validation_error}; solution repair did not return a JSON object.")
        repaired = self._semantic_normalize_solution(repaired, semantic_output)
        return self._validate_solution(repaired)

    def _retry_solution_json(
        self,
        semantic_output: dict[str, Any],
        response_preview: str,
        validation_error: str,
    ) -> dict[str, Any] | None:
        retry_response = self.llm_provider.chat(
            [
                {
                    "role": "user",
                    "content": self._build_correction_prompt(
                        semantic_output,
                        response_preview,
                        validation_error,
                        raw_response=True,
                    ),
                }
            ],
            temperature=0.0,
            max_tokens=self.config.get("retry_max_tokens", self.config.get("repair_max_tokens", 4096)),
            response_format={"type": "json_object"},
            stage="physics.solution.retry",
        )
        parsed = extract_json(retry_response)
        if not isinstance(parsed, dict):
            return None
        parsed = self._semantic_normalize_solution(parsed, semantic_output)
        return self._validate_solution(parsed)

    def _request_solution(
        self,
        prompt: str,
        semantic_output: dict[str, Any] | None = None,
        deterministic_fallback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = self.llm_provider.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=self.config.get("max_tokens", 4096),
                response_format={"type": "json_object"},
                stage="physics.solution",
            )
        except Exception:
            deterministic = self._validated_deterministic_fallback(semantic_output, deterministic_fallback)
            if deterministic is not None:
                return deterministic
            raise
        parsed = extract_json(response)
        if not isinstance(parsed, dict):
            response_preview = response[:200] if response else "(empty)"
            validation_error = "Physics solution response must be a JSON object."
            if semantic_output is None:
                raise ValueError(
                    f"{validation_error} "
                    f"Raw response preview: {response_preview}"
                )
            try:
                retried = self._retry_solution_json(semantic_output, response_preview, validation_error)
                if retried is not None:
                    return retried
            except ValueError:
                pass
            try:
                return self._repair_solution(
                    semantic_output,
                    {"raw_response_preview": response_preview},
                    validation_error,
                )
            except ValueError as repair_exc:
                deterministic = self._validated_deterministic_fallback(semantic_output, deterministic_fallback)
                if deterministic is not None:
                    return deterministic
                raise ValueError(
                    f"{repair_exc}; original response was not valid JSON. "
                    f"Raw response preview: {response_preview}"
                ) from repair_exc
        parsed = self._semantic_normalize_solution(parsed, semantic_output)
        try:
            return self._validate_solution(parsed)
        except ValueError as exc:
            if semantic_output is None:
                raise
            try:
                response_preview = json.dumps(parsed, ensure_ascii=False, default=str)[:500]
                try:
                    retried = self._retry_solution_json(semantic_output, response_preview, str(exc))
                    if retried is not None:
                        return retried
                except ValueError:
                    pass
                return self._repair_solution(semantic_output, parsed, str(exc))
            except ValueError as repair_exc:
                deterministic = self._validated_deterministic_fallback(semantic_output, deterministic_fallback)
                if deterministic is not None:
                    return deterministic
                raise ValueError(f"{repair_exc}; original validation error: {exc}") from repair_exc

    @classmethod
    def _validate_equation(cls, equation: str) -> None:
        if equation.count("=") != 1:
            raise ValueError(f"Formula must be an equation with one '=': {equation}")
        lhs, rhs = equation.split("=", 1)
        if not lhs.strip() or not rhs.strip():
            raise ValueError(f"Formula must contain two expressions: {equation}")
        names = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", equation))
        locals_map = {
            **cls.SYMPY_LOCALS,
            **{
                name: sp.Symbol(name)
                for name in names
                if name not in cls.SYMPY_LOCALS
            },
        }
        try:
            sp.sympify(lhs.strip(), locals=locals_map)
            sp.sympify(rhs.strip(), locals=locals_map)
        except (TypeError, ValueError, SyntaxError, sp.SympifyError) as exc:
            raise ValueError(f"Formula is not valid SymPy syntax: {equation}") from exc

    @classmethod
    def _validate_dependency_closure(
        cls,
        equations: list[str],
        known_values: dict[str, Any],
        target_symbol: str,
    ) -> None:
        known_symbols = {_canonical_symbol_name(symbol) for symbol in known_values}
        allowed = set(cls.SYMPY_LOCALS) | PHYSICAL_CONSTANT_NAMES | {_canonical_symbol_name(target_symbol)}
        defined: set[str] = set()
        used: set[str] = set()
        implicit_symbols: set[str] = set()
        for equation in equations:
            lhs, rhs = equation.split("=", 1)
            lhs_text = lhs.strip()
            rhs = rhs.strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs_text):
                defined.add(_canonical_symbol_name(lhs_text))
            else:
                lhs_symbols = {_canonical_symbol_name(sym) for sym in IDENTIFIER_RE.findall(lhs_text)}
                rhs_symbols = {_canonical_symbol_name(sym) for sym in IDENTIFIER_RE.findall(rhs)}
                implicit_symbols.update(lhs_symbols | rhs_symbols)
                used.update(lhs_symbols)
                used.update(rhs_symbols)
                continue
            used.update(_canonical_symbol_name(sym) for sym in IDENTIFIER_RE.findall(rhs))
        unresolved = sorted(used - defined - known_symbols - allowed)
        unresolved = [symbol for symbol in unresolved if symbol not in implicit_symbols]
        if unresolved:
            raise ValueError(f"Unresolved symbols: {', '.join(unresolved)}.")
        if target_symbol not in defined and target_symbol not in known_symbols:
            if target_symbol not in used:
                raise ValueError(f"Target symbol {target_symbol} must be defined by an equation or known value.")

    @staticmethod
    def _validate_vector_spec_components(
        solution: dict[str, Any],
        equations: list[str],
        known_values: dict[str, Any],
    ) -> None:
        vector_spec = solution.get("vector_spec")
        if not isinstance(vector_spec, dict):
            return
        component_symbols = vector_spec.get("component_symbols") or []
        if not isinstance(component_symbols, list):
            raise ValueError("vector_spec.component_symbols must be an array.")
        lhs_symbols = {
            _canonical_symbol_name(equation.split("=", 1)[0].strip())
            for equation in equations
            if isinstance(equation, str)
            and equation.count("=") == 1
            and _is_identifier(equation.split("=", 1)[0].strip())
        }
        known_symbols = {_canonical_symbol_name(symbol) for symbol in known_values}
        missing = sorted(
            _canonical_symbol_name(symbol)
            for symbol in component_symbols
            if _canonical_symbol_name(symbol) not in lhs_symbols
            and _canonical_symbol_name(symbol) not in known_symbols
        )
        if missing:
            raise ValueError(
                "vector_spec component symbols must be defined by equations or known values: "
                f"{', '.join(missing)}."
            )

    @classmethod
    def _validate_solution(cls, solution: dict[str, Any]) -> dict[str, Any]:
        solution = cls._normalize_solution_contract(solution)
        mode = solution.get("mode")
        answer_type = solution.get("answer_type")
        if mode == "direct":
            direct = solution.get("direct_answer")
            if answer_type not in {"yes_no", "multiple_choice", "conceptual"} or not isinstance(direct, dict):
                raise ValueError("Direct physics solution must contain a supported direct_answer.")
            if direct.get("answer") is None or not isinstance(direct.get("rationale_steps"), list):
                raise ValueError("Direct physics solution answer/rationale_steps are invalid.")
            return solution
        if mode != "computational" or answer_type not in {"numeric", "yes_no"}:
            raise ValueError("Physics solution must use a supported mode and answer_type.")

        spec = solution.get("sympy_spec")
        if not isinstance(spec, dict):
            raise ValueError("Computational physics solution requires sympy_spec.")
        if not isinstance(spec.get("target_symbol"), str) or not spec["target_symbol"].strip():
            raise ValueError("sympy_spec.target_symbol must be non-empty.")
        if not _is_identifier(spec["target_symbol"]):
            raise ValueError("sympy_spec.target_symbol must be a plain symbol, not an expression.")
        equations = spec.get("equations")
        if not isinstance(equations, list) or not equations:
            raise ValueError("sympy_spec.equations must be a non-empty list.")
        for equation in equations:
            if not isinstance(equation, str):
                raise ValueError("Every physics equation must be a string.")
            cls._validate_equation(equation)
        known_values = spec.get("known_values", {})
        if not isinstance(known_values, dict):
            raise ValueError("sympy_spec.known_values must be an object.")
        structural_error = solve_spec_structural_error(equations, {_canonical_symbol_name(symbol) for symbol in known_values})
        if structural_error:
            raise ValueError(structural_error)
        cls._validate_dependency_closure(equations, known_values, spec["target_symbol"])
        cls._validate_vector_spec_components(solution, equations, known_values)
        if not isinstance(solution.get("solution_steps", []), list):
            raise ValueError("solution_steps must be an array.")
        if answer_type == "yes_no":
            decision = solution.get("decision_spec")
            if not isinstance(decision, dict) or decision.get("operator") != "approximately_equal":
                raise ValueError("Computational yes/no solution requires approximately_equal decision_spec.")
            if decision.get("computed_symbol") != spec["target_symbol"] or not decision.get("expected_symbol"):
                raise ValueError("Computational yes/no decision symbols are invalid.")
        return solution

    @classmethod
    def deterministic_solution(cls, semantic_output: dict[str, Any]) -> dict[str, Any] | None:
        """Return a validated deterministic solution when the parsed question is rule-covered."""
        deterministic = cls._deterministic_solution(semantic_output)
        if deterministic is None:
            return None
        return cls._validate_solution(cls._semantic_normalize_solution(deterministic, semantic_output))

    def get_solution(self, question: str, semantic_output: dict[str, Any]) -> dict[str, Any]:
        """Request and validate one structured solution for the parsed question."""
        del question
        deterministic = self._deterministic_solution(semantic_output)
        if self._should_use_deterministic_direct(deterministic):
            return self._validated_deterministic_fallback(semantic_output, deterministic) or deterministic
        prompt = self._build_prompt(semantic_output, deterministic_solution=deterministic)
        llm_solution = self._request_solution(prompt, semantic_output, deterministic_fallback=deterministic)
        return self._merge_deterministic_metadata(llm_solution, deterministic)

    def repair_solution(
        self,
        question: str,
        semantic_output: dict[str, Any],
        invalid_solution: dict[str, Any],
        validation_error: str,
    ) -> dict[str, Any]:
        """Repair an already selected solution after execution-time validation failed."""
        del question
        return self._repair_solution(semantic_output, _normalize_value(invalid_solution), validation_error)
