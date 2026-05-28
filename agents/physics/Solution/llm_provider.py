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

from .solution_provider import SolutionProvider

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
PHYSICAL_CONSTANT_NAMES = {"k", "k_e", "epsilon_0", "mu_0", "c"}
SYMBOL_ALIAS_GROUPS = (
    ("U", "V", "U_rms", "V_rms"),
    ("I", "I_rms", "I_effective"),
    ("f", "f_res", "f0", "frequency"),
    ("XL", "X_L", "ZL", "Z_L"),
    ("XC", "X_C", "ZC", "Z_C"),
)
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
SYMBOL_REPLACEMENTS = {
    "ℓ": "ell",
    "φ": "phi",
    "Φ": "Phi",
    "θ": "theta",
    "ω": "omega",
    "Ω": "Ohm",
    "μ": "mu",
    "−": "-",
    "–": "-",
}
RAG_HINT_CHAR_LIMIT = 2100
RAG_HINT_ITEM_LIMIT = 3
RULE_PACKS = {
    "core": """Core formulas and constants:
- Point charge field magnitude: E = Abs(k * q / r**2); k may be known_values constant 9e9.
- Capacitor energy: W = C * U**2 / 2. Series resistors: R_total = R1 + R2. Ohm law: I = U / R.
- Series RLC impedance: X_L = 2*pi*f*L, X_C = 1/(2*pi*f*C), Z = sqrt(R**2 + (X_L - X_C)**2).
- Resonance: f_res = 1 / (2 * pi * sqrt(L * C)).
- Solenoid field: B = mu_0 * N * I / ell. Faraday EMF: E_ind = -N * (phi_final - phi_initial) / t.
- Self-inductance from current change: L_self = Abs(epsilon) * delta_t / Abs(I_final - I_initial).
- Always create helper equations before using helper symbols.""",
    "direct": """Direct mode rules:
- Use direct mode for conceptual and yes_no_conceptual questions.
- direct_answer.answer must be concise and direct; rationale_steps must be a non-empty list.
- Do not create sympy_spec for theory-only questions.""",
    "electric": """Electric field rules:
- For a point charge vector, define distance r first, then components such as Ex = k * q * dx / r**3 and Ey = k * q * dy / r**3.
- For a requested magnitude, combine components with sqrt(Ex**2 + Ey**2) or Abs(...) for one-dimensional fields.
- Use signed q values in component equations; use Abs only for final magnitudes.""",
    "perpendicular_bisector": """Perpendicular-bisector geometry:
- Prefer coordinates: xA = -d_AB / 2, xB = d_AB / 2, yA = 0, yB = 0, xM = 0, yM = ell.
- Define AM and BM with sqrt((xM - xA)**2 + (yM - yA)**2) before components.
- For opposite charges, perpendicular components may cancel and horizontal components may add; still compute from components.""",
    "triangle_geometry": """Geometry helper rules:
- If parsed geometry provides numeric derived_distances, put those symbols in known_values only when they are explicit parsed values.
- If a helper coordinate, side, projection, or radius is not parsed as numeric, define it by equation before use.
- Never leave x1, y1, r1, r2, dx, dy, or similar helpers unresolved.""",
    "induction": """Induction and EMF rules:
- If the question asks magnitude/strength or does not ask for Lenz-law sign, make public target nonnegative with Abs(E_signed) or a magnitude formula.
- Keep signed Faraday equations only when the question explicitly asks direction/sign.""",
    "ac_circuit": """AC circuit rules:
- If an RLC impedance question does not state topology, assume series only when dataset convention allows it and mark assumption circuit_type=series_assumed.
- Series impedance: X_L = 2*pi*f*L, X_C = 1/(2*pi*f*C), Z = sqrt(R**2 + (X_L - X_C)**2).
- For yes/no resonance comparisons, compute f_res and compare against parsed expected frequency with decision_spec.
- decision_spec.computed_symbol must equal target_symbol and expected_symbol must be present in known_values.""",
    "uncertainty": """Uncertainty rules:
- Use parsed numeric central values in known_values.
- If a target asks uncertainty and no explicit propagation variables are parsed, prefer direct conceptual explanation over invented formulas.""",
}


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


def _lookup_value(values: dict[str, float], *symbols: str) -> float | None:
    for symbol in symbols:
        if symbol in values:
            return values[symbol]
        for group in _alias_groups_for(symbol):
            for alias in group:
                if alias in values:
                    return values[alias]
    return None


def _is_close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


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
        "conjugate": sp.conjugate,
        "atan": sp.atan,
        "cos": sp.cos,
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
    def _selected_rule_packs(semantic_output: dict[str, Any]) -> list[tuple[str, str]]:
        domain = str(semantic_output.get("domain", "")).lower()
        question = str(semantic_output.get("question", "")).lower()
        question_kind = str(semantic_output.get("question_kind", "")).lower()
        requested_form = str(
            (semantic_output.get("answer_format") or {}).get("requested_form", "")
        ).lower()
        geometry = semantic_output.get("geometry") or {}
        geometry_type = str(geometry.get("type", "")).lower() if isinstance(geometry, dict) else ""
        joined = json.dumps(semantic_output, ensure_ascii=False, default=str).lower()

        selected: list[str] = ["core"]
        if "conceptual" in question_kind:
            selected.append("direct")
        if any(token in domain for token in ("electric", "charge", "gauss")):
            selected.append("electric")
        if "perpendicular_bisector" in geometry_type or "perpendicular bisector" in joined:
            selected.append("perpendicular_bisector")
        if any(token in geometry_type for token in ("triangle", "collinear", "geometry")):
            selected.append("triangle_geometry")
        if any(token in domain for token in ("alternating-current", "ac circuit", "circuit")):
            selected.append("ac_circuit")
        if any(token in domain for token in ("induction", "electromagnetic")) or "emf" in question:
            selected.append("induction")
        if "magnitude" in requested_form and "induction" not in selected and "emf" in joined:
            selected.append("induction")
        if "uncertainty" in domain or "uncertainty" in question:
            selected.append("uncertainty")

        deduped = list(dict.fromkeys(selected))
        return [(name, RULE_PACKS[name]) for name in deduped]

    @staticmethod
    def _format_rule_packs(rule_packs: list[tuple[str, str]]) -> str:
        return "\n\n".join(f"[{name}]\n{content.strip()}" for name, content in rule_packs)

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

        def put_value(symbol: Any, value: Any) -> None:
            if not isinstance(value, (int, float)):
                return
            clean_symbol = _clean_symbol_name(symbol)
            if not clean_symbol:
                return
            numeric = float(value)
            if clean_symbol in values and not math.isclose(values[clean_symbol], numeric, rel_tol=1e-9, abs_tol=1e-12):
                conflicted_symbols.add(clean_symbol)
            values[clean_symbol] = numeric

        for item in semantic_output.get("givens") or []:
            if isinstance(item, dict):
                put_value(item.get("symbol"), item.get("si_value"))
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
    def _numeric_sequence_items(cls, semantic_output: dict[str, Any], *symbols: str) -> list[tuple[str, float]]:
        accepted = set(symbols)
        for symbol in symbols:
            for group in _alias_groups_for(symbol):
                accepted.update(group)
        values: list[tuple[str, float]] = []
        for item in semantic_output.get("givens") or []:
            if not isinstance(item, dict) or not isinstance(item.get("si_value"), (int, float)):
                continue
            symbol = _clean_symbol_name(item.get("symbol"))
            if symbol in accepted:
                values.append((symbol, float(item["si_value"])))
        return values

    @classmethod
    def _numeric_sequence(cls, semantic_output: dict[str, Any], *symbols: str) -> list[float]:
        return [value for _, value in cls._numeric_sequence_items(semantic_output, *symbols)]

    @classmethod
    def _target_from_semantics(cls, semantic_output: dict[str, Any]) -> str:
        target = semantic_output.get("target") or {}
        if isinstance(target, dict):
            return _clean_symbol_name(target.get("symbol"))
        return _clean_symbol_name(target)

    @staticmethod
    def _target_unit_from_semantics(semantic_output: dict[str, Any], default: str = "") -> str:
        target = semantic_output.get("target") or {}
        if isinstance(target, dict):
            unit = str(target.get("unit") or "").strip()
            if unit:
                return _normalize_text(unit)
        return default

    @staticmethod
    def _semantic_text(semantic_output: dict[str, Any]) -> str:
        parts: list[str] = [
            str(semantic_output.get("question") or ""),
            str(semantic_output.get("domain") or ""),
        ]
        parts.extend(str(relation) for relation in semantic_output.get("relations") or [])
        target = semantic_output.get("target") or {}
        if isinstance(target, dict):
            parts.append(str(target.get("symbol") or ""))
            parts.append(str(target.get("unit") or ""))
        return _normalize_text(" ".join(parts)).lower()

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
        if _is_identifier(raw_target):
            spec["target_symbol"] = raw_target
            return

        semantic_target = cls._target_from_semantics(semantic_output)
        target_symbol = semantic_target if _is_identifier(semantic_target) else cleaned
        if not _is_identifier(target_symbol):
            target_symbol = "result"
        target_expr = _normalize_text(raw_target)
        if target_expr and target_expr != target_symbol:
            lhs_symbols = {
                equation.split("=", 1)[0].strip()
                for equation in equations
                if isinstance(equation, str) and equation.count("=") == 1
            }
            if target_symbol not in lhs_symbols:
                equations.append(f"{target_symbol} = {target_expr}")
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
            clean_symbol = _clean_symbol_name(symbol)
            if clean_symbol:
                normalized_knowns[clean_symbol] = value
        for symbol, value in cls._numeric_values(semantic_output).items():
            normalized_knowns.setdefault(symbol, value)
        spec["known_values"] = _expand_symbol_aliases(normalized_knowns)

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
        cls._normalize_target_symbol(normalized, semantic_output)
        return normalized

    @staticmethod
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

    @classmethod
    def _electric_equilateral_solution(cls, semantic_output: dict[str, Any]) -> dict[str, Any] | None:
        values = cls._numeric_values(semantic_output)
        q1 = values.get("q1")
        q2 = values.get("q2")
        side = values.get("a") or values.get("AB")
        if q1 is None or q2 is None or side is None:
            return None
        return cls._solution(
            ["electrostatics.point_charge_field_vector", "electrostatics.net_field_vector_cartesian"],
            "E_net_magnitude",
            "N/C",
            [
                "Ax = 0",
                "Ay = 0",
                "Bx = a",
                "By = 0",
                "Nx = a / 2",
                "Ny = a * sqrt(3) / 2",
                "dx1 = Nx - Ax",
                "dy1 = Ny - Ay",
                "r1 = sqrt(dx1**2 + dy1**2)",
                "dx2 = Nx - Bx",
                "dy2 = Ny - By",
                "r2 = sqrt(dx2**2 + dy2**2)",
                "E1x = k * q1 * dx1 / r1**3",
                "E1y = k * q1 * dy1 / r1**3",
                "E2x = k * q2 * dx2 / r2**3",
                "E2y = k * q2 * dy2 / r2**3",
                "Ex_net = E1x + E2x",
                "Ey_net = E1y + E2y",
                "E_net_magnitude = sqrt(Ex_net**2 + Ey_net**2)",
            ],
            {"q1": q1, "q2": q2, "a": side, "k": 9e9},
            [
                "Place A at the origin, B on the positive x-axis, and N above AB.",
                "Use the signed vector field formula E = k*q*r_vector/|r_vector|^3 for each charge.",
                "Add x/y components before computing magnitude and direction.",
            ],
            vector_spec={
                "component_symbols": ["Ex_net", "Ey_net"],
                "magnitude_symbol": "E_net_magnitude",
                "direction": "parallel to AB, from A to B when Ex_net is positive and Ey_net is zero",
            },
        )

    @classmethod
    def _electric_midpoint_solution(cls, semantic_output: dict[str, Any]) -> dict[str, Any] | None:
        values = cls._numeric_values(semantic_output)
        q1 = values.get("q1")
        q2 = values.get("q2")
        ab = values.get("AB") or values.get("a")
        if q1 is None or q2 is None or ab is None:
            return None
        return cls._solution(
            ["electrostatics.point_charge_field_vector", "electrostatics.net_field_vector_cartesian"],
            "E_net_magnitude",
            "N/C",
            [
                "Ax = 0",
                "Bx = AB",
                "Mx = AB / 2",
                "dx1 = Mx - Ax",
                "dx2 = Mx - Bx",
                "E1x = k * q1 * dx1 / Abs(dx1)**3",
                "E2x = k * q2 * dx2 / Abs(dx2)**3",
                "E_net_x = E1x + E2x",
                "E_net_magnitude = Abs(E_net_x)",
            ],
            {"q1": q1, "q2": q2, "AB": ab, "k": 9e9},
            [
                "Use a one-dimensional signed vector formula from each charge to the midpoint.",
                "For opposite charges at the midpoint the two field vectors point the same way and add.",
            ],
            vector_spec={
                "component_symbols": ["E_net_x"],
                "magnitude_symbol": "E_net_magnitude",
                "direction": "along AB according to the sign of E_net_x",
            },
        )

    @classmethod
    def _zero_field_solution(cls, semantic_output: dict[str, Any]) -> dict[str, Any] | None:
        values = cls._numeric_values(semantic_output)
        q1 = values.get("q1")
        q2 = values.get("q2")
        ab = values.get("AB") or values.get("a")
        if q1 is None or q2 is None or ab is None:
            return None
        if q1 == 0 or q2 == 0:
            return None
        if q1 * q2 > 0:
            return cls._solution(
                ["electrostatics.zero_field_point_two_charges_1d"],
                "BM",
                "m",
                ["BM = AB * sqrt(Abs(q2)) / (sqrt(Abs(q1)) + sqrt(Abs(q2)))"],
                {"q1": q1, "q2": q2, "AB": ab},
                [
                    "For same-sign charges between A and B, set k*Abs(q1)/AM**2 = k*Abs(q2)/BM**2.",
                    "Use AM + BM = AB, which gives a square-root distance ratio, not a cube-root relation.",
                ],
            )
        abs_q1 = abs(q1)
        abs_q2 = abs(q2)
        if math.isclose(abs_q1, abs_q2, rel_tol=1e-12, abs_tol=0.0):
            return {
                "mode": "direct",
                "answer_type": "conceptual",
                "direct_answer": {
                    "answer": "No finite zero-field point exists for equal-magnitude opposite charges.",
                    "rationale_steps": ["Outside the segment the fields never balance, and between the charges they add."],
                },
            }
        if abs_q1 < abs_q2:
            return cls._solution(
                ["electrostatics.zero_field_point_two_charges_1d"],
                "AM",
                "m",
                ["AM = AB * sqrt(Abs(q1)) / (sqrt(Abs(q2)) - sqrt(Abs(q1)))"],
                {"q1": q1, "q2": q2, "AB": ab},
                ["For opposite-sign charges, the zero-field point is outside the segment on the side of the smaller charge."],
            )
        return cls._solution(
            ["electrostatics.zero_field_point_two_charges_1d"],
            "BM",
            "m",
            ["BM = AB * sqrt(Abs(q2)) / (sqrt(Abs(q1)) - sqrt(Abs(q2)))"],
            {"q1": q1, "q2": q2, "AB": ab},
            ["For opposite-sign charges, the zero-field point is outside the segment on the side of the smaller charge."],
        )

    @staticmethod
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

    @staticmethod
    def _is_resonance_context(text: str) -> bool:
        return any(term in text for term in ("resonance", "resonant", "at resonance", "resonate"))

    @staticmethod
    def _is_ac_context(text: str) -> bool:
        return any(term in text for term in ("alternating-current", "ac circuit", "rlc", "reson", "reactance", "impedance"))

    @staticmethod
    def _target_is(target: str, *symbols: str) -> bool:
        normalized_target = _clean_symbol_name(target)
        normalized = {_clean_symbol_name(symbol) for symbol in symbols}
        return normalized_target in normalized

    @classmethod
    def _target_from_terms(cls, semantic_output: dict[str, Any], fallback: str) -> str:
        target = cls._target_from_semantics(semantic_output)
        return target if _is_identifier(target) else fallback

    @classmethod
    def _resonance_basic_solution(
        cls,
        semantic_output: dict[str, Any],
        values: dict[str, float],
        text: str,
    ) -> dict[str, Any] | None:
        if not cls._is_resonance_context(text):
            return None
        target = cls._target_from_terms(semantic_output, "result")
        unit = cls._target_unit_from_semantics(semantic_output)
        r_value = values.get("R")
        u_value = _lookup_value(values, "U")
        i_value = _lookup_value(values, "I")
        z_value = values.get("Z")

        if ("power" in text or target.startswith("P")) and r_value is not None:
            if u_value is not None:
                return cls._solution(
                    ["ac.resonance.power_from_voltage_resistance"],
                    target if target != "result" else "P",
                    unit or "W",
                    [f"{target if target != 'result' else 'P'} = U**2 / R"],
                    {"U": u_value, "R": r_value},
                    ["At resonance the series RLC impedance is purely resistive, so power is U**2/R."],
                )
            if i_value is not None:
                return cls._solution(
                    ["ac.resonance.power_from_current_resistance"],
                    target if target != "result" else "P",
                    unit or "W",
                    [f"{target if target != 'result' else 'P'} = I**2 * R"],
                    {"I": i_value, "R": r_value},
                    ["At resonance the series RLC impedance is purely resistive, so power is I**2*R."],
                )

        if cls._target_is(target, "I", "I_rms", "I_effective") and u_value is not None and r_value is not None:
            return cls._solution(
                ["ac.resonance.current"],
                target,
                unit or "A",
                [f"{target} = U / R"],
                {"U": u_value, "R": r_value},
                ["At resonance the net reactance is zero, so the effective current is U/R."],
            )

        if cls._target_is(target, "U", "U_rms", "V", "V_rms") and i_value is not None and r_value is not None:
            return cls._solution(
                ["ac.resonance.voltage"],
                target,
                unit or "V",
                [f"{target} = I * R"],
                {"I": i_value, "R": r_value},
                ["At resonance the total impedance is R, so the RMS voltage is I*R."],
            )

        if cls._target_is(target, "Z") and r_value is not None:
            return cls._solution(
                ["ac.resonance.impedance_equals_resistance"],
                target,
                unit or "Ohm",
                [f"{target} = R"],
                {"R": r_value},
                ["At resonance the inductive and capacitive reactances cancel, leaving Z = R."],
            )
        if cls._target_is(target, "R") and z_value is not None:
            return cls._solution(
                ["ac.resonance.resistance_equals_impedance"],
                "R",
                unit or "Ohm",
                ["R = Z"],
                {"Z": z_value},
                ["At resonance the circuit impedance is equal to the pure resistance."],
            )
        return None

    @classmethod
    def _resonance_frequency_solution(
        cls,
        semantic_output: dict[str, Any],
        values: dict[str, float],
        text: str,
    ) -> dict[str, Any] | None:
        target = cls._target_from_terms(semantic_output, "f_res")
        unit = cls._target_unit_from_semantics(semantic_output)
        l_value = values.get("L")
        c_value = values.get("C")
        f_value = values.get("f") or values.get("f_res")
        omega_value = values.get("omega") or values.get("omega_res")
        wants_resonance_formula = cls._is_resonance_context(text) or any(term in text for term in ("lc circuit", "rlc"))

        if not wants_resonance_formula:
            return None
        if cls._target_is(target, "f", "f_res", "f0") and l_value is not None and c_value is not None:
            answer_type = "numeric"
            extra: dict[str, Any] = {}
            comparison = semantic_output.get("comparison") or {}
            expected_symbol = ""
            if isinstance(comparison, dict):
                expected_symbol = _clean_symbol_name(comparison.get("given_quantity_symbol"))
            if not expected_symbol and values.get("f") is not None and target != "f":
                expected_symbol = "f"
            if "yes_no" in cls._normalized_label(semantic_output.get("question_kind")) and expected_symbol:
                answer_type = "yes_no"
                extra["decision_spec"] = {
                    "computed_symbol": target,
                    "expected_symbol": expected_symbol,
                    "operator": "approximately_equal",
                    "tolerance_policy": "significant_figures",
                    "answer_if_true": "Yes",
                    "answer_if_false": "No",
                }
            return cls._solution(
                ["ac.resonance.frequency"],
                target,
                unit or "Hz",
                [f"{target} = 1 / (2 * pi * sqrt(L * C))"],
                {"L": l_value, "C": c_value},
                ["Use the LC resonance relation f = 1/(2*pi*sqrt(L*C))."],
                answer_type=answer_type,
                **extra,
            )

        if cls._target_is(target, "omega", "omega_res", "omega0") and l_value is not None and c_value is not None:
            return cls._solution(
                ["ac.resonance.angular_frequency"],
                target,
                unit or "rad/s",
                [f"{target} = 1 / sqrt(L * C)"],
                {"L": l_value, "C": c_value},
                ["Use the LC resonance relation omega = 1/sqrt(L*C)."],
            )

        if cls._target_is(target, "L") and c_value is not None:
            if f_value is not None:
                return cls._solution(
                    ["ac.resonance.inductance_from_frequency_capacitance"],
                    "L",
                    unit or "H",
                    ["L = 1 / (4 * pi**2 * f**2 * C)"],
                    {"f": f_value, "C": c_value},
                    ["Rearrange f = 1/(2*pi*sqrt(L*C)) to solve for L."],
                )
            if omega_value is not None:
                return cls._solution(
                    ["ac.resonance.inductance_from_angular_frequency_capacitance"],
                    "L",
                    unit or "H",
                    ["L = 1 / (omega**2 * C)"],
                    {"omega": omega_value, "C": c_value},
                    ["Rearrange omega = 1/sqrt(L*C) to solve for L."],
                )

        if cls._target_is(target, "C") and l_value is not None:
            if f_value is not None:
                return cls._solution(
                    ["ac.resonance.capacitance_from_frequency_inductance"],
                    "C",
                    unit or "F",
                    ["C = 1 / (4 * pi**2 * f**2 * L)"],
                    {"f": f_value, "L": l_value},
                    ["Rearrange f = 1/(2*pi*sqrt(L*C)) to solve for C."],
                )
            if omega_value is not None:
                return cls._solution(
                    ["ac.resonance.capacitance_from_angular_frequency_inductance"],
                    "C",
                    unit or "F",
                    ["C = 1 / (omega**2 * L)"],
                    {"omega": omega_value, "L": l_value},
                    ["Rearrange omega = 1/sqrt(L*C) to solve for C."],
                )
        return None

    @classmethod
    def _reactance_solution(
        cls,
        semantic_output: dict[str, Any],
        values: dict[str, float],
        text: str,
    ) -> dict[str, Any] | None:
        target = cls._target_from_terms(semantic_output, "Z")
        unit = cls._target_unit_from_semantics(semantic_output)
        l_value = values.get("L")
        c_value = values.get("C")
        f_value = values.get("f")
        omega_value = values.get("omega")
        xl_value = _lookup_value(values, "XL")
        xc_value = _lookup_value(values, "XC")
        r_value = values.get("R")

        if cls._target_is(target, "ZL", "Z_L", "XL", "X_L") and l_value is not None:
            if omega_value is not None:
                return cls._solution(
                    ["ac.inductive_reactance"],
                    target,
                    unit or "Ohm",
                    [f"{target} = omega * L"],
                    {"omega": omega_value, "L": l_value},
                    ["Compute inductive reactance from angular frequency and inductance."],
                )
            if f_value is not None:
                return cls._solution(
                    ["ac.inductive_reactance"],
                    target,
                    unit or "Ohm",
                    [f"{target} = 2 * pi * f * L"],
                    {"f": f_value, "L": l_value},
                    ["Compute inductive reactance from frequency and inductance."],
                )

        if cls._target_is(target, "ZC", "Z_C", "XC", "X_C") and c_value is not None:
            if omega_value is not None:
                return cls._solution(
                    ["ac.capacitive_reactance"],
                    target,
                    unit or "Ohm",
                    [f"{target} = 1 / (omega * C)"],
                    {"omega": omega_value, "C": c_value},
                    ["Compute capacitive reactance from angular frequency and capacitance."],
                )
            if f_value is not None:
                return cls._solution(
                    ["ac.capacitive_reactance"],
                    target,
                    unit or "Ohm",
                    [f"{target} = 1 / (2 * pi * f * C)"],
                    {"f": f_value, "C": c_value},
                    ["Compute capacitive reactance from frequency and capacitance."],
                )

        if cls._target_is(target, "Z") and r_value is not None and xl_value is not None and xc_value is not None:
            return cls._solution(
                ["rlc.series.impedance_from_reactances"],
                "Z",
                unit or "Ohm",
                ["Z = sqrt(R**2 + (XL - XC)**2)"],
                {"R": r_value, "XL": xl_value, "XC": xc_value},
                ["Use the series RLC impedance formula with the parsed reactances."],
                assumptions={"circuit_type": "series_assumed"},
            )

        if cls._target_is(target, "factor", "frequency_factor", "omega_factor") and xl_value is not None and xc_value is not None:
            return cls._solution(
                ["ac.resonance.frequency_factor_from_reactances"],
                target,
                unit or "dimensionless",
                [f"{target} = sqrt(XC / XL)"],
                {"XL": xl_value, "XC": xc_value},
                ["Because XL scales with frequency and XC scales inversely, resonance requires multiplier sqrt(XC/XL)."],
            )
        return None

    @classmethod
    def _frequency_scaled_ac_solution(
        cls,
        semantic_output: dict[str, Any],
        values: dict[str, float],
        text: str,
    ) -> dict[str, Any] | None:
        ratio = cls._frequency_multiplier(text)
        xl_value = _lookup_value(values, "XL")
        xc_value = _lookup_value(values, "XC")
        u_value = _lookup_value(values, "U")
        r_value = values.get("R")
        if ratio is None or ratio <= 0 or xl_value is None or xc_value is None:
            return None

        target = cls._target_from_terms(semantic_output, "result")
        unit = cls._target_unit_from_semantics(semantic_output)
        resistor_voltage_target = (
            target in {"U_R", "V_R", "UR", "VR"}
            or ("voltage" in text and "across r" in text)
            or ("rms voltage" in text and "across r" in text)
        )
        if resistor_voltage_target and target in {"", "U", "V", "result"}:
            target = "U_R"

        scaled_xl = ratio * xl_value
        scaled_xc = xc_value / ratio
        at_resonance_after_scaling = _is_close(scaled_xl, scaled_xc)
        common_equations = [
            f"frequency_ratio = {ratio}",
            "XL_new = frequency_ratio * XL",
            "XC_new = XC / frequency_ratio",
        ]
        common_knowns = {"XL": xl_value, "XC": xc_value}

        if cls._target_is(target, "Z") and r_value is not None:
            return cls._solution(
                ["rlc.series.frequency_scaled_impedance"],
                "Z",
                unit or "Ohm",
                [*common_equations, "Z = sqrt(R**2 + (XL_new - XC_new)**2)"],
                {**common_knowns, "R": r_value},
                ["Scale XL and XC by the frequency change, then compute the series impedance."],
                assumptions={"circuit_type": "series_assumed"},
            )

        if cls._target_is(target, "I", "I_rms", "I_effective") and u_value is not None:
            if r_value is not None:
                return cls._solution(
                    ["rlc.series.frequency_scaled_current"],
                    target,
                    unit or "A",
                    [*common_equations, "Z_new = sqrt(R**2 + (XL_new - XC_new)**2)", f"{target} = U / Z_new"],
                    {**common_knowns, "U": u_value, "R": r_value},
                    ["Scale the reactances by the frequency change and apply I = U/Z."],
                    assumptions={"circuit_type": "series_assumed"},
                )
            if at_resonance_after_scaling:
                return None

        if resistor_voltage_target and u_value is not None:
            if r_value is not None:
                return cls._solution(
                    ["rlc.series.frequency_scaled_resistor_voltage"],
                    target,
                    unit or "V",
                    [*common_equations, "Z_new = sqrt(R**2 + (XL_new - XC_new)**2)", f"{target} = U * R / Z_new"],
                    {**common_knowns, "U": u_value, "R": r_value},
                    ["Scale the reactances by the frequency change, compute Z, then use U_R = U*R/Z."],
                    assumptions={"circuit_type": "series_assumed"},
                )
            if at_resonance_after_scaling:
                return cls._solution(
                    ["ac.resonance.resistor_voltage_equals_source_voltage"],
                    target,
                    unit or "V",
                    [*common_equations, f"{target} = U"],
                    {**common_knowns, "U": u_value},
                    ["After the frequency change XL equals XC, so all source RMS voltage is across R."],
                    assumptions={"circuit_type": "series_assumed"},
                )

        if ("power" in text or target.startswith("P")) and u_value is not None and r_value is not None:
            return cls._solution(
                ["rlc.series.frequency_scaled_power"],
                target if target != "result" else "P",
                unit or "W",
                [
                    *common_equations,
                    "Z_new = sqrt(R**2 + (XL_new - XC_new)**2)",
                    "U_R = U * R / Z_new",
                    f"{target if target != 'result' else 'P'} = U_R**2 / R",
                ],
                {**common_knowns, "U": u_value, "R": r_value},
                ["Scale the reactances by the frequency change, find the resistor voltage, then compute power."],
                assumptions={"circuit_type": "series_assumed"},
            )
        return None

    @classmethod
    def _resonance_shifted_reactance_solution(
        cls,
        semantic_output: dict[str, Any],
        values: dict[str, float],
        text: str,
    ) -> dict[str, Any] | None:
        target = cls._target_from_terms(semantic_output, "ZL")
        if not (
            cls._target_is(target, "ZL", "Z_L", "XL", "X_L", "ZC", "Z_C", "XC", "X_C")
            or "inductive reactance" in text
            or "capacitive reactance" in text
        ):
            return None
        if not cls._is_resonance_context(text):
            return None

        r_value = values.get("R")
        f_res_name = "f_res"
        f_new_name = "f_new"
        i_res_name = "I_res"
        i_new_name = "I_new"
        f_res = values.get(f_res_name) or values.get("f0")
        f_new = values.get(f_new_name)
        i_res = values.get(i_res_name) or values.get("I0")
        i_new = values.get(i_new_name)
        frequency_items = cls._numeric_sequence_items(semantic_output, "f", "f_res", "f_new", "f0")
        current_items = cls._numeric_sequence_items(semantic_output, "I", "I_rms", "I_res", "I_new", "I0")
        if len(frequency_items) >= 2:
            f_res_name, f_res = frequency_items[0]
            f_new_name, f_new = frequency_items[-1]
            if f_new_name == f_res_name:
                f_new_name = f"{f_new_name}_{len(frequency_items)}"
        elif f_res is None and values.get("f0") is not None:
            f_res_name = "f0"
            f_res = values["f0"]
        if len(current_items) >= 2:
            i_res_name, i_res = current_items[0]
            i_new_name, i_new = current_items[-1]
            if i_new_name == i_res_name:
                i_new_name = f"{i_new_name}_{len(current_items)}"
        elif i_res is None and values.get("I0") is not None:
            i_res_name = "I0"
            i_res = values["I0"]
        ratio = cls._frequency_multiplier(text)
        if ratio is None and f_res not in (None, 0) and f_new is not None:
            ratio = f_new / f_res
        if r_value is None or i_res is None or i_new is None or ratio is None or _is_close(ratio, 1.0):
            return None

        target = target if _is_identifier(target) and target != "result" else "ZL"
        unit = cls._target_unit_from_semantics(semantic_output, "Ohm")
        target_equation = f"{target} = frequency_ratio * X_res"
        formula_id = "ac.resonance_shift.inductive_reactance"
        if cls._target_is(target, "ZC", "Z_C", "XC", "X_C") or "capacitive reactance" in text:
            target_equation = f"{target} = X_res / frequency_ratio"
            formula_id = "ac.resonance_shift.capacitive_reactance"
        knowns = {"R": r_value, i_res_name: i_res, i_new_name: i_new}
        ratio_equation = f"frequency_ratio = {ratio}"
        if f_res is not None and f_new is not None:
            knowns[f_res_name] = f_res
            knowns[f_new_name] = f_new
            ratio_equation = f"frequency_ratio = {f_new_name} / {f_res_name}"
        equations = [
            ratio_equation,
            f"U = {i_res_name} * R",
            f"Z_new = U / {i_new_name}",
            "X_net_new = sqrt(Z_new**2 - R**2)",
            "X_res = X_net_new / Abs(frequency_ratio - 1 / frequency_ratio)",
            target_equation,
        ]
        return cls._solution(
            [formula_id],
            target,
            unit,
            equations,
            knowns,
            [
                "At resonance XL equals XC and the source voltage is I_res*R.",
                "At the changed frequency, use the current to find the new impedance and net reactance.",
                "Scale XL directly with frequency and XC inversely with frequency.",
            ],
            assumptions={"circuit_type": "series_assumed"},
        )

    @classmethod
    def _ac_deterministic_solution(cls, semantic_output: dict[str, Any]) -> dict[str, Any] | None:
        text = cls._semantic_text(semantic_output)
        if not cls._is_ac_context(text):
            return None
        values = cls._numeric_values(semantic_output)
        for builder in (
            cls._resonance_basic_solution,
            cls._resonance_shifted_reactance_solution,
            cls._frequency_scaled_ac_solution,
            cls._resonance_frequency_solution,
            cls._reactance_solution,
        ):
            solution = builder(semantic_output, values, text)
            if solution is not None:
                return solution
        return None

    @classmethod
    def _deterministic_solution(cls, semantic_output: dict[str, Any]) -> dict[str, Any] | None:
        values = cls._numeric_values(semantic_output)
        question = str(semantic_output.get("question") or "").lower()
        domain = str(semantic_output.get("domain") or "").lower()
        geometry = semantic_output.get("geometry") or {}
        geometry_type = str(geometry.get("type") or "").lower() if isinstance(geometry, dict) else ""

        if "electric" in domain or "charge" in domain:
            if "equilateral" in geometry_type or "equilateral triangle" in question:
                return cls._electric_equilateral_solution(semantic_output)
            if "midpoint" in geometry_type or "midpoint" in question:
                return cls._electric_midpoint_solution(semantic_output)
            if "field is zero" in question or "electric field is zero" in question or "zero-field" in question:
                return cls._zero_field_solution(semantic_output)

        ac_solution = cls._ac_deterministic_solution(semantic_output)
        if ac_solution is not None:
            return ac_solution

        if "rlc" in question and "impedance" in question and all(symbol in values for symbol in ("R", "L", "C", "f")):
            return cls._solution(
                ["rlc.series.impedance"],
                "Z",
                "Ohm",
                ["X_L = 2 * pi * f * L", "X_C = 1 / (2 * pi * f * C)", "Z = sqrt(R**2 + (X_L - X_C)**2)"],
                {symbol: values[symbol] for symbol in ("R", "L", "C", "f")},
                ["Assume a series RLC circuit when topology is absent but the dataset convention uses series RLC.", "Compute inductive reactance, capacitive reactance, and series impedance."],
                assumptions={"circuit_type": "series_assumed"},
            )
        if ("impedance" in question or "reactance" in question) and all(symbol in values for symbol in ("L", "C", "omega")):
            target = cls._target_from_semantics(semantic_output) or "Z"
            if target in {"ZL", "X_L", "XL"}:
                return cls._solution(
                    ["ac.inductive_reactance"],
                    "ZL",
                    "Ohm",
                    ["ZL = omega * L"],
                    {"omega": values["omega"], "L": values["L"]},
                    ["Compute inductive reactance from angular frequency and inductance."],
                )
            if target in {"ZC", "X_C", "XC"}:
                return cls._solution(
                    ["ac.capacitive_reactance"],
                    "ZC",
                    "Ohm",
                    ["ZC = 1 / (omega * C)"],
                    {"omega": values["omega"], "C": values["C"]},
                    ["Compute capacitive reactance from angular frequency and capacitance."],
                )
        if "rms current" in question and all(symbol in values for symbol in ("U", "R1", "R2")):
            return cls._solution(
                ["ac.rms_current_resistive_equivalent"],
                "I_rms",
                "A",
                ["R_total = R1 + R2", "I_rms = U / R_total"],
                {"U": values["U"], "R1": values["R1"], "R2": values["R2"]},
                ["Use the parsed equivalent resistance condition to reduce the circuit, then apply RMS Ohm law."],
            )

        if "solenoid" in question and "magnetic field" in question and all(symbol in values for symbol in ("N", "I", "ell")):
            return cls._solution(
                ["magnetism.solenoid_magnetic_field"],
                "B",
                "T",
                ["B = mu_0 * N * I / ell"],
                {symbol: values[symbol] for symbol in ("N", "I", "ell")},
                ["Use the long-solenoid magnetic-field formula with mu_0 as a physical constant."],
            )

        if ("self-inductance" in question or "self inductance" in question or "inductance" in domain) and all(
            symbol in values for symbol in ("epsilon", "I_initial", "I_final", "delta_t")
        ):
            return cls._solution(
                ["inductance.self_inductance_from_emf_current_change"],
                "L_self",
                "H",
                ["delta_I = I_final - I_initial", "L_self = Abs(epsilon) * delta_t / Abs(delta_I)"],
                {symbol: values[symbol] for symbol in ("epsilon", "I_initial", "I_final", "delta_t")},
                ["Compute the current change from verified endpoint currents, then use Abs(epsilon) = L*Abs(delta_I/delta_t)."],
            )
        return None

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

    def _build_prompt(
        self,
        semantic_output: dict[str, Any],
        retrieved_examples: list[dict[str, Any]] | None = None,
    ) -> str:
        rule_packs = self._selected_rule_packs(semantic_output)
        rag_hints = self._format_rag_hints(retrieved_examples)
        parsed_question = json.dumps(semantic_output, ensure_ascii=False, default=str)
        prompt = self.prompt_template
        prompt = (
            prompt.replace("{{RULE_PACKS}}", self._format_rule_packs(rule_packs))
            .replace("{{RAG_HINTS}}", rag_hints)
            .replace("{{PARSED_QUESTION}}", parsed_question)
        )
        self.last_prompt_diagnostics = {
            "prompt_chars": len(prompt),
            "rag_chars": 0 if rag_hints == "None." else len(rag_hints),
            "selected_rule_pack": [name for name, _ in rule_packs],
            "used_json_mode": True,
            "used_repair": False,
        }
        return prompt

    @staticmethod
    def _build_repair_prompt(
        semantic_output: dict[str, Any],
        invalid_solution: Any,
        validation_error: str,
    ) -> str:
        return (
            "Repair the physics solution specification. Return exactly one valid JSON object, "
            "with no markdown or commentary. Keep the same output schema. Use ASCII symbol names only. "
            "Always include top-level mode and answer_type. "
            "If mode is computational, include sympy_spec with target_symbol, target_unit, "
            "a non-empty equations array, and a known_values object, plus a solution_steps array. "
            "If mode is direct, include direct_answer with answer and a rationale_steps array. "
            "Use explicit '*' for multiplication between symbols (e.g., L*C, not LC). "
            "Allowed functions/constants in equations: Abs, abs, sqrt, sin, cos, tan, atan, exp, log, pi, Im, Re, conjugate, "
            "k, k_e, epsilon_0, mu_0, c. Every RHS symbol in sympy_spec.equations must be either a known value, "
            "an allowed physical constant/function, or defined by another equation. The target_symbol must be defined by an equation. "
            "For magnitude/strength/intensity answers, make the final target nonnegative using Abs(...) or a "
            "magnitude formula.\n\n"
            f"Validation error:\n{validation_error}\n\n"
            f"Parsed question:\n{json.dumps(semantic_output, ensure_ascii=False, default=str)}\n\n"
            f"Invalid solution:\n{json.dumps(invalid_solution, ensure_ascii=False, default=str)}\n\n"
            "Repaired JSON:"
        )

    @staticmethod
    def _build_retry_prompt(semantic_output: dict[str, Any], response_preview: str) -> str:
        return (
            "Return exactly one complete valid JSON object for this physics solution specification. "
            "No markdown, no prose, no final numeric calculation. Keep the object compact but complete. "
            "For computational questions include mode, answer_type, sympy_spec.target_symbol, "
            "sympy_spec.target_unit, sympy_spec.equations, sympy_spec.known_values, and solution_steps. "
            "For direct questions include mode, answer_type, direct_answer.answer, and "
            "direct_answer.rationale_steps. Use ASCII SymPy equations and explicit '*'.\n\n"
            f"Parsed question:\n{json.dumps(semantic_output, ensure_ascii=False, default=str)}\n\n"
            f"Previous invalid response preview:\n{response_preview}\n\n"
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
            [{"role": "user", "content": self._build_repair_prompt(semantic_output, invalid_solution, validation_error)}],
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

    def _retry_solution_json(self, semantic_output: dict[str, Any], response_preview: str) -> dict[str, Any] | None:
        retry_response = self.llm_provider.chat(
            [{"role": "user", "content": self._build_retry_prompt(semantic_output, response_preview)}],
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

    def _request_solution(self, prompt: str, semantic_output: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self.llm_provider.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=self.config.get("max_tokens", 4096),
            response_format={"type": "json_object"},
            stage="physics.solution",
        )
        parsed = extract_json(response)
        if not isinstance(parsed, dict):
            response_preview = response[:200] if response else "(empty)"
            validation_error = "Physics solution response must be a JSON object."
            if semantic_output is None:
                raise ValueError(
                    f"{validation_error} "
                    f"Raw response preview: {response_preview}"
                )
            deterministic = self._deterministic_solution(semantic_output)
            if deterministic is not None:
                return self._validate_solution(self._semantic_normalize_solution(deterministic, semantic_output))
            try:
                retried = self._retry_solution_json(semantic_output, response_preview)
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
                deterministic = self._deterministic_solution(semantic_output)
                if deterministic is not None:
                    return self._validate_solution(self._semantic_normalize_solution(deterministic, semantic_output))
                return self._repair_solution(semantic_output, parsed, str(exc))
            except ValueError as repair_exc:
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
        known_symbols = {str(symbol) for symbol in known_values}
        allowed = set(cls.SYMPY_LOCALS) | PHYSICAL_CONSTANT_NAMES
        defined: set[str] = set()
        used: set[str] = set()
        for equation in equations:
            lhs, rhs = equation.split("=", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs):
                defined.add(lhs)
            else:
                used.update(IDENTIFIER_RE.findall(lhs))
            used.update(IDENTIFIER_RE.findall(rhs))
        unresolved = sorted(used - defined - known_symbols - allowed)
        if unresolved:
            raise ValueError(f"Unresolved symbols: {', '.join(unresolved)}.")
        if target_symbol not in defined and target_symbol not in known_symbols:
            raise ValueError(f"Target symbol {target_symbol} must be defined by an equation or known value.")

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
        cls._validate_dependency_closure(equations, known_values, spec["target_symbol"])
        if not isinstance(solution.get("solution_steps", []), list):
            raise ValueError("solution_steps must be an array.")
        if answer_type == "yes_no":
            decision = solution.get("decision_spec")
            if not isinstance(decision, dict) or decision.get("operator") != "approximately_equal":
                raise ValueError("Computational yes/no solution requires approximately_equal decision_spec.")
            if decision.get("computed_symbol") != spec["target_symbol"] or not decision.get("expected_symbol"):
                raise ValueError("Computational yes/no decision symbols are invalid.")
        return solution

    def get_solution(self, question: str, semantic_output: dict[str, Any]) -> dict[str, Any]:
        """Request and validate one structured solution for the parsed question."""
        del question
        deterministic = self._deterministic_solution(semantic_output)
        if deterministic is not None:
            self.last_prompt_diagnostics = {
                "prompt_chars": 0,
                "rag_chars": 0,
                "selected_rule_pack": ["deterministic"],
                "used_json_mode": False,
                "used_repair": False,
            }
            return self._validate_solution(self._semantic_normalize_solution(deterministic, semantic_output))
        return self._request_solution(self._build_prompt(semantic_output), semantic_output)

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
