"""Generate verified physics explanations from parsed questions and solutions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

import sympy as sp

from agents.formatting.json_tools import extract_json
from agents.llm import LLMClientBase

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _comparison_tolerance(expected_value: float) -> float:
    rounding_tolerance = 0.5 if float(expected_value).is_integer() else 1e-2
    return max(rounding_tolerance, abs(expected_value) * 1e-3)


def _clean_steps(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


def _format_known_values(known_values: Dict[str, Any], limit: int = 4) -> str:
    items = [f"{key}={value}" for key, value in list(known_values.items())[:limit]]
    return ", ".join(items)


class ExplainAgent:
    """Create a final physics explanation using the prompt-driven Type 2 format."""

    DEFAULT_PROMPT_PATH = _PROJECT_ROOT / "prompts" / "explanation_type2.md"

    def __init__(
        self,
        llm_provider: LLMClientBase | None = None,
        prompt_path: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.llm_provider = llm_provider
        self.prompt_path = Path(prompt_path) if prompt_path else self.DEFAULT_PROMPT_PATH
        self.config = config or {}
        self.prompt_template = self.prompt_path.read_text(encoding="utf-8")

    def _build_verified_output(
        self,
        parsed_question: Dict[str, Any],
        solution_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        mode = solution_output.get("mode")
        answer_type = solution_output.get("answer_type")
        parsed_target = (parsed_question or {}).get("target") or {}
        default_symbol = parsed_target.get("symbol") or "answer"
        default_unit = parsed_target.get("unit") or ""

        if mode == "direct":
            direct_answer = solution_output.get("direct_answer") or {}
            final_answer = {
                "symbol": default_symbol,
                "value": direct_answer.get("answer"),
                "unit": default_unit,
            }
            return {
                "mode": mode,
                "answer_type": answer_type,
                "final_answer": final_answer,
                "solution_output": solution_output,
            }

        sympy_spec = solution_output.get("sympy_spec") or {}
        target_symbol = sympy_spec.get("target_symbol") or default_symbol
        target_unit = sympy_spec.get("target_unit") or default_unit
        equations = sympy_spec.get("equations") or []
        if not equations:
            raise ValueError("sympy_spec.equations is required for computation.")
        known_values = sympy_spec.get("known_values") or {}

        tokens: set[str] = set()
        for eq in equations:
            tokens.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", eq or ""))
        tokens.update(str(k) for k in known_values.keys() if k)
        tokens.add(target_symbol)
        reserved = {"sqrt", "sin", "cos", "tan", "atan", "Abs", "pi"}
        symbols = {name: sp.Symbol(name) for name in tokens if name not in reserved}
        locals_map = {
            **symbols,
            "sqrt": sp.sqrt,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "atan": sp.atan,
            "Abs": sp.Abs,
            "pi": sp.pi,
        }

        parsed_equations: list[sp.Eq] = []
        for eq in equations:
            if not eq or "=" not in eq:
                raise ValueError(f"Invalid equation format: {eq}")
            left, right = eq.split("=", 1)
            lhs = sp.sympify(left.strip(), locals=locals_map)
            rhs = sp.sympify(right.strip(), locals=locals_map)
            parsed_equations.append(sp.Eq(lhs, rhs))

        substitutions = {}
        for key, value in known_values.items():
            if key in symbols:
                substitutions[symbols[key]] = sp.sympify(value)
        reduced_equations = [eq.subs(substitutions) for eq in parsed_equations]

        target = symbols.get(target_symbol, sp.Symbol(target_symbol))
        solution = sp.solve(reduced_equations, target, dict=True)
        if not solution:
            solution = sp.solve(reduced_equations, dict=True)
        if not solution:
            raise ValueError(f"Unable to solve for {target_symbol} using provided equations.")
        picked = solution[0]
        value = picked.get(target) or picked.get(sp.Symbol(target_symbol))
        if value is None:
            raise ValueError(f"Solution did not contain target symbol {target_symbol}.")

        if isinstance(value, sp.Basic):
            if value.is_Number:
                computed_value: Any = float(value)
            elif value.free_symbols:
                computed_value = str(value)
            else:
                try:
                    computed_value = float(sp.N(value))
                except (TypeError, ValueError):
                    computed_value = str(value)
        else:
            computed_value = value

        trace = [f"{sym} = {sym_value}" for sym, sym_value in picked.items() if sym != target]
        final_answer = {
            "symbol": target_symbol,
            "value": computed_value,
            "unit": target_unit,
        }
        verified_output: Dict[str, Any] = {
            "mode": mode,
            "answer_type": answer_type,
            "final_answer": final_answer,
            "solution_output": solution_output,
            "sympy_result": {
                "symbol": target_symbol,
                "value": computed_value,
                "unit": target_unit,
                "trace": trace,
            },
        }

        if answer_type == "yes_no":
            decision_spec = solution_output.get("decision_spec") or {}
            expected_symbol = decision_spec.get("expected_symbol")
            if not expected_symbol:
                raise ValueError("decision_spec.expected_symbol is required for yes_no answers.")
            expected_value = sympy_spec.get("known_values", {}).get(expected_symbol)
            if expected_value is None:
                expected_sym = symbols.get(expected_symbol, sp.Symbol(expected_symbol))
                expected_value = sp.solve(parsed_equations, expected_sym)
                if isinstance(expected_value, list):
                    expected_value = expected_value[0] if expected_value else None
                elif isinstance(expected_value, dict):
                    expected_value = expected_value.get(expected_sym)
            if expected_value is None:
                raise ValueError("Unable to resolve expected comparison value for yes_no decision.")
            expected_value = float(sp.N(sp.sympify(expected_value)))
            if not isinstance(computed_value, (int, float)):
                raise ValueError("Computed value must be numeric for yes_no decisions.")
            tolerance = _comparison_tolerance(expected_value)
            difference = abs(float(computed_value) - expected_value)
            is_true = difference <= tolerance
            final_answer["value"] = decision_spec.get("answer_if_true") if is_true else decision_spec.get("answer_if_false")
            verified_output["decision_result"] = {
                "computed_value": float(computed_value),
                "expected_value": expected_value,
                "difference": difference,
                "tolerance": tolerance,
                "answer": final_answer["value"],
            }
        return verified_output

    def build_cot(
        self,
        parsed_question: Dict[str, Any],
        solution_output: Dict[str, Any],
        verified_output: Dict[str, Any] | None = None,
    ) -> list[str]:
        verified = verified_output or self._build_verified_output(parsed_question, solution_output)
        parsed_target = (parsed_question or {}).get("target") or {}
        mode = str(verified.get("mode") or solution_output.get("mode") or "").lower()
        answer_type = str(verified.get("answer_type") or solution_output.get("answer_type") or "").lower()
        final_answer = verified.get("final_answer") or {}
        target_symbol = str(final_answer.get("symbol") or parsed_target.get("symbol") or "answer")
        target_unit = str(final_answer.get("unit") or parsed_target.get("unit") or "")
        final_value = final_answer.get("value")

        if mode == "direct":
            rationale_steps = _clean_steps((solution_output.get("direct_answer") or {}).get("rationale_steps"))
            if not rationale_steps:
                rationale_steps = ["Use the verified direct conclusion from the parsed solution."]
            if len(rationale_steps) == 1:
                rationale_steps.append(f"That supports the verified answer {final_value}.")
            return rationale_steps

        sympy_spec = solution_output.get("sympy_spec") or {}
        equations = _clean_steps(sympy_spec.get("equations"))
        known_values = sympy_spec.get("known_values") or {}
        solution_steps = _clean_steps(solution_output.get("solution_steps"))
        trace = _clean_steps((verified.get("sympy_result") or {}).get("trace"))
        decision_result = verified.get("decision_result") if isinstance(verified.get("decision_result"), dict) else {}
        vector_result = verified.get("vector_result") if isinstance(verified.get("vector_result"), dict) else {}

        cot: list[str] = []
        if solution_steps:
            cot.extend(solution_steps)
        if not cot and equations:
            cot.append(f"Use the verified equation {equations[0]}.")
        if equations and len(equations) > 1:
            cot.append(f"Continue with {equations[-1]}.")
        if known_values:
            cot.append(f"Substitute the parsed values {_format_known_values(known_values)} into the equations.")
        if trace:
            cot.append(f"The symbolic solver reduces the system through {', '.join(trace[:3])}.")

        if answer_type == "yes_no" and decision_result:
            cot.append(
                "Compare the computed value "
                f"{decision_result.get('computed_value')} with the expected value "
                f"{decision_result.get('expected_value')} using tolerance {decision_result.get('tolerance')}; "
                f"the difference is {decision_result.get('difference')} so the answer is {decision_result.get('answer')}."
            )
        elif vector_result:
            cot.append(
                "Combine the resolved components "
                f"{vector_result.get('components')} to obtain magnitude {vector_result.get('magnitude')} "
                f"and direction {vector_result.get('direction')}."
            )
        if answer_type == "yes_no" and decision_result:
            cot.append(f"Therefore, the answer is {final_value}.")
        elif vector_result:
            cot.append(f"Therefore, the vector result has magnitude {vector_result.get('magnitude')} {target_unit} and direction {vector_result.get('direction')}.")
        elif target_unit:
            cot.append(f"Therefore, {target_symbol} = {final_value} {target_unit}.")
        else:
            cot.append(f"Therefore, {target_symbol} = {final_value}.")
        return [step for step in cot if step]

    def run(
        self,
        parsed_question: Dict[str, Any],
        solution_output: Dict[str, Any],
        verified_output: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if not solution_output:
            raise ValueError("solution_output is required for explanation.")
        if not self.llm_provider or not self.llm_provider.enabled:
            raise ValueError("llm_provider is required and must be enabled.")

        verified = verified_output or self._build_verified_output(parsed_question, solution_output)
        cot = self.build_cot(parsed_question, solution_output, verified)
        prompt = self.prompt_template
        prompt = prompt.replace(
            "{{PARSED_QUESTION}}",
            json.dumps(parsed_question or {}, ensure_ascii=False, default=str),
        )
        prompt = prompt.replace(
            "{{VERIFIED_OUTPUT}}",
            json.dumps(verified or {}, ensure_ascii=False, default=str),
        )
        prompt = prompt.replace(
            "{{SOLUTION_OUTPUT}}",
            json.dumps(solution_output or {}, ensure_ascii=False, default=str),
        )

        response = self.llm_provider.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=self.config.get("max_tokens", 1024),
            response_format={"type": "json_object"},
            stage="physics.explanation",
        )
        parsed = extract_json(response)
        if not parsed or "answer" not in parsed or "explanation" not in parsed:
            raise ValueError("LLM response must contain 'answer' and 'explanation'.")
        if parsed["answer"] != verified.get("final_answer"):
            raise ValueError("LLM answer does not match verified_output.final_answer.")
        parsed["cot"] = cot
        return parsed
