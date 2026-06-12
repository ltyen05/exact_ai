"""Generate verified physics explanations from parsed questions and solutions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.formatting.json_tools import extract_json
from agents.llm import LLMClientBase

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _clean_steps(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


def _format_known_values(known_values: dict[str, Any], limit: int = 4) -> str:
    items = [f"{key}={value}" for key, value in list(known_values.items())[:limit]]
    return ", ".join(items)


class ExplainAgent:
    """Create a final physics explanation using the prompt-driven Type 2 format."""

    DEFAULT_PROMPT_PATH = _PROJECT_ROOT / "prompts" / "physics" / "explanation_type2.md"

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

    @staticmethod
    def answer_object_to_text(answer: dict[str, Any]) -> str:
        if not isinstance(answer, dict) or answer.get("value") is None:
            return "Unknown"
        value = answer.get("value")
        unit = str(answer.get("unit") or "")
        if unit and unit.lower() != "dimensionless":
            return f"{value} {unit}".strip()
        return str(value)

    @staticmethod
    def _light_verified_output(
        parsed_question: dict[str, Any],
        solution_output: dict[str, Any],
        verified_output: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Use upstream verification when present; otherwise build only display context."""
        if isinstance(verified_output, dict) and verified_output:
            return dict(verified_output)
        parsed_target = (parsed_question or {}).get("target") or {}
        sympy_spec = (solution_output or {}).get("sympy_spec") or {}
        direct_answer = (solution_output or {}).get("direct_answer") or {}
        return {
            "mode": (solution_output or {}).get("mode"),
            "answer_type": (solution_output or {}).get("answer_type"),
            "final_answer": {
                "symbol": sympy_spec.get("target_symbol") or parsed_target.get("symbol") or "answer",
                "value": direct_answer.get("answer"),
                "unit": sympy_spec.get("target_unit") or parsed_target.get("unit") or "",
            },
            "solution_output": solution_output or {},
        }

    def build_cot(
        self,
        parsed_question: dict[str, Any],
        solution_output: dict[str, Any],
        verified_output: dict[str, Any] | None = None,
    ) -> list[str]:
        verified = self._light_verified_output(parsed_question, solution_output, verified_output)
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
            cot.append(f"Verified intermediate values include {', '.join(trace[:3])}.")

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
        parsed_question: dict[str, Any],
        solution_output: dict[str, Any],
        verified_output: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not solution_output:
            raise ValueError("solution_output is required for explanation.")
        if not self.llm_provider or not self.llm_provider.enabled:
            raise ValueError("llm_provider is required and must be enabled.")

        verified = self._light_verified_output(parsed_question, solution_output, verified_output)
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
            max_tokens=self.config.get("max_tokens", 512),
            response_format={"type": "json_object"},
            stage="physics.explanation",
        )
        parsed = extract_json(response)
        if not isinstance(parsed, dict):
            raise ValueError("LLM response must be a JSON object.")
        fallback_answer = self.answer_object_to_text(verified.get("final_answer") or {})
        if isinstance(parsed.get("answer"), dict):
            parsed["answer"] = self.answer_object_to_text(parsed["answer"])
        elif parsed.get("answer") is None:
            parsed["answer"] = fallback_answer
        if not isinstance(parsed.get("explanation"), str) or not parsed["explanation"].strip():
            parsed["explanation"] = " ".join(cot) or f"Therefore, the answer is {fallback_answer}."
        parsed["cot"] = cot
        return parsed
