"""LLM-backed provider for structured physics solution specifications."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agents.formatting import extract_json
from agents.llm import LLMClientBase

from .formula_lib import SYMBOL_REPLACEMENTS, deterministic_solution as formula_lib_solution
from .solution_provider import SolutionProvider

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAG_HINT_CHAR_LIMIT = 2100
RAG_HINT_ITEM_LIMIT = 2
DETERMINISTIC_HINT_CHAR_LIMIT = 2600
DOMAIN_PROMPT_DIR = _PROJECT_ROOT / "prompts" / "physics_solution_domains"
CONVERT_TO_SYMPY_PROMPT_PATH = _PROJECT_ROOT / "prompts" / "convert_to_sympy.md"
DOMAIN_PROMPT_FILES = {
    "Electric Charges and Fields": "electric_charges_and_fields.md",
    "Gauss's Law": "gausss_law.md",
    "Gauss Law": "gausss_law.md",
    "Electric Potential": "electric_potential.md",
    "Capacitance": "capacitance.md",
    "Current and Resistance": "current_and_resistance.md",
    "Direct-Current Circuits": "direct_current_circuits.md",
    "Direct Current Circuits": "direct_current_circuits.md",
    "Magnetic Forces and Fields": "magnetic_forces_and_fields.md",
    "Sources of Magnetic Fields": "sources_of_magnetic_fields.md",
    "Electromagnetic Induction": "electromagnetic_induction.md",
    "Inductance": "inductance.md",
    "Alternating-Current Circuits": "alternating_current_circuits.md",
    "Alternating Current Circuits": "alternating_current_circuits.md",
    "Electromagnetic Waves": "electromagnetic_waves.md",
    "Measurement and Uncertainty": "measurement_and_uncertainty.md",
}


def _normalize_text(value: Any) -> str:
    text = str(value or "")
    for source, replacement in SYMBOL_REPLACEMENTS.items():
        text = text.replace(source, replacement)
    return text


def _json_preview(value: Any, max_chars: int = 1600) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = str(value)
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


def _slugify_domain(value: Any) -> str:
    text = str(value or "").strip().lower().replace("'", "")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


class LLMSolutionProvider(SolutionProvider):
    """Build solution prompts and return the LLM's JSON specification."""

    DEFAULT_PROMPT_PATH = _PROJECT_ROOT / "prompts" / "solution_type2.md"

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
        self.use_domain_prompts = bool(self.config.get("use_domain_prompts", prompt_path is None))
        self.domain_prompt_dir = Path(self.config.get("domain_prompt_dir", DOMAIN_PROMPT_DIR))
        self.use_convert_to_sympy = bool(self.config.get("use_convert_to_sympy", True))
        self.convert_to_sympy_prompt_path = Path(
            self.config.get("convert_to_sympy_prompt_path", CONVERT_TO_SYMPY_PROMPT_PATH)
        )
        self.convert_to_sympy_prompt_template = self.convert_to_sympy_prompt_path.read_text(encoding="utf-8")
        self._domain_prompt_cache: dict[Path, str] = {}
        self.last_prompt_diagnostics: dict[str, Any] = {}

    @staticmethod
    def _truncate_text(value: Any, max_chars: int) -> str:
        text = _normalize_text(value).strip()
        if len(text) <= max_chars:
            return text
        return f"{text[: max_chars - 3].rstrip()}..."

    @classmethod
    def _extract_strategy_hints(cls, value: Any) -> list[str]:
        if isinstance(value, list):
            text = " ".join(str(item) for item in value)
        else:
            text = _normalize_text(value).replace("\r", "\n")
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
        hints = [cls._compact_rag_example(item) for item in retrieved_examples[:RAG_HINT_ITEM_LIMIT]]
        return cls._truncate_text(json.dumps(hints, ensure_ascii=False, default=str), RAG_HINT_CHAR_LIMIT)

    @classmethod
    def _format_deterministic_hints(cls, deterministic: dict[str, Any] | None) -> str:
        if not isinstance(deterministic, dict):
            return "None."
        compact: dict[str, Any] = {}
        for key in ("mode", "answer_type", "formula_ids", "assumptions", "vector_spec", "decision_spec"):
            if deterministic.get(key) is not None:
                compact[key] = deterministic[key]
        spec = deterministic.get("sympy_spec")
        if isinstance(spec, dict):
            compact["sympy_spec"] = {
                key: spec[key]
                for key in ("target_symbol", "target_unit", "equations", "known_values")
                if spec.get(key) is not None
            }
        direct = deterministic.get("direct_answer")
        if isinstance(direct, dict):
            compact["direct_answer"] = direct
        if deterministic.get("solution_steps"):
            compact["solution_steps"] = deterministic["solution_steps"][:4]
        return cls._truncate_text(json.dumps(compact, ensure_ascii=False, default=str), DETERMINISTIC_HINT_CHAR_LIMIT)

    @staticmethod
    def _deterministic_solution(semantic_output: dict[str, Any]) -> dict[str, Any] | None:
        return formula_lib_solution(semantic_output)

    def _domain_prompt_path(self, semantic_output: dict[str, Any]) -> Path | None:
        if not self.use_domain_prompts:
            return None
        domain = str(semantic_output.get("domain") or "").strip()
        file_name = DOMAIN_PROMPT_FILES.get(domain)
        if file_name is None:
            slug = _slugify_domain(domain)
            file_name = f"{slug}.md" if slug else None
        if not file_name:
            return None
        path = self.domain_prompt_dir / file_name
        return path if path.exists() else None

    def _select_prompt_template(self, semantic_output: dict[str, Any]) -> tuple[str, Path, str]:
        path = self._domain_prompt_path(semantic_output)
        if path is None:
            return self.prompt_template, self.prompt_path, "fallback"
        if path not in self._domain_prompt_cache:
            self._domain_prompt_cache[path] = path.read_text(encoding="utf-8")
        return self._domain_prompt_cache[path], path, "domain"

    def _build_prompt(
        self,
        semantic_output: dict[str, Any],
        retrieved_examples: list[dict[str, Any]] | None = None,
        deterministic_solution: dict[str, Any] | None = None,
    ) -> str:
        rag_hints = self._format_rag_hints(retrieved_examples)
        deterministic_hints = self._format_deterministic_hints(deterministic_solution)
        parsed_question = json.dumps(semantic_output, ensure_ascii=False, default=str)
        prompt_template, prompt_path, prompt_kind = self._select_prompt_template(semantic_output)
        prompt = (
            prompt_template.replace("{{RAG_HINTS}}", rag_hints)
            .replace("{{DETERMINISTIC_HINTS}}", deterministic_hints)
            .replace("{{PARSED_QUESTION}}", parsed_question)
        )
        self.last_prompt_diagnostics = {
            "prompt_chars": len(prompt),
            "rag_chars": 0 if rag_hints == "None." else len(rag_hints),
            "deterministic_chars": 0 if deterministic_hints == "None." else len(deterministic_hints),
            "selected_rule_pack": [prompt_kind, prompt_path.name],
            "selected_prompt": str(prompt_path),
            "selected_domain": str(semantic_output.get("domain") or ""),
            "used_json_mode": True,
            "used_repair": False,
            "used_convert_to_sympy": False,
        }
        return prompt

    def _build_convert_to_sympy_prompt(
        self,
        semantic_output: dict[str, Any],
        solution_draft: dict[str, Any],
        validation_error: str = "",
    ) -> str:
        parsed_question = json.dumps(semantic_output, ensure_ascii=False, default=str)
        draft = json.dumps(solution_draft, ensure_ascii=False, default=str)
        prompt = (
            self.convert_to_sympy_prompt_template.replace("{{PARSED_QUESTION}}", parsed_question)
            .replace("{{SOLUTION_DRAFT}}", draft)
            .replace("{{VALIDATION_ERROR}}", validation_error or "None.")
        )
        self.last_prompt_diagnostics.update(
            {
                "used_convert_to_sympy": True,
                "convert_prompt": str(self.convert_to_sympy_prompt_path),
                "convert_prompt_chars": len(prompt),
            }
        )
        return prompt

    def _chat_json(self, prompt: str, *, stage: str, max_tokens_key: str = "max_tokens") -> dict[str, Any]:
        response = self.llm_provider.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=self.config.get(max_tokens_key, 4096),
            response_format={"type": "json_object"},
            stage=stage,
        )
        parsed = extract_json(response)
        if not isinstance(parsed, dict):
            preview = self._truncate_text(response or "(empty)", 240)
            raise ValueError(f"Physics solution response must be a JSON object. Raw response preview: {preview}")
        return parsed

    def _request_solution(
        self,
        prompt: str,
        semantic_output: dict[str, Any] | None = None,
        deterministic_fallback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del semantic_output, deterministic_fallback
        return self._chat_json(prompt, stage="physics.solution")

    def _convert_to_sympy(
        self,
        semantic_output: dict[str, Any],
        solution_draft: dict[str, Any],
        validation_error: str = "",
    ) -> dict[str, Any]:
        if not self.use_convert_to_sympy:
            return solution_draft
        if solution_draft.get("mode") == "direct":
            self.last_prompt_diagnostics.update({"used_convert_to_sympy": False, "skipped_convert_to_sympy": "direct_mode"})
            return solution_draft
        prompt = self._build_convert_to_sympy_prompt(semantic_output, solution_draft, validation_error)
        return self._chat_json(prompt, stage="physics.solution.convert_to_sympy", max_tokens_key="convert_max_tokens")

    @classmethod
    def deterministic_solution(cls, semantic_output: dict[str, Any]) -> dict[str, Any] | None:
        """Return the deterministic formula-library result when the workflow has no LLM."""
        return cls._deterministic_solution(semantic_output)

    def get_solution(self, question: str, semantic_output: dict[str, Any]) -> dict[str, Any]:
        """Request one structured solution and normalize it into a SymPy-safe spec."""
        del question
        deterministic = self._deterministic_solution(semantic_output)
        prompt = self._build_prompt(semantic_output, deterministic_solution=deterministic)
        solution_draft = self._request_solution(prompt, semantic_output, deterministic)
        return self._convert_to_sympy(semantic_output, solution_draft)

    def repair_solution(
        self,
        question: str,
        semantic_output: dict[str, Any],
        invalid_solution: dict[str, Any],
        validation_error: str,
    ) -> dict[str, Any]:
        """Ask the LLM to revise a solution after executor-side validation fails."""
        del question
        self.last_prompt_diagnostics["used_repair"] = True
        prompt = (
            "Repair the physics solution JSON so it can be executed by SymPy. "
            "Return exactly one JSON object and no prose.\n\n"
            f"Validation error:\n{validation_error}\n\n"
            f"Parsed question:\n{json.dumps(semantic_output, ensure_ascii=False, default=str)}\n\n"
            f"Invalid solution:\n{_json_preview(invalid_solution)}\n\n"
            "JSON:"
        )
        repaired_draft = self._chat_json(prompt, stage="physics.solution.repair", max_tokens_key="repair_max_tokens")
        return self._convert_to_sympy(semantic_output, repaired_draft, validation_error)
