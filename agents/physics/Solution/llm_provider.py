"""LLM-backed provider for structured physics solution specifications."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agents.formatting import extract_json
from agents.llm import LLMClientBase
from tools.calculator import PHYSICAL_CONSTANTS

from .formula_lib import SYMBOL_REPLACEMENTS, deterministic_solution as formula_lib_solution
from .solution_provider import SolutionProvider

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAG_HINT_CHAR_LIMIT = 2100
RAG_HINT_ITEM_LIMIT = 2
DETERMINISTIC_HINT_CHAR_LIMIT = 2600
DOMAIN_PROMPT_DIR = _PROJECT_ROOT / "prompts" / "physics" / "physics_solution_domains"
CONVERT_TO_SYMPY_PROMPT_PATH = _PROJECT_ROOT / "prompts" / "physics" / "convert_to_sympy.md"
VERIFY_SOLUTION_PROMPT_PATH = _PROJECT_ROOT / "prompts" / "physics" / "verify_solution_against_parser.md"
IGNORED_PARSED_TARGETS = {"", "answer", "result", "each_energy"}
TARGET_EQUIVALENCE_GROUPS = (
    {"q", "Q", "Q_source", "q_source", "charge"},
    {"E", "E_net", "E_total", "E_field", "E_M", "E_N", "E_net_magnitude"},
    {"F", "F_net", "F_resultant", "force"},
    {"I", "I_rms", "I_max", "current"},
    {"U", "V", "U_new", "V_new", "epsilon", "emf", "E_ind"},
    {"W", "E", "W_B", "W_C", "W_L", "W_max", "E_total", "W_total"},
    {"f", "f_res", "frequency"},
    {"omega", "omega_res", "angular_frequency"},
    {"R", "R_total", "R_eq", "R2", "resistance"},
    {"C", "C_eq", "capacitance"},
    {"L", "L_self", "L_ind", "inductance"},
)
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


def _target_symbols_equivalent(left: str, right: str) -> bool:
    if left == right:
        return True
    return any(left in group and right in group for group in TARGET_EQUIVALENCE_GROUPS)


def _unit_key(unit: Any) -> str:
    return str(unit or "").strip().replace("µ", "u").replace("μ", "u").lower()


class LLMSolutionProvider(SolutionProvider):
    """Build solution prompts and return the LLM's JSON specification."""

    DEFAULT_PROMPT_PATH = _PROJECT_ROOT / "prompts" / "physics" / "solution_type2.md"

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
        self.use_solution_parser_verifier = bool(self.config.get("use_solution_parser_verifier", True))
        self.verify_solution_prompt_path = Path(
            self.config.get("verify_solution_prompt_path", VERIFY_SOLUTION_PROMPT_PATH)
        )
        self.verify_solution_prompt_template = self.verify_solution_prompt_path.read_text(encoding="utf-8")
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
            "used_solution_parser_verifier": False,
            "solution_parser_mismatches": [],
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

    def _build_solution_parser_verifier_prompt(
        self,
        question: str,
        semantic_output: dict[str, Any],
        solution_output: dict[str, Any],
        mismatches: list[str],
    ) -> str:
        prompt = (
            self.verify_solution_prompt_template.replace("{{ORIGINAL_QUESTION}}", str(question or ""))
            .replace("{{PARSED_QUESTION}}", json.dumps(semantic_output, ensure_ascii=False, default=str))
            .replace("{{SOLUTION_OUTPUT}}", json.dumps(solution_output, ensure_ascii=False, default=str))
            .replace("{{MISMATCHES}}", json.dumps(mismatches, ensure_ascii=False, default=str))
        )
        self.last_prompt_diagnostics.update(
            {
                "used_solution_parser_verifier": True,
                "solution_parser_verify_prompt": str(self.verify_solution_prompt_path),
                "solution_parser_verify_prompt_chars": len(prompt),
            }
        )
        return prompt

    @staticmethod
    def _lhs_symbols(equations: list[Any]) -> set[str]:
        symbols: set[str] = set()
        for equation in equations:
            if not isinstance(equation, str) or equation.count("=") != 1:
                continue
            lhs = equation.split("=", 1)[0].strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs):
                symbols.add(lhs)
        return symbols

    @staticmethod
    def _parsed_symbols(semantic_output: dict[str, Any]) -> set[str]:
        symbols: set[str] = set()

        def add_symbol(value: Any) -> None:
            text = str(value or "").strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
                symbols.add(text)

        for key in ("givens", "quantities"):
            raw = semantic_output.get(key) or []
            if isinstance(raw, dict):
                for symbol in raw:
                    add_symbol(symbol)
            elif isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict):
                        add_symbol(item.get("symbol"))
        geometry = semantic_output.get("geometry") or {}
        if isinstance(geometry, dict):
            for key in ("segments", "derived_distances"):
                for item in geometry.get(key) or []:
                    if isinstance(item, dict):
                        add_symbol(item.get("symbol"))
        comparison = semantic_output.get("comparison") or {}
        if isinstance(comparison, dict):
            add_symbol(comparison.get("given_quantity_symbol"))
            add_symbol(comparison.get("computed_quantity_symbol"))
        target = semantic_output.get("target") or {}
        if isinstance(target, dict):
            add_symbol(target.get("symbol"))
        return symbols

    def _solution_parser_mismatches(
        self,
        semantic_output: dict[str, Any],
        solution_output: dict[str, Any],
    ) -> list[str]:
        if not self.use_solution_parser_verifier:
            return []
        mismatches: list[str] = []
        question_kind = str(semantic_output.get("question_kind") or "").lower()
        answer_format = semantic_output.get("answer_format") or {}
        requested_form = str(answer_format.get("requested_form") or "").lower() if isinstance(answer_format, dict) else ""
        mode = str(solution_output.get("mode") or "")
        answer_type = str(solution_output.get("answer_type") or "")

        if mode == "direct":
            if question_kind in {"computational", "yes_no_computational"} and semantic_output.get("givens"):
                mismatches.append("Parsed question is computational with givens, but solution returned direct mode.")
            return mismatches

        if mode != "computational":
            mismatches.append(f"Solution mode is {mode or '(missing)'}, expected computational or direct.")
            return mismatches

        spec = solution_output.get("sympy_spec") or {}
        if not isinstance(spec, dict):
            return ["Computational solution is missing sympy_spec object."]
        equations = spec.get("equations") or []
        lhs_symbols = self._lhs_symbols(equations if isinstance(equations, list) else [])
        target_symbol = str(spec.get("target_symbol") or "").strip()
        target_unit = str(spec.get("target_unit") or "").strip()
        parsed_target = semantic_output.get("target") or {}
        parsed_symbol = str(parsed_target.get("symbol") or "").strip() if isinstance(parsed_target, dict) else str(parsed_target or "").strip()
        parsed_unit = str(parsed_target.get("unit") or "").strip() if isinstance(parsed_target, dict) else ""

        if question_kind == "yes_no_computational" and answer_type != "yes_no":
            mismatches.append("Parsed question is yes_no_computational, but solution answer_type is not yes_no.")
        if answer_type == "yes_no" and question_kind != "yes_no_computational" and requested_form != "yes_no":
            mismatches.append("Solution answer_type is yes_no, but parsed question did not request a yes/no computational answer.")
        if parsed_symbol not in IGNORED_PARSED_TARGETS and target_symbol and not _target_symbols_equivalent(parsed_symbol, target_symbol):
            mismatches.append(f"Parsed target symbol is {parsed_symbol}, but solution target_symbol is {target_symbol}.")
        if parsed_unit and target_unit and _unit_key(parsed_unit) != _unit_key(target_unit):
            mismatches.append(f"Parsed target unit is {parsed_unit}, but solution target_unit is {target_unit}.")
        if target_symbol and target_symbol not in lhs_symbols:
            mismatches.append(f"Solution target_symbol {target_symbol} is not defined by any equation lhs.")

        known_values = spec.get("known_values") or {}
        if isinstance(known_values, dict):
            parsed_symbols = self._parsed_symbols(semantic_output)
            fixed_symbols = set(PHYSICAL_CONSTANTS)
            unsupported = sorted(
                str(symbol)
                for symbol in known_values
                if str(symbol) not in parsed_symbols
                and str(symbol) not in fixed_symbols
                and str(symbol) not in lhs_symbols
            )
            if unsupported:
                mismatches.append(
                    "Solution known_values contains symbols not present in parsed_question and not defined by equations: "
                    + ", ".join(unsupported)
                    + "."
                )

        comparison = semantic_output.get("comparison") or {}
        decision = solution_output.get("decision_spec") or {}
        if question_kind == "yes_no_computational":
            if not isinstance(decision, dict) or not decision.get("expected_symbol"):
                mismatches.append("Parsed question is yes/no computational, but solution is missing decision_spec.expected_symbol.")
            elif isinstance(comparison, dict) and comparison.get("given_quantity_symbol"):
                expected = str(decision.get("expected_symbol") or "")
                parsed_expected = str(comparison.get("given_quantity_symbol") or "")
                if expected != parsed_expected:
                    mismatches.append(
                        f"Solution decision expected_symbol is {expected}, but parsed comparison symbol is {parsed_expected}."
                    )
        return mismatches

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

    def _verify_solution_against_parser(
        self,
        question: str,
        semantic_output: dict[str, Any],
        solution_output: dict[str, Any],
    ) -> dict[str, Any]:
        mismatches = self._solution_parser_mismatches(semantic_output, solution_output)
        self.last_prompt_diagnostics["solution_parser_mismatches"] = mismatches
        if not mismatches:
            return solution_output
        prompt = self._build_solution_parser_verifier_prompt(question, semantic_output, solution_output, mismatches)
        verified = self._chat_json(
            prompt,
            stage="physics.solution.verify_against_parser",
            max_tokens_key="verify_solution_max_tokens",
        )
        return self._convert_to_sympy(semantic_output, verified)

    def _finalize_solution(
        self,
        question: str,
        semantic_output: dict[str, Any],
        solution_draft: dict[str, Any],
        validation_error: str = "",
    ) -> dict[str, Any]:
        converted = self._convert_to_sympy(semantic_output, solution_draft, validation_error)
        return self._verify_solution_against_parser(question, semantic_output, converted)

    @classmethod
    def deterministic_solution(cls, semantic_output: dict[str, Any]) -> dict[str, Any] | None:
        """Return the deterministic formula-library result when the workflow has no LLM."""
        return cls._deterministic_solution(semantic_output)

    def get_solution(self, question: str, semantic_output: dict[str, Any]) -> dict[str, Any]:
        """Request one structured solution and normalize it into a SymPy-safe spec."""
        deterministic = self._deterministic_solution(semantic_output)
        prompt = self._build_prompt(semantic_output, deterministic_solution=deterministic)
        solution_draft = self._request_solution(prompt, semantic_output, deterministic)
        return self._finalize_solution(question, semantic_output, solution_draft)

    def repair_solution(
        self,
        question: str,
        semantic_output: dict[str, Any],
        invalid_solution: dict[str, Any],
        validation_error: str,
    ) -> dict[str, Any]:
        """Ask the LLM to revise a solution after executor-side validation fails."""
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
        return self._finalize_solution(question, semantic_output, repaired_draft, validation_error)
