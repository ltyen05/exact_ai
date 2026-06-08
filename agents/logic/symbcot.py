"""XAI-4-Edu style SymbCoT pipeline for educational logic questions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agents.formatting import extract_json
from agents.llm import LLMClientBase

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PROMPT_ROOT = _PROJECT_ROOT / "prompts" / "logic_symbcot" / "xai"

QUESTION_TYPES = ("YesNo", "MultiChoice", "OpenEnded", "ChainedQuestion", "Numerical")


def split_choices(question: str) -> dict[str, str]:
    """Return A-D multiple-choice options from common benchmark formats."""
    choices: dict[str, str] = {}
    matches = list(re.finditer(r"(?:^|\n)\s*([A-D])[\.\)]\s*", question or "", flags=re.I))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(question)
        choices[match.group(1).upper()] = question[start:end].strip()
    return choices


def normalize_question_type(value: Any) -> str:
    text = str(value or "").strip()
    for question_type in QUESTION_TYPES:
        if re.search(rf"\b{re.escape(question_type)}\b", text, flags=re.I):
            return question_type
    aliases = {
        "yes/no": "YesNo",
        "yes no": "YesNo",
        "multiple choice": "MultiChoice",
        "multi choice": "MultiChoice",
        "mcq": "MultiChoice",
        "chained": "ChainedQuestion",
        "chain": "ChainedQuestion",
        "numeric": "Numerical",
        "number": "Numerical",
        "open ended": "OpenEnded",
    }
    lower = text.lower()
    for alias, question_type in aliases.items():
        if alias in lower:
            return question_type
    return "OpenEnded"


def normalize_logic_answer(answer: Any, question: str) -> str:
    """Normalize model answers into the public logic answer vocabulary."""
    text = str(answer or "").strip()
    compact = re.sub(r"\s+", " ", text).strip(" .,:;`{}")
    compact = re.sub(r"^final\s+answer\s*:\s*", "", compact, flags=re.I).strip(" .,:;`{}")
    lower = compact.lower()
    aliases = {
        "true": "Yes",
        "yes": "Yes",
        "false": "No",
        "no": "No",
        "uncertain": "Unknown",
        "unknown": "Unknown",
        "cannot be determined": "Unknown",
        "not enough information": "Unknown",
        "not enough info": "Unknown",
    }
    if lower in aliases:
        return aliases[lower]
    choices = split_choices(question)
    if choices:
        match = re.search(r"\b([A-D])\b", compact, flags=re.I)
        if match:
            return match.group(1).upper()
    if lower.startswith("yes"):
        return "Yes"
    if lower.startswith("no"):
        return "No"
    if lower.startswith(("unknown", "uncertain", "cannot")):
        return "Unknown"
    return compact or "Unknown"


def valid_logic_answer(answer: str, question: str) -> bool:
    choices = split_choices(question)
    if choices:
        return answer in choices or answer == "Unknown"
    return answer in {"Yes", "No", "Unknown"} or bool(answer.strip())


def format_context(premises: list[str]) -> str:
    """Match the XAI prompt style: one-based premise numbering with no extra metadata."""
    return "\n".join(f"{index}.{premise}" for index, premise in enumerate(premises, 1))


def _parse_idx(value: Any, premise_count: int) -> list[int]:
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = re.findall(r"\d+", str(value or ""))
    idx = sorted(
        {
            int(item)
            for item in raw_items
            if isinstance(item, int) or (isinstance(item, str) and item.strip().isdigit())
        }
    )
    return [item for item in idx if 1 <= item <= premise_count]


def _extract_final_block(text: str) -> dict[str, Any] | None:
    parsed = extract_json(text)
    if isinstance(parsed, dict):
        return parsed

    answer_match = re.search(
        r"Final\s+answer\s*:\s*(?:\{)?([^\n\r}]+)(?:\})?",
        text or "",
        flags=re.I,
    )
    if not answer_match:
        return None
    idx_match = re.search(r"idx\s*:\s*(\[[^\]]*\]|[0-9,\s]+)", text or "", flags=re.I)
    explanation_match = re.search(r"explanation\s*:\s*(.+)", text or "", flags=re.I | re.S)
    return {
        "Final_answer": answer_match.group(1).strip(),
        "idx": idx_match.group(1).strip() if idx_match else [],
        "explanation": explanation_match.group(1).strip() if explanation_match else "",
    }


class LogicSymbCoTProvider:
    """Run the same conceptual stages as XAI-4-Edu: classify, plan, execute, parse."""

    DEFAULT_PROMPT_ROOT = _DEFAULT_PROMPT_ROOT

    def __init__(
        self,
        llm_provider: LLMClientBase | None,
        prompt_root: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.llm_provider = llm_provider
        self.prompt_root = Path(prompt_root) if prompt_root else self.DEFAULT_PROMPT_ROOT
        self.config = config or {}
        self.prompt_cache: dict[Path, str] = {}
        self.last_prompts: dict[str, str] = {}

    def _load_prompt(self, *parts: str) -> str:
        path = self.prompt_root.joinpath(*parts)
        if path not in self.prompt_cache:
            self.prompt_cache[path] = path.read_text(encoding="utf-8")
        return self.prompt_cache[path]

    def _chat(
        self,
        prompt: str,
        *,
        stage: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        if self.llm_provider is None:
            raise ValueError("llm_provider is required for logic SymbCoT.")
        self.last_prompts[stage] = prompt
        return self.llm_provider.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a highly intelligent and logical assistant. "
                        "Solve deductional and logical reasoning tasks with clear, step-by-step explanations. "
                        "Use only the given premises and avoid unstated assumptions."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            stage=stage,
        )

    def classify_question(self, question: str) -> str:
        prompt = self._load_prompt("classify_question.txt").replace("[[QUESTION]]", question)
        response = self._chat(
            prompt,
            stage="logic.xai.classify",
            max_tokens=int(self.config.get("classification_max_tokens", 64)),
        )
        return normalize_question_type(response)

    def build_plan_prompt(self, premises: list[str], question: str, question_type: str) -> str:
        template = self._load_prompt(question_type, "plan_generation.txt")
        return template.replace("[[CONTEXT]]", format_context(premises)).replace("[[QUESTION]]", question)

    def generate_plan(self, premises: list[str], question: str, question_type: str) -> str:
        prompt = self.build_plan_prompt(premises, question, question_type)
        return self._chat(
            prompt,
            stage="logic.xai.plan",
            max_tokens=int(self.config.get("plan_max_tokens", 1200)),
        )

    def build_solver_prompt(self, premises: list[str], question: str, question_type: str, plan: str) -> str:
        template = self._load_prompt(question_type, "solver.txt")
        return (
            template.replace("[[CONTEXT]]", format_context(premises))
            .replace("[[QUESTION]]", question)
            .replace("[[PLAN]]", plan)
        )

    def execute_reasoning(self, premises: list[str], question: str, question_type: str, plan: str) -> str:
        prompt = self.build_solver_prompt(premises, question, question_type, plan)
        retries = int(self.config.get("execution_retries", 5))
        max_tokens = int(self.config.get("execution_max_tokens", 2400))
        last_response = ""
        for attempt in range(retries):
            last_response = self._chat(
                prompt,
                stage="logic.xai.execute",
                temperature=0.0 if attempt == 0 else 0.2,
                max_tokens=max_tokens,
            )
            if _extract_final_block(last_response):
                return last_response
        return last_response

    def extract_answer(self, raw_execution: str, question: str, premise_count: int) -> dict[str, Any]:
        parsed = _extract_final_block(raw_execution)
        if not isinstance(parsed, dict):
            raise ValueError("Logic answer parser could not find Final answer, idx, and explanation.")
        raw_answer = (
            parsed.get("Final_answer")
            if parsed.get("Final_answer") is not None
            else parsed.get("final_answer")
            if parsed.get("final_answer") is not None
            else parsed.get("answer")
            if parsed.get("answer") is not None
            else parsed.get("answers")
        )
        answer = normalize_logic_answer(raw_answer, question)
        if not valid_logic_answer(answer, question):
            raise ValueError(f"Unsupported logic answer: {answer}")
        return {
            "final_answer": answer,
            "idx": _parse_idx(parsed.get("idx"), premise_count),
            "explanation": str(parsed.get("explanation") or "").strip(),
        }

    def solve(self, premises: list[str], question: str) -> dict[str, Any]:
        question_type = self.classify_question(question)
        plan = self.generate_plan(premises, question, question_type)
        raw_execution = self.execute_reasoning(premises, question, question_type, plan)
        extracted = self.extract_answer(raw_execution, question, len(premises))
        return {
            **extracted,
            "question_type": question_type,
            "plan": plan,
            "raw_execution": raw_execution,
        }


class LogicSymbCoTAgent:
    """Small domain agent wrapper mirroring the physics provider/agent split."""

    def __init__(
        self,
        llm_provider: LLMClientBase | None,
        prompt_root: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.provider = LogicSymbCoTProvider(llm_provider, prompt_root=prompt_root, config=config)

    def classify_question(self, question: str) -> str:
        return self.provider.classify_question(question)

    def generate_plan(self, premises: list[str], question: str, question_type: str) -> str:
        return self.provider.generate_plan(premises, question, question_type)

    def execute_reasoning(self, premises: list[str], question: str, question_type: str, plan: str) -> str:
        return self.provider.execute_reasoning(premises, question, question_type, plan)

    def extract_answer(self, raw_execution: str, question: str, premise_count: int) -> dict[str, Any]:
        return self.provider.extract_answer(raw_execution, question, premise_count)

    def run(self, *, premises: list[str], question: str) -> dict[str, Any]:
        return self.provider.solve(premises, question)
