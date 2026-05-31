"""Prompt construction for the physics semantic parser."""

from __future__ import annotations

from pathlib import Path


class ParserPromptBuilder:
    """Load and render parser prompts."""

    def __init__(self, prompt_path: Path) -> None:
        self.prompt_path = prompt_path
        self.prompt_template = self.prompt_path.read_text(encoding="utf-8")

    def build_prompt(self, question: str) -> str:
        """Insert the input question into the semantic-parser prompt."""
        if "{{QUESTION}}" in self.prompt_template:
            return self.prompt_template.replace("{{QUESTION}}", question)
        return f"{self.prompt_template.rstrip()}\n\nInput:\n{question}\n\nOutput:"


def build_repair_prompt(question: str, response: str) -> str:
    """Ask the LLM to convert a malformed parser response into schema JSON."""
    return (
        "Convert the physics parser response into exactly one valid JSON object. "
        "Return JSON only, with no markdown or commentary. "
        "Required fields: question, domain, target, givens, relations, question_kind. "
        "Use ASCII symbol names only: ell, phi, theta, omega, mu. "
        "Use question_kind computational, yes_no_computational, yes_no_conceptual, "
        "multiple_choice, or conceptual. Do not solve the problem.\n\n"
        f"Question:\n{question}\n\nMalformed response:\n{response}\n\nJSON:"
    )


def build_retry_prompt(question: str, response_preview: str) -> str:
    """Ask the LLM to retry parsing after an invalid JSON response."""
    return (
        "Return exactly one complete valid JSON object parsing this physics question. "
        "No markdown, no prose, no calculation. Required fields: question, domain, target, "
        "givens, relations, question_kind. Use ASCII symbols such as mu_0, ell, omega; "
        "convert stated numeric quantities to SI floats where possible.\n\n"
        f"Question:\n{question}\n\n"
        f"Previous invalid response preview:\n{response_preview}\n\n"
        "JSON:"
    )


def build_example_repair_prompt(question: str, response: str) -> str:
    """Provide a concrete JSON example to guide the LLM repair on second attempt."""
    return (
        "The following physics question must be parsed into a JSON object. "
        "The previous attempts failed to produce valid JSON. "
        "Return ONLY a valid JSON object with NO markdown, NO commentary, NO extra text. "
        "Do not wrap in ```json``` blocks.\n\n"
        f"Question:\n{question}\n\n"
        "Required JSON schema (fill in the values, do not solve the problem):\n"
        '{"question": "...", "domain": "...", "target": {"symbol": "...", "unit": "..."}, '
        '"givens": [{"symbol": "...", "value": ..., "si_value": ..., "si_unit": "..."}], '
        '"relations": ["..."], "question_kind": "computational"}\n\n'
        f"Previous malformed response (for reference only):\n{response[:500]}\n\n"
        "Valid JSON:"
    )
