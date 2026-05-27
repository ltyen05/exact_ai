from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents.formatter import extract_json
from agents.llm_client import LLMClient
from agents.models.state import BaseAgent, AgentInput, AgentOutput
from tools.calculator import solve_by_formula


class PhysicsRetriever:
    def __init__(self, kb_path: Optional[str] = None):
        self.examples: List[Dict[str, Any]] = []
        if kb_path and Path(kb_path).exists():
            with open(kb_path, encoding="utf-8") as f:
                data = json.load(f)
            self.examples = [r for r in data if not str(r.get("id", "")).startswith("QA")]

    @staticmethod
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9μµΩ.]+", " ", s.lower())).strip()

    def best(self, question: str) -> Tuple[Optional[Dict[str, Any]], float]:
        qn = self.norm(question)
        best_ex, best_score = None, 0.0
        for ex in self.examples:
            score = SequenceMatcher(None, qn, self.norm(ex.get("question", ""))).ratio()
            if score > best_score:
                best_ex, best_score = ex, score
        return best_ex, best_score


class PhysicsAgent(BaseAgent):
    """Physics pipeline: formula tool first, local LLM second, retrieval fallback third."""

    def __init__(self, kb_path: Optional[str] = None, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()
        self.retriever = PhysicsRetriever(kb_path)

    def solve_with_llm(self, question: str) -> Optional[Dict[str, Any]]:
        if not self.llm.enabled:
            return None
        system = (
            "Solve the physics problem using explicit steps and calculator-style formulas. "
            "Return JSON only: {answer: string, unit: string, cot: [steps], premises: [formulas], confidence: number}. "
            "Use concise final numeric answer."
        )
        try:
            text = self.llm.chat([
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ], temperature=0, max_tokens=1200, response_format={"type": "json_object"})
            obj = extract_json(text)
            if obj and obj.get("answer"):
                return obj
        except Exception:
            return None
        return None

    def solve(self, agent_input: AgentInput) -> AgentOutput:
        question = agent_input.question
        calc = solve_by_formula(question)
        if calc:
            return AgentOutput(
                answer=calc.answer,
                reasoning="The calculator tool matched a physics formula pattern and computed the result step by step.",
                confidence=calc.confidence,
                metadata={
                    "unit": calc.unit,
                    "cot": calc.cot,
                    "premises": ["symbolic calculator", "physics formula library"],
                },
                agent_name="PhysicsAgent",
            )

        llm_obj = self.solve_with_llm(question)
        if llm_obj:
            ans = llm_obj.get("answer", "Unknown")
            unit = llm_obj.get("unit", "")
            cot = llm_obj.get("cot", [])
            return AgentOutput(
                answer=ans,
                reasoning="A local open-source LLM produced a structured solution, then the formatter normalized the output.",
                confidence=llm_obj.get("confidence", 0.65),
                metadata={
                    "unit": unit,
                    "cot": cot,
                    "premises": llm_obj.get("premises", []),
                },
                agent_name="PhysicsAgent",
            )

        ex, score = self.retriever.best(question)
        if ex and score >= 0.78:
            cot = ex.get("cot", "")
            cot_list = cot.split("\n") if isinstance(cot, str) else list(cot or [])
            return AgentOutput(
                answer=ex.get("answer", "Unknown"),
                reasoning=f"No local LLM was configured. The retrieval fallback used the nearest solved training problem ({ex.get('id')}) with similarity {score:.3f}.",
                confidence=round(min(0.82, score), 3),
                metadata={
                    "unit": ex.get("unit", ""),
                    "cot": cot_list[:8],
                    "premises": [f"retrieved_example_id={ex.get('id')}", f"similarity={score:.3f}"],
                },
                agent_name="PhysicsAgent",
            )

        return AgentOutput(
            answer="Unknown",
            reasoning="No formula pattern matched, no local LLM endpoint was configured, and retrieval similarity was too low for a reliable answer.",
            confidence=0.15,
            metadata={
                "unit": "",
                "cot": ["Route=physics", "Formula solver failed", "LLM disabled or failed", "Retrieval fallback below threshold"],
            },
            agent_name="PhysicsAgent",
        )

