"""LangGraph logic subgraph following the XAI-4-Edu SymbCoT stages."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.logic.symbcot import LogicSymbCoTAgent, split_choices

from .state import WorkflowState
from .tracing import trace_step

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LOGIC_KB = _PROJECT_ROOT / "data" / "Logic_Based_Educational_Queries.json"

TYPE_PROMPTS: dict[str, str] = {
    "YesNo": "Classify as a binary Yes/No/Uncertain logic question.",
    "MultiChoice": "Classify as a fixed-option A/B/C/D question.",
    "Numerical": "Classify as a numerical calculation question.",
    "ChainedQuestion": "Classify as exactly two sequential subquestions.",
    "OpenEnded": "Classify as an open-ended premise-grounded question.",
}

_STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "be",
    "to",
    "for",
    "of",
    "with",
    "and",
    "or",
    "if",
    "then",
    "all",
    "every",
    "does",
    "do",
    "did",
    "can",
    "could",
    "should",
    "must",
    "based",
    "premises",
    "according",
    "about",
    "on",
    "in",
    "that",
    "it",
    "he",
    "she",
    "they",
}


def _llm_available(llm: Any) -> bool:
    return llm is not None and bool(getattr(llm, "enabled", True))


def _with_error(state: WorkflowState, error: str) -> list[str]:
    return [*state.get("errors", []), error]


def _tokens(text: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())
        if len(word) > 1 and word not in _STOP_WORDS
    ]


def tokenize(text: str) -> set[str]:
    return set(_tokens(text))


def classify_logic_question(question: str) -> str:
    """Heuristic helper retained for compatibility; the workflow uses the LLM classifier."""
    q = (question or "").strip()
    q_lower = q.lower()
    if split_choices(q):
        return "MultiChoice"
    if re.search(r"\b(calculate|compute|how many|how much|number|count|sum|difference|ratio|percentage)\b", q_lower):
        return "Numerical"
    if re.search(r"\b(first|second|after that|step|subquestion|part\s*[a-z0-9]|following chain|multi-part)\b", q_lower):
        return "ChainedQuestion"
    if re.match(r"^(does|do|did|is|are|was|were|can|could|should|must|will|would|has|have)\b", q_lower):
        return "YesNo"
    return "OpenEnded"


def prompt_for_type(question_type: str) -> str:
    return TYPE_PROMPTS.get(question_type, TYPE_PROMPTS["OpenEnded"])


class LogicRAGRetriever:
    """Dependency-free BM25 retriever kept for API compatibility.

    XAI-4-Edu's logic pipeline does not use RAG, so LogicWorkflow no longer calls
    this retriever. Existing imports can still instantiate it.
    """

    def __init__(self, dataset_path: str | Path | None = None, top_k: int = 3) -> None:
        self.dataset_path = str(dataset_path) if dataset_path else ""
        self.top_k = top_k
        self.records: list[dict[str, Any]] = []
        self.doc_tokens: list[list[str]] = []
        self.df: Counter[str] = Counter()
        self.avgdl = 1.0
        if dataset_path and Path(dataset_path).exists():
            self.load(dataset_path)

    @property
    def enabled(self) -> bool:
        return bool(self.records)

    def load(self, dataset_path: str | Path) -> None:
        data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
        self.records = [record for record in data if isinstance(record, dict)]
        self.doc_tokens = [_tokens(self._join_record(record)) for record in self.records]
        self.df = Counter()
        for tokens in self.doc_tokens:
            self.df.update(set(tokens))
        total_length = sum(len(tokens) for tokens in self.doc_tokens)
        self.avgdl = total_length / max(1, len(self.doc_tokens))

    @staticmethod
    def _join_record(record: dict[str, Any]) -> str:
        parts: list[str] = []
        parts.extend(str(item) for item in record.get("premises-NL", []))
        parts.extend(str(item) for item in record.get("questions", []))
        return "\n".join(parts)

    def _bm25(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        tf = Counter(doc_tokens)
        n = max(1, len(self.doc_tokens))
        k1, b = 1.5, 0.75
        score = 0.0
        dl = len(doc_tokens)
        for term in query_tokens:
            df = self.df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            freq = tf.get(term, 0)
            denom = freq + k1 * (1 - b + b * dl / self.avgdl)
            score += idf * (freq * (k1 + 1)) / max(1e-9, denom)
        return score

    def retrieve(
        self,
        question: str,
        premises_nl: list[str],
        *,
        top_k: int | None = None,
        exclude_record_index: int | None = None,
        exclude_idx: Any = None,
        exclude_question: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        query = "\n".join([*list(premises_nl or []), question or ""])
        query_tokens = _tokens(query)
        exclude_idx_s = str(exclude_idx) if exclude_idx is not None else None
        exclude_q_norm = re.sub(r"\s+", " ", str(exclude_question or "").strip().lower())
        scored: list[tuple[float, int]] = []
        for index, doc_tokens in enumerate(self.doc_tokens):
            record = self.records[index]
            if exclude_record_index is not None and index == int(exclude_record_index):
                continue
            if exclude_idx_s is not None and str(record.get("idx", "")) == exclude_idx_s:
                continue
            if exclude_q_norm:
                rec_questions = [
                    re.sub(r"\s+", " ", str(item or "").strip().lower())
                    for item in record.get("questions", [])
                ]
                if exclude_q_norm in rec_questions:
                    continue
            scored.append((self._bm25(query_tokens, doc_tokens), index))
        scored.sort(reverse=True)
        return [
            {
                "score": round(score, 4),
                "record_index": index,
                "premises-NL": self.records[index].get("premises-NL", []),
                "questions": self.records[index].get("questions", []),
                "answers": self.records[index].get("answers", []),
                "explanation": self.records[index].get("explanation", []),
            }
            for score, index in scored[: top_k or self.top_k]
        ]


class LogicWorkflow:
    """Question input -> classification -> plan -> execution -> answer extraction."""

    def __init__(
        self,
        llm: Any = None,
        fol_prompt_path: str | Path | None = None,
        explanation_prompt_path: str | Path | None = None,
        rag_path: str | Path | None = None,
        use_rag: bool = True,
        symbcot_prompt_root: str | Path | None = None,
        symbcot_prompt_path: str | Path | None = None,
    ) -> None:
        del fol_prompt_path, explanation_prompt_path, symbcot_prompt_path
        self.llm = llm
        resolved_rag_path = Path(rag_path) if rag_path else _DEFAULT_LOGIC_KB
        self.rag = LogicRAGRetriever(resolved_rag_path if resolved_rag_path.exists() else None)
        self.use_rag = use_rag
        self.symbcot_agent = LogicSymbCoTAgent(
            llm_provider=llm,
            prompt_root=symbcot_prompt_root,
            config={
                "classification_max_tokens": 64,
                "plan_max_tokens": 1200,
                "execution_max_tokens": 2400,
                "execution_retries": 5,
            },
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        graph = StateGraph(WorkflowState)
        graph.add_node("extract_logic", self.extract_logic)
        graph.add_node("classify_question", self.classify_question)
        graph.add_node("generate_plan", self.generate_plan)
        graph.add_node("execute_reasoning", self.execute_reasoning)
        graph.add_node("extract_answer", self.extract_answer)
        graph.add_node("explain_logic", self.explain_logic)
        graph.add_edge(START, "extract_logic")
        graph.add_edge("extract_logic", "classify_question")
        graph.add_edge("classify_question", "generate_plan")
        graph.add_edge("generate_plan", "execute_reasoning")
        graph.add_edge("execute_reasoning", "extract_answer")
        graph.add_edge("extract_answer", "explain_logic")
        graph.add_edge("explain_logic", END)
        return graph.compile()

    @staticmethod
    def _has_terminal_result(state: WorkflowState) -> bool:
        result = state.get("result")
        return isinstance(result, dict) and str(result.get("answer") or "")

    @trace_step("logic.extract_logic")
    def extract_logic(self, state: WorkflowState) -> dict[str, Any]:
        premises = list(state.get("premises", []))
        return {
            "logic_spec": {
                "premises": premises,
                "pipeline": "xai_symbcot_classify_plan_execute_extract",
                "rag_used": False,
            }
        }

    @trace_step("logic.classify_question")
    def classify_question(self, state: WorkflowState) -> dict[str, Any]:
        if self._has_terminal_result(state):
            return {}
        logic_spec = dict(state.get("logic_spec", {}))
        if not _llm_available(self.llm):
            return {
                "logic_spec": logic_spec,
                "result": {
                    "answer": "Unknown",
                    "unit": "",
                    "premises": logic_spec.get("premises", []),
                    "cot": ["Logic SymbCoT classification requires a configured LLM."],
                    "logic_solver": "xai_symbcot",
                },
                "errors": _with_error(state, "Logic SymbCoT requires a configured LLM."),
            }
        try:
            question_type = self.symbcot_agent.classify_question(state["question"])
            logic_spec["question_type"] = question_type
            logic_spec["type_prompt"] = prompt_for_type(question_type)
            return {"logic_spec": logic_spec}
        except Exception as exc:
            return {
                "logic_spec": logic_spec,
                "result": {
                    "answer": "Unknown",
                    "unit": "",
                    "premises": logic_spec.get("premises", []),
                    "cot": ["Question classification failed."],
                    "logic_solver": "xai_symbcot",
                },
                "errors": _with_error(state, f"Logic question classification failed: {exc}"),
            }

    @trace_step("logic.generate_plan")
    def generate_plan(self, state: WorkflowState) -> dict[str, Any]:
        if self._has_terminal_result(state):
            return {}
        logic_spec = dict(state.get("logic_spec", {}))
        premises = list(logic_spec.get("premises") or state.get("premises") or [])
        question_type = str(logic_spec.get("question_type") or "OpenEnded")
        try:
            plan = self.symbcot_agent.generate_plan(premises, state["question"], question_type)
            logic_spec["plan"] = plan
            return {"logic_spec": logic_spec}
        except Exception as exc:
            return {
                "logic_spec": logic_spec,
                "result": {
                    "answer": "Unknown",
                    "unit": "",
                    "premises": premises,
                    "cot": ["Plan generation failed."],
                    "logic_solver": "xai_symbcot",
                },
                "errors": _with_error(state, f"Logic plan generation failed: {exc}"),
            }

    @trace_step("logic.execute_reasoning")
    def execute_reasoning(self, state: WorkflowState) -> dict[str, Any]:
        if self._has_terminal_result(state):
            return {}
        logic_spec = dict(state.get("logic_spec", {}))
        premises = list(logic_spec.get("premises") or state.get("premises") or [])
        question_type = str(logic_spec.get("question_type") or "OpenEnded")
        plan = str(logic_spec.get("plan") or "")
        try:
            raw_execution = self.symbcot_agent.execute_reasoning(premises, state["question"], question_type, plan)
            logic_spec["raw_execution"] = raw_execution
            return {"logic_spec": logic_spec}
        except Exception as exc:
            return {
                "logic_spec": logic_spec,
                "result": {
                    "answer": "Unknown",
                    "unit": "",
                    "premises": premises,
                    "cot": ["Reasoning execution failed."],
                    "logic_solver": "xai_symbcot",
                },
                "errors": _with_error(state, f"Logic execution failed: {exc}"),
            }

    @trace_step("logic.extract_answer")
    def extract_answer(self, state: WorkflowState) -> dict[str, Any]:
        if self._has_terminal_result(state):
            return {}
        logic_spec = dict(state.get("logic_spec", {}))
        premises = list(logic_spec.get("premises") or state.get("premises") or [])
        raw_execution = str(logic_spec.get("raw_execution") or "")
        try:
            extracted = self.symbcot_agent.extract_answer(raw_execution, state["question"], len(premises))
        except Exception as exc:
            return {
                "result": {
                    "answer": "Unknown",
                    "unit": "",
                    "premises": premises,
                    "cot": ["Answer extraction failed."],
                    "logic_solver": "xai_symbcot",
                },
                "errors": _with_error(state, f"Logic answer extraction failed: {exc}"),
            }

        idx = extracted.get("idx") if isinstance(extracted.get("idx"), list) else []
        premises_used = sorted({int(item) - 1 for item in idx if isinstance(item, int) and 1 <= item <= len(premises)})
        premise_steps = [f"Premise {item}: {premises[item - 1]}" for item in idx if 1 <= int(item) <= len(premises)]
        cot = [
            f"Classified question as {logic_spec.get('question_type', 'OpenEnded')}.",
            "Generated a SymbCoT reasoning plan.",
            *premise_steps,
            "Executed the plan and parsed Final answer, idx, and explanation.",
        ]
        return {
            "result": {
                "answer": str(extracted.get("final_answer") or "Unknown"),
                "unit": "",
                "idx": idx,
                "premises_used": premises_used,
                "explanation": str(extracted.get("explanation") or "").strip(),
                "cot": cot,
                "premises": premises,
                "question_type": logic_spec.get("question_type", "OpenEnded"),
                "logic_solver": "xai_symbcot",
            }
        }

    @trace_step("logic.explain_logic")
    def explain_logic(self, state: WorkflowState) -> dict[str, Any]:
        result = dict(state.get("result", {}))
        explanation = str(result.get("explanation") or "").strip()
        if explanation:
            return {"result": result}
        cot = result.get("cot") if isinstance(result.get("cot"), list) else []
        result["explanation"] = (
            " ".join(str(step) for step in cot[:4])
            if cot
            else f"The answer is {result.get('answer', 'Unknown')} based on the available premises."
        )
        return {"result": result, "errors": state.get("errors", [])}
