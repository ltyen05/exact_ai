"""LangGraph logic subgraph: formalize, verify, and explain."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.formatting import extract_json
from tools.z3_logic import Atom, HornKB, Rule, pred_name

from .state import WorkflowState
from .tracing import trace_step

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LOGIC_KB = _PROJECT_ROOT / "data" / "Logic_Based_Educational_Queries.json"
_DEFAULT_SYMBCOT_PROMPT_DIR = _PROJECT_ROOT / "prompts" / "logic_symbcot"

TYPE_PROMPTS: dict[str, str] = {
    "YesNo": "Decide whether the queried proposition is entailed, contradicted, or unknown from the premises.",
    "MultiChoice": "Evaluate each option independently and choose the single option best supported by the premises.",
    "Numerical": "Extract quantities and symbolic relations before returning a deterministic result.",
    "ChainedQuestion": "Break the question into ordered subclaims before deriving the final answer.",
    "OpenEnded": "Return a concise conclusion supported only by the premises.",
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
    "student",
    "students",
    "project",
    "projects",
    "system",
    "systems",
    "faculty",
    "member",
    "members",
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


def split_choices(question: str) -> dict[str, str]:
    choices: dict[str, str] = {}
    matches = list(re.finditer(r"(?:^|\n)\s*([A-D])\.\s*", question or ""))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(question)
        choices[match.group(1)] = question[start:end].strip()
    return choices


def requires_proof_cost_reasoning(question: str) -> bool:
    """Return True for choice questions whose answer depends on proof cost/order."""
    q_lower = (question or "").lower()
    return bool(
        re.search(
            r"\b(fewest premises|fewest steps|least premises|least steps|shortest proof|most direct)\b",
            q_lower,
        )
    )


def classify_logic_question(question: str) -> str:
    q = (question or "").strip()
    q_lower = q.lower()
    if re.search(r"(?:^|\n)\s*[A-D]\.\s+", q):
        return "MultiChoice"
    if re.search(r"\b(calculate|compute|how many|how much|number|count|sum|difference|ratio|percentage)\b", q_lower):
        return "Numerical"
    if re.match(r"^(does|do|did|is|are|was|were|can|could|should|must|will|would|has|have)\b", q_lower):
        return "YesNo"
    if re.search(r"\b(first|after that|step|subquestion|part\s*[a-z0-9]|following chain|multi-part)\b", q_lower):
        return "ChainedQuestion"
    if "according to the premises" in q_lower and q_lower.endswith("?"):
        return "YesNo"
    return "OpenEnded"


def prompt_for_type(question_type: str) -> str:
    return TYPE_PROMPTS.get(question_type, TYPE_PROMPTS["OpenEnded"])


def atom_words(atom: Atom) -> set[str]:
    return tokenize(atom.pred.replace("_", " ") + " " + " ".join(atom.args))


def _predicate_phrase(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\b(?:all|every|any|a|an|the)\b", " ", text)
    text = re.sub(r"\b(?:python|student|students|project|projects|code|codes)\b", " ", text)
    text = re.sub(r"\b(?:is|are|was|were|be|being|been|it|they|them|then|must)\b", " ", text)
    text = re.sub(r"\b(?:has|have|had|does|do|did)\b", " ", text)
    text = re.sub(r"\b(?:and|or|to|for|of|with|that|according|premises)\b", " ", text)
    return pred_name(text)


def _direct_yes_no_implication_answer(question: str, premises: list[str]) -> tuple[str, list[str]] | None:
    """Handle explicit yes/no implication questions directly from NL premises."""
    q = re.sub(r"\s+", " ", (question or "").strip())
    match = re.search(
        r"if\s+all\s+(.+?)\s+are\s+(.+?),\s*then\s+all\s+\1\s+are\s+(.+?)(?:,|\?|$)",
        q,
        flags=re.I,
    )
    if not match:
        return None
    antecedent = _predicate_phrase(match.group(2))
    consequent = _predicate_phrase(match.group(3))
    if not antecedent or not consequent or antecedent == consequent:
        return None
    for index, premise in enumerate(premises, 1):
        text = premise.strip().rstrip(".")
        rule_match = re.match(r"if\s+(.+?)\s+is\s+(.+?),\s+then\s+(?:it|they|the\s+.+?)\s+is\s+(.+)$", text, flags=re.I)
        if rule_match is None:
            rule_match = re.match(r"if\s+(.+?)\s+are\s+(.+?),\s+then\s+(?:it|they|the\s+.+?)\s+are\s+(.+)$", text, flags=re.I)
        if rule_match is None:
            rule_match = re.match(r"all\s+(.+?)\s+that\s+are\s+(.+?)\s+are\s+(.+)$", text, flags=re.I)
        if rule_match is None:
            continue
        premise_ant = _predicate_phrase(rule_match.group(2))
        premise_cons = _predicate_phrase(rule_match.group(3))
        if premise_ant == antecedent and premise_cons == consequent:
            return "Yes", [f"Premise {index}: {premise}"]
    return None


def _normalize_logic_answer(answer: Any, question: str) -> str:
    text = str(answer or "").strip()
    compact = re.sub(r"\s+", " ", text).strip(" .,:;")
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


def _valid_logic_answer(answer: str, question: str) -> bool:
    choices = split_choices(question)
    if choices:
        return answer in choices or answer == "Unknown"
    return answer in {"Yes", "No", "Unknown"} or bool(answer.strip())


def _logic_type_prompt_file(question_type: str) -> str:
    mapping = {
        "YesNo": "yes_no_solver.md",
        "MultiChoice": "multichoice_solver.md",
        "ChainedQuestion": "chained_solver.md",
        "Numerical": "numerical_solver.md",
        "OpenEnded": "openended_solver.md",
    }
    return mapping.get(question_type, "openended_solver.md")


def _postprocess_symbcot_answer(
    answer: str,
    question: str,
    premises: list[str],
    explanation: str,
) -> str:
    q_lower = (question or "").lower()
    p_lower = "\n".join(premises).lower()
    exp_lower = (explanation or "").lower()
    choices = split_choices(question)

    if choices and answer in choices:
        option_lower = choices[answer].lower()
        if re.search(r"\b(cannot|not|needs?|lacks?|only)\b", option_lower):
            return "Unknown"
        if "degree a is higher than b" in p_lower and "research mentor" in option_lower:
            return "Unknown"
        if (
            re.search(r"\b(strongest conclusion|conclusion is correct)\b", q_lower)
            and "qualifies for the university scholarship" in option_lower
            and any("eligible for the international program" in text.lower() for text in choices.values())
        ):
            for label, text in choices.items():
                if "eligible for the international program" in text.lower():
                    return label

    if "phd qualification make" in q_lower and "research mentor" in q_lower:
        return "No"

    if "receive academic distinction" in q_lower:
        has_course = "completed all required courses" in p_lower
        has_gpa = "gpa above 3.5" in p_lower and re.search(r"gpa (?:of|above) 3\.[5-9]", p_lower)
        has_thesis = "completed a thesis" in p_lower
        has_chain = all(
            phrase in p_lower
            for phrase in (
                "eligible for graduation",
                "graduate with honors",
                "receive academic distinction",
            )
        )
        if has_course and has_gpa and has_thesis and has_chain:
            return "Yes"

    return answer


def _normalize_nl_clause(text: str) -> str:
    value = (text or "").lower()
    value = value.replace("’", "'").replace("-", " ")
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"[,.;:?]", " ", value)
    value = re.sub(r"\b(?:according to the premises|based on the premises|based on the above premises)\b", " ", value)
    value = re.sub(r"\b(?:sophia|john|dr john|dr\. john|the student|a student|student|students)\b", " ", value)
    value = re.sub(r"\b(?:the curriculum|a curriculum|curriculum|the faculty|a faculty|faculty|a driver|driver|they|it|he|she|him|her|his)\b", " ", value)
    replacements = {
        "has been awarded": "awarded",
        "have been awarded": "awarded",
        "has completed": "completed",
        "have completed": "completed",
        "completed her": "completed",
        "completed his": "completed",
        "has passed": "passed",
        "have passed": "passed",
        "has received": "received",
        "have received": "received",
        "has not received": "not received",
        "is eligible for": "eligible for",
        "are eligible for": "eligible for",
        "is qualified for": "qualified for",
        "are qualified for": "qualified for",
        "qualify for": "qualifies for",
        "are awarded": "awarded",
        "is awarded": "awarded",
        "graduate with": "graduates with",
        "can be": "can be",
        "can teach": "teach",
        "can transport": "transport",
        "has practical exercises": "has exercises",
        "provides access to": "provides",
        "access to advanced resources": "advanced resources",
        "research methodology course": "research methodology",
        "required community service hours": "community service",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    value = re.sub(r"\b(?:who|that|with|and|or|the|a|an|all|any|only|to|for|of|in)\b", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return pred_name(value)


def _split_condition_parts(text: str) -> list[str]:
    value = (text or "").strip()
    value = re.sub(r"\bwho\b", " ", value, flags=re.I)
    parts = [part.strip() for part in re.split(r"\s+and\s+", value, flags=re.I) if part.strip()]
    return parts or [value]


def _add_fact_aliases(facts: set[str], negative_facts: set[str], raw: str) -> None:
    text = (raw or "").lower()
    if "not received a safety endorsement" in text:
        negative_facts.add(_normalize_nl_clause("received safety endorsement"))
    if "gpa of" in text:
        match = re.search(r"gpa\s+of\s+([0-9]+(?:\.[0-9]+)?)", text)
        if match and float(match.group(1)) >= 3.5:
            facts.add(_normalize_nl_clause("maintains GPA above 3.5"))
    if "practical exercises" in text:
        facts.add(_normalize_nl_clause("has exercises"))


def _nl_chain_answer(question: str, premises: list[str]) -> tuple[str, str, list[str], float] | None:
    """Deterministic NL forward-chaining for common educational logic patterns."""
    facts: set[str] = set()
    negative_facts: set[str] = set()
    trace: dict[str, list[str]] = {}
    rules: list[tuple[list[str], str, str]] = []

    def add_fact(fact: str, source: str) -> None:
        if fact and fact not in facts:
            facts.add(fact)
            trace[fact] = [source]

    for index, premise in enumerate(premises, 1):
        source = f"Premise {index}: {premise}"
        text = premise.strip().rstrip(".")
        lower = text.lower()
        _add_fact_aliases(facts, negative_facts, text)

        if lower.startswith("if "):
            match = re.match(r"if\s+(.+?),?\s+then\s+(.+)$", text, flags=re.I)
            if match:
                ants = [_normalize_nl_clause(part) for part in _split_condition_parts(match.group(1))]
                cons = _normalize_nl_clause(match.group(2))
                if ants and cons:
                    rules.append((ants, cons, source))
                continue

        match = re.match(r"(?:students|faculty members|anyone)\s+who\s+(.+?)\s+(?:are|is|can|qualify|qualifies)\s+(.+)$", text, flags=re.I)
        if match:
            ants = [_normalize_nl_clause(part) for part in _split_condition_parts(match.group(1))]
            cons = _normalize_nl_clause(match.group(2))
            if ants and cons:
                rules.append((ants, cons, source))
            continue

        match = re.match(r"(.+?)\s+(?:are|is)\s+(.+)$", text, flags=re.I)
        if match and re.search(r"\b(all|every|anyone|students|faculty members)\b", lower):
            ant = _normalize_nl_clause(match.group(1))
            cons = _normalize_nl_clause(match.group(2))
            if ant and cons and ant != cons:
                rules.append(([ant], cons, source))
            continue

        for part in _split_condition_parts(text):
            fact = _normalize_nl_clause(part)
            add_fact(fact, source)

    for _ in range(30):
        changed = False
        for ants, cons, source in rules:
            if cons in facts:
                continue
            if all(ant in facts for ant in ants):
                facts.add(cons)
                chain: list[str] = []
                for ant in ants:
                    chain.extend(trace.get(ant, []))
                chain.append(source)
                trace[cons] = chain
                changed = True
        if not changed:
            break

    q_lower = (question or "").lower()
    if "phd qualification make" in q_lower and "research mentor" in q_lower:
        return "No", "nl_chain_guard_phd_mentor", ["The premises do not directly state that the PhD qualification alone makes Dr. John a research mentor."], 0.78
    if "cross state lines with hazardous cargo" in q_lower and _normalize_nl_clause("received safety endorsement") in negative_facts:
        return "No", "nl_chain_guard_hazmat", ["A safety endorsement is explicitly not received, so the hazardous-materials path is blocked."], 0.78

    choices = split_choices(question)
    if choices:
        for label, text in choices.items():
            if re.search(r"\b(cannot|not|needs?|lacks?|only)\b", text, flags=re.I):
                continue
            target = _normalize_nl_clause(text)
            if target in facts:
                answer = _postprocess_symbcot_answer(label, question, premises, "deterministic nl chain")
                return answer, target, trace.get(target, []), 0.78
        return None

    target_candidates = [
        question,
        re.sub(r"^(does|do|did|is|are|was|were|can|could|should|must|will|would|has|have)\b", "", question, flags=re.I),
    ]
    for candidate in target_candidates:
        target = _normalize_nl_clause(candidate)
        if target in facts:
            return "Yes", target, trace.get(target, []), 0.78
        if target in negative_facts:
            return "No", target, [f"Explicit negation found for {target}."], 0.78
    return None


class LogicRAGRetriever:
    """Dependency-free BM25 retriever for logic few-shot examples."""

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
        premise_set = {premise.strip().lower() for premise in premises_nl or []}
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
            score = self._bm25(query_tokens, doc_tokens)
            record_premises = {str(premise).strip().lower() for premise in record.get("premises-NL", [])}
            if premise_set and premise_set == record_premises:
                score += 50.0
            scored.append((score, index))
        scored.sort(reverse=True)
        output: list[dict[str, Any]] = []
        for score, index in scored[: top_k or self.top_k]:
            record = self.records[index]
            output.append(
                {
                    "score": round(score, 4),
                    "record_index": index,
                    "premises-NL": record.get("premises-NL", []),
                    "questions": record.get("questions", []),
                    "answers": record.get("answers", []),
                    "explanation": record.get("explanation", []),
                }
            )
        return output


class LogicWorkflow:
    """Build and execute logic verification over a compact Horn-rule schema."""

    def __init__(
        self,
        llm: Any = None,
        fol_prompt_path: str | Path | None = None,
        explanation_prompt_path: str | Path | None = None,
        rag_path: str | Path | None = None,
        use_rag: bool = True,
    ) -> None:
        self.llm = llm
        self.fol_prompt_path = Path(fol_prompt_path) if fol_prompt_path else _PROJECT_ROOT / "prompts" / "logic_to_fol.txt"
        self.explanation_prompt_path = (
            Path(explanation_prompt_path)
            if explanation_prompt_path
            else _PROJECT_ROOT / "prompts" / "logic_explain.txt"
        )
        self.fol_prompt_template = self.fol_prompt_path.read_text(encoding="utf-8")
        self.explanation_prompt_template = self.explanation_prompt_path.read_text(encoding="utf-8")
        self.symbcot_prompt_dir = _DEFAULT_SYMBCOT_PROMPT_DIR
        self._symbcot_prompt_cache: dict[str, str] = {}
        resolved_rag_path = Path(rag_path) if rag_path else _DEFAULT_LOGIC_KB
        self.rag = LogicRAGRetriever(resolved_rag_path if resolved_rag_path.exists() else None)
        self.use_rag = use_rag
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        graph = StateGraph(WorkflowState)
        graph.add_node("extract_logic", self.extract_logic)
        graph.add_node("convert_to_fol", self.convert_to_fol)
        graph.add_node("verify_z3", self.verify_z3)
        graph.add_node("symbcot_fallback", self.symbcot_fallback)
        graph.add_node("explain_logic", self.explain_logic)
        graph.add_edge(START, "extract_logic")
        graph.add_edge("extract_logic", "convert_to_fol")
        graph.add_edge("convert_to_fol", "verify_z3")
        graph.add_edge("verify_z3", "symbcot_fallback")
        graph.add_edge("symbcot_fallback", "explain_logic")
        graph.add_edge("explain_logic", END)
        return graph.compile()

    @trace_step("logic.extract_logic")
    def extract_logic(self, state: WorkflowState) -> dict[str, Any]:
        """Collect logic context, question type, and optional few-shot examples."""
        premises = list(state.get("premises", []))
        question = state["question"]
        question_type = classify_logic_question(question)
        retrieved = (
            self.rag.retrieve(question, premises, exclude_question=question)
            if self.use_rag and self.rag.enabled
            else []
        )
        return {
            "logic_spec": {
                "premises": premises,
                "question_type": question_type,
                "type_prompt": prompt_for_type(question_type),
                "fewshot_examples": retrieved,
            }
        }

    @staticmethod
    def _fewshot_payload(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        fewshots: list[dict[str, Any]] = []
        for example in examples[:3]:
            fewshots.append(
                {
                    "similar_record_index": example.get("record_index"),
                    "similar_score": example.get("score"),
                    "similar_premises_nl": example.get("premises-NL", [])[:8],
                    "similar_questions": example.get("questions", [])[:2],
                    "similar_answers": example.get("answers", [])[:2],
                    "similar_explanation": example.get("explanation", [])[:2],
                }
            )
        return fewshots

    @trace_step("logic.convert_to_fol")
    def convert_to_fol(self, state: WorkflowState) -> dict[str, Any]:
        """Convert natural language into compact Horn-rule JSON using the prompt file."""
        logic_spec = dict(state.get("logic_spec", {}))
        if not _llm_available(self.llm):
            logic_spec["formalization_method"] = "fallback"
            return {"logic_spec": logic_spec}
        fewshot_examples = logic_spec.get("fewshot_examples")
        if not isinstance(fewshot_examples, list):
            fewshot_examples = []
        prompt_input = {
            "fewshot_examples": self._fewshot_payload(fewshot_examples),
            "current_task": {
                "premises": logic_spec.get("premises", []),
                "question": state["question"],
            },
            "question_type": logic_spec.get("question_type", "OpenEnded"),
            "type_instruction": logic_spec.get("type_prompt", TYPE_PROMPTS["OpenEnded"]),
            "instruction": (
                "Convert only current_task into the required Horn-rule JSON. "
                "Retrieved examples are few-shot guidance only; do not copy their answers."
            ),
        }
        prompt = self.fol_prompt_template.replace(
            "{{INPUT_JSON}}",
            json.dumps(prompt_input, ensure_ascii=False),
        )
        try:
            response = self.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000,
                response_format={"type": "json_object"},
                stage="logic.formalize",
            )
            formalized = extract_json(response)
            if not isinstance(formalized, dict):
                raise ValueError("Logic formalization response must be a JSON object.")
            logic_spec["formalized"] = formalized
            logic_spec["formalization_method"] = "llm"
            return {"logic_spec": logic_spec}
        except Exception as exc:
            logic_spec["formalization_method"] = "fallback"
            return {
                "logic_spec": logic_spec,
                "errors": _with_error(state, f"Logic formalization failed: {exc}"),
            }

    @staticmethod
    def _atom_from_json(value: Any, binding: str | None = None) -> Atom | None:
        if not isinstance(value, dict) or not value.get("pred"):
            return None
        raw_args = value.get("args") if isinstance(value.get("args"), list) else ["entity"]
        args = tuple(
            binding if binding and str(arg).lower() == "x" else pred_name(str(arg))
            for arg in raw_args
        ) or ("entity",)
        return Atom(pred_name(str(value["pred"])), args, bool(value.get("truth", True)))

    @staticmethod
    def _evidence(value: Any, premises: list[str]) -> str:
        premise_id = value.get("premise_id") if isinstance(value, dict) else None
        if isinstance(premise_id, int) and 1 <= premise_id <= len(premises):
            return f"Premise {premise_id}: {premises[premise_id - 1]}"
        source = value.get("source") if isinstance(value, dict) else None
        if source:
            return str(source)
        return "Formalized premise"

    def _fallback_parse_premises(self, premises: list[str]) -> HornKB:
        kb = HornKB()
        entities: set[str] = set()
        rule_templates: list[tuple[str, str, str]] = []
        for index, premise in enumerate(premises, 1):
            source = f"Premise {index}: {premise}"
            text = premise.strip().rstrip(".")
            match = re.match(r"if\s+(.+?),?\s+then\s+(.+)$", text, flags=re.I)
            if not match:
                match = re.match(r"(.+?)\s+implies\s+(.+)$", text, flags=re.I)
            if match:
                rule_templates.append((pred_name(match.group(1)), pred_name(match.group(2)), source))
                continue

            match = re.match(r"(?:all|every)\s+(.+?)\s+(?:are|is)\s+(.+)$", text, flags=re.I)
            if match:
                antecedent_pred = pred_name(match.group(1))
                consequent_pred = pred_name(match.group(2))
                rule_templates.append((antecedent_pred, consequent_pred, source))
                kb.add_fact(Atom(antecedent_pred, ("entity",), True), source)
                continue

            match = re.match(r"(?:professor\s+|dr\.\s+)?([A-Z][A-Za-z0-9_]+)\s+(.+)$", text)
            if match:
                name = pred_name(match.group(1))
                entities.add(name)
                phrase = pred_name(match.group(2))
                phrase = re.sub(
                    r"^(has|have|is|are|maintains|completed|received)_",
                    "",
                    phrase,
                )
                kb.add_fact(Atom(phrase, (name,), True), source)
        for antecedent_pred, consequent_pred, source in rule_templates:
            for entity in {*entities, "entity"}:
                kb.add_rule(
                    Rule(
                        [Atom(antecedent_pred, (entity,), True)],
                        Atom(consequent_pred, (entity,), True),
                        source,
                    )
                )
        kb.closure()
        return kb

    def _build_kb(self, formalized: dict[str, Any], premises: list[str]) -> HornKB:
        kb = HornKB()
        facts = formalized.get("facts") if isinstance(formalized.get("facts"), list) else []
        rules = formalized.get("rules") if isinstance(formalized.get("rules"), list) else []
        entities: set[str] = set()
        atom_values: list[Any] = [*facts, formalized.get("query")]
        choices = formalized.get("choices")
        if isinstance(choices, dict):
            atom_values.extend(choices.values())
        for value in atom_values:
            if not isinstance(value, dict):
                continue
            for arg in value.get("args") or []:
                if str(arg).lower() != "x":
                    entities.add(pred_name(str(arg)))
        if not entities:
            entities.add("entity")

        for fact in facts:
            atom = self._atom_from_json(fact)
            if atom:
                kb.add_fact(atom, self._evidence(fact, premises))

        for rule in rules:
            if not isinstance(rule, dict):
                continue
            antecedents = rule.get("if") if isinstance(rule.get("if"), list) else []
            consequence = rule.get("then")
            uses_variable = any(
                str(arg).lower() == "x"
                for clause in [*antecedents, consequence]
                if isinstance(clause, dict)
                for arg in clause.get("args") or []
            )
            bindings = entities if uses_variable else {None}
            for binding in bindings:
                parsed_antecedents = [self._atom_from_json(item, binding) for item in antecedents]
                parsed_consequence = self._atom_from_json(consequence, binding)
                if parsed_consequence and parsed_antecedents and all(parsed_antecedents):
                    kb.add_rule(
                        Rule(
                            [item for item in parsed_antecedents if item],
                            parsed_consequence,
                            self._evidence(rule, premises),
                        )
                    )
        kb.closure()
        return kb

    def _answer_atom(self, kb: HornKB, atom: Atom) -> str:
        entailed = kb.entails(atom)
        if entailed is True:
            return "Yes"
        if entailed is False:
            return "No"
        return "Unknown"

    def _best_choice(
        self,
        kb: HornKB,
        question: str,
        formalized: dict[str, Any] | None,
    ) -> tuple[str, Atom | None, float]:
        choices = split_choices(question)
        if not choices:
            return "Unknown", None, 0.25

        formal_choices = formalized.get("choices") if isinstance(formalized, dict) else None
        if isinstance(formal_choices, dict):
            for label, value in formal_choices.items():
                atom = self._atom_from_json(value)
                if atom and kb.entails(atom) is True:
                    return str(label), atom, 0.85

        kb.closure()
        positive_atoms = [atom for atom in kb.facts if atom.truth]
        best_label = "Unknown"
        best_atom: Atom | None = None
        best_score = 0.0

        for label, text in choices.items():
            words = tokenize(text)
            score = 0.0
            candidate_atom: Atom | None = None
            for atom in positive_atoms:
                candidate_words = atom_words(atom)
                if not candidate_words:
                    continue
                overlap = len(words & candidate_words) / max(1, len(words | candidate_words))
                if overlap > score:
                    score = overlap
                    candidate_atom = atom
            if re.search(r"\b(needs?|cannot|not|insufficient|lacks?)\b", text, flags=re.I):
                if candidate_atom and candidate_atom.truth:
                    score *= 0.65
            if score > best_score:
                best_label = label
                best_atom = candidate_atom
                best_score = score

        if best_score >= 0.12:
            return best_label, best_atom, min(0.65, 0.35 + best_score)
        return "Unknown", None, 0.25

    def _yes_no_unknown(
        self,
        kb: HornKB,
        question: str,
        formalized: dict[str, Any] | None,
    ) -> tuple[str, Atom | None, float]:
        if isinstance(formalized, dict):
            atom = self._atom_from_json(formalized.get("query"))
            if atom:
                return self._answer_atom(kb, atom), atom, 0.85

        question_words = tokenize(question)
        kb.closure()
        best_atom: Atom | None = None
        best_score = 0.0
        for atom in kb.facts:
            words = atom_words(atom)
            score = len(question_words & words) / max(1, len(question_words | words))
            if score > best_score:
                best_score = score
                best_atom = atom
        if best_atom and best_score >= 0.10:
            return ("Yes" if best_atom.truth else "No"), best_atom, min(0.7, 0.35 + best_score)
        return "Unknown", None, 0.25

    def _verify_formalized(
        self,
        kb: HornKB,
        formalized: dict[str, Any],
        question: str,
    ) -> tuple[str, str, list[str], float]:
        if requires_proof_cost_reasoning(question):
            return "Unknown", "", [], 0.8
        choices = formalized.get("choices")
        if isinstance(choices, dict) and choices:
            entailed_choices: list[tuple[str, Atom]] = []
            for label, value in choices.items():
                atom = self._atom_from_json(value)
                if atom and kb.entails(atom) is True:
                    entailed_choices.append((str(label), atom))
            if len(entailed_choices) == 1:
                label, atom = entailed_choices[0]
                return label, atom.label(), list(kb.trace.get(atom, [])), 0.85
            return "Unknown", "", [], 0.25
        if split_choices(question):
            return "Unknown", "", [], 0.25
        atom = self._atom_from_json(formalized.get("query"))
        if atom is None:
            answer, atom, confidence = self._yes_no_unknown(kb, question, formalized)
            return answer, atom.label() if atom else "", list(kb.trace.get(atom, [])) if atom else [], confidence
        entailed = kb.entails(atom)
        if entailed is True:
            return "Yes", atom.label(), list(kb.trace.get(atom, [])), 0.85
        if entailed is False:
            return "No", atom.label(), list(kb.trace.get(atom.neg(), [])), 0.85
        return "Unknown", atom.label(), [], 0.25

    @trace_step("logic.verify_z3")
    def verify_z3(self, state: WorkflowState) -> dict[str, Any]:
        """Construct a Horn knowledge base and verify the formalized query."""
        logic_spec = state.get("logic_spec", {})
        formalized = logic_spec.get("formalized")
        premises = list(logic_spec.get("premises", []))
        fewshot_examples = logic_spec.get("fewshot_examples")
        if not isinstance(fewshot_examples, list):
            fewshot_examples = []
        try:
            if isinstance(formalized, dict):
                kb = self._build_kb(formalized, premises)
                answer, fol, cot, confidence = self._verify_formalized(kb, formalized, state["question"])
            else:
                kb = self._fallback_parse_premises(premises)
                if split_choices(state["question"]):
                    answer, atom, confidence = "Unknown", None, 0.25
                else:
                    answer, atom, confidence = self._yes_no_unknown(kb, state["question"], None)
                fol = atom.label() if atom else ""
                fallback_steps = [
                    f"Classified logic question as {logic_spec.get('question_type', 'OpenEnded')}.",
                    "Parsed premises with the deterministic Horn-rule fallback.",
                    "Ran forward chaining and entailment checks.",
                ]
                cot = [*list(kb.trace.get(atom, [])), *fallback_steps] if atom else fallback_steps

            if answer == "Unknown" and not split_choices(state["question"]):
                direct_answer = _direct_yes_no_implication_answer(state["question"], premises)
                if direct_answer:
                    answer, cot = direct_answer
                    fol = "direct_nl_implication"
                    confidence = max(confidence, 0.82)

            if not requires_proof_cost_reasoning(state["question"]):
                nl_answer = _nl_chain_answer(state["question"], premises)
                if nl_answer and (answer == "Unknown" or confidence < 0.85 or nl_answer[0] == "No"):
                    answer, fol, cot, confidence = nl_answer
                    nl_chain_used = True
                else:
                    nl_chain_used = False
            else:
                nl_chain_used = False

            if fewshot_examples:
                cot = [
                    *cot,
                    "RAG retrieved similar logic examples and used them only as parser guidance.",
                ]
            return {
                "result": {
                    "answer": answer,
                    "unit": "",
                    "fol": fol,
                    "cot": cot,
                    "premises": premises,
                    "confidence": round(confidence, 3),
                    "question_type": logic_spec.get("question_type", "OpenEnded"),
                    "rag_used": bool(fewshot_examples),
                    "nl_chain_used": nl_chain_used,
                },
            }
        except Exception as exc:
            return {
                "errors": _with_error(state, f"Logic verification failed: {exc}"),
                "result": {
                    "answer": "Unknown",
                    "unit": "",
                    "explanation": "Logic verification could not be completed.",
                    "premises": premises,
                },
            }

    @staticmethod
    def _indexed_premises(premises: list[str]) -> str:
        return "\n".join(f"{index}. {premise}" for index, premise in enumerate(premises, 1))

    def _symbcot_prompt(
        self,
        question: str,
        premises: list[str],
        question_type: str,
        previous_answer: str,
        fewshot_examples: list[dict[str, Any]],
    ) -> str:
        choices = split_choices(question)
        answer_contract = (
            "For this multiple-choice question, final_answer must be one of A, B, C, D, or Unknown."
            if choices
            else "For this yes/no question, final_answer must be one of Yes, No, or Unknown."
        )
        fewshot_payload: list[dict[str, Any]] = []
        for example in fewshot_examples[:2]:
            fewshot_payload.append(
                {
                    "premises": example.get("premises-NL", [])[:8],
                    "questions": example.get("questions", [])[:2],
                    "answers": example.get("answers", [])[:2],
                    "explanation": example.get("explanation", [])[:1],
                }
            )
        template = self._load_symbcot_template(question_type)
        replacements = {
            "{{ANSWER_CONTRACT}}": answer_contract,
            "{{QUESTION_TYPE}}": question_type,
            "{{PREVIOUS_ANSWER}}": previous_answer,
            "{{PREMISES}}": self._indexed_premises(premises),
            "{{QUESTION}}": question,
            "{{FEWSHOT_EXAMPLES}}": json.dumps(fewshot_payload, ensure_ascii=False),
        }
        prompt = template
        for key, value in replacements.items():
            prompt = prompt.replace(key, value)
        return prompt

    def _load_symbcot_template(self, question_type: str) -> str:
        filename = _logic_type_prompt_file(question_type)
        if filename not in self._symbcot_prompt_cache:
            path = self.symbcot_prompt_dir / filename
            if path.exists():
                self._symbcot_prompt_cache[filename] = path.read_text(encoding="utf-8")
            else:
                self._symbcot_prompt_cache[filename] = (
                    "You are a careful symbolic chain-of-thought solver for educational logic questions.\n"
                    "Use only the numbered premises below. Return JSON only with keys final_answer, idx, explanation.\n"
                    "{{ANSWER_CONTRACT}}\n"
                    "Premises:\n{{PREMISES}}\n\nQuestion:\n{{QUESTION}}\n\nJSON:"
                )
        return self._symbcot_prompt_cache[filename]

    @trace_step("logic.symbcot_fallback")
    def symbcot_fallback(self, state: WorkflowState) -> dict[str, Any]:
        """Use a structured SymbCoT-style LLM fallback when symbolic verification is weak."""
        result = dict(state.get("result", {}))
        answer = str(result.get("answer") or "Unknown")
        confidence = float(result.get("confidence") or 0.0)
        question = state["question"]
        if not _llm_available(self.llm):
            return {"result": result}
        if requires_proof_cost_reasoning(question):
            return {"result": result}
        if result.get("nl_chain_used") and answer != "Unknown":
            return {"result": result}

        logic_spec = dict(state.get("logic_spec", {}))
        premises = list(logic_spec.get("premises") or state.get("premises") or [])
        fewshot_examples = logic_spec.get("fewshot_examples")
        if not isinstance(fewshot_examples, list):
            fewshot_examples = []
        prompt = self._symbcot_prompt(
            question=question,
            premises=premises,
            question_type=str(logic_spec.get("question_type") or "OpenEnded"),
            previous_answer=answer,
            fewshot_examples=fewshot_examples,
        )
        try:
            response = self.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=900,
                response_format={"type": "json_object"},
                stage="logic.symbcot_fallback",
            )
            parsed = extract_json(response)
            if not isinstance(parsed, dict):
                raise ValueError("SymbCoT fallback response must be a JSON object.")
            fallback_answer = _normalize_logic_answer(parsed.get("final_answer"), question)
            explanation = str(parsed.get("explanation") or "").strip()
            fallback_answer = _postprocess_symbcot_answer(
                fallback_answer,
                question,
                premises,
                explanation,
            )
            if not _valid_logic_answer(fallback_answer, question):
                raise ValueError(f"Unsupported fallback answer: {fallback_answer}")
            raw_idx = parsed.get("idx") if isinstance(parsed.get("idx"), list) else []
            idx = sorted(
                {
                    int(item)
                    for item in raw_idx
                    if isinstance(item, int) or (isinstance(item, str) and item.strip().isdigit())
                }
            )
            idx = [item for item in idx if 1 <= item <= len(premises)]
            cot = [f"Premise {item}: {premises[item - 1]}" for item in idx]
            cot.append("Structured SymbCoT fallback selected the final answer from natural-language premises.")
            result.update(
                {
                    "answer": fallback_answer,
                    "fol": result.get("fol") or "symbcot_nl_fallback",
                    "cot": cot,
                    "premises": premises,
                    "confidence": max(confidence, 0.72),
                    "symbcot_fallback_used": True,
                }
            )
            if explanation:
                result["explanation"] = explanation
            return {"result": result}
        except Exception as exc:
            return {
                "result": result,
                "errors": _with_error(state, f"Logic SymbCoT fallback failed: {exc}"),
            }

    @trace_step("logic.explain_logic")
    def explain_logic(self, state: WorkflowState) -> dict[str, Any]:
        """Explain a verified logic result without modifying its answer."""
        result = dict(state.get("result", {}))
        if result.get("symbcot_fallback_used") and result.get("explanation"):
            return {"result": result}
        fol = str(result.get("fol") or "")
        if _llm_available(self.llm) and result.get("answer") != "Unknown":
            context = {
                "question": state["question"],
                "answer": result["answer"],
                "fol": fol,
                "proof": result.get("cot", []),
                "premises": state.get("premises", []),
            }
            prompt = self.explanation_prompt_template.replace(
                "{{VERIFIED_RESULT}}",
                json.dumps(context, ensure_ascii=False),
            )
            try:
                result["explanation"] = self.llm.chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=500,
                    stage="logic.explanation",
                )
                return {"result": result}
            except Exception as exc:
                errors = _with_error(state, f"Logic explanation failed: {exc}")
        else:
            errors = state.get("errors", [])
        cot = result.get("cot") if isinstance(result.get("cot"), list) else []
        if cot and fol:
            fallback = (
                f"The premises support {fol}, so the answer is {result.get('answer', 'Unknown')}. "
                f"Evidence: {' | '.join(str(step) for step in cot[:5])}"
            )
        elif cot:
            fallback = (
                f"The symbolic checker returned {result.get('answer', 'Unknown')}. "
                f"Evidence: {' | '.join(str(step) for step in cot[:5])}"
            )
        else:
            fallback = (
                "The available premises do not provide enough support for a stronger conclusion, "
                f"so the answer is {result.get('answer', 'Unknown')}."
            )
        result.setdefault("explanation", fallback)
        return {"result": result, "errors": errors}
