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
from tools.z3_logic import Atom, HornKB, Rule, parse_fol_to_kb, pred_name

from .orchestrator import WorkflowState
from .tracing import trace_step

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LOGIC_KB = _PROJECT_ROOT / "data" / "Logic_Based_Educational_Queries.json"

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
                    "premises-FOL": record.get("premises-FOL", []),
                    "questions": record.get("questions", []),
                    "answers": record.get("answers", []),
                    "explanation": record.get("explanation", []),
                }
            )
        return output

    def best_fol_if_same_premises(
        self,
        premises_nl: list[str],
        retrieved: list[dict[str, Any]],
    ) -> list[str] | None:
        premise_set = {premise.strip().lower() for premise in premises_nl or []}
        if not premise_set:
            return None
        for record in retrieved:
            record_set = {str(premise).strip().lower() for premise in record.get("premises-NL", [])}
            fol = record.get("premises-FOL") or []
            if fol and record_set == premise_set:
                return [str(item) for item in fol]
        return None


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
        resolved_rag_path = Path(rag_path) if rag_path else _DEFAULT_LOGIC_KB
        self.rag = LogicRAGRetriever(resolved_rag_path if resolved_rag_path.exists() else None)
        self.use_rag = use_rag
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        graph = StateGraph(WorkflowState)
        graph.add_node("extract_logic", self.extract_logic)
        graph.add_node("convert_to_fol", self.convert_to_fol)
        graph.add_node("verify_z3", self.verify_z3)
        graph.add_node("explain_logic", self.explain_logic)
        graph.add_edge(START, "extract_logic")
        graph.add_edge("extract_logic", "convert_to_fol")
        graph.add_edge("convert_to_fol", "verify_z3")
        graph.add_edge("verify_z3", "explain_logic")
        graph.add_edge("explain_logic", END)
        return graph.compile()

    @trace_step("logic.extract_logic")
    def extract_logic(self, state: WorkflowState) -> dict[str, Any]:
        """Collect logic context, question type, and optional few-shot examples."""
        premises = list(state.get("premises", []))
        premises_fol = list(state.get("premises_fol") or [])
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
                "premises_fol": premises_fol,
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
                    "similar_premises_fol": example.get("premises-FOL", [])[:12],
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
        choices = formalized.get("choices")
        if isinstance(choices, dict) and choices:
            for label, value in choices.items():
                atom = self._atom_from_json(value)
                if atom and kb.entails(atom) is True:
                    return str(label), atom.label(), list(kb.trace.get(atom, [])), 0.85
            answer, atom, confidence = self._best_choice(kb, question, formalized)
            return answer, atom.label() if atom else "", list(kb.trace.get(atom, [])) if atom else [], confidence
        if split_choices(question):
            answer, atom, confidence = self._best_choice(kb, question, formalized)
            return answer, atom.label() if atom else "", list(kb.trace.get(atom, [])) if atom else [], confidence
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
        premises_fol = list(logic_spec.get("premises_fol") or [])
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
                    answer, atom, confidence = self._best_choice(kb, state["question"], None)
                else:
                    answer, atom, confidence = self._yes_no_unknown(kb, state["question"], None)
                fol = atom.label() if atom else ""
                fallback_steps = [
                    f"Classified logic question as {logic_spec.get('question_type', 'OpenEnded')}.",
                    "Parsed premises with the deterministic Horn-rule fallback.",
                    "Ran forward chaining and entailment checks.",
                ]
                cot = [*list(kb.trace.get(atom, [])), *fallback_steps] if atom else fallback_steps

            fol_fallback_used = False
            if (answer == "Unknown" or confidence < 0.55) and premises_fol:
                fol_kb = parse_fol_to_kb(premises_fol)
                if split_choices(state["question"]):
                    fol_answer, fol_atom, fol_confidence = self._best_choice(
                        fol_kb,
                        state["question"],
                        formalized if isinstance(formalized, dict) else None,
                    )
                else:
                    fol_answer, fol_atom, fol_confidence = self._yes_no_unknown(
                        fol_kb,
                        state["question"],
                        formalized if isinstance(formalized, dict) else None,
                    )
                if fol_answer != "Unknown" or fol_confidence > confidence:
                    kb = fol_kb
                    answer = fol_answer
                    fol = fol_atom.label() if fol_atom else fol
                    cot = list(kb.trace.get(fol_atom, [])) if fol_atom else cot
                    confidence = max(fol_confidence, 0.78)
                    fol_fallback_used = True

            rag_fol_used = False
            if (answer == "Unknown" or confidence < 0.55) and fewshot_examples:
                rag_fol = self.rag.best_fol_if_same_premises(premises, fewshot_examples)
                if rag_fol:
                    rag_kb = parse_fol_to_kb(rag_fol)
                    if split_choices(state["question"]):
                        rag_answer, rag_atom, rag_confidence = self._best_choice(rag_kb, state["question"], formalized if isinstance(formalized, dict) else None)
                    else:
                        rag_answer, rag_atom, rag_confidence = self._yes_no_unknown(rag_kb, state["question"], formalized if isinstance(formalized, dict) else None)
                    if rag_answer != "Unknown" or rag_confidence > confidence:
                        kb = rag_kb
                        answer = rag_answer
                        fol = rag_atom.label() if rag_atom else ""
                        cot = list(kb.trace.get(rag_atom, [])) if rag_atom else cot
                        confidence = max(rag_confidence, 0.78)
                        rag_fol_used = True

            if fewshot_examples:
                cot = [
                    *cot,
                    "RAG retrieved similar logic examples and used them only as parser guidance.",
                ]
            if fol_fallback_used:
                cot = [
                    *cot,
                    "Provided FOL premises were used as a conservative symbolic fallback.",
                ]
            if rag_fol_used:
                cot = [
                    *cot,
                    "RAG FOL fallback was used only because a retrieved example had matching premises.",
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
                    "rag_fol_used": rag_fol_used,
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

    @trace_step("logic.explain_logic")
    def explain_logic(self, state: WorkflowState) -> dict[str, Any]:
        """Explain a verified logic result without modifying its answer."""
        result = dict(state.get("result", {}))
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
