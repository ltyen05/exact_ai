"""LangGraph logic subgraph: formalize, verify, and explain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.formatting import extract_json
from tools.z3_logic import Atom, HornKB, Rule, pred_name

from .state import WorkflowState
from .tracing import trace_step

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _llm_available(llm: Any) -> bool:
    return llm is not None and bool(getattr(llm, "enabled", True))


def _with_error(state: WorkflowState, error: str) -> list[str]:
    return [*state.get("errors", []), error]


class LogicWorkflow:
    """Build and execute logic verification over a compact Horn-rule schema."""

    def __init__(
        self,
        llm: Any = None,
        fol_prompt_path: str | Path | None = None,
        explanation_prompt_path: str | Path | None = None,
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
        """Collect the only public logic context: question and NL premises."""
        return {"logic_spec": {"premises": list(state.get("premises", []))}}

    @trace_step("logic.convert_to_fol")
    def convert_to_fol(self, state: WorkflowState) -> dict[str, Any]:
        """Convert natural language into compact Horn-rule JSON using the prompt file."""
        logic_spec = dict(state.get("logic_spec", {}))
        if not _llm_available(self.llm):
            return {"logic_spec": logic_spec}
        prompt_input = {
            "premises": logic_spec.get("premises", []),
            "question": state["question"],
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
            return {"logic_spec": logic_spec}
        except Exception as exc:
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
        return "Formalized premise"

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

    def _verify_formalized(
        self,
        kb: HornKB,
        formalized: dict[str, Any],
    ) -> tuple[str, str, list[str]]:
        choices = formalized.get("choices")
        if isinstance(choices, dict) and choices:
            for label, value in choices.items():
                atom = self._atom_from_json(value)
                if atom and kb.entails(atom) is True:
                    return str(label), atom.label(), list(kb.trace.get(atom, []))
            return "Unknown", "", []
        atom = self._atom_from_json(formalized.get("query"))
        if atom is None:
            return "Unknown", "", []
        entailed = kb.entails(atom)
        if entailed is True:
            return "Yes", atom.label(), list(kb.trace.get(atom, []))
        if entailed is False:
            return "No", atom.label(), list(kb.trace.get(atom.neg(), []))
        return "Unknown", atom.label(), []

    @trace_step("logic.verify_z3")
    def verify_z3(self, state: WorkflowState) -> dict[str, Any]:
        """Construct a Horn knowledge base and verify the formalized query."""
        logic_spec = state.get("logic_spec", {})
        formalized = logic_spec.get("formalized")
        premises = list(logic_spec.get("premises", []))
        if not isinstance(formalized, dict):
            return {
                "result": {
                    "answer": "Unknown",
                    "unit": "",
                    "explanation": "No formalized logic specification was available for verification.",
                    "premises": premises,
                }
            }
        try:
            kb = self._build_kb(formalized, premises)
            answer, fol, cot = self._verify_formalized(kb, formalized)
            return {
                "result": {
                    "answer": answer,
                    "unit": "",
                    "fol": fol,
                    "cot": cot,
                    "premises": premises,
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
        result.setdefault(
            "explanation",
            f"Horn-rule verification result for {fol or 'the requested conclusion'}: {result.get('answer', 'Unknown')}.",
        )
        return {"result": result, "errors": errors}
