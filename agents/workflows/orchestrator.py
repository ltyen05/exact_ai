"""Shared LangGraph state and top-level workflow nodes."""

from __future__ import annotations

import re
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.classify import TfidfLogisticClassifier

from .state import WorkflowExecutionError, WorkflowState
from .tracing import trace_step


class ClassifyNode:
    """Route by explicit competition type, with classifier fallback for old calls."""

    def __init__(self, classifier: TfidfLogisticClassifier | None = None) -> None:
        """Store a supplied classifier or configure the default trained classifier."""
        self.classifier = classifier or TfidfLogisticClassifier()

    @trace_step("router.classify")
    def __call__(self, state: WorkflowState) -> dict[str, Any]:
        """Classify the current question into its workflow route."""
        route = state.get("route")
        if route in {"logic", "physics"}:
            return {"route": route}
        classified = self.classifier.run(state["question"])
        return {"route": classified["Type"]}


class FormatterNode:
    """Unify outputs from both subgraphs into the public response contract."""

    _OPTION_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    @staticmethod
    def _format_cot(value: Any) -> list[str] | None:
        if isinstance(value, str):
            text = value.strip()
            return [line.strip() for line in text.splitlines() if line.strip()] or None
        if isinstance(value, list):
            steps = [str(step).strip() for step in value if str(step).strip()]
            return steps or None
        return None

    @staticmethod
    def _ascii_unit(unit: Any) -> str:
        text = str(unit or "").strip()
        if not text or text.lower() == "dimensionless":
            return ""
        replacements = {
            "\\Omega": "ohm",
            "Ω": "ohm",
            "Ω": "ohm",
            "\\mu": "u",
            "μ": "u",
            "µ": "u",
            "Ohm": "ohm",
            "ohms": "ohm",
            "micro": "u",
            "degree": "deg",
            "degrees": "deg",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text.strip()

    @classmethod
    def _match_option(cls, answer: str, options: list[str]) -> str:
        if not options:
            return answer
        normalized = answer.strip().lower()
        for option in options:
            if normalized == option.strip().lower():
                return option
        aliases = {
            "unknown": ("uncertain", "cannot be determined", "not enough information"),
            "uncertain": ("unknown", "cannot be determined", "not enough information"),
            "true": ("yes",),
            "false": ("no",),
        }
        candidates = aliases.get(normalized, ())
        for option in options:
            if option.strip().lower() in candidates:
                return option
        label_match = re.match(r"^\s*([A-Z])(?:[.)]|\b)", answer.strip(), flags=re.I)
        if label_match:
            index = cls._OPTION_LABELS.find(label_match.group(1).upper())
            if 0 <= index < len(options):
                return options[index]
        return answer

    @staticmethod
    def _strip_unit(answer: str, unit: str) -> str:
        if not unit:
            return answer
        return re.sub(rf"\s+{re.escape(unit)}\s*$", "", answer).strip()

    @staticmethod
    def _premises_used(result: dict[str, Any], query_type: str) -> list[int]:
        if query_type == "type2":
            return []
        raw = result.get("premises_used")
        if isinstance(raw, list):
            return sorted({int(item) for item in raw if isinstance(item, int) and item >= 0})
        idx = result.get("idx")
        if not isinstance(idx, list):
            return []
        used: set[int] = set()
        for item in idx:
            if isinstance(item, int) and item > 0:
                used.add(item - 1)
        return sorted(used)

    @trace_step("formatter.output")
    def __call__(self, state: WorkflowState) -> dict[str, Any]:
        """Publish the unified EXACT 2026 result object."""
        result = state.get("result", {})
        explanation = str(result.get("explanation", "No solution produced."))
        answer = str(result.get("answer", "Unknown"))
        query_type = str(state.get("query_type") or "")
        unit = "" if query_type == "type1" else self._ascii_unit(result.get("unit"))
        if query_type == "type2":
            answer = self._strip_unit(answer, unit)
        cot = result.get("cot") if isinstance(result.get("cot"), list) else []
        options = [str(option) for option in state.get("options", []) if isinstance(option, str)]
        if query_type == "type1":
            answer = self._match_option(answer, options)
        cot_steps = self._format_cot(cot)
        reasoning_type = "cot" if query_type == "type2" or state.get("route") == "physics" else "fol"
        output: dict[str, Any] = {
            "query_id": str(state.get("query_id") or ""),
            "answer": answer,
            "unit": unit,
            "explanation": explanation,
            "premises_used": self._premises_used(result, query_type),
            "reasoning": {"type": reasoning_type, "steps": cot_steps} if cot_steps else None,
        }
        return {"output": output}


class ExactGraph:
    """Route each request through one traced LangGraph domain subgraph."""

    def __init__(
        self,
        llm: Any = None,
        physics_kb_path: str | None = None,
        logic_kb_path: str | None = None,
        classifier: TfidfLogisticClassifier | None = None,
        use_abstract_templates: bool = True,
        use_rag: bool = True,
    ) -> None:
        """Configure the nested workflows and compile the orchestration graph."""
        from .logic import LogicWorkflow
        from .physics import PhysicsWorkflow

        self.llm = llm
        self.physics_kb_path = physics_kb_path
        self.logic_kb_path = logic_kb_path
        self.use_abstract_templates = use_abstract_templates
        self.use_rag = use_rag
        self.router = ClassifyNode(classifier)
        self.physics = PhysicsWorkflow(llm=llm, kb_path=physics_kb_path)
        self.logic = LogicWorkflow(llm=llm, rag_path=logic_kb_path, use_rag=use_rag)
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Build router, conditional subgraphs, and formatter as one LangGraph."""
        graph = StateGraph(WorkflowState)
        graph.add_node("classify_route", self.router)
        graph.add_node("physics_subgraph", self.physics.graph)
        graph.add_node("logic_subgraph", self.logic.graph)
        graph.add_node("format_output", FormatterNode())
        graph.add_edge(START, "classify_route")
        graph.add_conditional_edges(
            "classify_route",
            lambda state: state["route"],
            {"physics": "physics_subgraph", "logic": "logic_subgraph"},
        )
        graph.add_edge("physics_subgraph", "format_output")
        graph.add_edge("logic_subgraph", "format_output")
        graph.add_edge("format_output", END)
        return graph.compile()

    @staticmethod
    def _route_from_type(query_type: str) -> str:
        if query_type == "type1":
            return "logic"
        if query_type == "type2":
            return "physics"
        raise ValueError("type must be either 'type1' or 'type2'.")

    @classmethod
    def _question_with_options(cls, question: str, options: list[str]) -> str:
        if not options:
            return question
        normalized_options = {option.strip().lower() for option in options}
        if normalized_options <= {"yes", "no", "uncertain", "unknown", "true", "false"}:
            return question
        if all(option and option in question for option in options):
            return question
        lines = [question.rstrip(), "", "Options:"]
        for index, option in enumerate(options):
            label = FormatterNode._OPTION_LABELS[index] if index < len(FormatterNode._OPTION_LABELS) else str(index + 1)
            lines.append(f"{label}. {option}")
        return "\n".join(lines)

    @classmethod
    def _normalize_payload(cls, payload: dict[str, Any]) -> dict[str, Any]:
        if "query" in payload or "type" in payload:
            unknown_fields = sorted(set(payload) - {"query_id", "type", "query", "premises", "options"})
            if unknown_fields:
                raise ValueError(f"Unsupported input fields: {', '.join(unknown_fields)}.")
            query_id = payload.get("query_id")
            query_type = payload.get("type")
            question = payload.get("query")
            premises = payload.get("premises")
            options = payload.get("options")
            if not isinstance(query_id, str):
                raise ValueError("query_id must be a string.")
            if not isinstance(query_type, str):
                raise ValueError("type must be a string.")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("A non-empty query is required.")
            if not isinstance(premises, list) or not all(isinstance(item, str) for item in premises):
                raise ValueError("premises must be an array of strings.")
            if not isinstance(options, list) or not all(isinstance(item, str) for item in options):
                raise ValueError("options must be an array of strings.")
            route = cls._route_from_type(query_type)
            return {
                "query_id": query_id,
                "query_type": query_type,
                "question": cls._question_with_options(question.strip(), list(options)) if route == "logic" else question.strip(),
                "premises": list(premises),
                "options": list(options),
                "route": route,
                "errors": [],
            }

        unknown_fields = sorted(set(payload) - {"question", "premises"})
        if unknown_fields:
            raise ValueError(f"Unsupported input fields: {', '.join(unknown_fields)}.")
        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("A non-empty question is required.")
        premises = payload.get("premises") or []
        if not isinstance(premises, list) or not all(isinstance(item, str) for item in premises):
            raise ValueError("premises must be an array of strings.")
        return {
            "query_id": "",
            "query_type": "",
            "question": question.strip(),
            "premises": list(premises),
            "options": [],
            "errors": [],
        }

    @trace_step("workflow.predict")
    def predict(
        self,
        payload: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke one traceable graph run and return formatted output."""
        initial_state = self._normalize_payload(payload)
        run_config = {
            "run_name": "exact_2026_workflow",
            "tags": ["exact-2026", "langgraph"],
        }
        if config:
            run_config.update(config)
        final_state = self.graph.invoke(
            initial_state,
            config=run_config,
        )
        return final_state["output"]

    def predict_record(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        """Process a record once per contained question."""
        questions = record.get("questions")
        if not isinstance(questions, list):
            return [self.predict(record)]
        outputs: list[dict[str, Any]] = []
        for index, question in enumerate(questions):
            payload = {"question": question}
            premises = record.get("premises")
            if not isinstance(premises, list):
                premises = record.get("premises-NL")
            if isinstance(premises, list):
                payload["premises"] = premises
            output = self.predict(payload)
            output["question_index"] = index
            outputs.append(output)
        return outputs
