"""Shared LangGraph state and top-level workflow nodes."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.classify import TfidfLogisticClassifier

from .state import WorkflowExecutionError, WorkflowState
from .tracing import trace_step


class ClassifyNode:
    """Route a question through the trained classifier in Classify_Agent.py."""

    def __init__(self, classifier: TfidfLogisticClassifier | None = None) -> None:
        """Store a supplied classifier or configure the default trained classifier."""
        self.classifier = classifier or TfidfLogisticClassifier()

    @trace_step("router.classify")
    def __call__(self, state: WorkflowState) -> dict[str, Any]:
        """Classify the current question into its workflow route."""
        classified = self.classifier.run(state["question"])
        return {"route": classified["Type"]}


class FormatterNode:
    """Unify outputs from both subgraphs into the public response contract."""

    @staticmethod
    def _format_cot(value: Any) -> list[str] | None:
        if isinstance(value, str):
            text = value.strip()
            return [line.strip() for line in text.splitlines() if line.strip()] or None
        if isinstance(value, list):
            steps = [str(step).strip() for step in value if str(step).strip()]
            return steps or None
        return None

    @trace_step("formatter.output")
    def __call__(self, state: WorkflowState) -> dict[str, Any]:
        """Publish only answer, explanation and encouraged reasoning fields."""
        result = state.get("result", {})
        explanation = str(result.get("explanation", "No solution produced."))
        answer = str(result.get("answer", "Unknown"))
        unit = str(result.get("unit") or "")
        if (
            answer != "Unknown"
            and result.get("append_unit")
            and unit
            and unit.lower() != "dimensionless"
            and not answer.endswith(f" {unit}")
        ):
            answer = f"{answer} {unit}"
        cot = result.get("cot") if isinstance(result.get("cot"), list) else []
        premises = result.get("premises") if isinstance(result.get("premises"), list) else []
        output: dict[str, Any] = {
            "answer": answer,
            "explanation": explanation,
        }
        fol = str(result.get("fol") or "").strip()
        if fol:
            output["fol"] = fol
        cot_steps = self._format_cot(cot)
        if cot_steps:
            output["cot"] = cot_steps
        if premises:
            output["premises"] = [str(premise) for premise in premises]
        return {"output": output}


class ExactGraph:
    """Route each request through one traced LangGraph domain subgraph."""

    def __init__(
        self,
        llm: Any = None,
        physics_kb_path: str | None = None,
        classifier: TfidfLogisticClassifier | None = None,
        use_abstract_templates: bool = True,
    ) -> None:
        """Configure the nested workflows and compile the orchestration graph."""
        from .logic import LogicWorkflow
        from .physics import PhysicsWorkflow

        self.llm = llm
        self.physics_kb_path = physics_kb_path
        self.use_abstract_templates = use_abstract_templates
        self.router = ClassifyNode(classifier)
        self.physics = PhysicsWorkflow(llm=llm, kb_path=physics_kb_path)
        self.logic = LogicWorkflow(llm=llm)
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

    @trace_step("workflow.predict")
    def predict(
        self,
        payload: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke one traceable graph run and return formatted output."""
        unknown_fields = sorted(set(payload) - {"question", "premises"})
        if unknown_fields:
            raise ValueError(f"Unsupported input fields: {', '.join(unknown_fields)}.")
        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("A non-empty question is required.")
        premises = payload.get("premises") or []
        if not isinstance(premises, list) or not all(isinstance(item, str) for item in premises):
            raise ValueError("premises must be an array of strings.")
        run_config = {
            "run_name": "exact_2026_workflow",
            "tags": ["exact-2026", "langgraph"],
        }
        if config:
            run_config.update(config)
        final_state = self.graph.invoke(
            {
                "question": question.strip(),
                "premises": list(premises),
                "errors": [],
            },
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
            if isinstance(record.get("premises"), list):
                payload["premises"] = record["premises"]
            output = self.predict(payload)
            output["question_index"] = index
            outputs.append(output)
        return outputs
