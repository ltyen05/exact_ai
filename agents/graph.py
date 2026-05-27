"""
LangGraph workflow for EXACT 2026 multi-agent system.

Graph structure:
  START
    ↓
  [Router Node] → Classifies query (logic/physics)
    ↓
  ├─ [Physics Subgraph]
  │   1. Extract quantities (LLM)
  │   2. Classify problem type (LLM)
  │   3. Select abstract/RAG (Retrieval)
  │   4. Compute (SymPy)
  │   5. Explain (LLM)
  │
  └─ [Logic Subgraph]
      1. Extract logic (LLM)
      2. Convert to FOL
      3. Verify with Z3
      4. Explain (LLM)
    ↓
  [Formatter Node] → Unify output format
    ↓
  END
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from langgraph.graph import StateGraph
except ImportError:
    raise ImportError("langgraph not installed. Run: pip install langgraph")

from agents.models.state import WorkflowState, AgentOutput
from agents.nodes.router_node import RouterNode
from agents.subgraphs.physics_subgraph import build_physics_subgraph
from agents.subgraphs.logic_subgraph import build_logic_subgraph
from agents.llm.base import LLMClientBase
from agents.formatter import standard_response


class FormatterNode:
    """Unify output format from different subgraphs."""

    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Format final output."""
        if not state.agent_output:
            state.agent_output = AgentOutput(
                answer="Unknown",
                reasoning="No solution produced",
                confidence=0.0,
                agent_name="Formatter",
            )
        return state


class ExactGraph:
    """
    LangGraph-based workflow for EXACT 2026.
    
    Orchestrates:
    - Router: Classify physics/logic
    - Physics Subgraph: Multi-step physics problem solving
    - Logic Subgraph: Multi-step logic problem solving
    - Formatter: Unify output
    """

    def __init__(self, llm: LLMClientBase, physics_kb_path: Optional[str] = None, use_abstract_templates: bool = True):
        self.llm = llm
        self.physics_kb_path = physics_kb_path
        self.use_abstract_templates = use_abstract_templates

        # Build subgraphs
        self.physics_graph = build_physics_subgraph(llm, physics_kb_path, use_abstract_templates=use_abstract_templates)
        self.logic_graph = build_logic_subgraph(llm)

        # Build main graph
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Build the main LangGraph workflow."""
        graph = StateGraph(WorkflowState)

        # Create nodes
        router = RouterNode()
        formatter = FormatterNode()

        # Add nodes
        graph.add_node("router", router)
        graph.add_node("physics_subgraph", self._physics_subgraph_node)
        graph.add_node("logic_subgraph", self._logic_subgraph_node)
        graph.add_node("formatter", formatter)

        # Set entry point
        graph.set_entry_point("router")

        # Router decides destination
        graph.add_conditional_edges(
            "router",
            lambda state: "physics" if state.query_type == "physics" else "logic",
            {"physics": "physics_subgraph", "logic": "logic_subgraph"},
        )

        # Both subgraphs go to formatter
        graph.add_edge("physics_subgraph", "formatter")
        graph.add_edge("logic_subgraph", "formatter")

        # Formatter is endpoint
        graph.add_edge("formatter", "__end__")

        # Compile
        return graph.compile()

    def _physics_subgraph_node(self, state: WorkflowState) -> WorkflowState:
        """Execute physics subgraph."""
        res = self.physics_graph.invoke(state)
        is_dict = isinstance(res, dict)
        res_metadata = res.get("metadata", {}) if is_dict else getattr(res, "metadata", {})

        # Extract final answer from subgraph
        if res_metadata.get("compute_result"):
            compute = res_metadata["compute_result"]
            explanation = res_metadata.get("explanation", "")

            # Create final output
            state.agent_output = AgentOutput(
                answer=compute.get("answer", "Unknown"),
                reasoning=explanation or "Physics calculation completed",
                confidence=compute.get("confidence", 0.7),
                metadata={
                    "unit": compute.get("unit", ""),
                    "cot": compute.get("cot", []),
                    "problem_type": res_metadata.get("problem_type"),
                    "extract_quantities": res_metadata.get("extract_quantities"),
                    "abstract_source": res_metadata.get("abstract_source"),
                },
                agent_name="PhysicsAgent",
            )
        else:
            # No computation, might use RAG or fallback
            abstract_data = res_metadata.get("abstract_data", {})
            state.agent_output = AgentOutput(
                answer="Unknown",
                reasoning=res_metadata.get("explanation", "Could not solve physics problem"),
                confidence=0.3,
                metadata={
                    "approach": abstract_data.get("approach", "Unknown"),
                    "abstract_source": res_metadata.get("abstract_source"),
                },
                agent_name="PhysicsAgent",
            )

        state.physics_output = state.agent_output
        state.metadata.update(res_metadata)
        return state

    def _logic_subgraph_node(self, state: WorkflowState) -> WorkflowState:
        """Execute logic subgraph."""
        res = self.logic_graph.invoke(state)
        is_dict = isinstance(res, dict)
        res_agent_output = res.get("agent_output") if is_dict else getattr(res, "agent_output", None)
        res_metadata = res.get("metadata", {}) if is_dict else getattr(res, "metadata", {})

        # Extract final answer from subgraph
        if res_agent_output:
            state.agent_output = res_agent_output
        else:
            state.agent_output = AgentOutput(
                answer="Unknown",
                reasoning="Logic reasoning incomplete",
                confidence=0.0,
                agent_name="LogicAgent",
            )

        state.logic_output = state.agent_output
        state.metadata.update(res_metadata)
        return state

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the workflow on a single query.

        Args:
            payload: Query payload with question, type, premises, etc.

        Returns:
            Standardized output dict
        """
        # Create initial state
        state = WorkflowState(
            original_payload=payload,
            question=str(payload.get("question") or payload.get("query") or ""),
            premises_nl=payload.get("premises-NL") or payload.get("premises_nl") or payload.get("premises") or [],
            premises_fol=payload.get("premises-FOL") or payload.get("premises_fol"),
            record_id=payload.get("id"),
        )

        # Run the graph
        result_state = self.graph.invoke(state)

        # Convert to output format
        if isinstance(result_state, dict):
            agent_output = result_state.get("agent_output")
            if isinstance(agent_output, dict):
                answer = agent_output.get("answer", "Unknown")
                reasoning = agent_output.get("reasoning", "")
                confidence = agent_output.get("confidence", 0.0)
                metadata = agent_output.get("metadata", {})
            elif agent_output is not None:
                answer = agent_output.answer
                reasoning = agent_output.reasoning
                confidence = agent_output.confidence
                metadata = agent_output.metadata
            else:
                answer = "Unknown"
                reasoning = ""
                confidence = 0.0
                metadata = {}

            return {
                "type": result_state.get("query_type"),
                "question": result_state.get("question"),
                "answer": answer,
                "reasoning": reasoning,
                "confidence": confidence,
                "metadata": metadata,
                "errors": result_state.get("errors", []),
            }
        
        return result_state.to_dict()

    def predict_record(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a record that may contain multiple questions.

        Args:
            record: Record with possibly multiple questions

        Returns:
            List of predictions, one per question
        """
        if isinstance(record.get("questions"), list):
            outputs = []
            for i, q in enumerate(record["questions"]):
                payload = dict(record)
                payload["question"] = q
                payload.pop("questions", None)
                out = self.predict(payload)
                out["question_index"] = i
                outputs.append(out)
            return outputs
        return [self.predict(record)]

