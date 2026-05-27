"""
Physics node for LangGraph - solves physics problems.
"""

from __future__ import annotations

from typing import Optional

from agents.llm.base import LLMClientBase
from agents.models.state import AgentOutput, WorkflowState, AgentInput
from agents.physics_agent import PhysicsAgent


class PhysicsNode:
    """Physics problem solver node."""

    def __init__(self, llm: LLMClientBase, kb_path: Optional[str] = None):
        self.physics_agent = PhysicsAgent(kb_path=kb_path, llm=llm)

    def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Physics node for LangGraph.
        Solves physics problems and returns answer with confidence.
        """
        try:
            agent_input = AgentInput(
                question=state.question,
                query_type="physics",
                payload=state.original_payload,
                id=state.record_id,
            )

            # Use existing physics agent logic
            result = self.physics_agent.solve(agent_input)

            # Convert to standardized output
            output = AgentOutput(
                answer=result.answer,
                reasoning=result.reasoning,
                confidence=result.confidence,
                metadata=result.metadata,
                agent_name="PhysicsAgent",
            )

            state.agent_output = output
            state.physics_output = output

        except Exception as e:
            state.add_error(f"Physics agent error: {str(e)}")
            state.agent_output = AgentOutput(
                answer="Unknown",
                reasoning=f"Error in physics reasoning: {str(e)}",
                confidence=0.0,
                agent_name="PhysicsAgent",
            )

        return state
