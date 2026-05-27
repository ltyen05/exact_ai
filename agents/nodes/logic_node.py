"""
Logic node for LangGraph - solves logic problems.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents.llm.base import LLMClientBase
from agents.models.state import AgentOutput, WorkflowState, AgentInput
from agents.logic_agent import LogicAgent


class LogicNode:
    """Logic problem solver node."""

    def __init__(self, llm: LLMClientBase):
        self.logic_agent = LogicAgent(llm)

    def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Logic node for LangGraph.
        Processes logical premises and returns answer with confidence.
        """
        try:
            agent_input = AgentInput(
                question=state.question,
                query_type="logic",
                premises_nl=state.premises_nl or [],
                premises_fol=state.premises_fol,
                payload=state.original_payload,
                id=state.record_id,
            )

            # Use unified logic agent
            result = self.logic_agent.solve(agent_input)

            state.agent_output = result
            state.logic_output = result

        except Exception as e:
            state.add_error(f"Logic agent error: {str(e)}")
            state.agent_output = AgentOutput(
                answer="Unknown",
                reasoning=f"Error in logic reasoning: {str(e)}",
                confidence=0.0,
                agent_name="LogicAgent",
            )

        return state

