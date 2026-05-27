"""
Unified state management for LangGraph workflow.
All agents communicate through this state class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class BaseAgent(ABC):
    """Abstract base class for all agents to unify inputs and outputs."""

    @abstractmethod
    def solve(self, agent_input: AgentInput) -> AgentOutput:
        """Process the unified agent input and return a unified agent output."""
        pass


@dataclass
class AgentInput:
    """Standardized input for all agents."""
    question: str
    query_type: str  # "logic" or "physics"
    premises_nl: List[str] = field(default_factory=list)  # For logic
    premises_fol: Optional[List[str]] = None  # For logic - First Order Logic
    payload: Dict[str, Any] = field(default_factory=dict)  # Original payload
    id: Optional[str] = None  # Record ID


@dataclass
class AgentOutput:
    """Standardized output from all agents."""
    answer: str
    reasoning: str  # Chain of thought / explanation
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extra info (unit, cot, etc.)
    agent_name: str = ""  # Which agent produced this


@dataclass
class WorkflowState:
    """
    Central state object passed through LangGraph workflow.
    Nodes read input and write output to this state.
    Edges determine routing based on state content.
    """

    # Input
    original_payload: Dict[str, Any]
    question: str
    query_type: Optional[str] = None  # "logic" or "physics" - set by router node
    premises_nl: List[str] = field(default_factory=list)
    premises_fol: Optional[List[str]] = None
    record_id: Optional[str] = None

    # Processing states
    router_confidence: float = 0.0  # Confidence of the router decision
    parsed_kb: Any = None  # For logic: parsed knowledge base (HornKB object)

    # Outputs from agents
    agent_output: Optional[AgentOutput] = None
    logic_output: Optional[AgentOutput] = None
    physics_output: Optional[AgentOutput] = None

    # Error tracking
    errors: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, error: str) -> None:
        """Log an error during processing."""
        self.errors.append(error)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dict for JSON serialization (for API/output)."""
        return {
            "type": self.query_type,
            "question": self.question,
            "answer": self.agent_output.answer if self.agent_output else "Unknown",
            "reasoning": self.agent_output.reasoning if self.agent_output else "",
            "confidence": self.agent_output.confidence if self.agent_output else 0.0,
            "metadata": self.agent_output.metadata if self.agent_output else {},
            "errors": self.errors,
        }

    def route_to_agent(self) -> str:
        """
        Determine which agent to route to based on query_type.
        This replaces the old router logic - returns edge destination.
        """
        if self.query_type == "logic":
            return "logic_agent"
        elif self.query_type == "physics":
            return "physics_agent"
        else:
            return "physics_agent"  # Default fallback
