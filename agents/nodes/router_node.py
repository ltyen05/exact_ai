"""
Router node for LangGraph - classifies queries into physics or logic (edu) using ML.
"""

from __future__ import annotations

from agents.models.state import WorkflowState
from agents.router import RouterAgent


class RouterNode:
    """Route queries to Physics or Logic (Educational) processing using ML Classifier."""

    def __init__(self):
        self.router = RouterAgent()

    def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Router node for LangGraph.
        Classifies query using TF-IDF + Naive Bayes and updates state.
        """
        # Get overall classification
        query_type = self.router.classify(state.original_payload)
        
        # Get ML prediction details for metadata
        ml_type, confidence = self.router.classify_with_confidence(state.question)

        state.query_type = query_type
        state.router_confidence = confidence
        state.metadata["router_classification"] = query_type
        state.metadata["router_ml_prediction"] = ml_type
        state.metadata["router_ml_confidence"] = confidence
        
        return state

