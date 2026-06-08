"""Logic domain agents and utilities."""

from __future__ import annotations

from agents.logic.symbcot import LogicSymbCoTAgent, LogicSymbCoTProvider

__all__ = [
    "LogicAgent",
    "LogicWorkflow",
    "LogicRAGRetriever",
    "LogicSymbCoTAgent",
    "LogicSymbCoTProvider",
    "classify_logic_question",
    "prompt_for_type",
]


def __getattr__(name: str):
    if name == "LogicAgent":
        from agents.logic.agent import LogicAgent

        return LogicAgent
    if name in {"LogicWorkflow", "LogicRAGRetriever", "classify_logic_question", "prompt_for_type"}:
        from agents.workflows.logic import (
            LogicRAGRetriever,
            LogicWorkflow,
            classify_logic_question,
            prompt_for_type,
        )

        values = {
            "LogicWorkflow": LogicWorkflow,
            "LogicRAGRetriever": LogicRAGRetriever,
            "classify_logic_question": classify_logic_question,
            "prompt_for_type": prompt_for_type,
        }
        return values[name]
    raise AttributeError(name)
