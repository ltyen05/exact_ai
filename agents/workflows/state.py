"""Shared workflow state and exceptions."""

from __future__ import annotations

from typing import Any, TypedDict


class WorkflowState(TypedDict, total=False):
    """Represent the data shared by the routed LangGraph nodes."""

    question: str
    query_id: str
    query_type: str
    premises: list[str]
    options: list[str]
    route: str
    parsed_question: dict[str, Any]
    solution_output: dict[str, Any]
    verified_output: dict[str, Any]
    logic_spec: dict[str, Any]
    result: dict[str, Any]
    errors: list[str]
    output: dict[str, Any]


class WorkflowExecutionError(RuntimeError):
    """Raised when a valid request cannot complete the selected workflow."""
