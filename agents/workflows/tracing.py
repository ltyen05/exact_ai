"""LangSmith tracing decorators with compact workflow observability."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TypeVar

import langsmith as ls

F = TypeVar("F", bound=Callable[..., Any])

_BLOCKED_KEYS = {"confidence", "metadata", "raw_type", "source", "tracing"}
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _tracing_enabled() -> bool:
    """Keep submission runs free of tracing network calls unless explicitly enabled."""
    return os.getenv("EXACT_ENABLE_LANGSMITH", "").strip().lower() in _TRUE_VALUES


def _clean(value: Any) -> Any:
    """Remove disallowed diagnostics from a compact trace value."""
    if isinstance(value, dict):
        return {
            key: _clean(item)
            for key, item in value.items()
            if key not in _BLOCKED_KEYS
        }
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def _state_input_summary(state: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("question", "premises", "route"):
        value = state.get(key)
        if value:
            summary[key] = _clean(value)
    parsed = state.get("parsed_question")
    if isinstance(parsed, dict):
        summary["parsed_question"] = {
            key: _clean(parsed[key])
            for key in ("domain", "target", "question_kind")
            if parsed.get(key)
        }
    return summary


def _process_step_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    state = inputs.get("state")
    if isinstance(state, dict):
        return _state_input_summary(state)
    payload = inputs.get("payload")
    if isinstance(payload, dict):
        return {
            key: _clean(payload[key])
            for key in ("question", "premises")
            if payload.get(key)
        }
    return {}


def _process_step_outputs(output: Any) -> dict[str, Any]:
    if not isinstance(output, dict):
        return {"status": "uncompleted"}
    if isinstance(output.get("output"), dict):
        return {"output": _clean(output["output"])}
    if output.get("route"):
        return {"route": output["route"]}
    if isinstance(output.get("parsed_question"), dict):
        return {"parsed_question": _clean(output["parsed_question"])}
    if isinstance(output.get("solution_output"), dict):
        solution = output["solution_output"]
        return {
            "solution_output": _clean(
                {
                    "mode": solution.get("mode"),
                    "answer_type": solution.get("answer_type"),
                    "sympy_spec": solution.get("sympy_spec"),
                    "decision_spec": solution.get("decision_spec"),
                }
            )
        }
    if isinstance(output.get("result"), dict):
        return {"result": _clean(output["result"])}
    return _clean(output)


def _message_chars(payload: dict[str, Any]) -> int:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return 0
    return sum(
        len(str(message.get("content") or ""))
        for message in messages
        if isinstance(message, dict)
    )


def _process_llm_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    client = inputs.get("self")
    payload = inputs.get("payload") if isinstance(inputs.get("payload"), dict) else {}
    reasoning = payload.get("reasoning") if isinstance(payload.get("reasoning"), dict) else {}
    effort = str(reasoning.get("effort") or "") if isinstance(reasoning, dict) else ""
    return {
        "stage": str(inputs.get("stage") or "llm"),
        "provider": str(getattr(client, "provider", "unknown")),
        "model": str(getattr(client, "model", "unknown")),
        "json_mode": bool(payload.get("response_format")),
        "thinking_disabled": effort == "none",
        "attempt": int(inputs.get("attempt") or 1),
        "request_chars": _message_chars(payload),
    }


def _process_llm_outputs(output: Any) -> dict[str, Any]:
    if not isinstance(output, dict):
        return {"status": "unknown"}
    return {
        key: output[key]
        for key in ("status", "duration_ms", "response_chars")
        if key in output
    }


def trace_step(name: str) -> Callable[[F], F]:
    """Decorate one workflow step while storing only compact business data."""

    def decorate(function: F) -> F:
        if not _tracing_enabled():
            return function
        return ls.traceable(
            name=name,
            process_inputs=_process_step_inputs,
            process_outputs=_process_step_outputs,
        )(function)  # type: ignore[return-value]

    return decorate


def trace_llm_attempt(function: F) -> F:
    """Decorate one HTTP LLM attempt without persisting prompt content."""
    if not _tracing_enabled():
        return function
    return ls.traceable(
        name="llm.http_attempt",
        run_type="llm",
        process_inputs=_process_llm_inputs,
        process_outputs=_process_llm_outputs,
    )(function)  # type: ignore[return-value]
