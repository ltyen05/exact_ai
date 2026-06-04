"""Compatibility wrapper for running the logic workflow as a domain agent."""

from __future__ import annotations

from typing import Any

from agents.workflows.logic import LogicWorkflow


class LogicAgent:
    """Expose the LangGraph logic workflow through the old `solve` interface."""

    def __init__(
        self,
        llm: Any = None,
        rag_path: str | None = None,
        use_rag: bool = True,
    ) -> None:
        self.workflow = LogicWorkflow(llm=llm, rag_path=rag_path, use_rag=use_rag)

    def solve(
        self,
        question: str,
        premises_nl: list[str],
        premises_fol: list[str] | None = None,
        exclude_record_index: int | None = None,
        exclude_idx: Any = None,
        use_rag: bool = True,
    ) -> dict[str, Any]:
        del premises_fol, exclude_record_index, exclude_idx
        previous_use_rag = self.workflow.use_rag
        self.workflow.use_rag = use_rag
        try:
            final_state = self.workflow.graph.invoke(
                {
                    "question": str(question or "").strip(),
                    "premises": list(premises_nl or []),
                    "errors": [],
                }
            )
        finally:
            self.workflow.use_rag = previous_use_rag
        return dict(final_state.get("result", {}))
