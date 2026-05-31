"""Direct answer handling for non-computational physics solutions."""

from __future__ import annotations

from typing import Any

from agents.workflows.state import WorkflowExecutionError


class DirectAnswerHandler:
    """Validate and format direct physics answer specs."""

    def handle(
        self,
        parsed_question: dict[str, Any],
        solution_output: dict[str, Any],
    ) -> dict[str, Any]:
        answer_type = solution_output.get("answer_type")
        direct = solution_output.get("direct_answer") or {}
        if answer_type not in {"yes_no", "multiple_choice", "conceptual"}:
            raise WorkflowExecutionError("Direct physics solution has an unsupported answer_type.")
        if direct.get("answer") is None or not isinstance(direct.get("rationale_steps"), list):
            raise WorkflowExecutionError("Direct physics solution answer/rationale_steps are invalid.")

        answer = direct.get("answer", "Unknown")
        selected_option = str(direct.get("selected_option") or "").strip()
        parsed_options = parsed_question.get("options") if isinstance(parsed_question, dict) else None
        if answer_type == "multiple_choice" and selected_option and isinstance(parsed_options, list):
            option_text = next(
                (
                    str(option.get("text") or "").strip()
                    for option in parsed_options
                    if isinstance(option, dict) and str(option.get("label") or "").strip() == selected_option
                ),
                "",
            )
            if option_text:
                answer = f"{selected_option}. {option_text}"

        cot = [str(step) for step in direct.get("rationale_steps") or []]
        final_answer = {"symbol": "answer", "value": str(answer), "unit": ""}
        return {
            "verified_output": {
                "mode": "direct",
                "answer_type": answer_type,
                "final_answer": final_answer,
                "solution_output": solution_output,
            },
            "result": {
                "answer": str(answer),
                "unit": "",
                "append_unit": False,
                "explanation": " ".join(cot) or "A direct physics conclusion was selected.",
                "cot": cot,
                "premises": [],
                "fol": "",
            },
        }
