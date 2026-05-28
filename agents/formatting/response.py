"""Standard response shaping."""

from typing import Any, Dict


def standard_response(answer: Any, explanation: str, **extra: Any) -> Dict[str, Any]:
    out = {
        "answer": str(answer) if answer is not None else "Unknown",
        "explanation": explanation or "No explanation was generated.",
    }
    for k, v in extra.items():
        if v is not None:
            out[k] = v
    return out
