import re
from typing import Any, Dict, Literal

Route = Literal["logic", "physics"]


class RouterAgent:
    """Route a unified query stream into Logic Type 1 or Physics Type 2."""

    PHYSICS_HINTS = re.compile(
        r"\b(calculate|find|determine|voltage|current|resistance|capacitor|capacitance|charge|electric|force|rlc|ohm|power|energy|field|circuit|μf|uf|pf|μc|uc|\bohm\b|Ω|\bN\b|\bV\b)\b",
        re.I,
    )

    def classify(self, payload: Dict[str, Any]) -> Route:
        t = str(payload.get("type") or payload.get("query_type") or "").lower()
        if t in {"logic", "type1", "type_1", "educational_logic"}:
            return "logic"
        if t in {"physics", "type2", "type_2"}:
            return "physics"
        if payload.get("premises-NL") or payload.get("premises") or payload.get("premises_nl"):
            return "logic"
        if "unit" in payload or str(payload.get("id", "")).startswith(("TD", "LD", "CH", "DD", "DT", "NL", "TH")):
            return "physics"
        q = str(payload.get("question") or payload.get("query") or "")
        if self.PHYSICS_HINTS.search(q):
            return "physics"
        # Multiple choice + educational rules usually belongs to logic.
        if re.search(r"\n\s*A\.", q):
            return "logic"
        return "logic"
