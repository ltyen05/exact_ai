from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents.llm_client import VLLMClient
from agents.logic_agent import LogicAgent
from agents.physics_agent import PhysicsAgent
from agents.router import RouterAgent


class ExactPipeline:
    def __init__(self, physics_kb_path: Optional[str] = None):
        self.llm = VLLMClient()
        self.router = RouterAgent()
        self.logic = LogicAgent(self.llm)
        self.physics = PhysicsAgent(kb_path=physics_kb_path, llm=self.llm)

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        q = payload.get("question") or payload.get("query")
        if isinstance(q, list):
            raise ValueError("Payload contains a list of questions. Use predict_record or explode before calling predict.")
        question = str(q or "")
        route = self.router.classify(payload)
        if route == "logic":
            premises_nl = payload.get("premises-NL") or payload.get("premises_nl") or payload.get("premises") or []
            premises_fol = payload.get("premises-FOL") or payload.get("premises_fol")
            res = self.logic.solve(question=question, premises_nl=list(premises_nl), premises_fol=premises_fol)
        else:
            res = self.physics.solve(question=question)
        res["type"] = route
        return res

    def predict_record(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        if isinstance(record.get("questions"), list):
            outputs = []
            for i, q in enumerate(record["questions"]):
                payload = dict(record)
                payload["question"] = q
                payload.pop("questions", None)
                out = self.predict(payload)
                out["question_index"] = i
                outputs.append(out)
            return outputs
        return [self.predict(record)]
