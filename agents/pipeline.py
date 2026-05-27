from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents.llm_client import LLMClient
from agents.logic_agent import LogicAgent
from agents.physics_agent import PhysicsAgent
from agents.router import RouterAgent
from agents.models.state import AgentInput


class ExactPipeline:
    def __init__(self, physics_kb_path: Optional[str] = None):
        self.llm = LLMClient()
        self.router = RouterAgent()
        self.logic = LogicAgent(self.llm)
        self.physics = PhysicsAgent(kb_path=physics_kb_path, llm=self.llm)

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        q = payload.get("question") or payload.get("query")
        if isinstance(q, list):
            raise ValueError("Payload contains a list of questions. Use predict_record or explode before calling predict.")
        question = str(q or "")
        route = self.router.classify(payload)
        
        agent_input = AgentInput(
            question=question,
            query_type=route,
            premises_nl=payload.get("premises-NL") or payload.get("premises_nl") or payload.get("premises") or [],
            premises_fol=payload.get("premises-FOL") or payload.get("premises_fol"),
            payload=payload,
            id=payload.get("id"),
        )
        
        if route == "logic":
            output = self.logic.solve(agent_input)
        else:
            output = self.physics.solve(agent_input)
            
        res = {
            "answer": output.answer,
            "explanation": output.reasoning,
            "confidence": output.confidence,
            "type": route,
        }
        res.update(output.metadata)
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

