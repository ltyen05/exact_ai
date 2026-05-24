from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel

from agents.pipeline import ExactPipeline

ROOT = Path(__file__).resolve().parent
pipe = ExactPipeline(physics_kb_path=str(ROOT / "data" / "Physics_Problems_Text_Only_removeQA.json"))
app = FastAPI(title="EXACT 2026 Multi-Agent QA")


class QueryPayload(BaseModel):
    type: str | None = None
    query_type: str | None = None
    question: str
    premises_NL: list[str] | None = None
    premises: list[str] | None = None
    premises_FOL: list[str] | None = None
    id: str | None = None


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Normalize possible API names into dataset names.
    if "premises_NL" in payload and "premises-NL" not in payload:
        payload["premises-NL"] = payload.pop("premises_NL")
    if "premises_FOL" in payload and "premises-FOL" not in payload:
        payload["premises-FOL"] = payload.pop("premises_FOL")
    return pipe.predict(payload)
