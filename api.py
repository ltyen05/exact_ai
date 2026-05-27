from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

# Load environment variables
load_dotenv()

from agents.graph import ExactGraph
from agents.llm.openrouter_client import OpenRouterClient

ROOT = Path(__file__).resolve().parent
DEFAULT_PHYSICS = ROOT / "data" / "Physics_Problems_Text_Only_removeQA.json"

# Initialize LLM and graph
llm = OpenRouterClient()
graph = ExactGraph(llm=llm, physics_kb_path=str(DEFAULT_PHYSICS))

app = FastAPI(title="EXACT 2026 Multi-Agent QA with LangGraph")


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
    """Health check endpoint."""
    return {"status": "ok", "llm_enabled": llm.enabled}


@app.post("/predict")
def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict answer for a single query.
    Uses LangGraph workflow with routing based on query type.
    
    Example:
    {
        "type": "logic",
        "question": "Does John graduate with honors?",
        "premises-NL": ["If student has GPA > 3.5, they graduate with honors.", "John has GPA 3.8."]
    }
    """
    # Normalize possible API names into dataset names
    if "premises_NL" in payload and "premises-NL" not in payload:
        payload["premises-NL"] = payload.pop("premises_NL")
    if "premises_FOL" in payload and "premises-FOL" not in payload:
        payload["premises-FOL"] = payload.pop("premises_FOL")
    
    return graph.predict(payload)


@app.get("/info")
def info() -> Dict[str, Any]:
    """Get system information."""
    return {
        "version": "2.0-langgraph",
        "system": "EXACT 2026 Multi-Agent QA",
        "llm": {
            "enabled": llm.enabled,
            "model": llm.model if llm.enabled else None,
            "base_url": llm.base_url if llm.enabled else None,
        },
        "features": [
            "Logic problem solving with Z3",
            "Physics problem solving",
            "LangGraph workflow routing",
            "LangSmith tracing",
        ],
    }

