"""FastAPI entry point for the EXACT 2026 LangGraph workflow."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import uvicorn

from agents.llm import OpenRouterClient, VLLMClient
from agents.workflows import ExactGraph, WorkflowExecutionError


load_dotenv()
TRUE_VALUES = {"1", "true", "yes", "on"}
if os.getenv("EXACT_ENABLE_LANGSMITH", "").strip().lower() not in TRUE_VALUES:
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

ROOT = Path(__file__).resolve().parent
DEFAULT_PHYSICS_KB = ROOT / "data" / "Physics_Problems_Text_Only_removeQA.json"
DEFAULT_LOGIC_KB = ROOT / "data" / "Logic_Based_Educational_Queries.json"

OPENROUTER_API_KEY_ENV = "OR_TOKEN"
LLM_PROVIDER_ENV = "EXACT_LLM_PROVIDER"


def build_llm() -> Any:
    """Use vLLM for submission, while keeping OpenRouter available for demos."""
    provider = os.getenv(LLM_PROVIDER_ENV, "vllm").strip().lower()
    if provider in {"openrouter", "or"}:
        return OpenRouterClient(api_key_env=OPENROUTER_API_KEY_ENV)
    return VLLMClient()


llm = build_llm()
graph = ExactGraph(
    llm=llm,
    physics_kb_path=str(DEFAULT_PHYSICS_KB),
    logic_kb_path=str(DEFAULT_LOGIC_KB),
)

app = FastAPI(title="EXACT 2026 Multi-Agent QA", version="2.0-langgraph")


class QueryPayload(BaseModel):
    """Accept the unified EXACT 2026 competition payload."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    type: Literal["type1", "type2"]
    query: str
    premises: list[str]
    options: list[str]


def get_graph() -> ExactGraph:
    """Return the initialized LangGraph service used by prediction endpoints."""
    return graph


@app.get("/health")
def health() -> dict[str, Any]:
    """Report whether the API and configured LLM are available."""
    return {
        "status": "ok",
        "llm_provider": llm.provider,
        "llm_enabled": llm.enabled,
    }


@app.get("/info")
def info() -> dict[str, Any]:
    """Describe the workflow capabilities and configured LLM endpoint."""
    return {
        "version": "2.0-langgraph",
        "system": "EXACT 2026 Multi-Agent QA",
        "llm": {
            "provider": llm.provider,
            "enabled": llm.enabled,
            "model": llm.model,
            "base_url": llm.base_url,
        },
        "features": [
            "Logic XAI SymbCoT classification, planning, execution, and answer extraction",
            "Logic premise-index explainability without RAG injection",
            "Physics formula generation and SymPy computation",
            "LangGraph workflow routing",
            "LangSmith step tracing",
        ],
    }


@app.post("/predict")
def predict(
    payload: QueryPayload,
    workflow: ExactGraph = Depends(get_graph),
) -> list[dict[str, Any]]:
    """Route one competition query and return the required one-item result list."""
    data = payload.model_dump(exclude_none=True)
    question = data["query"].strip()
    if not question:
        raise HTTPException(status_code=422, detail="A non-empty query is required.")
    data["query"] = question
    try:
        return [workflow.predict(data)]
    except WorkflowExecutionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000)
