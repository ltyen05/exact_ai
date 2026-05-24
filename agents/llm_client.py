import os
import requests
from typing import Any, Dict, List, Optional


class VLLMClient:
    """Tiny OpenAI-compatible client for a self-hosted vLLM server.

    Environment variables:
      EXACT_LLM_BASE_URL=http://localhost:8000/v1
      EXACT_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
      EXACT_LLM_API_KEY=EMPTY

    The code intentionally does not call closed-source APIs. If no endpoint is
    configured, the rest of the pipeline falls back to symbolic/heuristic logic.
    """

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None, timeout: int = 45):
        self.base_url = (base_url or os.getenv("EXACT_LLM_BASE_URL") or "").rstrip("/")
        self.model = model or os.getenv("EXACT_LLM_MODEL") or ""
        self.api_key = api_key or os.getenv("EXACT_LLM_API_KEY") or "EMPTY"
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 1024, response_format: Optional[Dict[str, Any]] = None) -> str:
        if not self.enabled:
            raise RuntimeError("No local vLLM/OpenAI-compatible endpoint configured.")
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        r = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
