from __future__ import annotations

import os

from .llm_provider import LLMClientBase


class VLLMClient(LLMClientBase):
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "EXACT_LLM_API_KEY",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout_s: float | None = None,
    ) -> None:
        resolved_model = model or os.getenv("EXACT_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        resolved_base_url = base_url or os.getenv("EXACT_LLM_BASE_URL", "http://localhost:8000/v1")
        api_key = os.getenv(api_key_env)
        if api_key:
            api_key = api_key.strip()
        if api_key and api_key.upper() == "EMPTY":
            api_key = None

        super().__init__(
            provider="vllm",
            model=resolved_model,
            base_url=resolved_base_url,
            api_key_env=api_key_env,
            api_key=api_key,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout_s=timeout_s,
            require_api_key=False,
        )
