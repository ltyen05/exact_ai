from __future__ import annotations

import os

from .llm_provider import LLMClientBase


class HFClient(LLMClientBase):
    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    DEFAULT_BASE_URL = "https://router.huggingface.co/v1"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "HF_TOKEN",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout_s: float | None = None,
    ) -> None:
        resolved_model = model or os.getenv("HF_MODEL") or self.DEFAULT_MODEL
        resolved_base_url = base_url or os.getenv("HF_BASE_URL") or self.DEFAULT_BASE_URL

        super().__init__(
            provider="hf",
            model=resolved_model,
            base_url=resolved_base_url,
            api_key_env=api_key_env,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout_s=timeout_s,
            require_api_key=True,
        )
