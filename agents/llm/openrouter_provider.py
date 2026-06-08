from __future__ import annotations

import os
from typing import Any

from .llm_provider import LLMClientBase


class OpenRouterClient(LLMClientBase):
    DEFAULT_MODEL = "qwen/qwen-2.5-7b-instruct"
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "OR_TOKEN",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        http_referer: str | None = None,
        app_title: str | None = None,
    ) -> None:
        default_headers: dict[str, str] = {}
        if http_referer:
            default_headers["HTTP-Referer"] = http_referer
        if app_title:
            default_headers["X-Title"] = app_title

        resolved_model = model or os.getenv("OPENROUTER_MODEL") or self.DEFAULT_MODEL
        resolved_base_url = base_url or os.getenv("OPENROUTER_BASE_URL") or self.DEFAULT_BASE_URL

        super().__init__(
            provider="openrouter",
            model=self._resolve_model_route(resolved_model),
            base_url=resolved_base_url,
            api_key_env=api_key_env,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            default_headers=default_headers,
        )
        self.http_referer = http_referer
        self.app_title = app_title

    @classmethod
    def _resolve_model_route(cls, model: str) -> str:
        if ":" in model:
            return model
        provider = cls.MODEL_ROUTE_PROVIDERS.get(model)
        if provider:
            return f"{model}:{provider}"
        return model

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
        stage: str = "llm.chat",
    ) -> str:
        del reasoning
        return super().chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            reasoning={"effort": "none"},
            stage=stage,
        )
