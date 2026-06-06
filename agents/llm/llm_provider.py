"""
Abstract base class for LLM clients.
Allows easy switching between different LLM platforms.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC
from typing import Any, Dict, List, Optional

import requests

from agents.workflows.tracing import trace_llm_attempt


class LLMClientBase(ABC):
    """Base class for all LLM implementations (OpenAI-compatible HTTP API)."""

    MODEL_ROUTE_PROVIDERS: Dict[str, str] = {}
    DEFAULT_TIMEOUT_S = 60.0

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        base_url: str,
        api_key_env: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        default_headers: Optional[Dict[str, str]] = None,
        timeout_s: Optional[float] = None,
        require_api_key: bool = True,
        api_key: Optional[str] = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.default_headers = dict(default_headers or {})
        self.timeout_s = timeout_s or self.DEFAULT_TIMEOUT_S
        self.require_api_key = require_api_key
        self._explicit_api_key = api_key
        self._session = requests.Session()

    def _resolve_api_key(self) -> Optional[str]:
        if self._explicit_api_key is not None:
            return self._explicit_api_key
        if not self.api_key_env:
            return None
        value = os.getenv(self.api_key_env)
        if value is None:
            return None
        value = value.strip()
        return value or None

    @property
    def api_key(self) -> Optional[str]:
        return self._resolve_api_key()

    @property
    def enabled(self) -> bool:
        """Check if this LLM client is properly configured."""
        if not self.base_url or not self.model:
            return False
        if self.require_api_key and not self.api_key:
            return False
        return True

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        headers.update(self.default_headers)
        api_key = self.api_key
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _endpoint(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            raise RuntimeError(f"{self.provider} LLM is not configured.")
        url = self._endpoint(path)
        try:
            response = self._session.post(
                url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout_s,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"{self.provider} LLM request failed: {exc}") from exc
        if response.status_code >= 400:
            detail: Any
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise RuntimeError(f"{self.provider} LLM request failed ({response.status_code}): {detail}")
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError(f"{self.provider} LLM returned non-JSON response") from exc

    @trace_llm_attempt
    def _request_attempt(
        self,
        path: str,
        payload: Dict[str, Any],
        stage: str,
        attempt: int,
    ) -> Dict[str, Any]:
        """Execute one traced HTTP attempt and retain data only for the caller."""
        started = time.perf_counter()
        try:
            data = self._post(path, payload)
        except RuntimeError as exc:
            return {
                "status": "error",
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "response_chars": 0,
                "_error": str(exc),
            }
        return {
            "status": "ok",
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "response_chars": len(json.dumps(data, ensure_ascii=False, default=str)),
            "_data": data,
        }

    def _extract_content(self, payload: Dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not choices or not isinstance(choices, list):
            raise RuntimeError("LLM response missing choices.")
        first = choices[0] if choices else None
        if not isinstance(first, dict):
            raise RuntimeError("LLM response choices format is invalid.")
        message = first.get("message") or {}
        content: Any = message.get("content")
        if content is None:
            content = first.get("text")
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
            content = "".join(parts) if parts else None
        if content is None:
            raise RuntimeError("LLM response missing message content.")
        return str(content)

    def _retry_without_response_format(self, error: RuntimeError) -> bool:
        """Allow one prompt-only retry when OpenRouter cannot serve JSON mode."""
        if self.provider != "openrouter":
            return False
        message = str(error).lower()
        if any(term in message for term in ("invalid model", "model id", "authentication", "unauthorized", "api key")):
            return False
        return any(
            term in message
            for term in (
                "provider returned error",
                "require_parameters",
                "response_format",
                "no endpoints found",
                "unsupported parameter",
            )
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning: Optional[Dict[str, Any]] = None,
        stage: str = "llm.chat",
    ) -> str:
        """
        Send a chat request to the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
            response_format: Optional format constraint (e.g., {"type": "json_object"})

        Returns:
            The LLM response text

        Raises:
            RuntimeError: If LLM is not properly configured
        """
        if not self.enabled:
            raise RuntimeError(f"{self.provider} LLM is not configured.")
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_new_tokens,
            "top_p": self.top_p,
            "stream": False,
        }
        if response_format:
            payload["response_format"] = response_format
            if self.provider == "openrouter":
                payload["provider"] = {"require_parameters": True}
        if reasoning:
            payload["reasoning"] = reasoning
        first = self._request_attempt("/chat/completions", payload, stage, 1)
        if first["status"] == "ok":
            data = first["_data"]
        else:
            error = RuntimeError(str(first.get("_error") or f"{self.provider} LLM request failed."))
            if not response_format or not self._retry_without_response_format(error):
                raise error
            fallback_payload = dict(payload)
            fallback_payload.pop("response_format", None)
            fallback_payload.pop("provider", None)
            second = self._request_attempt("/chat/completions", fallback_payload, stage, 2)
            if second["status"] != "ok":
                raise RuntimeError(str(second.get("_error") or f"{self.provider} LLM request failed."))
            data = second["_data"]
        return self._extract_content(data)

    def health_check(self) -> bool:
        """Test connectivity to the LLM service."""
        if not self.enabled:
            return False
        try:
            response = self._session.get(
                self._endpoint("/models"),
                headers=self._headers(),
                timeout=self.timeout_s,
            )
            if response.ok:
                return True
            if response.status_code == 404:
                response = self._session.get(
                    self._endpoint("/health"),
                    headers=self._headers(),
                    timeout=self.timeout_s,
                )
                return response.ok
        except requests.RequestException:
            return False
        return False
