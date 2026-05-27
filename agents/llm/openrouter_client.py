"""
OpenRouter LLM client implementation.

OpenRouter provides unified access to multiple LLM providers:
- Claude (Anthropic)
- GPT-4 (OpenAI)
- Llama (Meta)
- And many others

Environment variables:
  OPENROUTER_API_KEY: Your OpenRouter API key (get from https://openrouter.ai)
  OPENROUTER_MODEL: Model name (default: claude-3-sonnet-20240229)
  OPENROUTER_BASE_URL: API endpoint (default: https://openrouter.ai/api/v1)
"""

from __future__ import annotations

import os
import requests
from typing import Any, Dict, List, Optional

from agents.llm.base import LLMClientBase


class OpenRouterClient(LLMClientBase):
    """
    OpenRouter LLM client - unified access to multiple models.
    
    Supported models (examples):
    - claude-3-sonnet-20240229 (Anthropic)
    - claude-3-opus-20240229
    - claude-3-haiku-20240307
    - gpt-4-turbo
    - gpt-4o
    - llama-2-70b
    - mistral-large
    """

    # Default model (good balance of speed/quality)
    DEFAULT_MODEL = "claude-3-sonnet-20240229"
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
    ):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key (from https://openrouter.ai)
            model: Model name (default: claude-3-sonnet)
            base_url: API endpoint (default: OpenRouter's endpoint)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or ""
        self.model = model or os.getenv("OPENROUTER_MODEL") or self.DEFAULT_MODEL
        self.base_url = (base_url or os.getenv("OPENROUTER_BASE_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        """Check if OpenRouter is configured."""
        return bool(self.api_key and self.model)

    def health_check(self) -> bool:
        """Test connectivity to OpenRouter."""
        if not self.enabled:
            return False
        try:
            url = f"{self.base_url}/models"
            headers = self._get_headers()
            r = requests.get(url, headers=headers, timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers for OpenRouter."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/exact2026",  # Required by OpenRouter
            "X-Title": "EXACT 2026",
        }

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Send chat request to OpenRouter.
        
        Args:
            messages: List of messages with role and content
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Max tokens in response
            response_format: Optional format constraint (e.g., {"type": "json_object"})
        
        Returns:
            Response text from model
        
        Raises:
            RuntimeError: If OpenRouter not configured or request fails
        """
        if not self.enabled:
            raise RuntimeError(
                "OpenRouter not configured. "
                "Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
            )

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # OpenRouter supports JSON mode for certain models
        if response_format and response_format.get("type") == "json_object":
            if "claude" in self.model.lower() or "gpt" in self.model.lower():
                payload["response_format"] = {"type": "json_object"}

        headers = self._get_headers()
        url = f"{self.base_url}/chat/completions"

        try:
            r = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            r.raise_for_status()

            data = r.json()
            
            # Handle different response formats
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0].get("message", {}).get("content", "")
                return content
            else:
                raise RuntimeError(f"Unexpected OpenRouter response format: {data}")

        except requests.exceptions.Timeout:
            raise RuntimeError(f"OpenRouter request timeout (>{self.timeout}s)")
        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get("error", {}).get("message", error_msg)
                except Exception:
                    pass
            raise RuntimeError(f"OpenRouter API error: {error_msg}")
        except Exception as e:
            raise RuntimeError(f"OpenRouter request failed: {str(e)}")

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models on OpenRouter.
        
        Returns:
            List of model information dicts
        """
        if not self.api_key:
            raise RuntimeError("OpenRouter API key not configured")

        try:
            url = f"{self.base_url}/models"
            headers = self._get_headers()
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            return data.get("data", [])
        except Exception as e:
            raise RuntimeError(f"Failed to list models: {str(e)}")
