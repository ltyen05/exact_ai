"""
LLM client implementation.
Exclusively uses OpenRouter Client.
"""

from __future__ import annotations

from agents.llm.openrouter_client import OpenRouterClient

# Direct alias for OpenRouterClient
LLMClient = OpenRouterClient

__all__ = [
    "OpenRouterClient",
    "LLMClient",
]


