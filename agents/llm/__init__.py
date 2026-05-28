"""LLM clients package."""

from agents.llm.llm_provider import LLMClientBase
from agents.llm.openrouter_provider import OpenRouterClient
from agents.llm.vllm_provider import VLLMClient

# Default client alias
LLMClient = OpenRouterClient

__all__ = ["LLMClientBase", "OpenRouterClient", "VLLMClient", "LLMClient"]
