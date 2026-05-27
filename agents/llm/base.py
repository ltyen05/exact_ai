"""
Abstract base class for LLM clients.
Allows easy switching between different LLM platforms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMClientBase(ABC):
    """Abstract base class for all LLM implementations."""

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Check if this LLM client is properly configured."""
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: Optional[Dict[str, Any]] = None,
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
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Test connectivity to the LLM service."""
        pass
