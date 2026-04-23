from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMRequest:
    prompt: str
    model: str
    max_tokens: int | None = 4096
    temperature: float = 0.7


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a completion for the given request"""
        pass

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models from this provider.

        Returns:
            List of model metadata dictionaries.
            Default implementation returns empty list.
        """
        return []
