"""Base client components."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from serving.base import LLMRequest


class DataLoader(ABC):
    """Abstract base class for loading datasets."""

    @abstractmethod
    async def load_requests(self) -> AsyncGenerator[LLMRequest, None]:
        """Load requests from data source."""
        pass


class RequestGenerator(ABC):
    """Abstract base class for generating requests."""

    @abstractmethod
    async def generate_requests(self) -> AsyncGenerator[LLMRequest, None]:
        """Generate requests for benchmarking."""
        pass
