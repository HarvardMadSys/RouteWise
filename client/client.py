"""Main client class."""

from serving.base import LLMRequest, LLMResponse

from .base import DataLoader, RequestGenerator
from .metrics import MetricsCollector


class HybridClient:
    """Client for the hybrid inference system."""

    def __init__(
        self,
        data_loader: DataLoader | None = None,
        request_generator: RequestGenerator | None = None,
        metrics_collector: MetricsCollector | None = None,
    ):
        self.data_loader = data_loader
        self.request_generator = request_generator
        self.metrics_collector = metrics_collector or MetricsCollector()

    async def send_request(self, request: LLMRequest) -> LLMResponse:
        """Send a single request to the system."""
        # This will be implemented to call the serving layer
        # For now, just record the request
        self.metrics_collector.record_request(request)
        raise NotImplementedError("Will connect to serving layer")

    async def run_benchmark(self):
        """Run benchmark using configured data loader or request generator."""
        if self.data_loader:
            async for request in self.data_loader.load_requests():
                await self.send_request(request)
        elif self.request_generator:
            async for request in self.request_generator.generate_requests():
                await self.send_request(request)
        else:
            raise ValueError("No data loader or request generator configured")
