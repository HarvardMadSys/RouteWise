"""Metrics collection for hybrid inference system."""

import time
from dataclasses import dataclass


@dataclass
class RequestMetrics:
    """Metrics for a single request."""

    request_id: str
    provider: str
    model: str
    timestamp: float

    # Latency metrics (in ms)
    request_latency_ms: float | None = None
    time_to_first_token_ms: float | None = None

    # Token counts
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Status
    success: bool = True
    error: str | None = None

    # Cost (if available)
    cost_usd: float | None = None


@dataclass
class ProviderMetrics:
    """Aggregated metrics for a provider."""

    provider_id: str

    # Counters
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Latency stats (in ms)
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # TTFT stats (in ms)
    avg_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p90_ttft_ms: float = 0.0

    # Throughput
    requests_per_second: float = 0.0

    # Quality metrics
    error_rate: float = 0.0
    availability: float = 1.0

    # Cost metrics
    total_cost_usd: float = 0.0
    avg_cost_per_request: float = 0.0
    tokens_per_dollar: float = 0.0


class MetricsCollector:
    """Collects and aggregates metrics across providers."""

    def __init__(self):
        self.request_metrics: list[RequestMetrics] = []
        self.provider_metrics: dict[str, ProviderMetrics] = {}
        self._start_time = time.time()

    def record_request_start(
        self, request_id: str, provider: str, model: str, prompt_tokens: int
    ) -> RequestMetrics:
        """Record the start of a request."""
        metrics = RequestMetrics(
            request_id=request_id,
            provider=provider,
            model=model,
            timestamp=time.time(),
            prompt_tokens=prompt_tokens,
        )
        self.request_metrics.append(metrics)
        return metrics

    def record_request_end(
        self,
        request_id: str,
        completion_tokens: int,
        request_latency_ms: float,
        ttft_ms: float | None = None,
        success: bool = True,
        error: str | None = None,
        cost_usd: float | None = None,
    ):
        """Record the completion of a request."""
        for metrics in self.request_metrics:
            if metrics.request_id == request_id:
                metrics.completion_tokens = completion_tokens
                metrics.request_latency_ms = request_latency_ms
                metrics.time_to_first_token_ms = ttft_ms
                metrics.success = success
                metrics.error = error
                metrics.cost_usd = cost_usd

                # Update provider aggregates
                self._update_provider_metrics(metrics)
                break

    def _update_provider_metrics(self, request: RequestMetrics):
        """Update aggregated provider metrics."""
        provider_id = request.provider

        if provider_id not in self.provider_metrics:
            self.provider_metrics[provider_id] = ProviderMetrics(provider_id=provider_id)

        provider = self.provider_metrics[provider_id]
        provider.total_requests += 1

        if request.success:
            provider.successful_requests += 1
        else:
            provider.failed_requests += 1

        # Update cost tracking
        if request.cost_usd:
            provider.total_cost_usd += request.cost_usd

        # Recalculate aggregates
        self._recalculate_provider_stats(provider_id)

    def _recalculate_provider_stats(self, provider_id: str):
        """Recalculate statistics for a provider."""
        provider = self.provider_metrics[provider_id]

        # Get all requests for this provider
        provider_requests = [
            r
            for r in self.request_metrics
            if r.provider == provider_id and r.success and r.request_latency_ms
        ]

        if not provider_requests:
            return

        # Calculate error rate
        provider.error_rate = provider.failed_requests / provider.total_requests

        # Calculate latency percentiles
        latencies = sorted([r.request_latency_ms for r in provider_requests])
        ttfts = sorted(
            [r.time_to_first_token_ms for r in provider_requests if r.time_to_first_token_ms]
        )

        if latencies:
            provider.avg_latency_ms = sum(latencies) / len(latencies)
            provider.p50_latency_ms = self._percentile(latencies, 0.5)
            provider.p90_latency_ms = self._percentile(latencies, 0.9)
            provider.p99_latency_ms = self._percentile(latencies, 0.99)

        if ttfts:
            provider.avg_ttft_ms = sum(ttfts) / len(ttfts)
            provider.p50_ttft_ms = self._percentile(ttfts, 0.5)
            provider.p90_ttft_ms = self._percentile(ttfts, 0.9)

        # Calculate throughput
        duration_seconds = time.time() - self._start_time
        if duration_seconds > 0:
            provider.requests_per_second = provider.successful_requests / duration_seconds

        # Calculate cost metrics
        if provider.total_cost_usd > 0:
            provider.avg_cost_per_request = provider.total_cost_usd / provider.total_requests
            total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in provider_requests)
            if total_tokens > 0:
                provider.tokens_per_dollar = total_tokens / provider.total_cost_usd

    def _percentile(self, data: list[float], p: float) -> float:
        """Calculate percentile of a sorted list."""
        if not data:
            return 0.0
        k = (len(data) - 1) * p
        f = int(k)
        c = f + 1 if f < len(data) - 1 else f
        return data[f] if f == c else data[f] * (c - k) + data[c] * (k - f)

    def get_provider_metrics(self, provider_id: str) -> ProviderMetrics | None:
        """Get metrics for a specific provider."""
        return self.provider_metrics.get(provider_id)

    def get_all_provider_metrics(self) -> dict[str, ProviderMetrics]:
        """Get metrics for all providers."""
        return self.provider_metrics.copy()

    def get_summary(self) -> dict:
        """Get overall system summary."""
        total_requests = len(self.request_metrics)
        successful_requests = sum(1 for r in self.request_metrics if r.success)
        total_cost = sum(r.cost_usd for r in self.request_metrics if r.cost_usd)

        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": total_requests - successful_requests,
            "total_cost_usd": total_cost,
            "providers": len(self.provider_metrics),
            "uptime_seconds": time.time() - self._start_time,
        }
