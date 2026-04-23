#!/usr/bin/env python3
r"""Phase 5 Online Evaluation: Head-to-Head Policy Comparison on Real OpenRouter API.

This script performs real-time evaluation of routing policies against OpenRouter API
to prove our routing strategies beat OpenRouter's default routing.

Key features:
- N-way interleaved trace replay (each request -> one policy, round-robin)
- Real streaming TTFT measurement
- True concurrent hedging with delayed backup dispatch
- Cost tracking from OpenRouter usage response (actual billing)
- Budget guardrails with cost estimation bounds
- Safety controls (cost cap, error threshold, validity checks)

Policies:
- openrouter_auto: OpenRouter default routing (baseline)
- lp_mix: Phase 3 LP-based optimal mixing
- smart_hedge: Phase 4 with true concurrent hedging (delayed backup)

Two replay modes:
- P0 (Arrival Pattern Only): Fixed short prompt, only replay arrival times (low cost)
- P1 (Full Trace): Real prompts + variable lengths (high cost, high realism)

Usage:
    # Smoke test (30-50 requests, ~$1)
    python phase5_online_evaluation.py \\
        --trace experiment/data/burstgpt_sample.csv \\
        --max-requests 50 --cost-cap 1.0

    # Full run (300-500 requests, ~$5)
    python phase5_online_evaluation.py \\
        --trace experiment/data/burstgpt_trace.csv \\
        --max-requests 500 --cost-cap 5.0 --warmup 900
"""

import argparse
import csv
import importlib.util
import json
import os
import pickle
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import Phase 3/4 modules
_script_dir = os.path.dirname(os.path.abspath(__file__))
_module_path = os.path.join(_script_dir, "..", "strategies", "online_latency_router.py")
_module_path = os.path.normpath(_module_path)

_spec = importlib.util.spec_from_file_location("online_latency_router", _module_path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

FailureMode = _module.FailureMode
OnlineLatencyRouter = _module.OnlineLatencyRouter
ProviderProfile = _module.ProviderProfile
load_pricing = _module.load_pricing

# Import Phase 4 hedging module
_hedging_path = os.path.join(_script_dir, "..", "strategies", "smart_hedging.py")
_hedging_path = os.path.normpath(_hedging_path)

_hedging_spec = importlib.util.spec_from_file_location("smart_hedging", _hedging_path)
_hedging_module = importlib.util.module_from_spec(_hedging_spec)
sys.modules[_hedging_spec.name] = _hedging_module
_hedging_spec.loader.exec_module(_hedging_module)

HedgingStrategy = _hedging_module.HedgingStrategy
BackupSelectionMethod = _hedging_module.BackupSelectionMethod
HedgingParams = _hedging_module.HedgingParams
SmartHedger = _hedging_module.SmartHedger
select_backup = _hedging_module.select_backup

# Import V2 router
_v2_path = os.path.join(_script_dir, "..", "strategies", "v2_router.py")
_v2_path = os.path.normpath(_v2_path)
_v2_spec = importlib.util.spec_from_file_location("v2_router", _v2_path)
_v2_module = importlib.util.module_from_spec(_v2_spec)
sys.modules[_v2_spec.name] = _v2_module
_v2_spec.loader.exec_module(_v2_module)
V2Router = _v2_module.V2Router

# Import data loader for trace replay
# Add project root to sys.path for experiment.data imports
_project_root = os.path.normpath(os.path.join(_script_dir, "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from experiment.data.loader import DataLoader

# OpenRouter API configuration
API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
BASE_URL = "https://openrouter.ai/api/v1"

# Default settings
DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct"
PROBE_PROMPT = "Say hello in exactly 5 words."
DEFAULT_MAX_TOKENS = 16

DEFAULT_TIMEOUT_SEC = 60
DEFAULT_USER_AGENT = "hybridInference/phase5_online_evaluation"

# Sentinel prefix for OpenRouter sort-based routing policies. The suffix is
# the `sort` key passed to the OpenRouter provider field (price/throughput/
# latency). Using a prefix lets us keep _select_provider_for_policy's signature
# as str | None while disambiguating from real provider names.
SORT_PREFIX = "__or_sort__:"
SORT_POLICIES = {
    "sort_price": f"{SORT_PREFIX}price",
    "sort_throughput": f"{SORT_PREFIX}throughput",
    "sort_latency": f"{SORT_PREFIX}latency",
}


@dataclass
class BudgetConfig:
    """Budget configuration for trace replay evaluation."""

    max_requests: int = 500
    cost_cap_usd: float = 5.0
    max_tokens: int = 16
    # Baselines: OpenRouter sort modes cover the "latency" baseline via
    # sort_latency (OpenRouter picks the fastest provider dynamically).
    policies: tuple[str, ...] = (
        "openrouter_auto",  # OpenRouter default (baseline)
        "sort_price",  # OpenRouter sort=price (cheapest baseline)
        "sort_latency",  # OpenRouter sort=latency (fastest baseline)
        "lp_mix",  # LP optimal mixing (our method V1)
        "smart_hedge",  # LP + hedging (our method V1)
        "v2_p50_hedge",  # P50-rank + hedging (our method V2)
        "cheapest_fixed",  # Always cheapest provider (cost baseline)
    )
    extra_fixed_providers: tuple[str, ...] = ()  # Additional single-provider baselines


@dataclass
class TraceRequest:
    """A single request from the trace for replay."""

    arrival_time_sec: float  # Relative to trace start
    prompt: str = PROBE_PROMPT
    max_tokens: int = DEFAULT_MAX_TOKENS
    original_prompt_tokens: int = 0
    original_completion_tokens: int = 0
    use_real_prompt: bool = False  # P1 mode: use real prompt from trace


@dataclass
class Checkpoint:
    """Checkpoint for resuming trace replay."""

    completed_requests: int  # Number of requests fully processed
    total_cost: float  # Cumulative cost
    router_samples: dict[str, list[tuple[float, float]]]  # provider -> [(timestamp, ttft_ms)]
    router_errors: dict[str, list[tuple[float, str | None]]]  # provider -> [(timestamp, error)]
    policy_costs: dict[str, float]  # policy -> total_cost
    policy_errors: dict[str, int]  # policy -> error_count
    policy_results: dict[str, list[dict]]  # policy -> list of result dicts for analysis
    last_save_time: float  # Unix timestamp of checkpoint

    def save(self, path: Path) -> None:
        """Save checkpoint to file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  [Checkpoint] Saved at request {self.completed_requests}")

    @classmethod
    def load(cls, path: Path) -> "Checkpoint | None":
        """Load checkpoint from file, returns None if not found."""
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                checkpoint = pickle.load(f)
            print(f"  [Checkpoint] Loaded from {path}")
            print(f"    Completed requests: {checkpoint.completed_requests}")
            print(f"    Total cost: ${checkpoint.total_cost:.4f}")
            return checkpoint
        except Exception as e:
            print(f"  [Checkpoint] Failed to load: {e}")
            return None


def load_trace_from_csv(
    filepath: str,
    max_requests: int | None = None,
    time_limit_sec: float | None = None,
) -> list[TraceRequest]:
    """Load trace from CSV file using existing DataLoader.

    Args:
        filepath: Path to trace CSV file (BurstGPT or HybridInference format).
        max_requests: Maximum number of requests to load.
        time_limit_sec: Only load requests within this time window.

    Returns:
        List of TraceRequest objects with relative arrival times.
    """
    loader = DataLoader(config={})
    requests = loader.load(filepath)

    if not requests:
        return []

    # Convert to TraceRequest with relative arrival times
    first_timestamp = requests[0].timestamp
    trace = []

    for req in requests:
        arrival_time_sec = float(req.timestamp - first_timestamp)

        # Apply time limit
        if time_limit_sec is not None and arrival_time_sec > time_limit_sec:
            break

        trace_req = TraceRequest(
            arrival_time_sec=arrival_time_sec,
            prompt=PROBE_PROMPT,  # P0 mode: fixed short prompt
            max_tokens=DEFAULT_MAX_TOKENS,
            original_prompt_tokens=req.request_tokens,
            original_completion_tokens=req.response_tokens,
        )
        trace.append(trace_req)

        # Apply max_requests limit
        if max_requests is not None and len(trace) >= max_requests:
            break

    return trace


def load_trace_from_jsonl(
    filepath: str,
    max_requests: int | None = None,
    time_limit_sec: float | None = None,
    use_real_prompts: bool = True,
) -> list[TraceRequest]:
    """Load trace from JSONL file with real prompts (P1 mode).

    Args:
        filepath: Path to JSONL file (sharegpt_prompts_burstgpt_timestamps.jsonl).
        max_requests: Maximum number of requests to load.
        time_limit_sec: Only load requests within this time window.
        use_real_prompts: If True, use real prompt_text from trace.

    Returns:
        List of TraceRequest objects with real prompts.
    """
    trace = []
    first_timestamp = None

    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            timestamp = record.get("arrived_at", 0)
            if first_timestamp is None:
                first_timestamp = timestamp

            arrival_time_sec = float(timestamp - first_timestamp)

            # Apply time limit
            if time_limit_sec is not None and arrival_time_sec > time_limit_sec:
                break

            prompt_text = record.get("prompt_text", PROBE_PROMPT)
            num_decode_tokens = record.get("num_decode_tokens", DEFAULT_MAX_TOKENS)
            num_prefill_tokens = record.get("num_prefill_tokens", 0)

            # Cap max_tokens to avoid extremely long responses
            # Most providers have limits, and we want reasonable costs
            max_tokens = min(num_decode_tokens, 512)

            trace_req = TraceRequest(
                arrival_time_sec=arrival_time_sec,
                prompt=prompt_text if use_real_prompts else PROBE_PROMPT,
                max_tokens=max_tokens if use_real_prompts else DEFAULT_MAX_TOKENS,
                original_prompt_tokens=num_prefill_tokens,
                original_completion_tokens=num_decode_tokens,
                use_real_prompt=use_real_prompts,
            )
            trace.append(trace_req)

            # Apply max_requests limit
            if max_requests is not None and len(trace) >= max_requests:
                break

    return trace


def estimate_cost_bounds(
    n_requests: int,
    n_policies: int,
    pricing: dict[str, float],
    hedge_rate: float = 0.3,
) -> tuple[float, float]:
    """Estimate cost bounds for trace replay.

    Args:
        n_requests: Total number of trace requests.
        n_policies: Number of policies (each request goes to one policy).
        pricing: Provider -> cost per request.
        hedge_rate: Expected hedge trigger rate for smart_hedge.

    Returns:
        (lower_bound, upper_bound) in USD.
    """
    if not pricing:
        return 0.0, 0.0

    cheapest = min(pricing.values())
    most_expensive = max(pricing.values())

    # Lower bound: all requests -> cheapest provider, no hedging
    lower = n_requests * cheapest

    # Upper bound: all requests -> most expensive, smart_hedge always hedges
    # smart_hedge hedge cost = 2x (primary + backup)
    hedge_multiplier = 1 + hedge_rate  # Average across policies
    upper = n_requests * most_expensive * hedge_multiplier

    return lower, upper


def estimate_trace_duration(
    trace: list[TraceRequest],
    speedup: float = 1.0,
) -> float:
    """Estimate wall-clock duration for replaying a trace.

    Args:
        trace: List of trace requests.
        speedup: Time compression factor (1.0 = real-time).

    Returns:
        Estimated duration in seconds.
    """
    if not trace:
        return 0.0
    return (trace[-1].arrival_time_sec - trace[0].arrival_time_sec) / speedup


@dataclass
class SingleRequestResult:
    """Result of a single API request (internal)."""

    ttft_ms: float
    e2e_ms: float
    status: str
    actual_provider: str
    error_message: str | None
    prompt_tokens: int
    completion_tokens: int
    cost: float  # Actual cost from OpenRouter response
    start_ts: float
    first_token_ts: float | None


@dataclass
class RequestResult:
    """Result of a single API request."""

    request_id: str
    timestamp: float
    policy: str
    selected_provider: str | None
    actual_provider: str
    ttft_ms: float
    e2e_ms: float
    status: str
    slo_violated: bool
    cost_usd: float
    hedge_triggered: bool = False
    hedge_provider: str | None = None
    hedge_winner: str | None = None  # "primary" or "backup"
    error_message: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_is_estimated: bool = False  # True if cost is estimated (no token usage)


@dataclass
class EvaluationConfig:
    """Configuration for online evaluation."""

    model: str = DEFAULT_MODEL
    slo_sec: float = 2.0
    warmup_sec: float = 1800  # 30 minutes
    duration_sec: float = 7200  # 2 hours
    request_interval_sec: float = 2.0  # Time between policy cycles
    cost_cap_usd: float = 10.0  # Safety: abort if total cost exceeds
    error_threshold: float = 0.1  # Safety: abort if error rate > 10%
    error_window: int = 50  # Window for error rate calculation
    max_tokens: int = DEFAULT_MAX_TOKENS
    lp_update_interval: float = 300.0  # 5 minutes
    window_sec: float = 15 * 60  # 15 minutes - single moving window
    dispatch_overhead_sec: float = 0.05
    policy_backoff_sec: float = 60.0  # Backoff when error rate is high.
    probing_interval_sec: float = 300.0  # 5 minutes - periodic probing of all providers
    extra_fixed_providers: tuple[str, ...] = ()  # Additional single-provider baselines


@dataclass
class PolicyState:
    """Track state for each policy."""

    name: str
    results: list[RequestResult] = field(default_factory=list)
    total_cost: float = 0.0
    error_count: int = 0

    @property
    def success_count(self) -> int:
        """Return number of successful requests."""
        return sum(1 for r in self.results if r.status == "success")

    @property
    def slo_violations(self) -> int:
        """Return number of SLO violations."""
        return sum(1 for r in self.results if r.slo_violated)

    @property
    def slo_violation_rate(self) -> float:
        """Return SLO violation rate."""
        if not self.results:
            return 0.0
        return self.slo_violations / len(self.results)

    @property
    def avg_cost(self) -> float:
        """Return average cost per request."""
        if not self.results:
            return 0.0
        return self.total_cost / len(self.results)

    def get_latency_percentile(self, p: float) -> float:
        """Get latency percentile for successful requests."""
        latencies = [r.ttft_ms for r in self.results if r.status == "success"]
        if not latencies:
            return float("inf")
        return float(np.percentile(latencies, p))


class Phase5OnlineEvaluator:
    """Online evaluator for head-to-head policy comparison."""

    def __init__(
        self,
        config: EvaluationConfig,
        pricing: dict[str, float],
        output_dir: Path,
    ):
        """Initialize the evaluator.

        Args:
            config: Evaluation configuration.
            pricing: Provider -> cost per request.
            output_dir: Directory for output files.
        """
        self.config = config
        self.pricing = pricing
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._thread_local = threading.local()

        # Initialize router for LP-mix policy
        self.router = OnlineLatencyRouter(
            costs=pricing,
            slo_sec=config.slo_sec,
            window_sec=config.window_sec,
            failure_mode=FailureMode.INFINITY,
            kappa=0.0,
            weight_smoothing=0.3,
            lp_update_interval=config.lp_update_interval,
        )

        # Initialize hedger for smart_hedge policy
        # Use SMART_ECONOMIC strategy: cost-benefit model
        # Hedge when P(violation|waited t) * P(backup succeeds) > C_b/V
        # cost_ratio=0.05 chosen from Phase 4 ablation (best Pareto point)
        hedging_params = HedgingParams(
            strategy=HedgingStrategy.SMART_ECONOMIC,
            slo_sec=config.slo_sec,
            cost_ratio=0.05,
            dispatch_overhead_sec=config.dispatch_overhead_sec,
            backup_method=BackupSelectionMethod.FASTEST,
        )
        self.hedger = SmartHedger(hedging_params, pricing)

        # V2 router: P50-rank + Pareto filter (drop-in alongside V1 router)
        self.v2_router = V2Router(
            costs=pricing,
            slo_sec=config.slo_sec,
            window_sec=config.window_sec,
            failure_mode=FailureMode.INFINITY,
            lp_update_interval=config.lp_update_interval,
        )

        # Policy states
        self.policies = {
            "openrouter_auto": PolicyState(name="openrouter_auto"),
            "sort_price": PolicyState(name="sort_price"),
            "sort_throughput": PolicyState(name="sort_throughput"),
            "sort_latency": PolicyState(name="sort_latency"),
            "lp_mix": PolicyState(name="lp_mix"),
            "smart_hedge": PolicyState(name="smart_hedge"),
            "cheapest_fixed": PolicyState(name="cheapest_fixed"),
            "fastest_fixed": PolicyState(name="fastest_fixed"),
            "v2_p50_hedge": PolicyState(name="v2_p50_hedge"),
        }

        # Add extra fixed-provider baselines
        self.extra_fixed_providers: dict[str, str] = {}  # policy_name -> provider_name
        for provider_name in config.extra_fixed_providers:
            policy_name = f"{provider_name.lower()}_fixed"
            self.policies[policy_name] = PolicyState(name=policy_name)
            self.extra_fixed_providers[policy_name] = provider_name

        # Determine fixed providers
        self.eval_providers = sorted(pricing.keys())
        self.cheapest_provider = min(pricing, key=lambda p: pricing[p])
        self.fastest_provider = "Groq" if "Groq" in pricing else self.cheapest_provider
        if "Groq" not in pricing:
            print(
                "Warning: 'Groq' not found in pricing file. "
                f"Using '{self.fastest_provider}' as fastest_fixed baseline."
            )
        # Validate extra fixed providers exist in pricing
        for provider_name in config.extra_fixed_providers:
            if provider_name not in pricing:
                print(
                    f"Warning: '{provider_name}' not found in pricing. "
                    f"{provider_name.lower()}_fixed will be disabled."
                )

        self.avg_cost_estimate = float(np.mean(list(pricing.values()))) if pricing else 0.0
        self.policy_disabled_until: dict[str, float] = {}

        # CSV log file
        self.csv_path = output_dir / "evaluation_log.csv"
        self._csv_lock = threading.Lock()  # Thread-safe CSV writing
        self._init_csv_log()

        # Total cost tracking
        self.total_cost = 0.0
        self.probing_cost = 0.0  # Track probing cost separately

        # Router lock for thread-safe access
        self._router_lock = threading.Lock()

    def _get_session(self) -> requests.Session:
        """Return a per-thread requests session.

        requests.Session is not thread-safe, so we keep one session per thread.
        """
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
            self._thread_local.session = session
        return session

    def _init_csv_log(self) -> None:
        """Initialize CSV log file with headers."""
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "request_id",
                    "policy",
                    "selected_provider",
                    "actual_provider",
                    "ttft_ms",
                    "e2e_ms",
                    "status",
                    "slo_violated",
                    "cost_usd",
                    "cost_is_estimated",
                    "hedge_triggered",
                    "hedge_provider",
                    "hedge_winner",
                    "prompt_tokens",
                    "completion_tokens",
                    "error_message",
                ]
            )

    def _log_result(self, result: RequestResult) -> None:
        """Append result to CSV log (thread-safe)."""
        with self._csv_lock, open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    result.timestamp,
                    result.request_id,
                    result.policy,
                    result.selected_provider,
                    result.actual_provider,
                    result.ttft_ms,
                    result.e2e_ms,
                    result.status,
                    result.slo_violated,
                    result.cost_usd,
                    result.cost_is_estimated,
                    result.hedge_triggered,
                    result.hedge_provider,
                    result.hedge_winner,
                    result.prompt_tokens,
                    result.completion_tokens,
                    result.error_message,
                ]
            )

    def _send_request(
        self,
        provider: str | None = None,
        timeout: int = 60,
        ttft_event: threading.Event | None = None,
        ttft_info: dict[str, Any] | None = None,
        prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> SingleRequestResult:
        """Send a single request to OpenRouter API with streaming.

        Args:
            provider: Provider to use (None for OpenRouter auto).
            timeout: Request timeout in seconds.
            prompt: Custom prompt text (default: PROBE_PROMPT).
            max_tokens: Custom max tokens (default: config.max_tokens).

        Returns:
            SingleRequestResult with TTFT, status, provider, and token usage.
        """
        if prompt is None:
            prompt = PROBE_PROMPT
        if max_tokens is None:
            max_tokens = self.config.max_tokens

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "HTTP-Referer": "https://github.com/HarvardSys/hybridInference",
            "X-Title": "hybridInference phase5 evaluation",
        }

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if provider is not None:
            if provider.startswith(SORT_PREFIX):
                # OpenRouter sort-based routing (no explicit provider order).
                sort_key = provider[len(SORT_PREFIX):]
                payload["provider"] = {"sort": sort_key}
            else:
                payload["provider"] = {
                    "order": [provider],
                    "allow_fallbacks": False,
                }

        # Use perf_counter for precise timing, time.time() for absolute timestamps
        start_perf = time.perf_counter()
        start_time = time.time()  # For absolute timestamp (logging, hedging)
        if provider and not provider.startswith(SORT_PREFIX):
            actual_provider = provider
        else:
            actual_provider = "unknown"
        ttft_ms = -1.0
        e2e_ms = -1.0
        status = "success"
        error_message = None
        prompt_tokens = 0
        completion_tokens = 0
        cost = 0.0  # Actual cost from OpenRouter
        first_token_ts: float | None = None
        first_token_perf: float | None = None  # perf_counter value at first token

        if ttft_info is not None:
            ttft_info.setdefault("ttft_ms", -1.0)
            ttft_info.setdefault("first_token_ts", None)
            ttft_info.setdefault("status", None)

        try:
            session = self._get_session()
            response = session.post(
                url=f"{BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
                stream=True,
            )

            if response.status_code != 200:
                try:
                    error_data = response.json() if response.text else {}
                except Exception:
                    error_data = {}
                error_message = error_data.get("error", {}).get("message", "Unknown error")

                # If stream_options is not supported, retry once without it.
                if (
                    response.status_code in (400, 422)
                    and "stream_options" in json.dumps(error_data)
                    and "stream_options" in payload
                ):
                    payload.pop("stream_options", None)
                    response.close()
                    response = session.post(
                        url=f"{BASE_URL}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=timeout,
                        stream=True,
                    )
                    if response.status_code != 200:
                        response.close()
                        if ttft_event is not None:
                            ttft_event.set()
                        if ttft_info is not None:
                            ttft_info["status"] = f"HTTP {response.status_code}"
                        return SingleRequestResult(
                            ttft_ms=-1.0,
                            e2e_ms=-1.0,
                            status=f"HTTP {response.status_code}",
                            actual_provider=actual_provider,
                            error_message=error_message,
                            prompt_tokens=0,
                            completion_tokens=0,
                            cost=0.0,
                            start_ts=start_time,
                            first_token_ts=None,
                        )
                else:
                    response.close()
                    if ttft_event is not None:
                        ttft_event.set()
                    if ttft_info is not None:
                        ttft_info["status"] = f"HTTP {response.status_code}"
                    return SingleRequestResult(
                        ttft_ms=-1.0,
                        e2e_ms=-1.0,
                        status=f"HTTP {response.status_code}",
                        actual_provider=actual_provider,
                        error_message=error_message,
                        prompt_tokens=0,
                        completion_tokens=0,
                        cost=0.0,
                        start_ts=start_time,
                        first_token_ts=None,
                    )

            # Best-effort: provider may appear in headers (case-insensitive).
            for k, v in response.headers.items():
                if "provider" in k.lower() and v:
                    actual_provider = v
                    break

            # Process streaming response.
            for line in response.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8", errors="ignore")
                if not line_str.startswith("data: "):
                    continue
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Record TTFT on first content token.
                if first_token_ts is None:
                    choices = chunk.get("choices", [])
                    if choices and choices[0].get("delta", {}).get("content"):
                        first_token_perf = time.perf_counter()
                        first_token_ts = time.time()  # For hedging comparison
                        ttft_ms = (
                            first_token_perf - start_perf
                        ) * 1000  # Use perf_counter for precision
                        if ttft_info is not None:
                            ttft_info["ttft_ms"] = ttft_ms
                            ttft_info["first_token_ts"] = first_token_ts
                        if ttft_event is not None:
                            ttft_event.set()

                # Extract provider if present (best-effort).
                provider_field = chunk.get("provider")
                if isinstance(provider_field, str) and provider_field:
                    actual_provider = provider_field
                elif isinstance(provider_field, dict):
                    name = provider_field.get("name")
                    if isinstance(name, str) and name:
                        actual_provider = name

                # Extract token usage and cost from response (usually in final chunk).
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens) or 0)
                    completion_tokens = int(usage.get("completion_tokens", completion_tokens) or 0)
                    # OpenRouter returns actual billed cost directly
                    cost = float(usage.get("cost", cost) or 0.0)

            end_perf = time.perf_counter()
            e2e_ms = (end_perf - start_perf) * 1000  # Use perf_counter for precision

            # Handle case where no first token was detected.
            if ttft_ms < 0 and e2e_ms > 0:
                ttft_ms = e2e_ms  # Fallback to E2E

            if ttft_event is not None and not ttft_event.is_set():
                ttft_event.set()
            if ttft_info is not None and ttft_info.get("status") is None:
                ttft_info["status"] = status

            response.close()

        except requests.exceptions.Timeout:
            if ttft_event is not None:
                ttft_event.set()
            if ttft_info is not None:
                ttft_info["status"] = "timeout"
            return SingleRequestResult(
                ttft_ms=-1.0,
                e2e_ms=-1.0,
                status="timeout",
                actual_provider=actual_provider,
                error_message="Request timeout",
                prompt_tokens=0,
                completion_tokens=0,
                cost=0.0,
                start_ts=start_time,
                first_token_ts=None,
            )
        except Exception as e:
            if ttft_event is not None:
                ttft_event.set()
            if ttft_info is not None:
                ttft_info["status"] = "error"
            return SingleRequestResult(
                ttft_ms=-1.0,
                e2e_ms=-1.0,
                status="error",
                actual_provider=actual_provider,
                error_message=str(e),
                prompt_tokens=0,
                completion_tokens=0,
                cost=0.0,
                start_ts=start_time,
                first_token_ts=None,
            )

        # If we explicitly requested a provider and did not reliably extract
        # the routed provider name, fall back to the requested provider for
        # consistent accounting. Skip sort-based sentinels since they do not
        # name a single provider.
        if (
            provider is not None
            and not provider.startswith(SORT_PREFIX)
            and actual_provider not in self.pricing
        ):
            actual_provider = provider

        return SingleRequestResult(
            ttft_ms=ttft_ms,
            e2e_ms=e2e_ms,
            status=status,
            actual_provider=actual_provider,
            error_message=error_message,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            start_ts=start_time,
            first_token_ts=first_token_ts,
        )

    def _send_hedged_request(
        self,
        primary_provider: str,
        backup_provider: str,
        hedge_time_sec: float,
        prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> tuple[SingleRequestResult, SingleRequestResult | None, str, bool]:
        """Send hedged request with delayed backup dispatch.

        Implements true concurrent hedging:
        1. Start primary request immediately
        2. If primary doesn't return TTFT by hedge_time, start backup
        3. Use whichever returns first with valid TTFT
        4. Both requests are billed

        Args:
            primary_provider: Primary provider to try first.
            backup_provider: Backup provider for hedging.
            hedge_time_sec: Time threshold to trigger backup (seconds).
            prompt: Custom prompt text.
            max_tokens: Custom max tokens.

        Returns:
            (primary_result, backup_result, winner, hedge_triggered)
            where winner is "primary" or "backup".
        """
        if not np.isfinite(hedge_time_sec):
            # Never hedge.
            primary_result = self._send_request(
                provider=primary_provider,
                timeout=DEFAULT_TIMEOUT_SEC,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            return primary_result, None, "primary", False

        hedge_time_sec = max(0.0, float(hedge_time_sec))

        primary_result: SingleRequestResult | None = None
        backup_result: SingleRequestResult | None = None
        primary_ttft = threading.Event()
        backup_ttft = threading.Event()
        primary_ttft_info: dict[str, Any] = {}
        backup_ttft_info: dict[str, Any] = {}
        hedge_triggered = False

        def run_primary():
            nonlocal primary_result
            primary_result = self._send_request(
                provider=primary_provider,
                timeout=DEFAULT_TIMEOUT_SEC,
                ttft_event=primary_ttft,
                ttft_info=primary_ttft_info,
                prompt=prompt,
                max_tokens=max_tokens,
            )

        def run_backup():
            nonlocal backup_result
            backup_result = self._send_request(
                provider=backup_provider,
                timeout=DEFAULT_TIMEOUT_SEC,
                ttft_event=backup_ttft,
                ttft_info=backup_ttft_info,
                prompt=prompt,
                max_tokens=max_tokens,
            )

        # Start primary request
        primary_thread = threading.Thread(target=run_primary)
        primary_thread.start()

        # Wait for primary TTFT (or terminal error) up to hedge_time.
        primary_ttft.wait(timeout=hedge_time_sec)

        if not primary_ttft.is_set() or primary_ttft_info.get("ttft_ms", -1.0) <= 0:
            # Primary has not produced a TTFT within hedge_time (or errored), start backup.
            hedge_triggered = True
            overhead = max(0.0, float(self.config.dispatch_overhead_sec))
            if overhead > 0:
                time.sleep(overhead)
            backup_thread = threading.Thread(target=run_backup)
            backup_thread.start()

            primary_thread.join()
            backup_thread.join()
        else:
            # Primary produced TTFT before hedge_time. No hedging.
            primary_thread.join()

        # Determine winner
        if not hedge_triggered:
            return primary_result, None, "primary", False

        def first_token_or_inf(r: SingleRequestResult | None) -> float:
            if r is None or r.status != "success" or r.first_token_ts is None:
                return float("inf")
            return float(r.first_token_ts)

        primary_first = first_token_or_inf(primary_result)
        backup_first = first_token_or_inf(backup_result)
        winner = "primary" if primary_first <= backup_first else "backup"

        return primary_result, backup_result, winner, True

    def _select_provider_for_policy(self, policy: str, now: float) -> str | None:
        """Select provider based on policy.

        Args:
            policy: Policy name.
            now: Current timestamp.

        Returns:
            Provider name (None for openrouter_auto).
        """
        if policy == "openrouter_auto":
            return None
        elif policy in SORT_POLICIES:
            return SORT_POLICIES[policy]
        elif policy == "lp_mix":
            return self.router.route(now)
        elif policy == "smart_hedge":
            # Primary from LP-mix
            return self.router.route(now)
        elif policy == "v2_p50_hedge":
            # V2: P50-rank primary selection (with hedging in execute)
            return self.v2_router.route(now)
        elif policy == "cheapest_fixed":
            return self.cheapest_provider
        elif policy == "fastest_fixed":
            return self.fastest_provider
        elif policy in self.extra_fixed_providers:
            return self.extra_fixed_providers[policy]
        else:
            raise ValueError(f"Unknown policy: {policy}")

    def _execute_policy_request(
        self,
        policy: str,
        now: float,
        prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> RequestResult:
        """Execute a single request for a policy.

        Args:
            policy: Policy name.
            now: Current timestamp.
            prompt: Custom prompt text (P1 mode).
            max_tokens: Custom max tokens (P1 mode).

        Returns:
            RequestResult with metrics.
        """
        request_id = str(uuid.uuid4())[:8]
        selected_provider = self._select_provider_for_policy(policy, now)
        hedge_triggered = False
        hedge_provider = None
        hedge_winner = None
        prompt_tokens = 0
        completion_tokens = 0
        cost_is_estimated = False

        # Execute request based on policy
        if policy == "v2_p50_hedge" and selected_provider is not None:
            # V2: same hedging logic as smart_hedge, but uses v2_router profiles
            backup = select_backup(
                method=BackupSelectionMethod.FASTEST,
                profiles=self.v2_router.profiles,
                costs=self.pricing,
                primary=selected_provider,
                slo_sec=self.config.slo_sec,
                now=now,
                lp_weights=self.v2_router.last_lp_weights,
            )
            hedge_provider = backup

            if backup is not None and backup != selected_provider:
                h = self.hedger.compute_hedge_time(
                    selected_provider,
                    backup,
                    self.v2_router.profiles,
                    now,
                )

                primary_result, backup_result, winner, hedge_triggered = self._send_hedged_request(
                    primary_provider=selected_provider,
                    backup_provider=backup,
                    hedge_time_sec=h,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )

                hedge_winner = winner

                if winner == "primary" or backup_result is None:
                    result = primary_result
                    actual_provider = result.actual_provider if result else "unknown"
                else:
                    result = backup_result
                    actual_provider = result.actual_provider if result else "unknown"

                overall_start = primary_result.start_ts if primary_result else now
                if result and result.first_token_ts is not None and result.status == "success":
                    ttft_ms = (result.first_token_ts - overall_start) * 1000
                else:
                    ttft_ms = result.ttft_ms if result else -1

                e2e_ms = result.e2e_ms if result else -1
                status = result.status if result else "error"
                error_message = result.error_message if result else "hedging_failed"
                cost = (primary_result.cost if primary_result else 0) + (
                    backup_result.cost if backup_result and hedge_triggered else 0
                )
                prompt_tokens = result.prompt_tokens if result else 0
                completion_tokens = result.completion_tokens if result else 0
            else:
                # No backup or same provider -- send without hedging
                result = self._send_request(
                    provider=selected_provider,
                    timeout=DEFAULT_TIMEOUT_SEC,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
                actual_provider = result.actual_provider if result else "unknown"
                ttft_ms = result.ttft_ms if result else -1
                e2e_ms = result.e2e_ms if result else -1
                status = result.status if result else "error"
                error_message = result.error_message if result else "request_failed"
                cost = result.cost if result else 0
                prompt_tokens = result.prompt_tokens if result else 0
                completion_tokens = result.completion_tokens if result else 0

        elif policy == "smart_hedge" and selected_provider is not None:
            # True concurrent hedging implementation
            backup = select_backup(
                method=BackupSelectionMethod.FASTEST,
                profiles=self.router.profiles,
                costs=self.pricing,
                primary=selected_provider,
                slo_sec=self.config.slo_sec,
                now=now,
                lp_weights=self.router.last_lp_weights,
            )
            hedge_provider = backup

            if backup is not None and backup != selected_provider:
                # Compute hedge time
                h = self.hedger.compute_hedge_time(
                    selected_provider,
                    backup,
                    self.router.profiles,
                    now,
                )

                # Execute hedged request with true concurrency
                primary_result, backup_result, winner, hedge_triggered = self._send_hedged_request(
                    primary_provider=selected_provider,
                    backup_provider=backup,
                    hedge_time_sec=h,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )

                hedge_winner = winner

                # Determine final result based on winner
                if winner == "primary" or backup_result is None:
                    result = primary_result
                    actual_provider = result.actual_provider if result else "unknown"
                else:
                    result = backup_result
                    actual_provider = result.actual_provider if result else "unknown"

                overall_start = primary_result.start_ts if primary_result else now
                if result and result.first_token_ts is not None and result.status == "success":
                    ttft_ms = (result.first_token_ts - overall_start) * 1000
                else:
                    ttft_ms = result.ttft_ms if result else -1.0

                if result and result.e2e_ms > 0:
                    end_ts = result.start_ts + (result.e2e_ms / 1000.0)
                    e2e_ms = (end_ts - overall_start) * 1000
                else:
                    e2e_ms = result.e2e_ms if result else -1.0
                status = result.status
                error_msg = result.error_message

                # Log total token usage (both requests if hedged).
                prompt_tokens = 0
                completion_tokens = 0
                if primary_result is not None:
                    prompt_tokens += primary_result.prompt_tokens
                    completion_tokens += primary_result.completion_tokens
                if hedge_triggered and backup_result is not None:
                    prompt_tokens += backup_result.prompt_tokens
                    completion_tokens += backup_result.completion_tokens

                # Use actual cost from OpenRouter (both requests billed if hedged)
                cost = 0.0
                cost_is_estimated = False
                if primary_result:
                    if primary_result.cost > 0:
                        cost += primary_result.cost
                    else:
                        cost_is_estimated = True
                if hedge_triggered and backup_result:
                    if backup_result.cost > 0:
                        cost += backup_result.cost
                    else:
                        cost_is_estimated = True
            else:
                # No valid backup, just send primary
                result = self._send_request(
                    provider=selected_provider,
                    timeout=DEFAULT_TIMEOUT_SEC,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
                ttft_ms = result.ttft_ms
                e2e_ms = result.e2e_ms
                status = result.status
                actual_provider = result.actual_provider
                error_msg = result.error_message
                prompt_tokens = result.prompt_tokens
                completion_tokens = result.completion_tokens
                cost = result.cost
                cost_is_estimated = cost <= 0
        else:
            # Non-hedging policies: single request
            result = self._send_request(
                provider=selected_provider,
                timeout=DEFAULT_TIMEOUT_SEC,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            ttft_ms = result.ttft_ms
            e2e_ms = result.e2e_ms
            status = result.status
            actual_provider = result.actual_provider
            error_msg = result.error_message
            prompt_tokens = result.prompt_tokens
            completion_tokens = result.completion_tokens

            # Use actual cost from OpenRouter
            cost = result.cost
            cost_is_estimated = cost <= 0

        # Check SLO violation
        slo_violated = (status != "success") or (ttft_ms / 1000.0 > self.config.slo_sec)

        return RequestResult(
            request_id=request_id,
            timestamp=now,
            policy=policy,
            selected_provider=selected_provider,
            actual_provider=actual_provider,
            ttft_ms=ttft_ms,
            e2e_ms=e2e_ms,
            status=status,
            slo_violated=slo_violated,
            cost_usd=cost,
            hedge_triggered=hedge_triggered,
            hedge_provider=hedge_provider,
            hedge_winner=hedge_winner,
            error_message=error_msg,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_is_estimated=cost_is_estimated,
        )

    def _execute_policy_request_with_provider(
        self,
        policy: str,
        selected_provider: str | None,
        hedge_info: tuple[str | None, float] | None,
        now: float,
        prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> RequestResult:
        """Execute a policy request with pre-computed provider selection.

        This is the thread-safe version that accepts pre-computed provider
        and hedge info, avoiding concurrent access to the router.

        Args:
            policy: Policy name.
            selected_provider: Pre-computed provider selection.
            hedge_info: Pre-computed (backup_provider, hedge_time) for smart_hedge.
            now: Current timestamp.
            prompt: Custom prompt text (P1 mode).
            max_tokens: Custom max tokens (P1 mode).

        Returns:
            RequestResult with metrics.
        """
        request_id = str(uuid.uuid4())[:8]
        hedge_triggered = False
        hedge_provider = None
        hedge_winner = None
        prompt_tokens = 0
        completion_tokens = 0
        cost_is_estimated = False

        # Execute request based on policy
        if policy in ("smart_hedge", "v2_p50_hedge") and selected_provider is not None and hedge_info is not None:
            backup, hedge_time = hedge_info
            hedge_provider = backup

            if backup is not None and backup != selected_provider:
                # Execute hedged request with true concurrency
                primary_result, backup_result, winner, hedge_triggered = self._send_hedged_request(
                    primary_provider=selected_provider,
                    backup_provider=backup,
                    hedge_time_sec=hedge_time,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )

                hedge_winner = winner

                # Determine final result based on winner
                if winner == "primary" or backup_result is None:
                    result = primary_result
                    actual_provider = result.actual_provider if result else "unknown"
                else:
                    result = backup_result
                    actual_provider = result.actual_provider if result else "unknown"

                overall_start = primary_result.start_ts if primary_result else now
                if result and result.first_token_ts is not None and result.status == "success":
                    ttft_ms = (result.first_token_ts - overall_start) * 1000
                else:
                    ttft_ms = result.ttft_ms if result else -1.0

                if result and result.e2e_ms > 0:
                    end_ts = result.start_ts + (result.e2e_ms / 1000.0)
                    e2e_ms = (end_ts - overall_start) * 1000
                else:
                    e2e_ms = result.e2e_ms if result else -1.0
                status = result.status
                error_msg = result.error_message

                # Log total token usage (both requests if hedged).
                prompt_tokens = 0
                completion_tokens = 0
                if primary_result is not None:
                    prompt_tokens += primary_result.prompt_tokens
                    completion_tokens += primary_result.completion_tokens
                if hedge_triggered and backup_result is not None:
                    prompt_tokens += backup_result.prompt_tokens
                    completion_tokens += backup_result.completion_tokens

                # Use actual cost from OpenRouter (both requests billed if hedged)
                cost = 0.0
                cost_is_estimated = False
                if primary_result:
                    if primary_result.cost > 0:
                        cost += primary_result.cost
                    else:
                        cost_is_estimated = True
                if hedge_triggered and backup_result:
                    if backup_result.cost > 0:
                        cost += backup_result.cost
                    else:
                        cost_is_estimated = True
            else:
                # No valid backup, just send primary
                result = self._send_request(
                    provider=selected_provider,
                    timeout=DEFAULT_TIMEOUT_SEC,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
                ttft_ms = result.ttft_ms
                e2e_ms = result.e2e_ms
                status = result.status
                actual_provider = result.actual_provider
                error_msg = result.error_message
                prompt_tokens = result.prompt_tokens
                completion_tokens = result.completion_tokens
                cost = result.cost
                cost_is_estimated = cost <= 0
        else:
            # Non-hedging policies: single request
            result = self._send_request(
                provider=selected_provider,
                timeout=DEFAULT_TIMEOUT_SEC,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            ttft_ms = result.ttft_ms
            e2e_ms = result.e2e_ms
            status = result.status
            actual_provider = result.actual_provider
            error_msg = result.error_message
            prompt_tokens = result.prompt_tokens
            completion_tokens = result.completion_tokens

            # Use actual cost from OpenRouter
            cost = result.cost
            cost_is_estimated = cost <= 0

        # Check SLO violation
        slo_violated = (status != "success") or (ttft_ms / 1000.0 > self.config.slo_sec)

        return RequestResult(
            request_id=request_id,
            timestamp=now,
            policy=policy,
            selected_provider=selected_provider,
            actual_provider=actual_provider,
            ttft_ms=ttft_ms,
            e2e_ms=e2e_ms,
            status=status,
            slo_violated=slo_violated,
            cost_usd=cost,
            hedge_triggered=hedge_triggered,
            hedge_provider=hedge_provider,
            hedge_winner=hedge_winner,
            error_message=error_msg,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_is_estimated=cost_is_estimated,
        )

    def _update_router_from_result(self, result: RequestResult) -> None:
        """Update router profiles from a request result (thread-safe).

        Args:
            result: Request result to learn from.
        """
        if result.selected_provider is None:
            return  # Can't safely learn from OpenRouter auto without provider attribution.
        if result.actual_provider not in self.pricing:
            return

        error_type = None if result.status == "success" else "server_error"
        ttft_ms = result.ttft_ms if result.status == "success" else -1.0

        with self._router_lock:
            self.router.add_sample(
                provider=result.actual_provider,
                timestamp=result.timestamp,
                ttft_ms=ttft_ms,
                error_type=error_type,
            )
            # V2 router learns from the same samples.
            self.v2_router.add_sample(
                provider=result.actual_provider,
                timestamp=result.timestamp,
                ttft_ms=ttft_ms,
                error_type=error_type,
            )

    def _run_probing_round(self, log_to_csv: bool = True) -> dict[str, float]:
        """Run a probing round to refresh all provider profiles.

        Sends one probe request to each provider to keep profiles fresh.
        This prevents provider profiles from going stale during long evaluations.

        Args:
            log_to_csv: Whether to log probing results to CSV (default: True).

        Returns:
            Dict of provider -> latency_ms (-1 if failed).
        """
        results = {}
        for provider in self.eval_providers:
            # Check cost cap
            if self.total_cost > self.config.cost_cap_usd:
                break

            now = time.time()
            result = self._send_request(
                provider=provider,
                timeout=DEFAULT_TIMEOUT_SEC,
            )

            # Update router (thread-safe)
            error_type = None if result.status == "success" else "server_error"
            ttft_ms = result.ttft_ms if result.status == "success" else -1.0
            with self._router_lock:
                self.router.add_sample(
                    provider=result.actual_provider,
                    timestamp=now,
                    ttft_ms=ttft_ms,
                    error_type=error_type,
                )
                # V2 router also learns from probing.
                self.v2_router.add_sample(
                    provider=result.actual_provider,
                    timestamp=now,
                    ttft_ms=ttft_ms,
                    error_type=error_type,
                )

            # Track cost (both total and probing-specific)
            probe_cost = result.cost if result.cost > 0 else self.avg_cost_estimate
            self.total_cost += probe_cost
            self.probing_cost += probe_cost

            # Log probing result to CSV for cost accountability
            if log_to_csv:
                probe_result = RequestResult(
                    request_id=f"probe_{provider[:8]}",
                    timestamp=now,
                    policy="_probing",  # Special policy tag for probing
                    selected_provider=provider,
                    actual_provider=result.actual_provider,
                    ttft_ms=ttft_ms,
                    e2e_ms=result.e2e_ms,
                    status=result.status,
                    slo_violated=False,  # Probing doesn't count for SLO
                    cost_usd=probe_cost,
                    cost_is_estimated=(result.cost <= 0),
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    error_message=result.error_message,
                )
                self._log_result(probe_result)

            results[provider] = ttft_ms

            # Small delay between probes to avoid rate limiting
            time.sleep(0.5)

        return results

    def warmup(self, duration_sec: float | None = None) -> None:
        """Run warmup phase to seed router profiles.

        Sends requests to all providers to build initial latency profiles.

        Args:
            duration_sec: Warmup duration. If None, uses config.warmup_sec.
        """
        if duration_sec is None:
            duration_sec = self.config.warmup_sec

        print(f"\n{'='*60}")
        print(f"Phase 5: Warmup Phase ({duration_sec/60:.1f} minutes)")
        print(f"{'='*60}")

        start_time = time.time()
        probe_count = 0

        while time.time() - start_time < duration_sec:
            # Round-robin through all providers in pricing file.
            for provider in self.eval_providers:
                if time.time() - start_time >= duration_sec:
                    break

                probe_count += 1
                now = time.time()

                result = self._send_request(
                    provider=provider,
                    timeout=DEFAULT_TIMEOUT_SEC,
                )

                # Update router
                error_type = None if result.status == "success" else "server_error"
                self.router.add_sample(
                    provider=result.actual_provider,
                    timestamp=now,
                    ttft_ms=result.ttft_ms if result.status == "success" else -1.0,
                    error_type=error_type,
                )

                # Track warmup cost using actual cost from OpenRouter.
                warmup_cost = result.cost
                # For warmup, use estimated cost if no cost returned (safety cap check).
                if warmup_cost <= 0:
                    warmup_cost = self.pricing.get(provider, self.avg_cost_estimate)
                self.total_cost += warmup_cost

                if self.total_cost > self.config.cost_cap_usd:
                    print(
                        f"\nCost cap exceeded during warmup (${self.total_cost:.2f}). Aborting warmup."
                    )
                    self.router.update_lp(force=True)
                    self.v2_router.update_lp(force=True)
                    return

                status_str = (
                    f"TTFT={result.ttft_ms:.0f}ms" if result.status == "success" else result.status
                )
                elapsed = time.time() - start_time
                print(
                    f"  [Warmup {probe_count}] {provider}: {status_str} "
                    f"(tokens: {result.prompt_tokens}+{result.completion_tokens}) "
                    f"({elapsed/60:.1f}/{duration_sec/60:.1f} min)"
                )

                time.sleep(1.0)  # Rate limiting

        # Force LP update after warmup
        self.router.update_lp(force=True)
        self.v2_router.update_lp(force=True)

        print(f"\nWarmup complete: {probe_count} probes")
        print(f"Router LP status: {self.router.last_lp_status}")
        print(f"Router weights: {self.router.last_lp_weights}")

    def run_evaluation(
        self,
        duration_sec: float | None = None,
        policies: list[str] | None = None,
    ) -> dict[str, PolicyState]:
        """Run main evaluation phase with interleaved requests.

        Args:
            duration_sec: Evaluation duration. If None, uses config.duration_sec.
            policies: List of policies to evaluate. If None, evaluates all.

        Returns:
            Dict of policy name -> PolicyState with results.
        """
        if duration_sec is None:
            duration_sec = self.config.duration_sec

        if policies is None:
            policies = list(self.policies.keys())

        print(f"\n{'='*60}")
        print(f"Phase 5: Evaluation Phase ({duration_sec/60:.1f} minutes)")
        print(f"Policies: {policies}")
        print(f"{'='*60}")

        start_time = time.time()
        cycle_count = 0

        while time.time() - start_time < duration_sec:
            cycle_count += 1
            cycle_start = time.time()

            # Safety check: total cost
            if self.total_cost > self.config.cost_cap_usd:
                print(f"\nCost cap exceeded (${self.total_cost:.2f}). Aborting.")
                break

            # Interleaved requests: one per policy per cycle
            # Rotate policy order each cycle to reduce ordering bias.
            rotate = cycle_count % len(policies) if policies else 0
            cycle_policies = policies[rotate:] + policies[:rotate]

            for policy in cycle_policies:
                now = time.time()

                # Check duration
                if now - start_time >= duration_sec:
                    break

                # Skip policy if currently in backoff.
                disabled_until = self.policy_disabled_until.get(policy, 0.0)
                if now < disabled_until:
                    continue

                # Execute request
                result = self._execute_policy_request(policy, now)

                # Record result
                self.policies[policy].results.append(result)
                self.policies[policy].total_cost += result.cost_usd
                if result.status != "success":
                    self.policies[policy].error_count += 1

                self.total_cost += result.cost_usd

                # Update router from all results (not just lp_mix)
                self._update_router_from_result(result)

                # Log result
                self._log_result(result)

                # Progress output
                status_str = (
                    f"TTFT={result.ttft_ms:.0f}ms" if result.status == "success" else result.status
                )
                slo_str = " [SLO VIOLATED]" if result.slo_violated else ""
                elapsed = now - start_time
                print(
                    f"[{elapsed/60:.1f}/{duration_sec/60:.1f} min] "
                    f"{policy}: {result.actual_provider} {status_str}{slo_str}"
                )

                # Safety check: error rate
                recent_errors = sum(
                    1
                    for r in self.policies[policy].results[-self.config.error_window :]
                    if r.status != "success"
                )
                if (
                    len(self.policies[policy].results) >= self.config.error_window
                    and recent_errors / self.config.error_window > self.config.error_threshold
                ):
                    until = now + self.config.policy_backoff_sec
                    self.policy_disabled_until[policy] = until
                    print(
                        f"\nHigh error rate for {policy}. Backing off for "
                        f"{self.config.policy_backoff_sec:.0f}s."
                    )
                    continue

                # Rate limiting between requests
                time.sleep(0.5)

            # Wait between cycles
            cycle_elapsed = time.time() - cycle_start
            if cycle_elapsed < self.config.request_interval_sec * len(policies):
                time.sleep(self.config.request_interval_sec * len(policies) - cycle_elapsed)

        print(f"\nEvaluation complete: {cycle_count} cycles")
        print(f"Total cost: ${self.total_cost:.4f}")

        return self.policies

    def analyze_results(self) -> dict[str, dict]:
        """Analyze evaluation results and compute summary statistics.

        Returns:
            Dict of policy name -> stats dict.
        """
        stats = {}

        for policy_name, state in self.policies.items():
            if not state.results:
                continue

            # Count requests with real vs estimated cost
            real_cost_count = sum(1 for r in state.results if not r.cost_is_estimated)
            estimated_cost_count = sum(1 for r in state.results if r.cost_is_estimated)

            # Compute avg cost only from requests with real token-based cost
            real_cost_results = [r for r in state.results if not r.cost_is_estimated]
            real_cost_total = sum(r.cost_usd for r in real_cost_results)
            avg_cost_real = real_cost_total / len(real_cost_results) if real_cost_results else 0.0

            stats[policy_name] = {
                "total_requests": len(state.results),
                "successful_requests": state.success_count,
                "error_rate": (state.error_count / len(state.results) if state.results else 0.0),
                "slo_violation_rate": state.slo_violation_rate,
                "total_cost": state.total_cost,
                "avg_cost": state.avg_cost,
                "avg_cost_real": avg_cost_real,  # Only from real token-based billing
                "real_cost_count": real_cost_count,
                "estimated_cost_count": estimated_cost_count,
                "p50_ms": state.get_latency_percentile(50),
                "p90_ms": state.get_latency_percentile(90),
                "p99_ms": state.get_latency_percentile(99),
                "provider_distribution": self._compute_provider_distribution(state),
            }

        return stats

    def _compute_provider_distribution(self, state: PolicyState) -> dict[str, float]:
        """Compute provider selection distribution for a policy."""
        if not state.results:
            return {}

        counts: dict[str, int] = {}
        for r in state.results:
            provider = r.actual_provider
            counts[provider] = counts.get(provider, 0) + 1

        total = sum(counts.values())
        return {p: c / total for p, c in counts.items()}

    def print_summary(self, stats: dict[str, dict]) -> None:
        """Print summary of evaluation results."""
        print(f"\n{'='*60}")
        print("Phase 5: Evaluation Summary")
        print(f"{'='*60}")

        # Cost breakdown
        print("\nCost Breakdown:")
        print(f"  Total cost: ${self.total_cost:.4f}")
        print(
            f"  Probing cost: ${self.probing_cost:.4f} ({self.probing_cost/self.total_cost*100:.1f}%)"
            if self.total_cost > 0
            else f"  Probing cost: ${self.probing_cost:.4f}"
        )
        print(f"  Experiment cost: ${self.total_cost - self.probing_cost:.4f}")

        for policy_name, s in stats.items():
            print(f"\n{policy_name}:")
            print(f"  Requests: {s['total_requests']}")
            print(f"  Error rate: {s['error_rate']:.2%}")
            print(f"  SLO violation rate: {s['slo_violation_rate']:.2%}")
            print(f"  Avg cost (all): ${s['avg_cost']:.6f}")
            print(
                f"  Avg cost (real billing only): ${s['avg_cost_real']:.6f} ({s['real_cost_count']}/{s['total_requests']} requests)"
            )
            print(
                f"  P50/P90/P99 TTFT: {s['p50_ms']:.0f} / {s['p90_ms']:.0f} / {s['p99_ms']:.0f} ms"
            )
            print(f"  Provider distribution: {s['provider_distribution']}")

        # Comparative analysis
        if "openrouter_auto" in stats and "lp_mix" in stats:
            baseline = stats["openrouter_auto"]
            lp_mix = stats["lp_mix"]

            print(f"\n{'='*60}")
            print("Comparison: lp_mix vs openrouter_auto")
            print(f"{'='*60}")

            if baseline["avg_cost"] > 0:
                cost_reduction = (baseline["avg_cost"] - lp_mix["avg_cost"]) / baseline["avg_cost"]
                print(f"  Cost reduction: {cost_reduction:.1%}")

            if baseline["slo_violation_rate"] > 0:
                slo_reduction = (
                    baseline["slo_violation_rate"] - lp_mix["slo_violation_rate"]
                ) / baseline["slo_violation_rate"]
                print(f"  SLO violation reduction: {slo_reduction:.1%}")

            p99_improvement = baseline["p99_ms"] - lp_mix["p99_ms"]
            print(f"  P99 latency improvement: {p99_improvement:.0f}ms")

    def save_summary(self, stats: dict[str, dict]) -> None:
        """Save summary statistics to JSON and CSV."""
        # JSON summary
        json_path = self.output_dir / "evaluation_summary.json"
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved JSON summary to {json_path}")

        # CSV summary
        csv_summary_path = self.output_dir / "evaluation_summary.csv"
        with open(csv_summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "policy",
                    "total_requests",
                    "successful_requests",
                    "error_rate",
                    "slo_violation_rate",
                    "total_cost",
                    "avg_cost",
                    "avg_cost_real",
                    "real_cost_count",
                    "estimated_cost_count",
                    "p50_ms",
                    "p90_ms",
                    "p99_ms",
                ]
            )
            for policy, s in stats.items():
                writer.writerow(
                    [
                        policy,
                        s["total_requests"],
                        s["successful_requests"],
                        s["error_rate"],
                        s["slo_violation_rate"],
                        s["total_cost"],
                        s["avg_cost"],
                        s["avg_cost_real"],
                        s["real_cost_count"],
                        s["estimated_cost_count"],
                        s["p50_ms"],
                        s["p90_ms"],
                        s["p99_ms"],
                    ]
                )
        print(f"Saved CSV summary to {csv_summary_path}")

    def check_validity(self, stats: dict[str, dict]) -> tuple[bool, list[str]]:
        """Check if the evaluation run is valid.

        A run is valid only if:
        - All costs are real (no estimated costs)
        - Error rate < 10% for all policies

        Args:
            stats: Analysis results from analyze_results().

        Returns:
            (is_valid, list of failure reasons)
        """
        failures = []

        for policy, s in stats.items():
            # Check for estimated costs
            if s["estimated_cost_count"] > 0:
                failures.append(
                    f"{policy}: {s['estimated_cost_count']} requests with estimated cost"
                )

            # Check error rate
            if s["error_rate"] >= 0.10:
                failures.append(f"{policy}: error rate {s['error_rate']:.1%} >= 10%")

        is_valid = len(failures) == 0
        return is_valid, failures


class TraceReplayEvaluator(Phase5OnlineEvaluator):
    """Evaluator that replays trace with N-way interleaving.

    Key design: Each trace request is assigned to exactly ONE policy via round-robin.
    This keeps cost at ~1x (not Nx) while still allowing fair statistical comparison.
    """

    def __init__(
        self,
        config: EvaluationConfig,
        pricing: dict[str, float],
        output_dir: Path,
        budget_config: BudgetConfig | None = None,
        checkpoint_interval: int = 100,  # Save checkpoint every N requests
    ):
        """Initialize trace replay evaluator.

        Args:
            config: Evaluation configuration.
            pricing: Provider -> cost per request.
            output_dir: Directory for output files.
            budget_config: Budget configuration for cost guardrails.
            checkpoint_interval: Save checkpoint every N requests.
        """
        # Check if checkpoint exists BEFORE calling parent init (which creates CSV)
        self._pending_checkpoint_path = output_dir / "checkpoint.pkl"
        self._has_checkpoint = self._pending_checkpoint_path.exists()

        super().__init__(config, pricing, output_dir)
        self.budget_config = budget_config or BudgetConfig()
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_path = self._pending_checkpoint_path

        # Override policies to only use the configured ones
        self.policies = {name: PolicyState(name=name) for name in self.budget_config.policies}

    def _init_csv_log(self) -> None:
        """Initialize CSV log file with headers.

        If resuming from checkpoint, append to existing file instead of overwriting.
        """
        # Check if we're resuming with existing checkpoint
        if getattr(self, "_has_checkpoint", False) and self.csv_path.exists():
            # Don't overwrite - we'll append to existing file
            print(f"  [CSV] Appending to existing log: {self.csv_path}")
            return

        # Create new CSV with headers
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "request_id",
                    "policy",
                    "selected_provider",
                    "actual_provider",
                    "ttft_ms",
                    "e2e_ms",
                    "status",
                    "slo_violated",
                    "cost_usd",
                    "cost_is_estimated",
                    "hedge_triggered",
                    "hedge_provider",
                    "hedge_winner",
                    "prompt_tokens",
                    "completion_tokens",
                    "error_message",
                ]
            )

    def _create_checkpoint(self, completed: int) -> Checkpoint:
        """Create a checkpoint from current state."""
        router_samples = {}
        router_errors = {}
        for provider, profile in self.router.profiles.items():
            router_samples[provider] = list(profile.samples)
            router_errors[provider] = list(profile.error_samples)

        policy_costs = {p: self.policies[p].total_cost for p in self.policies}
        policy_errors = {p: self.policies[p].error_count for p in self.policies}

        # Serialize results for analysis (include fields for provider_distribution)
        policy_results = {}
        for p, state in self.policies.items():
            policy_results[p] = [
                {
                    "ttft_ms": r.ttft_ms,
                    "e2e_ms": r.e2e_ms,
                    "status": r.status,
                    "slo_violated": r.slo_violated,
                    "cost_usd": r.cost_usd,
                    "cost_is_estimated": r.cost_is_estimated,
                    "actual_provider": r.actual_provider,
                    "hedge_triggered": r.hedge_triggered,
                }
                for r in state.results
            ]

        return Checkpoint(
            completed_requests=completed,
            total_cost=self.total_cost,
            router_samples=router_samples,
            router_errors=router_errors,
            policy_costs=policy_costs,
            policy_errors=policy_errors,
            policy_results=policy_results,
            last_save_time=time.time(),
        )

    def _restore_from_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Restore state from checkpoint."""
        self.total_cost = checkpoint.total_cost

        # Restore router profiles
        for provider, samples in checkpoint.router_samples.items():
            if provider in self.router.profiles:
                self.router.profiles[provider].samples = list(samples)
        for provider, errors in checkpoint.router_errors.items():
            if provider in self.router.profiles:
                self.router.profiles[provider].error_samples = list(errors)

        # Restore policy states
        for policy, cost in checkpoint.policy_costs.items():
            if policy in self.policies:
                self.policies[policy].total_cost = cost
        for policy, errors in checkpoint.policy_errors.items():
            if policy in self.policies:
                self.policies[policy].error_count = errors

        # Restore results for analysis (minimal RequestResult objects)
        if hasattr(checkpoint, "policy_results") and checkpoint.policy_results:
            for policy, results in checkpoint.policy_results.items():
                if policy in self.policies:
                    self.policies[policy].results = [
                        RequestResult(
                            request_id="restored",
                            timestamp=0.0,
                            policy=policy,
                            selected_provider=None,
                            actual_provider=r.get("actual_provider", "unknown"),
                            ttft_ms=r["ttft_ms"],
                            e2e_ms=r.get("e2e_ms", r.get("ttft_ms", 0.0)),
                            status=r["status"],
                            slo_violated=r["slo_violated"],
                            cost_usd=r["cost_usd"],
                            cost_is_estimated=r.get("cost_is_estimated", False),
                            hedge_triggered=r.get("hedge_triggered", False),
                        )
                        for r in results
                    ]

        # Force LP update with restored data
        self.router.update_lp(force=True)
        self.v2_router.update_lp(force=True)
        print(
            f"  [Checkpoint] Restored state, resuming from request {checkpoint.completed_requests}"
        )

    def run_trace_replay(
        self,
        trace: list[TraceRequest],
        policies: list[str] | None = None,
        speedup: float = 1.0,
        resume: bool = True,
    ) -> dict[str, PolicyState]:
        """Replay trace with all policies receiving each request.

        Fair comparison: Each request goes to ALL policies simultaneously.
        - Request 0 -> [openrouter_auto, lp_mix, smart_hedge] in parallel
        - Request 1 -> [openrouter_auto, lp_mix, smart_hedge] in parallel
        ...

        This uses shared router/probing (1x probing overhead) and allows
        direct comparison of the same request across different policies.

        Supports checkpoint/resume: saves progress periodically and can resume
        from the last checkpoint if the process was interrupted.

        Args:
            trace: List of trace requests with arrival times.
            policies: Policies to evaluate. If None, uses budget_config.policies.
            speedup: Time compression factor (1.0 = real-time).
            resume: Whether to resume from checkpoint if available.

        Returns:
            Policy states with results.
        """
        if policies is None:
            policies = list(self.budget_config.policies)

        n_policies = len(policies)
        n_requests = len(trace)

        # Check for existing checkpoint
        start_from = 0
        if resume:
            checkpoint = Checkpoint.load(self.checkpoint_path)
            if checkpoint is not None:
                self._restore_from_checkpoint(checkpoint)
                start_from = checkpoint.completed_requests
                print(f"  [Checkpoint] Resuming from request {start_from}/{n_requests}")

        # Pre-flight checks (cost is now n_policies x higher)
        remaining_requests = n_requests - start_from
        lower_cost, upper_cost = estimate_cost_bounds(
            remaining_requests * n_policies, n_policies, self.pricing
        )
        duration_sec = (
            estimate_trace_duration(trace[start_from:], speedup) if start_from < n_requests else 0
        )

        print(f"\n{'='*60}")
        print("Phase 5: Trace Replay Evaluation (All Policies Per Request)")
        print(f"{'='*60}")
        print(f"Trace: {n_requests} requests over {duration_sec/60:.1f} min (speedup={speedup}x)")
        if start_from > 0:
            print(f"Resuming from request {start_from} ({remaining_requests} remaining)")
        print(f"Policies: {policies}")
        print(f"Each request -> all {n_policies} policies (fair comparison)")
        print(f"Cost estimate: ${lower_cost:.2f} - ${upper_cost:.2f} (remaining)")
        print(f"Cost cap: ${self.budget_config.cost_cap_usd:.2f}")
        print(f"Max tokens: {self.budget_config.max_tokens}")
        print(f"Checkpoint interval: every {self.checkpoint_interval} requests")

        if upper_cost > self.budget_config.cost_cap_usd * 2:
            print(f"\nWARNING: Upper cost estimate (${upper_cost:.2f}) exceeds 2x cost cap!")
            print("Consider reducing --max-requests or increasing --cost-cap.")

        print(f"\n{'='*60}")
        print("Starting replay...")
        print(f"{'='*60}")

        start_time = time.time()
        trace_start = trace[start_from].arrival_time_sec if start_from < n_requests else 0.0
        last_probe_time = start_time  # Track last probing round
        probing_interval = self.config.probing_interval_sec

        completed = start_from
        for i, req in enumerate(trace[start_from:], start=start_from):
            # Wait until scheduled arrival time
            target_time = (req.arrival_time_sec - trace_start) / speedup
            elapsed = time.time() - start_time
            if target_time > elapsed:
                sleep_time = target_time - elapsed
                if sleep_time > 0.01:  # Don't sleep for tiny intervals
                    time.sleep(sleep_time)

            # Check budget
            if self.total_cost > self.budget_config.cost_cap_usd:
                print(f"\nCost cap exceeded (${self.total_cost:.2f}). Stopping at request {i}.")
                break

            # Execute request for ALL policies in parallel
            now = time.time()
            results: dict[str, RequestResult] = {}
            threads: list[threading.Thread] = []

            # Use real prompt from trace if available (P1 mode)
            req_prompt = req.prompt if req.use_real_prompt else None
            req_max_tokens = req.max_tokens if req.use_real_prompt else None

            # Pre-compute provider selections BEFORE spawning threads (thread-safety)
            # This ensures router.route() is called sequentially, avoiding race conditions
            with self._router_lock:
                provider_selections: dict[str, str | None] = {}
                hedge_selections: dict[str, tuple[str | None, float]] = {}  # (backup, hedge_time)
                for policy in policies:
                    provider = self._select_provider_for_policy(policy, now)
                    provider_selections[policy] = provider
                    # Pre-compute hedge info for hedging policies
                    if policy in ("smart_hedge", "v2_p50_hedge") and provider is not None:
                        # Use V2 profiles for v2_p50_hedge, V1 for smart_hedge
                        _profiles = self.v2_router.profiles if policy == "v2_p50_hedge" else self.router.profiles
                        _lp_weights = self.v2_router.last_lp_weights if policy == "v2_p50_hedge" else self.router.last_lp_weights
                        backup = select_backup(
                            method=BackupSelectionMethod.FASTEST,
                            profiles=_profiles,
                            costs=self.pricing,
                            primary=provider,
                            slo_sec=self.config.slo_sec,
                            now=now,
                            lp_weights=_lp_weights,
                        )
                        if backup is not None and backup != provider:
                            h = self.hedger.compute_hedge_time(
                                provider, backup, _profiles, now
                            )
                            hedge_selections[policy] = (backup, h)
                        else:
                            hedge_selections[policy] = (None, float("inf"))

            def execute_policy(
                policy: str,
                provider: str | None,
                hedge_info: tuple[str | None, float] | None,
                prompt: str | None,
                max_tokens: int | None,
                results: dict = results,
                now: float = now,
            ):
                results[policy] = self._execute_policy_request_with_provider(
                    policy=policy,
                    selected_provider=provider,
                    hedge_info=hedge_info,
                    now=now,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )

            for policy in policies:
                provider = provider_selections[policy]
                hedge_info = hedge_selections.get(policy)
                t = threading.Thread(
                    target=execute_policy,
                    args=(policy, provider, hedge_info, req_prompt, req_max_tokens),
                )
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # Record results for all policies
            for policy in policies:
                result = results[policy]
                self.policies[policy].results.append(result)
                self.policies[policy].total_cost += result.cost_usd
                if result.status != "success":
                    self.policies[policy].error_count += 1
                self.total_cost += result.cost_usd
                self._update_router_from_result(result)
                self._log_result(result)

            completed += 1

            # Periodic probing to keep all provider profiles fresh (shared, 1x)
            now = time.time()
            if probing_interval > 0 and (now - last_probe_time) >= probing_interval:
                print("\n  [Probing] Refreshing all provider profiles...")
                probe_results = self._run_probing_round()
                n_success = sum(1 for v in probe_results.values() if v > 0)
                print(f"  [Probing] {n_success}/{len(probe_results)} providers responded")
                last_probe_time = now
                # Force LP update after probing
                self.router.update_lp(force=True)
                self.v2_router.update_lp(force=True)

            # Progress output (every 10 requests)
            if completed % 10 == 0:
                elapsed_min = (time.time() - start_time) / 60
                summary_parts = []
                for policy in policies:
                    r = results[policy]
                    if r.status == "success":
                        summary_parts.append(f"{policy}={r.ttft_ms:.0f}ms")
                    else:
                        summary_parts.append(f"{policy}=ERR")
                print(
                    f"[{completed}/{n_requests}] {', '.join(summary_parts)} "
                    f"(${self.total_cost:.3f}, {elapsed_min:.1f}min)"
                )

            # Save checkpoint periodically
            if self.checkpoint_interval > 0 and completed % self.checkpoint_interval == 0:
                checkpoint = self._create_checkpoint(completed)
                checkpoint.save(self.checkpoint_path)

        # Save final checkpoint
        if self.checkpoint_interval > 0:
            checkpoint = self._create_checkpoint(completed)
            checkpoint.save(self.checkpoint_path)

        # Summary per policy
        counts = {p: len(self.policies[p].results) for p in policies}

        print(f"\n{'='*60}")
        print(
            f"Replay complete: {completed} requests x {n_policies} policies = {sum(counts.values())} total"
        )
        print(f"Total cost: ${self.total_cost:.4f}")
        print(f"Duration: {(time.time() - start_time)/60:.1f} min")
        print(f"{'='*60}")

        return self.policies


def main():
    """Run Phase 5 online evaluation."""
    parser = argparse.ArgumentParser(description="Phase 5: Head-to-Head Online Evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--slo",
        type=float,
        default=2.0,
        help="SLO in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=1800,
        help="Warmup duration in seconds (default: 1800 = 30 min)",
    )
    parser.add_argument(
        "--probing-interval",
        type=float,
        default=300,
        help="Periodic probing interval in seconds (default: 300 = 5 min, 0 to disable)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=7200,
        help="Evaluation duration in seconds (default: 7200 = 2 hours)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiment/results/phase5_online",
        help="Output directory",
    )
    parser.add_argument(
        "--pricing",
        type=str,
        default="experiment/data/openrouter_llama33_70b.json",
        help="Path to pricing JSON file",
    )
    parser.add_argument(
        "--cost-cap",
        type=float,
        default=10.0,
        help="Maximum total cost in USD (default: 10.0)",
    )
    parser.add_argument(
        "--policies",
        type=str,
        nargs="+",
        default=None,
        help="Policies to evaluate (default: openrouter_auto, sort_price, sort_throughput, sort_latency, lp_mix, smart_hedge, cheapest_fixed)",
    )
    parser.add_argument(
        "--extra-fixed",
        type=str,
        nargs="+",
        default=None,
        help="Additional single-provider baselines (e.g., --extra-fixed Nebius Inceptron)",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip warmup phase",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run without API calls",
    )
    # Trace replay arguments
    parser.add_argument(
        "--trace",
        type=str,
        default=None,
        help="Path to trace CSV file for replay mode (BurstGPT or HybridInference format)",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Maximum number of trace requests to replay",
    )
    parser.add_argument(
        "--speedup",
        type=float,
        default=1.0,
        help="Time compression factor for trace replay (1.0 = real-time, 2.0 = 2x faster)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Only replay requests within this time window (seconds)",
    )
    parser.add_argument(
        "--real-prompts",
        action="store_true",
        help="Use real prompts from trace (P1 mode). Requires JSONL trace file with prompt_text field.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=512,
        help="Maximum output tokens per request when using real prompts (default: 512)",
    )
    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save checkpoint every N requests (default: 100, 0 to disable)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignoring any existing checkpoint",
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="Resume from existing run directory (use this instead of --output to continue a run)",
    )

    args = parser.parse_args()

    # Check API key
    if not API_KEY and not args.dry_run:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)

    # Load pricing
    print(f"Loading pricing from {args.pricing}...")
    if not os.path.exists(args.pricing):
        print(f"Error: Pricing file not found at {args.pricing}")
        sys.exit(1)

    pricing = load_pricing(args.pricing)
    print(f"  Loaded pricing for {len(pricing)} providers")

    # Determine policies.
    # Baselines cover the "fast" dimension via sort_latency (OpenRouter picks
    # the fastest provider dynamically), which is a stronger and more realistic
    # baseline than picking one static "fastest" provider.
    if args.policies:
        policies = tuple(args.policies)
    else:
        policies = (
            "openrouter_auto",  # OpenRouter default (baseline)
            "sort_price",  # OpenRouter sort=price (equivalent to :floor)
            "sort_throughput",  # OpenRouter sort=throughput (equivalent to :nitro)
            "sort_latency",  # OpenRouter sort=latency (direct OR latency baseline)
            "lp_mix",  # LP optimal mixing (our method V1)
            "smart_hedge",  # LP + hedging (our method V1)
            "v2_p50_hedge",  # P50-rank + hedging (our method V2)
            "cheapest_fixed",  # Always cheapest provider (cost baseline)
        )

    # Add extra fixed-provider baselines
    extra_fixed_providers: tuple[str, ...] = ()
    if args.extra_fixed:
        extra_fixed_providers = tuple(args.extra_fixed)
        extra_policies = tuple(f"{p.lower()}_fixed" for p in args.extra_fixed)
        policies = policies + extra_policies

    # Create config
    config = EvaluationConfig(
        model=args.model,
        slo_sec=args.slo,
        warmup_sec=args.warmup,
        duration_sec=args.duration,
        cost_cap_usd=args.cost_cap,
        probing_interval_sec=args.probing_interval,
        extra_fixed_providers=extra_fixed_providers,
    )

    # Create output directory with timestamp (or use resume-dir)
    if args.resume_dir:
        output_dir = Path(args.resume_dir)
        if not output_dir.exists():
            print(f"Error: Resume directory not found: {args.resume_dir}")
            sys.exit(1)
        print(f"Resuming from: {output_dir}")
        resume_from_checkpoint = not args.no_resume
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output) / f"run_{timestamp}"
        resume_from_checkpoint = False  # New run, no checkpoint to resume from

    # Trace replay mode
    if args.trace:
        print("\n=== Trace Replay Mode ===")
        print(f"Loading trace from {args.trace}...")

        if not os.path.exists(args.trace):
            print(f"Error: Trace file not found at {args.trace}")
            sys.exit(1)

        # Determine if we should use real prompts (P1 mode)
        use_real_prompts = args.real_prompts or args.trace.endswith(".jsonl")

        if use_real_prompts:
            print("  Mode: P1 (real prompts from trace)")
            trace = load_trace_from_jsonl(
                args.trace,
                max_requests=args.max_requests,
                time_limit_sec=args.time_limit,
                use_real_prompts=True,
            )
        else:
            print("  Mode: P0 (fixed short prompt, arrival times only)")
            trace = load_trace_from_csv(
                args.trace,
                max_requests=args.max_requests,
                time_limit_sec=args.time_limit,
            )
        print(f"  Loaded {len(trace)} requests")

        if not trace:
            print("Error: No requests in trace")
            sys.exit(1)

        # Show sample prompt info for P1 mode
        if use_real_prompts and trace:
            sample = trace[0]
            print(f"  Sample prompt tokens: {sample.original_prompt_tokens}")
            print(f"  Sample max output tokens: {sample.max_tokens}")
            print(f"  Sample prompt preview: {sample.prompt[:80]}...")

        # Create budget config
        budget_config = BudgetConfig(
            max_requests=args.max_requests or len(trace),
            cost_cap_usd=args.cost_cap,
            max_tokens=args.max_output_tokens if use_real_prompts else config.max_tokens,
            policies=policies,
        )

        # Create trace replay evaluator
        evaluator = TraceReplayEvaluator(
            config=config,
            pricing=pricing,
            output_dir=output_dir,
            budget_config=budget_config,
            checkpoint_interval=args.checkpoint_interval,
        )

        print("\nConfiguration:")
        print(f"  Model: {config.model}")
        print(f"  SLO: {config.slo_sec}s")
        print(f"  Policies: {policies}")
        print(f"  Cost cap: ${budget_config.cost_cap_usd}")
        print(f"  Max requests: {budget_config.max_requests}")
        print(f"  Real prompts (P1 mode): {use_real_prompts}")
        print(f"  Max output tokens: {budget_config.max_tokens}")
        print(f"  Speedup: {args.speedup}x")
        print(f"  Checkpoint interval: {args.checkpoint_interval} requests")
        print(f"  Resume from checkpoint: {resume_from_checkpoint}")
        print(f"  Output: {output_dir}")

        if args.dry_run:
            print("\n[DRY RUN] Skipping actual API calls")
            return

        # Run warmup (optional but recommended) - skip if resuming
        if not args.skip_warmup and args.warmup > 0 and not resume_from_checkpoint:
            evaluator.warmup(
                duration_sec=min(args.warmup, 300)
            )  # Cap warmup at 5 min for trace replay

        # Run trace replay
        evaluator.run_trace_replay(
            trace=trace,
            policies=list(policies),
            speedup=args.speedup,
            resume=resume_from_checkpoint,
        )

    else:
        # Original duration-based mode
        evaluator = Phase5OnlineEvaluator(
            config=config,
            pricing=pricing,
            output_dir=output_dir,
        )

        print("\nConfiguration:")
        print(f"  Model: {config.model}")
        print(f"  SLO: {config.slo_sec}s")
        print(f"  Warmup: {config.warmup_sec/60:.1f} min")
        print(f"  Duration: {config.duration_sec/60:.1f} min")
        print(f"  Cost cap: ${config.cost_cap_usd}")
        print(f"  Output: {output_dir}")

        if args.dry_run:
            print("\n[DRY RUN] Skipping actual API calls")
            return

        # Run warmup
        if not args.skip_warmup:
            evaluator.warmup()

        # Run evaluation
        evaluator.run_evaluation(policies=list(policies) if args.policies else None)

    # Analyze and print results
    stats = evaluator.analyze_results()
    evaluator.print_summary(stats)
    evaluator.save_summary(stats)

    # Check validity
    is_valid, failures = evaluator.check_validity(stats)
    if is_valid:
        print("\nValidity check: PASSED")
    else:
        print("\nValidity check: FAILED")
        for failure in failures:
            print(f"  - {failure}")

    print("\nPhase 5 evaluation complete!")


if __name__ == "__main__":
    main()
