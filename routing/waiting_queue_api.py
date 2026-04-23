"""Abstraction layer for SLO-aware request outsourcing in serving engines."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from .const import (
    ATTENTION_FLOP_MULTIPLIER,
    DEFAULT_DEVICE_TFLOPS,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_INPUT_PRICE_PER_TOKEN,
    DEFAULT_MAX_BATCH_SIZE,
    DEFAULT_NUM_ATTENTION_HEADS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_OUTPUT_PRICE_PER_TOKEN,
    DEFAULT_PREFILL_SLO_BASE_SECONDS,
    DEFAULT_PREFILL_SLO_SLACK_FACTOR,
    DEFAULT_UTILIZATION_TARGET,
    FFN_FLOP_MULTIPLIER,
    MIN_VALUE_FOR_RATIO,
    MIN_WEIGHT_VALUE,
    TFLOPS_TO_FLOPS,
)

# ============================================================================
# Request Abstraction
# ============================================================================


class RequestStatus(Enum):
    """Status of a request in the serving pipeline."""

    WAITING = "waiting"  # In queue, not yet scheduled
    RUNNING = "running"  # Currently being processed
    COMPLETED = "completed"  # Finished successfully
    OUTSOURCED = "outsourced"  # Sent to external service


@dataclass
class OutsourcingRequestInfo:
    """Request information needed for outsourcing decisions.

    This is a lightweight view that can be constructed from any serving engine's
    request object.
    """

    # Identity
    request_id: str

    # Timing information
    arrival_time: float  # When request entered the system
    queue_time: float = 0.0  # Time spent in queue so far

    # Token information
    num_prompt_tokens: int = 0  # Input/prefill tokens
    num_output_tokens: int = 0  # Expected output tokens (may be estimate)
    num_processed_tokens: int = 0  # Tokens already processed

    # SLO information (optional)
    prefill_slo_seconds: float | None = None  # TTFT deadline
    total_slo_seconds: float | None = None  # End-to-end deadline

    # Status
    status: RequestStatus = RequestStatus.WAITING
    is_prefill_complete: bool = False

    # Pricing/value (for knapsack)
    input_price_per_token: float = DEFAULT_INPUT_PRICE_PER_TOKEN
    output_price_per_token: float = DEFAULT_OUTPUT_PRICE_PER_TOKEN

    # Metadata
    metadata: dict = field(default_factory=dict)

    @property
    def estimated_value(self) -> float:
        """Revenue potential of this request."""
        return (
            self.num_prompt_tokens * self.input_price_per_token
            + self.num_output_tokens * self.output_price_per_token
        )

    @property
    def remaining_prompt_tokens(self) -> int:
        """Tokens left to process in prefill phase."""
        if self.is_prefill_complete:
            return 0
        return max(0, self.num_prompt_tokens - self.num_processed_tokens)

    @property
    def remaining_output_tokens(self) -> int:
        """Tokens left to generate in decode phase."""
        if not self.is_prefill_complete:
            return self.num_output_tokens  # Haven't started decode yet
        decode_done = max(0, self.num_processed_tokens - self.num_prompt_tokens)
        return max(0, self.num_output_tokens - decode_done)

    @property
    def prefill_deadline(self) -> float | None:
        """Absolute timestamp when prefill must complete."""
        if self.prefill_slo_seconds is None:
            return None
        return self.arrival_time + self.prefill_slo_seconds

    @property
    def total_deadline(self) -> float | None:
        """Absolute timestamp when entire request must complete."""
        if self.total_slo_seconds is None:
            return None
        return self.arrival_time + self.total_slo_seconds


# ============================================================================
# Waiting Queue Abstraction
# ============================================================================


class WaitingQueueInterface(ABC):
    """Abstract interface for the waiting request queue.

    Implementations wrap the serving engine's actual queue.
    """

    @abstractmethod
    def get_all_waiting(self) -> list[OutsourcingRequestInfo]:
        """Get snapshot of all waiting requests in queue order (FCFS).

        Returns a copy/snapshot to avoid iterator invalidation.
        """
        pass

    @abstractmethod
    def remove_requests(self, request_ids: set[str]) -> list[OutsourcingRequestInfo]:
        """Remove specified requests from the waiting queue.

        Returns the removed requests for logging/handoff.
        """
        pass

    @abstractmethod
    def get_length(self) -> int:
        """Current number of waiting requests."""
        pass

    @abstractmethod
    def peek(self) -> OutsourcingRequestInfo | None:
        """Look at the head of the queue without removing."""
        pass


# ============================================================================
# FLOP Calculator Abstraction
# ============================================================================


class FLOPCalculatorInterface(ABC):
    """Abstract interface for computing FLOPs required for request processing.

    Hides model architecture details from the outsourcing logic.
    """

    @abstractmethod
    def compute_prefill_flops(self, request: OutsourcingRequestInfo, num_tokens: int) -> float:
        """Calculate FLOPs needed to prefill `num_tokens` for this request.

        Accounts for: hidden_dim, num_layers, num_attention_heads, etc.
        """
        pass

    @abstractmethod
    def get_device_flops_per_second(self) -> float:
        """Peak FLOPs/second of the serving device (e.g., A100 = 312 TFLOPs)."""
        pass

    @abstractmethod
    def get_effective_flops_per_second(self, utilization: float = 0.8) -> float:
        """Effective FLOPs/second accounting for utilization efficiency."""
        pass


# ============================================================================
# Outsourcing Decision Engine
# ============================================================================


@dataclass
class OutsourcingDecision:
    """Result of an outsourcing decision cycle."""

    should_outsource: bool
    requests_to_outsource: list[str]  # Request IDs
    requests_to_keep: list[str]  # Request IDs
    reason: str  # Human-readable explanation
    metrics: dict = field(default_factory=dict)


class OutsourcingEngine:
    """Main engine that makes outsourcing decisions.

    Uses the abstractions above to remain engine-agnostic.
    """

    def __init__(
        self,
        waiting_queue: WaitingQueueInterface,
        flop_calculator: FLOPCalculatorInterface,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        prefill_slo_base_seconds: float = DEFAULT_PREFILL_SLO_BASE_SECONDS,
        prefill_slo_slack_factor: float = DEFAULT_PREFILL_SLO_SLACK_FACTOR,
        utilization_target: float = DEFAULT_UTILIZATION_TARGET,
    ):
        self.waiting_queue = waiting_queue
        self.flop_calculator = flop_calculator
        self.max_batch_size = max_batch_size
        self.prefill_slo_base_seconds = prefill_slo_base_seconds
        self.prefill_slo_slack_factor = prefill_slo_slack_factor
        self.utilization_target = utilization_target

        self.outsourced_request_ids: set[str] = set()

    def should_outsource(self, current_time: float) -> OutsourcingDecision:
        """Main entry point: decide if outsourcing is needed.

        Call this before each scheduling cycle.
        """
        waiting_requests = self.waiting_queue.get_all_waiting()

        if not waiting_requests:
            return OutsourcingDecision(
                should_outsource=False,
                requests_to_outsource=[],
                requests_to_keep=[],
                reason="No waiting requests",
            )

        # Check for SLO violations
        at_risk_requests = self._check_slo_violations(waiting_requests, current_time)

        if not at_risk_requests:
            return OutsourcingDecision(
                should_outsource=False,
                requests_to_outsource=[],
                requests_to_keep=[r.request_id for r in waiting_requests],
                reason="No SLO violations detected",
            )

        # Run knapsack to decide what to keep vs outsource
        keep_ids, outsource_ids = self._knapsack_selection(waiting_requests, current_time)

        return OutsourcingDecision(
            should_outsource=len(outsource_ids) > 0,
            requests_to_outsource=outsource_ids,
            requests_to_keep=keep_ids,
            reason=f"SLO violations detected for {len(at_risk_requests)} requests",
            metrics={
                "at_risk_count": len(at_risk_requests),
                "total_waiting": len(waiting_requests),
                "outsource_count": len(outsource_ids),
            },
        )

    def apply_outsourcing(self, decision: OutsourcingDecision) -> list[OutsourcingRequestInfo]:
        """Execute the outsourcing decision by removing requests from queue.

        Returns the outsourced requests for handoff to external service.
        """
        if not decision.should_outsource:
            return []

        outsourced = self.waiting_queue.remove_requests(set(decision.requests_to_outsource))

        for req in outsourced:
            self.outsourced_request_ids.add(req.request_id)

        return outsourced

    def _check_slo_violations(
        self, waiting_requests: list[OutsourcingRequestInfo], current_time: float
    ) -> list[OutsourcingRequestInfo]:
        """Check which waiting requests face imminent TTFT violations.

        Uses FLOP-based estimation for accuracy.
        """
        effective_flops = self.flop_calculator.get_effective_flops_per_second(
            self.utilization_target
        )

        at_risk = []

        # Compute cumulative FLOP load for FCFS queue
        cumulative_flops = 0.0
        for req in waiting_requests:
            # FLOPs needed ahead of this request
            ahead_flops = cumulative_flops

            # FLOPs needed for this request's remaining prefill
            req_flops = self.flop_calculator.compute_prefill_flops(req, req.remaining_prompt_tokens)

            # Total time to TTFT
            total_flops = ahead_flops + req_flops
            estimated_ttft = total_flops / effective_flops if effective_flops > 0 else float("inf")

            # Determine SLO (explicit or derived)
            if req.prefill_slo_seconds is not None:
                slo = req.prefill_slo_seconds
            else:
                # Derive default SLO based on request size
                slo = self.prefill_slo_base_seconds + (
                    self.prefill_slo_slack_factor * req_flops / effective_flops
                )

            # Check violation
            time_left = (req.arrival_time + slo) - current_time
            if estimated_ttft > time_left:
                at_risk.append(req)

            # Update cumulative for next request
            cumulative_flops += req_flops

        return at_risk

    def _knapsack_selection(
        self, waiting_requests: list[OutsourcingRequestInfo], current_time: float
    ) -> tuple[list[str], list[str]]:
        """Greedy knapsack: keep highest value/FLOP ratio requests locally.

        Returns: (keep_request_ids, outsource_request_ids)
        """
        # Build knapsack items
        items = []
        for req in waiting_requests:
            # Weight = total FLOPs remaining
            weight = 0.0
            if req.remaining_prompt_tokens > 0:
                weight += self.flop_calculator.compute_prefill_flops(
                    req, req.remaining_prompt_tokens
                )
            if req.remaining_output_tokens > 0:
                # Decode FLOPs scale with KV cache size, approximate per-token
                weight += self.flop_calculator.compute_decode_flops(
                    req, req.remaining_output_tokens
                )

            # Value = revenue potential
            value = req.estimated_value

            items.append(
                {
                    "id": req.request_id,
                    "weight": max(MIN_WEIGHT_VALUE, weight),
                    "value": max(MIN_VALUE_FOR_RATIO, value),
                    "ratio": value / max(MIN_WEIGHT_VALUE, weight),
                }
            )

        # Budget: ensure we can outsource at least the cheapest request
        min_weight = min(item["weight"] for item in items) if items else 0
        budget = max(0, sum(item["weight"] for item in items) - min_weight)

        # Greedy by value/weight ratio
        items.sort(key=lambda x: x["ratio"], reverse=True)

        keep = []
        total_weight = 0.0
        for item in items:
            if total_weight + item["weight"] <= budget:
                keep.append(item["id"])
                total_weight += item["weight"]

        keep_set = set(keep)
        outsource = [item["id"] for item in items if item["id"] not in keep_set]

        return keep, outsource


# ============================================================================
# Example Implementations (SGLang adapters)
# ============================================================================


class SGLangWaitingQueueAdapter(WaitingQueueInterface):
    """Example adapter for SGLang's waiting queue. Replace with actual SGLang API calls."""

    def __init__(self, sglang_scheduler):
        self.scheduler = sglang_scheduler

    def get_all_waiting(self) -> list[OutsourcingRequestInfo]:
        """Get snapshot of all waiting requests in queue order."""
        # Example: access SGLang's internal waiting queue
        # waiting_reqs = self.scheduler.waiting_queue
        waiting_reqs = []  # Placeholder

        return [
            OutsourcingRequestInfo(
                request_id=req.request_id,
                arrival_time=req.created_time,
                num_prompt_tokens=len(req.prompt_tokens),
                num_output_tokens=req.sampling_params.max_new_tokens,
                num_processed_tokens=0,
                status=RequestStatus.WAITING,
            )
            for req in waiting_reqs
        ]

    def remove_requests(self, request_ids: set[str]) -> list[OutsourcingRequestInfo]:
        """Remove requests from the waiting queue."""
        removed = []
        # Example: filter out requests from SGLang's queue
        # self.scheduler.waiting_queue = [
        #     req for req in self.scheduler.waiting_queue
        #     if req.request_id not in request_ids
        # ]
        return removed

    def get_length(self) -> int:
        """Current number of waiting requests."""
        # return len(self.scheduler.waiting_queue)
        return 0

    def peek(self) -> OutsourcingRequestInfo | None:
        """Peek at the first request in the waiting queue."""
        # if self.scheduler.waiting_queue:
        #     return self._convert_to_info(self.scheduler.waiting_queue[0])
        return None


class SimpleFLOPCalculator(FLOPCalculatorInterface):
    """Simple FLOP calculator based on transformer architecture."""

    def __init__(
        self,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_layers: int = DEFAULT_NUM_LAYERS,
        num_attention_heads: int = DEFAULT_NUM_ATTENTION_HEADS,
        device_tflops: float = DEFAULT_DEVICE_TFLOPS,  # A100
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.device_flops = device_tflops * TFLOPS_TO_FLOPS

    def compute_prefill_flops(self, request: OutsourcingRequestInfo, num_tokens: int) -> float:
        """Prefill: O(n^2 * d) attention + O(n * d^2) FFN, where n = sequence length, d = hidden_dim."""
        # Attention: 2 * n^2 * d per layer (QK^T + softmax * V)
        attn_flops = (
            ATTENTION_FLOP_MULTIPLIER * num_tokens * num_tokens * self.hidden_dim * self.num_layers
        )
        # FFN: 4 * n * d^2 per layer (two linear projections)
        ffn_flops = (
            FFN_FLOP_MULTIPLIER * num_tokens * self.hidden_dim * self.hidden_dim * self.num_layers
        )
        return attn_flops + ffn_flops

    def get_device_flops_per_second(self) -> float:
        """Return the theoretical device FLOPS per second."""
        return self.device_flops

    def get_effective_flops_per_second(self, utilization: float = 0.8) -> float:
        """Return effective FLOPS/second accounting for utilization."""
        return self.device_flops * utilization


# ============================================================================
# Usage Example (AKA Integration Snippet with other parts of the system)
# ============================================================================


def example_usage():
    """Example of how to integrate outsourcing into a serving loop."""
    # 1. Create adapters for serving engine
    # waiting_queue = SGLangWaitingQueueAdapter(sglang_scheduler)
    waiting_queue = None  # Replace with actual adapter

    flop_calculator = SimpleFLOPCalculator(
        hidden_dim=DEFAULT_HIDDEN_DIM,
        num_layers=DEFAULT_NUM_LAYERS,
        device_tflops=DEFAULT_DEVICE_TFLOPS,
    )

    # 2. Initialize outsourcing engine
    outsourcing = OutsourcingEngine(
        waiting_queue=waiting_queue,
        flop_calculator=flop_calculator,
        max_batch_size=DEFAULT_MAX_BATCH_SIZE,
        prefill_slo_base_seconds=DEFAULT_PREFILL_SLO_BASE_SECONDS,
        utilization_target=DEFAULT_UTILIZATION_TARGET,
    )

    # 3. In serving loop (before scheduling)
    current_time = time.time()

    # Check if outsourcing is needed
    decision = outsourcing.should_outsource(current_time)

    if decision.should_outsource:
        print(f"Outsourcing {decision.metrics['outsource_count']} requests: {decision.reason}")

        # Remove from local queue and get request info
        outsourced_requests = outsourcing.apply_outsourcing(decision)

        # Send to external service (e.g., OpenAI API)
        for req in outsourced_requests:
            print(
                f"  - Outsourcing request {req.request_id}: "
                f"{req.num_prompt_tokens} prompt tokens, "
                f"{req.num_output_tokens} expected output tokens"
            )
            # send_to_external_service(req)

    # 4. Continue with normal scheduling
    # scheduler.schedule_next_batch()


if __name__ == "__main__":
    example_usage()
