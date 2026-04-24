"""Baseline strategies for Stage 2 dual-subscription evaluation.

This module implements baseline strategies:
- B2: Daily-Quota-Only (Offline Optimal) - best possible with only S_Q
- B3: Concurrency-Only (Offline Optimal via ILP) - best possible with only S_C
- B4: Greedy-Online (FCFS) - practical online approach with both

System Semantics (Zero-wait / Loss system):
- All strategies use zero-wait semantics: requests cannot be queued.
- For S_C: if concurrency is full at arrival, request goes to API immediately.
- B3 uses latency_slo=0 to enforce this constraint in ILP.

Design rationale:
- B2 uses greedy selection (top-Q by cost) which is provably optimal for S_Q
- B3 uses ILP with daily_quota=0 and latency_slo=0 (zero-wait)
- B4 uses FCFS to show practical dual-subscription without optimization
- This allows clean comparisons:
  - ILP vs B2: value of adding S_C
  - ILP vs B3: value of adding S_Q
  - ILP vs B4: value of offline optimization
"""

import logging

import numpy as np

from rwsim.offline.schemas import Request, RoutingDecision
from rwsim.offline.strategy import RoutingStrategy

logger = logging.getLogger(__name__)


class DailyQuotaOnlyStrategy(RoutingStrategy):
    """B2: Daily-Quota-Only strategy (Offline Optimal).

    Uses only the daily-quota subscription (S_Q).
    Offline optimal: select top-Q requests by API cost each day.

    This is provably optimal for S_Q because:
    - Each day is independent (no time coupling)
    - Selecting highest-cost requests minimizes API spend

    Allows clean comparison: ILP vs B2 = value of adding S_C.
    """

    def __init__(self, *args, daily_quota: int = 5000, **kwargs):
        super().__init__(*args, **kwargs)
        self.daily_quota = daily_quota
        self.assignments: dict[int, str] = {}

        # Load model compatibility from config
        subscriptions = self.config.get("subscriptions", {})
        chutes_config = subscriptions.get("chutes", {})
        self.sq_supported_models: set[str] = set(chutes_config.get("supported_models", []))

    @property
    def name(self) -> str:
        """Return strategy name."""
        return "Daily-Quota-Only"

    def precompute(self, requests: list[Request]) -> None:
        """Precompute optimal daily quota allocation."""
        if not requests:
            return

        logger.info(f"Precomputing Daily-Quota-Only (Optimal) for {len(requests)} requests")

        # Log compatibility info
        if self.sq_supported_models:
            compatible_count = sum(1 for r in requests if r.model in self.sq_supported_models)
            logger.info(f"  Model compatibility: {compatible_count}/{len(requests)} for S_Q")

        # Group by day
        days: dict[int, list[Request]] = {}
        for req in requests:
            if req.day not in days:
                days[req.day] = []
            days[req.day].append(req)

        # For each day, select top Q by API cost (highest cost gets quota)
        for day_requests in days.values():
            api_costs = self._calculate_api_costs(day_requests)

            # Filter to only compatible models
            compatible_indices = [
                i
                for i, req in enumerate(day_requests)
                if (not self.sq_supported_models) or (req.model in self.sq_supported_models)
            ]

            if len(compatible_indices) <= self.daily_quota:
                # All compatible can use quota
                for idx in compatible_indices:
                    self.assignments[day_requests[idx].id] = "daily"
                # Incompatible go to API
                for idx, req in enumerate(day_requests):
                    if idx not in compatible_indices:
                        self.assignments[req.id] = "api"
            else:
                # Select top Q by highest API cost among compatible
                compatible_costs = api_costs[compatible_indices]
                top_k = min(self.daily_quota, len(compatible_indices))

                if top_k == 0:
                    # No quota available, all go to API
                    for req in day_requests:
                        self.assignments[req.id] = "api"
                else:
                    top_indices_in_compatible = np.argpartition(compatible_costs, -top_k)[-top_k:]
                    top_q_indices = {compatible_indices[i] for i in top_indices_in_compatible}

                    for idx, req in enumerate(day_requests):
                        if idx in top_q_indices:
                            self.assignments[req.id] = "daily"
                        else:
                            self.assignments[req.id] = "api"

        daily_count = sum(1 for v in self.assignments.values() if v == "daily")
        api_count = sum(1 for v in self.assignments.values() if v == "api")
        logger.info(f"Daily-Quota-Only: {daily_count} daily, {api_count} API")

    def _calculate_api_costs(self, requests: list[Request]) -> np.ndarray:
        """Calculate API costs for requests."""
        n = len(requests)
        request_tokens = np.array([r.request_tokens for r in requests])
        response_tokens = np.array([r.response_tokens for r in requests])
        models = [r.model for r in requests]

        model_pricing = self.config.get("model_pricing", {})

        if model_pricing and any(models):
            costs = np.zeros(n)
            for i in range(n):
                model = models[i]
                if not model:
                    raise ValueError(f"Request at index {i} has no model specified")
                pricing = model_pricing.get(model)
                if pricing is None:
                    raise ValueError(f"Model '{model}' not found in model_pricing")
                costs[i] = (
                    request_tokens[i] / 1_000_000.0 * pricing["input"]
                    + response_tokens[i] / 1_000_000.0 * pricing["output"]
                )
            return costs
        else:
            api_provider_id = self.config["simulation"].get(
                "default_api_fallback", "openai-chatgpt"
            )
            provider = self.config["providers"][api_provider_id]
            return (
                request_tokens / 1000.0 * provider.input_price_per_1k
                + response_tokens / 1000.0 * provider.output_price_per_1k
            )

    def route(self, request: Request) -> RoutingDecision:
        """Route using precomputed assignment."""
        self.quota_manager.reset_if_new_day(request.day)
        provider_type = self.assignments.get(request.id, "api")

        if provider_type == "daily":
            self.quota_manager.use_quota()
            self.subscription_used += 1
            return RoutingDecision(
                request=request,
                provider="daily-quota",
                cost=0.0,
                quota_used=1,
                timestamp=request.timestamp,
            )
        else:
            model_pricing = self.config.get("model_pricing", {})
            if model_pricing and request.model:
                cost = self.cost_calculator.calculate_cost_by_model(request)
                provider = "api"
            else:
                provider, cost = self.cost_calculator.get_cheapest_api_provider(request)
            self.api_used += 1
            return RoutingDecision(
                request=request,
                provider=provider,
                cost=cost,
                quota_used=0,
                timestamp=request.timestamp,
            )


class ConcurrencyOnlyStrategy(RoutingStrategy):
    """B3: Concurrency-Only strategy (Offline Optimal via ILP).

    Uses only the concurrency-limited subscription (S_C).
    Offline optimal: uses ILP solver with daily_quota=0 to find true optimal.

    This delegates to ILPOptimalStrategy with S_Q disabled, ensuring
    we get the true upper bound for S_C-only performance.

    Allows clean comparison: ILP vs B3 = value of adding S_Q.
    """

    def __init__(
        self,
        *args,
        concurrency_limit: int = 8,
        delta: float = 60.0,
        dataset_name: str | None = None,
        use_cache: bool = True,
        max_workers: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.concurrency_limit = concurrency_limit
        self.delta = delta
        self.dataset_name = dataset_name
        self.use_cache = use_cache
        self.max_workers = max_workers
        self.assignments: dict[int, str] = {}
        self._ilp_strategy = None

    @property
    def name(self) -> str:
        """Return strategy name."""
        return "Concurrency-Only"

    def precompute(self, requests: list[Request]) -> None:
        """Precompute optimal concurrency allocation using ILP."""
        if not requests:
            return

        logger.info(f"Precomputing Concurrency-Only (ILP Optimal) for {len(requests)} requests")

        # Import here to avoid circular dependency
        from experiments.offline_stage.strategies.stage2_optimal import ILPOptimalStrategy

        # Create ILP strategy with daily_quota=0 (S_Q disabled)
        # latency_slo=0: no queueing, must start immediately or go to API
        self._ilp_strategy = ILPOptimalStrategy(
            self.cost_calculator,
            self.quota_manager,
            self.config,
            delta=self.delta,
            daily_quota=0,  # Disable S_Q
            concurrency_limit=self.concurrency_limit,
            dataset_name=f"{self.dataset_name}_conconly" if self.dataset_name else None,
            use_cache=self.use_cache,
            max_workers=self.max_workers,
            latency_slo=0,  # Zero-wait: no queueing allowed
        )

        # Run ILP optimization
        self._ilp_strategy.precompute(requests)

        # Copy assignments, mapping 'concurrency' to our format
        for req_id, provider in self._ilp_strategy.assignments.items():
            if provider == "concurrency":
                self.assignments[req_id] = "concurrency"
            else:
                # Both 'daily' and 'api' from ILP become 'api' for us
                # (since we only have S_C, daily_quota=0 means no 'daily' assignments)
                self.assignments[req_id] = "api"

        conc_count = sum(1 for v in self.assignments.values() if v == "concurrency")
        api_count = sum(1 for v in self.assignments.values() if v == "api")
        logger.info(f"Concurrency-Only: {conc_count} concurrency, {api_count} API")

    def route(self, request: Request) -> RoutingDecision:
        """Route using precomputed assignment."""
        provider_type = self.assignments.get(request.id, "api")

        if provider_type == "concurrency":
            self.subscription_used += 1
            return RoutingDecision(
                request=request,
                provider="concurrency",
                cost=0.0,
                quota_used=0,
                timestamp=request.timestamp,
            )
        else:
            model_pricing = self.config.get("model_pricing", {})
            if model_pricing and request.model:
                cost = self.cost_calculator.calculate_cost_by_model(request)
                provider = "api"
            else:
                provider, cost = self.cost_calculator.get_cheapest_api_provider(request)
            self.api_used += 1
            return RoutingDecision(
                request=request,
                provider=provider,
                cost=cost,
                quota_used=0,
                timestamp=request.timestamp,
            )


class GreedyOnlineStrategy(RoutingStrategy):
    """B4: Greedy-Online strategy (FCFS).

    Online decision-making without future knowledge.
    At each request arrival:
    1. If model compatible with S_Q and has remaining daily quota -> use S_Q
    2. Else if model compatible with S_C and has available slot -> use S_C
    3. Else -> use API

    This shows PRACTICAL performance with dual subscriptions,
    allowing clean comparison: ILP vs B4 = value of offline optimization.
    """

    def __init__(self, *args, daily_quota: int = 5000, concurrency_limit: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.daily_quota = daily_quota
        self.concurrency_limit = concurrency_limit
        # Online state
        self.current_day = -1
        self.daily_used = 0
        self.active_slots: list[tuple[int, int]] = []  # (finish_time, slots_used)

        # Load model compatibility from config
        subscriptions = self.config.get("subscriptions", {})

        # S_Q (Chutes)
        chutes_config = subscriptions.get("chutes", {})
        self.sq_supported_models: set[str] = set(chutes_config.get("supported_models", []))

        # S_C (Featherless)
        featherless_config = subscriptions.get("featherless", {})
        featherless_models = featherless_config.get("supported_models", {})

        self.sc_supported_models: set[str] = set()
        self.sc_multipliers: dict[str, int] = {}

        if isinstance(featherless_models, dict):
            for model, props in featherless_models.items():
                self.sc_supported_models.add(model)
                self.sc_multipliers[model] = props.get("multiplier", 1)
        elif isinstance(featherless_models, list):
            self.sc_supported_models = set(featherless_models)
            for model in featherless_models:
                self.sc_multipliers[model] = 1

    @property
    def name(self) -> str:
        """Return strategy name."""
        return "Greedy-Online"

    def precompute(self, requests: list[Request]) -> None:
        """No precomputation for online strategy."""
        logger.info("Greedy-Online: no precomputation (online strategy)")
        if self.sq_supported_models or self.sc_supported_models:
            logger.info(
                f"  S_Q models: {len(self.sq_supported_models)}, "
                f"S_C models: {len(self.sc_supported_models)}"
            )
        self.current_day = -1
        self.daily_used = 0
        self.active_slots = []

    def route(self, request: Request) -> RoutingDecision:
        """Route request using online greedy decision with model compatibility."""
        # Reset daily quota if new day
        if request.day != self.current_day:
            self.current_day = request.day
            self.daily_used = 0
            # Reset quota manager for accurate stats
            self.quota_manager.reset_if_new_day(request.day)

        # Update active slots (remove finished)
        self.active_slots = [
            (ft, slots) for ft, slots in self.active_slots if ft > request.timestamp
        ]
        current_usage = sum(slots for _, slots in self.active_slots)

        # Check model compatibility
        sq_compatible = (not self.sq_supported_models) or (
            request.model in self.sq_supported_models
        )
        sc_compatible = (not self.sc_supported_models) or (
            request.model in self.sc_supported_models
        )
        multiplier = self.sc_multipliers.get(request.model, 1) if sc_compatible else 1

        # Decision: S_Q first (if compatible), then S_C (if compatible), then API
        if sq_compatible and self.daily_used < self.daily_quota:
            self.daily_used += 1
            self.quota_manager.use_quota()
            self.subscription_used += 1
            return RoutingDecision(
                request=request,
                provider="daily-quota",
                cost=0.0,
                quota_used=1,
                timestamp=request.timestamp,
            )
        elif sc_compatible and current_usage + multiplier <= self.concurrency_limit:
            finish_time = request.timestamp + int(request.latency_seconds)
            self.active_slots.append((finish_time, multiplier))
            self.subscription_used += 1
            return RoutingDecision(
                request=request,
                provider="concurrency",
                cost=0.0,
                quota_used=0,
                timestamp=request.timestamp,
            )
        else:
            model_pricing = self.config.get("model_pricing", {})
            if model_pricing and request.model:
                cost = self.cost_calculator.calculate_cost_by_model(request)
                provider = "api"
            else:
                provider, cost = self.cost_calculator.get_cheapest_api_provider(request)
            self.api_used += 1
            return RoutingDecision(
                request=request,
                provider=provider,
                cost=cost,
                quota_used=0,
                timestamp=request.timestamp,
            )
