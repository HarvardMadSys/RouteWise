"""ILP-based optimal routing strategy for dual subscriptions (Phase 2).

This module implements the offline optimal solution for routing requests across:
- Daily-quota subscription (S_Q): Fixed daily quota, zero marginal cost
- Concurrency-limited subscription (S_C): Fixed concurrency, zero marginal cost
- Pay-per-token API (S_A): Unlimited capacity, per-token pricing

The optimization is formulated as a Mixed Integer Linear Program (MILP) that
jointly optimizes assignment and scheduling to minimize total API cost.

System Semantics:
- Default: Zero-wait / Loss system (latency_slo=0)
  Requests must start immediately upon arrival or be routed to API.
  This models real-world systems where queueing is not allowed.
- Optional: Queueing allowed (latency_slo>0 or None for unlimited)
  Requests can be delayed up to latency_slo slots before starting.
  Use --latency-slo=-1 for sensitivity analysis with unlimited queueing.
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pulp
from tqdm import tqdm

from llm_routewise.offline.cache import get_ilp_cache_key, load_cached_ilp_result, save_ilp_cache
from llm_routewise.offline.schemas import Request, RoutingDecision
from llm_routewise.offline.strategy import RoutingStrategy

logger = logging.getLogger(__name__)


@dataclass
class ILPParams:
    """Parameters for ILP solver (picklable for multiprocessing)."""

    delta: float
    daily_quota: int
    concurrency_limit: int
    latency_slo: int | None  # Max queueing delay in slots (0=zero-wait, None=unlimited)
    sq_supported_models: set[str]
    sc_supported_models: set[str]
    sc_concurrency_costs: dict[str, int]
    model_pricing: dict | None
    api_provider_input_price: float
    api_provider_output_price: float
    solver_name: str = "cbc"  # "cbc" or "gurobi"
    solver_time_limit: int | None = None  # Time limit in seconds (None = no limit)
    gurobi_license_file: str | None = None  # Path to Gurobi license file


def _solve_day_ilp_worker(
    day_idx: int,
    day_requests: list[Request],
    params: ILPParams,
) -> tuple[int, dict[int, str], dict[int, tuple[int, int]]]:
    """Solve ILP for a single day (module-level function for multiprocessing).

    Args:
        day_idx: Day index for logging
        day_requests: Requests for this day
        params: ILP parameters

    Returns:
        Tuple of (day_idx, assignments dict, schedules dict)
    """
    # Set Gurobi license file in subprocess (env vars not inherited from parent)
    if params.gurobi_license_file:
        os.environ["GRB_LICENSE_FILE"] = params.gurobi_license_file

    assignments: dict[int, str] = {}
    schedules: dict[int, tuple[int, int]] = {}

    n = len(day_requests)
    if n == 0:
        return day_idx, assignments, schedules

    # Time discretization
    min_time = min(r.timestamp for r in day_requests)

    # Calculate API costs
    request_tokens = np.array([r.request_tokens for r in day_requests])
    response_tokens = np.array([r.response_tokens for r in day_requests])
    models = [r.model for r in day_requests]

    if params.model_pricing and any(models):
        api_costs = np.zeros(n)
        for i in range(n):
            model = models[i]
            if model and model in params.model_pricing:
                pricing = params.model_pricing[model]
                api_costs[i] = request_tokens[i] / 1_000_000.0 * pricing.get(
                    "input", 0
                ) + response_tokens[i] / 1_000_000.0 * pricing.get("output", 0)
            else:
                api_costs[i] = (
                    request_tokens[i] / 1000.0 * params.api_provider_input_price
                    + response_tokens[i] / 1000.0 * params.api_provider_output_price
                )
    else:
        api_costs = (
            request_tokens / 1000.0 * params.api_provider_input_price
            + response_tokens / 1000.0 * params.api_provider_output_price
        )

    latencies = np.array([r.latency_seconds for r in day_requests])
    slots_needed = np.ceil(latencies / params.delta).astype(int)
    arrival_slots = (
        (np.array([r.timestamp for r in day_requests]) - min_time) / params.delta
    ).astype(int)

    # Horizon
    latest_finish_slot = int(np.max(arrival_slots + slots_needed))
    T = latest_finish_slot + 1

    # Create ILP problem
    prob = pulp.LpProblem(f"DualSubscriptionRouting_Day{day_idx}", pulp.LpMinimize)

    # Compatibility flags
    sq_compatible = [
        (not params.sq_supported_models) or (models[i] in params.sq_supported_models)
        for i in range(n)
    ]
    sc_compatible = [
        (not params.sc_supported_models) or (models[i] in params.sc_supported_models)
        for i in range(n)
    ]
    concurrency_costs = [
        params.sc_concurrency_costs.get(models[i], 1) if sc_compatible[i] else 1 for i in range(n)
    ]

    # Decision variables
    x_Q = [pulp.LpVariable(f"x_Q_{i}", cat="Binary") for i in range(n)]
    x_C = [pulp.LpVariable(f"x_C_{i}", cat="Binary") for i in range(n)]
    x_A = [pulp.LpVariable(f"x_A_{i}", cat="Binary") for i in range(n)]

    # Start time and occupancy variables
    tau = {}
    start_windows: dict[int, range] = {}
    active_windows: dict[int, range] = {}
    z = {}

    for i in range(n):
        latest_start = T - slots_needed[i]
        if params.latency_slo is not None:
            latest_start = min(latest_start, arrival_slots[i] + params.latency_slo)
        if latest_start < arrival_slots[i]:
            start_windows[i] = range(0, 0)
            active_windows[i] = range(0, 0)
            continue

        start_windows[i] = range(arrival_slots[i], latest_start + 1)
        active_windows[i] = range(arrival_slots[i], latest_start + slots_needed[i])

        for t in start_windows[i]:
            tau[i, t] = pulp.LpVariable(f"tau_{i}_{t}", cat="Binary")

    for i in range(n):
        for t in active_windows[i]:
            z[i, t] = pulp.LpVariable(f"z_{i}_{t}", cat="Binary")

    # Objective
    prob += pulp.lpSum(x_A[i] * api_costs[i] for i in range(n))

    # Constraints
    for i in range(n):
        prob += x_Q[i] + x_C[i] + x_A[i] == 1
        if not sq_compatible[i]:
            prob += x_Q[i] == 0
        if not sc_compatible[i]:
            prob += x_C[i] == 0

    prob += pulp.lpSum(x_Q[i] for i in range(n) if sq_compatible[i]) <= params.daily_quota

    for i in range(n):
        prob += pulp.lpSum(tau.get((i, t), 0) for t in start_windows[i]) == x_C[i]
        prob += pulp.lpSum(z.get((i, t), 0) for t in active_windows[i]) == slots_needed[i] * x_C[i]
        for t in active_windows[i]:
            window_start = max(arrival_slots[i], t - slots_needed[i] + 1)
            window_end = min(t, start_windows[i].stop - 1) if start_windows[i] else t
            prob += z.get((i, t), 0) <= pulp.lpSum(
                tau.get((i, t_prime), 0) for t_prime in range(window_start, window_end + 1)
            )

    for t in range(T):
        prob += (
            pulp.lpSum(z.get((i, t), 0) * concurrency_costs[i] for i in range(n) if (i, t) in z)
            <= params.concurrency_limit
        )

    # Select solver based on configuration (no fallback in worker - detection done in main process)
    if params.solver_name.lower() == "gurobi":
        solver = pulp.GUROBI(msg=0, timeLimit=params.solver_time_limit)
    else:
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=params.solver_time_limit)

    # Solve with explicit error handling for Gurobi runtime errors
    try:
        prob.solve(solver)
    except Exception as e:
        error_msg = str(e)
        if "gurobi" in error_msg.lower() or "license" in error_msg.lower():
            raise RuntimeError(
                f"Gurobi solver failed for day {day_idx}: {error_msg}\n"
                f"Try using CBC solver instead: --solver cbc"
            ) from e
        raise

    if prob.status != pulp.LpStatusOptimal:
        raise RuntimeError(
            f"ILP solver failed for day {day_idx} with {n} requests "
            f"(status={pulp.LpStatus[prob.status]}). Consider increasing delta."
        )

    # Extract solution
    for i, req in enumerate(day_requests):
        if pulp.value(x_Q[i]) > 0.5:
            assignments[req.id] = "daily"
        elif pulp.value(x_C[i]) > 0.5:
            assignments[req.id] = "concurrency"
            for t in start_windows[i]:
                if (i, t) in tau and pulp.value(tau[i, t]) > 0.5:
                    start_time = int(min_time + t * params.delta)
                    finish_time = int(start_time + slots_needed[i] * params.delta)
                    schedules[req.id] = (start_time, finish_time)
                    break
        else:
            assignments[req.id] = "api"

    return day_idx, assignments, schedules


class ILPOptimalStrategy(RoutingStrategy):
    """ILP-based optimal strategy for dual subscriptions.

    This strategy solves a joint assignment and scheduling problem using
    Integer Linear Programming to find the globally optimal routing that
    minimizes total cost across two subscription types and an API fallback.

    Time discretization:
    - Divides time into slots of size delta (e.g., 1.0 second)
    - Each request occupies multiple consecutive slots based on its latency
    - Concurrency constraint enforced per slot

    Decision variables:
    - x_i^Q, x_i^C, x_i^A: Binary assignment to daily quota, concurrency, or API
    - tau_i^t: Binary start time for request i at slot t (if assigned to S_C)
    - z_i,t: Binary occupancy of request i at slot t

    Attributes:
        delta: Time slot size in seconds (default: 1.0)
        daily_quota: Daily quota limit (Q)
        concurrency_limit: Concurrency limit (C)
        assignments: Precomputed routing assignments
        schedules: Precomputed start times for concurrency requests
    """

    def __init__(
        self,
        *args,
        delta: float = 1.0,
        daily_quota: int = 5000,
        concurrency_limit: int = 8,
        latency_slo: int | None = 0,  # Max queueing delay in slots (0=zero-wait, None=unlimited)
        solver: str = "cbc",
        solver_time_limit: int | None = None,
        dataset_name: str | None = None,
        use_cache: bool = True,
        max_workers: int | None = None,
        **kwargs,
    ):
        """Initialize ILP optimal strategy.

        Args:
            delta: Time slot size in seconds
            daily_quota: Daily quota limit
            concurrency_limit: Maximum concurrent requests (total slots)
            latency_slo: Max queueing delay in slots (0=zero-wait/loss system, None=unlimited)
            solver: Solver to use ("cbc" or "gurobi")
            solver_time_limit: Time limit per day in seconds (None = no limit)
            dataset_name: Dataset name for caching (e.g., "freeinference")
            use_cache: Whether to use ILP result caching
            max_workers: Max parallel workers for ILP solving (None = auto)
        """
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.daily_quota = daily_quota
        self.concurrency_limit = concurrency_limit
        self.assignments: dict[int, str] = {}  # request_id -> provider type
        self.schedules: dict[int, tuple[int, int]] = {}  # request_id -> (start_time, finish_time)
        self.latency_slo = latency_slo
        self.solver = solver.lower()
        self.solver_time_limit = solver_time_limit
        self.dataset_name = dataset_name
        self.max_workers = max_workers
        self.use_cache = use_cache

        # Load model compatibility from config
        self._load_subscription_config()

    def _load_subscription_config(self) -> None:
        """Load model compatibility from subscription config."""
        subscriptions = self.config.get("subscriptions", {})

        # S_Q (Chutes) - Daily Quota
        chutes_config = subscriptions.get("chutes", {})
        self.sq_supported_models: set[str] = set(chutes_config.get("supported_models", []))

        # S_C - Concurrency (read from active_sc_subscription set by runner)
        sc_config = self.config.get("active_sc_subscription", {})
        featherless_models = sc_config.get("supported_models", {})

        # Build supported models set and concurrency_cost dict
        self.sc_supported_models: set[str] = set()
        self.sc_concurrency_costs: dict[str, int] = {}

        if isinstance(featherless_models, dict):
            # New format: {model: {concurrency_cost: N}}
            for model, props in featherless_models.items():
                self.sc_supported_models.add(model)
                self.sc_concurrency_costs[model] = props.get("concurrency_cost", 1)
        elif isinstance(featherless_models, list):
            # Old format: [model1, model2, ...]
            self.sc_supported_models = set(featherless_models)
            for model in featherless_models:
                self.sc_concurrency_costs[model] = 1

        # Log configuration
        if self.sq_supported_models or self.sc_supported_models:
            logger.info("Model compatibility loaded:")
            logger.info(f"  S_Q (Chutes): {len(self.sq_supported_models)} models")
            logger.info(
                f"  S_C (Featherless): {len(self.sc_supported_models)} models, "
                f"concurrency_costs: {self.sc_concurrency_costs}"
            )

    @property
    def name(self) -> str:
        """Return strategy name."""
        return "ILP-Optimal"

    def _check_solver_availability(self) -> None:
        """Check if requested solver is available, fallback to CBC if not."""
        if self.solver == "gurobi":
            try:
                # Try to create a Gurobi solver instance
                pulp.GUROBI(msg=0)
            except Exception as e:
                logger.warning(
                    f"Gurobi solver not available ({e}), falling back to CBC. "
                    f"To suppress this warning, use --solver cbc"
                )
                self.solver = "cbc"

    def precompute(self, requests: list[Request]) -> None:
        """Precompute optimal routing using ILP with parallel processing.

        Args:
            requests: All requests in the dataset
        """
        if not requests:
            return

        # Check solver availability before computing cache key
        self._check_solver_availability()

        # Try loading from cache first
        cache_key = None
        if self.use_cache and self.dataset_name:
            cache_key = get_ilp_cache_key(
                dataset_name=self.dataset_name,
                num_requests=len(requests),
                delta=self.delta,
                daily_quota=self.daily_quota,
                concurrency_limit=self.concurrency_limit,
                solver=self.solver,
                latency_slo=self.latency_slo,
                model_pricing=self.config.get("model_pricing"),
                subscriptions=self.config.get("active_sc_subscription"),
                model_override=self.config.get("model_override"),
            )
            cached = load_cached_ilp_result(cache_key)
            if cached:
                self.assignments = cached["assignments"]
                self.schedules = cached["schedules"]
                logger.info(
                    f"Loaded from cache: "
                    f"{sum(1 for v in self.assignments.values() if v == 'daily')} daily quota, "
                    f"{sum(1 for v in self.assignments.values() if v == 'concurrency')} concurrency, "
                    f"{sum(1 for v in self.assignments.values() if v == 'api')} API"
                )
                return

        # Determine number of workers
        max_workers = self.max_workers if self.max_workers else (os.cpu_count() or 4)
        logger.info(f"Precomputing ILP-optimal routing for {len(requests)} requests")
        logger.info(
            f"Parameters: Q={self.daily_quota}, C={self.concurrency_limit}, "
            f"delta={self.delta}s, workers={max_workers}"
        )

        # Group by day
        days: dict[int, list[Request]] = {}
        for req in requests:
            if req.day not in days:
                days[req.day] = []
            days[req.day].append(req)

        logger.info(f"Solving ILP for {len(days)} days in parallel")

        # Prepare ILP parameters (picklable)
        api_provider_id = self.config["simulation"].get("default_api_fallback", "openai-chatgpt")
        api_provider = self.config["providers"][api_provider_id]

        params = ILPParams(
            delta=self.delta,
            daily_quota=self.daily_quota,
            concurrency_limit=self.concurrency_limit,
            latency_slo=self.latency_slo,
            sq_supported_models=self.sq_supported_models,
            sc_supported_models=self.sc_supported_models,
            sc_concurrency_costs=self.sc_concurrency_costs,
            model_pricing=self.config.get("model_pricing"),
            api_provider_input_price=api_provider.input_price_per_1k,
            api_provider_output_price=api_provider.output_price_per_1k,
            solver_name=self.solver,
            solver_time_limit=self.solver_time_limit,
            gurobi_license_file=os.environ.get("GRB_LICENSE_FILE"),
        )

        # Submit all days to process pool
        sorted_days = sorted(days.items())

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(_solve_day_ilp_worker, day_idx, day_requests, params): day_idx
                for day_idx, day_requests in sorted_days
            }

            # Collect results with progress bar
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="ILP solving days",
                unit="day",
                ncols=100,
            ):
                _day_idx, day_assignments, day_schedules = future.result()
                self.assignments.update(day_assignments)
                self.schedules.update(day_schedules)

        # Save to cache
        if cache_key:
            save_ilp_cache(cache_key, self.assignments, self.schedules)

        logger.info(
            f"ILP optimization complete: "
            f"{sum(1 for v in self.assignments.values() if v == 'daily')} daily quota, "
            f"{sum(1 for v in self.assignments.values() if v == 'concurrency')} concurrency, "
            f"{sum(1 for v in self.assignments.values() if v == 'api')} API"
        )

    def _calculate_api_costs(self, requests: list[Request]) -> np.ndarray:
        """Calculate API costs for all requests.

        Supports multi-model pricing when model_pricing is configured.

        Args:
            requests: List of requests

        Returns:
            Array of API costs
        """
        n = len(requests)
        request_tokens = np.array([r.request_tokens for r in requests])
        response_tokens = np.array([r.response_tokens for r in requests])
        models = [r.model for r in requests]

        # Check if multi-model pricing is available
        model_pricing = self.config.get("model_pricing", {})

        if model_pricing and any(models):
            # Multi-model pricing: calculate cost per model
            costs = np.zeros(n)
            for i in range(n):
                model = models[i]
                if not model:
                    raise ValueError(f"Request at index {i} has no model specified")

                pricing = model_pricing.get(model)
                if pricing is None:
                    raise ValueError(
                        f"Model '{model}' not found in model_pricing. "
                        f"Please add pricing in experiments/offline_stage/configs/experiment.yaml. "
                        f"Available: {list(model_pricing.keys())}"
                    )

                # Pricing is per 1M tokens
                input_price = pricing.get("input")
                output_price = pricing.get("output")
                if input_price is None or output_price is None:
                    raise ValueError(f"Model '{model}' has incomplete pricing")

                costs[i] = (
                    request_tokens[i] / 1_000_000.0 * input_price
                    + response_tokens[i] / 1_000_000.0 * output_price
                )
            return costs
        else:
            # Single-model pricing: use default API provider
            api_provider_id = self.config["simulation"].get(
                "default_api_fallback", "openai-chatgpt"
            )
            provider = self.config["providers"][api_provider_id]

            input_price = provider.input_price_per_1k
            output_price = provider.output_price_per_1k

            return request_tokens / 1000.0 * input_price + response_tokens / 1000.0 * output_price

    def route(self, request: Request) -> RoutingDecision:
        """Route request using precomputed assignment.

        Args:
            request: Request to route

        Returns:
            Routing decision
        """
        # Reset quota if entering new day
        self.quota_manager.reset_if_new_day(request.day)

        # Use precomputed assignment (keyed by request.id for uniqueness)
        provider_type = self.assignments.get(request.id, "api")

        if provider_type == "daily":
            # Use daily quota subscription
            self.quota_manager.use_quota()
            self.subscription_used += 1

            return RoutingDecision(
                request=request,
                provider="daily-quota",
                cost=0.0,
                quota_used=1,
                timestamp=request.timestamp,
            )
        elif provider_type == "concurrency":
            # Use concurrency subscription
            self.subscription_used += 1
            schedule = self.schedules.get(request.id)
            if schedule:
                start_time, finish_time = schedule
            else:
                start_time = request.timestamp
                finish_time = int(start_time + request.latency_seconds)

            return RoutingDecision(
                request=request,
                provider="concurrency",
                cost=0.0,
                quota_used=0,
                timestamp=request.timestamp,
                start_time=start_time,
                finish_time=finish_time,
            )
        else:
            # Use API with model-specific pricing
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
