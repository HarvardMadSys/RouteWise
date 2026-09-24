"""Offline fixed-start routing oracle for cost-layer simulator sections.

This module defines the offline baseline used by the simulator as a routing
oracle, not as a queueing or scheduling oracle. The oracle receives the full
request trace up front, including arrivals, token counts, API fallback costs,
and provider-model service-duration estimates. With that future knowledge it
chooses the provider assignment that minimizes marginal API cost under the
selected provider-capacity constraints.

The main semantics are fixed-start and no-queueing:

* Each request must be assigned to exactly one provider at its arrival time.
* If a request uses a concurrency-limited subscription provider, its occupied
  interval is fixed to [arrival_time, arrival_time + service_duration].
* The oracle may choose which fixed intervals receive subscription capacity,
  but it may not delay, reorder, or preempt requests to create future capacity.

Service-duration assumption:

* The main simulator uses provider-model duration rather than BurstGPT's
  recorded elapsed-time field. Arrival times and token lengths come from the
  workload trace; service duration for concurrency occupancy is derived from
  provider TTFT and throughput profiles.
* Current online simulation samples service time at dispatch as
  sampled_TTFT + response_tokens / sampled_TPS, unless an explicit provider
  service-time distribution is configured.
* Current offline routing baselines use a deterministic counterpart,
  p50_TTFT + response_tokens / p50_TPS. This keeps the oracle fixed-start and
  provider-model based, but it is not yet a per-request realized-duration
  oracle. A stricter future implementation should materialize provider-model
  durations before the run and feed the same realized durations to both the
  simulator's capacity accounting and the offline oracle.
* BurstGPT `elapsed_time_sec` is preserved as metadata and can be used for a
  separate robustness check, but it is not mixed into the main capacity model.

Under those semantics, an exact offline oracle is allowed to exploit future
knowledge to reserve quota or concurrency capacity for requests with higher
API fallback value. It is not allowed to introduce start-time variables or
time-discretized scheduling decisions. A time-discretized model with chosen
start slots would be a stronger offline scheduler and should be reported
separately as a queueing/scheduling upper bound.

The intended scalable formulation for joint quota/concurrency routing is a
sparse event-driven MILP. Quota constraints are applied per provider window.
Concurrency constraints should be expressed as cumulative load over sorted
start/end events, so each request contributes only at its start and end events
instead of scanning every interval at every event point.

Current implementation notes:

* API-only, single-window quota, and fixed-start concurrency baselines have
  exact implementations for their supported settings.
* Multi-window quota currently uses a value-greedy baseline and is not a
  globally exact oracle.
* Joint quota/concurrency exact solving is intentionally bounded by request
  count until the sparse event-driven MILP path is implemented for full traces
  and multiple quota/concurrency providers.
"""

from __future__ import annotations

import bisect
import heapq
import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import pulp

from rwsim.metrics import PerRequestRecord, Run, RunAggregator, Status
from rwsim.world.capacity import ProviderTier

if TYPE_CHECKING:
    from rwsim.schemas import Request
    from rwsim.world.providers import TieredProvider
    from rwsim.world.scenarios import ScenarioConfig

OFFLINE_POLICY = "offline"
_DEFAULT_JOINT_EXACT_MAX_REQUESTS = 5_000
_DEFAULT_MILP_SEED = 42
_DEFAULT_MILP_SOLVER = "cbc"


class OfflineOracleKind(str, Enum):
    """Named offline oracle semantics used by cost-layer baselines."""

    API_ONLY = "api_only"
    STAGE_Q_EXACT_SINGLE_WINDOW = "stage_q_exact_single_window"
    STAGE_Q_GREEDY_VALUE_MULTI_WINDOW = "stage_q_greedy_value_multi_window"
    STAGE_C_GREEDY_INTERVAL = "stage_c_greedy_interval"
    STAGE_C_EXACT = "stage_c_exact"
    STAGE_QC_EXACT = "stage_qc_exact"
    STAGE_QC_BEST_DECOMPOSITION = "stage_qc_best_decomposition"
    STAGE_QC_EXACT_SLICE = "stage_qc_exact_slice"


@dataclass(frozen=True)
class OfflineAssignment:
    """One deterministic offline assignment decision."""

    request_id: int
    provider_name: str
    provider_tier: ProviderTier
    avoided_api_cost_usd: float
    oracle_kind: OfflineOracleKind
    start_sec: float | None = None
    finish_sec: float | None = None


def run_offline_oracle_policy(
    scenario: ScenarioConfig,
    requests: list[Request],
    seed: int,
    retain_records: bool = True,
) -> Run:
    """Run the cost-layer offline baseline with full trace knowledge.

    ``seed`` is accepted for runner-interface compatibility. The oracle is
    deterministic under stable value tie-breaks.
    """
    del seed
    assignments = assign_offline(scenario, requests)
    providers_by_name = {provider.name: provider for provider in scenario.providers}
    aggregator = RunAggregator(
        policy=OFFLINE_POLICY,
        scenario_name=scenario.name,
        source="simulation",
        retain_records=retain_records,
    )
    for request in requests:
        assignment = assignments[request.id]
        aggregator.observe(
            _offline_record(
                scenario=scenario,
                request=request,
                assignment=assignment,
                provider=providers_by_name[assignment.provider_name],
            )
        )
    return aggregator.finalize()


def assign_offline(
    scenario: ScenarioConfig,
    requests: list[Request],
    *,
    kind: OfflineOracleKind | str = "auto",
) -> dict[int, OfflineAssignment]:
    """Return deterministic offline assignments for a cost-layer scenario."""
    providers = list(scenario.providers)
    api_provider = _cheapest_api_provider(providers)
    if api_provider is None:
        raise ValueError(f"offline baseline requires at least one API provider: {scenario.name}")

    quota_providers = [provider for provider in providers if provider.tier == ProviderTier.S_Q]
    concurrency_providers = [
        provider for provider in providers if provider.tier == ProviderTier.S_C
    ]
    oracle_kind = _resolve_oracle_kind(
        kind,
        quota_providers=quota_providers,
        concurrency_providers=concurrency_providers,
    )

    provider_assignments = {request.id: api_provider for request in requests}
    if quota_providers and oracle_kind not in {
        OfflineOracleKind.STAGE_QC_EXACT,
        OfflineOracleKind.STAGE_QC_EXACT_SLICE,
    }:
        provider_assignments.update(
            _quota_assignments(requests, quota_providers, api_provider)
        )
    if concurrency_providers:
        if oracle_kind in {
            OfflineOracleKind.STAGE_QC_EXACT,
            OfflineOracleKind.STAGE_QC_EXACT_SLICE,
        }:
            provider_assignments.update(
                _joint_exact_assignments(
                    requests,
                    quota_providers,
                    concurrency_providers,
                    api_provider,
                )
            )
        elif oracle_kind == OfflineOracleKind.STAGE_C_EXACT:
            provider_assignments.update(
                _concurrency_exact_assignments(
                    requests,
                    concurrency_providers,
                    api_provider,
                )
            )
        else:
            provider_assignments.update(
                _concurrency_greedy_assignments(
                    requests,
                    concurrency_providers,
                    api_provider,
                )
            )

    return {
        request.id: OfflineAssignment(
            request_id=request.id,
            provider_name=provider_assignments[request.id].name,
            provider_tier=provider_assignments[request.id].tier,
            avoided_api_cost_usd=_api_cost(api_provider, request),
            oracle_kind=oracle_kind,
        )
        for request in requests
    }


def _resolve_oracle_kind(
    kind: OfflineOracleKind | str,
    *,
    quota_providers: list[TieredProvider],
    concurrency_providers: list[TieredProvider],
) -> OfflineOracleKind:
    if kind != "auto":
        resolved = OfflineOracleKind(kind)
    elif quota_providers and concurrency_providers:
        resolved = OfflineOracleKind.STAGE_QC_EXACT
    elif concurrency_providers:
        resolved = OfflineOracleKind.STAGE_C_EXACT
    elif quota_providers:
        has_multi_window = any(
            len(_provider_quota_windows(provider)) > 1 for provider in quota_providers
        )
        resolved = (
            OfflineOracleKind.STAGE_Q_GREEDY_VALUE_MULTI_WINDOW
            if has_multi_window
            else OfflineOracleKind.STAGE_Q_EXACT_SINGLE_WINDOW
        )
    else:
        resolved = OfflineOracleKind.API_ONLY

    if resolved in {
        OfflineOracleKind.STAGE_C_EXACT,
        OfflineOracleKind.STAGE_C_GREEDY_INTERVAL,
    } and (quota_providers or not concurrency_providers):
        raise ValueError(f"{resolved.value} requires S_C + S_A and no S_Q provider")
    if resolved in {
        OfflineOracleKind.STAGE_QC_EXACT,
        OfflineOracleKind.STAGE_QC_EXACT_SLICE,
    } and (not quota_providers or not concurrency_providers):
        raise ValueError(f"{resolved.value} requires both S_Q and S_C providers")
    if resolved in {
        OfflineOracleKind.STAGE_QC_BEST_DECOMPOSITION,
    }:
        raise NotImplementedError("joint decomposition baseline is not an oracle")
    return resolved


def _quota_assignments(
    requests: list[Request],
    quota_providers: list[TieredProvider],
    api_provider: TieredProvider,
) -> dict[int, TieredProvider]:
    assignments: dict[int, TieredProvider] = {}
    trace_start = float(requests[0].timestamp) if requests else 0.0
    quota_usage: dict[str, dict[tuple[int, int], int]] = {
        provider.name: {} for provider in quota_providers
    }
    ranked = sorted(
        requests,
        key=lambda request: (
            _api_cost(api_provider, request),
            -(request.timestamp),
            -request.id,
        ),
        reverse=True,
    )
    for request in ranked:
        for provider in quota_providers:
            quota_windows = _provider_quota_windows(provider)
            if _quota_windows_have_capacity(
                quota_usage[provider.name],
                quota_windows,
                request,
                trace_start,
            ):
                _charge_quota_windows(
                    quota_usage[provider.name],
                    quota_windows,
                    request,
                    trace_start,
                )
                assignments[request.id] = provider
                break
    return assignments


def _provider_quota_windows(provider: TieredProvider) -> tuple[tuple[int, float], ...]:
    quota = provider.quota
    if quota is None:
        return ()
    if hasattr(quota, "windows"):
        return tuple((window.size, window.window_sec) for window in quota.windows)
    return ((quota.size, quota.window_sec),)


def _quota_windows_have_capacity(
    usage: dict[tuple[int, int], int],
    quota_windows: tuple[tuple[int, float], ...],
    request: Request,
    trace_start: float,
) -> bool:
    return all(
        usage.get((index, _quota_window_id(request, window_sec, trace_start)), 0)
        < size
        for index, (size, window_sec) in enumerate(quota_windows)
    )


def _charge_quota_windows(
    usage: dict[tuple[int, int], int],
    quota_windows: tuple[tuple[int, float], ...],
    request: Request,
    trace_start: float,
) -> None:
    for index, (_, window_sec) in enumerate(quota_windows):
        key = (index, _quota_window_id(request, window_sec, trace_start))
        usage[key] = usage.get(key, 0) + 1


def _quota_window_id(request: Request, window_sec: float, trace_start: float) -> int:
    return int((float(request.timestamp) - trace_start) // float(window_sec))


def _concurrency_greedy_assignments(
    requests: list[Request],
    concurrency_providers: list[TieredProvider],
    api_provider: TieredProvider,
) -> dict[int, TieredProvider]:
    slots: list[tuple[TieredProvider, list[tuple[float, float]]]] = [
        (provider, [])
        for provider in concurrency_providers
        for _ in range(provider.concurrency.limit if provider.concurrency is not None else 0)
    ]
    assignments: dict[int, TieredProvider] = {}
    ranked = sorted(
        requests,
        key=lambda request: (
            _api_cost(api_provider, request),
            -(request.timestamp),
            -request.id,
        ),
        reverse=True,
    )
    for request in ranked:
        start = float(request.timestamp)
        for provider, intervals in slots:
            end = start + _offline_service_time_sec(provider, request)
            if _interval_fits(intervals, start, end):
                bisect.insort(intervals, (start, end))
                assignments[request.id] = provider
                break
    return assignments


class _MinCostFlow:
    """Small min-cost flow implementation for exact interval selection."""

    def __init__(self, n_nodes: int) -> None:
        self.graph: list[list[list[float | int]]] = [[] for _ in range(n_nodes)]

    def add_edge(
        self,
        source: int,
        target: int,
        capacity: int,
        cost: float,
    ) -> int:
        forward = [target, capacity, cost, len(self.graph[target])]
        reverse = [source, 0, -cost, len(self.graph[source])]
        self.graph[source].append(forward)
        self.graph[target].append(reverse)
        return len(self.graph[source]) - 1

    def min_cost(self, source: int, sink: int, flow: int) -> float:
        if flow <= 0 or source == sink:
            return 0.0
        potential = self._initial_potentials(source)
        total_cost = 0.0
        sent = 0
        while sent < flow:
            dist = [float("inf")] * len(self.graph)
            parent_node = [-1] * len(self.graph)
            parent_edge = [-1] * len(self.graph)
            dist[source] = 0.0
            heap: list[tuple[float, int]] = [(0.0, source)]
            while heap:
                distance, node = heapq.heappop(heap)
                if distance > dist[node] + 1e-18:
                    continue
                for edge_index, edge in enumerate(self.graph[node]):
                    if int(edge[1]) <= 0:
                        continue
                    target = int(edge[0])
                    reduced_cost = float(edge[2]) + potential[node] - potential[target]
                    next_distance = distance + max(reduced_cost, 0.0)
                    if next_distance + 1e-18 < dist[target]:
                        dist[target] = next_distance
                        parent_node[target] = node
                        parent_edge[target] = edge_index
                        heapq.heappush(heap, (next_distance, target))

            if parent_node[sink] < 0:
                break

            for node, distance in enumerate(dist):
                if distance < float("inf"):
                    potential[node] += distance

            add = flow - sent
            node = sink
            while node != source:
                edge = self.graph[parent_node[node]][parent_edge[node]]
                add = min(add, int(edge[1]))
                node = parent_node[node]

            node = sink
            while node != source:
                prev = parent_node[node]
                edge_index = parent_edge[node]
                edge = self.graph[prev][edge_index]
                reverse = self.graph[node][int(edge[3])]
                edge[1] = int(edge[1]) - add
                reverse[1] = int(reverse[1]) + add
                total_cost += float(edge[2]) * add
                node = prev
            sent += add
        return total_cost

    def _initial_potentials(self, source: int) -> list[float]:
        distances = [float("inf")] * len(self.graph)
        distances[source] = 0.0
        for node in range(source, len(self.graph)):
            if distances[node] == float("inf"):
                continue
            for edge in self.graph[node]:
                if int(edge[1]) <= 0:
                    continue
                target = int(edge[0])
                if target <= node:
                    continue
                candidate = distances[node] + float(edge[2])
                if candidate < distances[target]:
                    distances[target] = candidate
        return [0.0 if distance == float("inf") else distance for distance in distances]


def _concurrency_exact_assignments(
    requests: list[Request],
    concurrency_providers: list[TieredProvider],
    api_provider: TieredProvider,
) -> dict[int, TieredProvider]:
    if not requests:
        return {}
    if len(concurrency_providers) != 1:
        raise NotImplementedError("stage_c_exact currently supports one S_C provider")

    provider = concurrency_providers[0]
    limit = provider.concurrency.limit if provider.concurrency is not None else 0
    if limit <= 0:
        return {}

    intervals: list[tuple[float, float, float, int]] = []
    endpoints: set[float] = set()
    for request in requests:
        start = float(request.timestamp)
        end = start + _offline_service_time_sec(provider, request)
        value = _api_cost(api_provider, request)
        if end <= start or value <= 0.0:
            continue
        intervals.append((start, end, value, request.id))
        endpoints.add(start)
        endpoints.add(end)
    if not intervals:
        return {}

    ordered_endpoints = sorted(endpoints)
    endpoint_index = {value: index for index, value in enumerate(ordered_endpoints)}
    flow = _MinCostFlow(len(ordered_endpoints))
    for index in range(len(ordered_endpoints) - 1):
        flow.add_edge(index, index + 1, limit, 0.0)

    request_edges: dict[int, tuple[int, int]] = {}
    for start, end, value, request_id in intervals:
        source = endpoint_index[start]
        target = endpoint_index[end]
        edge_index = flow.add_edge(source, target, 1, -value)
        request_edges[request_id] = (source, edge_index)

    flow.min_cost(0, len(ordered_endpoints) - 1, limit)
    assignments: dict[int, TieredProvider] = {}
    for request_id, (source, edge_index) in request_edges.items():
        edge = flow.graph[source][edge_index]
        if int(edge[1]) == 0:
            assignments[request_id] = provider
    return assignments


def _joint_exact_assignments(
    requests: list[Request],
    quota_providers: list[TieredProvider],
    concurrency_providers: list[TieredProvider],
    api_provider: TieredProvider,
) -> dict[int, TieredProvider]:
    if not requests:
        return {}
    if len(quota_providers) != 1:
        raise NotImplementedError("stage_qc_exact currently supports one S_Q provider")
    if len(concurrency_providers) != 1:
        raise NotImplementedError("stage_qc_exact currently supports one S_C provider")
    max_requests = _joint_exact_max_requests()
    if len(requests) > max_requests:
        raise NotImplementedError(
            "stage_qc_exact currently supports bounded slices up to "
            f"{max_requests} requests; full-trace joint solving needs "
            "a batched/optimized ILP path before paper use"
        )

    quota_provider = quota_providers[0]
    concurrency_provider = concurrency_providers[0]
    concurrency_limit = (
        concurrency_provider.concurrency.limit
        if concurrency_provider.concurrency is not None
        else 0
    )
    quota_windows = _provider_quota_windows(quota_provider)
    if not quota_windows:
        raise ValueError("stage_qc_exact requires S_Q provider quota windows")

    trace_start = min(float(request.timestamp) for request in requests)
    request_indices = {request.id: index for index, request in enumerate(requests)}
    values = [_api_cost(api_provider, request) for request in requests]
    intervals = [
        (
            float(request.timestamp),
            float(request.timestamp) + _offline_service_time_sec(concurrency_provider, request),
        )
        for request in requests
    ]

    problem = pulp.LpProblem("StageQCExactOfflineOracle", pulp.LpMaximize)
    x_quota = [
        pulp.LpVariable(f"x_q_{index}", lowBound=0, upBound=1, cat="Binary")
        for index in range(len(requests))
    ]
    x_concurrency = [
        pulp.LpVariable(f"x_c_{index}", lowBound=0, upBound=1, cat="Binary")
        for index in range(len(requests))
    ]

    problem += pulp.lpSum(
        values[index] * (x_quota[index] + x_concurrency[index])
        for index in range(len(requests))
    )

    for index, value in enumerate(values):
        problem += x_quota[index] + x_concurrency[index] <= 1
        if value <= 0.0:
            problem += x_quota[index] == 0
            problem += x_concurrency[index] == 0
        if concurrency_limit <= 0 or intervals[index][1] <= intervals[index][0]:
            problem += x_concurrency[index] == 0

    quota_usage: dict[tuple[int, int], list[pulp.LpVariable]] = {}
    for request in requests:
        index = request_indices[request.id]
        for window_index, (_, window_sec) in enumerate(quota_windows):
            key = (window_index, _quota_window_id(request, window_sec, trace_start))
            quota_usage.setdefault(key, []).append(x_quota[index])
    for window_index, (size, _) in enumerate(quota_windows):
        for key, variables in quota_usage.items():
            if key[0] == window_index:
                problem += pulp.lpSum(variables) <= size

    event_points = sorted({point for interval in intervals for point in interval})
    for point in event_points:
        active = [
            x_concurrency[index]
            for index, (start, end) in enumerate(intervals)
            if start <= point < end
        ]
        if active:
            problem += pulp.lpSum(active) <= concurrency_limit

    solver = _make_milp_solver()
    try:
        problem.solve(solver)
    finally:
        close_solver = getattr(solver, "close", None)
        if close_solver is not None:
            close_solver()
    if problem.status != pulp.LpStatusOptimal:
        status = pulp.LpStatus.get(problem.status, str(problem.status))
        raise RuntimeError(f"stage_qc_exact solver failed with status={status}")

    assignments: dict[int, TieredProvider] = {}
    for request in requests:
        index = request_indices[request.id]
        if pulp.value(x_quota[index]) > 0.5:
            assignments[request.id] = quota_provider
        elif pulp.value(x_concurrency[index]) > 0.5:
            assignments[request.id] = concurrency_provider
    return assignments


def _make_milp_solver() -> pulp.LpSolver:
    solver_name = _milp_solver_name()
    seed = _milp_seed()
    time_limit = _milp_time_limit_sec()
    if solver_name == "cbc":
        return pulp.PULP_CBC_CMD(
            msg=0,
            timeLimit=time_limit,
            options=[f"randomCbcSeed {seed}"],
        )
    if solver_name == "gurobi":
        return pulp.GUROBI(
            msg=0,
            timeLimit=time_limit,
            manageEnv=True,
            Seed=seed,
        )
    if solver_name == "gurobi_cmd":
        return pulp.GUROBI_CMD(
            msg=0,
            timeLimit=time_limit,
            options=[("Seed", seed)],
        )
    raise ValueError(
        "ROUTEWISE_OFFLINE_MILP_SOLVER must be one of cbc, gurobi, gurobi_cmd"
    )


def _milp_solver_name() -> str:
    return os.environ.get("ROUTEWISE_OFFLINE_MILP_SOLVER", _DEFAULT_MILP_SOLVER).lower()


def _milp_seed() -> int:
    return _env_int("ROUTEWISE_OFFLINE_MILP_SEED", _DEFAULT_MILP_SEED)


def _milp_time_limit_sec() -> float | None:
    raw = os.environ.get("ROUTEWISE_OFFLINE_MILP_TIME_LIMIT_SEC")
    if raw is None or not raw.strip():
        return None
    return float(raw)


def _joint_exact_max_requests() -> int:
    return _env_int(
        "ROUTEWISE_OFFLINE_JOINT_MAX_REQUESTS",
        _DEFAULT_JOINT_EXACT_MAX_REQUESTS,
    )


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _interval_fits(
    sorted_intervals: list[tuple[float, float]],
    start: float,
    end: float,
) -> bool:
    """Return whether [start, end) fits in a start-sorted interval list."""
    index = bisect.bisect_left(sorted_intervals, (start, end))
    if index > 0 and sorted_intervals[index - 1][1] > start:
        return False
    return index >= len(sorted_intervals) or end <= sorted_intervals[index][0]


def _offline_record(
    *,
    scenario: ScenarioConfig,
    request: Request,
    assignment: OfflineAssignment,
    provider: TieredProvider,
) -> PerRequestRecord:
    ttft_ms = provider.true_p50_ms(float(request.timestamp))
    cost_usd = provider.marginal_cost_for_request(request, float(request.timestamp))
    metadata = {
        "offline_cost_baseline": True,
        "offline_oracle_kind": assignment.oracle_kind.value,
        "offline_avoided_api_cost_usd": assignment.avoided_api_cost_usd,
    }
    if assignment.oracle_kind in {
        OfflineOracleKind.STAGE_QC_EXACT,
        OfflineOracleKind.STAGE_QC_EXACT_SLICE,
    }:
        metadata.update(
            {
                "offline_milp_solver": _milp_solver_name(),
                "offline_milp_seed": _milp_seed(),
                "offline_milp_time_limit_sec": _milp_time_limit_sec(),
                "offline_joint_max_requests": _joint_exact_max_requests(),
            }
        )
    return PerRequestRecord(
        request_id=str(request.id),
        elapsed_sec=float(request.timestamp),
        policy=OFFLINE_POLICY,
        prompt_tokens=int(request.request_tokens),
        completion_tokens_budget=(
            request.estimated_response_tokens
            if request.estimated_response_tokens is not None
            else request.response_tokens
        ),
        completion_tokens_actual=request.response_tokens,
        primary_provider=provider.name,
        primary_tier=provider.tier.value,
        final_provider=provider.name,
        final_tier=provider.tier.value,
        ttft_ms=ttft_ms,
        primary_local_ttft_ms=ttft_ms,
        slo_ms=scenario.primary_slo_ms,
        slo_violated=ttft_ms > scenario.primary_slo_ms,
        total_cost_usd=cost_usd,
        primary_cost_usd=cost_usd,
        status=Status.SUCCESS,
        metadata=metadata,
    )


def _cheapest_api_provider(providers: list[TieredProvider]) -> TieredProvider | None:
    api_providers = [provider for provider in providers if provider.tier == ProviderTier.S_A]
    if not api_providers:
        return None
    return min(
        api_providers,
        key=lambda provider: (
            provider.effective_input_cost_per_token,
            provider.effective_output_cost_per_token,
            provider.name,
        ),
    )


def _api_cost(provider: TieredProvider, request: Request) -> float:
    return provider.marginal_cost_for_request(request, float(request.timestamp))


def _offline_service_time_sec(provider: TieredProvider, request: Request) -> float:
    """Return deterministic provider-model service time for the offline oracle."""
    ttft_ms = provider.true_p50_ms(float(request.timestamp))
    tps = max(provider.tps_dist.p50(), 1.0)
    response_tokens = request.response_tokens or 1
    return (ttft_ms + (response_tokens / tps) * 1000.0) / 1000.0


__all__ = [
    "OFFLINE_POLICY",
    "OfflineAssignment",
    "OfflineOracleKind",
    "assign_offline",
    "run_offline_oracle_policy",
]
