"""Shared helpers for section-based simulator experiments."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rwsim.engine.simulator import Simulator
from rwsim.policies import build_policy
from rwsim.schemas import Request
from rwsim.world.capacity import ConcurrencyState, ProviderTier, QuotaState
from rwsim.world.distributions import LogNormal, Normal, Uniform
from rwsim.world.providers import TieredProvider

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from rwsim.metrics import Run
    from rwsim.world.scenarios import ScenarioConfig

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "simulation"

P_SWEEP = (0.0, 0.25, 0.50, 0.75, 1.0)
COST_RATIO_PER_MILLION = (1.0, 2.0, 4.0)
COST_LAYER_P50_MS = 300.0
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_WORKLOAD = "sharegpt_burstgpt"
DEFAULT_TPS_P50 = 150.0

_WORKLOAD_PATHS = {
    "sharegpt_burstgpt": DATA_DIR / "burstgpt_30d.jsonl",
    "burstgpt": DATA_DIR / "burstgpt_30d.jsonl",
}


def p_label(value: float) -> str:
    """Return a stable policy-name suffix for one p-sweep value."""
    pct = round(float(value) * 100)
    if abs(float(value) - pct / 100.0) > 1e-9:
        raise ValueError(f"p values must be percent-like decimals, got {value!r}")
    return f"p{pct}"


def routewise_lp_policy_name(p_value: float) -> str:
    """Return the simulator policy name for LP-only RouteWise at one p value."""
    return f"ablation_lp_only_{p_label(p_value)}"


def routewise_hedging_policy_name(p_value: float) -> str:
    """Return the simulator policy name for LP+hedging RouteWise at one p value."""
    return f"ablation_lp_hedging_{p_label(p_value)}"


def make_routewise_presets(
    *,
    p_values: tuple[float, ...] = P_SWEEP,
    include_hedging: bool = False,
) -> dict[str, dict[str, Any]]:
    """Build section-local policy presets with explorer disabled."""
    presets: dict[str, dict[str, Any]] = {
        "greedy_cost": {"policy": "BaselinePolicy", "params": {"mode": "greedy_cost"}},
        "greedy_latency": {"policy": "BaselinePolicy", "params": {"mode": "greedy_latency"}},
        "random": {"policy": "BaselinePolicy", "params": {"mode": "random"}},
    }
    for value in p_values:
        presets[routewise_lp_policy_name(value)] = {
            "policy": "RouteWisePolicy",
            "params": {
                "hedging": False,
                "explorer": False,
                "p": float(value),
            },
        }
        if include_hedging:
            presets[routewise_hedging_policy_name(value)] = {
                "policy": "RouteWisePolicy",
                "params": {
                    "hedging": "probability_target",
                    "explorer": False,
                    "p": float(value),
                },
            }
    return presets


def make_ttft_distribution(family: str, p50_ms: float):
    """Construct the synthetic latency family used by the current section."""
    if family == "uniform":
        return Uniform(low=0.5 * p50_ms, high=1.5 * p50_ms)
    if family == "normal":
        return Normal(mean_ms=p50_ms, sigma=0.3 * p50_ms)
    if family in {"heavy_tail", "lognormal"}:
        return LogNormal(mu=math.log(p50_ms), sigma=0.5)
    raise ValueError(f"unknown latency family: {family!r}")


def make_tps_distribution() -> LogNormal:
    """Return the default token-throughput distribution for synthetic providers."""
    return LogNormal(mu=math.log(DEFAULT_TPS_P50), sigma=0.3)


def make_api_provider(
    name: str,
    *,
    cost_per_million_tokens: float,
    latency_family: str,
    p50_ms: float = COST_LAYER_P50_MS,
) -> TieredProvider:
    """Build an on-demand API provider."""
    return TieredProvider(
        name=name,
        cost_per_token=cost_per_million_tokens / 1_000_000.0,
        ttft_dist=make_ttft_distribution(latency_family, p50_ms),
        tps_dist=make_tps_distribution(),
        tier=ProviderTier.S_A,
    )


def make_quota_provider(
    name: str,
    *,
    quota_size: int,
    latency_family: str = "heavy_tail",
    p50_ms: float = COST_LAYER_P50_MS,
    quota_window_sec: float = 86400.0,
) -> TieredProvider:
    """Build a subscription/quota provider with zero marginal request cost."""
    return TieredProvider(
        name=name,
        cost_per_token=0.0,
        ttft_dist=make_ttft_distribution(latency_family, p50_ms),
        tps_dist=make_tps_distribution(),
        tier=ProviderTier.S_Q,
        quota=QuotaState(size=quota_size, window_sec=quota_window_sec),
    )


def make_concurrency_provider(
    name: str,
    *,
    concurrency_limit: int,
    latency_family: str = "heavy_tail",
    p50_ms: float = COST_LAYER_P50_MS,
) -> TieredProvider:
    """Build a subscription/concurrency provider with zero marginal request cost."""
    return TieredProvider(
        name=name,
        cost_per_token=0.0,
        ttft_dist=make_ttft_distribution(latency_family, p50_ms),
        tps_dist=make_tps_distribution(),
        tier=ProviderTier.S_C,
        concurrency=ConcurrencyState(limit=concurrency_limit),
    )


@cache
def _load_jsonl_workload(dataset: str) -> tuple[Request, ...]:
    path = _workload_path(dataset)
    return tuple(_read_jsonl_workload(path))


def _workload_path(dataset: str) -> Path:
    try:
        return _WORKLOAD_PATHS[dataset]
    except KeyError as exc:
        known = ", ".join(sorted(_WORKLOAD_PATHS))
        raise ValueError(f"unknown workload {dataset!r}; expected one of: {known}") from exc


def _read_jsonl_workload(
    path: Path,
    *,
    duration_sec: float | None = None,
    max_requests: int | None = None,
) -> list[Request]:
    if not path.exists():
        raise FileNotFoundError(f"workload file not found: {path}")

    requests: list[Request] = []
    first_timestamp: float | None = None
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            timestamp = float(record["arrived_at"])
            if first_timestamp is None:
                first_timestamp = timestamp
            relative_timestamp = timestamp - first_timestamp
            if duration_sec is not None and relative_timestamp > duration_sec:
                break
            request_tokens = int(record["num_prefill_tokens"])
            response_tokens = int(record["num_decode_tokens"])
            total_tokens = request_tokens + response_tokens
            if total_tokens <= 0:
                continue
            metadata = {
                key: record[key]
                for key in (
                    "session_id",
                    "sharegpt_conversation_id",
                    "sharegpt_turn_index",
                    "log_type",
                    "elapsed_time_sec",
                )
                if key in record
            }
            requests.append(
                Request(
                    id=idx,
                    timestamp=relative_timestamp,
                    request_tokens=request_tokens,
                    response_tokens=response_tokens,
                    total_tokens=total_tokens,
                    model=str(record.get("model") or "sharegpt"),
                    metadata=metadata,
                )
            )
            if max_requests is not None and len(requests) >= max_requests:
                break
    return requests


def load_workload(
    *,
    dataset: str = DEFAULT_WORKLOAD,
    duration_sec: float | None = None,
    max_requests: int | None = None,
) -> list[Request]:
    """Load and optionally truncate the canonical trace workload."""
    if duration_sec is None and max_requests is None:
        selected = list(_load_jsonl_workload(dataset))
    else:
        selected = _read_jsonl_workload(
            _workload_path(dataset),
            duration_sec=duration_sec,
            max_requests=max_requests,
        )

    return [
        Request(
            id=idx,
            timestamp=float(item.timestamp),
            request_tokens=item.request_tokens,
            estimated_response_tokens=item.estimated_response_tokens,
            response_tokens=item.response_tokens,
            total_tokens=item.total_tokens,
            model=item.model,
            provider=item.provider,
            actual_cost=item.actual_cost,
            latency_ms=item.latency_ms,
            ttft_ms=item.ttft_ms,
            slo_ms=item.slo_ms,
            metadata=dict(item.metadata),
        )
        for idx, item in enumerate(selected)
    ]


def run_policy(
    scenario: ScenarioConfig,
    requests: list[Request],
    policy_name: str,
    *,
    presets: dict[str, dict[str, Any]],
    seed: int,
) -> Run:
    """Run a section-local policy preset on one request stream."""
    policy = build_policy(policy_name, presets=presets, seed=seed)
    simulator = Simulator(scenario=scenario, seed=seed)
    return simulator.run(requests, policy, policy_name=policy_name)


def summarize_runs(
    *,
    scenario: ScenarioConfig,
    policy: str,
    seeds: tuple[int, ...],
    runs: list[Run],
) -> dict[str, Any]:
    """Aggregate section metrics across seeds."""
    records = [record for run in runs for record in run.records]
    ttft = np.asarray([record.ttft_ms for record in records], dtype=float)
    costs = np.asarray([record.total_cost_usd for record in records], dtype=float)
    provider_counts = Counter(record.final_provider for record in records)
    tier_counts = Counter(record.final_tier for record in records if record.final_tier)
    total = len(records)

    def percentile(pct: float) -> float:
        if ttft.size == 0:
            return float("nan")
        return float(np.percentile(ttft, pct))

    return {
        "scenario": scenario.name,
        "policy": policy,
        "seeds": list(seeds),
        "n_requests": total,
        "mean_ttft_ms": float(np.mean(ttft)) if ttft.size else float("nan"),
        "p10_ms": percentile(10),
        "p25_ms": percentile(25),
        "p50_ms": percentile(50),
        "p75_ms": percentile(75),
        "p90_ms": percentile(90),
        "p99_ms": percentile(99),
        "mean_cost_usd": float(np.mean(costs)) if costs.size else float("nan"),
        "total_cost_usd": float(np.sum(costs)) if costs.size else 0.0,
        "slo_violation_rate": (
            float(np.mean([record.slo_violated for record in records])) if records else 0.0
        ),
        "hedge_rate": (
            float(np.mean([record.hedge_triggered for record in records])) if records else 0.0
        ),
        "provider_mix": _fraction_map(provider_counts, total),
        "tier_mix": _fraction_map(tier_counts, total),
    }


def run_section(
    *,
    section_name: str,
    scenarios: dict[str, ScenarioConfig],
    policies: tuple[str, ...],
    presets: dict[str, dict[str, Any]],
    seeds: tuple[int, ...],
    section_runners: Mapping[str, Callable[[ScenarioConfig, list[Request], int], Run]] | None = None,
    workload_dataset: str = DEFAULT_WORKLOAD,
    duration_sec: float | None = None,
    max_requests: int | None = None,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Run one section and write machine-readable summaries."""
    root = output_dir or (OUTPUT_DIR / section_name.replace("-", "_"))
    root.mkdir(parents=True, exist_ok=True)
    requests = load_workload(
        dataset=workload_dataset,
        duration_sec=duration_sec,
        max_requests=max_requests,
    )
    rows: list[dict[str, Any]] = []
    local_runners = section_runners or {}
    for scenario in scenarios.values():
        for policy in policies:
            runs = [
                (
                    local_runners[policy](scenario, requests, seed)
                    if policy in local_runners
                    else run_policy(
                        scenario,
                        requests,
                        policy,
                        presets=presets,
                        seed=seed,
                    )
                )
                for seed in seeds
            ]
            rows.append(
                summarize_runs(
                    scenario=scenario,
                    policy=policy,
                    seeds=seeds,
                    runs=runs,
                )
            )

    metadata = {
        "section": section_name,
        "scenarios": list(scenarios),
        "policies": list(policies),
        "seeds": list(seeds),
        "workload_dataset": workload_dataset,
        "duration_sec": duration_sec,
        "max_requests": max_requests,
        "loaded_requests": len(requests),
    }
    write_json(root / "metadata.json", metadata)
    write_json(root / "summary.json", rows)
    write_summary_csv(root / "summary.csv", rows)
    return rows


def write_json(path: Path, payload: Any) -> None:
    """Write stable JSON output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write section summary rows with nested maps JSON-encoded."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "policy",
        "seeds",
        "n_requests",
        "mean_ttft_ms",
        "p10_ms",
        "p25_ms",
        "p50_ms",
        "p75_ms",
        "p90_ms",
        "p99_ms",
        "mean_cost_usd",
        "total_cost_usd",
        "slo_violation_rate",
        "hedge_rate",
        "provider_mix",
        "tier_mix",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(row[key], sort_keys=True)
                        if isinstance(row.get(key), (dict, list))
                        else row.get(key)
                    )
                    for key in fieldnames
                }
            )


def _fraction_map(counts: Counter[str], total: int) -> dict[str, float]:
    if total <= 0:
        return {}
    return {key: counts[key] / total for key in sorted(counts)}


__all__ = [
    "COST_LAYER_P50_MS",
    "COST_RATIO_PER_MILLION",
    "DEFAULT_SEEDS",
    "DEFAULT_WORKLOAD",
    "OUTPUT_DIR",
    "P_SWEEP",
    "load_workload",
    "make_api_provider",
    "make_concurrency_provider",
    "make_quota_provider",
    "make_routewise_presets",
    "make_ttft_distribution",
    "p_label",
    "routewise_hedging_policy_name",
    "routewise_lp_policy_name",
    "run_policy",
    "run_section",
    "summarize_runs",
    "write_json",
    "write_summary_csv",
]
