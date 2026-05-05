"""Policy evaluation helpers for the simulator experiment."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from experiments.simulation import list_scenarios, load_world_scenario
from experiments.simulation.simple_scenarios import make_simple_scenarios
from rwsim.data import DataLoader
from rwsim.runner import POLICIES, run_policy
from rwsim.schemas import Request

if TYPE_CHECKING:
    from rwsim.metrics import Run
    from rwsim.world.scenarios import ScenarioConfig


MAIN_VARIANTS = tuple(POLICIES)
BACKUP_EXPLORATION_VARIANTS: tuple[str, ...] = ()
HEDGE_ABLATION_VARIANTS: tuple[str, ...] = ()
PROVIDER_PERCENTILE_ABLATION_VARIANTS: tuple[str, ...] = ()
CONTROL_VARIANTS = ("greedy_cost", "greedy_latency", "random")
TRACE_WORKLOAD_DATASETS = ("burstgpt", "freeinference", "rednote", "sharegpt")
BACKUP_SCOPES = ("any_provider", "cross_tier")

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_TRACE_DATA_ROOT = _WORKSPACE_ROOT / "data"
_DATASET_CACHE_ROOT = _WORKSPACE_ROOT / "outputs" / "cache" / "dataset"
_DATA_LOADER_CONFIG = {"dataset": {}}

_TRACE_DATASET_PATHS = {
    "freeinference": [
        _TRACE_DATA_ROOT / "freeinference.csv",
        _TRACE_DATA_ROOT / "freeinference_logs.csv",
    ],
    "rednote": [
        _TRACE_DATA_ROOT / "enterprise.csv",
        _TRACE_DATA_ROOT / "rednote_logs.csv",
    ],
    "burstgpt": [
        _TRACE_DATA_ROOT / "burstgpt_30d.jsonl",
    ],
    "sharegpt": [
        _TRACE_DATA_ROOT / "sharegpt_prompts_7d.jsonl",
        _TRACE_DATA_ROOT / "sharegpt_burstgpt" / "sharegpt_prompts_7d.jsonl",
        _TRACE_DATA_ROOT / "sharegpt_burstgpt" / "converted.csv",
    ],
}


@dataclass
class RunDiagnostics:
    """Small diagnostics summary for one policy run."""

    variant: str
    scenario_name: str
    seed: int
    total_decisions: int = 0
    backup_selection_counts: Counter[str] = field(default_factory=Counter)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "total_decisions": self.total_decisions,
            "backup_selection_counts": dict(sorted(self.backup_selection_counts.items())),
        }


@dataclass
class EvaluatedRun:
    """One seed of one policy on one scenario."""

    run: Run
    diagnostics: RunDiagnostics


def canonicalize_variant_name(variant: str) -> str:
    """Policy names are already canonical."""
    return variant


def _body_latency_proxy_ms(provider, profile=None, now: float = 0.0) -> tuple[float, str]:
    """Return the latency objective used by the RouteWise LP body.

    This compatibility helper keeps eval-grid tests pointed at the same
    policy-facing quantity while the implementation lives in
    ``rwsim.policies.routewise``.
    """
    del profile
    return float(provider.true_mean_ms(now)), "provider_mean"


def _resolve_trace_dataset_path(dataset_name: str) -> Path | None:
    """Return the first available canonical path for one trace dataset."""
    for candidate in _TRACE_DATASET_PATHS.get(dataset_name, []):
        if candidate.exists():
            return candidate
    return None


def _dataset_cache_path(dataset_name: str) -> Path:
    return _DATASET_CACHE_ROOT / f"{dataset_name}.pkl"


def _load_sharegpt_jsonl_requests(filepath: Path) -> list[Request]:
    """Load ShareGPT-style JSONL trace records into Request objects."""
    requests: list[Request] = []
    first_timestamp: float | None = None
    with filepath.open() as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            timestamp = float(record["arrived_at"])
            if first_timestamp is None:
                first_timestamp = timestamp
            request_tokens = int(record["num_prefill_tokens"])
            response_tokens = int(record["num_decode_tokens"])
            total_tokens = request_tokens + response_tokens
            if total_tokens <= 0:
                continue
            metadata = {
                key: record[key]
                for key in (
                    "session_id",
                    "prompt_text",
                    "response_text",
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
                    timestamp=float(timestamp - first_timestamp),
                    request_tokens=request_tokens,
                    response_tokens=response_tokens,
                    total_tokens=total_tokens,
                    model=str(record.get("model") or "sharegpt"),
                    metadata=metadata,
                )
            )
    return requests


@cache
def _load_trace_dataset_requests(dataset_name: str) -> tuple[Request, ...]:
    """Load one real workload dataset from cache or raw trace."""
    if dataset_name not in TRACE_WORKLOAD_DATASETS:
        raise ValueError(
            f"Unknown trace-driven dataset: {dataset_name!r}. "
            f"Expected one of {TRACE_WORKLOAD_DATASETS!r}."
        )

    from experiments.simulation.dataset_cache import (
        CacheStalenessError,
        build_cache,
        load_cached,
        verify_cache,
    )

    cache_path = _dataset_cache_path(dataset_name)
    if cache_path.exists():
        try:
            verify_cache(dataset_name, quick=True)
            return load_cached(dataset_name)
        except CacheStalenessError:
            pass

    source_path = _resolve_trace_dataset_path(dataset_name)
    if source_path is not None:
        build_cache(dataset_name, force=True)
        return load_cached(dataset_name)

    raise FileNotFoundError(
        "Could not locate trace data for dataset "
        f"{dataset_name!r}. Checked {list(_TRACE_DATASET_PATHS.get(dataset_name, []))} "
        f"and cache path {cache_path}."
    )


def _generate_trace_driven_workload(
    scenario: ScenarioConfig,
    *,
    dataset_name: str,
    seed: int,
) -> list[Request]:
    """Replay the entire trace at its natural arrival rate."""
    del scenario, seed
    full_trace = _load_trace_dataset_requests(dataset_name)
    if not full_trace:
        return []

    base = float(full_trace[0].timestamp)
    return [
        Request(
            id=idx,
            timestamp=float(float(req.timestamp) - base),
            request_tokens=req.request_tokens,
            response_tokens=req.response_tokens,
            total_tokens=req.total_tokens,
            model=req.model,
            provider=req.provider,
            actual_cost=req.actual_cost,
            latency_ms=req.latency_ms,
            ttft_ms=req.ttft_ms,
            metadata=req.metadata,
        )
        for idx, req in enumerate(full_trace)
    ]


def generate_scenario_workload(
    scenario: ScenarioConfig,
    *,
    seed: int = 0,
    dataset_name: str,
) -> list[Request]:
    """Load the shared real-trace workload for one scenario."""
    return _generate_trace_driven_workload(
        scenario,
        dataset_name=dataset_name,
        seed=seed,
    )


def run_variant(
    scenario: ScenarioConfig,
    requests: list[Request],
    variant: str,
    *,
    seed: int,
    backup_scope: str = "any_provider",
) -> EvaluatedRun:
    """Run one policy preset on one scenario for one seed."""
    del backup_scope
    variant = canonicalize_variant_name(variant)
    if variant not in MAIN_VARIANTS:
        raise ValueError(f"Unknown policy variant: {variant!r}")
    run = run_policy(scenario, requests, variant, seed=seed)
    diagnostics = RunDiagnostics(
        variant=variant,
        scenario_name=scenario.name,
        seed=seed,
        total_decisions=len(requests),
    )
    return EvaluatedRun(run=run, diagnostics=diagnostics)


def build_all_scenarios() -> dict[str, ScenarioConfig]:
    """Return every registered scenario available to the simulator."""
    scenarios = {name: load_world_scenario(name) for name in list_scenarios()}
    scenarios.update(make_simple_scenarios())
    return scenarios


def summarize_main_metrics(
    scenario: ScenarioConfig,
    evaluated_runs: list[EvaluatedRun],
) -> dict[str, object]:
    """Aggregate user-facing metrics across seeds for one policy."""
    runs = [item.run for item in evaluated_runs]
    if not runs:
        return {}
    mean_cost_usd = float(np.mean([run.mean_cost_usd() for run in runs]))
    provider_fractions = _mean_provider_fraction(runs)
    tier_fractions = _mean_tier_fraction(runs)
    summary = {
        "scenario": scenario.name,
        "seeds": [item.diagnostics.seed for item in evaluated_runs],
        "mean_ttft_ms": float(np.mean([run.mean_ttft_ms() for run in runs])),
        "mean_cost_usd": mean_cost_usd,
        "p50_ms": float(np.mean([run.p50_ms() for run in runs])),
        "p90_ms": float(np.mean([run.p90_ms() for run in runs])),
        "p99_ms": float(np.mean([run.p99_ms() for run in runs])),
        "slo_violation_rate": float(
            np.mean([run.slo_violation_rate(scenario.primary_slo_ms) for run in runs])
        ),
        "hedge_rate": float(np.mean([run.hedge_rate() for run in runs])),
        "provider_fractions": provider_fractions,
        "tier_fractions": tier_fractions,
    }
    # Temporary output-schema aliases for experiment-layer CSV writers. The
    # canonical names above are policy-facing; old runner-local plot helpers use
    # these display names.
    summary["avg_cost_usd"] = mean_cost_usd
    summary["provider_mix"] = provider_fractions
    summary["tier_mix"] = tier_fractions
    return summary


def summarize_diagnostics(evaluated_runs: list[EvaluatedRun]) -> dict[str, object]:
    """Aggregate diagnostics across seeds."""
    seed_summaries = {
        str(item.diagnostics.seed): item.diagnostics.to_summary_dict() for item in evaluated_runs
    }
    return {
        "seeds": seed_summaries,
        "mean_B_tau": 0.0,
        "mean_E_pi_c_eff": 0.0,
        "mean_budget_utilization": 0.0,
        "mean_budget_slack": 0.0,
        "budget_utilization_p10": 0.0,
        "budget_utilization_p50": 0.0,
        "budget_utilization_p90": 0.0,
        "budget_slack_p10": 0.0,
        "budget_slack_p50": 0.0,
        "budget_slack_p90": 0.0,
        "solver_status_counts": {},
        "fallback_counts": {},
        "non_optimal_decisions": 0,
        "single_feasible_provider_decisions": 0,
        "trivial_single_provider_outcomes": 0,
        "true_p50_fallback_count": 0,
        "explorer_feedback_count": 0,
        "backup_selection_counts": {},
        "mean_backup_random_prob": 0.0,
        "total_decisions": int(sum(item.diagnostics.total_decisions for item in evaluated_runs)),
    }


def build_hedge_delta(
    no_hedge: dict[str, object],
    hedge: dict[str, object],
) -> dict[str, object]:
    """Return simple hedge-vs-no-hedge deltas for CSV builders."""
    return {
        "delta_p99_ms": float(hedge.get("p99_ms", 0.0)) - float(no_hedge.get("p99_ms", 0.0)),
        "delta_cost_usd": float(hedge.get("mean_cost_usd", 0.0))
        - float(no_hedge.get("mean_cost_usd", 0.0)),
        "delta_slo_violation_rate": float(hedge.get("slo_violation_rate", 0.0))
        - float(no_hedge.get("slo_violation_rate", 0.0)),
    }


def _mean_provider_fraction(runs: list[Run]) -> dict[str, float]:
    provider_names = sorted(
        {provider_name for run in runs for provider_name in run.provider_fractions()}
    )
    return {
        name: float(np.mean([run.provider_fractions().get(name, 0.0) for run in runs]))
        for name in provider_names
    }


def _mean_tier_fraction(runs: list[Run]) -> dict[str, float]:
    tier_names = sorted({tier_name for run in runs for tier_name in run.tier_fractions()})
    return {
        name: float(np.mean([run.tier_fractions().get(name, 0.0) for run in runs]))
        for name in tier_names
    }


def load_trace_file(path: Path) -> list[Request]:
    """Load one trace file directly."""
    if path.suffix == ".jsonl":
        return _load_sharegpt_jsonl_requests(path)
    return DataLoader(_DATA_LOADER_CONFIG).load(path)


__all__ = [
    "BACKUP_EXPLORATION_VARIANTS",
    "BACKUP_SCOPES",
    "CONTROL_VARIANTS",
    "HEDGE_ABLATION_VARIANTS",
    "MAIN_VARIANTS",
    "PROVIDER_PERCENTILE_ABLATION_VARIANTS",
    "TRACE_WORKLOAD_DATASETS",
    "_DATASET_CACHE_ROOT",
    "_DATA_LOADER_CONFIG",
    "_TRACE_DATASET_PATHS",
    "EvaluatedRun",
    "RunDiagnostics",
    "_body_latency_proxy_ms",
    "_dataset_cache_path",
    "_load_sharegpt_jsonl_requests",
    "_resolve_trace_dataset_path",
    "build_all_scenarios",
    "build_hedge_delta",
    "canonicalize_variant_name",
    "generate_scenario_workload",
    "load_trace_file",
    "run_variant",
    "summarize_diagnostics",
    "summarize_main_metrics",
]
