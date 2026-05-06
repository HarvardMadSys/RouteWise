"""Shared helpers for section-based simulator experiments."""

from __future__ import annotations

import csv
import inspect
import json
import math
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rwsim.engine.simulator import Simulator
from rwsim.metrics import RunAggregate
from rwsim.metrics.histogram import merge_histograms
from rwsim.policies import build_policy
from rwsim.schemas import Request
from rwsim.world.capacity import ConcurrencyState, ProviderTier, QuotaState
from rwsim.world.distributions import LogNormal, Normal, Uniform
from rwsim.world.providers import TieredProvider

if TYPE_CHECKING:
    from collections import Counter
    from collections.abc import Callable, Mapping

    from rwsim.metrics import Run
    from rwsim.world.scenarios import ScenarioConfig

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "simulation"

P_SWEEP = (0.0, 0.25, 0.50, 0.75, 1.0)
COST_RATIO_PER_MILLION = (1.0, 2.0, 4.0)
OUTPUT_COST_MULTIPLIER = 5.0
COST_LAYER_P50_MS = 300.0
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_WORKLOAD = "sharegpt_burstgpt"
DEFAULT_TPS_P50 = 150.0
_WORKLOAD_CACHE_VERSION = 1

_WORKLOAD_PATHS = {
    "sharegpt_burstgpt": DATA_DIR / "burstgpt_30d.jsonl",
    "burstgpt": DATA_DIR / "burstgpt_30d.jsonl",
}


@dataclass(frozen=True)
class SectionCell:
    """One independent section simulation cell."""

    scenario_name: str
    policy: str
    seed: int


@dataclass(frozen=True)
class SectionCellResult:
    """Result for one independent section simulation cell."""

    scenario_name: str
    policy: str
    seed: int
    run: Run


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
    output_cost_per_million_tokens: float | None = None,
    latency_family: str,
    p50_ms: float = COST_LAYER_P50_MS,
) -> TieredProvider:
    """Build an on-demand API provider."""
    output_price = (
        cost_per_million_tokens * OUTPUT_COST_MULTIPLIER
        if output_cost_per_million_tokens is None
        else output_cost_per_million_tokens
    )
    return TieredProvider(
        name=name,
        cost_per_token=cost_per_million_tokens / 1_000_000.0,
        input_cost_per_token=cost_per_million_tokens / 1_000_000.0,
        output_cost_per_token=output_price / 1_000_000.0,
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
        input_cost_per_token=0.0,
        output_cost_per_token=0.0,
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
        input_cost_per_token=0.0,
        output_cost_per_token=0.0,
        ttft_dist=make_ttft_distribution(latency_family, p50_ms),
        tps_dist=make_tps_distribution(),
        tier=ProviderTier.S_C,
        concurrency=ConcurrencyState(limit=concurrency_limit),
    )


@cache
def _load_cached_workload(dataset: str) -> tuple[Request, ...]:
    path = _workload_path(dataset)
    cache_path, manifest_path = _workload_cache_paths(path)
    if _workload_cache_is_valid(path, cache_path, manifest_path):
        try:
            with cache_path.open("rb") as handle:
                return tuple(pickle.load(handle))
        except (
            AttributeError,
            EOFError,
            ImportError,
            ValueError,
            pickle.UnpicklingError,
        ):
            pass
    return _build_workload_cache(path, cache_path, manifest_path)


def ensure_workload_cache(dataset: str = DEFAULT_WORKLOAD) -> Path:
    """Build the compact simulator workload cache if it is missing or stale."""
    path = _workload_path(dataset)
    cache_path, manifest_path = _workload_cache_paths(path)
    if not _workload_cache_is_valid(path, cache_path, manifest_path):
        _build_workload_cache(path, cache_path, manifest_path)
    return cache_path


def _workload_cache_paths(path: Path) -> tuple[Path, Path]:
    resolved = path.resolve()
    cache_path = resolved.with_name(f"{resolved.name}.simcache.pkl")
    manifest_path = resolved.with_name(f"{resolved.name}.simcache.manifest.json")
    return cache_path, manifest_path


def _workload_cache_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "cache_version": _WORKLOAD_CACHE_VERSION,
        "source_path": str(path),
        "source_resolved": str(resolved),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }


def _workload_cache_is_valid(path: Path, cache_path: Path, manifest_path: Path) -> bool:
    if not cache_path.exists() or not manifest_path.exists():
        return False
    try:
        with manifest_path.open() as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    fingerprint = _workload_cache_fingerprint(path)
    return all(manifest.get(key) == value for key, value in fingerprint.items())


def _build_workload_cache(
    path: Path,
    cache_path: Path,
    manifest_path: Path,
) -> tuple[Request, ...]:
    requests = tuple(_read_jsonl_workload(path))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(f".pkl.tmp.{multiprocessing.current_process().pid}")
    with tmp_path.open("wb") as handle:
        pickle.dump(requests, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(cache_path)

    manifest = _workload_cache_fingerprint(path)
    manifest["n_requests"] = len(requests)
    manifest["cache_size"] = cache_path.stat().st_size
    tmp_manifest = manifest_path.with_suffix(
        f".json.tmp.{multiprocessing.current_process().pid}"
    )
    with tmp_manifest.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    tmp_manifest.replace(manifest_path)
    _load_cached_workload.cache_clear()
    return requests


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
        for line in handle:
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
                    id=len(requests),
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
    """Load and optionally truncate the canonical trace workload.

    Full-trace loads use a compact pickle cache and return shared immutable
    ``Request`` objects from that cache. Callers must treat the returned
    requests and their metadata as read-only. Request IDs are dense simulator
    request indexes, not raw JSONL line numbers.
    """
    path = _workload_path(dataset)
    cache_path, manifest_path = _workload_cache_paths(path)
    if (
        duration_sec is not None or max_requests is not None
    ) and not _workload_cache_is_valid(path, cache_path, manifest_path):
        return _read_jsonl_workload(
            path,
            duration_sec=duration_sec,
            max_requests=max_requests,
        )

    selected = _select_cached_requests(
        _load_cached_workload(dataset),
        duration_sec=duration_sec,
        max_requests=max_requests,
    )
    return list(selected)


def _select_cached_requests(
    requests: tuple[Request, ...],
    *,
    duration_sec: float | None,
    max_requests: int | None,
) -> tuple[Request, ...]:
    stop = len(requests)
    if duration_sec is not None:
        for index, request in enumerate(requests):
            if request.timestamp > duration_sec:
                stop = index
                break
    if max_requests is not None:
        stop = min(stop, max_requests)
    return requests[:stop]


def run_policy(
    scenario: ScenarioConfig,
    requests: list[Request],
    policy_name: str,
    *,
    presets: dict[str, dict[str, Any]],
    seed: int,
    retain_records: bool = True,
) -> Run:
    """Run a section-local policy preset on one request stream."""
    policy = build_policy(policy_name, presets=presets, seed=seed)
    simulator = Simulator(scenario=scenario, seed=seed, retain_records=retain_records)
    return simulator.run(requests, policy, policy_name=policy_name)


def summarize_runs(
    *,
    scenario: ScenarioConfig,
    policy: str,
    seeds: tuple[int, ...],
    runs: list[Run],
) -> dict[str, Any]:
    """Aggregate section metrics across seeds."""
    aggregate = _merge_run_aggregates(runs)
    total = aggregate.n
    ttft_histogram = aggregate.ttft_histogram

    def percentile(pct: float) -> float:
        if total == 0:
            return float("nan")
        return ttft_histogram.quantile(pct / 100.0)

    return {
        "scenario": scenario.name,
        "policy": policy,
        "seeds": list(seeds),
        "n_requests": total,
        "mean_ttft_ms": ttft_histogram.mean(),
        "p10_ms": percentile(10),
        "p25_ms": percentile(25),
        "p50_ms": percentile(50),
        "p75_ms": percentile(75),
        "p90_ms": percentile(90),
        "p99_ms": percentile(99),
        "mean_cost_usd": (
            aggregate.total_cost_usd / aggregate.cost_count
            if aggregate.cost_count
            else float("nan")
        ),
        "total_cost_usd": float(aggregate.total_cost_usd),
        "slo_violation_rate": (
            aggregate.slo_violated_count / total if total else 0.0
        ),
        "hedge_rate": (
            aggregate.hedge_triggered_count / aggregate.hedge_total_count
            if aggregate.hedge_total_count
            else 0.0
        ),
        "provider_mix": _fraction_map(aggregate.provider_counts, total),
        "tier_mix": _fraction_map(aggregate.tier_counts, total),
        "percentile_source": "histogram",
        "histogram_bins": int(ttft_histogram.bin_edges_ms.size - 1),
    }


def run_section(
    *,
    section_name: str,
    scenarios: dict[str, ScenarioConfig],
    policies: tuple[str, ...],
    presets: dict[str, dict[str, Any]],
    seeds: tuple[int, ...],
    section_runners: Mapping[str, Callable[[ScenarioConfig, list[Request], int], Run]] | None = None,
    parallel_cell_runner: Callable[
        [SectionCell, dict[str, dict[str, Any]], str, float | None, int | None, bool],
        SectionCellResult,
    ]
    | None = None,
    workload_dataset: str = DEFAULT_WORKLOAD,
    duration_sec: float | None = None,
    max_requests: int | None = None,
    output_dir: Path | None = None,
    retain_records: bool = False,
    jobs: int = 1,
) -> list[dict[str, Any]]:
    """Run one section and write machine-readable summaries."""
    if jobs < 1:
        raise ValueError(f"jobs must be >= 1, got {jobs}")
    root = output_dir or (OUTPUT_DIR / section_name.replace("-", "_"))
    root.mkdir(parents=True, exist_ok=True)
    cells = [
        SectionCell(scenario.name, policy, seed)
        for scenario in scenarios.values()
        for policy in policies
        for seed in seeds
    ]
    if jobs == 1:
        results, loaded_requests = _run_section_serial(
            scenarios=scenarios,
            cells=cells,
            presets=presets,
            section_runners=section_runners or {},
            workload_dataset=workload_dataset,
            duration_sec=duration_sec,
            max_requests=max_requests,
            retain_records=retain_records,
        )
        execution_mode = "serial"
    else:
        if parallel_cell_runner is None:
            raise ValueError(
                "parallel run_section requires a section-local parallel_cell_runner"
            )
        ensure_workload_cache(workload_dataset)
        results = _run_section_parallel(
            cells=cells,
            presets=presets,
            workload_dataset=workload_dataset,
            duration_sec=duration_sec,
            max_requests=max_requests,
            retain_records=retain_records,
            jobs=jobs,
            parallel_cell_runner=parallel_cell_runner,
        )
        loaded_requests = _processed_request_count(results)
        execution_mode = "parallel"

    rows = _write_section_outputs(
        root=root,
        section_name=section_name,
        scenarios=scenarios,
        policies=policies,
        seeds=seeds,
        results=results,
        workload_dataset=workload_dataset,
        duration_sec=duration_sec,
        max_requests=max_requests,
        loaded_requests=loaded_requests,
        jobs=jobs,
        execution_mode=execution_mode,
    )
    return rows


def _run_section_serial(
    *,
    scenarios: dict[str, ScenarioConfig],
    cells: list[SectionCell],
    presets: dict[str, dict[str, Any]],
    section_runners: Mapping[str, Callable[[ScenarioConfig, list[Request], int], Run]],
    workload_dataset: str,
    duration_sec: float | None,
    max_requests: int | None,
    retain_records: bool,
) -> tuple[list[SectionCellResult], int]:
    requests = load_workload(
        dataset=workload_dataset,
        duration_sec=duration_sec,
        max_requests=max_requests,
    )
    results: list[SectionCellResult] = []
    for cell in cells:
        scenario = scenarios[cell.scenario_name]
        if cell.policy in section_runners:
            run = section_runners[cell.policy](
                scenario,
                requests,
                cell.seed,
                retain_records=retain_records,
            )
        else:
            run = run_policy(
                scenario,
                requests,
                cell.policy,
                presets=presets,
                seed=cell.seed,
                retain_records=retain_records,
            )
        results.append(
            SectionCellResult(
                scenario_name=cell.scenario_name,
                policy=cell.policy,
                seed=cell.seed,
                run=run,
            )
        )
    return results, len(requests)


def _run_section_parallel(
    *,
    cells: list[SectionCell],
    presets: dict[str, dict[str, Any]],
    workload_dataset: str,
    duration_sec: float | None,
    max_requests: int | None,
    retain_records: bool,
    jobs: int,
    parallel_cell_runner: Callable[
        [SectionCell, dict[str, dict[str, Any]], str, float | None, int | None, bool],
        SectionCellResult,
    ],
) -> list[SectionCellResult]:
    if not cells:
        return []
    context = multiprocessing.get_context("spawn")
    max_workers = min(jobs, len(cells))
    results: list[SectionCellResult] = []
    executor_kwargs: dict[str, Any] = {
        "max_workers": max_workers,
        "mp_context": context,
    }
    if "max_tasks_per_child" in inspect.signature(ProcessPoolExecutor).parameters:
        executor_kwargs["max_tasks_per_child"] = 10
    with ProcessPoolExecutor(**executor_kwargs) as executor:
        futures = [
            executor.submit(
                parallel_cell_runner,
                cell,
                presets,
                workload_dataset,
                duration_sec,
                max_requests,
                retain_records,
            )
            for cell in cells
        ]
        for future in as_completed(futures):
            results.append(future.result())
    order = {cell: index for index, cell in enumerate(cells)}
    return sorted(
        results,
        key=lambda result: order[
            SectionCell(
                scenario_name=result.scenario_name,
                policy=result.policy,
                seed=result.seed,
            )
        ],
    )


def _write_section_outputs(
    *,
    root: Path,
    section_name: str,
    scenarios: dict[str, ScenarioConfig],
    policies: tuple[str, ...],
    seeds: tuple[int, ...],
    results: list[SectionCellResult],
    workload_dataset: str,
    duration_sec: float | None,
    max_requests: int | None,
    loaded_requests: int,
    jobs: int,
    execution_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    histogram_rows: list[dict[str, Any]] = []
    per_seed_histogram_rows: list[dict[str, Any]] = []
    by_group: dict[tuple[str, str], list[Run]] = {
        (scenario.name, policy): [] for scenario in scenarios.values() for policy in policies
    }
    by_seed: dict[tuple[str, str, int], Run] = {}
    for result in results:
        by_group[(result.scenario_name, result.policy)].append(result.run)
        by_seed[(result.scenario_name, result.policy, result.seed)] = result.run

    for scenario in scenarios.values():
        for policy in policies:
            runs = by_group[(scenario.name, policy)]
            aggregate = _merge_run_aggregates(runs)
            rows.append(
                summarize_runs(
                    scenario=scenario,
                    policy=policy,
                    seeds=seeds,
                    runs=runs,
                )
            )
            histogram_rows.append(
                {
                    "scenario": scenario.name,
                    "policy": policy,
                    "seeds": list(seeds),
                    "histogram": aggregate.ttft_histogram.to_dict(),
                }
            )
            for seed in seeds:
                try:
                    run = by_seed[(scenario.name, policy, seed)]
                except KeyError as exc:
                    raise ValueError(
                        "missing section cell result for "
                        f"scenario={scenario.name!r}, policy={policy!r}, seed={seed}"
                    ) from exc
                if run.aggregate is None:
                    raise ValueError(
                        "run_section requires every run to carry a streaming "
                        f"aggregate; missing for scenario={scenario.name!r}, "
                        f"policy={policy!r}, seed={seed}"
                    )
                per_seed_histogram_rows.append(
                    {
                        "scenario": scenario.name,
                        "policy": policy,
                        "seed": seed,
                        "histogram": run.aggregate.ttft_histogram.to_dict(),
                    }
                )

    metadata = {
        "section": section_name,
        "scenarios": list(scenarios),
        "policies": list(policies),
        "seeds": list(seeds),
        "workload_dataset": workload_dataset,
        "duration_sec": duration_sec,
        "max_requests": max_requests,
        "loaded_requests": loaded_requests,
        "processed_requests_per_cell": loaded_requests,
        "jobs": jobs,
        "execution_mode": execution_mode,
    }
    write_json(root / "metadata.json", metadata)
    write_json(root / "summary.json", rows)
    write_json(root / "ttft_histograms.json", histogram_rows)
    write_json(root / "ttft_histograms_by_seed.json", per_seed_histogram_rows)
    write_summary_csv(root / "summary.csv", rows)
    return rows


def _processed_request_count(results: list[SectionCellResult]) -> int:
    if not results:
        return 0
    run = results[0].run
    if run.aggregate is not None:
        return run.aggregate.n
    return len(run.records)


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
        "percentile_source",
        "histogram_bins",
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


def _merge_run_aggregates(runs: list[Run]) -> RunAggregate:
    for run in runs:
        if run.aggregate is None:
            raise ValueError(
                "run_section requires every run to carry a streaming aggregate"
            )
    aggregate = RunAggregate(
        ttft_histogram=merge_histograms([run.ttft_histogram() for run in runs])
    )
    for run in runs:
        source = run.aggregate
        aggregate.n += source.n
        if source.e2e_histogram is not None:
            if aggregate.e2e_histogram is None:
                aggregate.e2e_histogram = source.e2e_histogram.copy()
            else:
                aggregate.e2e_histogram = aggregate.e2e_histogram.merge(
                    source.e2e_histogram
                )
        aggregate.total_cost_usd += source.total_cost_usd
        aggregate.cost_count += source.cost_count
        aggregate.status_counts.update(source.status_counts)
        aggregate.slo_violated_count += source.slo_violated_count
        aggregate.cost_by_tier.update(source.cost_by_tier)
        aggregate.cost_by_provider.update(source.cost_by_provider)
        aggregate.provider_counts.update(source.provider_counts)
        aggregate.tier_counts.update(source.tier_counts)
        aggregate.hedge_triggered_count += source.hedge_triggered_count
        aggregate.hedge_total_count += source.hedge_total_count
        aggregate.hedge_winner_counts.update(source.hedge_winner_counts)
    return aggregate


__all__ = [
    "COST_LAYER_P50_MS",
    "COST_RATIO_PER_MILLION",
    "DEFAULT_SEEDS",
    "DEFAULT_WORKLOAD",
    "OUTPUT_COST_MULTIPLIER",
    "OUTPUT_DIR",
    "P_SWEEP",
    "SectionCell",
    "SectionCellResult",
    "ensure_workload_cache",
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
