"""Capture and compare golden baselines for simulator refactors.

This script is the Phase 0 baseline tool from the simulator merge plan.
It captures normalized JSON artifacts for all supported scenario families
and can later re-run the same families and compare them against the stored
golden baselines.

The implementation intentionally runs each family in an isolated subprocess.
`routewise-simulator/` and `routewise-simulator-joint/` both expose an
`experiment` package; subprocess isolation avoids import-path contamination
between the worktrees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
MAIN_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = MAIN_ROOT.parent
JOINT_ROOT = WORKSPACE_ROOT / "routewise-simulator-joint"
JOINT_LP_BUDGET_ROOT = WORKSPACE_ROOT / "routewise-simulator-joint-wt-lp-budget"
DEFAULT_GOLDEN_ROOT = MAIN_ROOT / "tests" / "golden"
AUDIT_ROOTS = [
    root
    for root in [MAIN_ROOT, JOINT_ROOT, JOINT_LP_BUDGET_ROOT]
    if root.exists()
]

SEEDS = [42, 43, 44]
SANITY_N_REQUESTS = 1000
SANITY_DURATION_SECONDS = 3600.0

FAMILY_SPECS = {
    "latency_synthetic": {
        "repo_root": MAIN_ROOT,
        "output_relpath": Path("latency/synthetic.json"),
    },
    "latency_sanity": {
        "repo_root": MAIN_ROOT,
        "output_relpath": Path("latency/sanity.json"),
    },
    "tiered": {
        "repo_root": MAIN_ROOT,
        "output_relpath": Path("tiered/scenarios.json"),
    },
    "calibrated": {
        "repo_root": MAIN_ROOT,
        "output_relpath": Path("calibrated/mm25.json"),
    },
    "stress": {
        "repo_root": MAIN_ROOT,
        "output_relpath": Path("stress/scenarios.json"),
    },
}


def _sha256_bytes(payload: bytes) -> str:
    """Return the SHA256 hex digest for bytes."""
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(payload: str) -> str:
    """Return the SHA256 hex digest for a UTF-8 string."""
    return _sha256_bytes(payload.encode("utf-8"))


def _sorted_dict(input_map: dict[str, Any]) -> dict[str, Any]:
    """Return a key-sorted shallow copy for stable JSON output."""
    return {key: input_map[key] for key in sorted(input_map)}


def _strategy_hash(strategy_names: list[str]) -> str:
    """Return a stable digest for the ordered strategy list."""
    return _sha256_text("\0".join(strategy_names))


def _float_array_digest(values: Any) -> str:
    """Return a stable digest for a numeric array-like object."""
    arr = np.asarray(values, dtype=np.float64)
    return _sha256_bytes(arr.tobytes())


def _bool_array_digest(values: Any) -> str:
    """Return a stable digest for a boolean array-like object."""
    arr = np.asarray(values, dtype=np.bool_)
    return _sha256_bytes(arr.tobytes())


def _string_list_digest(values: list[str]) -> str:
    """Return a stable digest for an ordered string list."""
    return _sha256_text("\0".join(values))


def _bootstrap_repo(repo_root: Path) -> None:
    """Prepare imports for a target worktree in an isolated worker process."""
    os.chdir(repo_root)
    repo_str = str(repo_root)
    if repo_str in sys.path:
        sys.path.remove(repo_str)
    sys.path.insert(0, repo_str)


def _provider_metadata(provider: Any) -> dict[str, Any]:
    """Serialize provider metadata used to audit scenario definitions."""
    metadata: dict[str, Any] = {
        "name": provider.name,
        "cost_per_token": float(provider.cost_per_token),
        "cost_per_m_tokens": float(provider.cost_per_token * 1e6),
        "p50_ms": float(provider.true_p50_ms()),
        "p99_ms": float(provider.true_p99_ms()),
    }

    tier = getattr(provider, "tier", None)
    if tier is not None and provider.__class__.__name__ == "TieredProvider":
        metadata["tier"] = str(getattr(tier, "value", tier))

    quota = getattr(provider, "quota", None)
    if quota is not None:
        metadata["quota_size"] = int(quota.size)
        metadata["quota_window_sec"] = float(quota.window_sec)

    concurrency = getattr(provider, "concurrency", None)
    if concurrency is not None:
        metadata["concurrency_limit"] = int(concurrency.limit)

    shift_time = getattr(provider, "shift_time", None)
    if shift_time is not None:
        metadata["shift_time"] = float(shift_time)
        metadata["p50_ms_after"] = float(provider.true_p50_ms(float(shift_time)))
        metadata["p99_ms_after"] = float(provider.true_p99_ms(float(shift_time)))

    service_time_dist = getattr(provider, "service_time_dist", None)
    if service_time_dist is not None:
        metadata["service_time_p50_ms"] = float(service_time_dist.p50())
        metadata["service_time_p99_ms"] = float(service_time_dist.p99())

    return metadata


def _seed_run_payload(run: Any, slo_thresholds_ms: list[float]) -> dict[str, Any]:
    """Serialize one seeded strategy execution."""
    hedge_rate = (
        float(run.hedge_rate())
        if hasattr(run, "hedge_rate")
        else float(np.mean(np.asarray(run.hedge_triggered, dtype=np.bool_)))
    )
    if hasattr(run, "provider_fractions"):
        provider_fractions = {
            name: float(frac) for name, frac in run.provider_fractions().items()
        }
    else:
        provider_fractions = {
            name: float(run.provider.count(name) / len(run.provider))
            for name in sorted(set(run.provider))
        }
    payload: dict[str, Any] = {
        "metrics": {
            f"slo_violation_rate_{int(slo)}ms": float(run.slo_violation_rate(slo))
            for slo in slo_thresholds_ms
        },
        "mean_cost_usd": float(run.mean_cost_usd()),
        "p50_ms": float(run.p50_ms()),
        "p99_ms": float(run.p99_ms()),
        "hedge_rate": hedge_rate,
        "provider_fractions": _sorted_dict(provider_fractions),
        "digests": {
            "timestamp": _float_array_digest(run.timestamp),
            "ttft_ms": _float_array_digest(run.ttft_ms),
            "cost_usd": _float_array_digest(run.cost_usd),
            "provider": _string_list_digest(run.provider),
            "hedge_triggered": _bool_array_digest(run.hedge_triggered),
        },
    }

    if hasattr(run, "tier_fractions"):
        tier_fractions = {
            name: float(frac) for name, frac in run.tier_fractions().items()
        }
        if tier_fractions:
            payload["tier_fractions"] = _sorted_dict(tier_fractions)

    for field_name in [
        "tier",
        "quota_fraction_used",
        "concurrency_utilization",
        "shadow_price_q",
        "rejected",
    ]:
        field_value = getattr(run, field_name, None)
        if field_value is None:
            continue
        arr = np.asarray(field_value)
        if arr.size == 0:
            continue
        if arr.dtype == np.bool_:
            payload["digests"][field_name] = _bool_array_digest(arr)
        elif np.issubdtype(arr.dtype, np.number):
            payload["digests"][field_name] = _float_array_digest(arr)
        else:
            payload["digests"][field_name] = _string_list_digest(list(field_value))

    return payload


def _aggregate_seed_runs(
    runs: list[Any],
    seeds: list[int],
    slo_thresholds_ms: list[float],
) -> dict[str, Any]:
    """Aggregate multi-seed strategy runs into deterministic JSON."""
    seed_runs: dict[str, Any] = {}
    for seed, run in zip(seeds, runs, strict=True):
        seed_runs[str(seed)] = _seed_run_payload(run, slo_thresholds_ms)

    aggregate: dict[str, Any] = {
        f"slo_violation_rate_{int(slo)}ms": float(
            np.mean([run.slo_violation_rate(slo) for run in runs])
        )
        for slo in slo_thresholds_ms
    }
    aggregate["mean_cost_usd"] = float(np.mean([run.mean_cost_usd() for run in runs]))
    aggregate["p50_ms"] = float(np.mean([run.p50_ms() for run in runs]))
    aggregate["p99_ms"] = float(np.mean([run.p99_ms() for run in runs]))
    aggregate["hedge_rate"] = float(
        np.mean(
            [
                (
                    float(run.hedge_rate())
                    if hasattr(run, "hedge_rate")
                    else float(np.mean(np.asarray(run.hedge_triggered, dtype=np.bool_)))
                )
                for run in runs
            ]
        )
    )

    provider_fraction_lists: dict[str, list[float]] = {}
    for run in runs:
        provider_fractions = (
            run.provider_fractions()
            if hasattr(run, "provider_fractions")
            else {
                name: float(run.provider.count(name) / len(run.provider))
                for name in sorted(set(run.provider))
            }
        )
        for provider_name, fraction in provider_fractions.items():
            provider_fraction_lists.setdefault(provider_name, []).append(float(fraction))
    aggregate["provider_fractions"] = _sorted_dict(
        {
            provider_name: float(np.mean(values))
            for provider_name, values in provider_fraction_lists.items()
        }
    )

    if hasattr(runs[0], "tier_fractions"):
        tier_fraction_lists: dict[str, list[float]] = {}
        for run in runs:
            for tier_name, fraction in run.tier_fractions().items():
                tier_fraction_lists.setdefault(tier_name, []).append(float(fraction))
        if tier_fraction_lists:
            aggregate["tier_fractions"] = _sorted_dict(
                {
                    tier_name: float(np.mean(values))
                    for tier_name, values in tier_fraction_lists.items()
                }
            )

    return {
        "aggregate": aggregate,
        "seed_runs": _sorted_dict(seed_runs),
    }


def _scenario_payload(
    scenario: Any,
    strategy_names: list[str],
    results: dict[str, list[Any]],
) -> dict[str, Any]:
    """Serialize one scenario and all strategy results."""
    payload: dict[str, Any] = {
        "name": scenario.name,
        "description": scenario.description,
        "n_requests": int(scenario.n_requests),
        "duration_seconds": float(scenario.duration_seconds),
        "arrival_process": str(scenario.arrival_process),
        "primary_slo_ms": float(scenario.primary_slo_ms),
        "slo_thresholds_ms": [float(value) for value in scenario.slo_thresholds_ms],
        "provider_names": list(scenario.provider_names),
        "providers": _sorted_dict(
            {
                provider.name: _provider_metadata(provider)
                for provider in scenario.providers
            }
        ),
        "strategy_order": list(strategy_names),
        "strategy_hash": _strategy_hash(strategy_names),
        "strategies": _sorted_dict(
            {
                strategy_name: _aggregate_seed_runs(
                    results[strategy_name],
                    SEEDS,
                    [float(value) for value in scenario.slo_thresholds_ms],
                )
                for strategy_name in strategy_names
            }
        ),
    }
    return payload


def _capture_latency_synthetic(repo_root: Path) -> dict[str, Any]:
    """Capture S1-S5 latency scenarios."""
    _bootstrap_repo(repo_root)

    from experiments.synthetic_latency import load_all_world_scenarios  # noqa: E402
    from rwsim.runner import LATENCY_STRATEGIES, run_registered_strategy  # noqa: E402
    from rwsim.world import generate_workload  # noqa: E402

    strategy_names = list(LATENCY_STRATEGIES)
    scenarios_payload: dict[str, Any] = {}
    scenarios = {scenario.name: scenario for scenario in load_all_world_scenarios()}
    for scenario_id, scenario in scenarios.items():
        requests = generate_workload(
            n_requests=scenario.n_requests,
            duration_seconds=scenario.duration_seconds,
            seed=0,
            start_time=0.0,
            arrival_process=scenario.arrival_process,
        )
        results = {
            strategy_name: [
                run_registered_strategy(scenario, requests, strategy_name, seed=seed)
                for seed in SEEDS
            ]
            for strategy_name in strategy_names
        }
        scenarios_payload[scenario_id] = _scenario_payload(
            scenario,
            strategy_names,
            results,
        )

    return {
        "family": "latency_synthetic",
        "seed_order": list(SEEDS),
        "strategy_order": strategy_names,
        "strategy_hash": _strategy_hash(strategy_names),
        "scenarios": _sorted_dict(scenarios_payload),
    }


def _capture_latency_sanity(repo_root: Path) -> dict[str, Any]:
    """Capture Step 1-Step 5 sanity scenarios."""
    _bootstrap_repo(repo_root)

    from experiments.synthetic_latency import (  # noqa: E402
        make_sanity_steps,
    )
    from rwsim.runner import LATENCY_STRATEGIES, run_registered_strategy  # noqa: E402
    from rwsim.world import generate_workload  # noqa: E402

    strategy_names = list(LATENCY_STRATEGIES)
    steps_payload: dict[str, Any] = {}
    for step in make_sanity_steps():
        scenario_payloads: dict[str, Any] = {}
        for scenario in step.scenarios:
            requests = generate_workload(
                n_requests=SANITY_N_REQUESTS,
                duration_seconds=SANITY_DURATION_SECONDS,
                seed=0,
                start_time=0.0,
                arrival_process="poisson",
            )
            results = {
                strategy_name: [
                    run_registered_strategy(scenario, requests, strategy_name, seed=seed)
                    for seed in SEEDS
                ]
                for strategy_name in strategy_names
            }
            scenario_payloads[scenario.name] = _scenario_payload(
                scenario,
                strategy_names,
                results,
            )
        steps_payload[step.step_id] = {
            "step_id": step.step_id,
            "description": step.description,
            "is_sweep": bool(step.is_sweep),
            "scenarios": _sorted_dict(scenario_payloads),
        }

    return {
        "family": "latency_sanity",
        "seed_order": list(SEEDS),
        "strategy_order": strategy_names,
        "strategy_hash": _strategy_hash(strategy_names),
        "sanity_n_requests": SANITY_N_REQUESTS,
        "sanity_duration_seconds": SANITY_DURATION_SECONDS,
        "steps": _sorted_dict(steps_payload),
    }


def _capture_tiered_family(repo_root: Path, family: str) -> dict[str, Any]:
    """Capture one tiered-family scenario set."""
    _bootstrap_repo(repo_root)

    from legacy.experiment.scripts.simulate.synthetic.tiered.runner import (  # noqa: E402
        run_tiered_scenario,
    )
    from legacy.experiment.scripts.simulate.synthetic.tiered.strategies import (  # noqa: E402
        TIERED_STRATEGIES,
    )
    from legacy.experiment.scripts.simulate.synthetic.workload import (  # noqa: E402
        generate_workload,
    )

    if family == "tiered":
        from legacy.experiment.scripts.simulate.synthetic.tiered.scenarios import (  # noqa: E402
            make_tiered_scenarios,
        )

        scenario_factory = make_tiered_scenarios
    elif family == "calibrated":
        from experiments.tiered_capacity import (  # noqa: E402
            make_mm25_scenarios,
        )

        scenario_factory = make_mm25_scenarios
    elif family == "stress":
        from experiments.tiered_capacity import (  # noqa: E402
            make_stress_scenarios,
        )

        scenario_factory = make_stress_scenarios
    else:
        raise ValueError(f"Unsupported tiered family: {family}")

    strategy_names = list(TIERED_STRATEGIES)
    scenarios_payload: dict[str, Any] = {}
    for scenario_id, scenario in scenario_factory().items():
        requests = generate_workload(
            n_requests=scenario.n_requests,
            duration_seconds=scenario.duration_seconds,
            seed=0,
            start_time=0.0,
            arrival_process=scenario.arrival_process,
        )
        results = run_tiered_scenario(
            scenario,
            requests,
            seeds=SEEDS,
            strategies=strategy_names,
        )
        scenarios_payload[scenario_id] = _scenario_payload(
            scenario,
            strategy_names,
            results,
        )

    return {
        "family": family,
        "seed_order": list(SEEDS),
        "strategy_order": strategy_names,
        "strategy_hash": _strategy_hash(strategy_names),
        "scenarios": _sorted_dict(scenarios_payload),
    }


def _capture_family_payload(family: str, repo_root: Path) -> dict[str, Any]:
    """Capture one family in the current process."""
    if family == "latency_synthetic":
        return _capture_latency_synthetic(repo_root)
    if family == "latency_sanity":
        return _capture_latency_sanity(repo_root)
    if family in {"tiered", "calibrated", "stress"}:
        return _capture_tiered_family(repo_root, family)
    raise ValueError(f"Unknown family: {family}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _run_family_worker(family: str, output_root: Path) -> Path:
    """Run one family in an isolated subprocess and return the output path."""
    spec = FAMILY_SPECS[family]
    repo_root = Path(spec["repo_root"])
    output_path = output_root / spec["output_relpath"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--mode",
        "capture-family",
        "--family",
        family,
        "--repo-root",
        str(repo_root),
        "--output",
        str(output_path),
    ]
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    subprocess.run(
        cmd,
        cwd=repo_root,
        check=True,
        text=True,
        env=env,
    )
    return output_path


def _gather_grep_hits(root: Path, pattern: re.Pattern[str]) -> list[tuple[str, str]]:
    """Collect grep hits for an audit regex pattern."""
    hits: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*.py")):
        if ".git" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if pattern.search(line):
                hits.append((str(path.relative_to(WORKSPACE_ROOT)), line.rstrip()))
    return hits


def _write_grep_audit(output_root: Path) -> Path:
    """Write Phase 0 grep audit results."""
    audit_path = output_root / "GREP_AUDIT.md"
    checks = [
        (
            "TieredProvider isinstance",
            re.compile(r"isinstance\([^\n]*TieredProvider"),
        ),
        (
            "SyntheticProvider isinstance",
            re.compile(r"isinstance\([^\n]*SyntheticProvider"),
        ),
        (
            "_active_dist hasattr",
            re.compile(r"hasattr\([^\n]*[\"']_active_dist[\"']"),
        ),
        (
            "quota hasattr",
            re.compile(r"hasattr\([^\n]*[\"']quota[\"']"),
        ),
        (
            "concurrency hasattr",
            re.compile(r"hasattr\([^\n]*[\"']concurrency[\"']"),
        ),
    ]
    lines = [
        "# Phase 0 Grep Audit",
        "",
        f"Scope: {', '.join(root.name for root in AUDIT_ROOTS)}.",
        "",
    ]

    for title, pattern in checks:
        lines.append(f"## {title}")
        lines.append("")
        pattern_hits: list[tuple[str, str]] = []
        for root in AUDIT_ROOTS:
            pattern_hits.extend(_gather_grep_hits(root, pattern))
        if not pattern_hits:
            lines.append("- No matches found.")
            lines.append("")
            continue
        for relpath, line in pattern_hits:
            lines.append(f"- `{relpath}`: `{line}`")
        lines.append("")

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("\n".join(lines), encoding="utf-8")
    return audit_path


def _capture_all(output_root: Path, families: list[str]) -> None:
    """Capture all requested families and the grep audit."""
    output_root.mkdir(parents=True, exist_ok=True)
    for family in families:
        output_path = _run_family_worker(family, output_root)
        print(f"[capture] {family}: wrote {output_path.relative_to(output_root)}")
    audit_path = _write_grep_audit(output_root)
    print(f"[capture] grep audit: wrote {audit_path.relative_to(output_root)}")


def _load_json(path: Path) -> Any:
    """Load JSON from disk."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _compare_json(
    expected: Any,
    actual: Any,
    path: str,
    *,
    rel_tol: float,
    abs_tol: float,
    diffs: list[str],
) -> None:
    """Recursively compare parsed JSON payloads."""
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if math.isnan(float(expected)) and math.isnan(float(actual)):
            return
        if not math.isclose(float(expected), float(actual), rel_tol=rel_tol, abs_tol=abs_tol):
            diffs.append(f"{path}: expected {expected!r}, got {actual!r}")
        return

    if type(expected) is not type(actual):
        diffs.append(
            f"{path}: expected type {type(expected).__name__}, "
            f"got {type(actual).__name__}"
        )
        return

    if isinstance(expected, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        if missing:
            diffs.append(f"{path}: missing keys {missing}")
        if extra:
            diffs.append(f"{path}: unexpected keys {extra}")
        for key in sorted(expected_keys & actual_keys):
            _compare_json(
                expected[key],
                actual[key],
                f"{path}.{key}" if path else key,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                diffs=diffs,
            )
        return

    if isinstance(expected, list):
        if len(expected) != len(actual):
            diffs.append(f"{path}: expected len {len(expected)}, got len {len(actual)}")
            return
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            _compare_json(
                expected_item,
                actual_item,
                f"{path}[{index}]",
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                diffs=diffs,
            )
        return

    if expected != actual:
        diffs.append(f"{path}: expected {expected!r}, got {actual!r}")


def _compare_capture_to_golden(
    golden_root: Path,
    families: list[str],
    *,
    rel_tol: float,
    abs_tol: float,
) -> list[str]:
    """Re-capture requested families and compare them to golden JSON."""
    diffs: list[str] = []
    with tempfile.TemporaryDirectory(prefix="rwsim-golden-") as temp_dir:
        temp_root = Path(temp_dir)
        _capture_all(temp_root, families)
        for family in families:
            relpath = FAMILY_SPECS[family]["output_relpath"]
            expected_path = golden_root / relpath
            actual_path = temp_root / relpath
            if not expected_path.exists():
                diffs.append(f"{relpath}: golden file missing")
                continue
            if not actual_path.exists():
                diffs.append(f"{relpath}: recaptured file missing")
                continue
            _compare_json(
                _load_json(expected_path),
                _load_json(actual_path),
                str(relpath),
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                diffs=diffs,
            )

        expected_audit = golden_root / "GREP_AUDIT.md"
        actual_audit = temp_root / "GREP_AUDIT.md"
        if expected_audit.exists() and actual_audit.exists():
            if expected_audit.read_text(encoding="utf-8") != actual_audit.read_text(
                encoding="utf-8"
            ):
                diffs.append("GREP_AUDIT.md: content differs")
        elif expected_audit.exists() != actual_audit.exists():
            diffs.append("GREP_AUDIT.md: existence differs")

    return diffs


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["capture", "compare", "capture-family"],
        default="capture",
    )
    parser.add_argument(
        "--families",
        nargs="*",
        choices=sorted(FAMILY_SPECS),
        default=sorted(FAMILY_SPECS),
        help="Subset of family names to process.",
    )
    parser.add_argument(
        "--family",
        choices=sorted(FAMILY_SPECS),
        help="Single family name for worker mode.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="Target repo root for worker mode.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_GOLDEN_ROOT,
        help="Root directory for capture mode outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Single output file path for worker mode.",
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=0.0,
        help="Relative tolerance for compare mode.",
    )
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=0.0,
        help="Absolute tolerance for compare mode.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the capture or compare command."""
    args = _parse_args()

    if args.mode == "capture-family":
        if args.family is None or args.repo_root is None or args.output is None:
            raise ValueError(
                "--family, --repo-root, and --output are required for capture-family mode."
            )
        payload = _capture_family_payload(args.family, args.repo_root)
        _write_json(args.output, payload)
        return 0

    families = list(args.families)
    if args.mode == "capture":
        _capture_all(args.output_root, families)
        return 0

    diffs = _compare_capture_to_golden(
        args.output_root,
        families,
        rel_tol=args.rel_tol,
        abs_tol=args.abs_tol,
    )
    if diffs:
        print("[compare] mismatches found:")
        for diff in diffs:
            print(f"  - {diff}")
        return 1

    print("[compare] all requested families match golden baselines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
