from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_monitor_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "monitor_real_eval.py"
    spec = importlib.util.spec_from_file_location("monitor_real_eval", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compute_concurrency_usage_reports_busy_time_and_peak() -> None:
    monitor = _load_monitor_module()
    usage = monitor.compute_concurrency_usage(
        [monitor.ConcurrencySpec("Featherless_SC", 2)],
        {"Featherless_SC": [(0.0, 10.0), (5.0, 15.0)]},
        window_start=0.0,
        window_end=20.0,
    )

    item = usage["Featherless_SC"]
    assert item.busy_time_pct == 75.0
    assert item.peak_used == 2
    assert item.limit == 2
    assert item.requests == 2
    assert monitor.fmt_concurrency_usage(usage) == "Featherless_SC:75%"


def test_peak_concurrency_treats_adjacent_intervals_as_non_overlapping() -> None:
    monitor = _load_monitor_module()

    assert monitor.peak_concurrency([(0.0, 1.0), (1.0, 2.0)]) == 1
    assert monitor.peak_concurrency([(0.0, 2.0), (1.0, 3.0)]) == 2


def test_union_interval_duration_merges_overlaps() -> None:
    monitor = _load_monitor_module()

    assert monitor.union_interval_duration([(0.0, 2.0), (1.0, 3.0), (4.0, 5.0)]) == 4.0


def test_load_inventory_specs_expands_subscription_plan_quota_windows(tmp_path) -> None:
    monitor = _load_monitor_module()
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "name": "MiniMax_Plus_SQ",
                        "tier": "quota",
                        "subscription_plan": "minimax_subscription_plus",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (policy_dir / "args.json").write_text(
        json.dumps({"inventory": str(inventory_path)}),
        encoding="utf-8",
    )

    _, quota_specs, _ = monitor.load_inventory_specs(policy_dir)
    specs_by_window = {spec.window_name: spec for spec in quota_specs}

    assert specs_by_window["five_hour"].quota_requests == 4500
    assert specs_by_window["five_hour"].quota_window_sec == 18000
    assert specs_by_window["weekly_allowance"].quota_requests == 45000
    assert specs_by_window["weekly_allowance"].quota_window_sec == 604800


def test_compute_quota_usage_renders_5h_and_weekly_headroom() -> None:
    monitor = _load_monitor_module()
    specs = [
        monitor.QuotaSpec("MiniMax_Plus_SQ", "five_hour", 4500, 18000.0),
        monitor.QuotaSpec("MiniMax_Plus_SQ", "weekly_allowance", 45000, 604800.0),
    ]

    usage = monitor.compute_quota_usage(
        specs,
        {"MiniMax_Plus_SQ": [0.0, 1000.0, 19000.0]},
        now=20000.0,
    )

    assert usage["MiniMax_Plus_SQ"]["five_hour"].used == 1
    assert usage["MiniMax_Plus_SQ"]["weekly_allowance"].used == 3
    assert (
        monitor.fmt_quota_usage(usage)
        == "MiniMax_Plus_SQ(5h:4499/4500,week:44997/45000)"
    )


def test_collect_policy_counts_hedge_backup_quota_even_when_primary_wins(
    tmp_path,
) -> None:
    monitor = _load_monitor_module()
    policy_dir = tmp_path / "budget_range_alpha75_hedge"
    policy_dir.mkdir()
    with (policy_dir / "requests.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "ts",
            "status",
            "ttft_ms",
            "e2e_ms",
            "actual_provider",
            "tier",
            "billed_cost_usd",
            "primary_provider",
            "backup_provider",
            "hedge_triggered",
            "backup_dispatch_ts",
            "backup_start_ts",
            "primary_start_ts",
            "rate_limited",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "ts": "120.0",
                "status": "success",
                "ttft_ms": "300",
                "e2e_ms": "1000",
                "actual_provider": "Featherless_SC",
                "tier": "concurrency",
                "billed_cost_usd": "0",
                "primary_provider": "Featherless_SC",
                "backup_provider": "MiniMax_Plus_SQ",
                "hedge_triggered": "1",
                "backup_dispatch_ts": "110.0",
                "backup_start_ts": "110.1",
                "primary_start_ts": "100.0",
                "rate_limited": "0",
            }
        )

    stats = monitor.collect_policy(
        policy_dir,
        slo_ms=3000.0,
        quota_specs=[
            monitor.QuotaSpec("MiniMax_Plus_SQ", "five_hour", 4500, 18000.0)
        ],
        now=130.0,
    )

    assert stats is not None
    assert stats.quota_usage["MiniMax_Plus_SQ"]["five_hour"].used == 1
    assert (
        monitor.fmt_quota_usage(stats.quota_usage)
        == "MiniMax_Plus_SQ(5h:4499/4500)"
    )
