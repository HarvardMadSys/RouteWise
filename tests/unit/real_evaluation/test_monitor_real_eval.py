from __future__ import annotations

import importlib.util
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
