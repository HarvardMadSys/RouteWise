#!/usr/bin/env python3
"""Live progress monitor for the multi-policy real-eval runs.

Reads the per-policy ``requests.csv`` + ``run.log`` files written by
``scripts/run_real_eval_8h_policy_processes.sh`` and prints a single
snapshot table with the key health metrics the user wants while a long
8h experiment is in flight:

  - wall-clock elapsed since the run started
  - trace replay progress (by request count and by trace time)
  - per-policy totals: requests, cost, latency percentiles, SLO
    violation rate, tier mix, top providers, hedge stats

Designed to be safe to run repeatedly against a live output dir; it only
reads the CSVs and never writes back. Use ``--watch N`` to refresh every
N seconds (Ctrl+C to stop).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

# Matches the leading INFO line written by runner.py at startup, e.g.
# "17:14:13 [INFO] run plan: 14233 trace requests over 29233.0s trace time;"
RUN_PLAN_RE = re.compile(
    r"run plan: (\d+) trace requests over ([\d.]+)s trace time"
)

# We track time-to-first-token (the routing-quality signal) for the
# percentiles and SLO check. e2e is mostly a function of output-token
# count, which the policy can't control; ttft is what the routing
# decision actually moves.
LATENCY_COL = "ttft_ms"

# Inventory-name prefix used by the OpenRouter auto/sort baselines. The
# runner writes ``actual_provider`` as e.g. ``__or_auto__@DeepInfra``,
# where the suffix is the sub-provider OpenRouter's server-side router
# picked. Treat that suffix as the meaningful provider attribution for
# these baselines.
OR_SENTINEL_PREFIXES = ("__or_auto__", "__or_sort_")


@dataclass
class PolicyStats:
    policy: str
    total_requests: int
    successes: int
    failures: int
    total_cost_usd: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    slo_violation_pct: float
    tier_mix: dict[str, float]
    provider_mix: list[tuple[str, float]]
    hedge_requests: int
    hedge_wins: int
    rate_limited_429: int
    earliest_ts: float | None
    latest_ts: float | None


def parse_run_plan(run_log: Path) -> tuple[int | None, float | None]:
    """Return (total_trace_requests, trace_time_sec) from the first run-plan line."""
    if not run_log.exists():
        return None, None
    try:
        with run_log.open() as f:
            for line in f:
                m = RUN_PLAN_RE.search(line)
                if m:
                    return int(m.group(1)), float(m.group(2))
                if "[INFO]" in line and "trace requests" not in line:
                    continue
    except OSError:
        return None, None
    return None, None


def safe_float(s: str) -> float | None:
    if s == "" or s is None:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if math.isnan(v):
        return None
    return v


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile. ``q`` in [0, 100]."""
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (q / 100.0) * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def normalize_provider(name: str) -> str:
    """Collapse a raw ``actual_provider`` value into a stable label.

    Three shapes show up:
      * ``OR_WandB@WandB`` (pinned OR provider) — base is meaningful, the
        ``@WandB`` suffix is redundant. Keep ``OR_WandB``.
      * ``__or_auto__@DeepInfra`` / ``__or_sort_latency__@Friendli`` (OR
        sentinel baseline) — base is just the sentinel name; the suffix
        is the actual sub-provider OR picked. Render as ``auto:DeepInfra``
        / ``sort_latency:Friendli`` so the user can still see the mix.
      * ``Featherless_SC`` / ``Chutes_SQ`` (direct transports) — identity.
    """
    if "@" in name:
        base, sub = name.split("@", 1)
        if base.startswith("__") and base.endswith("__"):
            return sub or name
        return base
    return name


def infer_tier(tier: str, actual_provider: str) -> str:
    """Map ``unknown`` tier rows from OR sentinel baselines to ``api``.

    The runner restricts the sentinel allowlist in ``_send_or_sentinel`` to
    inventory providers with ``tier == "api"``, so every sentinel
    dispatch is API-tier by construction. The CSV column is just missing
    that metadata because the sentinel isn't a single ProviderSpec.
    """
    if tier and tier != "unknown":
        return tier
    if actual_provider.startswith("__") and "@" in actual_provider:
        return "api"
    return tier or "unknown"


def collect_policy(policy_dir: Path, slo_ms: float) -> PolicyStats | None:
    """Parse one policy's ``requests.csv`` into aggregate stats.

    Returns ``None`` if the CSV is missing entirely (policy hasn't started
    dispatching yet); returns a zero-filled stats object if the CSV exists
    but has no rows (header-only).
    """
    csv_path = policy_dir / "requests.csv"
    if not csv_path.exists():
        return None

    total = 0
    successes = 0
    failures = 0
    total_cost = 0.0
    latencies: list[float] = []
    slo_violations = 0
    tier_counter: Counter[str] = Counter()
    provider_counter: Counter[str] = Counter()
    hedge_requests = 0
    hedge_wins = 0
    rate_limited_429 = 0
    earliest_ts: float | None = None
    latest_ts: float | None = None

    try:
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                ts = safe_float(row.get("ts", ""))
                if ts is not None:
                    earliest_ts = ts if earliest_ts is None else min(earliest_ts, ts)
                    latest_ts = ts if latest_ts is None else max(latest_ts, ts)

                status = row.get("status", "")
                e2e = safe_float(row.get(LATENCY_COL, ""))
                # Failed/canceled requests count as SLO violations even if
                # they don't contribute a latency sample to percentiles.
                if status == "success":
                    successes += 1
                    if e2e is not None and e2e > 0:
                        latencies.append(e2e)
                        if e2e > slo_ms:
                            slo_violations += 1
                else:
                    failures += 1
                    slo_violations += 1

                cost = safe_float(row.get("billed_cost_usd", ""))
                if cost is not None:
                    total_cost += cost

                raw_provider = row.get("actual_provider") or "unknown"
                tier = infer_tier(row.get("tier") or "unknown", raw_provider)
                tier_counter[tier] += 1
                provider_counter[normalize_provider(raw_provider)] += 1

                if row.get("hedge_triggered") == "1":
                    hedge_requests += 1
                    if row.get("hedge_winner") == "backup":
                        hedge_wins += 1
                if row.get("rate_limited") == "1":
                    rate_limited_429 += 1
    except OSError as exc:
        print(f"warning: could not read {csv_path}: {exc}", file=sys.stderr)
        return None

    latencies.sort()
    tier_mix = {k: v / total for k, v in tier_counter.items()} if total else {}
    provider_mix = (
        sorted(
            ((k, v / total) for k, v in provider_counter.items()),
            key=lambda kv: kv[1],
            reverse=True,
        )
        if total
        else []
    )

    return PolicyStats(
        policy=policy_dir.name,
        total_requests=total,
        successes=successes,
        failures=failures,
        total_cost_usd=total_cost,
        latency_mean_ms=statistics.fmean(latencies) if latencies else float("nan"),
        latency_p50_ms=percentile(latencies, 50),
        latency_p90_ms=percentile(latencies, 90),
        latency_p95_ms=percentile(latencies, 95),
        latency_p99_ms=percentile(latencies, 99),
        slo_violation_pct=(slo_violations / total * 100.0) if total else 0.0,
        tier_mix=tier_mix,
        provider_mix=provider_mix,
        hedge_requests=hedge_requests,
        hedge_wins=hedge_wins,
        rate_limited_429=rate_limited_429,
        earliest_ts=earliest_ts,
        latest_ts=latest_ts,
    )


def fmt_duration(sec: float) -> str:
    if sec < 0 or math.isnan(sec):
        return "n/a"
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}h{m:02d}m{s:02d}s"


def fmt_ms(v: float) -> str:
    if math.isnan(v):
        return "    n/a"
    return f"{v:7.0f}"


def fmt_pct(v: float) -> str:
    return f"{v:5.1f}%"


def fmt_mix(mix: dict[str, float], max_entries: int = 3) -> str:
    """Compact ``tier=p%, ...`` rendering. Sorted by share, descending."""
    if not mix:
        return "-"
    items = sorted(mix.items(), key=lambda kv: kv[1], reverse=True)
    parts = [f"{k}:{v * 100:.0f}%" for k, v in items[:max_entries]]
    if len(items) > max_entries:
        parts.append("...")
    return " ".join(parts)


def fmt_provider_mix(mix: list[tuple[str, float]], max_entries: int = 3) -> str:
    if not mix:
        return "-"
    parts = [f"{name}:{share * 100:.0f}%" for name, share in mix[:max_entries]]
    if len(mix) > max_entries:
        parts.append(f"+{len(mix) - max_entries}")
    return " ".join(parts)


def render_table(rows: list[list[str]], headers: list[str]) -> str:
    """Plain-text aligned table. Avoids a pandas/rich dependency."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "  "
    out_lines = [
        sep.join(h.ljust(widths[i]) for i, h in enumerate(headers)),
        sep.join("-" * widths[i] for i in range(len(headers))),
    ]
    for row in rows:
        out_lines.append(sep.join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(out_lines)


def load_slo_ms(policy_dir: Path, fallback_slo_ms: float) -> float:
    """Read SLO threshold from this policy's args.json, falling back to the
    base directory default. Keeps the per-policy view honest if a user ever
    runs a mixed-SLO sweep."""
    args_path = policy_dir / "args.json"
    if not args_path.exists():
        return fallback_slo_ms
    try:
        with args_path.open() as f:
            args = json.load(f)
    except (OSError, json.JSONDecodeError):
        return fallback_slo_ms
    v = args.get("slo_ms")
    if v is None or v == "None":
        return fallback_slo_ms
    try:
        return float(v)
    except (TypeError, ValueError):
        return fallback_slo_ms


def discover_policies(base_dir: Path) -> list[Path]:
    """Return policy subdirs in the order recorded in ``policies.txt`` when
    that file exists, otherwise alphabetical. This keeps the table layout
    stable across snapshots."""
    policies_txt = base_dir / "policies.txt"
    ordered: list[Path] = []
    if policies_txt.exists():
        for name in policies_txt.read_text().splitlines():
            name = name.strip()
            if not name:
                continue
            p = base_dir / name
            if p.is_dir():
                ordered.append(p)
    if ordered:
        return ordered
    return sorted(p for p in base_dir.iterdir() if p.is_dir() and (p / "args.json").exists())


def load_run_env(base_dir: Path) -> dict[str, str]:
    """Parse the ``run_env.txt`` snapshot written by the launcher.

    Only used for default SLO_MS / DURATION_SEC; missing file is fine."""
    env_path = base_dir / "run_env.txt"
    env: dict[str, str] = {}
    if not env_path.exists():
        return env
    try:
        for line in env_path.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    except OSError:
        pass
    return env


def render_snapshot(base_dir: Path) -> str:
    run_env = load_run_env(base_dir)
    default_slo_ms = float(run_env.get("SLO_MS") or 3000.0)
    duration_cap_sec = (
        float(run_env["DURATION_SEC"]) if run_env.get("DURATION_SEC") else None
    )

    policy_dirs = discover_policies(base_dir)
    if not policy_dirs:
        return f"no policy subdirectories found under {base_dir}"

    trace_total: int | None = None
    trace_time_sec: float | None = None
    for pdir in policy_dirs:
        total, trace_secs = parse_run_plan(pdir / "run.log")
        if total is not None:
            trace_total = total
            trace_time_sec = trace_secs
            break

    stats: list[PolicyStats] = []
    earliest_ts_global: float | None = None
    latest_ts_global: float | None = None
    for pdir in policy_dirs:
        slo_ms = load_slo_ms(pdir, default_slo_ms)
        s = collect_policy(pdir, slo_ms=slo_ms)
        if s is None:
            stats.append(
                PolicyStats(
                    policy=pdir.name,
                    total_requests=0,
                    successes=0,
                    failures=0,
                    total_cost_usd=0.0,
                    latency_mean_ms=float("nan"),
                    latency_p50_ms=float("nan"),
                    latency_p90_ms=float("nan"),
                    latency_p95_ms=float("nan"),
                    latency_p99_ms=float("nan"),
                    slo_violation_pct=0.0,
                    tier_mix={},
                    provider_mix=[],
                    hedge_requests=0,
                    hedge_wins=0,
                    rate_limited_429=0,
                    earliest_ts=None,
                    latest_ts=None,
                )
            )
            continue
        stats.append(s)
        if s.earliest_ts is not None:
            earliest_ts_global = (
                s.earliest_ts
                if earliest_ts_global is None
                else min(earliest_ts_global, s.earliest_ts)
            )
        if s.latest_ts is not None:
            latest_ts_global = (
                s.latest_ts
                if latest_ts_global is None
                else max(latest_ts_global, s.latest_ts)
            )

    now = time.time()
    elapsed_sec = (now - earliest_ts_global) if earliest_ts_global else 0.0

    total_completed = sum(s.total_requests for s in stats)
    n_policies = len(stats)
    expected_per_policy = trace_total or 0
    completion_count_pct = (
        (total_completed / (expected_per_policy * n_policies) * 100.0)
        if expected_per_policy and n_policies
        else float("nan")
    )
    completion_time_pct = (
        min(elapsed_sec / trace_time_sec * 100.0, 100.0)
        if trace_time_sec
        else float("nan")
    )

    header_lines = [
        f"snapshot @ {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}",
        f"output_dir       : {base_dir}",
        f"policies         : {n_policies}",
        f"trace dataset    : "
        + (f"{trace_total} requests over {trace_time_sec / 3600:.1f} hours" if trace_total and trace_time_sec else "unknown"),
        f"wall-clock elapsed : {fmt_duration(elapsed_sec)}"
        + (f"  (cap {fmt_duration(duration_cap_sec)})" if duration_cap_sec else ""),
        f"trace replayed   : "
        + (f"{completion_time_pct:5.1f}% by trace time" if not math.isnan(completion_time_pct) else "n/a")
        + (
            f", {completion_count_pct:5.1f}% by per-policy request count"
            if not math.isnan(completion_count_pct)
            else ""
        ),
        f"SLO threshold    : {default_slo_ms:.0f} ms (ttft)",
        f"total cost (all) : ${sum(s.total_cost_usd for s in stats):.4f}",
    ]

    headers = [
        "policy",
        "reqs",
        "fail",
        "cost$",
        "mean",
        "p50",
        "p90",
        "p95",
        "p99",
        "SLO viol",
        "tier mix",
        "top providers",
        "hedge",
        "win",
        "429s",
    ]
    rows: list[list[str]] = []
    for s in stats:
        rows.append(
            [
                s.policy,
                str(s.total_requests),
                str(s.failures),
                f"{s.total_cost_usd:.4f}",
                fmt_ms(s.latency_mean_ms),
                fmt_ms(s.latency_p50_ms),
                fmt_ms(s.latency_p90_ms),
                fmt_ms(s.latency_p95_ms),
                fmt_ms(s.latency_p99_ms),
                fmt_pct(s.slo_violation_pct),
                fmt_mix(s.tier_mix),
                fmt_provider_mix(s.provider_mix),
                str(s.hedge_requests),
                str(s.hedge_wins),
                str(s.rate_limited_429),
            ]
        )

    table_text = "\n".join(header_lines) + "\n\n" + render_table(rows, headers)
    return table_text, {
        "snapshot_ts": now,
        "snapshot_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        "output_dir": str(base_dir),
        "n_policies": n_policies,
        "trace_total_requests": trace_total,
        "trace_time_sec": trace_time_sec,
        "wall_elapsed_sec": elapsed_sec,
        "duration_cap_sec": duration_cap_sec,
        "completion_time_pct": completion_time_pct if not math.isnan(completion_time_pct) else None,
        "completion_count_pct": completion_count_pct if not math.isnan(completion_count_pct) else None,
        "slo_ms": default_slo_ms,
        "total_cost_usd": sum(s.total_cost_usd for s in stats),
        "policies": [_stats_to_dict(s) for s in stats],
    }


def _stats_to_dict(s: PolicyStats) -> dict:
    d = asdict(s)
    for key in ("latency_mean_ms", "latency_p50_ms", "latency_p90_ms",
                "latency_p95_ms", "latency_p99_ms"):
        if d[key] is not None and math.isnan(d[key]):
            d[key] = None
    return d


def save_snapshot(base_dir: Path, label: str | None = None) -> Path:
    """Save a timestamped snapshot: JSON metrics, text table, and copies of
    every policy's current ``requests.csv`` and ``run.log``."""
    table_text, data = render_snapshot(base_dir)
    ts_str = time.strftime("%Y%m%d_%H%M%S")
    name = f"snapshot_{ts_str}" + (f"_{label}" if label else "")
    snap_dir = base_dir / "snapshots" / name
    snap_dir.mkdir(parents=True, exist_ok=True)

    (snap_dir / "summary.txt").write_text(table_text + "\n")
    with (snap_dir / "summary.json").open("w") as f:
        json.dump(data, f, indent=2)

    policy_dirs = discover_policies(base_dir)
    for pdir in policy_dirs:
        dest = snap_dir / pdir.name
        dest.mkdir(exist_ok=True)
        for fname in ("requests.csv", "run.log", "args.json"):
            src = pdir / fname
            if src.exists():
                shutil.copy2(src, dest / fname)

    for extra in ("run_env.txt", "policies.txt", "initial_profile.json",
                   "policy_key_assignments.tsv"):
        src = base_dir / extra
        if src.exists():
            shutil.copy2(src, snap_dir / extra)

    return snap_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "base_dir",
        type=Path,
        help="OUTPUT_BASE directory passed to run_real_eval_8h_policy_processes.sh",
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=0.0,
        help="If > 0, refresh the snapshot every N seconds (Ctrl+C to stop).",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Save a timestamped snapshot (JSON + text + CSV copies) into "
             "<base_dir>/snapshots/ and print the path.",
    )
    parser.add_argument(
        "--snapshot-label",
        type=str,
        default=None,
        help="Optional label appended to the snapshot directory name.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base = args.base_dir.resolve()
    if not base.is_dir():
        print(f"error: {base} is not a directory", file=sys.stderr)
        return 2

    if args.snapshot:
        snap_dir = save_snapshot(base, label=args.snapshot_label)
        table_text, _ = render_snapshot(base)
        print(table_text)
        print(f"\nsnapshot saved -> {snap_dir}")
        return 0

    if args.watch <= 0:
        table_text, _ = render_snapshot(base)
        print(table_text)
        return 0
    try:
        while True:
            sys.stdout.write("\x1b[2J\x1b[H")
            table_text, _ = render_snapshot(base)
            sys.stdout.write(table_text)
            sys.stdout.write("\n")
            sys.stdout.flush()
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
