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

Pass ``--profiles`` to additionally render the shared latency-profile event
log (warmup/probe/natural samples) so runs can inspect the provider latencies
the policies know about, including providers that were probed but not selected
for real trace requests.
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
from dataclasses import asdict, dataclass, field
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.subscriptions import load_subscription_plans

# Matches the leading INFO line written by runner.py at startup, e.g.
# "17:14:13 [INFO] run plan: 14233 trace requests over 29233.0s trace time;"
RUN_PLAN_RE = re.compile(r"run plan: (\d+) trace requests over ([\d.]+)s trace time")

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
    # ``{quota_provider_name: {quota_window_name: QuotaUsage}}``.
    # Computed by counting per-policy dispatches against the inventory's
    # explicit quota fields or its subscription-plan windows. Empty when
    # the inventory has no quota providers or hasn't been loaded yet.
    quota_usage: dict[str, dict[str, "QuotaUsage"]] = field(default_factory=dict)
    # ``{concurrency_provider_name: ConcurrencyUsage}`` computed from
    # completed request intervals in requests.csv. This is observed
    # utilization, not a live in-flight counter.
    concurrency_usage: dict[str, "ConcurrencyUsage"] = field(default_factory=dict)


@dataclass(frozen=True)
class QuotaSpec:
    """Subset of an inventory provider entry needed for quota accounting."""

    name: str
    window_name: str
    quota_requests: int
    quota_window_sec: float


@dataclass(frozen=True)
class QuotaUsage:
    """Observed usage for one quota provider window."""

    used: int
    limit: int
    window_sec: float

    @property
    def left(self) -> int:
        return max(self.limit - self.used, 0)


@dataclass(frozen=True)
class ConcurrencySpec:
    """Subset of an inventory provider entry needed for slot accounting."""

    name: str
    concurrency_limit: int


@dataclass(frozen=True)
class ConcurrencyUsage:
    """Observed concurrency busy-time for one provider."""

    busy_time_pct: float
    peak_used: int
    limit: int
    requests: int


@dataclass
class ProfileProviderStats:
    """Aggregated shared-profile events for one provider."""

    provider: str
    successes: int
    failures: int
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    slo_violation_pct: float
    source_counts: dict[str, int]
    last_ts: float | None
    last_source: str | None


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


def capacity_provider_key(raw_provider: str) -> str:
    """Return the inventory provider name used for capacity accounting."""
    return raw_provider.split("@", 1)[0] if "@" in raw_provider else raw_provider


def collect_policy(
    policy_dir: Path,
    slo_ms: float,
    provider_ttft: dict[str, list[float]] | None = None,
    provider_failures: Counter[str] | None = None,
    quota_specs: list[QuotaSpec] | None = None,
    concurrency_specs: list[ConcurrencySpec] | None = None,
    now: float | None = None,
) -> PolicyStats | None:
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
    # Per-policy dispatch timestamps keyed by raw ``actual_provider``,
    # used downstream to recompute the current quota-window usage.
    dispatch_ts_by_provider: dict[str, list[float]] = {}
    concurrency_names = {spec.name for spec in concurrency_specs or []}
    concurrency_intervals: dict[str, list[tuple[float, float]]] = {}
    observed_start_ts: float | None = None
    observed_end_ts: float | None = None

    def add_concurrency_interval(
        provider: str | None,
        start_ts: float | None,
        end_ts: float | None,
    ) -> None:
        if not provider or provider not in concurrency_names:
            return
        if start_ts is None or end_ts is None or end_ts <= start_ts:
            return
        concurrency_intervals.setdefault(provider, []).append((start_ts, end_ts))

    def add_quota_dispatch(provider: str | None, dispatch_ts: float | None) -> None:
        if not provider or dispatch_ts is None:
            return
        quota_key = capacity_provider_key(provider)
        if not quota_key:
            return
        dispatch_ts_by_provider.setdefault(quota_key, []).append(dispatch_ts)

    try:
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                ts = safe_float(row.get("ts", ""))
                if ts is not None:
                    earliest_ts = ts if earliest_ts is None else min(earliest_ts, ts)
                    latest_ts = ts if latest_ts is None else max(latest_ts, ts)
                    observed_end_ts = ts if observed_end_ts is None else max(observed_end_ts, ts)

                status = row.get("status", "")
                e2e = safe_float(row.get(LATENCY_COL, ""))
                e2e_ms = safe_float(row.get("e2e_ms", ""))
                inferred_start_ts = None
                if ts is not None and e2e_ms is not None and e2e_ms > 0:
                    inferred_start_ts = ts - (e2e_ms / 1000.0)
                    observed_start_ts = (
                        inferred_start_ts
                        if observed_start_ts is None
                        else min(observed_start_ts, inferred_start_ts)
                    )
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
                norm_provider = normalize_provider(raw_provider)
                provider_counter[norm_provider] += 1

                # Estimate concurrency occupancy from completed request
                # intervals. Newer CSVs record provider-local start times;
                # older rows fall back to ``ts - e2e_ms``. Hedges can occupy
                # both primary and backup slots, so count both when present.
                primary_key = capacity_provider_key(row.get("primary_provider") or "")
                backup_key = capacity_provider_key(row.get("backup_provider") or "")
                actual_key = capacity_provider_key(raw_provider)
                primary_start_ts = safe_float(row.get("primary_start_ts", ""))
                backup_start_ts = safe_float(row.get("backup_start_ts", ""))
                backup_dispatch_ts = safe_float(row.get("backup_dispatch_ts", ""))
                primary_start_ts = (
                    primary_start_ts if primary_start_ts is not None else inferred_start_ts
                )
                backup_start_ts = (
                    backup_start_ts if backup_start_ts is not None else backup_dispatch_ts
                )
                primary_dispatch_ts = primary_start_ts or inferred_start_ts or ts
                backup_charge_ts = backup_dispatch_ts or backup_start_ts or ts
                if primary_key:
                    add_quota_dispatch(primary_key, primary_dispatch_ts)
                else:
                    add_quota_dispatch(actual_key, ts)
                if backup_key:
                    add_quota_dispatch(backup_key, backup_charge_ts)
                if actual_key and actual_key not in {primary_key, backup_key}:
                    # 429 fallback attempts also reserve capacity. The CSV
                    # preserves the original primary and final provider, but
                    # not every intermediate fallback provider when multiple
                    # 429s occur, so this is the best reconstructable lower
                    # bound from the row schema.
                    add_quota_dispatch(actual_key, ts)
                if primary_start_ts is not None:
                    observed_start_ts = (
                        primary_start_ts
                        if observed_start_ts is None
                        else min(observed_start_ts, primary_start_ts)
                    )
                if backup_start_ts is not None:
                    observed_start_ts = (
                        backup_start_ts
                        if observed_start_ts is None
                        else min(observed_start_ts, backup_start_ts)
                    )
                add_concurrency_interval(primary_key, primary_start_ts, ts)
                add_concurrency_interval(backup_key, backup_start_ts, ts)
                if actual_key not in {primary_key, backup_key}:
                    add_concurrency_interval(actual_key, primary_start_ts or inferred_start_ts, ts)

                # Accumulate cross-policy per-provider TTFT samples for
                # the second table. Only successful, positive-TTFT rows
                # contribute to the latency distribution; everything else
                # increments the per-provider failure count.
                if provider_ttft is not None:
                    if status == "success" and e2e is not None and e2e > 0:
                        provider_ttft.setdefault(norm_provider, []).append(e2e)
                    elif provider_failures is not None:
                        provider_failures[norm_provider] += 1

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

    quota_usage: dict[str, dict[str, QuotaUsage]] = {}
    if quota_specs:
        quota_usage = compute_quota_usage(
            quota_specs,
            dispatch_ts_by_provider,
            now=time.time(),
            latest_activity_ts=latest_ts,
        )
    concurrency_usage: dict[str, ConcurrencyUsage] = {}
    if concurrency_specs:
        reference_end_ts = observed_end_ts
        now_ts = time.time() if now is None else now
        if observed_end_ts is not None and now_ts - observed_end_ts <= 60.0:
            reference_end_ts = now_ts
        concurrency_usage = compute_concurrency_usage(
            concurrency_specs,
            concurrency_intervals,
            window_start=observed_start_ts,
            window_end=reference_end_ts,
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
        quota_usage=quota_usage,
        concurrency_usage=concurrency_usage,
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


def quota_window_label(window_name: str, window_sec: float) -> str:
    """Return a compact display label for a quota window."""
    name = str(window_name or "").strip()
    normalized = name.lower()
    if normalized == "five_hour":
        return "5h"
    if normalized in {"weekly_allowance", "weekly"}:
        return "week"
    if normalized == "daily":
        return "day"
    if name and normalized != "window":
        return name.replace("_allowance", "").replace("_", "-")

    seconds = int(window_sec)
    if seconds == 604800:
        return "week"
    if seconds == 86400:
        return "day"
    if seconds > 0 and seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds > 0 and seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def quota_window_sort_key(window_name: str, usage: QuotaUsage) -> tuple[float, str]:
    """Sort shorter quota windows before longer ones."""
    return (usage.window_sec, window_name)


def fmt_quota_usage(
    quota_usage: dict[str, dict[str, QuotaUsage]] | None,
) -> str:
    """Render quota status as ``Provider(window:left/limit,...)``.

    ``left`` is shown (not ``used``) so the column makes the headroom
    immediately obvious — a value of ``0/X`` means the policy will not
    route there until the window rolls.
    """
    if not quota_usage:
        return "-"
    parts = []
    for name in sorted(quota_usage):
        windows = quota_usage[name]
        window_parts = []
        for window_name, usage in sorted(
            windows.items(),
            key=lambda kv: quota_window_sort_key(kv[0], kv[1]),
        ):
            label = quota_window_label(window_name, usage.window_sec)
            window_parts.append(f"{label}:{usage.left}/{usage.limit}")
        parts.append(f"{name}({','.join(window_parts)})")
    return " ".join(parts)


def fmt_concurrency_usage(
    concurrency_usage: dict[str, ConcurrencyUsage] | None,
) -> str:
    """Render observed concurrency busy-time.

    Format: ``Provider:busy%``. ``busy`` is the percent of the policy's
    observed experiment time where at least one slot from that provider was
    occupied. The monitor intentionally does not print a reconstructed
    peak-overlap value here: ``requests.csv`` is written after capacity
    release, so deriving exact lease overlap from completed rows can
    overstate true concurrency for hedged requests.
    """
    if not concurrency_usage:
        return "-"
    parts = []
    for name in sorted(concurrency_usage):
        usage = concurrency_usage[name]
        parts.append(f"{name}:{usage.busy_time_pct:.0f}%")
    return " ".join(parts)


def fmt_counts(counts: dict[str, int], max_entries: int = 3) -> str:
    """Compact ``key:n`` rendering. Sorted by count, descending."""
    if not counts:
        return "-"
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    parts = [f"{k}:{v}" for k, v in items[:max_entries]]
    if len(items) > max_entries:
        parts.append(f"+{len(items) - max_entries}")
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


def load_inventory_specs(
    policy_dir: Path,
) -> tuple[Path | None, list[QuotaSpec], list[ConcurrencySpec]]:
    """Read the inventory file referenced by this policy's ``args.json``.

    Returns ``(inventory_path, quota_specs, concurrency_specs)``. Returns
    ``(None, [], [])`` when the args/inventory aren't readable yet.
    """
    args_path = policy_dir / "args.json"
    if not args_path.exists():
        return None, [], []
    try:
        with args_path.open() as f:
            args = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None, [], []
    inv_path_raw = args.get("inventory")
    if not inv_path_raw or inv_path_raw == "None":
        return None, [], []
    # ``args.json`` records the inventory path as it was passed on the
    # command line (typically repo-relative). Resolve it against the
    # workspace root, but also try the snapshot directory in case a
    # snapshot was archived without the inventory file copied alongside.
    candidates = [Path(inv_path_raw), policy_dir.parent / inv_path_raw, policy_dir / inv_path_raw]
    inv_path: Path | None = None
    for cand in candidates:
        if cand.is_file():
            inv_path = cand
            break
    if inv_path is None:
        return None, [], []
    try:
        with inv_path.open() as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return inv_path, [], []
    quota_specs: list[QuotaSpec] = []
    concurrency_specs: list[ConcurrencySpec] = []
    for entry in raw.get("providers", []):
        qr = entry.get("quota_requests")
        qw = entry.get("quota_window_sec")
        if qr is not None and qw is not None:
            try:
                quota_specs.append(
                    QuotaSpec(
                        name=str(entry["name"]),
                        window_name=str(entry.get("quota_window_name") or "window"),
                        quota_requests=int(qr),
                        quota_window_sec=float(qw),
                    )
                )
            except (KeyError, TypeError, ValueError):
                pass
        elif entry.get("subscription_plan") is not None:
            try:
                subscription_count = int(entry.get("subscription_count", 1))
                plan = load_subscription_plans()[str(entry["subscription_plan"])]
                for window in plan.quota_windows:
                    quota_specs.append(
                        QuotaSpec(
                            name=str(entry["name"]),
                            window_name=str(window.name),
                            quota_requests=int(window.quota_requests) * subscription_count,
                            quota_window_sec=float(window.quota_window_sec),
                        )
                    )
            except (KeyError, TypeError, ValueError, OSError):
                pass
        limit = entry.get("concurrency_limit")
        if limit is not None:
            try:
                concurrency_specs.append(
                    ConcurrencySpec(
                        name=str(entry["name"]),
                        concurrency_limit=int(limit),
                    )
                )
            except (KeyError, TypeError, ValueError):
                pass
    return inv_path, quota_specs, concurrency_specs


def load_quota_specs(policy_dir: Path) -> tuple[Path | None, list[QuotaSpec]]:
    """Backward-compatible helper for callers that only need quota specs."""
    inv_path, quota_specs, _ = load_inventory_specs(policy_dir)
    return inv_path, quota_specs


def compute_quota_usage(
    quota_specs: list[QuotaSpec],
    dispatch_ts_by_provider: dict[str, list[float]],
    now: float,
    latest_activity_ts: float | None = None,
) -> dict[str, dict[str, QuotaUsage]]:
    """Per-quota-provider ``(used_in_current_window, limit)`` view.

    Counts dispatches that fall inside a trailing window of length
    ``quota_window_sec`` ending at the reference time. For a *live* run we
    use ``now`` as the reference; for a *finished* snapshot we use the
    latest dispatch timestamp so the column shows the peak window load
    rather than 0 (the window the policy would currently be allowed to
    use again). The displayed ``used`` is always within ``[0, limit]`` for
    a correctly-enforcing runner.
    """
    out = empty_quota_usage(quota_specs)
    # Anchor the window to the latest observed activity when the run is no
    # longer dispatching. Using ``now`` directly would silently drop all
    # historical usage once a full ``quota_window_sec`` had elapsed since
    # the last dispatch, even though that usage really did count against
    # the policy's quota counter while the run was active.
    reference_ts = now
    if latest_activity_ts is not None and now - latest_activity_ts > 60.0:
        reference_ts = latest_activity_ts
    for spec in quota_specs:
        timestamps = dispatch_ts_by_provider.get(spec.name, [])
        if not timestamps:
            continue
        cutoff = reference_ts - spec.quota_window_sec
        used = sum(1 for ts in timestamps if ts > cutoff)
        out.setdefault(spec.name, {})[spec.window_name] = QuotaUsage(
            used=used,
            limit=spec.quota_requests,
            window_sec=spec.quota_window_sec,
        )
    return out


def empty_quota_usage(
    quota_specs: list[QuotaSpec],
) -> dict[str, dict[str, QuotaUsage]]:
    """Return a zero-usage quota map with every configured window present."""
    out: dict[str, dict[str, QuotaUsage]] = {}
    for spec in quota_specs:
        out.setdefault(spec.name, {})[spec.window_name] = QuotaUsage(
            used=0,
            limit=spec.quota_requests,
            window_sec=spec.quota_window_sec,
        )
    return out


def compute_concurrency_usage(
    concurrency_specs: list[ConcurrencySpec],
    intervals_by_provider: dict[str, list[tuple[float, float]]],
    *,
    window_start: float | None,
    window_end: float | None,
) -> dict[str, ConcurrencyUsage]:
    """Compute observed concurrency busy-time from completed intervals."""
    out: dict[str, ConcurrencyUsage] = {}
    span = (
        max(float(window_end) - float(window_start), 0.0)
        if window_start is not None and window_end is not None
        else 0.0
    )
    for spec in concurrency_specs:
        limit = max(int(spec.concurrency_limit), 0)
        intervals = [
            (max(start, window_start), min(end, window_end))
            for start, end in intervals_by_provider.get(spec.name, [])
            if window_start is not None
            and window_end is not None
            and end > start
            and end > window_start
            and start < window_end
        ]
        busy_sec = union_interval_duration(intervals)
        busy_pct = (busy_sec / span * 100.0) if span > 0 else 0.0
        out[spec.name] = ConcurrencyUsage(
            busy_time_pct=busy_pct,
            peak_used=peak_concurrency(intervals),
            limit=limit,
            requests=len(intervals),
        )
    return out


def union_interval_duration(intervals: list[tuple[float, float]]) -> float:
    """Return total wall time covered by at least one interval."""
    valid = sorted((start, end) for start, end in intervals if end > start)
    if not valid:
        return 0.0
    total = 0.0
    current_start, current_end = valid[0]
    for start, end in valid[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    total += current_end - current_start
    return total


def peak_concurrency(intervals: list[tuple[float, float]]) -> int:
    """Return max overlap count for half-open intervals [start, end)."""
    events: list[tuple[float, int]] = []
    for start, end in intervals:
        if end <= start:
            continue
        events.append((start, 1))
        events.append((end, -1))
    current = 0
    peak = 0
    # End events sort before start events at the same timestamp, so adjacent
    # intervals do not count as overlapping.
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        current += delta
        peak = max(peak, current)
    return peak


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


def build_profile_events_table(
    base_dir: Path,
    *,
    slo_ms: float,
    now: float,
) -> tuple[str, list[dict]]:
    """Build a table from ``shared_profile_events.jsonl``.

    Unlike the per-provider request table, this reflects the shared latency
    evidence available to routing policies: warmup samples, sidecar probes,
    and natural request feedback. It is therefore useful when a provider is
    probed but never selected for real trace traffic.
    """
    path = base_dir / "shared_profile_events.jsonl"
    headers = [
        "provider",
        "events",
        "fail",
        "mean",
        "p50",
        "p90",
        "p95",
        "p99",
        "SLO viol",
        "sources",
        "last",
    ]
    if not path.exists():
        return (
            "profile TTFT (shared warmup/probe/natural events):\n"
            f"no shared profile event log found at {path}",
            [],
        )

    samples_by_provider: dict[str, list[float]] = {}
    failures_by_provider: Counter[str] = Counter()
    sources_by_provider: dict[str, Counter[str]] = {}
    last_ts_by_provider: dict[str, float] = {}
    last_source_by_provider: dict[str, str] = {}

    try:
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                provider = str(event.get("provider") or "unknown")
                source = str(event.get("source") or "unknown")
                sources_by_provider.setdefault(provider, Counter())[source] += 1

                ts = safe_float(str(event.get("ts", "")))
                if ts is not None and ts >= last_ts_by_provider.get(provider, -math.inf):
                    last_ts_by_provider[provider] = ts
                    last_source_by_provider[provider] = source

                error_type = event.get("error_type")
                ttft_ms = safe_float(str(event.get("ttft_ms", "")))
                if error_type not in (None, "") or ttft_ms is None or ttft_ms <= 0:
                    failures_by_provider[provider] += 1
                    continue
                samples_by_provider.setdefault(provider, []).append(ttft_ms)
    except OSError as exc:
        return (
            f"profile TTFT (shared warmup/probe/natural events):\ncould not read {path}: {exc}",
            [],
        )

    records: list[ProfileProviderStats] = []
    all_providers = set(samples_by_provider) | set(failures_by_provider) | set(sources_by_provider)
    for provider in sorted(all_providers):
        samples = sorted(samples_by_provider.get(provider, []))
        failures = failures_by_provider.get(provider, 0)
        total = len(samples) + failures
        if samples:
            mean_v = statistics.fmean(samples)
            p50 = percentile(samples, 50)
            p90 = percentile(samples, 90)
            p95 = percentile(samples, 95)
            p99 = percentile(samples, 99)
            slo_violations = sum(1 for v in samples if v > slo_ms) + failures
            slo_v_pct = (slo_violations / total * 100.0) if total else 0.0
        else:
            mean_v = p50 = p90 = p95 = p99 = float("nan")
            slo_v_pct = 100.0 if total else 0.0
        records.append(
            ProfileProviderStats(
                provider=provider,
                successes=len(samples),
                failures=failures,
                latency_mean_ms=mean_v,
                latency_p50_ms=p50,
                latency_p90_ms=p90,
                latency_p95_ms=p95,
                latency_p99_ms=p99,
                slo_violation_pct=slo_v_pct,
                source_counts=dict(sources_by_provider.get(provider, Counter())),
                last_ts=last_ts_by_provider.get(provider),
                last_source=last_source_by_provider.get(provider),
            )
        )

    records.sort(
        key=lambda record: (
            -(record.successes + record.failures),
            record.provider,
        )
    )

    rows: list[list[str]] = []
    json_records: list[dict] = []
    for record in records:
        total = record.successes + record.failures
        age = (
            f"{max(now - record.last_ts, 0.0):.0f}s:{record.last_source}"
            if record.last_ts is not None
            else "-"
        )
        rows.append(
            [
                record.provider,
                str(total),
                str(record.failures),
                fmt_ms(record.latency_mean_ms),
                fmt_ms(record.latency_p50_ms),
                fmt_ms(record.latency_p90_ms),
                fmt_ms(record.latency_p95_ms),
                fmt_ms(record.latency_p99_ms),
                fmt_pct(record.slo_violation_pct),
                fmt_counts(record.source_counts),
                age,
            ]
        )
        json_records.append(
            {
                "provider": record.provider,
                "events": total,
                "successes": record.successes,
                "failures": record.failures,
                "ttft_mean_ms": (
                    None if math.isnan(record.latency_mean_ms) else record.latency_mean_ms
                ),
                "ttft_p50_ms": (
                    None if math.isnan(record.latency_p50_ms) else record.latency_p50_ms
                ),
                "ttft_p90_ms": (
                    None if math.isnan(record.latency_p90_ms) else record.latency_p90_ms
                ),
                "ttft_p95_ms": (
                    None if math.isnan(record.latency_p95_ms) else record.latency_p95_ms
                ),
                "ttft_p99_ms": (
                    None if math.isnan(record.latency_p99_ms) else record.latency_p99_ms
                ),
                "slo_violation_pct": record.slo_violation_pct,
                "source_counts": record.source_counts,
                "last_ts": record.last_ts,
                "last_source": record.last_source,
            }
        )

    table = "profile TTFT (shared warmup/probe/natural events):\n" + render_table(rows, headers)
    return table, json_records


def render_snapshot(base_dir: Path) -> tuple[str, dict[str, Any]]:
    now = time.time()
    run_env = load_run_env(base_dir)
    default_slo_ms = float(run_env.get("SLO_MS") or 3000.0)
    duration_cap_sec = float(run_env["DURATION_SEC"]) if run_env.get("DURATION_SEC") else None

    policy_dirs = discover_policies(base_dir)
    if not policy_dirs:
        msg = f"no policy subdirectories found under {base_dir}"
        return msg, {
            "snapshot_ts": now,
            "output_dir": str(base_dir),
            "n_policies": 0,
            "message": msg,
        }

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
    # Cross-policy per-provider TTFT samples. Aggregated across every
    # policy so each provider's latency distribution is computed from
    # the largest available sample size.
    provider_ttft: dict[str, list[float]] = {}
    provider_failures: Counter[str] = Counter()
    for pdir in policy_dirs:
        slo_ms = load_slo_ms(pdir, default_slo_ms)
        _, quota_specs, concurrency_specs = load_inventory_specs(pdir)
        s = collect_policy(
            pdir,
            slo_ms=slo_ms,
            provider_ttft=provider_ttft,
            provider_failures=provider_failures,
            quota_specs=quota_specs,
            concurrency_specs=concurrency_specs,
            now=now,
        )
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
                    quota_usage=empty_quota_usage(quota_specs),
                    concurrency_usage={
                        spec.name: ConcurrencyUsage(
                            busy_time_pct=0.0,
                            peak_used=0,
                            limit=spec.concurrency_limit,
                            requests=0,
                        )
                        for spec in concurrency_specs
                    },
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
                s.latest_ts if latest_ts_global is None else max(latest_ts_global, s.latest_ts)
            )

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
        min(elapsed_sec / trace_time_sec * 100.0, 100.0) if trace_time_sec else float("nan")
    )

    header_lines = [
        f"snapshot @ {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}",
        f"output_dir       : {base_dir}",
        f"policies         : {n_policies}",
        f"trace dataset    : "
        + (
            f"{trace_total} requests over {trace_time_sec / 3600:.1f} hours"
            if trace_total and trace_time_sec
            else "unknown"
        ),
        f"wall-clock elapsed : {fmt_duration(elapsed_sec)}"
        + (f"  (cap {fmt_duration(duration_cap_sec)})" if duration_cap_sec else ""),
        f"trace replayed   : "
        + (
            f"{completion_time_pct:5.1f}% by trace time"
            if not math.isnan(completion_time_pct)
            else "n/a"
        )
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
        "quota left",
        "conc busy",
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
                fmt_quota_usage(s.quota_usage),
                fmt_concurrency_usage(s.concurrency_usage),
            ]
        )

    provider_rows, provider_table_data = build_provider_table(
        provider_ttft, provider_failures, slo_ms=default_slo_ms
    )
    provider_headers = [
        "provider",
        "reqs",
        "fail",
        "mean",
        "p50",
        "p90",
        "p95",
        "p99",
        "SLO viol",
    ]
    provider_section = "\n\nper-provider TTFT (aggregated across all policies):\n" + render_table(
        provider_rows, provider_headers
    )
    profile_section, profile_table_data = build_profile_events_table(
        base_dir,
        slo_ms=default_slo_ms,
        now=now,
    )

    policy_table_text = render_table(rows, headers)
    header_text = "\n".join(header_lines)
    table_text = header_text + "\n\n" + policy_table_text + provider_section
    return table_text, {
        "_sections": {
            "header": header_text,
            "policy_table": policy_table_text,
            "provider_table": provider_section.lstrip("\n"),
            "profile_table": profile_section,
        },
        "snapshot_ts": now,
        "snapshot_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        "output_dir": str(base_dir),
        "n_policies": n_policies,
        "trace_total_requests": trace_total,
        "trace_time_sec": trace_time_sec,
        "wall_elapsed_sec": elapsed_sec,
        "duration_cap_sec": duration_cap_sec,
        "completion_time_pct": completion_time_pct if not math.isnan(completion_time_pct) else None,
        "completion_count_pct": completion_count_pct
        if not math.isnan(completion_count_pct)
        else None,
        "slo_ms": default_slo_ms,
        "total_cost_usd": sum(s.total_cost_usd for s in stats),
        "policies": [_stats_to_dict(s) for s in stats],
        "providers_ttft": provider_table_data,
        "profile_events": profile_table_data,
    }


def build_provider_table(
    provider_ttft: dict[str, list[float]],
    provider_failures: Counter[str],
    slo_ms: float,
) -> tuple[list[list[str]], list[dict]]:
    """Build per-provider TTFT distribution rows.

    Returns ``(rendered_rows, json_records)``. Sorted by sample count
    descending so the providers with the most traffic appear first.
    """
    json_records: list[dict] = []
    rendered: list[list[str]] = []
    all_providers = set(provider_ttft) | set(provider_failures)
    sortable = sorted(
        all_providers,
        key=lambda p: (
            -len(provider_ttft.get(p, [])),
            -provider_failures.get(p, 0),
            p,
        ),
    )
    for prov in sortable:
        samples = sorted(provider_ttft.get(prov, []))
        fail = provider_failures.get(prov, 0)
        total = len(samples) + fail
        if not samples:
            mean_v = float("nan")
            p50 = p90 = p95 = p99 = float("nan")
            slo_v_pct = 100.0 if total else 0.0
        else:
            mean_v = statistics.fmean(samples)
            p50 = percentile(samples, 50)
            p90 = percentile(samples, 90)
            p95 = percentile(samples, 95)
            p99 = percentile(samples, 99)
            over = sum(1 for v in samples if v > slo_ms) + fail
            slo_v_pct = (over / total * 100.0) if total else 0.0
        json_records.append(
            {
                "provider": prov,
                "requests": total,
                "successes": len(samples),
                "failures": fail,
                "ttft_mean_ms": None if math.isnan(mean_v) else mean_v,
                "ttft_p50_ms": None if math.isnan(p50) else p50,
                "ttft_p90_ms": None if math.isnan(p90) else p90,
                "ttft_p95_ms": None if math.isnan(p95) else p95,
                "ttft_p99_ms": None if math.isnan(p99) else p99,
                "slo_violation_pct": slo_v_pct,
            }
        )
        rendered.append(
            [
                prov,
                str(total),
                str(fail),
                fmt_ms(mean_v),
                fmt_ms(p50),
                fmt_ms(p90),
                fmt_ms(p95),
                fmt_ms(p99),
                fmt_pct(slo_v_pct),
            ]
        )
    return rendered, json_records


def _stats_to_dict(s: PolicyStats) -> dict:
    d = asdict(s)
    for key in (
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p90_ms",
        "latency_p95_ms",
        "latency_p99_ms",
    ):
        if d[key] is not None and math.isnan(d[key]):
            d[key] = None
    # ``asdict`` turns tuples into lists, but downstream JSON consumers
    # benefit from named fields. Expand into ``{used, limit, left}`` per
    # quota provider so summary.json is self-describing.
    if s.quota_usage:
        d["quota_usage"] = {
            name: {
                window_name: {
                    "used": usage.used,
                    "limit": usage.limit,
                    "left": usage.left,
                    "window_sec": usage.window_sec,
                }
                for window_name, usage in windows.items()
            }
            for name, windows in s.quota_usage.items()
        }
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
    json_data = {k: v for k, v in data.items() if not k.startswith("_")}
    with (snap_dir / "summary.json").open("w") as f:
        json.dump(json_data, f, indent=2)

    policy_dirs = discover_policies(base_dir)
    for pdir in policy_dirs:
        dest = snap_dir / pdir.name
        dest.mkdir(exist_ok=True)
        for fname in ("requests.csv", "run.log", "args.json"):
            src = pdir / fname
            if src.exists():
                shutil.copy2(src, dest / fname)

    for extra in (
        "run_env.txt",
        "policies.txt",
        "initial_profile.json",
        "policy_key_assignments.tsv",
    ):
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
        "--providers-only",
        action="store_true",
        help="Print only the per-provider TTFT distribution table, skipping "
        "the per-policy summary.",
    )
    parser.add_argument(
        "--policies-only",
        action="store_true",
        help="Print only the per-policy summary table, skipping the "
        "per-provider TTFT distribution.",
    )
    parser.add_argument(
        "--profiles",
        action="store_true",
        help="Also print the shared profile event table (warmup/probe/natural latency samples).",
    )
    parser.add_argument(
        "--profiles-only",
        action="store_true",
        help="Print only the shared profile event table, skipping request summaries.",
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

    exclusive = [args.providers_only, args.policies_only, args.profiles_only]
    if sum(bool(v) for v in exclusive) > 1:
        print(
            "error: --providers-only, --policies-only, and --profiles-only are mutually exclusive",
            file=sys.stderr,
        )
        return 2

    def render_for_display() -> str:
        text, data = render_snapshot(base)
        sections = data.get("_sections", {})
        if args.profiles_only:
            return sections.get("header", "") + "\n\n" + sections.get("profile_table", "")
        if args.providers_only:
            return sections.get("header", "") + "\n\n" + sections.get("provider_table", "")
        if args.policies_only:
            return sections.get("header", "") + "\n\n" + sections.get("policy_table", "")
        if args.profiles:
            return text + "\n\n" + sections.get("profile_table", "")
        return text

    if args.snapshot:
        snap_dir = save_snapshot(base, label=args.snapshot_label)
        print(render_for_display())
        print(f"\nsnapshot saved -> {snap_dir}")
        return 0

    if args.watch <= 0:
        print(render_for_display())
        return 0
    try:
        while True:
            sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.write(render_for_display())
            sys.stdout.write("\n")
            sys.stdout.flush()
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
