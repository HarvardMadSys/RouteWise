"""Inventory + per-provider state for real online evaluation.

Defines the static specs (read from inventory JSON) and the dynamic state
(rolling latency profile, quota, concurrency) that policies operate over.

Migrated from
``NSDI2027_RouteWise/experiment/scripts/phase6_joint_online_evaluation.py``
lines 97-298. Data structures are intentionally simpler than ``rwsim.world``
counterparts: real experiments only need empirical rolling profiles and
advisory quota/concurrency tracking, not interval-aware admission.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from experiments.real_evaluation.transports import (
    TransportConfig,
    resolve_transport_config,
)
from experiments.subscriptions import load_subscription_plans

PROFILE_WINDOW_SEC: float = 15 * 60.0


@dataclass(frozen=True)
class ProviderQuotaWindow:
    """One quota window materialized from a subscription plan or inventory."""

    window_sec: float
    requests: int


@dataclass
class ProviderSpec:
    """Static config for one provider (read from inventory JSON)."""

    name: str
    tier: str  # "api" | "quota" | "concurrency"
    transport_cfg: TransportConfig
    quota_window_sec: float | None = None
    quota_requests: int | None = None
    quota_windows: tuple[ProviderQuotaWindow, ...] = ()
    subscription_plan: str | None = None
    subscription_count: int = 1
    concurrency_limit: int | None = None

    @property
    def input_price_per_m(self) -> float:
        return self.transport_cfg.input_price_per_m

    @property
    def output_price_per_m(self) -> float:
        return self.transport_cfg.output_price_per_m


@dataclass
class InventoryConfig:
    """Top-level inventory: model family + SLO + provider list."""

    model_family: str
    openrouter_model_id: str
    primary_slo_ms: float
    slo_thresholds_ms: list[float]
    providers: list[ProviderSpec]


def load_inventory(path: Path | str) -> InventoryConfig:
    """Read an inventory JSON and return the parsed ``InventoryConfig``."""
    raw = json.loads(Path(path).read_text())
    provider_specs: list[ProviderSpec] = []
    for entry in raw["providers"]:
        transport_cfg = resolve_transport_config(entry)
        plan_id = entry.get("subscription_plan")
        subscription_count = int(entry.get("subscription_count", 1))
        quota_windows = _quota_windows_from_entry(
            entry,
            subscription_count=subscription_count,
        )
        first_window = quota_windows[0] if quota_windows else None
        provider_specs.append(
            ProviderSpec(
                name=entry["name"],
                tier=entry["tier"],
                transport_cfg=transport_cfg,
                quota_window_sec=(
                    first_window.window_sec
                    if first_window is not None
                    else float(entry["quota_window_sec"])
                    if entry.get("quota_window_sec") is not None
                    else None
                ),
                quota_requests=(
                    first_window.requests
                    if first_window is not None
                    else int(entry["quota_requests"])
                    if entry.get("quota_requests") is not None
                    else None
                ),
                quota_windows=quota_windows,
                subscription_plan=str(plan_id) if plan_id is not None else None,
                subscription_count=subscription_count,
                concurrency_limit=(
                    int(entry["concurrency_limit"])
                    if entry.get("concurrency_limit") is not None
                    else None
                ),
            )
        )
    return InventoryConfig(
        model_family=raw["model_family"],
        openrouter_model_id=raw["openrouter_model_id"],
        primary_slo_ms=float(raw["primary_slo_ms"]),
        slo_thresholds_ms=[float(t) for t in raw["slo_thresholds_ms"]],
        providers=provider_specs,
    )


def _quota_windows_from_entry(
    entry: dict,
    *,
    subscription_count: int,
) -> tuple[ProviderQuotaWindow, ...]:
    plan_id = entry.get("subscription_plan")
    if plan_id is None:
        return ()
    if entry.get("quota_window_sec") is not None or entry.get("quota_requests") is not None:
        raise ValueError(
            f"provider {entry.get('name')!r} uses subscription_plan={plan_id!r}; "
            "quota_window_sec/quota_requests must come from experiments/subscription_plans.yaml"
        )
    if subscription_count <= 0:
        raise ValueError(
            f"provider {entry.get('name')!r}: subscription_count must be > 0"
        )
    plans = load_subscription_plans()
    try:
        plan = plans[str(plan_id)]
    except KeyError as exc:
        known = ", ".join(sorted(plans))
        raise ValueError(
            f"provider {entry.get('name')!r}: unknown subscription_plan "
            f"{plan_id!r}; known plans: {known}"
        ) from exc
    return tuple(
        ProviderQuotaWindow(
            window_sec=float(window.quota_window_sec),
            requests=int(window.quota_requests) * subscription_count,
        )
        for window in plan.quota_windows
    )


@dataclass
class QuotaState:
    """Rolling-window quota counter (advisory; physical quotas live at provider).

    Mirrors ``rwsim.world.capacity.QuotaState`` semantics with a simpler
    interface — we only need ``charge`` / ``can_admit`` / ``fraction_used``.
    """

    window_sec: float
    limit: int
    window_start: float = 0.0
    used: int = 0

    def tick(self, now: float) -> None:
        # Fixed-boundary semantics: ``window_start`` is the largest multiple
        # of ``window_sec`` that is <= ``now``. The default ``window_start = 0.0``
        # means the very first call snaps to ``floor(now / window_sec) * window_sec``,
        # so a 1-hour quota is always anchored to the top of the hour
        # (epoch-aligned), regardless of when the first request arrives.
        # Earlier "anchor at first request" semantics drifted: a quota
        # configured for [t0, t0+3600) would extend until 2*t0+3600 once
        # the second window opened. Matches :class:`rwsim.world.capacity.QuotaState`.
        if self.window_sec <= 0:
            self.window_start = now
            self.used = 0
            return
        elapsed = now - self.window_start
        if elapsed >= self.window_sec:
            n_windows = int(elapsed // self.window_sec)
            self.window_start += n_windows * self.window_sec
            self.used = 0

    def is_available(self, now: float) -> bool:
        self.tick(now)
        return self.used < self.limit

    def fraction_used(self, now: float) -> float:
        self.tick(now)
        if self.limit <= 0:
            return 0.0
        return float(min(self.used / self.limit, 0.9999))

    def charge(self, now: float) -> None:
        self.tick(now)
        self.used += 1


@dataclass
class MultiWindowQuotaState:
    """Quota tracker for plans with simultaneous reset windows."""

    windows: tuple[QuotaState, ...]

    def is_available(self, now: float) -> bool:
        return all(window.is_available(now) for window in self.windows)

    def fraction_used(self, now: float) -> float:
        if not self.windows:
            return 0.0
        return max(window.fraction_used(now) for window in self.windows)

    def charge(self, now: float) -> None:
        for window in self.windows:
            window.charge(now)


@dataclass
class ConcurrencyState:
    """Active-request tracker for concurrency-limited providers.

    Tracks ``(request_id, expected_finish_ts)`` pairs. ``_prune`` evicts
    finished entries lazily on every read.
    """

    limit: int
    active: list[tuple[int, float]] = field(default_factory=list)

    def _prune(self, now: float) -> None:
        self.active = [
            (rid, deadline) for rid, deadline in self.active if deadline > now
        ]

    def is_available(self, now: float) -> bool:
        self._prune(now)
        return len(self.active) < self.limit

    def utilization(self, now: float) -> float:
        self._prune(now)
        if self.limit <= 0:
            return 0.0
        return float(min(len(self.active) / self.limit, 0.9999))

    def admit(self, request_id: int, now: float, expected_hold_sec: float) -> None:
        self._prune(now)
        self.active.append((request_id, now + max(0.1, expected_hold_sec)))


@dataclass
class LatencyProfile:
    """Rolling window of ``(timestamp, ttft_ms)`` with optional error tracking.

    Supports the empirical-CDF based hedge math (``cdf_at``) and the LP body
    selector (``mean_ms``). Errors are tracked separately so we can compute
    rates and treat them as misses in CDF.
    """

    window_sec: float
    samples: list[tuple[float, float]] = field(default_factory=list)
    error_samples: list[tuple[float, str]] = field(default_factory=list)

    def add_sample(
        self,
        ts: float,
        ttft_ms: float,
        error_type: str | None = None,
    ) -> None:
        if error_type is None and ttft_ms > 0:
            self.samples.append((ts, float(ttft_ms)))
        elif error_type is not None:
            self.error_samples.append((ts, error_type))

    def _active_samples(self, now: float) -> list[float]:
        cutoff = now - self.window_sec
        self.samples = [(ts, v) for ts, v in self.samples if ts >= cutoff]
        self.error_samples = [(ts, e) for ts, e in self.error_samples if ts >= cutoff]
        return [v for _, v in self.samples]

    def sample_count(self, now: float) -> int:
        return len(self._active_samples(now))

    def mean_ms(self, now: float) -> float | None:
        values = self._active_samples(now)
        if not values:
            return None
        return float(np.mean(values))

    def median_ms(self, now: float) -> float | None:
        values = self._active_samples(now)
        if not values:
            return None
        return float(np.percentile(values, 50))

    def percentile_ms(self, now: float, p: float) -> float | None:
        values = self._active_samples(now)
        if not values:
            return None
        return float(np.percentile(values, p))

    def cdf_at(self, threshold_ms: float, now: float) -> float:
        """Return ``P(TTFT <= threshold_ms)`` treating errors as misses."""
        values = self._active_samples(now)
        n_errors = len(self.error_samples)
        n_total = len(values) + n_errors
        if n_total == 0:
            return 0.0
        n_under = int(sum(1 for v in values if v <= threshold_ms))
        return float(n_under / n_total)

    def error_rate(self, now: float) -> float:
        values = self._active_samples(now)
        n_errors = len(self.error_samples)
        n_total = len(values) + n_errors
        if n_total == 0:
            return 0.0
        return float(n_errors / n_total)


@dataclass
class ProviderState:
    """Per-policy dynamic state for one provider.

    Each policy owns an isolated ``ProviderState`` so its routing decisions
    are not influenced by other policies sharing the harness. The underlying
    backend quotas / concurrency slots are physically shared, but at pilot
    trace rates the aggregate load stays under the subscription budget.
    """

    spec: ProviderSpec
    profile: LatencyProfile
    quota: QuotaState | MultiWindowQuotaState | None
    concurrency: ConcurrencyState | None

    @classmethod
    def from_spec(
        cls, spec: ProviderSpec, window_sec: float = PROFILE_WINDOW_SEC
    ) -> ProviderState:
        quota_windows = spec.quota_windows
        if not quota_windows and (
            spec.quota_requests is not None and spec.quota_window_sec is not None
        ):
            quota_windows = (
                ProviderQuotaWindow(
                    window_sec=spec.quota_window_sec,
                    requests=spec.quota_requests,
                ),
            )
        quota_states = tuple(
            QuotaState(window_sec=window.window_sec, limit=window.requests)
            for window in quota_windows
        )
        if len(quota_states) == 1:
            quota = quota_states[0]
        elif quota_states:
            quota = MultiWindowQuotaState(quota_states)
        else:
            quota = None
        concurrency = (
            ConcurrencyState(limit=spec.concurrency_limit)
            if spec.concurrency_limit is not None
            else None
        )
        return cls(
            spec=spec,
            profile=LatencyProfile(window_sec=window_sec),
            quota=quota,
            concurrency=concurrency,
        )

    def is_available(self, now: float) -> bool:
        if self.quota is not None and not self.quota.is_available(now):
            return False
        return not (
            self.concurrency is not None
            and not self.concurrency.is_available(now)
        )


def build_provider_states(
    inventory: InventoryConfig,
    profile_window_sec: float = PROFILE_WINDOW_SEC,
) -> dict[str, ProviderState]:
    """Materialize one isolated ``ProviderState`` per spec from an inventory."""
    return {
        spec.name: ProviderState.from_spec(spec, profile_window_sec)
        for spec in inventory.providers
    }


__all__ = [
    "PROFILE_WINDOW_SEC",
    "ConcurrencyState",
    "InventoryConfig",
    "LatencyProfile",
    "MultiWindowQuotaState",
    "ProviderQuotaWindow",
    "ProviderSpec",
    "ProviderState",
    "QuotaState",
    "build_provider_states",
    "load_inventory",
]
