"""Inventory + per-provider state for real online evaluation.

Defines the static specs (read from inventory JSON) and the dynamic state
(rolling latency profile, quota, concurrency) that policies operate over.

Migrated from
``NSDI2027_RouteWise/experiment/scripts/phase6_joint_online_evaluation.py``
lines 97-298. Real experiments share quota capacity primitives with
``llm_routewise.sim.world`` while keeping empirical rolling profiles and live-run
concurrency state local to this package.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from experiments.real_evaluation.transports import (
    TransportConfig,
    resolve_transport_config,
)
from experiments.subscriptions import load_subscription_plans, subscription_fixed_cost_usd
from llm_routewise.capacity import MultiWindowQuotaState, QuotaState
from llm_routewise.core.latency_profile import DEFAULT_PROFILE_WINDOW_SEC, RollingLatencyProfile

# Shared with the simulator policies; see llm_routewise.core.latency_profile.
PROFILE_WINDOW_SEC: float = DEFAULT_PROFILE_WINDOW_SEC


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
    fixed_cost_usd_override: float | None = None
    concurrency_limit: int | None = None

    @property
    def billing_mode(self) -> str:
        return self.transport_cfg.billing_mode

    @property
    def input_price_per_m(self) -> float:
        return self.transport_cfg.input_price_per_m

    @property
    def cached_input_price_per_m(self) -> float | None:
        return self.transport_cfg.cached_input_price_per_m

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
    openrouter_provider_only: tuple[str, ...] = ()
    openrouter_provider_ignore: tuple[str, ...] = ()
    openrouter_stream_cancel_billing_by_provider: dict[str, str] = field(default_factory=dict)
    billing_duration_sec: float | None = None


def load_inventory(path: Path | str) -> InventoryConfig:
    """Read an inventory JSON and return the parsed ``InventoryConfig``."""
    raw = json.loads(Path(path).read_text())
    openrouter_provider_only = _provider_filter(raw.get("openrouter_provider_only"))
    openrouter_provider_ignore = _provider_filter(raw.get("openrouter_provider_ignore"))
    provider_specs: list[ProviderSpec] = []
    for entry in raw["providers"]:
        if _skip_openrouter_provider_entry(
            entry,
            provider_only=openrouter_provider_only,
            provider_ignore=openrouter_provider_ignore,
        ):
            continue
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
                fixed_cost_usd_override=(
                    float(entry["fixed_cost_usd_override"])
                    if entry.get("fixed_cost_usd_override") is not None
                    else None
                ),
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
        openrouter_provider_only=openrouter_provider_only,
        openrouter_provider_ignore=openrouter_provider_ignore,
        openrouter_stream_cancel_billing_by_provider=(
            _openrouter_stream_cancel_billing_by_provider(provider_specs)
        ),
        billing_duration_sec=(
            float(raw["billing_duration_sec"])
            if raw.get("billing_duration_sec") is not None
            else None
        ),
    )


def _openrouter_stream_cancel_billing_by_provider(
    specs: list[ProviderSpec],
) -> dict[str, str]:
    out: dict[str, str] = {}
    for spec in specs:
        cfg = spec.transport_cfg
        if cfg.transport != "openrouter" or cfg.provider_hint is None:
            continue
        if cfg.stream_cancel_billing is None:
            continue
        out[cfg.provider_hint.strip().lower()] = cfg.stream_cancel_billing
    return out


def _provider_filter(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        raise ValueError(f"OpenRouter provider filter must be a string or list: {value!r}")
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        cleaned = str(item).strip()
        if cleaned and cleaned not in seen:
            out.append(cleaned)
            seen.add(cleaned)
    return tuple(out)


def _skip_openrouter_provider_entry(
    entry: dict,
    *,
    provider_only: tuple[str, ...],
    provider_ignore: tuple[str, ...],
) -> bool:
    if entry.get("transport") != "openrouter":
        return False
    provider_name = str(entry.get("provider_hint") or "").strip()
    if not provider_name:
        return False
    if provider_only and provider_name not in provider_only:
        return True
    return provider_name in provider_ignore


def _quota_windows_from_entry(
    entry: dict,
    *,
    subscription_count: int,
) -> tuple[ProviderQuotaWindow, ...]:
    plan_id = entry.get("subscription_plan")
    if plan_id is None:
        return ()
    if entry.get("quota_window_sec") is not None or entry.get("quota_requests") is not None:
        # Live runs sometimes use OpenRouter to physically emulate a
        # subscription tier while scaling quota to the experiment window.
        # Keep subscription_plan for fixed-cost accounting, but honor the
        # explicit quota fields for capacity.
        return ()
    if subscription_count <= 0:
        raise ValueError(f"provider {entry.get('name')!r}: subscription_count must be > 0")
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


def subscription_fixed_cost_for_inventory(
    inventory: InventoryConfig,
    *,
    billing_duration_sec: float,
) -> float:
    """Return prorated fixed subscription cost for one real-eval policy run."""
    if billing_duration_sec <= 0.0:
        return 0.0
    plans = load_subscription_plans()
    total = 0.0
    for spec in inventory.providers:
        if spec.billing_mode != "subscription" or spec.subscription_plan is None:
            continue
        if spec.fixed_cost_usd_override is not None:
            if spec.fixed_cost_usd_override < 0.0:
                raise ValueError(
                    f"provider {spec.name!r}: fixed_cost_usd_override must be >= 0"
                )
            total += float(spec.fixed_cost_usd_override)
            continue
        plan = plans[str(spec.subscription_plan)]
        total += subscription_fixed_cost_usd(
            plan,
            subscription_count=spec.subscription_count,
            trace_days=billing_duration_sec / 86400.0,
        )
    return total


@dataclass
class ConcurrencyState:
    """Active-request tracker for concurrency-limited providers.

    Tracks ``(request_id, deadline_ts)`` pairs. A deadline of ``inf`` means
    the caller must explicitly release the slot when the real request finishes.
    Finite deadlines are still supported for tests and legacy callers.
    """

    limit: int
    active: list[tuple[int, float]] = field(default_factory=list)

    def _prune(self, now: float) -> None:
        self.active = [(rid, deadline) for rid, deadline in self.active if deadline > now]

    def is_available(self, now: float) -> bool:
        self._prune(now)
        return len(self.active) < self.limit

    def utilization(self, now: float) -> float:
        self._prune(now)
        if self.limit <= 0:
            return 0.0
        return float(min(len(self.active) / self.limit, 0.9999))

    def admit(
        self,
        request_id: int,
        now: float,
        expected_hold_sec: float | None = None,
    ) -> int:
        self._prune(now)
        deadline = float("inf")
        if expected_hold_sec is not None:
            deadline = now + max(0.1, expected_hold_sec)
        self.active.append((request_id, deadline))
        return request_id

    def release(self, request_id: int, now: float | None = None) -> None:
        if now is not None:
            self._prune(now)
        self.active = [(rid, deadline) for rid, deadline in self.active if rid != request_id]


class LatencyProfile(RollingLatencyProfile):
    """Real-eval profile API over the shared core rolling estimator.

    ``add_sample`` feeds the core :class:`RollingLatencyProfile` machinery so
    the RouteWise router's latency beliefs learn on the exact profile objects
    held by :class:`ProviderState`. The historical query methods below keep
    their pre-router semantics verbatim — in particular they include samples
    with ``ts > now`` (the core queries are strictly causal). Live feedback
    always lands in the past, so both readings agree in production; the
    distinction only shows up for synthetic timestamps in tests and probes.

    Not thread-safe; every caller goes through the owning policy's lock.
    """

    def add_sample(
        self,
        ts: float,
        ttft_ms: float,
        error_type: str | None = None,
    ) -> None:
        if error_type is None and ttft_ms > 0:
            super().add_sample(ts, float(ttft_ms))
        elif error_type is not None:
            self.add_error(ts, error_type)

    def _active_samples(self, now: float) -> list[float]:
        cutoff = now - self.window_sec
        self.samples = deque((ts, v) for ts, v in self.samples if ts >= cutoff)
        self.error_samples = deque(
            (ts, error_type) for ts, error_type in self.error_samples if ts >= cutoff
        )
        return [v for _, v in self.samples]

    def sample_count(self, now: float) -> int:
        return len(self._active_samples(now))

    def total_count(self, now: float) -> int:
        values = self._active_samples(now)
        return len(values) + len(self.error_samples)

    def mean_ms(self, now: float) -> float | None:
        values = self._active_samples(now)
        if not values:
            return None
        return float(np.mean(values))

    def mean_with_errors_ms(self, now: float, *, error_penalty_ms: float) -> float | None:
        """Return mean over successes plus synthetic latency for failed attempts."""
        values = self._active_samples(now)
        n_errors = len(self.error_samples)
        n_total = len(values) + n_errors
        if n_total == 0:
            return None
        return float((sum(values) + n_errors * float(error_penalty_ms)) / n_total)

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
    def from_spec(cls, spec: ProviderSpec, window_sec: float = PROFILE_WINDOW_SEC) -> ProviderState:
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
            QuotaState(size=window.requests, window_sec=window.window_sec)
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
        if self.quota is not None and not self.quota.can_admit(now):
            return False
        return not (self.concurrency is not None and not self.concurrency.is_available(now))


def build_provider_states(
    inventory: InventoryConfig,
    profile_window_sec: float = PROFILE_WINDOW_SEC,
) -> dict[str, ProviderState]:
    """Materialize one isolated ``ProviderState`` per spec from an inventory."""
    return {
        spec.name: ProviderState.from_spec(spec, profile_window_sec) for spec in inventory.providers
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
    "subscription_fixed_cost_for_inventory",
]
