"""Public, dependency-free RouteWise facade for API providers.

This module owns lifecycle, learning, and accounting while the algorithmic
primitives in :mod:`llm_routewise.core` remain pure. The ``0.2.0`` facade
supports metered API-provider prices only.
"""

from __future__ import annotations

import math
import random
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from threading import RLock
from types import MappingProxyType
from typing import Any

from llm_routewise._capacity_controller import (
    _NoopCapacityController,
    _Reservation,
)
from llm_routewise._output_length import _OutputLengthEstimator
from llm_routewise.const import DISPATCH_OVERHEAD_MS
from llm_routewise.core.hedging import (
    BackupCandidate,
    combined_success_probability,
    hedge_checkpoints_for_slo,
    select_probability_backup,
)
from llm_routewise.core.latency_profile import RollingLatencyProfile
from llm_routewise.core.lp import (
    BudgetLPCandidate,
    cost_tiebroken_objective,
    solve_budget_lp,
)
from llm_routewise.errors import NoProviderError, OutcomeError, ValidationError

_KNOWN_ERROR_CODES = frozenset(
    {
        "rate_limited",
        "timeout",
        "server_error",
        "connection",
        "bad_request",
        "auth",
        "unsupported",
    }
)
_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled", "declined"})
_UNKNOWN_LATENCY_MS = 1_000_000_000.0


def _real(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a real number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValidationError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValidationError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValidationError(f"{name} must be positive")
    if not positive and result < 0.0:
        raise ValidationError(f"{name} must be non-negative")
    return result


def _token_count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValidationError(f"{name} must be a non-negative integer")
    return value


def _alpha(value: object, name: str = "alpha") -> float:
    result = _real(value, name)
    if result > 1.0:
        raise ValidationError(f"{name} must be within [0, 1]")
    return result


def _clock_seconds(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError("clock must return finite numeric seconds")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError("clock must return finite numeric seconds")
    return result


def _freeze(value: Any) -> Any:
    """Return a recursively immutable snapshot suitable for public output."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True)
class Provider:
    """One API provider and its per-million-token prices."""

    name: str
    price_in: float
    price_out: float
    price_cached: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValidationError("provider name must be a non-empty string")
        object.__setattr__(self, "price_in", _real(self.price_in, "price_in"))
        object.__setattr__(self, "price_out", _real(self.price_out, "price_out"))
        if self.price_cached is not None:
            object.__setattr__(
                self,
                "price_cached",
                _real(self.price_cached, "price_cached"),
            )


@dataclass(frozen=True, kw_only=True)
class Tuning:
    """Calibrated algorithm constants for callers that need an escape hatch."""

    hedge_target: float = 0.99
    penalty_ms: float = 60_000.0
    window_min: float = 15.0
    cooldown_sec: float = 30.0
    cooldown_after: int = 3
    hedge_min_samples: int = 5
    exploration_lease_sec: float = 60.0

    def __post_init__(self) -> None:
        target = _real(self.hedge_target, "hedge_target", positive=True)
        if target > 1.0:
            raise ValidationError("hedge_target must be within (0, 1]")
        object.__setattr__(self, "hedge_target", target)
        object.__setattr__(
            self,
            "penalty_ms",
            _real(self.penalty_ms, "penalty_ms", positive=True),
        )
        object.__setattr__(
            self,
            "window_min",
            _real(self.window_min, "window_min", positive=True),
        )
        object.__setattr__(self, "cooldown_sec", _real(self.cooldown_sec, "cooldown_sec"))
        if isinstance(self.cooldown_after, bool) or not isinstance(self.cooldown_after, int):
            raise ValidationError("cooldown_after must be a positive integer")
        if self.cooldown_after < 1:
            raise ValidationError("cooldown_after must be a positive integer")
        if isinstance(self.hedge_min_samples, bool) or not isinstance(self.hedge_min_samples, int):
            raise ValidationError("hedge_min_samples must be a positive integer")
        if self.hedge_min_samples < 1:
            raise ValidationError("hedge_min_samples must be a positive integer")
        object.__setattr__(
            self,
            "exploration_lease_sec",
            _real(
                self.exploration_lease_sec,
                "exploration_lease_sec",
                positive=True,
            ),
        )


@dataclass(frozen=True)
class StatsSnapshot:
    """Deeply immutable snapshot of the router's observable state."""

    providers: Mapping[str, Mapping[str, object]]
    hedges: Mapping[str, object]
    exploration: Mapping[str, int]
    decisions_without_adoption: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "providers", _freeze(self.providers))
        object.__setattr__(self, "hedges", _freeze(self.hedges))
        object.__setattr__(self, "exploration", _freeze(self.exploration))


@dataclass
class _ProviderState:
    profile: RollingLatencyProfile
    failure_streak: int = 0
    cooldown_until: float = 0.0
    primary_selections: int = 0
    errors: dict[str, Counter[str]] = field(
        default_factory=lambda: {"health": Counter(), "request": Counter()}
    )
    actual_spend_usd: float = 0.0
    calculated_spend_usd: float = 0.0
    unsettled_attempts: int = 0


@dataclass(frozen=True)
class _ExplorationLease:
    attempt_id: str
    expires_at: float
    generation: int


class Router:
    """Learning API-provider router with typed outcome reporting."""

    def __init__(
        self,
        providers: Iterable[Provider],
        *,
        alpha: float = 0.25,
        slo_ms: float | None = None,
        seed: int | None = None,
        cold_start: str = "explore",
        clock: Callable[[], float] | None = None,
        tuning: Tuning | None = None,
    ) -> None:
        try:
            provider_list = list(providers)
        except TypeError as exc:
            raise ValidationError("providers must be an iterable of Provider objects") from exc
        if not provider_list:
            raise ValidationError("providers must contain at least one provider")
        if any(not isinstance(provider, Provider) for provider in provider_list):
            raise ValidationError("providers must contain Provider objects")
        names = [provider.name for provider in provider_list]
        if len(set(names)) != len(names):
            raise ValidationError("provider names must be unique")
        if not isinstance(seed, (int, type(None))) or isinstance(seed, bool):
            raise ValidationError("seed must be an integer or None")
        if cold_start not in {"explore", "require_observations"}:
            raise ValidationError('cold_start must be "explore" or "require_observations"')
        alpha_value = _alpha(alpha)
        slo_value = None if slo_ms is None else _real(slo_ms, "slo_ms", positive=True)
        tuning_value = Tuning() if tuning is None else tuning
        if not isinstance(tuning_value, Tuning):
            raise ValidationError("tuning must be a Tuning object or None")
        clock_value = time.monotonic if clock is None else clock
        if not callable(clock_value):
            raise ValidationError("clock must be a zero-argument callable")
        try:
            initial_clock = _clock_seconds(clock_value())
        except Exception as exc:
            if isinstance(exc, ValidationError):
                raise
            raise ValidationError("clock must be a zero-argument callable") from exc

        self._providers = {provider.name: provider for provider in provider_list}
        self._states = {
            provider.name: _ProviderState(
                RollingLatencyProfile(window_sec=tuning_value.window_min * 60.0)
            )
            for provider in provider_list
        }
        self._alpha = alpha_value
        self._slo_ms = slo_value
        self._cold_start = cold_start
        self._clock = clock_value
        self._last_now = initial_clock
        self._tuning = tuning_value
        self._rng = random.Random(seed)
        self._lock = RLock()
        self._estimator = _OutputLengthEstimator()
        self._capacity = _NoopCapacityController()
        self._leases: dict[str, _ExplorationLease] = {}
        self._lease_generation: Counter[str] = Counter()
        self._hedges: dict[str, float | int] = {
            "offered": 0,
            "declined": 0,
            "won": 0,
            "actual_spend_usd": 0.0,
            "calculated_spend_usd": 0.0,
        }
        self._exploration = {"decisions": 0, "target_selected": 0}
        self._decisions_without_adoption = 0

    def _now_locked(self) -> float:
        try:
            raw = _clock_seconds(self._clock())
        except Exception as exc:
            if isinstance(exc, ValidationError):
                raise
            raise ValidationError("clock must return finite numeric seconds") from exc
        now = max(self._last_now, raw)
        self._last_now = now
        return now

    def route(
        self,
        *,
        input_tokens: int,
        estimated_output_tokens: float | None = None,
        estimated_cached_tokens: int | Mapping[str, int] = 0,
        alpha: float | None = None,
        exclude: Iterable[str] = (),
    ) -> Decision:
        input_count = _token_count(input_tokens, "input_tokens")
        output_estimate = (
            None
            if estimated_output_tokens is None
            else _real(estimated_output_tokens, "estimated_output_tokens")
        )
        alpha_value = self._alpha if alpha is None else _alpha(alpha)
        try:
            excluded = frozenset(exclude)
        except TypeError as exc:
            raise ValidationError("exclude must be an iterable of provider names") from exc
        if any(not isinstance(name, str) for name in excluded):
            raise ValidationError("exclude must contain provider names")
        unknown_excluded = excluded.difference(self._providers)
        if unknown_excluded:
            raise ValidationError(
                f"exclude contains unknown provider(s): {', '.join(sorted(unknown_excluded))}"
            )
        cached = self._validate_cached_tokens(estimated_cached_tokens, input_count)

        with self._lock:
            now = self._now_locked()
            self._expire_leases_locked(now)
            predicted_output = (
                self._estimator.predict(input_count)
                if output_estimate is None
                else output_estimate
            )
            costs = {
                name: self._price(
                    self._providers[name],
                    input_count,
                    cached[name],
                    predicted_output,
                )
                for name in self._providers
            }
            capacity_excluded: set[str] = set()
            replan_count = 0
            rng_state = self._rng.getstate()

            while True:
                transaction_excluded = frozenset(set(excluded) | capacity_excluded)
                eligible, reasons = self._eligible_locked(now, transaction_excluded)
                for name in capacity_excluded:
                    reasons[name] = "capacity"
                if not eligible:
                    self._rng.setstate(rng_state)
                    raise NoProviderError(self._no_provider_message(reasons))

                c_min_name = min(eligible, key=lambda name: (costs[name], name))
                c_max_name = max(eligible, key=lambda name: (costs[name], name))
                c_min = costs[c_min_name]
                c_max = costs[c_max_name]
                budget = c_min + alpha_value * (c_max - c_min)
                target = self._exploration_target_locked(eligible, now, costs)
                exploration_q = 0.0
                reason = "latency_optimized_lp"
                if target is not None:
                    if costs[target] <= budget:
                        exploration_q = 1.0
                    elif costs[target] > c_min:
                        exploration_q = max(
                            0.0,
                            min(1.0, (budget - c_min) / (costs[target] - c_min)),
                        )

                if target is not None and exploration_q > 0.0:
                    reason = "cold_start_exploration"
                    if target == c_min_name:
                        mutable_weights = {target: 1.0}
                    else:
                        mutable_weights = {
                            c_min_name: 1.0 - exploration_q,
                            target: exploration_q,
                        }
                        mutable_weights = {
                            name: weight
                            for name, weight in mutable_weights.items()
                            if weight > 1e-12
                        }
                    sampled = self._sample_locked(mutable_weights)
                    expected_latency = None
                else:
                    mutable_weights = self._plain_lp_locked(eligible, costs, budget, now)
                    sampled = self._sample_locked(mutable_weights)
                    if any(self._is_unprofiled_locked(name, now) for name in mutable_weights):
                        expected_latency = None
                    else:
                        expected_latency = sum(
                            weight * self._latency_mean_locked(name, now)
                            for name, weight in mutable_weights.items()
                        )

                attempt_id = uuid.uuid4().hex
                reservation = self._reserve_locked(sampled, attempt_id, now)
                if reservation is not None:
                    break
                capacity_excluded.add(sampled)
                replan_count += 1

            expected_cost = sum(costs[name] * weight for name, weight in mutable_weights.items())
            lease_generation: int | None = None
            if target is not None and exploration_q > 0.0:
                self._exploration["decisions"] += 1
                if sampled == target:
                    self._lease_generation[target] += 1
                    lease_generation = self._lease_generation[target]
                    self._leases[target] = _ExplorationLease(
                        attempt_id=attempt_id,
                        expires_at=now + self._tuning.exploration_lease_sec,
                        generation=lease_generation,
                    )
                    self._exploration["target_selected"] += 1

            self._states[sampled].primary_selections += 1
            trace = {
                "reason": reason,
                "alpha": alpha_value,
                "budget_usd": budget,
                "c_min_usd": c_min,
                "c_min_provider": c_min_name,
                "c_max_usd": c_max,
                "c_max_provider": c_max_name,
                "weights": mutable_weights,
                "sampled_provider": sampled,
                "excluded": tuple(sorted(excluded)),
                "eligibility_exclusions": reasons,
                "capacity_exclusions": tuple(sorted(capacity_excluded)),
                "capacity_replans": replan_count,
            }
            if target is not None and exploration_q > 0.0:
                trace.update(
                    {
                        "exploration_target": target,
                        "exploration_probability": exploration_q,
                        "latency_estimate": "unprofiled",
                    }
                )
            return Decision(
                router=self,
                provider=sampled,
                weights=mutable_weights,
                expected_cost_usd=expected_cost,
                expected_latency_ms=expected_latency,
                checkpoints_ms=tuple(
                    checkpoint * 1000.0
                    for checkpoint in (
                        hedge_checkpoints_for_slo(self._slo_ms) if self._slo_ms is not None else ()
                    )
                ),
                trace=trace,
                input_tokens=input_count,
                estimated_cached_tokens=cached,
                predicted_output_tokens=predicted_output,
                original_exclude=excluded,
                primary_attempt_id=attempt_id,
                primary_reservation=reservation,
                primary_lease_generation=lease_generation,
            )

    def observe(
        self,
        provider: str,
        *,
        ttft_ms: float | None = None,
        kind: str | None = None,
        code: str | None = None,
    ) -> None:
        if not isinstance(provider, str) or provider not in self._providers:
            raise ValidationError(f"unknown provider: {provider!r}")
        if (ttft_ms is None) == (kind is None):
            raise ValidationError("observe requires exactly one of ttft_ms or kind")
        if code is not None and kind is None:
            raise ValidationError("code may only be supplied with kind")
        if ttft_ms is not None:
            latency = _real(ttft_ms, "ttft_ms")
            normalized_kind = None
        else:
            latency = None
            normalized_kind = self._validate_kind(kind)
            self._validate_code(code)
        with self._lock:
            now = self._now_locked()
            if latency is not None:
                self._record_success_locked(provider, latency, now, lease_attempt_id=None)
            else:
                self._record_failure_locked(provider, normalized_kind, code, now, before_ttft=True)

    def stats(self) -> StatsSnapshot:
        with self._lock:
            now = self._now_locked()
            self._expire_leases_locked(now)
            providers: dict[str, dict[str, object]] = {}
            for name, state in self._states.items():
                values = state.profile.values(now)
                providers[name] = {
                    "primary_selections": state.primary_selections,
                    "ttft_p50_ms": _percentile(values, 0.50),
                    "ttft_p95_ms": _percentile(values, 0.95),
                    "errors": {kind: dict(counts) for kind, counts in state.errors.items()},
                    "cooldown_remaining_sec": max(state.cooldown_until - now, 0.0),
                    "actual_spend_usd": state.actual_spend_usd,
                    "calculated_spend_usd": state.calculated_spend_usd,
                    "unsettled_attempts": state.unsettled_attempts,
                }
            return StatsSnapshot(
                providers=providers,
                hedges=dict(self._hedges),
                exploration=dict(self._exploration),
                decisions_without_adoption=self._decisions_without_adoption,
            )

    def _validate_cached_tokens(
        self,
        value: int | Mapping[str, int],
        input_tokens: int,
    ) -> dict[str, int]:
        if isinstance(value, Mapping):
            unknown = set(value).difference(self._providers)
            if unknown:
                raise ValidationError(
                    "estimated_cached_tokens contains unknown provider(s): "
                    + ", ".join(sorted(str(name) for name in unknown))
                )
            result = dict.fromkeys(self._providers, 0)
            for name, count in value.items():
                if not isinstance(name, str):
                    raise ValidationError("estimated_cached_tokens keys must be provider names")
                result[name] = min(
                    _token_count(count, f"estimated_cached_tokens[{name!r}]"), input_tokens
                )
            return result
        count = min(_token_count(value, "estimated_cached_tokens"), input_tokens)
        return dict.fromkeys(self._providers, count)

    @staticmethod
    def _price(
        provider: Provider,
        input_tokens: int,
        cached_tokens: int,
        output_tokens: int | float,
    ) -> float:
        try:
            cached = min(max(int(cached_tokens), 0), input_tokens)
            cached_price = (
                provider.price_in if provider.price_cached is None else provider.price_cached
            )
            cost = (
                provider.price_in * (input_tokens - cached)
                + cached_price * cached
                + provider.price_out * float(output_tokens)
            ) / 1_000_000.0
        except (OverflowError, ValueError) as exc:
            raise ValidationError("derived request cost must be finite") from exc
        if not math.isfinite(cost) or cost < 0.0:
            raise ValidationError("derived request cost must be finite and non-negative")
        return cost

    def _expire_leases_locked(self, now: float) -> None:
        expired = [name for name, lease in self._leases.items() if lease.expires_at <= now]
        for name in expired:
            del self._leases[name]

    def _eligible_locked(
        self,
        now: float,
        excluded: frozenset[str],
    ) -> tuple[list[str], dict[str, str]]:
        eligible: list[str] = []
        reasons: dict[str, str] = {}
        for name in self._providers:
            if name in excluded:
                reasons[name] = "excluded"
                continue
            state = self._states[name]
            if state.cooldown_until > now:
                reasons[name] = "cooldown"
                continue
            unprofiled = self._is_unprofiled_locked(name, now)
            if self._cold_start == "require_observations" and unprofiled:
                reasons[name] = "unprofiled"
                continue
            if self._cold_start == "explore" and unprofiled and name in self._leases:
                reasons[name] = "exploration_lease"
                continue
            eligible.append(name)
        return eligible, reasons

    @staticmethod
    def _no_provider_message(reasons: Mapping[str, str]) -> str:
        if reasons and all(reason == "unprofiled" for reason in reasons.values()):
            return "no provider is eligible; seed profiles with router.observe() first"
        if reasons and all(reason == "exploration_lease" for reason in reasons.values()):
            return "no provider is eligible; cold-start exploration leases are already in flight"
        details = ", ".join(f"{name}={reason}" for name, reason in sorted(reasons.items()))
        return "no provider is eligible" + (f" ({details})" if details else "")

    def _is_unprofiled_locked(self, provider: str, now: float) -> bool:
        profile = self._states[provider].profile
        return profile.count(now) + profile.error_count(now) == 0

    def _exploration_target_locked(
        self,
        eligible: Sequence[str],
        now: float,
        costs: Mapping[str, float],
    ) -> str | None:
        if self._cold_start != "explore":
            return None
        candidates = [name for name in eligible if self._is_unprofiled_locked(name, now)]
        if not candidates:
            return None
        return min(candidates, key=lambda name: (costs[name], name))

    def _latency_mean_locked(self, provider: str, now: float) -> float:
        value = self._states[provider].profile.mean_with_errors(
            now,
            error_penalty_ms=self._tuning.penalty_ms,
        )
        return _UNKNOWN_LATENCY_MS if value is None else value

    def _plain_lp_locked(
        self,
        eligible: Sequence[str],
        costs: Mapping[str, float],
        budget: float,
        now: float,
    ) -> dict[str, float]:
        latencies = [self._latency_mean_locked(name, now) for name in eligible]
        effective_costs = [costs[name] for name in eligible]
        objective = cost_tiebroken_objective(latencies, effective_costs)
        result = solve_budget_lp(
            [
                BudgetLPCandidate(
                    name=name,
                    objective=objective[index],
                    effective_cost=costs[name],
                )
                for index, name in enumerate(eligible)
            ],
            budget=budget,
        )
        if not result.feasible or not result.weights:
            cheapest = min(eligible, key=lambda name: (costs[name], name))
            return {cheapest: 1.0}
        return result.weights

    def _sample_locked(self, weights: Mapping[str, float]) -> str:
        draw = self._rng.random()
        cumulative = 0.0
        selected = next(iter(weights))
        for name, weight in weights.items():
            cumulative += float(weight)
            selected = name
            if draw < cumulative:
                return name
        return selected

    def _reserve_locked(
        self,
        provider: str,
        attempt_id: str,
        now: float,
    ) -> _Reservation | None:
        snapshot = self._capacity.snapshot(resource_key=provider, now=now)
        return self._capacity.try_reserve(
            resource_key=provider,
            attempt_id=attempt_id,
            snapshot=snapshot,
        )

    def _release_lease_locked(
        self,
        provider: str,
        *,
        attempt_id: str | None,
    ) -> None:
        lease = self._leases.get(provider)
        if lease is None:
            return
        if attempt_id is None or lease.attempt_id == attempt_id:
            del self._leases[provider]

    def _record_success_locked(
        self,
        provider: str,
        ttft_ms: float,
        now: float,
        *,
        lease_attempt_id: str | None,
    ) -> None:
        state = self._states[provider]
        state.profile.add_sample(now, ttft_ms)
        state.failure_streak = 0
        state.cooldown_until = 0.0
        self._release_lease_locked(provider, attempt_id=lease_attempt_id)

    def _record_failure_locked(
        self,
        provider: str,
        kind: str,
        code: str | None,
        now: float,
        *,
        before_ttft: bool,
        lease_attempt_id: str | None = None,
    ) -> None:
        state = self._states[provider]
        label = code if code in _KNOWN_ERROR_CODES else "other"
        state.errors[kind][label] += 1
        if kind == "health":
            if before_ttft:
                state.profile.add_error(now, label)
                self._release_lease_locked(provider, attempt_id=lease_attempt_id)
            state.failure_streak += 1
            if state.failure_streak >= self._tuning.cooldown_after:
                state.cooldown_until = max(
                    state.cooldown_until,
                    now + self._tuning.cooldown_sec,
                )

    @staticmethod
    def _validate_kind(kind: object) -> str:
        if kind not in {"health", "request"}:
            raise ValidationError('kind must be "health" or "request"')
        return str(kind)

    @staticmethod
    def _validate_code(code: object) -> None:
        if code is not None and not isinstance(code, str):
            raise ValidationError("code must be a string or None")

    def _preview_settlement_locked(
        self,
        attempt: Attempt,
        *,
        output_tokens: int | None = None,
        cached_tokens: int | None = None,
        cost_usd: float | None = None,
    ) -> tuple[str, float]:
        output = attempt._output_tokens if output_tokens is None else output_tokens
        cached = attempt._cached_tokens if cached_tokens is None else cached_tokens
        cost = attempt._cost_usd if cost_usd is None else cost_usd
        if cost is not None:
            new_state = "actual"
            new_amount = cost
        elif output is not None:
            billed_cached = attempt._estimated_cached_tokens if cached is None else cached
            new_state = "calculated"
            new_amount = self._price(
                attempt._price_snapshot,
                attempt._input_tokens,
                min(billed_cached, attempt._input_tokens),
                output,
            )
        else:
            new_state = "unknown"
            new_amount = 0.0

        state = self._states[attempt.provider]
        provider_calculated = state.calculated_spend_usd
        provider_actual = state.actual_spend_usd
        hedge_calculated = float(self._hedges["calculated_spend_usd"])
        hedge_actual = float(self._hedges["actual_spend_usd"])
        if attempt._billing_state == "calculated":
            provider_calculated -= attempt._billing_amount
            if attempt._is_backup:
                hedge_calculated -= attempt._billing_amount
        elif attempt._billing_state == "actual":
            provider_actual -= attempt._billing_amount
            if attempt._is_backup:
                hedge_actual -= attempt._billing_amount
        if new_state == "calculated":
            provider_calculated += new_amount
            if attempt._is_backup:
                hedge_calculated += new_amount
        elif new_state == "actual":
            provider_actual += new_amount
            if attempt._is_backup:
                hedge_actual += new_amount
        if not all(
            math.isfinite(value)
            for value in (
                provider_calculated,
                provider_actual,
                hedge_calculated,
                hedge_actual,
            )
        ):
            raise ValidationError("settlement would overflow spend aggregates")
        return (new_state, new_amount)

    def _settle_attempt_locked(
        self,
        attempt: Attempt,
        *,
        target: tuple[str, float] | None = None,
    ) -> None:
        old_state = attempt._billing_state
        old_amount = attempt._billing_amount
        new_state, new_amount = (
            self._preview_settlement_locked(attempt) if target is None else target
        )

        if old_state == new_state and old_amount == new_amount:
            return

        state = self._states[attempt.provider]
        hedge_key = attempt._is_backup
        if old_state == "calculated":
            state.calculated_spend_usd -= old_amount
            if hedge_key:
                self._hedges["calculated_spend_usd"] -= old_amount
        elif old_state == "actual":
            state.actual_spend_usd -= old_amount
            if hedge_key:
                self._hedges["actual_spend_usd"] -= old_amount

        if new_state == "calculated":
            state.calculated_spend_usd += new_amount
            if hedge_key:
                self._hedges["calculated_spend_usd"] += new_amount
        elif new_state == "actual":
            state.actual_spend_usd += new_amount
            if hedge_key:
                self._hedges["actual_spend_usd"] += new_amount

        if new_state != "unknown" and attempt._unsettled_counted:
            state.unsettled_attempts -= 1
            attempt._unsettled_counted = False
        attempt._billing_state = new_state
        attempt._billing_amount = new_amount

    def _maybe_count_unsettled_locked(self, attempt: Attempt) -> None:
        if (
            attempt._state in _TERMINAL_STATES
            and attempt._state != "declined"
            and attempt._billing_state == "unknown"
            and not attempt._unsettled_counted
        ):
            self._states[attempt.provider].unsettled_attempts += 1
            attempt._unsettled_counted = True

    def _maybe_train_estimator_locked(self, attempt: Attempt) -> None:
        decision = attempt._decision
        if (
            attempt._state == "completed"
            and decision._adopted is attempt
            and attempt._is_backup
            and not attempt._hedge_win_counted
        ):
            self._hedges["won"] += 1
            attempt._hedge_win_counted = True
        if (
            attempt._state == "completed"
            and decision._adopted is attempt
            and attempt._output_tokens is not None
            and attempt._output_tokens > 0
            and not attempt._estimator_trained
        ):
            self._estimator.update(attempt._input_tokens, attempt._output_tokens)
            attempt._estimator_trained = True

    def _refresh_decision_resolution_locked(self, decision: Decision) -> None:
        if decision._adopted is not None:
            adopted = decision._adopted
            if adopted._state in _TERMINAL_STATES:
                decision._state = adopted._state
            elif adopted._state == "streaming":
                decision._state = "streaming"
            else:
                decision._state = "pending"
            return
        if not all(attempt._state in _TERMINAL_STATES for attempt in decision._attempts):
            decision._state = "pending"
            return
        states = {attempt._state for attempt in decision._attempts}
        if "completed" in states:
            decision._state = "unresolved"
            if not decision._unresolved_counted:
                self._decisions_without_adoption += 1
                decision._unresolved_counted = True
        elif "failed" in states:
            decision._state = "failed"
        elif "cancelled" in states:
            decision._state = "cancelled"
        else:
            decision._state = "declined"

    def _hedge_candidates_locked(
        self,
        decision: Decision,
        elapsed_ms: float,
        now: float,
    ) -> list[BackupCandidate[str]]:
        """Build the currently eligible backup set at one checkpoint."""
        assert self._slo_ms is not None
        primary = decision._primary
        primary_state = self._states[primary.provider]
        candidates: list[BackupCandidate[str]] = []
        for name in self._providers:
            if name == primary.provider or name in decision._original_exclude:
                continue
            provider_state = self._states[name]
            if provider_state.cooldown_until > now:
                continue
            if provider_state.profile.count(now) < self._tuning.hedge_min_samples:
                continue
            remaining_ms = self._slo_ms - elapsed_ms - DISPATCH_OVERHEAD_MS
            backup_profile = provider_state.profile

            def backup_cdf(
                threshold: float,
                profile: RollingLatencyProfile = backup_profile,
            ) -> float:
                return float(profile.cdf_counting_errors(threshold, now) or 0.0)

            def primary_cdf(threshold: float) -> float:
                return float(primary_state.profile.cdf_counting_errors(threshold, now) or 0.0)

            primary_survival = max(1.0 - primary_cdf(elapsed_ms), 0.0)
            if primary_survival <= 1e-9:
                probability = 0.0 if remaining_ms <= 0.0 else backup_cdf(remaining_ms)
            else:
                probability = combined_success_probability(
                    primary_cdf,
                    backup_cdf,
                    elapsed_ms=elapsed_ms,
                    slo_ms=self._slo_ms,
                    dispatch_overhead_ms=DISPATCH_OVERHEAD_MS,
                )
            cost = self._price(
                self._providers[name],
                decision._input_tokens,
                decision._estimated_cached_tokens[name],
                decision._predicted_output_tokens,
            )
            candidates.append(
                BackupCandidate(
                    provider=name,
                    success_probability=probability,
                    marginal_cost=cost,
                    true_mean_ms=self._latency_mean_locked(name, now),
                    success_target=self._tuning.hedge_target,
                )
            )
        return candidates

    def _hedge_locked(self, decision: Decision, elapsed_ms: float, now: float) -> Attempt | None:
        if self._slo_ms is None or decision._hedge_slot_used or decision._hedge_slot_closed:
            return None
        if decision._state in _TERMINAL_STATES or decision._state == "unresolved":
            decision._hedge_slot_closed = True
            return None
        primary = decision._primary
        if primary._state != "pending":
            decision._hedge_slot_closed = True
            return None
        primary_state = self._states[primary.provider]
        if primary_state.profile.count(now) < self._tuning.hedge_min_samples:
            return None
        capacity_excluded: set[str] = set()
        selected: BackupCandidate[str] | None = None
        reservation: _Reservation | None = None
        attempt_id = ""
        while True:
            candidates = [
                candidate
                for candidate in self._hedge_candidates_locked(decision, elapsed_ms, now)
                if str(candidate.provider) not in capacity_excluded
            ]
            selected = select_probability_backup(candidates)
            if selected is None:
                if capacity_excluded:
                    prior = tuple(decision._trace.get("capacity_exclusions", ()))
                    decision._replace_trace(
                        {
                            **dict(decision._trace),
                            "capacity_exclusions": tuple(
                                dict.fromkeys((*prior, *sorted(capacity_excluded)))
                            ),
                            "hedge_capacity_exclusions": tuple(sorted(capacity_excluded)),
                        }
                    )
                return None

            # Latest-safe dispatch: a currently feasible hedge is withheld
            # while any later scheduled checkpoint still has a feasible
            # non-contended backup.
            future_feasible = False
            for future_elapsed_ms in decision.checkpoints_ms:
                if future_elapsed_ms <= elapsed_ms + 1e-9:
                    continue
                future_candidates = [
                    candidate
                    for candidate in self._hedge_candidates_locked(
                        decision,
                        future_elapsed_ms,
                        now,
                    )
                    if str(candidate.provider) not in capacity_excluded
                ]
                if select_probability_backup(future_candidates) is not None:
                    future_feasible = True
                    break
            if future_feasible:
                return None

            attempt_id = uuid.uuid4().hex
            reservation = self._reserve_locked(str(selected.provider), attempt_id, now)
            if reservation is not None:
                break
            capacity_excluded.add(str(selected.provider))

        assert selected is not None
        assert reservation is not None
        if capacity_excluded:
            prior = tuple(decision._trace.get("capacity_exclusions", ()))
            decision._replace_trace(
                {
                    **dict(decision._trace),
                    "capacity_exclusions": tuple(
                        dict.fromkeys((*prior, *sorted(capacity_excluded)))
                    ),
                    "hedge_capacity_exclusions": tuple(sorted(capacity_excluded)),
                }
            )
        attempt = Attempt(
            decision=decision,
            router=self,
            provider=str(selected.provider),
            attempt_id=attempt_id,
            input_tokens=decision._input_tokens,
            estimated_cached_tokens=decision._estimated_cached_tokens[str(selected.provider)],
            is_backup=True,
            reservation=reservation,
        )
        decision._attempts.append(attempt)
        decision._backups.append(attempt)
        decision._hedge_slot_used = True
        self._hedges["offered"] += 1
        return attempt


class Decision:
    """One logical request and the primary attempt selected for it."""

    def __init__(
        self,
        *,
        router: Router,
        provider: str,
        weights: Mapping[str, float],
        expected_cost_usd: float,
        expected_latency_ms: float | None,
        checkpoints_ms: tuple[float, ...],
        trace: Mapping[str, object],
        input_tokens: int,
        estimated_cached_tokens: Mapping[str, int],
        predicted_output_tokens: float,
        original_exclude: frozenset[str],
        primary_attempt_id: str,
        primary_reservation: _Reservation,
        primary_lease_generation: int | None,
    ) -> None:
        self._router = router
        self._provider = provider
        self._weights = _freeze(dict(weights))
        self._expected_cost_usd = float(expected_cost_usd)
        self._expected_latency_ms = expected_latency_ms
        self._checkpoints_ms = tuple(checkpoints_ms)
        self._trace = _freeze(dict(trace))
        self._input_tokens = input_tokens
        self._estimated_cached_tokens = dict(estimated_cached_tokens)
        self._predicted_output_tokens = predicted_output_tokens
        self._original_exclude = original_exclude
        self._state = "pending"
        self._adopted: Attempt | None = None
        self._hedge_slot_used = False
        self._hedge_slot_closed = False
        self._unresolved_counted = False
        self._primary = Attempt(
            decision=self,
            router=router,
            provider=provider,
            attempt_id=primary_attempt_id,
            input_tokens=input_tokens,
            estimated_cached_tokens=estimated_cached_tokens[provider],
            is_backup=False,
            reservation=primary_reservation,
            lease_generation=primary_lease_generation,
        )
        self._attempts = [self._primary]
        self._backups: list[Attempt] = []

    @property
    def trace(self) -> Mapping[str, object]:
        return self._trace

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def weights(self) -> Mapping[str, float]:
        return self._weights

    @property
    def expected_cost_usd(self) -> float:
        return self._expected_cost_usd

    @property
    def expected_latency_ms(self) -> float | None:
        return self._expected_latency_ms

    @property
    def checkpoints_ms(self) -> tuple[float, ...]:
        return self._checkpoints_ms

    @property
    def state(self) -> str:
        return self._state

    @property
    def primary(self) -> Attempt:
        return self._primary

    @property
    def backups(self) -> tuple[Attempt, ...]:
        return tuple(self._backups)

    def _replace_trace(self, trace: Mapping[str, object]) -> None:
        self._trace = _freeze(dict(trace))

    def explain(self) -> str:
        trace = self._trace
        mix = ", ".join(f"{name} {weight:.2f}" for name, weight in self.weights.items())
        explanation = (
            f"budget=${float(trace['budget_usd']):.6f} "
            f"(alpha={float(trace['alpha']):.2f}, "
            f"c_min=${float(trace['c_min_usd']):.6f}@{trace['c_min_provider']}, "
            f"c_max=${float(trace['c_max_usd']):.6f}@{trace['c_max_provider']}); "
            f"mix: {mix} -> sampled: {self.provider}"
        )
        exclusions = trace.get("eligibility_exclusions", {})
        if exclusions:
            rendered = ", ".join(f"{name} ({reason})" for name, reason in exclusions.items())
            explanation += f"; excluded: {rendered}"
        return explanation

    def first_token(self, *, ttft_ms: float, adopted: bool | None = None) -> None:
        self._primary.first_token(ttft_ms=ttft_ms, adopted=adopted)

    def completed(
        self,
        *,
        output_tokens: int | None = None,
        ttft_ms: float | None = None,
        cached_tokens: int | None = None,
        cost_usd: float | None = None,
        adopted: bool | None = None,
    ) -> None:
        self._primary.completed(
            output_tokens=output_tokens,
            ttft_ms=ttft_ms,
            cached_tokens=cached_tokens,
            cost_usd=cost_usd,
            adopted=adopted,
        )

    def failed(self, *, kind: str, code: str | None = None) -> None:
        self._primary.failed(kind=kind, code=code)

    def cancelled(self) -> None:
        self._primary.cancelled()

    def declined(self) -> None:
        self._primary.declined()

    def settle(
        self,
        *,
        output_tokens: int | None = None,
        cached_tokens: int | None = None,
        cost_usd: float | None = None,
    ) -> None:
        self._primary.settle(
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost_usd,
        )

    def hedge_now(self, *, elapsed_ms: float) -> Attempt | None:
        elapsed = _real(elapsed_ms, "elapsed_ms")
        with self._router._lock:
            now = self._router._now_locked()
            return self._router._hedge_locked(self, elapsed, now)


class Attempt:
    """A handle for one concrete provider dispatch."""

    def __init__(
        self,
        *,
        decision: Decision,
        router: Router,
        provider: str,
        attempt_id: str,
        input_tokens: int,
        estimated_cached_tokens: int,
        is_backup: bool,
        reservation: _Reservation,
        lease_generation: int | None = None,
    ) -> None:
        self._decision = decision
        self._router = router
        self._provider = provider
        self._attempt_id = attempt_id
        self._input_tokens = input_tokens
        self._estimated_cached_tokens = estimated_cached_tokens
        self._is_backup = is_backup
        self._reservation = reservation
        self._lease_generation = lease_generation
        self._price_snapshot = router._providers[provider]
        self._state = "pending"
        self._first_token_ms: float | None = None
        self._terminal_signature: tuple[object, ...] | None = None
        self._output_tokens: int | None = None
        self._cached_tokens: int | None = None
        self._cost_usd: float | None = None
        self._billing_state = "unknown"
        self._billing_amount = 0.0
        self._unsettled_counted = False
        self._estimator_trained = False
        self._hedge_win_counted = False
        self._hedge_declined_counted = False

    @property
    def state(self) -> str:
        return self._state

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def billing_state(self) -> str:
        return self._billing_state

    def _validate_adoption_locked(self, adopted: bool | None) -> None:
        if adopted is not None and adopted is not True:
            raise ValidationError("adopted must be True or None")
        if adopted is True:
            existing = self._decision._adopted
            if existing is not None and existing is not self:
                raise OutcomeError("a different attempt is already adopted")

    def _apply_adoption_locked(self, adopted: bool | None) -> None:
        if adopted is True:
            self._decision._adopted = self
            self._decision._hedge_slot_closed = True

    def first_token(self, *, ttft_ms: float, adopted: bool | None = None) -> None:
        latency = _real(ttft_ms, "ttft_ms")
        with self._router._lock:
            self._validate_adoption_locked(adopted)
            if self._state in _TERMINAL_STATES:
                raise OutcomeError("first_token cannot follow a terminal outcome")
            if self._first_token_ms is not None:
                if self._first_token_ms == latency and (
                    adopted is None or self._decision._adopted is self
                ):
                    return
                raise OutcomeError("first_token was already reported with different values")
            now = self._router._now_locked()
            self._first_token_ms = latency
            self._state = "streaming"
            self._apply_adoption_locked(adopted)
            self._router._record_success_locked(
                self.provider,
                latency,
                now,
                lease_attempt_id=self._attempt_id,
            )
            if not self._is_backup:
                self._decision._hedge_slot_closed = True
            self._router._refresh_decision_resolution_locked(self._decision)

    def completed(
        self,
        *,
        output_tokens: int | None = None,
        ttft_ms: float | None = None,
        cached_tokens: int | None = None,
        cost_usd: float | None = None,
        adopted: bool | None = None,
    ) -> None:
        output = None if output_tokens is None else _token_count(output_tokens, "output_tokens")
        latency = None if ttft_ms is None else _real(ttft_ms, "ttft_ms")
        cached = None if cached_tokens is None else _token_count(cached_tokens, "cached_tokens")
        cost = None if cost_usd is None else _real(cost_usd, "cost_usd")
        signature = ("completed", output, latency, cached, cost, adopted)
        with self._router._lock:
            self._validate_adoption_locked(adopted)
            self._validate_billing_fields_locked(output, cached, cost)
            if self._state in _TERMINAL_STATES:
                if self._terminal_signature == signature:
                    return
                raise OutcomeError("attempt already has a conflicting terminal outcome")
            if (
                self._first_token_ms is not None
                and latency is not None
                and latency != self._first_token_ms
            ):
                raise OutcomeError("completed ttft_ms conflicts with first_token")
            if (
                adopted is True
                and self._first_token_ms is not None
                and self._decision._adopted is not self
            ):
                raise OutcomeError("streaming adoption must be declared on first_token")
            billing_target = self._router._preview_settlement_locked(
                self,
                output_tokens=output,
                cached_tokens=cached,
                cost_usd=cost,
            )
            now = (
                self._router._now_locked()
                if self._first_token_ms is None and latency is not None
                else None
            )
            self._apply_billing_fields_locked(output, cached, cost)
            self._state = "completed"
            self._terminal_signature = signature
            if (
                not self._decision._hedge_slot_used
                and self is self._decision._primary
                and self._decision._adopted is None
            ):
                self._decision._adopted = self
            self._apply_adoption_locked(adopted)
            if latency is not None and self._first_token_ms is None:
                self._first_token_ms = latency
                assert now is not None
                self._router._record_success_locked(
                    self.provider,
                    latency,
                    now,
                    lease_attempt_id=self._attempt_id,
                )
            self._reservation.release()
            self._router._settle_attempt_locked(self, target=billing_target)
            self._router._maybe_count_unsettled_locked(self)
            self._router._maybe_train_estimator_locked(self)
            self._decision._hedge_slot_closed = True
            self._router._refresh_decision_resolution_locked(self._decision)

    def failed(self, *, kind: str, code: str | None = None) -> None:
        normalized_kind = self._router._validate_kind(kind)
        self._router._validate_code(code)
        signature = ("failed", normalized_kind, code)
        with self._router._lock:
            if self._state in _TERMINAL_STATES:
                if self._terminal_signature == signature:
                    return
                raise OutcomeError("attempt already has a conflicting terminal outcome")
            now = self._router._now_locked()
            before_ttft = self._first_token_ms is None
            self._state = "failed"
            self._terminal_signature = signature
            self._router._record_failure_locked(
                self.provider,
                normalized_kind,
                code,
                now,
                before_ttft=before_ttft,
                lease_attempt_id=self._attempt_id,
            )
            self._reservation.release()
            self._router._settle_attempt_locked(self)
            self._router._maybe_count_unsettled_locked(self)
            self._decision._hedge_slot_closed = True
            self._router._refresh_decision_resolution_locked(self._decision)

    def cancelled(self) -> None:
        self._terminal_without_payload("cancelled")

    def declined(self) -> None:
        with self._router._lock:
            if self._state == "declined" and self._terminal_signature == ("declined",):
                return
            if self._state != "pending":
                raise OutcomeError("declined is legal only before dispatch starts")
            self._state = "declined"
            self._terminal_signature = ("declined",)
            self._reservation.release()
            if self._is_backup and not self._hedge_declined_counted:
                self._router._hedges["declined"] += 1
                self._hedge_declined_counted = True
            self._router._refresh_decision_resolution_locked(self._decision)

    def _terminal_without_payload(self, state: str) -> None:
        signature = (state,)
        with self._router._lock:
            if self._state in _TERMINAL_STATES:
                if self._terminal_signature == signature:
                    return
                raise OutcomeError("attempt already has a conflicting terminal outcome")
            self._state = state
            self._terminal_signature = signature
            self._reservation.release()
            self._router._settle_attempt_locked(self)
            self._router._maybe_count_unsettled_locked(self)
            self._decision._hedge_slot_closed = True
            self._router._refresh_decision_resolution_locked(self._decision)

    def settle(
        self,
        *,
        output_tokens: int | None = None,
        cached_tokens: int | None = None,
        cost_usd: float | None = None,
    ) -> None:
        output = None if output_tokens is None else _token_count(output_tokens, "output_tokens")
        cached = None if cached_tokens is None else _token_count(cached_tokens, "cached_tokens")
        cost = None if cost_usd is None else _real(cost_usd, "cost_usd")
        with self._router._lock:
            if self._state not in _TERMINAL_STATES or self._state == "declined":
                raise OutcomeError("settle requires a terminal attempt other than declined")
            self._validate_billing_fields_locked(output, cached, cost)
            billing_target = self._router._preview_settlement_locked(
                self,
                output_tokens=output,
                cached_tokens=cached,
                cost_usd=cost,
            )
            self._apply_billing_fields_locked(output, cached, cost)
            self._router._settle_attempt_locked(self, target=billing_target)
            self._router._maybe_count_unsettled_locked(self)
            self._router._maybe_train_estimator_locked(self)

    def _validate_billing_fields_locked(
        self,
        output_tokens: int | None,
        cached_tokens: int | None,
        cost_usd: float | None,
    ) -> None:
        fields = (
            ("output_tokens", self._output_tokens, output_tokens),
            ("cached_tokens", self._cached_tokens, cached_tokens),
            ("cost_usd", self._cost_usd, cost_usd),
        )
        for name, existing, supplied in fields:
            if supplied is not None and existing is not None and supplied != existing:
                raise OutcomeError(f"{name} was already settled with a different value")

    def _apply_billing_fields_locked(
        self,
        output_tokens: int | None,
        cached_tokens: int | None,
        cost_usd: float | None,
    ) -> None:
        if output_tokens is not None:
            self._output_tokens = output_tokens
        if cached_tokens is not None:
            self._cached_tokens = cached_tokens
        if cost_usd is not None:
            self._cost_usd = cost_usd


__all__ = ["Attempt", "Decision", "Provider", "Router", "StatsSnapshot", "Tuning"]
