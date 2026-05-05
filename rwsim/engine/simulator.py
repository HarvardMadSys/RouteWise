"""Canonical request loop for RouteWise policies."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from rwsim.engine.state import SimulationState
from rwsim.metrics import SimulationRun
from rwsim.policies.base import Policy
from rwsim.schemas import HedgeDispatch, Request, RoutingDecision, RoutingOutcome
from rwsim.world.capacity import ProviderTier
from rwsim.world.providers import Provider
from rwsim.world.scenarios import ScenarioConfig


_HEDGE_REQUEST_ID_OFFSET = 10_000_000
_DISPATCH_OVERHEAD_MS = 50.0


@dataclass
class Simulator:
    """Run one policy over one request stream."""

    scenario: ScenarioConfig
    seed: int = 42
    dispatch_overhead_ms: float = _DISPATCH_OVERHEAD_MS

    def run(
        self,
        requests: Sequence[Request],
        policy: Policy,
        *,
        policy_name: str,
    ) -> SimulationRun:
        """Run a policy over pre-loaded requests."""
        rng = np.random.default_rng(self.seed)
        for provider in self.scenario.providers:
            provider.reset_state()

        providers = {provider.name: provider for provider in self.scenario.providers}
        state = SimulationState.from_providers(providers)

        ttft_ms: list[float] = []
        cost_usd: list[float] = []
        provider_sel: list[str] = []
        tier_sel: list[str] = []
        timestamps: list[float] = []
        hedge_flags: list[bool] = []
        rejected: list[bool] = []
        quota_fraction_used: list[float] = []
        concurrency_utilization: list[float] = []

        for request in requests:
            state.now = float(request.timestamp)
            decision = policy.route(request, state)
            outcome = self._execute_request(request, decision, policy, state, rng)
            policy.observe(request, decision, outcome)

            final_provider = providers.get(outcome.final_provider)
            ttft_ms.append(outcome.ttft_ms)
            cost_usd.append(outcome.cost_usd)
            provider_sel.append(outcome.final_provider)
            tier_sel.append(final_provider.tier.value if final_provider is not None else "")
            timestamps.append(float(request.timestamp))
            hedge_flags.append(outcome.hedge_triggered)
            rejected.append(outcome.rejected)
            quota_fraction_used.append(self._max_quota_fraction(providers, state.now))
            concurrency_utilization.append(self._max_concurrency_utilization(providers, state.now))

            user_id = _request_user_id(request)
            if user_id is not None and not outcome.rejected:
                state.user_last_provider[user_id] = outcome.final_provider

        return SimulationRun(
            policy=policy_name,
            ttft_ms=np.array(ttft_ms),
            cost_usd=np.array(cost_usd),
            provider=provider_sel,
            tier=tier_sel,
            timestamp=np.array(timestamps),
            hedge_triggered=np.array(hedge_flags, dtype=bool),
            quota_fraction_used=np.array(quota_fraction_used),
            concurrency_utilization=np.array(concurrency_utilization),
            rejected=np.array(rejected, dtype=bool),
        )

    def _execute_request(
        self,
        request: Request,
        decision: RoutingDecision,
        policy: Policy,
        state: SimulationState,
        rng: np.random.Generator,
    ) -> RoutingOutcome:
        providers = state.providers
        primary = providers[decision.primary_provider]
        now = float(request.timestamp)
        state.now = now

        primary_ttft_ms = primary.sample_ttft(rng, now)
        primary_service_time = _sample_service_time(
            primary,
            rng,
            now,
            request.response_tokens or 1,
            primary_ttft_ms,
        )

        if not _can_admit(primary, now, primary_service_time):
            fallback = _fallback_provider(providers.values(), now)
            if fallback is None:
                return RoutingOutcome(
                    request_id=request.id,
                    primary_provider=primary.name,
                    final_provider=primary.name,
                    ttft_ms=float("inf"),
                    cost_usd=0.0,
                    rejected=True,
                    error="no_capacity",
                )
            primary = fallback
            primary_ttft_ms = primary.sample_ttft(rng, now)
            primary_service_time = _sample_service_time(
                primary,
                rng,
                now,
                request.response_tokens or 1,
                primary_ttft_ms,
            )

        primary.account_request(request.id, now, primary_service_time)
        billed_cost = primary.marginal_cost(request.total_tokens or 0, now)

        final_provider = primary.name
        final_ttft_ms = primary_ttft_ms
        hedge_triggered = False
        hedge_dispatch: HedgeDispatch | None = None
        backup_ttft_ms: float | None = None
        backup_observed_at: float | None = None

        for elapsed in sorted(decision.hedge_checkpoints):
            elapsed_ms = float(elapsed) * 1000.0
            if elapsed_ms >= primary_ttft_ms:
                break
            state.now = now + float(elapsed)
            hedge_dispatch = policy.tick(request, decision, float(elapsed), state)
            if hedge_dispatch is not None:
                break

        if hedge_dispatch is not None:
            backup = providers[hedge_dispatch.backup_provider]
            dispatch_time = state.now
            backup_ttft_ms = backup.sample_ttft(rng, dispatch_time)
            backup_service_time = _sample_service_time(
                backup,
                rng,
                dispatch_time,
                request.response_tokens or 1,
                backup_ttft_ms,
            )
            if _can_admit(backup, dispatch_time, backup_service_time):
                backup.account_request(
                    request.id + _HEDGE_REQUEST_ID_OFFSET,
                    dispatch_time,
                    backup_service_time,
                )
                billed_cost += backup.marginal_cost(request.total_tokens or 0, dispatch_time)
                backup_observed_at = dispatch_time + backup_ttft_ms / 1000.0
                hedged_ttft = (
                    (dispatch_time - now) * 1000.0
                    + self.dispatch_overhead_ms
                    + backup_ttft_ms
                )
                if hedged_ttft < final_ttft_ms:
                    final_ttft_ms = hedged_ttft
                    final_provider = backup.name
                hedge_triggered = True

        state.now = now
        return RoutingOutcome(
            request_id=request.id,
            primary_provider=primary.name,
            final_provider=final_provider,
            ttft_ms=final_ttft_ms,
            cost_usd=billed_cost,
            hedge_triggered=hedge_triggered,
            metadata={
                "primary_ttft_ms": primary_ttft_ms,
                "primary_observed_at": now + primary_ttft_ms / 1000.0,
                "hedge_provider": hedge_dispatch.backup_provider if hedge_dispatch else None,
                "backup_ttft_ms": backup_ttft_ms,
                "backup_observed_at": backup_observed_at,
            },
        )

    @staticmethod
    def _max_quota_fraction(providers: dict[str, Provider], now: float) -> float:
        values = [
            provider.quota.fraction_used(now)
            for provider in providers.values()
            if provider.quota is not None
        ]
        return float(max(values, default=0.0))

    @staticmethod
    def _max_concurrency_utilization(providers: dict[str, Provider], now: float) -> float:
        values = [
            provider.concurrency.utilization(now)
            for provider in providers.values()
            if provider.concurrency is not None
        ]
        return float(max(values, default=0.0))


def _sample_service_time(
    provider: Provider,
    rng: np.random.Generator,
    now: float,
    output_tokens: int,
    ttft_ms: float,
) -> float:
    """Return service time in seconds for capacity accounting."""
    if provider.tier != ProviderTier.S_C:
        return 0.0
    if provider.service_time_dist is not None:
        return float(provider.service_time_dist.sample(rng)[0]) / 1000.0
    tps = max(float(provider.tps_dist.sample(rng)[0]), 1.0)
    return (ttft_ms + (output_tokens / tps) * 1000.0) / 1000.0


def _can_admit(provider: Provider, now: float, service_time: float) -> bool:
    if provider.tier == ProviderTier.S_C and provider.concurrency is not None:
        return provider.concurrency.can_admit_interval(now, now + service_time)
    return provider.is_available(now)


def _fallback_provider(providers: Sequence[Provider], now: float) -> Provider | None:
    available = [provider for provider in providers if provider.is_available(now)]
    if not available:
        return None
    return min(available, key=lambda provider: (provider.marginal_cost(1, now), provider.true_p50_ms(now)))


def _request_user_id(request: Request) -> str | None:
    for key in ("user_id", "session_id", "sharegpt_conversation_id"):
        value = request.metadata.get(key)
        if value is not None:
            return str(value)
    return None


__all__ = ["Simulator"]
