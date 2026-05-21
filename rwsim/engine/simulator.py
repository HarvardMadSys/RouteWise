"""Canonical request loop for RouteWise policies.

For concurrency capacity accounting, the simulator derives service time from
the selected provider model: sampled TTFT plus sampled decode time
(``response_tokens / TPS``), unless the provider defines an explicit service
time distribution. BurstGPT elapsed-time fields are preserved as trace metadata
but are not used by the simulator's main capacity model.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from rwsim.const import DISPATCH_OVERHEAD_MS
from rwsim.engine.state import SimulationState
from rwsim.metrics import PerRequestRecord, Run, RunAggregator, Status
from rwsim.policies.prefix_cache import cached_input_tokens
from rwsim.schemas import HedgeDispatch, Request, RoutingDecision, RoutingOutcome
from rwsim.world.capacity import ProviderTier

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rwsim.policies.base import Policy
    from rwsim.world.providers import Provider
    from rwsim.world.scenarios import ScenarioConfig


_HEDGE_REQUEST_ID_OFFSET = 10_000_000


@dataclass
class Simulator:
    """Run one policy over one request stream."""

    scenario: ScenarioConfig
    seed: int = 42
    dispatch_overhead_ms: float = DISPATCH_OVERHEAD_MS
    retain_records: bool = True

    def run(
        self,
        requests: Sequence[Request],
        policy: Policy,
        *,
        policy_name: str,
    ) -> Run:
        """Run a policy over pre-loaded requests."""
        quota_window_start = float(requests[0].timestamp) if requests else 0.0
        for provider in self.scenario.providers:
            provider.reset_state(quota_window_start=quota_window_start)

        providers = {provider.name: provider for provider in self.scenario.providers}
        provider_rngs = _provider_rngs(self.seed, self.scenario.providers)
        state = SimulationState.from_providers(providers)
        state.metadata["prefix_cache_enabled"] = self.scenario.metadata.get(
            "prefix_cache_enabled",
            False,
        )
        aggregator = RunAggregator(
            policy=policy_name,
            scenario_name=self.scenario.name,
            source="simulation",
            retain_records=self.retain_records,
        )

        prev_trace_time: float | None = None
        for request in requests:
            now = float(request.timestamp)
            # Garbage-collect capacity ledger entries that completed before
            # the previous outer trace timestamp. The hedge tick loop inside
            # _execute_request may temporarily advance state.now to future
            # checkpoints; using prev_trace_time as the watermark (strictly
            # earlier than any subsequent query) keeps GC safe under such
            # non-monotonic queries.
            if prev_trace_time is not None and now >= prev_trace_time:
                for provider in providers.values():
                    if provider.concurrency is not None:
                        provider.concurrency.gc_before(prev_trace_time)

            state.now = now
            decision = policy.route(request, state)
            outcome = self._execute_request(
                request,
                decision,
                policy,
                state,
                provider_rngs,
            )
            policy.observe(request, decision, outcome)
            aggregator.observe(
                self._build_record(
                    request=request,
                    decision=decision,
                    outcome=outcome,
                    providers=providers,
                    policy_name=policy_name,
                )
            )

            user_id = _request_user_id(request)
            if user_id is not None and not outcome.rejected:
                state.user_last_provider[user_id] = outcome.final_provider

            prev_trace_time = now

        return aggregator.finalize()

    def _execute_request(
        self,
        request: Request,
        decision: RoutingDecision,
        policy: Policy,
        state: SimulationState,
        provider_rngs: dict[str, np.random.Generator],
    ) -> RoutingOutcome:
        providers = state.providers
        primary = providers[decision.primary_provider]
        now = float(request.timestamp)
        state.now = now

        primary_rng = provider_rngs[primary.name]
        primary_ttft_ms = primary.sample_ttft(primary_rng, now)
        primary_service_time = _sample_service_time(
            primary,
            primary_rng,
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
            primary_rng = provider_rngs[primary.name]
            primary_ttft_ms = primary.sample_ttft(primary_rng, now)
            primary_service_time = _sample_service_time(
                primary,
                primary_rng,
                now,
                request.response_tokens or 1,
                primary_ttft_ms,
            )

        primary_cached_input_tokens = cached_input_tokens(primary, request, state)
        primary.account_request(request.id, now, primary_service_time)
        primary_cost_usd = primary.marginal_cost_for_request(
            request,
            now,
            cached_input_tokens=primary_cached_input_tokens,
        )
        billed_cost = primary_cost_usd

        final_provider = primary.name
        final_ttft_ms = primary_ttft_ms
        hedge_triggered = False
        hedge_dispatch: HedgeDispatch | None = None
        backup_ttft_ms: float | None = None
        backup_observed_at: float | None = None
        backup_cost_usd = 0.0
        backup_cached_input_tokens = 0
        hedge_delay_ms: float | None = None

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
            backup_rng = provider_rngs[backup.name]
            backup_ttft_ms = backup.sample_ttft(backup_rng, dispatch_time)
            backup_service_time = _sample_service_time(
                backup,
                backup_rng,
                dispatch_time,
                request.response_tokens or 1,
                backup_ttft_ms,
            )
            if _can_admit(backup, dispatch_time, backup_service_time):
                backup_cached_input_tokens = cached_input_tokens(backup, request, state)
                backup.account_request(
                    request.id + _HEDGE_REQUEST_ID_OFFSET,
                    dispatch_time,
                    backup_service_time,
                )
                backup_cost_usd = backup.marginal_cost_for_request(
                    request,
                    dispatch_time,
                    cached_input_tokens=backup_cached_input_tokens,
                )
                billed_cost += backup_cost_usd
                hedge_delay_ms = (dispatch_time - now) * 1000.0
                backup_observed_at = dispatch_time + backup_ttft_ms / 1000.0
                hedged_ttft = (
                    (dispatch_time - now) * 1000.0 + self.dispatch_overhead_ms + backup_ttft_ms
                )
                if hedged_ttft < final_ttft_ms:
                    final_ttft_ms = hedged_ttft
                    final_provider = backup.name
                hedge_triggered = True

                # TODO(routewise-hedging): model cancellation after the first
                # visible token wins the hedge race. Today both primary and
                # backup consume their full modeled service time and billed
                # cost. A cancel-aware simulator should truncate loser
                # concurrency occupancy at winner first-token time and record
                # both no-cancel and cancel-adjusted costs.

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
                "hedge_delay_ms": hedge_delay_ms,
                "backup_ttft_ms": backup_ttft_ms,
                "backup_observed_at": backup_observed_at,
                "primary_cost_usd": primary_cost_usd,
                "backup_cost_usd": backup_cost_usd,
                "primary_cached_input_tokens": primary_cached_input_tokens,
                "backup_cached_input_tokens": backup_cached_input_tokens,
                "sim_dispatch_overhead_ms": self.dispatch_overhead_ms,
            },
        )

    def _build_record(
        self,
        *,
        request: Request,
        decision: RoutingDecision,
        outcome: RoutingOutcome,
        providers: dict[str, Provider],
        policy_name: str,
    ) -> PerRequestRecord:
        primary_provider = providers.get(outcome.primary_provider)
        final_provider = providers.get(outcome.final_provider)
        backup_name = outcome.metadata.get("hedge_provider") if outcome.hedge_triggered else None
        backup_provider = providers.get(backup_name) if backup_name else None
        slo_ms = float(request.slo_ms or self.scenario.primary_slo_ms)
        backup_cost = outcome.metadata.get("backup_cost_usd")
        hedge_winner = None
        if outcome.hedge_triggered:
            hedge_winner = "backup" if outcome.final_provider == backup_name else "primary"

        metadata: dict[str, object] = {
            "sim_quota_fraction_used": self._quota_fraction_by_provider(
                providers,
                float(request.timestamp),
            ),
            "sim_concurrency_utilization": self._concurrency_utilization_by_provider(
                providers,
                float(request.timestamp),
            ),
        }
        if "weights" in decision.metadata:
            metadata["sim_lp_weights"] = decision.metadata["weights"]
        if "budget" in decision.metadata:
            metadata["sim_lp_budget"] = decision.metadata["budget"]
        if "c_eff" in decision.metadata:
            metadata["sim_c_eff"] = decision.metadata["c_eff"]
        for key in (
            "primary_cached_input_tokens",
            "backup_cached_input_tokens",
        ):
            if key in outcome.metadata:
                metadata[key] = outcome.metadata[key]
        if final_provider is not None:
            metadata["sim_true_p99_ms"] = final_provider.true_p99_ms(float(request.timestamp))
        for key in (
            "primary_observed_at",
            "backup_observed_at",
            "sim_dispatch_overhead_ms",
        ):
            if key in outcome.metadata:
                metadata[key] = outcome.metadata[key]
        if outcome.error:
            metadata["sim_error"] = outcome.error

        return PerRequestRecord(
            request_id=str(outcome.request_id),
            elapsed_sec=float(request.timestamp),
            policy=policy_name,
            prompt_tokens=int(request.request_tokens),
            completion_tokens_budget=(
                request.estimated_response_tokens
                if request.estimated_response_tokens is not None
                else request.response_tokens
            ),
            completion_tokens_actual=None if outcome.rejected else request.response_tokens,
            primary_provider=outcome.primary_provider,
            primary_tier=_provider_tier_value(primary_provider),
            final_provider=outcome.final_provider,
            final_tier=_provider_tier_value(final_provider),
            backup_provider=str(backup_name) if backup_name else None,
            backup_tier=_provider_tier_value(backup_provider) if backup_provider else None,
            ttft_ms=float(outcome.ttft_ms),
            e2e_ms=None,
            primary_local_ttft_ms=outcome.metadata.get("primary_ttft_ms"),
            backup_local_ttft_ms=(
                outcome.metadata.get("backup_ttft_ms") if outcome.hedge_triggered else None
            ),
            slo_ms=slo_ms,
            slo_violated=outcome.rejected or float(outcome.ttft_ms) > slo_ms,
            total_cost_usd=float(outcome.cost_usd),
            primary_cost_usd=outcome.metadata.get("primary_cost_usd", 0.0),
            backup_cost_usd=backup_cost if outcome.hedge_triggered else None,
            hedge_triggered=outcome.hedge_triggered,
            hedge_delay_ms=outcome.metadata.get("hedge_delay_ms"),
            hedge_winner=hedge_winner,
            status=Status.REJECTED if outcome.rejected else Status.SUCCESS,
            error_class=outcome.error,
            metadata=metadata,
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

    @staticmethod
    def _quota_fraction_by_provider(
        providers: dict[str, Provider],
        now: float,
    ) -> dict[str, float]:
        return {
            name: float(provider.quota.fraction_used(now))
            for name, provider in providers.items()
            if provider.quota is not None
        }

    @staticmethod
    def _concurrency_utilization_by_provider(
        providers: dict[str, Provider],
        now: float,
    ) -> dict[str, float]:
        return {
            name: float(provider.concurrency.utilization(now))
            for name, provider in providers.items()
            if provider.concurrency is not None
        }


def _provider_tier_value(provider: Provider | None) -> str:
    if provider is None:
        return ""
    return provider.tier.value


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
    return min(
        available,
        key=lambda provider: (
            provider.effective_input_cost_per_token,
            provider.effective_output_cost_per_token,
            provider.true_p50_ms(now),
        ),
    )


def _request_user_id(request: Request) -> str | None:
    for key in ("user_id", "session_id", "sharegpt_conversation_id"):
        value = request.metadata.get(key)
        if value is not None:
            return str(value)
    return None


def _provider_rngs(seed: int, providers: Sequence[Provider]) -> dict[str, np.random.Generator]:
    """Create independent, reproducible RNG streams for provider sampling."""
    rngs: dict[str, np.random.Generator] = {}
    for provider in providers:
        if provider.name in rngs:
            raise ValueError(f"duplicate provider name: {provider.name!r}")
        digest = hashlib.blake2b(
            f"{seed}:{provider.name}".encode(),
            digest_size=16,
        ).digest()
        rngs[provider.name] = np.random.default_rng(int.from_bytes(digest, "little"))
    return rngs


__all__ = ["Simulator"]
