"""Canonical request loop for RouteWise policies.

For concurrency capacity accounting, the simulator derives service time from
the selected provider model: sampled TTFT plus sampled decode time
(``response_tokens / TPS``), unless the provider defines an explicit service
time distribution. BurstGPT elapsed-time fields are preserved as trace metadata
but are not used by the simulator's main capacity model.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from routewise.capacity import ProviderTier
from routewise.const import DISPATCH_OVERHEAD_MS
from routewise.core import CheckpointBackupDispatch
from routewise.metrics import PerRequestRecord, Run, RunAggregator, Status
from routewise.schemas import Request, RoutingDecision, RoutingOutcome
from routewise.sim.engine.state import SimulationState
from routewise.sim.policies.prefix_cache import cached_input_tokens

if TYPE_CHECKING:
    from collections.abc import Sequence

    from routewise.sim.policies.base import Policy
    from routewise.sim.world.providers import Provider
    from routewise.sim.world.scenarios import ScenarioConfig


_HEDGE_REQUEST_ID_OFFSET = 10_000_000


@dataclass(frozen=True)
class _FallbackAdmission:
    provider: Provider
    ttft_ms: float
    service_time: float


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
            source="sim",
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
            fallback = _fallback_admission(
                providers.values(),
                now,
                request.response_tokens or 1,
                provider_rngs,
                exclude={primary.name},
            )
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
            primary = fallback.provider
            primary_ttft_ms = fallback.ttft_ms
            primary_service_time = fallback.service_time

        primary_cached_input_tokens = cached_input_tokens(primary, request, state)
        primary.account_request(request.id, now, primary_service_time)
        primary_cost_usd = primary.marginal_cost_for_request(
            request,
            now,
            cached_input_tokens=primary_cached_input_tokens,
        )
        primary_cancel_cost_usd = primary.marginal_prefill_cost_for_request(
            request,
            now,
            cached_input_tokens=primary_cached_input_tokens,
        )
        primary_uncanceled_cost_usd = primary_cost_usd
        billed_cost = primary_cost_usd

        final_provider = primary.name
        final_ttft_ms = primary_ttft_ms
        hedge_triggered = False
        checkpoint_backup_dispatch: CheckpointBackupDispatch[str] | None = None
        backup_ttft_ms: float | None = None
        backup_observed_at: float | None = None
        backup_start_time: float | None = None
        backup_full_service_end_at: float | None = None
        backup_cost_usd = 0.0
        backup_cancel_cost_usd = 0.0
        backup_uncanceled_cost_usd = 0.0
        backup_cached_input_tokens = 0
        hedge_delay_ms: float | None = None
        primary_observed_at = now + primary_ttft_ms / 1000.0
        primary_full_service_end_at = now + primary_service_time
        hedge_loser: str | None = None
        hedge_cancel_time: float | None = None
        hedge_loser_canceled = False

        for elapsed in sorted(decision.hedge_checkpoints):
            elapsed_ms = float(elapsed) * 1000.0
            if elapsed_ms >= primary_ttft_ms:
                break
            state.now = now + float(elapsed)
            tick_dispatch = policy.tick(request, decision, float(elapsed), state)
            if tick_dispatch is not None:
                checkpoint_backup_dispatch = CheckpointBackupDispatch(
                    backup=tick_dispatch.backup_provider,
                    elapsed_sec=float(elapsed),
                    metadata=tick_dispatch.metadata,
                )
                break

        if checkpoint_backup_dispatch is not None:
            backup = providers[checkpoint_backup_dispatch.backup]
            dispatch_time = state.now
            backup_start_time = dispatch_time + self.dispatch_overhead_ms / 1000.0
            backup_rng = provider_rngs[backup.name]
            backup_ttft_ms = backup.sample_ttft(backup_rng, backup_start_time)
            backup_service_time = _sample_service_time(
                backup,
                backup_rng,
                backup_start_time,
                request.response_tokens or 1,
                backup_ttft_ms,
            )
            if _can_admit(backup, backup_start_time, backup_service_time):
                backup_cached_input_tokens = cached_input_tokens(backup, request, state)
                backup.account_request(
                    request.id + _HEDGE_REQUEST_ID_OFFSET,
                    backup_start_time,
                    backup_service_time,
                )
                backup_full_service_end_at = backup_start_time + backup_service_time
                backup_cost_usd = backup.marginal_cost_for_request(
                    request,
                    backup_start_time,
                    cached_input_tokens=backup_cached_input_tokens,
                )
                backup_cancel_cost_usd = backup.marginal_prefill_cost_for_request(
                    request,
                    backup_start_time,
                    cached_input_tokens=backup_cached_input_tokens,
                )
                backup_uncanceled_cost_usd = backup_cost_usd
                hedge_delay_ms = (dispatch_time - now) * 1000.0
                backup_observed_at = backup_start_time + backup_ttft_ms / 1000.0
                hedged_ttft = (backup_observed_at - now) * 1000.0
                if hedged_ttft < final_ttft_ms:
                    final_ttft_ms = hedged_ttft
                    final_provider = backup.name
                    primary_cost_usd = primary_cancel_cost_usd
                    hedge_loser = "primary"
                    hedge_cancel_time = backup_observed_at
                    hedge_loser_canceled = primary.cancel_request(
                        request.id,
                        backup_observed_at,
                    )
                else:
                    backup_cost_usd = backup_cancel_cost_usd
                    hedge_loser = "backup"
                    hedge_cancel_time = primary_observed_at
                    hedge_loser_canceled = backup.cancel_request(
                        request.id + _HEDGE_REQUEST_ID_OFFSET,
                        primary_observed_at,
                    )
                billed_cost = primary_cost_usd + backup_cost_usd
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
                "primary_observed_at": primary_observed_at,
                "primary_full_service_end_at": primary_full_service_end_at,
                "hedge_provider": (
                    checkpoint_backup_dispatch.backup
                    if checkpoint_backup_dispatch
                    else None
                ),
                "hedge_delay_ms": hedge_delay_ms,
                "backup_ttft_ms": backup_ttft_ms,
                "backup_observed_at": backup_observed_at,
                "backup_start_time": backup_start_time,
                "backup_full_service_end_at": backup_full_service_end_at,
                "hedge_loser": hedge_loser,
                "hedge_cancel_time": hedge_cancel_time,
                "hedge_loser_canceled": hedge_loser_canceled,
                "primary_cost_usd": primary_cost_usd,
                "backup_cost_usd": backup_cost_usd,
                "primary_cancel_cost_usd": primary_cancel_cost_usd,
                "backup_cancel_cost_usd": backup_cancel_cost_usd,
                "primary_uncanceled_cost_usd": primary_uncanceled_cost_usd,
                "backup_uncanceled_cost_usd": backup_uncanceled_cost_usd,
                "primary_cached_input_tokens": primary_cached_input_tokens,
                "backup_cached_input_tokens": backup_cached_input_tokens,
                "sim_dispatch_overhead_ms": self.dispatch_overhead_ms,
                "sim_cancel_assumes_provider_cancelable": True,
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
        # Canonical LP state (promoted from decision metadata; was sim_lp_weights / sim_lp_budget).
        lp_weights = decision.metadata.get("weights")
        lp_budget_usd = decision.metadata.get("budget")
        if "c_eff" in decision.metadata:
            # sim-only oracle quantity (no canonical equivalent — per-provider effective cost
            # is reproducible only inside the simulator).
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
            "primary_full_service_end_at",
            "backup_observed_at",
            "backup_start_time",
            "backup_full_service_end_at",
            "hedge_loser",
            "hedge_cancel_time",
            "hedge_loser_canceled",
            "primary_cancel_cost_usd",
            "backup_cancel_cost_usd",
            "primary_uncanceled_cost_usd",
            "backup_uncanceled_cost_usd",
            "sim_cancel_assumes_provider_cancelable",
            "sim_dispatch_overhead_ms",
        ):
            if key in outcome.metadata:
                metadata[key] = outcome.metadata[key]
        if outcome.error:
            metadata["sim_error"] = outcome.error

        # Hedging schedule / algorithm metadata (canonical fields).
        # SIM main path runs probability-target hedging on SLO-relative checkpoints.
        hedging_active = bool(decision.hedge_checkpoints)
        hedge_algorithm = "probability_target" if hedging_active else "disabled"
        hedge_schedule = "slo_relative_checkpoints" if hedging_active else None

        # routing_estimated_cost_usd is the LP's decision-time cost estimate
        # (predicted tokens, not realized). The policy emits it on
        # RoutingDecision.metadata; outcome.cost_usd is a different (realized)
        # quantity and must not be substituted here.
        routing_estimated_cost_usd: float | None = decision.metadata.get(
            "routing_estimated_cost_usd"
        )
        # In SIM the backup is chosen at a hedge checkpoint (tick), not at route
        # time, so only the primary leg has a route-time estimate.
        primary_routing_estimated_cost_usd: float | None = routing_estimated_cost_usd
        backup_routing_estimated_cost_usd: float | None = None

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
            model=request.model,
            source="sim",
            timestamp_sec=None,
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
            routing_estimated_cost_usd=routing_estimated_cost_usd,
            primary_routing_estimated_cost_usd=primary_routing_estimated_cost_usd,
            backup_routing_estimated_cost_usd=backup_routing_estimated_cost_usd,
            physical_cost_usd=None,
            primary_physical_cost_usd=None,
            backup_physical_cost_usd=None,
            hedge_triggered=outcome.hedge_triggered,
            hedge_delay_ms=outcome.metadata.get("hedge_delay_ms"),
            hedge_winner=hedge_winner,
            hedge_algorithm=hedge_algorithm,
            hedge_schedule=hedge_schedule,
            lp_weights=lp_weights,
            lp_budget_usd=lp_budget_usd,
            lp_status=decision.metadata.get("lp_status"),
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


def _fallback_admission(
    providers: Sequence[Provider],
    now: float,
    output_tokens: int,
    provider_rngs: dict[str, np.random.Generator],
    *,
    exclude: set[str] | None = None,
) -> _FallbackAdmission | None:
    excluded = exclude or set()
    candidates = sorted(
        (
            provider
            for provider in providers
            if provider.name not in excluded and provider.is_available(now)
        ),
        key=lambda provider: (
            provider.effective_input_cost_per_token,
            provider.effective_output_cost_per_token,
            provider.true_p50_ms(now),
        ),
    )
    for provider in candidates:
        rng = provider_rngs[provider.name]
        rng_state = copy.deepcopy(rng.bit_generator.state)
        ttft_ms = provider.sample_ttft(rng, now)
        service_time = _sample_service_time(provider, rng, now, output_tokens, ttft_ms)
        if _can_admit(provider, now, service_time):
            return _FallbackAdmission(
                provider=provider,
                ttft_ms=ttft_ms,
                service_time=service_time,
            )
        rng.bit_generator.state = rng_state
    return None


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
