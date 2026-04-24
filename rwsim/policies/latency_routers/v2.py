"""V2 Router: Probe-rank by P50, hedge for tail.

Replaces LP mixing (min cost s.t. F(SLO) >= 0.99) with a simpler
Pareto-optimal P50 ranking. The primary provider is selected as the
cheapest provider within a near-best P50 band.

Justified by three analyses:
- ACF: P50 is predictable from probing, P99 is not
- LP diversification: CDF constraint over-diversifies
- Counterfactual: LP can't beat strongest fixed baseline

Drop-in replacement for OnlineLatencyRouter (same interface).
"""

from __future__ import annotations

import time
import numpy as np

from rwsim.policies.latency_routers.online_lp import (
    FailureMode,
    ProviderProfile,
    pre_filter,
)


class V2Router:
    """P50-rank router with Pareto filtering and near-best-band selection.

    Selection rule (every update_interval):
      1. Compute P50 for each provider from rolling profile window.
      2. Apply pre_filter (error rate, min CDF hard constraints).
      3. Pareto prune: remove providers dominated on (P50, cost).
      4. Near-best P50 band: best_p50 * (1 + p50_band).
      5. Primary = cheapest provider in the band.

    Attributes match OnlineLatencyRouter for compatibility with
    phase5_online_evaluation.py and SmartHedger.
    """

    def __init__(
        self,
        costs: dict[str, float],
        slo_sec: float = 2.0,
        window_sec: float = 15 * 60,
        failure_mode: FailureMode = FailureMode.INFINITY,
        update_interval: float = 300.0,
        p50_band: float = 0.10,
        # Compatibility params (unused but accepted for interface parity).
        kappa: float = 0.0,
        weight_smoothing: float = 0.3,
        lp_update_interval: float | None = None,
        relaxation_factors: list[float] | None = None,
    ):
        """Initialize the V2 router.

        Args:
            costs: Provider name -> cost per request.
            slo_sec: SLO latency threshold in seconds.
            window_sec: Rolling window for profile samples (default 15 min).
            failure_mode: How to handle failures in CDF estimation.
            update_interval: Seconds between ranking updates.
            p50_band: Fractional band around best P50 (0.10 = 10%).
        """
        self.costs = costs
        self.slo_sec = slo_sec
        self.window_sec = window_sec
        self.failure_mode = failure_mode
        self.update_interval = (
            lp_update_interval if lp_update_interval is not None else update_interval
        )
        self.p50_band = p50_band

        # Initialize provider profiles (same as OnlineLatencyRouter).
        self.profiles: dict[str, ProviderProfile] = {}
        for provider in costs:
            self.profiles[provider] = ProviderProfile(
                provider=provider,
                window_sec=window_sec,
                failure_mode=failure_mode,
            )

        # Routing state.
        self.primary_provider: str | None = None
        self.pareto_set: list[str] = []
        self.is_collapsed: bool = False
        self._provider_p50s: dict[str, float] = {}

        # Compatibility with OnlineLatencyRouter / SmartHedger interface.
        self.last_lp_update: float = 0.0
        self.last_lp_status: str = "not_run"
        self.last_lp_weights: dict[str, float] = {}
        # SWRR-compatible sampler shim (always returns primary).
        self.sampler = _SingleProviderSampler()

    def add_sample(
        self,
        provider: str,
        timestamp: float,
        ttft_ms: float,
        error_type: str | None = None,
    ) -> None:
        """Record a latency observation. Passthrough to ProviderProfile."""
        if provider not in self.profiles:
            return
        self.profiles[provider].add_sample(timestamp, ttft_ms, error_type)

    def update_lp(
        self,
        current_time: float | None = None,
        force: bool = False,
    ) -> bool:
        """Recompute P50 ranking (named update_lp for interface compat).

        Returns True if ranking was updated, False if skipped.
        """
        if current_time is None:
            current_time = time.time()

        if not force and (current_time - self.last_lp_update) < self.update_interval:
            return False

        # Step 1: Get eligible providers via hard filter.
        # BUGFIX: pre_filter's L_min defaults to 1.0 s, but V2's SLO is
        # configurable (default 2.0 s). Tying L_min to slo_sec makes the
        # "obviously broken" filter match the SLO we actually care about,
        # rather than a hardcoded 1 s threshold that is neither a probing
        # signal nor aligned with routing intent.
        eligible = pre_filter(self.profiles, current_time, L_min=self.slo_sec)
        if not eligible:
            eligible = list(self.profiles.keys())

        # Step 2: Compute P50 for each eligible provider.
        provider_p50s: dict[str, float] = {}
        for prov in eligible:
            samples = self.profiles[prov].get_samples_before(current_time)
            if not samples:
                continue
            provider_p50s[prov] = float(np.median(samples))

        if not provider_p50s:
            # No data yet -- keep current state.
            self.last_lp_update = current_time
            self.last_lp_status = "no_data"
            return True

        # Step 3: Pareto prune on (P50, cost).
        pareto = self._pareto_filter(provider_p50s)

        # Step 4: Near-best P50 band.
        best_p50 = min(provider_p50s[p] for p in pareto)
        band_threshold = best_p50 * (1.0 + self.p50_band)
        in_band = [p for p in pareto if provider_p50s[p] <= band_threshold]

        # Step 5: Cheapest in band.
        primary = min(in_band, key=lambda p: self.costs.get(p, float("inf")))

        # Update state.
        self.primary_provider = primary
        self.pareto_set = pareto
        self._provider_p50s = provider_p50s

        # Collapse detection (diagnostic only).
        cheapest_overall = min(self.costs, key=lambda p: self.costs[p])
        self.is_collapsed = (primary == cheapest_overall)

        # Compatibility fields.
        self.last_lp_weights = {primary: 1.0}
        self.last_lp_status = (
            f"v2_pareto_{len(pareto)}_band_{len(in_band)}"
            + ("_collapsed" if self.is_collapsed else "")
        )
        self.last_lp_update = current_time
        self.sampler.set_provider(primary)

        return True

    def route(self, current_time: float | None = None) -> str | None:
        """Select provider for the next request.

        Always returns self.primary_provider (single-provider routing).
        Triggers ranking update if interval has elapsed.
        """
        if current_time is None:
            current_time = time.time()
        self.update_lp(current_time)
        return self.primary_provider

    def get_status(self) -> dict:
        """Return current routing state for debugging."""
        return {
            "lp_status": self.last_lp_status,
            "lp_weights": self.last_lp_weights,
            "sampler_weights": self.last_lp_weights,
            "primary": self.primary_provider,
            "pareto_set": self.pareto_set,
            "is_collapsed": self.is_collapsed,
            "provider_p50s": self._provider_p50s,
            "profile_summary": {
                prov: {
                    "p50": self._provider_p50s.get(prov),
                    "cost": self.costs.get(prov),
                    "error_rate": self.profiles[prov].get_error_rate_before(
                        self.last_lp_update
                    ) if self.last_lp_update > 0 else None,
                }
                for prov in self.profiles
            },
        }

    def _pareto_filter(self, provider_p50s: dict[str, float]) -> list[str]:
        """Remove providers dominated on (P50, cost).

        A provider P is dominated if there exists Q with:
          P50(Q) <= P50(P) AND cost(Q) <= cost(P)
          with at least one strict inequality.

        Returns list of non-dominated providers, sorted by P50 ascending.
        """
        providers = list(provider_p50s.keys())
        non_dominated = []

        for p in providers:
            p50_p = provider_p50s[p]
            cost_p = self.costs.get(p, float("inf"))
            dominated = False

            for q in providers:
                if q == p:
                    continue
                p50_q = provider_p50s[q]
                cost_q = self.costs.get(q, float("inf"))

                if p50_q <= p50_p and cost_q <= cost_p:
                    if p50_q < p50_p or cost_q < cost_p:
                        dominated = True
                        break

            if not dominated:
                non_dominated.append(p)

        return sorted(non_dominated, key=lambda p: provider_p50s[p])


class _SingleProviderSampler:
    """Shim that mimics SWRRSampler but always returns one provider.

    Needed for compatibility: SmartHedger accesses router.sampler.
    """

    def __init__(self):
        self._provider: str | None = None
        self._weights: dict[str, float] = {}

    def set_provider(self, provider: str) -> None:
        self._provider = provider
        self._weights = {provider: 1.0}

    def next(self) -> str | None:
        return self._provider

    def get_weights(self) -> dict[str, float]:
        return self._weights

    def update_weights(self, new_weights: dict[str, float], smoothing: float = 0.3):
        # Accept but ignore -- V2 doesn't use SWRR smoothing.
        pass
