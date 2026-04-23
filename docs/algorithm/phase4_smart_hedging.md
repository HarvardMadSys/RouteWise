# Phase 4: Smart Hedging

## 1. Overview

### 1.1 Objective

Implement **smart hedging** to reduce tail latency while controlling cost overhead. When a request to the primary provider is at risk of violating the SLO, proactively send a duplicate request to a faster backup provider.

### 1.2 Latency Metric

Phase 4 targets **TTFT** (time-to-first-token) to primarily capture network + queueing + prefill variability, consistent with Phase 3 profiling.

### 1.3 Key Insight (from advisor)

> "If we have a SLO of x milliseconds, we can calculate the probability of maintaining SLO if we continue to wait. At certain point, we have to send request to a different provider so we can use the first response back to reply user."

Mathematical formulation:
- Given elapsed time `t` and SLO `L`
- **Hedge condition**: `P(T_primary > L | T_primary > t) > P(T_backup > L - t)`
- Equivalently: `S_primary(L) / S_primary(t) > S_backup(L - t)`

### 1.4 Scope

- **In scope**: Smart hedging logic, simulation evaluation, cost modeling
- **Out of scope (Phase 5)**: Real OpenRouter evaluation, production deployment

### 1.5 Relationship to Phase 3

Phase 4 is an **add-on layer** to Phase 3:
1. Phase 3 LP-Mix selects the **primary provider**
2. Phase 4 decides **when to hedge** and **which backup** to use
3. Final TTFT = `min(T_primary, h + T_backup)` if hedged, else `T_primary`

### 1.6 Modeling Assumptions (Simulation)

- **Causality**: At decision time `now`, hedging uses only profiles built from probe samples with timestamps `< now`.
- **Latency independence (approx.)**: Primary/backup latencies are sampled independently across providers. This is an approximation; correlated queueing bursts are analyzed as a limitation.
- **Failure time**: A failure is represented by `TTFT = 30s` (timeout) in simulation. This simplification is acceptable because Phase 4 primarily targets **queueing tail** (TTFT heavy-tail), not error handling. Failures trigger hedging explicitly in the simulation logic (see Section 9).
- **Hedging scope**: Smart hedging is designed as a **straggler mitigation** technique, not an error recovery mechanism. The hedge decision logic operates on successful latency samples only.

---

## 2. Hedging Strategies

### 2.1 Strategy Comparison

| Strategy | Description | Cost | Latency |
|----------|-------------|------|---------|
| **Never-hedge** | Phase 3 LP-Mix only | 1x | Baseline |
| **Always-hedge** | Always send 2 requests | ~2x | Optimal |
| **Fixed-timeout** | Hedge if no response after `h = αL` | 1-2x | Good |
| **Smart-hedge** | Hedge based on conditional probability | 1-2x | Near-optimal |

### 2.2 Never-hedge (Baseline)

```python
def never_hedge(primary: str, now: float) -> float:
    """Phase 3 LP-Mix, no hedging."""
    ttft_sec, _ = sample_latency(primary, now)
    return ttft_sec
```

### 2.3 Always-hedge (Upper Bound)

```python
def always_hedge(primary: str, backup: str, now: float) -> float:
    """Always send two requests, use the faster one."""
    ttft_primary_sec, err_primary = sample_latency(primary, now)
    ttft_backup_sec, err_backup = sample_latency(backup, now)

    if err_primary is not None:
        return ttft_backup_sec
    if err_backup is not None:
        return ttft_primary_sec
    return min(ttft_primary_sec, ttft_backup_sec)
```

Cost is `2x` baseline since both requests are billed regardless of which one is used.

### 2.4 Fixed-timeout Hedge

```python
def fixed_timeout_hedge(
    primary: str,
    backup: str,
    now: float,
    alpha: float,
    slo_sec: float,
    dispatch_overhead_sec: float = 0.0,
) -> float:
    """
    Hedge if primary doesn't respond within h = alpha * L.

    Args:
        alpha: Timeout fraction, e.g., 0.5, 0.7, 0.9
        slo_sec: SLO in seconds
        dispatch_overhead_sec: Backup launch overhead (network + scheduling).
    """
    h = alpha * slo_sec
    ttft_primary_sec, err_primary = sample_latency(primary, now)

    if err_primary is not None or ttft_primary_sec > h:
        # Hedge triggered
        ttft_backup_sec, err_backup = sample_latency(backup, now)
        if err_backup is not None:
            return ttft_primary_sec
        return min(ttft_primary_sec, h + dispatch_overhead_sec + ttft_backup_sec)
    return ttft_primary_sec
```

### 2.5 Smart-hedge (Main Contribution)

**Key Design Choice: SLO-first, not cost-aware.**

The smart-hedge strategies are designed as **SLO-first upper bounds**: they hedge as early as possible whenever hedging reduces the conditional violation probability, **without considering cost**. This positions them as:

| Strategy | Positioning |
|----------|-------------|
| `fixed_timeout_0.5` | **Cost-performance baseline** (low hedge rate, low cost overhead) |
| `smart_survival` | **SLO-first upper bound** (aggressive hedging, higher cost) |
| `always` | **Theoretical lower bound for latency** (maximum cost) |

A **cost-aware** variant would weigh the expected benefit against the hedge cost:
```
hedge if: P(violation_reduction) > threshold * (backup_cost / primary_cost)
```
This is out of scope for Phase 4 but noted as future work.

Two SLO-first variants implemented for ablation:

#### 2.5.1 Survival Threshold Method

```python
def smart_hedge_survival(
    primary: str,
    backup: str,
    elapsed: float,
    slo_sec: float,
    profiles: Dict[str, ProviderProfile],
    now: float,
    dispatch_overhead_sec: float = 0.0,
) -> bool:
    """
    Hedge condition based on conditional survival probability.

    P(violation if wait) > P(violation if hedge)

    S_primary(L) / S_primary(elapsed) > S_backup(L - elapsed - delta)
    """
    # Primary's conditional violation probability
    F_primary_L = profiles[primary].get_cdf_at(slo_sec, now)
    F_primary_elapsed = profiles[primary].get_cdf_at(elapsed, now)

    S_primary_L = 1 - F_primary_L
    S_primary_elapsed = 1 - F_primary_elapsed

    if S_primary_elapsed < 1e-6:
        return True  # Already exceeded all samples, must hedge

    P_violation_wait = S_primary_L / S_primary_elapsed

    # Backup's violation probability
    remaining_budget = slo_sec - elapsed - dispatch_overhead_sec
    if remaining_budget <= 0:
        return True  # No time left, must hedge

    F_backup = profiles[backup].get_cdf_at(remaining_budget, now)
    P_violation_hedge = 1 - F_backup

    return P_violation_wait > P_violation_hedge
```

#### 2.5.2 Residual Life Method (Advisor's Original)

```python
def smart_hedge_residual(
    primary: str,
    backup: str,
    elapsed: float,
    slo_sec: float,
    profiles: Dict[str, ProviderProfile],
    now: float,
    dispatch_overhead_sec: float = 0.0,
) -> bool:
    """
    Hedge condition based on residual life expectation.

    From advisor: "E[latency | elapsed time] + E[latency of fastest provider] > SLO"

    Formally: elapsed + E[T_primary - elapsed | T_primary > elapsed] +
              delta + E[T_backup] > SLO
    """
    E_remaining = compute_conditional_expectation(
        profiles[primary],
        elapsed,
        now=now,
    )
    E_backup = compute_expected_latency(profiles[backup], now=now)

    return elapsed + E_remaining + dispatch_overhead_sec + E_backup > slo_sec
```

**Stability protections for residual life estimation**:

```python
def compute_conditional_expectation(
    profile: ProviderProfile,
    elapsed: float,
    now: float,
    min_samples: int = 5,
    max_latency: float = 30.0,
) -> float:
    """
    E[T - elapsed | T > elapsed] with stability protections.

    Args:
        now: Decision timestamp. Uses samples with timestamps < now.
        min_samples: Minimum surviving samples required
        max_latency: Upper bound truncation to avoid outlier inflation
    """
    # Use long-window for stability (consistent with Phase 3)
    samples = profile.get_samples_before(now, use_short=False)
    survived = [s for s in samples if s > elapsed]

    # Protection 1: Minimum sample threshold
    if len(survived) < min_samples:
        # Fallback: use P99 as conservative upper bound
        return profile.get_p99() - elapsed

    # Protection 2: Upper bound truncation
    survived_capped = [min(s, max_latency) for s in survived]

    return np.mean(survived_capped) - elapsed
```

---

## 3. Backup Provider Selection

### 3.1 Default: Fastest Provider

```python
def get_fastest_provider(profiles: Dict[str, ProviderProfile], current_time: float) -> str:
    """Select provider with lowest P50 TTFT in long-window."""
    return min(
        profiles.keys(),
        key=lambda p: np.percentile(
            profiles[p].get_samples_before(current_time, use_short=False),
            50
        )
    )
```

### 3.2 Ablation: LP Other Endpoint

Since Phase 3 LP typically selects 2 providers, use the "other" one as backup:

```python
def get_lp_backup(lp_weights: Dict[str, float], primary: str) -> Optional[str]:
    """Get the other provider from LP solution."""
    candidates = [p for p in lp_weights.keys() if p != primary and lp_weights[p] > 0.01]
    if candidates:
        return candidates[0]
    return None
```

### 3.3 Ablation: Cheapest Viable

```python
def get_cheapest_viable(
    profiles: Dict[str, ProviderProfile],
    costs: Dict[str, float],
    slo_sec: float,
    now: float,
    min_cdf: float = 0.90,
) -> Optional[str]:
    """Select cheapest provider that can likely meet SLO."""
    viable = [
        p for p in profiles.keys()
        if profiles[p].get_cdf_at(slo_sec, now) >= min_cdf
    ]
    if viable:
        return min(viable, key=lambda p: costs[p])
    return None
```

---

## 4. Cost Modeling

### 4.1 OpenRouter Billing Model

OpenRouter bills for **all requests**, including cancelled ones:
- Client can disconnect, but tokens already generated are still billed
- For hedging, both primary and backup requests are billed in full

### 4.2 Total Cost Calculation

```python
@dataclass
class HedgingResult:
    primary_provider: str
    backup_provider: Optional[str]
    hedged: bool
    hedge_time_sec: Optional[float]
    winner: str  # "primary" or "backup"
    final_ttft_ms: float
    primary_cost: float
    backup_cost: float

    @property
    def total_cost(self) -> float:
        if not self.hedged:
            return self.primary_cost
        # Both requests are billed regardless of winner
        return self.primary_cost + self.backup_cost
```

---

## 5. Simulation Design

### 5.1 Hedging Simulation Logic

```python
def simulate_hedged_request(
    primary: str,
    backup: str,
    strategy: str,
    params: HedgingParams,
    profiles: Dict[str, ProviderProfile],
    costs: Dict[str, float],
    now: float,
) -> HedgingResult:
    """
    Simulate a single request with hedging.

    Key: Sample BOTH T_primary and T_backup upfront (for reproducibility),
    but only "use" T_backup if hedge is triggered.
    """
    # Sample latencies
    T_primary, err_primary = sample_latency(primary, now)
    T_backup, err_backup = sample_latency(backup, now)

    # Determine hedge time based on strategy
    if strategy == "never":
        hedged = False
        h = None
    elif strategy == "always":
        hedged = True
        h = 0.0  # Immediate
    elif strategy == "fixed_timeout":
        h = params.alpha * params.slo_sec
        hedged = (T_primary > h) or (err_primary is not None)
    elif strategy == "smart_survival":
        h = find_optimal_hedge_time_survival(
            primary,
            backup,
            profiles,
            now=now,
            slo_sec=params.slo_sec,
            dispatch_overhead_sec=params.dispatch_overhead_sec,
        )
        hedged = (T_primary > h) or (err_primary is not None)
    elif strategy == "smart_residual":
        h = find_optimal_hedge_time_residual(
            primary,
            backup,
            profiles,
            now=now,
            slo_sec=params.slo_sec,
            dispatch_overhead_sec=params.dispatch_overhead_sec,
        )
        hedged = (T_primary > h) or (err_primary is not None)

    # Compute final TTFT
    if not hedged:
        final_ttft = T_primary
        winner = "primary"
        backup_cost = 0.0
    else:
        # Both requests in flight
        if err_primary is not None:
            # Primary failed, use backup
            final_ttft = h + params.dispatch_overhead_sec + T_backup
            winner = "backup"
        elif err_backup is not None:
            # Backup failed, use primary
            final_ttft = T_primary
            winner = "primary"
        else:
            # Both succeeded, use faster
            if T_primary <= h + params.dispatch_overhead_sec + T_backup:
                final_ttft = T_primary
                winner = "primary"
            else:
                final_ttft = h + params.dispatch_overhead_sec + T_backup
                winner = "backup"

        backup_cost = costs[backup]

    return HedgingResult(
        primary_provider=primary,
        backup_provider=backup if hedged else None,
        hedged=hedged,
        hedge_time_sec=h,
        winner=winner,
        final_ttft_ms=final_ttft * 1000,
        primary_cost=costs[primary],
        backup_cost=backup_cost,
    )
```

### 5.2 Finding Optimal Hedge Time

For smart-hedge, we need to find the optimal `h` where hedging becomes beneficial:

```python
def find_optimal_hedge_time_survival(
    primary: str,
    backup: str,
    profiles: Dict[str, ProviderProfile],
    now: float,
    slo_sec: float,
    dispatch_overhead_sec: float = 0.0,
    resolution: float = 0.1,
) -> float:
    """
    Find minimum h where P(violation|wait) > P(violation|hedge).

    Grid search with resolution steps.
    """
    for h in np.arange(0, slo_sec, resolution):
        if smart_hedge_survival(
            primary,
            backup,
            elapsed=h,
            slo_sec=slo_sec,
            profiles=profiles,
            now=now,
            dispatch_overhead_sec=dispatch_overhead_sec,
        ):
            return h
    return slo_sec  # Never hedge if condition never met
```

---

## 6. Experiment Design

### 6.1 Experiment Matrix

| Dimension | Values |
|-----------|--------|
| **Strategy** | never, always, fixed_0.5, fixed_0.7, fixed_0.9, smart_survival, smart_residual |
| **SLO (L)** | {1, 2, 3, 5, 10} seconds |
| **Backup selection** | fastest, lp_other, cheapest_viable |

### 6.2 Metrics

| Metric | Definition |
|--------|------------|
| **Hedge Rate** | Fraction of requests that triggered hedging |
| **Cost Overhead** | `(total_cost - never_hedge_cost) / never_hedge_cost` |
| **P99 Reduction** | `(never_hedge_p99 - hedged_p99) / never_hedge_p99` |
| **Violation Rate** | Fraction of requests with TTFT > SLO (including errors) |
| **Error Rate** | Fraction of failed requests |

### 6.3 Per-Request Logging

```python
@dataclass
class HedgingLogEntry:
    timestamp: float
    primary_provider: str
    backup_provider: Optional[str]
    hedged: bool
    hedge_time_sec: Optional[float]
    winner: str
    final_ttft_ms: float
    slo_violated: bool
    error: Optional[str]
    primary_cost: float
    backup_cost: float
    total_cost: float
    lp_weights: Dict[str, float]
    lp_status: str
```

### 6.4 Expected Figures

1. **p99_vs_slo.png**: P99 latency for each strategy across SLOs
2. **violation_rate_vs_slo.png**: SLO violation rate comparison
3. **hedge_rate_vs_slo.png**: How often hedging triggers
4. **cost_overhead_vs_slo.png**: Cost increase from hedging
5. **cost_vs_violation_pareto.png**: Pareto frontier of cost-violation trade-off

### 6.5 Robustness Ablation: INFINITY Survival Mode

To demonstrate robustness under alternative error handling, we include an **optional ablation** using INFINITY mode for survival computation:

```python
def get_survival_infinity_mode(profile, t, now, timeout_sec=30.0):
    """S(t) treating failures as T = timeout (always miss SLO)."""
    # ProviderProfile stores successful TTFT samples and exposes them via get_samples_before().
    # Use long-window samples for stability.
    success_samples = profile.get_samples_before(now, use_short=False)
    error_rate = profile.get_error_rate_before(now)

    # Failures contribute to survival (they "survive" past any finite t < timeout)
    n_success_survived = sum(1 for s in success_samples if s > t)
    S_success = n_success_survived / len(success_samples) if success_samples else 0.0

    # S_infinity(t) = (1 - err_rate) * S_success(t) + err_rate * I(t < timeout)
    return (1 - error_rate) * S_success + error_rate * (1.0 if t < timeout_sec else 0.0)
```

**Purpose**: Verify that main conclusions (smart > fixed > never for SLO compliance) hold even when errors are modeled as guaranteed SLO violations.

**Note**: This ablation is for **robustness verification only**. The main results use SEPARATE mode because:
1. Hedging is a **straggler mitigation** technique, not error recovery
2. Failures already trigger hedging explicitly in the simulation logic

---

## 7. Implementation Plan

### 7.1 File Structure

```
experiment/
├── strategies/
│   ├── online_latency_router.py    # Phase 3 (existing)
│   └── smart_hedging.py            # Phase 4 (new)
├── scripts/
│   ├── run_phase3_simulation.py    # Phase 3 (existing)
│   ├── run_phase4_simulation.py    # Phase 4 (new)
│   └── plot_phase4_results.py      # Phase 4 plots (new)
└── results/
    └── latency_phase4/             # Output
```

### 7.2 Implementation Order

1. **HedgingParams dataclass**: Configuration for hedging strategies
2. **HedgingResult dataclass**: Result with cost breakdown
3. **Hedge decision functions**: survival method, residual method
4. **Backup selection functions**: fastest, lp_other, cheapest_viable
5. **SmartHedger class**: Integrates with Phase 3 OnlineLatencyRouter
6. **Simulation harness**: Replay with hedging
7. **Plotting scripts**: 5 core figures

### 7.3 Testing Strategy

1. **Unit tests**: Hedge decision logic, cost calculation
2. **Sanity checks**:
   - always-hedge should have lowest P99
   - always-hedge should have highest cost (~2x)
   - smart-hedge should be between never and always (both P99 and cost)
3. **Deterministic replay**: Fixed seed for reproducibility

---

## 8. Milestones

### Phase 4a: Simulation (Current)

- [ ] Implement smart_hedging.py
- [ ] Implement run_phase4_simulation.py
- [ ] Run experiment matrix on Phase 1 data
- [ ] Generate 5 core figures
- [ ] Write ablation table

**Deliverables**: Figures showing hedge improves P99 with controlled cost overhead.

### Phase 4b: Real OpenRouter (Future)

- [ ] Budget: <$20 for confirmatory experiment
- [ ] 200-500 requests with select strategies
- [ ] Head-to-head vs OpenRouter default
- [ ] Validate simulation predictions

---

## 9. Failure Handling in Survival Functions

### 9.1 The Problem

Phase 3 introduced two failure modes for CDF computation:
- **INFINITY**: Failures are treated as `latency = ∞`, contributing to violation probability
- **SEPARATE**: Only successful samples contribute to CDF (conditional on success)

How should hedging decisions handle failures in the survival function `S(t)`?

### 9.2 Recommended Approach: SEPARATE for Hedging

For hedging decisions, we use **SEPARATE mode** (conditional CDF).

**Precise Probability Semantics**:

The survival ratio in smart-hedge is:
```
S_primary(L) / S_primary(t) = P(TTFT > L | TTFT > t, success)
```

This is **conditional on success**, NOT the unconditional `P(T > L | T > t)`.

**How failures are handled** (two-stage logic):
1. **Explicit failure trigger**: In the simulation loop, `if err_primary is not None: hedge = True` (run_phase4_simulation.py)
2. **Survival-based decision**: Only executed for non-failed requests, using success-only samples

This separation ensures:
- Failures **always** trigger hedging (correct behavior)
- Survival function measures **straggler risk**, not failure risk
- No probability conflation between "slow" and "failed"

**Rationale**:
1. **Failure is not a latency event**: A failure at `t=timeout` doesn't mean the request was slow; it means it never completed
2. **Hedging already handles failures**: The hedge trigger condition `err_primary is not None` covers the failure case explicitly
3. **Cleaner probability interpretation**: `P(TTFT > L | TTFT > t, success)` is well-defined; mixing failures conflates two different events

**Implementation**:

```python
def get_survival_for_hedging(
    profile: ProviderProfile,
    t: float,
    now: float,
) -> float:
    """
    S(t) for hedging decisions.

    Uses SEPARATE mode: only successful samples.
    Failures are handled separately in hedge trigger logic.
    """
    # ProviderProfile stores successful TTFT samples and exposes them via get_samples_before().
    samples = profile.get_samples_before(now, use_short=False)
    if not samples:
        return 1.0  # No data, assume high latency (conservative)

    n_survived = sum(1 for s in samples if s > t)
    return n_survived / len(samples)
```

### 9.3 Interaction with Phase 3 FailureMode

| Phase 3 FailureMode | Hedging Survival Mode | Reason |
|---------------------|----------------------|--------|
| INFINITY | SEPARATE | LP weights account for error rate; hedging uses conditional probability |
| SEPARATE | SEPARATE | Consistent treatment |

### 9.4 Complete Hedge Decision Logic

```python
def should_hedge(
    primary: str,
    elapsed: float,
    slo_sec: float,
    profiles: Dict[str, ProviderProfile],
    now: float,
    dispatch_overhead_sec: float,
    backup: str,
) -> bool:
    """
    Complete hedge decision including failure handling.

    Returns True if:
    1. Primary already failed (detected), OR
    2. Conditional violation probability exceeds backup's violation probability
    """
    # Case 1: Primary failed - always hedge
    # (This is checked in the simulation loop, not here)

    # Case 2: Survival-based decision (SEPARATE mode)
    S_primary_L = get_survival_for_hedging(profiles[primary], slo_sec, now)
    S_primary_t = get_survival_for_hedging(profiles[primary], elapsed, now)

    if S_primary_t < 1e-6:
        return True  # All samples already exceeded, must hedge

    P_violation_wait = S_primary_L / S_primary_t

    remaining = slo_sec - elapsed - dispatch_overhead_sec
    if remaining <= 0:
        return True

    S_backup_remaining = get_survival_for_hedging(profiles[backup], remaining, now)
    P_violation_hedge = S_backup_remaining  # P(backup > remaining)

    return P_violation_wait > P_violation_hedge
```

### 9.5 Why Not INFINITY for Hedging?

If we used INFINITY mode (failures = ∞ latency), the survival function would include failure probability:

```
S_infinity(t) = P(T > t) = P(success) * P(T > t | success) + P(failure) * 1
             = (1 - err_rate) * S_success(t) + err_rate
```

This leads to:
1. **Double-counting**: Failures already trigger hedging explicitly
2. **Confusing semantics**: `S(∞) > 0` because of failure mass
3. **Unstable estimates**: Error rate variance dominates at high percentiles

---

## 10. Open Questions (Defer to Phase 5)

1. **Streaming hedging**: Can we hedge mid-stream for long outputs?
2. **Multi-hedge**: Hedge to multiple backups simultaneously?
3. **Production integration**: How to integrate with OpenRouter's routing?

---

## Appendix A: Mathematical Details

### A.1 Conditional Survival Probability

Given that request has not completed by time `t`:

```
P(T > L | T > t) = P(T > L) / P(T > t) = S(L) / S(t)
```

where `S(x) = 1 - F(x)` is the survival function.

### A.2 Residual Life Expectation

The expected remaining time given survival to `t`:

```
E[T - t | T > t] = ∫_{t}^{∞} P(T > x | T > t) dx
                = ∫_{t}^{∞} S(x) / S(t) dx
```

Empirical estimate:

```
E[T - t | T > t] ≈ mean({x_i - t : x_i > t})
```

### A.3 Optimal Hedge Time

Find `h*` that minimizes expected violation probability:

```
h* = argmin_h [ P(T_primary > L, T_primary ≤ h) + P(min(T_primary, h + T_backup) > L, T_primary > h) ]
```

For independent T_primary and T_backup:

```
h* = argmin_h [ F_primary(L) - F_primary(h) + S_primary(h) * S_backup(L - h) ]
```

### A.4 Cost-Violation Trade-off

Expected cost:

```
E[Cost] = c_primary + P(hedge) * c_backup
```

where `P(hedge) = P(T_primary > h)` for fixed-timeout.

The Pareto frontier plots `E[Cost]` vs `P(violation)` for different strategies and parameters.
