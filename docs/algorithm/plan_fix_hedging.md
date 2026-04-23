# Plan: Fix Smart Hedging - Formula/Implementation Alignment

## Background

### Problem Identified

The current smart hedging implementation has a fundamental inconsistency:
the **trigger formula** assumes serial execution (cancel primary, then send backup),
but the **actual execution** is parallel (keep primary running, send backup concurrently,
use whichever returns first).

Additionally, the paper, simulation, and production code each use **different algorithms**
while the paper text conflates them as if they were the same.

### Current State

| Component | Algorithm Used | Hedge Rate |
|---|---|---|
| Paper Section 4b (theory) | Survival formula (Eq. hedge_survival) | - |
| Paper Section 5 (experiment text) | Mentions "P90 threshold" + "residual-based" | - |
| Paper Appendix | "survival-analysis-based (Eq. hedge_survival)" | - |
| Phase 4 simulation (`smart_hedging.py`) | `smart_residual` (serial formula) | 7%-100% (unstable) |
| Phase 4 simulation (`smart_hedging.py`) | `smart_survival` (backup-dependent) | 28%-87% (unstable) |
| Phase 5 production (`phase5_online_evaluation.py`) | `PERCENTILE_BASED(P90)` | ~15% |

Key issues:
1. `smart_residual` formula: `t + E[rem] + delta + E[T_backup] > SLO`
   - Serial semantics (accounts for backup latency as if it runs after primary fails)
   - But code runs primary and backup **in parallel**
   - `E[T_backup]` term steals time budget from primary, triggering hedge too early
   - Hedge rate wildly varies by backup provider and SLO (1.2% to 100%)

2. `smart_survival` formula: `S_p(L)/S_p(t) > S_b(L-t-delta)`
   - Compares primary violation probability against backup's ability
   - Not a pure bug: it does encode "only hedge when backup is useful"
   - But when backup is fast (e.g., Groq), right side approaches 0, making trigger
     nearly unconditional -> 87% hedge rate
   - Hedge rate heavily dependent on backup provider choice

3. Production uses `PERCENTILE_BASED(P90)` which has no connection to either formula

### Root Cause

Juncheng's original description (from discussion notes):
> "if the E[latency | elapsed time] + E[latency of fastest provider] > P99,
>  then we **cancel** the current request and send a duplicate request"

This describes a **cancel-and-resend (serial)** strategy. The formula is correct for
serial execution. But the code implements **parallel duplicate hedge** (no cancel).

For parallel hedge, the completion time after hedge at time t is:
`t + min(R_primary, delta + T_backup)` (not the serial sum).
The mathematically rigorous trigger would compare:
- No-hedge violation: `P(T_p > L | T_p > t)`
- Hedge violation: `P(min(T_p, t + delta + T_b) > L | T_p > t)`

Simply removing `E[T_backup]` from the residual formula is not mathematically rigorous
(it ignores the min operation).

### Proposed Solution: Cost-Benefit Economic Model

Instead of ad-hoc heuristics (threshold + viability guard), we adopt a **cost-benefit
analysis** framework. The hedge decision is framed as: hedge when the expected cost
savings from avoiding a violation exceeds the cost of the backup request.

**Core formula**:
```
Hedge at time t  iff  P_viol(t) * F_b(remaining) > C_b / V
```

Where:
- `P_viol(t) = S_p(L) / S_p(t)` : conditional probability primary violates SLO,
  given it has already survived to time t
- `F_b(remaining)` : CDF of backup at `remaining = L - t - delta`, i.e., probability
  that backup finishes within the remaining SLO budget
- `C_b` : cost of the backup request (known from pricing)
- `V` : penalty for an SLO violation (tunable parameter reflecting business value)
- `C_b / V` : cost ratio threshold (single tunable parameter)

**Why this is better than the two-stage heuristic (theta + min_backup_cdf)**:

1. **Single parameter**: `C_b/V` replaces two parameters (`theta`, `min_backup_cdf`).
   Fewer knobs to tune, simpler to reason about.

2. **Economically interpretable**: The left side `P_viol * F_b` is the probability that
   hedging actually prevents a violation. The right side `C_b/V` is the cost-benefit
   threshold. Hedge when expected benefit > expected cost.

3. **Multiplicative tradeoff**: The two-stage heuristic applies AND logic (risk gate
   fires AND backup is viable). This means a request with 90% violation probability but
   only 40% backup viability is blocked (if min_cdf=0.5). The economic model allows
   tradeoff: `0.9 * 0.4 = 0.36` may still exceed `C_b/V` if violations are expensive.

4. **Natural adaptivity**: `C_b` varies by backup provider. Cheap backup (Parasail) has
   low `C_b/V`, hedges more aggressively. Expensive backup has high `C_b/V`, hedges
   more conservatively. This is economically rational without needing separate logic.

5. **Paper-ready**: The formula has a clean economic interpretation suitable for an
   NSDI/OSDI paper. It connects to the broader LP cost-optimization framework
   (Phase 3) through the shared cost model.

**Relationship to existing `smart_survival`**:
The existing `smart_survival` formula `S_p(L)/S_p(t) > S_b(L-t-delta)` can be seen as
a special case where `F_b(remaining)` is moved to the left side but compared against
`1 - S_b(L-t-delta)` (the inequality direction differs). The economic model generalizes
this by (a) using product form instead of ratio, and (b) introducing the cost threshold
`C_b/V` instead of implicitly assuming V -> infinity.

### Blockers: Phase 4 Script Broken

Two issues prevent `run_phase4_simulation.py` from running:

1. **Stale API**: Script passes `prior_strength` parameter (lines 64, 220, 294)
   to `ProviderProfile` and `OnlineLatencyRouter`, but both classes have been
   refactored and no longer accept this parameter.

2. **Missing data path**: Default `--data` path points to
   `ICML2026_HybridInference/data/latency_llama70b_24h.csv` which does not exist.
   Actual data is at `experiment/data/data/latency_llama70b_24h.csv`.
   Default `--pricing` path `experiment/data/openrouter_llama33_70b.json` is correct.

---

## Plan

### Step 0: Fix Phase 4 Script to Runnable State

**File**: `experiment/scripts/run_phase4_simulation.py`

#### 0a. Remove stale `prior_strength` references

- Line 64: Remove `prior_strength` from `SimulationConfig`
- Line 220: Remove `prior_strength` from `ProviderProfile()` constructor call.
  Use `window_sec` from existing `OnlineLatencyRouter` default (15 * 60 = 900 sec),
  no new config field needed.
- Line 294: Remove `prior_strength` from `OnlineLatencyRouter()` constructor call.
  Use default `window_sec` parameter from `OnlineLatencyRouter.__init__`.

#### 0b. Fix default data path

- Line 578 (`--data` default): Change from
  `ICML2026_HybridInference/data/latency_llama70b_24h.csv` to
  `experiment/data/data/latency_llama70b_24h.csv`
- Line 654 (alt_path fallback): Update to match new default

#### 0c. Verification

Run smoke test with actual data:
```bash
cd /home/murphy/test/hybridInference
python experiment/scripts/run_phase4_simulation.py \
    --data experiment/data/data/latency_llama70b_24h.csv \
    --strategy never \
    --max-eval-points 10 \
    --output /tmp/phase4_smoke_test
```

Must complete without import errors or missing-parameter exceptions.

**Deliverables**:
- Fixed `run_phase4_simulation.py` that runs end-to-end on existing data
- Smoke test output in `/tmp/phase4_smoke_test/`

### Step 1: Add `SMART_ECONOMIC` Strategy

**File**: `experiment/strategies/smart_hedging.py`

Add a new hedging strategy based on cost-benefit economic model.

#### 1a. Strategy enum and parameters

```python
class HedgingStrategy(Enum):
    ...
    SMART_ECONOMIC = "smart_economic"  # NEW: cost-benefit model
```

New fields in `HedgingParams`:
```python
@dataclass
class HedgingParams:
    ...
    cost_ratio: float = 0.1  # C_b / V: backup cost relative to violation penalty
```

#### 1b. Core hedge decision function

```python
def smart_hedge_economic(
    primary: str,
    backup: str,
    elapsed_sec: float,
    slo_sec: float,
    profiles: dict[str, ProviderProfile],
    now: float,
    cost_ratio: float = 0.1,
    dispatch_overhead_sec: float = 0.05,
) -> bool:
    """Hedge when expected benefit of avoiding violation exceeds backup cost.

    Decision rule:
        P_viol(t) * F_b(remaining) > C_b / V

    Left side: probability that hedging actually prevents a violation.
      - P_viol(t) = S_p(L)/S_p(t) = P(primary violates | survived to t)
      - F_b(remaining) = P(backup finishes within remaining budget)
    Right side: cost ratio (backup cost / violation penalty).

    When left > right, the expected savings from hedging exceed its cost.
    """
    remaining = slo_sec - elapsed_sec - dispatch_overhead_sec

    # If no time remains for backup, hedge is pointless regardless of primary risk.
    if remaining <= 0:
        return False

    S_L = get_survival_for_hedging(profiles[primary], slo_sec, now)
    S_t = get_survival_for_hedging(profiles[primary], elapsed_sec, now)

    if S_t < 1e-6:
        # Primary has exceeded all historical samples -> P_viol ~ 1.0
        P_viol = 1.0
    else:
        P_viol = S_L / S_t

    F_backup = get_cdf_for_hedging(profiles[backup], remaining, now)

    # Cost-benefit decision: hedge if expected benefit > cost
    return P_viol * F_backup > cost_ratio
```

**Key properties**:
- **Single parameter** (`cost_ratio`): replaces two parameters (`theta`, `min_backup_cdf`)
- **Edge cases handled uniformly**: `remaining <= 0` -> False (no time for backup).
  `S_t < 1e-6` -> `P_viol = 1.0`, still goes through cost-benefit check with `F_backup`.
  No special-case branches that bypass the main logic.
- **Backup-adaptive**: expensive backup (high `C_b`) naturally hedges less;
  cheap backup naturally hedges more. No separate viability guard needed.

#### 1c. Fix "no trigger" return value in grid search

Existing `find_optimal_hedge_time_survival()` and `find_optimal_hedge_time_residual()`
return `slo_sec` when the condition never triggers (lines 319, 359). But in
`simulate_request()`, only `float("inf")` means "never hedge"; `slo_sec` falls into
the `else` branch and triggers a hedge at the SLO boundary -- too late to help, but
still costs money.

The new `find_optimal_hedge_time_economic()` MUST return `float("inf")` when no
trigger point is found. Add a unit test that verifies: when `cost_ratio` is very high
(e.g., 0.99) and primary is fast, the returned hedge time is `float("inf")` and
`simulate_request` does NOT hedge.

Also fix the existing helpers (`find_optimal_hedge_time_survival`,
`find_optimal_hedge_time_residual`) to return `float("inf")` instead of `slo_sec`,
since this is a latent bug that affects existing strategies too.

#### 1d. Integration

- Add `cost_ratio` field to `HedgingParams`
- `dispatch_overhead_sec` already exists in `HedgingParams`, pass it through
- Add `SMART_ECONOMIC` branch in `SmartHedger.compute_hedge_time()`
- Add `find_optimal_hedge_time_economic()` for grid search (returns `float("inf")`
  when no trigger point found)

**Deliverables**:
- New strategy in `smart_hedging.py`
- Unit tests covering:
  - `smart_hedge_economic()`: basic trigger behavior
  - Cost-benefit check blocks hedge when backup is too expensive (`cost_ratio` high)
  - Cost-benefit check blocks hedge when `remaining <= 0`
  - Cost-benefit check blocks hedge when backup is too slow (`F_backup` low)
  - Monotonicity: higher `cost_ratio` -> lower hedge rate
  - `find_optimal_hedge_time_economic()`: returns `float("inf")` when no trigger
    (NOT `slo_sec`), and `simulate_request` correctly does not hedge
  - Fixed `find_optimal_hedge_time_survival/residual`: also return `float("inf")`
  - `SmartHedger.compute_hedge_time()` with `SMART_ECONOMIC`
  - Phase 4 CLI accepts `smart_economic` strategy name
  - Phase 5 threaded hedging path works with new strategy

**Note on hedge rate stability**: The economic model's hedge decision depends on both
primary risk (`P_viol`) and backup viability (`F_backup`). Unlike the two-stage
heuristic where the "risk gate" was backup-independent, the economic model's trigger
inherently couples primary and backup. However, the coupling is through a principled
cost-benefit product, not through arbitrary threshold comparisons. The `cost_ratio`
parameter has clear economic meaning (how much are we willing to pay per avoided
violation), making cross-configuration comparisons more interpretable.

### Step 2: Re-run Phase 4 Simulation

**File**: `experiment/scripts/run_phase4_simulation.py`

Update ablation study to include `SMART_ECONOMIC` with `cost_ratio` sweep:
- `cost_ratio` in {0.01, 0.05, 0.1, 0.2, 0.5}
- All backup methods (fastest, lp_other, cheapest_viable)
- SLO in {1.0, 2.0, 3.0, 5.0}

**Hypotheses to test** (not pre-determined conclusions):
- H1: `cost_ratio` monotonically controls hedge rate
  (higher cost_ratio -> lower hedge rate)
- H2: Hedge rate variance across backup methods is smaller than `smart_survival`
  (28%-87%) and `smart_residual` (1%-100%). Note: some variance is expected because
  `F_backup` differs by provider; the claim is that the economic model produces more
  stable rates than the existing strategies.
- H3: There exists a `cost_ratio` range where violation rate is near-zero with hedge
  rate comparable to fixed_timeout (~10-20%)
- H4: For the same `cost_ratio`, cheap backup providers (low `C_b`) trigger more
  hedges than expensive ones, consistent with economic rationality.

If H3 is not supported, we report the actual tradeoff curve honestly and may fall
back to recommending `PERCENTILE_BASED` as the production strategy.

**Deliverables**:
- Ablation CSV results for all configurations
- Comparison plots: hedge rate vs cost_ratio, violation rate vs cost_ratio,
  Pareto frontier (cost vs violation rate) across all strategies
- Written analysis of whether hypotheses are supported

### Step 3: Update Phase 5 Production Code

**Depends on**: Step 1 (strategy code) AND Step 2 (to determine best cost_ratio)

**File**: `experiment/scripts/phase5_online_evaluation.py`

Replace `PERCENTILE_BASED(P90)` with `SMART_ECONOMIC` (using best cost_ratio from Step 2):

```python
# Before (line 472-478):
hedging_params = HedgingParams(
    strategy=HedgingStrategy.PERCENTILE_BASED,
    alpha_percentile=90.0,
    ...
)

# After:
hedging_params = HedgingParams(
    strategy=HedgingStrategy.SMART_ECONOMIC,
    cost_ratio=<best_from_step2>,
    ...
)
```

**Setting `cost_ratio` from actual costs**: In production, `C_b` is the per-request
cost of the backup provider (from pricing data). `V` is the violation penalty which
can be estimated from the SLO contract or set as a business parameter. If the operator
does not specify `V`, a default can be derived from Step 2 ablation results
(the `cost_ratio` that achieves the best Pareto tradeoff).

**Cost accounting clarification**:
- **Real cost** (always reported): Both primary and backup are billed when hedge
  triggers. This is `c_primary + c_backup` for hedged requests. Phase 5 already
  tracks this correctly in `_send_hedged_request()`.
- **Cancel-aware cost** (estimated, reported separately): Assumes the losing request
  is cancelled and only `c_winner` is billed. This is a **counterfactual estimate**,
  not an actual billing measurement. Must be clearly labeled as "estimated" in
  output CSV and paper.
  Formula: `(1 - p_hedge) * c_primary + p_hedge * c_winner`

Both values are reported. The paper must clearly distinguish which is which.

**Deliverables**:
- Updated Phase 5 config
- Clarified cost reporting (real vs cancel-aware columns in output CSV)

### Step 4: Re-run Phase 5 Production Evaluation

Run 24h evaluation against OpenRouter with new strategy.

Compare against baselines:
- OpenRouter Auto
- Fastest Fixed (Groq)
- Cheapest Fixed (Parasail)
- LP-Mix (no hedging)
- Smart Hedge (SMART_ECONOMIC)

Metrics to collect:
- P50, P90, P99 TTFT
- SLO violation rate
- Hedge rate
- Real cost (both requests billed)
- Cancel-aware cost (estimated, clearly labeled)

### Step 5: Update Paper

**Section 4b (Smart Hedging for Tail Guarantees)**:
- Present hedging as "cost-benefit speculative hedging"
- Core formula: `P_viol(t) * F_b(remaining) > C_b / V`
- Interpretation: hedge when expected benefit of avoiding violation exceeds backup cost
- `P_viol(t) = S_p(L)/S_p(t)` is the conditional violation probability (primary-side)
- `F_b(remaining)` is the backup success probability (backup-side)
- `C_b/V` is the cost ratio (single tunable parameter with economic meaning)
- Connect to LP framework: `C_b` comes from the same pricing model used in Phase 3
- Clearly state this is a principled heuristic for parallel hedge, not a claim of
  global optimality

**Section 5 (Experiments)**:
- Report results from Step 4 (Phase 5 with SMART_ECONOMIC)
- Remove inconsistent references to "P90 threshold" and "residual-based"
- Show cost_ratio sweep results (hedge rate vs violation rate tradeoff)
- Clearly distinguish real cost vs cancel-aware estimated cost

**Appendix**:
- Keep `smart_survival` as "Alternative: Survival Comparison Rule" with honest
  discussion of its aggressive hedge rate and backup-dependency
- Keep `smart_residual` as "Alternative: Residual Life Decision Rule" with note on
  serial-vs-parallel semantics mismatch
- Show that `smart_survival` is a special case of the economic model where `V -> inf`
  (i.e., violations are infinitely costly, so always hedge when primary is at risk)
- Note: these are design alternatives with different tradeoffs

---

## Risk Assessment

- **Low risk**: Step 0 (fix script compatibility) is a mechanical fix
- **Low risk**: Step 1 (new strategy) is purely additive, does not modify existing code
- **Medium risk**: Step 2 (simulation) depends on Step 0 fix being correct
- **Medium risk**: Step 4 (production re-run) costs real money (~$5-10 for 24h eval)
- **Low risk**: Step 5 (paper update) is text changes only

## Dependencies

```
Step 0 (fix Phase 4 script)
  |
  v
Step 1 (add SMART_ECONOMIC + tests)
  |
  v
Step 2 (Phase 4 simulation, pick best cost_ratio)
  |
  v
Step 3 (update Phase 5 config with best cost_ratio)
  |
  v
Step 4 (Phase 5 production eval)
  |
  v
Step 5 (update paper)
```

All steps are sequential. Step 3 explicitly depends on Step 2 results to set cost_ratio.

## Open Decisions

1. **Default `cost_ratio` value**: Starting with {0.01, 0.05, 0.1, 0.2, 0.5} sweep.
   The best value depends on the actual `C_b` distribution across providers and the
   operator's tolerance for violations.

2. **How to set `V` in production**: Options:
   - Operator-specified: part of SLO contract (e.g., "each violation costs $X")
   - Derived from ablation: use the `cost_ratio` from the Pareto-optimal point in Step 2
   - Rule-of-thumb: `V = 10 * C_b` (i.e., a violation is worth 10x the backup cost),
     giving `cost_ratio = 0.1`

3. **What if SMART_ECONOMIC doesn't beat PERCENTILE_BASED(P90)?**
   If simulation shows P90 is comparable or better, we should honestly report that
   and use P90 in production. The paper would then present both strategies and
   recommend P90 for simplicity with SMART_ECONOMIC as an alternative when cost
   sensitivity matters.

4. **Paper narrative**: The economic model has stronger theoretical grounding than P90
   (connects to cost optimization, has interpretable threshold). Even if P90 performs
   similarly in our evaluation, the economic model is more defensible for NSDI review
   because it explains *why* the trigger is set where it is, not just *what* percentile.
