# Synthetic Simulation — Design & Implementation

Post-refactor path note: this document records the design history. The
canonical implementation now lives under `experiments/synthetic_latency/` and
`rwsim/`

## Motivation

The LP Mix router (`OnlineLatencyRouter`) was found to over-diversify traffic when one provider is both cheapest and fastest, causing more SLO violations than a trivial "always-pick-cheapest" baseline. The V2 P50 router and Smart Hedging approach are proposed as fixes. Before relying on noisy real-world data, we want to verify the algorithms in a controlled environment where:

- The true latency distribution is known exactly.
- "Correct" routing behavior is analytically obvious (e.g. 100% to the dominant provider).
- We can isolate specific failure modes: dominant provider, cost/latency trade-offs, heavy tails, distribution shift, and similar providers.

This document describes the design of the synthetic simulation built to satisfy that need.

---

## File Structure

```
experiments/synthetic_latency/
    configs/        — S1–S5 scenario configurations
    experiment.py   — Config loading and runner helpers
    materialize.py  — Config -> runnable world objects
    plots.py        — Four plot types per scenario

rwsim/world/        — providers, distributions, workload, metrics
rwsim/strategies/   — latency strategy loops
rwsim/policies/     — pipeline-stage policy components
run_synthetic.py    — Top-level orchestrator
results/synthetic/  — Output directory
    s{N}_*/
        summary.json
        slo_violation.png
        cost_comparison.png
        provider_selection.png
        latency_cdf.png
```

---

## Provider Model

### Distribution

TTFT (time to first token) is sampled from a **Log-Normal** distribution, which matches real provider behaviour: right-skewed, heavy right tail, always positive.

```
X ~ LogNormal(mu, sigma)
P50 = exp(mu)
P99 = exp(mu + 2.326 * sigma)
```

Given target P50 and P99:
```
mu    = ln(P50)
sigma = (ln(P99) - ln(P50)) / 2.326
```

Generation speed (tokens per second) is also Log-Normal. E2E latency = TTFT + output_tokens / TPS × 1000. However, **SLO is evaluated on TTFT only**, consistent with the production evaluation logs.

### `SyntheticProvider`

```python
@dataclass
class SyntheticProvider:
    name: str
    cost_per_token: float      # USD per token
    ttft_dist: LogNormal       # TTFT distribution in ms
    tps_dist: LogNormal        # Tokens-per-second distribution

    def sample_ttft(rng, current_time=0.0) -> float
    def sample_request(output_tokens, rng, current_time=0.0) -> (ttft_ms, e2e_ms)
    def true_p50_ms(current_time=0.0) -> float   # analytical, no noise
```

### `ShiftingProvider`

Subclass of `SyntheticProvider`. Has a second `ttft_dist_after` that activates at `shift_time` (simulated seconds). Used in S4 to model a provider degradation event.

```python
@dataclass
class ShiftingProvider(SyntheticProvider):
    shift_time: float
    ttft_dist_after: LogNormal
```

---

## Workload Generator

**Arrival process**: inter-arrival times drawn from Exp(rate) for Poisson, or a mixture (60 % at 3× rate, 40 % at 0.25× rate) for bursty.

**Output tokens**: LogNormal(mu=4.0, sigma=1.0) → P50 ≈ 55 tokens, E[tokens] ≈ 90.

**Input tokens**: fixed at 100 per request.

**Seed**: workload is generated once with seed=0 and shared across all strategies and all random seeds, so provider and cost differences are the only source of variation.

---

## Scenarios

All five scenarios use 2 000 requests over 3 600 s (1 hour) with Poisson arrivals. SLO thresholds for reporting: 1 000, 2 000, 3 000, 5 000 ms. Router SLO (used internally by LP and V2): **2 000 ms**.

| ID | Name | Providers | Expected winner |
|----|------|-----------|----------------|
| S1 | Dominant | A: slow+expensive; B: fast+cheap | 100 % B |
| S2 | Trade-off | A: cheap+slow; B: expensive+fast | Depends on SLO |
| S3 | Tail-heavy | A: great P50, catastrophic P99; B: mediocre but stable | B (SLO tight); hedging helps on A |
| S4 | Shift | A: fast until t=30 min, then slow; B: constant | A first, then B |
| S5 | Similar | A/B/C all near-equal P50, C cheapest | C (cheapest in near-best band) |

Full provider parameters are in `SYNTHETIC_RESULTS.md`.

---

## Strategy Implementations

### Stateless baselines

| Strategy | Logic |
|----------|-------|
| `cheapest_fixed` | Pre-compute `argmin(cost_per_token)` once; route all requests there. |
| `fastest_fixed` | Pre-compute empirical P50 across the full simulation (10 000 samples per provider, averaging pre- and post-shift for ShiftingProvider); route all requests to the globally fastest. |
| `round_robin` | Index modulo number of providers; no data dependency. |
| `oracle_per_window` | For each 15-minute window midpoint, compute the **analytical** P50 (`exp(mu)`) of each provider at that time; route to the one with lowest P50. This is the hindsight upper bound. |

### Adaptive strategies — shared infrastructure

Both `lp_mix` and `v2_p50_hedge` use production router classes unchanged. Before the main loop, two mechanisms seed the router profiles:

**Warm-up**: 50 samples per provider, drawn from each provider's true distribution, spread uniformly over the 14 minutes preceding the first request. This simulates the initial probing data that production routers have before any window starts. Because the samples span 14 min, they remain within the 15-min rolling window throughout the first hour.

**Periodic probing**: After each production routing decision, each non-primary provider receives a probe sample with probability 5 % (rate ≈ 1 probe per 20 requests per provider). This models the background probing that production systems send to maintain fresh latency profiles even for providers not currently receiving traffic. Without probing, a non-primary provider's profile empties after 15 min, making the router effectively blind to it. This is critical for S4: once Provider A degrades, the router needs current data on B to detect and switch.

### `v2_only`

Uses `V2Router` with the same warm-up, probing, and `lp_update_interval = 60 s` as `v2_p50_hedge`, but **no hedger is applied**. Provider selection follows the same P50-rank + Pareto + near-best-band logic; the chosen primary receives 100 % of traffic with no backup dispatch. Comparing `v2_only` versus `v2_p50_hedge` isolates the contribution of SmartHedger.

### `lp_mix`

Uses `OnlineLatencyRouter` with `lp_update_interval = 60 s`. The router solves:

```
minimize   Σ π_j · c_j
subject to Σ π_j · F_j(SLO) ≥ 0.99
           Σ π_j = 1,  π_j ≥ 0
```

Routes via Smooth Weighted Round-Robin (SWRR) with weight-smoothing factor 0.3. Latency samples are fed back after each request.

### `lp_hedge`

Uses `OnlineLatencyRouter` (same LP routing logic as `lp_mix`) combined with SmartHedger (`SMART_ECONOMIC`, `cost_ratio = 0.1`, `dispatch_overhead = 50 ms`). After the LP SWRR sampler selects the primary provider, the hedger applies the same trigger rule as `v2_p50_hedge`. Only the primary response is fed back to the router profile. Comparing `lp_hedge` versus `lp_mix` isolates hedging; comparing `lp_hedge` versus `v2_p50_hedge` isolates the routing algorithm under equivalent hedging.

### `v2_p50_hedge`

Uses `V2Router` with `lp_update_interval = 60 s` (same as LP for a fair comparison). Selection rule per update:

1. Pre-filter: remove providers with error rate > 5 % or CDF(1 s) < 80 %.
2. Pareto-prune: remove providers dominated on (P50, cost).
3. Near-best band: threshold = best_P50 × 1.10 (10 %).
4. Primary = cheapest provider in band.

After selecting the primary, `SmartHedger` with `SMART_ECONOMIC` strategy decides the hedge trigger time:

```
hedge when  P_viol(t) × F_backup(remaining_budget) > cost_ratio
```

where `cost_ratio = 0.1` and `dispatch_overhead = 50 ms`. The backup provider is the one with the lowest P50 in the current profile (fastest non-primary). If the hedge triggers, both requests are billed. Only the primary sample is fed back to the router profile.

---

## Strategy Comparison Design

The eight strategies form a 2×2 factorial plus three stateless references and one oracle:

|                | No hedging | With hedging  |
|----------------|------------|---------------|
| LP routing     | `lp_mix`   | `lp_hedge`    |
| V2 routing     | `v2_only`  | `v2_p50_hedge`|

Stateless references: `cheapest_fixed`, `fastest_fixed`, `round_robin`. Analytical upper bound: `oracle_per_window`.

This design enables clean attribution:

- **`lp_mix` vs `lp_hedge`**: effect of adding hedging to LP routing.
- **`v2_only` vs `v2_p50_hedge`**: effect of adding hedging to V2 routing.
- **`lp_mix` vs `v2_only`**: effect of routing algorithm alone (no hedging).
- **`lp_hedge` vs `v2_p50_hedge`**: effect of routing algorithm when both use hedging.

Key finding from S3: `lp_hedge` is cheaper than `v2_p50_hedge` ($2.07e-4 vs $2.53e-4) because LP's diversification toward stable provider B reduces the hedge frequency from 33.2 % to 9.2 %. Hedging solves the same SLO problem in both cases, but the routing algorithm determines how often hedging is needed.

---

## Cost Accounting

The router's `costs` dict (used by LP and V2 for optimisation) is a fixed value per provider:

```
costs[p] = p.cost_per_token × 200   (representing a typical 200-token request)
```

The per-request cost reported in results uses the **actual** sampled token count:

```
cost_usd = cost_per_token × req.total_tokens
```

For `v2_p50_hedge`, if a hedge triggers, the backup request is also billed:

```
cost_usd = primary_cost_per_token × total_tokens
         + backup_cost_per_token × total_tokens
```

---

## Oracle Definition

The oracle selects the provider with the lowest **analytical** P50 at the midpoint of each 15-minute window. This is not achievable online (it requires knowing the true distribution), but it represents the best a P50-ranking strategy could do with perfect knowledge. It is intentionally defined by P50, not by SLO compliance rate, to match the V2 router's selection criterion and isolate the question of whether adaptation helps.

Implication for S3: the oracle routes to A (P50 = 100 ms) despite A having a catastrophic P99, resulting in high SLO violations. This shows that P50-based routing is not always SLO-optimal when tails are heavy.

---

## Reproducibility

- Workload seed: 0 (fixed, shared).
- Strategy seeds: [42, 43, 44] — three trials; summary metrics are averaged across trials.
- Platform: NumPy `default_rng` (PCG64), SciPy `linprog` (HiGHS).
- No real provider data is used; all latency values are drawn from the parameterised distributions.

---

## Key Design Assumptions

| Assumption | Rationale |
|------------|-----------|
| SLO evaluated on TTFT, not E2E | Consistent with production evaluation logs. |
| Router costs dict uses 200-token estimate | Only relative ordering matters for LP; absolute value does not affect routing. |
| Probing rate 5 % per non-primary | Matches approximate production probe frequency; prevents profile starvation without dominating traffic. |
| 50 warm-up samples over 14 min | Ensures all providers have enough data for the first LP update; samples expire naturally before the 15-min window rolls over. |
| Oracle uses analytical P50 | Removes sampling noise from the upper-bound baseline; any noise would only make the oracle look worse. |
| Both primary and backup charged if hedge fires | Matches production billing semantics. |
| S5 P50 spread ≤ 10 % | Required for all three providers to fall within V2's near-best band. The README's 250 ms figure (25 % spread) would place C outside the 10 % band; adjusted to 215 ms (7.5 % spread). |
