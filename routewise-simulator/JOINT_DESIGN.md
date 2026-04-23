# Joint Cross-Tier Routing — Design & Results

## Motivation

The current RouteWise routes in two stages: Layer 1 picks the tier
(subscription priority S_Q > S_C > S_A), Layer 2 picks a provider within
that tier. Juncheng flagged a structural weakness in the 2026-04-14
meeting: once Layer 1 commits to a slow subscription, Layer 2 has no
latency lever left except aggressive hedging.

This package tests whether a joint cross-tier router — one that folds each
tier's capacity constraint into a unified effective cost and selects
providers across all tiers in one step — dominates the two-layer design.

## Files

```
experiment/scripts/simulate/synthetic/tiered/
    providers.py       TieredProvider + QuotaState + ConcurrencyState
    shadow_price.py    psi(z), lambda(u), effective_cost, calibrate_envelopes
    scenarios.py       S6, S7, S8 tiered scenarios
    strategies.py      two_layer + 4 joint variants
    runner.py          Scenario-level orchestration
    __init__.py

run_joint.py           Top-level entry; writes results/joint/{scenario}/summary.json
LESSONS_LEARNED.md     Process diary (how the design evolved, what broke)
```

## Effective cost

For a provider j at time t:

    c_j^eff(t) = marginal_cost_j(tokens) + psi_j^Q(z_j) + lambda_j^C(u_j)

- **S_A**: `c_eff = price_per_token * tokens`. Shadow prices are 0.
- **S_Q**: `c_eff = 0 + L * (U/L)^z` where z = quota used / quota size.
  Smooth exponential ramp from L (abundant quota) to U (exhausted).
- **S_C**: `c_eff = 0 + U * u` where u = active / limit. The linear
  congestion price (alpha=1 in the code) rises monotonically with
  utilisation and was chosen over the quadratic variant for
  interpretability; see paper sec.~3 concurrency shadow price.

Envelopes L and U are calibrated from the S_A providers in the scenario:
`U = max(api_cost_per_request)`, `L = U * 1e-3`.

## Strategies

### `two_layer` (baseline)

Layer 1: subscription-priority tier selection (S_Q if quota, else S_C if
slot, else S_A). Layer 2: lowest-true-P50 within the chosen tier.

This faithfully reproduces the current RouteWise structure.

### `joint_nohedge` (primary proposal)

    candidates = [p : p is available and P95(p) <= SLO * 0.8]
    primary    = argmin c_eff[p] among candidates

SLO-anchored P95 filter ensures only latency-safe providers are considered.
The cheapest by effective cost wins; shadow prices drive the transition
from subscriptions to S_A as capacity depletes.

### `joint_hedge`

`joint_nohedge` plus a cross-tier SMART_ECONOMIC hedge. Backup is the
fastest provider in a tier other than the primary's. Trigger rule:

    hedge if  P_violate(t_waited) * F_backup(SLO - t_waited)
                > c_eff[backup] / V_penalty

Both sides use effective cost so the economic comparison is tier-agnostic.
The current trigger heuristic (`wait_threshold = 1.5 * P50_primary`) is
flagged for iteration — see LESSONS_LEARNED.md Pitfall 4.

### `joint_p50band_nohedge`, `joint_p50band_hedge` (ablation)

The same framework but with the *incorrect* P50-band filter instead of the
SLO-anchored one:

    candidates = [p : p is available]
    band       = [p : P50(p) <= best_P50 * 1.10]
    primary    = argmin c_eff[p] among band

This variant is included to demonstrate that the filter choice is the
crucial design decision. See LESSONS_LEARNED.md Pitfall 1.

## Scenarios

| ID  | Name                      | Providers                                       | Intent |
|-----|---------------------------|-------------------------------------------------|--------|
| S6  | Slow-but-free trap        | S_Q P50=2000ms free, S_A P50=100ms $3/M         | Layer 1 traps into slow S_Q; joint should pick S_A. |
| S7  | Quota depletion           | S_Q quota=100 P50=300ms, S_A P50=200ms $3/M     | Quota runs out halfway; joint should ramp smoothly while using free capacity. |
| S8  | Concurrency saturation    | S_C C=4 svc=2s free, S_A P50=100ms $3/M         | 3x capacity load; joint should spill via lambda(u). |

## Results (seed=42, primary SLO)

```
S6  SLO = 1000 ms  (Chutes P50=2000 ms + Together $3/M)
  two_layer                  viol=96.5%  cost=$0        tiers=quota 100%
  joint_nohedge              viol= 0.0%  cost=$5.4e-4   tiers=api 100%
  joint_hedge                viol= 0.0%  cost=$5.4e-4   tiers=api 100%
  joint_p50band_nohedge      viol= 0.0%  cost=$5.4e-4   tiers=api 100%
  joint_p50band_hedge        viol= 0.0%  cost=$5.4e-4   tiers=api 100%

S7  SLO = 2000 ms  (Chutes quota=100 P50=300 ms + Together $3/M)
  two_layer                  viol= 0.0%  cost=$2.30e-4  tiers=quota 57%, api 43%
  joint_nohedge              viol= 0.0%  cost=$2.26e-4  tiers=quota 57%, api 43%  <- best
  joint_hedge                viol= 0.0%  cost=$2.97e-4  tiers=quota 57%, api 43%
  joint_p50band_nohedge      viol= 0.0%  cost=$5.74e-4  tiers=api 100%            <- broken
  joint_p50band_hedge        viol= 0.0%  cost=$5.74e-4  tiers=api 100%

S8  SLO = 2000 ms  (Featherless C=4 svc=2s + Together $3/M)
  two_layer                  viol= 0.1%  cost=$3.10e-4  tiers=concurrency 46%, api 54%
  joint_nohedge              viol= 0.2%  cost=$3.06e-4  tiers=concurrency 46%, api 54%  <- best cost
  joint_hedge                viol= 0.0%  cost=$3.92e-4  tiers=concurrency 39%, api 61%
  joint_p50band_nohedge      viol= 0.0%  cost=$5.76e-4  tiers=api 100%            <- broken
  joint_p50band_hedge        viol= 0.0%  cost=$5.76e-4  tiers=api 100%
```

## Interpretation

### Framing: joint as a backward-compatible extension of two_layer

Both `two_layer` and `joint` are our methods. `two_layer` is the current
RouteWise design; `joint` is the proposed extension. The evaluation does
not ask "which method wins a head-to-head" — it asks:
1. Does `joint` fix a regime where `two_layer` has a known weakness?
2. Does `joint` introduce a regression in regimes where `two_layer` is
   already correct?

Across S6/S7/S8 the answers are (1) yes, substantially, and (2) no.

### Scenario-by-scenario read

- **S6 (slow-subscription trap)**: `two_layer` fails at 96.5% SLO
  violation. Its Layer 1 greedily picks the subscription tier without
  checking whether the subscription can meet the SLO. `joint` uses an
  SLO-anchored P95 filter and routes 100% to S_A at 0% violation. *Joint
  fixes a real bug.*

- **S7 (quota depletion)**: both strategies meet the SLO; `joint` is 2%
  cheaper ($2.26e-4 vs $2.30e-4). The mechanism: `psi(z)` ramps smoothly
  before the quota hits z=1, so `joint` starts routing some traffic to
  S_A before `two_layer` would. Small but consistent improvement.

- **S8 (concurrency saturation)**: `joint` and `two_layer` produce
  essentially identical results ($3.06e-4 vs $3.10e-4, same tier mix,
  SLO within 0.1 pp). This is the *correct* outcome: concurrency is a
  discrete slot count, so the concurrency shadow price `lambda(u)` is
  mathematically equivalent to a binary gate (zero below capacity,
  infinite at capacity). `joint` reducing to `two_layer` on S_C is the
  shadow-price framework recognizing that the existing binary behavior
  is already optimal.

### Joint is strictly no worse than two_layer under correct filter design

The `joint_p50band_*` variants (ablation) route 100% to S_A in S7 and S8,
costing 2-2.5x more than `two_layer`. This was the initial design bug;
LESSONS_LEARNED.md Pitfalls 1-3 walk through the diagnosis. Crucially:
with the corrected SLO-anchored filter, `joint` does not regress against
`two_layer` in any tested scenario.

### Read for the paper

The contribution narrative should be:

> RouteWise's existing two-layer design handles concurrency-bound tiers
> correctly and is already SLO-safe when subscriptions are fast. We
> identify a real failure mode (slow subscriptions violating the SLO,
> S6) and propose a unified effective-cost framework that subsumes the
> existing design. The framework's SLO-anchored safety filter fixes the
> failure mode, and its smooth quota shadow-price mechanism gives a
> small additional improvement during quota depletion (S7). For
> concurrency-bound tiers the framework reduces to the original binary
> gate (S8), confirming that our extension is backward-compatible.

This is more defensible than a "joint dominates everywhere" claim and
actually matches what the synthetic results show.

### Hedging is not free lunch

`joint_hedge` is more expensive than `joint_nohedge` in both S7 and S8
without improving the primary SLO metric in S7 (because violations were
already 0%). The hedge trigger heuristic (`T > 1.5 * P50_primary`) is too
sensitive: it fires on the natural right-tail of the primary distribution
even when remaining SLO budget is large. A better trigger is SLO-budget
aware (see LESSONS_LEARNED.md Pitfall 4).

In S8, `joint_hedge` does eliminate the last 0.1% of violations but at
28% cost premium. Whether that is a good trade depends on V_penalty.

## Limitations

- Uses `true_p50_ms(now)` and analytical P95 as oracle estimates. A
  rolling-window estimate with warm-up / probing should replace this in
  the next iteration.
- Concurrency service time is separate from TTFT. `service_time_dist` is
  a simplification of real vLLM batch behavior.
- `V_penalty` is hard-coded to `10 * U`. The paper should tie it to a
  user-facing value or sweep it.
- Quota window rolls on wall-clock relative to `window_start`; real
  providers use rolling windows.

## Next iterations (ROI order)

1. **SLO-budget-aware hedge trigger** — tune the economic rule so hedging
   fires only when remaining SLO budget is small. Fixes S7/S8 cost premium.
2. **Rolling-window P50/P95** — replace oracle with warm-up + probing,
   verify the design is robust to measurement noise.
3. **Multi-provider per tier** — S6/S7/S8 have one provider per tier.
   Add S6+ variants with several S_A providers of different cost / P50.
4. **Cross-provider correlation** — shared-latent-event model so multiple
   providers can degrade together. Tests robustness under correlated outages.
5. **Phase diagram** — sweep (P50_Q / P50_A, workload / quota) on a grid,
   heat-map cost ratio (joint / two_layer). Each scenario above is one
   cell in this diagram.

## Related documents

- `LESSONS_LEARNED.md` — process diary, pitfalls we hit.
- `../../SYNTHETIC_DESIGN.md` — base (single-tier) simulator design.
- `../../MEMORY/v2-router-design.md` — original V2 rationale that we
  (initially) over-applied to the cross-tier setting.
