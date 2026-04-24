# Experiment Talk Practice Plan

This document prepares a short discussion with Juncheng about the experiment
we plan to run and the results we already have. It is written as speaker notes:
use it to practice the narrative, not as a final paper section.

## Goal

Practice explaining three things clearly:

1. What problem RouteWise is solving.
2. What experiment we plan to run next.
3. What current results already support the design.

The main message:

> RouteWise routes one LLM request across providers with different pricing
> contracts and latency profiles. The cost router prices scarce quota and
> concurrency capacity with shadow prices; the latency router and hedger then
> trade a small amount of extra cost for much lower tail latency.

## Talk Shape

Use this structure for a 10-12 minute discussion.

| Time | Section | What to Say |
| --- | --- | --- |
| 0:00-1:00 | Motivation | Same model, many providers, different contracts: API, quota subscription, concurrency slots. Naive routing wastes scarce capacity or violates latency SLOs. |
| 1:00-3:00 | System | RouteWise is a pipeline: value estimator -> cost router -> latency router -> hedger. This maps directly to `docs/ALGORITHMS.md`. |
| 3:00-5:00 | Planned experiment | End-to-end replay: compare two-layer/greedy, joint cost routing, joint+hedging, and latency-only baselines on tiered and calibrated scenarios. |
| 5:00-9:00 | Current results | Show cost-only paper results, OpenRouter latency results, and local simulator golden results for S3/S6/unified_pool. |
| 9:00-11:00 | What is still pending | Full engine+composer migration and known divergences are tracked; current results are reproducible through CLI and golden baselines. |
| 11:00-12:00 | Ask | Ask whether the planned experiment answers the right paper claim and what comparison Juncheng wants emphasized. |

## Planned Experiment

### Research Questions

1. Does joint routing across `S_Q`, `S_C`, and `S_A` avoid the traps of a
   category-first two-layer policy?
2. How much latency improvement do we get from hedging on top of joint routing?
3. When does RouteWise intentionally pay more money, and is the latency gain
   worth that cost?
4. Are the paper claims reproducible through the cleaned-up experiment entrypoints?

### Policies to Compare

| Policy | Meaning | Code Entry |
| --- | --- | --- |
| `two_layer` | Historical tier-priority baseline: quota -> concurrency -> API | `rwsim/strategies/tiered_impl.py` |
| `joint_nohedge` | Joint effective-cost selection with P95 SLO filtering, no hedging | `rwsim/strategies/tiered_impl.py` |
| `joint_hedge` | Joint routing plus smart hedging | `rwsim/strategies/tiered_impl.py` |
| `joint_p50band_nohedge` | P50-band latency selector ablation | `rwsim/strategies/tiered_impl.py` |
| `joint_p50band_hedge` | P50-band selector plus hedging | `rwsim/strategies/tiered_impl.py` |

The pipeline mapping for these names is documented in
`docs/ALGORITHMS.md` and registered in `rwsim/policies/composer.py`.

### Workloads and Commands

Run the local reproducible suites:

```bash
source .venv/bin/activate
python -m routewise_cli.main list --suites
python -m routewise_cli.main suite joint
python -m routewise_cli.main suite joint_mm25_baselines
python -m routewise_cli.main suite stress
python tests/golden_capture.py --mode compare
```

Useful focused smoke runs:

```bash
python -m routewise_cli.main run tiered_capacity \
  --scenario s6_slow_q_trap --strategy joint_hedge --seed 42

python -m routewise_cli.main run synthetic_latency \
  --scenario s3_tail --strategy v2_p50_hedge --seed 42
```

Expected output locations:

| Suite | Output |
| --- | --- |
| `joint` | `outputs/joint/<scenario>/summary.json` |
| `joint_mm25_baselines` | `outputs/alpha_joint_mm25/<scenario>/baselines.json` |
| `stress` | `outputs/stress/<scenario>/summary.json` |
| golden compare | `tests/golden/*` |

### Metrics

Report four metrics consistently:

| Metric | Why It Matters |
| --- | --- |
| Mean cost per request | Does the policy overspend? |
| P50 latency | Body latency and user-perceived normal case |
| P99 latency | Tail latency and production reliability |
| SLO violation rate | Direct operational objective |
| Hedge rate | How often we pay backup cost |

## Current Results to Discuss

### 1. Paper Cost-Routing Results

From the paper evaluation:

| Claim | Result |
| --- | --- |
| Quota-only greedy leaves savings on the table | Greedy saves 17-81%, while optimal saves 35-98% depending on dataset |
| Online PD routing is near-optimal on large traces | On BurstGPT, PD-EMA reaches 1.18x relative cost vs greedy 1.30x |
| Joint use of quota and concurrency matters | On FreeInference, PD-EMA reaches 1.74x relative cost vs greedy 5.3x |
| Simple predictors are enough | EMA point estimate is best in the 2x2 ablation at 1.18x relative cost |

How to say it:

> The offline analysis tells us what optimal routing should value: scarce
> subscription capacity should go to high-value requests. The online algorithm
> approximates this with a shadow price. The important result is not that the
> predictor is complex; it is that even a simple EMA is good enough once the
> resource price is correct.

### 2. Paper Latency Results

From the 24-hour OpenRouter experiment:

| Policy | P50 | P99 | SLO Viol. | Avg Cost |
| --- | ---: | ---: | ---: | ---: |
| OpenRouter Auto | 747 ms | 5988 ms | 14.1% | $5.1e-6 |
| OpenRouter `sort=latency` | 270 ms | 2090 ms | 1.5% | $11.3e-6 |
| Ours: Latency-Aware | 308 ms | 1975 ms | 2.3% | $3.5e-6 |
| Ours: Smart Hedge | 280 ms | 1335 ms | 0.3% | $4.2e-6 |

How to say it:

> OpenRouter Auto has poor tail behavior because it spreads traffic across
> many providers. The latency-aware router finds a small cheap mix with good
> latency, and smart hedging handles the residual tail. Compared with Auto,
> smart hedging cuts P99 by 4.5x and SLO violations by 47x while reducing cost.

### 3. Local Synthetic-Latency Golden Results

These are reproducible from `tests/golden/latency/synthetic.json`.

Key example: `s3_tail`, where the fastest/cheapest provider has a bad tail.

| Strategy | Mean Cost | P99 | SLO Viol. >1s | Hedge Rate |
| --- | ---: | ---: | ---: | ---: |
| `v2_only` | 1.90e-4 | 5316 ms | 8.50% | 0% |
| `v2_p50_hedge` | 2.53e-4 | 916 ms | 0.57% | 33.25% |
| `lp_mix` | 1.90e-4 | 2319 ms | 3.17% | 0% |
| `lp_hedge` | 2.08e-4 | 1022 ms | 1.11% | 9.32% |

Interpretation:

> Hedging fixes the tail, but routing determines how often hedging is needed.
> LP routing sends more traffic to stable providers, so it needs fewer backups
> than the P50-only route.

### 4. Local Tiered Golden Results

These are reproducible from `tests/golden/tiered/scenarios.json`.

Key example: `s6_slow_q_trap`, where the quota tier is cheap but too slow.

| Strategy | Mean Cost | P50 | P99 | SLO Viol. >1s |
| --- | ---: | ---: | ---: | ---: |
| `two_layer` | 0 | 2037 ms | 4735 ms | 96.48% |
| `joint_nohedge` | 5.38e-4 | 103 ms | 369 ms | 0% |
| `joint_hedge` | 5.38e-4 | 103 ms | 369 ms | 0% |

Interpretation:

> This is the cleanest demonstration that category-first routing is wrong when
> the cheap tier is slow. Joint routing pays API cost, but it avoids almost all
> SLO failures.

Key example: `unified_pool`, where many providers across all categories are
available.

| Strategy | Mean Cost | P99 | SLO Viol. >1s | SLO Viol. >2s |
| --- | ---: | ---: | ---: | ---: |
| `two_layer` | 6.45e-7 | 3537 ms | 51.43% | 11.94% |
| `joint_nohedge` | 5.69e-5 | 1854 ms | 14.07% | 0.67% |
| `joint_hedge` | 8.46e-5 | 1404 ms | 14.55% | 0.03% |

Interpretation:

> In the unified pool, joint routing spends more than two-layer because it is
> buying reliability. The right claim is not "joint is always cheaper"; the
> claim is "joint exposes the cost-latency tradeoff and avoids hidden SLO
> failure."

### 5. Calibrated MM25 Golden Result

Key example: `s8m_multi_sq_hierarchy`.

| Strategy | Mean Cost | P50 | P99 | SLO Viol. >1s |
| --- | ---: | ---: | ---: | ---: |
| `two_layer` | 3.08e-5 | 842 ms | 5398 ms | 31.69% |
| `joint_nohedge` | 5.22e-5 | 659 ms | 1408 ms | 20.91% |
| `joint_p50band_nohedge` | 1.03e-4 | 435 ms | 1446 ms | 6.72% |

Interpretation:

> The calibrated scenario shows the same pattern with more realistic provider
> names and latencies. More aggressive latency filtering costs more, but it
> can sharply reduce SLO violations.

## What We Can Safely Claim

Safe claims:

- The experiment code is now reproducible through `routewise_cli`.
- The architecture maps to the paper pipeline: value estimator, cost router,
  latency router, hedger.
- The cost router's shadow-price logic is tested for endpoints, monotonicity,
  scale invariance, and infeasibility.
- The smart economic hedger has analytical lognormal reference tests.
- The current golden results reproduce the intended qualitative behavior:
  category-first routing can be cheap but unsafe; joint routing makes the
  cost-latency tradeoff explicit.

Do not overclaim yet:

- Do not say the new `engine + composer` path fully replaces
  `rwsim/strategies/{latency,tiered}_impl.py`. It does not yet.
- Do not say all known algorithm divergences are fixed. They are documented,
  but some remain.
- Do not present stress `st2_s_q_degradation` as a final algorithm claim until
  the `provider_p95_at()` drift issue is fixed.

## Known Risks and How to Explain Them

| Risk | Status | How to Explain |
| --- | --- | --- |
| `provider_p95_at()` ignores `TieredProvider` drift | Known divergence | Stress degradation results are useful as regression coverage but should be fixed before making strong claims. |
| Optional `response_tokens` can crash estimators | Known divergence | Current golden workloads are labeled; next correctness pass should make online estimators robust to unlabeled requests. |
| `lat_term` historically appears in effective cost | Known divergence | Canonical design moves latency into the latency router; fix should be a behavior-change commit. |
| Old strategy loops remain | Transitional architecture | They are still canonical for reproducibility, but the next migration should move one strategy, likely `joint_nohedge`, through `engine + composer`. |

## Questions Juncheng May Ask

### Why is two-layer not enough?

Answer:

> Two-layer commits to a category before considering latency. In S6, the quota
> provider is free but slow, so two-layer gets near-zero cost and terrible SLO
> violations. Joint routing compares all providers on a common effective-cost
> scale, then filters by latency, so it can spend API cost when the cheap tier
> is not viable.

### Are we just paying more to get better latency?

Answer:

> Sometimes yes, and that is the point. The old baseline can look cheap because
> it ignores SLO failure. RouteWise exposes the tradeoff explicitly: if a cheap
> provider is unsafe, the router either pays for a safer provider or the user
> accepts higher violation risk. The experiment should quantify this tradeoff,
> not hide it.

### Why does hedging not always help?

Answer:

> Hedging only helps if there is remaining SLO budget and a backup provider can
> realistically finish in time. The economic rule triggers only when expected
> violation reduction exceeds backup cost. In some scenarios, routing already
> avoids the tail, so hedge rate is near zero.

### What is the difference between paper results and local golden results?

Answer:

> Paper results include real traces and a 24-hour OpenRouter production
> experiment. Local golden results are deterministic simulator baselines used
> to protect refactors and validate algorithm behavior. They should agree
> qualitatively with the paper claims, but they are not a replacement for the
> production latency experiment.

### What should we run next?

Answer:

> First rerun the local suites after the refactor to show reproducibility.
> Then run the end-to-end paper experiment that combines tiered cost routing,
> latency-aware provider selection, and hedging on a production-realistic trace.
> The missing piece is not entrypoint cleanup anymore; it is making the full
> engine/composer path own one strategy end to end.

## Practice Script

Use this as a 2-minute version.

> The experiment I want to discuss is an end-to-end replay of RouteWise across
> three provider categories: quota subscriptions, concurrency subscriptions,
> and pay-per-token APIs. The main hypothesis is that category-first routing is
> not enough: it can minimize cost by filling free quota, but it can also pick
> providers that are clearly unsafe for latency. RouteWise instead computes an
> effective cost for every provider, using shadow prices for scarce capacity,
> and then lets the latency router and hedger choose a provider that respects
> the SLO.
>
> The local simulator already shows the qualitative behavior. In `s6_slow_q_trap`,
> two-layer pays zero API cost but has 96% violation above 1 second and a P99 of
> 4.7 seconds. `joint_nohedge` pays about 5.4e-4 per request, but its P99 drops
> to 369 ms and violations above 1 second go to zero. In `unified_pool`, two-layer
> has a P99 of 3.5 seconds and 51% violation above 1 second, while joint routing
> cuts P99 to 1.85 seconds, and joint+hedging cuts it further to 1.4 seconds.
>
> This matches the paper story: RouteWise is not just trying to use the cheapest
> provider; it is exposing the cost-latency tradeoff. The paper's production
> OpenRouter result shows the same idea in a live setting: smart hedging reduces
> P99 from 6.0 seconds to 1.3 seconds and SLO violations from 14.1% to 0.3%,
> while remaining cheaper than OpenRouter's latency mode.
>
> The next experiment I plan to run is the full end-to-end suite through the new
> CLI, then compare the generated summaries against the paper claims and golden
> baselines. The main caveat is that the engine/composer migration is not fully
> complete yet; the current strategy loops are still the canonical reproducible
> path. I want feedback on whether this experiment directly supports the paper's
> central claim or whether we should add another baseline.

## Concrete Meeting Ask

End the discussion with one specific request:

> Does this experiment isolate the right comparison: category-first routing
> versus joint cost-latency routing with hedging? If not, which baseline or
> workload should we add before presenting this as the main paper result?

## One-Page Checklist Before the Meeting

- Run `python tests/golden_capture.py --mode compare`.
- Run `python -m routewise_cli.main list --suites`.
- Pick two examples to show: `s6_slow_q_trap` and `unified_pool`.
- Bring the paper OpenRouter table for the production latency result.
- Be explicit that `engine + composer` is next-stage migration, not completed.
- Ask for feedback on baselines, not only on implementation cleanliness.
