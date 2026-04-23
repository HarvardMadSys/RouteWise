# Latency Experiment Plan (Optional ICML Extension)

This document proposes a concrete plan to integrate end-to-end latency into the hybrid routing
framework, with an emphasis on (i) principled tail-SLO guarantees (e.g., P99) and (ii) practical
router mechanisms (provider selection + hedging) that can be deployed in OpenRouter-like settings.

The scope is intentionally modular: we can implement the measurement + offline evaluation first,
then add online hedging and (optionally) integrate with Stage 1/Stage 2 resource constraints.

## Problem Statement

We consider a set of providers (or backends) that can serve the same model family. Provider `j` has:

- A (possibly request-dependent) cost `C_j(x)` where `x` includes the observable request features
  at arrival time (e.g., model, input tokens, predicted output tokens).
- A (possibly request-dependent) latency random variable `T_j(x)` with CDF `F_j(t | x)`.

Goal: minimize expected cost while satisfying a tail latency SLO for the routed traffic.

Two common formulations:

1. **Constrained cost minimization**
   - Minimize `E[C]` subject to `P(T <= L) >= 1 - alpha` (e.g., `alpha = 0.01` for P99).
2. **Scalarized multi-objective optimization**
   - Minimize `E[C] + lambda * Risk(T)` for a user-chosen `lambda`, where `Risk` can be `P(T > L)`,
     CVaR, or a quantile proxy.

## Latency Metrics and Definitions

We should report multiple latency definitions, because providers differ in streaming behavior:

- **TTFT (time-to-first-token)**: user-perceived responsiveness.
- **E2E latency**: request completion time (including full decode).
- **TPS (tokens-per-second)**: decode throughput, `output_tokens / (E2E - TTFT)`.

For a fixed SLO, be explicit about which metric is constrained (typically E2E P99).

## Measurement Plan: Provider Latency Profiles

### Providers and Categories

Define *categories* as `(model, region, modality)` or a coarser partition that is stable across
providers. For each category, identify one or more eligible providers (local + multiple APIs).

### Dummy Request Probing

To estimate `F_j(t | x)`, periodically send dummy requests that span a grid over:

- Input token buckets (e.g., `n_in` in `{0-256, 256-1k, 1k-4k, 4k+}`).
- Output token targets (or early-stop lengths) to control decode time.
- Time-of-day / day-of-week (to capture diurnal load).

Record TTFT, E2E, output tokens, and any provider-side timing fields.

### Decomposing Latency into Prefill and Decode

When possible, fit a simple decomposition:

`T_j(x) ≈ TTFT_j(n_in, m) + n_out / TPS_j(n_in, m) + network_j`

This helps:

- Compare providers fairly (prefill vs decode bottlenecks).
- Support extrapolation to unseen token lengths.
- Provide a sanity check for the assumption "prefill and decode throughput are similar across
  providers" (or quantify how they differ).

If decomposition is too noisy, default to empirical CDFs per bucket.

### Estimators

For each `(provider j, category, feature bucket)`:

- Estimate empirical CDF `F_j(t)` and survival `S_j(t) = 1 - F_j(t)`.
- Estimate conditional residual life:
  `r_j(t) = E[T_j - t | T_j > t] = (∫_t^∞ S_j(u) du) / S_j(t)`
- Use bootstrap confidence intervals for P99 (important for ICML-level claims).

## SLO Under Mixing: "Dynamic P99 Tolerance"

If the router chooses provider `j` with probability `pi_j` within a category, the mixture latency
distribution is:

`F_mix(t) = Σ_j pi_j F_j(t)`

Therefore, the P99 constraint `F_mix(L) >= 0.99` does **not** require each provider to have
`F_j(L) >= 0.99`. Some providers can exceed the SLO individually if their routing probability is
sufficiently small and compensated by faster providers. This is the precise meaning of "dynamic
P99 tolerance".

Practically: we filter providers using a *soft* criterion first (e.g., `F_j(L) >= 0.90`) and then
solve for `pi` to satisfy the final `0.99` target.

## Offline Provider Selection Under a Tail SLO

### Linear Program (LP) for a Fixed SLO Level

For a category, assume known costs `c_j` (expected cost per request) and CDF values `F_j(L)` at the
target deadline `L`. Then:

- Decision variables: `pi_j >= 0`, `Σ_j pi_j = 1`
- Objective: minimize `Σ_j pi_j c_j`
- Constraint: `Σ_j pi_j F_j(L) >= 1 - alpha`

This is a linear program. A useful structural consequence:

- At an optimal extreme point, at most **two** providers have non-zero `pi_j`.

This gives a principled justification for "choose at most two providers on the Pareto frontier"
and makes the system behavior easier to reason about and deploy.

### Pareto Frontier and User-Controlled Tradeoff

For each provider, define the point `(cost, p99)` (or `(cost, violation_prob)`).

- Filter dominated providers (Pareto pruning).
- For the remaining candidates, compute either:
  - The LP solution at target `L`, or
  - A scalarized objective `cost + lambda * violation_prob` for a user-specific `lambda`.

Expose `lambda` (or equivalently a target `L`) as a user-controlled knob in the router.

## Online Hedging (Tail-Latency Control)

### Basic Hedging Model

Suppose we start on provider `j` at time `0`. If no response by time `h`, we send a duplicate to a
fast provider `f` (or the current fastest feasible provider). The completion time is:

`T_hedge = min(T_j, h + T_f)`

Given empirical `F_j` and `F_f`, we can compute:

`P(T_hedge > L) = P(T_j > L, T_f > L - h)`

Under independence, `= S_j(L) * S_f(L - h)`. This yields a direct way to choose `h` to satisfy a
tail SLO.

### Conditional-Expectation Trigger ("Residual Life")

An alternative trigger that does not require full independence modeling:

- Compute residual life `r_j(t) = E[T_j - t | T_j > t]` from the survival curve.
- Hedge when `t + r_j(t) + E[T_fast] > L`.

This is interpretable: if the expected remaining time plus a fallback provider would violate the
deadline, we hedge now.

### Cost Accounting Under Hedging

We must state a realistic billing model:

- **Optimistic**: canceled requests are not billed.
- **Conservative**: both attempts are billed once started.

Report both if possible; ICML reviewers will ask for this. In OpenRouter integration, cancellation
semantics depend on provider support and streaming behavior.

### Updating the Latency Profile Under Hedging

Hedging changes the effective latency distribution. For each hedging policy, estimate the induced
`F_policy(t)` via:

- Analytical computation under independence (fast for offline studies), or
- Replay simulation using sampled latencies from empirical distributions (more robust).

This enables offering a stronger SLO externally (e.g., turning a "P99=2s" provider into a "P99=1s"
policy at additional cost).

## Integration with Existing Stages (Optional)

This latency module can be layered on top of existing constraints:

- **Stage 1 (quota-only):** treat subscription as zero-marginal-cost but quota-limited; latency
  filtering and hedging apply within the subscription-compatible provider set and the API fallbacks.
- **Stage 2 (concurrency + API overflow):** treat local capacity as a loss system (no queueing) and
  route overflow to API. Latency-aware routing can decide *which* API provider to use, and whether
  to hedge local with API after `h`.

## Evaluation Protocol (ICML-Oral Quality)

### Datasets and Workloads

- Use existing request traces (e.g., ShareGPT/BurstGPT-style) for arrival patterns and token stats.
- For latency, either:
  - Use real latency traces from probing and map requests to buckets, or
  - Build a simulator that samples `T_j(x)` from measured bucketed distributions.

Avoid leakage: routing decisions must only use arrival-time observable features.

### Baselines

At minimum:

- Single provider: cheapest, fastest, and OpenRouter default (if applicable).
- Cost-only router (ignoring latency) + hard SLO filter.
- Latency-only router (always fastest feasible).
- Pareto/LP mixture without hedging.
- Pareto/LP mixture with hedging.

### Metrics

Report jointly:

- Cost: mean cost/request, total cost, cost breakdown by provider.
- Latency: P50/P90/P99, SLO violation probability, TTFT statistics.
- Hedging overhead: duplicate rate, cancellation rate, extra cost due to hedges.
- Stability: sensitivity to time-of-day, prompt length, output length.

### Statistical Rigor

- Use bootstrap confidence intervals for P99 and violation probability.
- Use time-based splits (train/warmup vs test days) for any learned predictors.
- Run ablations: with/without hedging, different `lambda`, different `h`, different bucket schemes.

## Milestones and Deliverables

1. **Week 1: Measurement**
   - Probing harness + logging schema; initial CDFs for each provider/category.
2. **Week 2: Offline selection**
   - Pareto pruning + LP mixture solver; reproduce cost-vs-P99 curves.
3. **Week 3: Hedging**
   - Implement hedging policy; compute induced latency distributions; cost accounting.
4. **Week 4: Integration + paper-ready results**
   - Compare against OpenRouter baselines; produce main plots + ablation tables; write paper section.

## Open Questions (Decide Early)

- Which SLO should be the headline metric: TTFT P99 or E2E P99?
- What is the correct billing model for canceled/hedged requests for each provider?
- Do we need token-conditional latency modeling (likely yes for fairness)?
- How to incorporate throughput variability during decoding (TPS drift) in a tractable way?
