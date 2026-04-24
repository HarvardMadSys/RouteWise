# Joint Cross-Tier Routing — Process Diary & Pitfalls

A record of how the joint-tier extension was designed, what broke, and what
the simulator caught before we could spend money on a live experiment.

## Timeline

**Stage 1 — meeting input (2026-04-14)**
Juncheng flagged the structural weakness of the current two-layer RouteWise
design: Layer 1 commits to a tier based on shadow price alone, and once a
slow subscription tier is chosen the latency router has no lever left.

**Stage 2 — whiteboard design (2026-04-17)**
We sketched a "joint" alternative: fold per-tier capacity constraints into
a unified effective cost c_eff, then run a single cross-tier selection that
considers both cost and latency. Proposed decomposition:

    c_eff[j] = marginal_cost[j] + psi_Q(z_j) + lambda_C(u_j)

    V2 P50-rank  ->  near-best P50 band  ->  cheapest-in-band by c_eff

**Stage 3 — skeleton (this commit)**
Implemented the scenario infrastructure (S_A/S_Q/S_C providers, shadow price
functions, three tier scenarios S6/S7/S8) and the two baseline strategies.

**Stage 4 — first run**
`two_layer` blew the SLO in S6 as expected. But `joint_v2` also looked bad
in S7 and S8 — 2-2.5x more expensive than `two_layer` by routing 100% to
S_A and wasting all the free subscription capacity.

**Stage 5 — root cause**
The P50-band filter is correct for within-tier diversification (V2's
original use) but wrong for cross-tier: a subscription provider can be
slower on P50 by a wide margin and still be comfortably inside the SLO.
Excluding it leaves free money on the table.

**Stage 6 — fix**
Replaced the P50-band filter with an SLO-anchored P95 safety filter:
    candidates = [p : P95(p) <= SLO * 0.8]
    primary    = argmin c_eff[p]

This matches or beats `two_layer` in all three scenarios.

## The Pitfalls

### Pitfall 1 — Importing a within-tier rule into a cross-tier setting

**What happened**
V2 router was designed to solve over-diversification *within* a single tier
of S_A providers that have similar P50s. Its "pick lowest-P50 among the
near-best band, then cheapest" rule is tuned for a specific regime: all
candidates are fast by the SLO's standard and cost is the tiebreaker.

When we ported the rule to cross-tier, that regime assumption silently
broke. Across tiers the P50 spread is much larger (a subscription provider
might be 5-10x slower on P50 than S_A), and cost matters more than P50
ordering because subscription marginal cost is 0.

**Why we missed it initially**
The failure is invisible in S6 (where P50 differs by 20x and the P50-band
coincidentally picks the right answer). The failure only shows in S7 and
S8, which we hadn't written yet when we first sketched the design. So the
bug was baked into the design by the time we got to implementation.

**How the simulator caught it**
S7 and S8 were written in the same sitting as the selector code. The first
end-to-end run dumped the full five-scenario table. Joint's 2.5x cost in
S7 stood out immediately against two_layer's $2.30e-4. Root-causing took
about twenty minutes of staring at the filter logic.

**Lesson**
When reusing a design across a new regime, re-derive the decision rule from
first principles. Don't assume the rule survives the change of setting just
because the surface API looks the same.

### Pitfall 2 — Misreading the ACF finding

**What we thought it said**
"P50 is predictable, P99 is not -> therefore use P50 as the primary
selection signal."

**What it actually says**
"P50 *estimates* are temporally stable; P99 estimates are noisy." This
justifies using P50-based inputs for our selection code, but says nothing
about what the *objective* should be. In a cost-latency tradeoff setting,
the objective is still cost subject to latency safety — it was never "pick
the lowest P50."

**Fix**
Use P95 (robust like P50, still captures the tail) as the SLO-safety input.
Use cost as the selection objective.

### Pitfall 3 — Conflating "fast" with "safe"

The V2 P50-band says "among the fastest, pick cheapest." The SLO-anchored
filter says "among the safe, pick cheapest." The distinction matters:
- Fast means "near-best P50."
- Safe means "P95 within SLO."

A provider can be safe without being fast. Subscriptions commonly fall in
that zone (moderately slower P50, still SLO-safe, free). The cross-tier
router must recognize the safe-but-slow regime as a first-class case.

### Pitfall 4 — Hedging default that does not compose

The hedge trigger we inherited fires whenever the primary's observed TTFT
exceeds `1.5 * P50`. In S7, S_Q's P50 is 300 ms and the SLO is 2000 ms, so
the primary can comfortably exceed 1.5x P50 (e.g. 450 ms) without any SLO
threat. The hedge still fires, dispatches a real-money backup, and inflates
cost by ~30% with zero SLO benefit.

The fix is not to change the economic rule — it is already correct — but
to tighten the trigger so hedging is not considered until the primary is
within SLO budget risk. The current `joint_hedge` strategy shows the
symptom, and the hedge logic is flagged for a follow-up iteration.

### Pitfall 5 — Designing with only one scenario

S6 alone would have validated the P50-band filter because the P50 spread
is extreme. S7 and S8 exposed the failure mode. The pattern is general:
**any single scenario is enough to ship a bug; diversity of scenarios is
what keeps the design honest.** The corrective effort was trivial (~20
lines of selector code), but the diagnostic effort would have been much
larger if caught in a live run.

### Pitfall 6 — Framing "joint vs two_layer" as a competition

**What happened**
After fixing the P50-band bug, the S8 results showed joint and two_layer
producing essentially identical outcomes ($3.10e-4 vs $3.06e-4, same tier
mix). The immediate reaction was "joint is not winning" — and we briefly
considered adding more scenarios to find a case where joint beats two_layer
on concurrency. That framing was wrong.

**The correct framing**
Two_layer and joint are *both* our methods. Two_layer is the existing
RouteWise design; joint is the proposed extension. The paper is not
claiming "method A beats method B in a head-to-head" — it is claiming
"the proposed extension fixes a regime where the existing design has a
real bug (S6) and provides a small improvement where the existing design
works (S7), without breaking the cases it already handles correctly (S8)."

In fact, for concurrency the binary-availability gate is *provably* the
correct shadow-price behavior:
- Slot free  →  lambda(u) = 0          (admit)
- Slot full  →  lambda(u) = infinity   (never admit)

There is no smooth intermediate because concurrency is discrete. Any
smooth proxy like `lambda = U * u^alpha` is an approximation, and in the
limit of correct primal-dual logic it collapses back to the binary gate.
So joint-reducing-to-two_layer on S_C is not a bug in joint — it is
joint correctly recognizing that the binary gate *is* the right answer
in that regime.

**Why it matters for the paper**
The claim "joint dominates two_layer across all tier types" is both
unnecessary and wrong. The correct (and easier to defend) claim is:

    Joint is a unified effective-cost framework that subsumes the
    existing two-layer design. It yields measurable improvement in
    two regimes:
      (i)  SLO-unsafe subscription tiers (S6): existing two-layer
           traps into the slow tier; joint's P95 filter avoids it.
      (ii) Gradually-depleting quota (S7): joint's smooth psi(z)
           handoff beats the binary tier switch.
    For binary-capacity tiers (S8), joint reduces to the existing
    two-layer behavior, which is already optimal.

This reframing also removes a headache for the evaluation section: we
don't need to justify why joint is "only slightly" better in S8. It's
exactly as good as the current design, and that's the point.

**Lesson**
When proposing an extension to an existing system, frame the evaluation
around "what did we fix and what did we not break", not around "does the
new thing win every benchmark". Equivalence on scenarios the existing
system already handles is evidence of correctness, not weakness.

### Pitfall 7 — Trusting a single-seed heatmap

**What happened**
The first phase-diagram sweep used one seed per cell. On the boundary
between regimes (p50_ratio = 10x), closed-form analysis predicts
8.3 % SLO violation for the subscription. The single-seed cell with n=90
requests showed only 2.22 %, because sample noise (stderr ~2.9 pp at that
sample size) swamped the signal. The cell rendered as "barely any
difference between joint and two_layer" when the underlying process has
a clear +6 pp SLO advantage for joint.

**Fix**
Average each cell over 5 seeds. Standard error drops from 2.9 pp to
1.3 pp, and the boundary region reports the expected ~8 pp instead of
2 pp. The cost stays within a fraction of a percent across seeds
because its dominant factor (quota consumption + API fallback mix) is
nearly deterministic once the routing decision is fixed.

**Lesson**
Whenever a metric is a proportion (SLO violation rate, hedge rate, tier
fraction), compute the sample-level standard error before trusting a
single-seed result. Cells that look "just noise" might actually be
signal dominated by undersampling.

### Pitfall 8 — Missing a hidden benefit of c_eff

**What happened**
At p50_ratio=1.0 and saturation=1.5, the phase-diagram cell reports
joint 9% cheaper than two_layer despite both strategies consuming the
full quota (200 requests) and spilling the same count (63 requests) to
S_A. The initial assumption was "if the tier counts are identical, the
costs must be too" — and the discrepancy looked like a bug.

It wasn't a bug. Both strategies send 200 requests to S_Q and 63 to
S_A, but they don't send *the same* 200. Joint uses `c_eff` which
compares `psi(z)` (shadow price, independent of token count) against
`price * tokens` (real API cost, scales with tokens). For a
small-token request, the S_A cost is small, and joint routes it to
S_A — saving the quota for larger-token requests. Two_layer ignores
token count entirely and fills the quota greedily in arrival order.

Measurement confirmed the hypothesis:
    two_layer  S_Q mean tokens = 184,  S_A mean tokens = 190
    joint      S_Q mean tokens = 189,  S_A mean tokens = 174
The 8 % lower mean-tokens-on-S_A in joint exactly explains the 9 %
cost reduction.

**Lesson**
This is the primal-dual *value-estimator* mechanism emerging
automatically from the unified effective-cost framework. It was not an
explicit design goal of the tiered extension — we built it for smooth
quota handoff — but it falls out of the same `c_eff` comparison. The
paper should credit this explicitly: joint has *three* benefits over
two_layer, not two:

    1. SLO-anchored filter (S6 fix)
    2. Smooth quota transition via psi(z) (S7 + low-p50_ratio cells)
    3. Value-aware routing via per-request `c_eff` comparison
       (emerges in every cell with both quota and overflow)

Benefit 3 is a free side effect of the design, and it is worth
naming.

### Pitfall 9 — Interpreting the cost panel in isolation

**What happened**
In the phase diagram, the rightmost columns (p50_ratio >= 10x) show
cost_ratio > 1 — meaning joint is more expensive than two_layer. Read
alone, this looks like a failure: "joint charges more money".

It isn't. In those cells two_layer is "cheap" precisely because it
routes to the slow subscription and violates SLO heavily (50 - 97 pp).
Joint pays real API money to deliver SLO compliance. The right read is:
"joint trades money for correctness in this regime."

**Fix**
Add a third panel that combines cost and SLO into an effective-cost
ratio using an explicit per-violation penalty V:

    c_effective = mean_cost_usd + V * violation_rate
    ratio = c_effective(joint) / c_effective(two_layer)

At V = $1e-3 per violation the phase diagram shows:

    Left (p50_ratio <= 5x): joint wins or ties across all saturations.
    Middle (p50_ratio == 10x): joint's SLO gain does not cover its cost
        premium -- two_layer wins effectively.
    Right (p50_ratio >= 20x): joint wins decisively because the
        violation rate difference is so large.

The V-sensitive band at 10x is exactly the kind of honest regime
description reviewers want. It should be in the paper.

**Lesson**
Whenever one axis of a comparison is a cost and another is a quality
metric, present the combined economic ratio explicitly. A naked cost
panel invites the wrong reading.

## What the Simulator Is For (Confirmed)

Before this work it was plausible that synthetic simulation was a nice-to-
have but real experiments were the source of truth. This episode is direct
counter-evidence:

- Time to detect the joint bug in simulator: ~20 minutes (implementation +
  one run + staring at the table).
- Time it would have taken in a live A/B: 24 hours minimum, plus pre-work
  to buy and configure a Chutes subscription ($20) and a Featherless
  subscription ($10), plus writing the interleaved evaluation harness
  (several hours), plus analysis turnaround.
- Estimated savings: one week of calendar time plus $30 in subscription
  fees plus engineering time for infrastructure we ended up not using.

The simulator also generated a clean paper ablation for free: the
`joint_p50band_*` strategies are now a useful comparison point ("what if
we had used the P50-band filter?") for the final writeup.

## What to Iterate Next

### A — Hedge trigger should be SLO-budget aware

Current: `fire when T_primary_observed > 1.5 * P50_primary`.
Better: `fire when T_primary_observed > SLO * (1 - safety_margin)` or when
the remaining SLO budget shrinks below the backup's P50 plus dispatch
overhead.

### B — Use rolling-window P50/P95 instead of oracle

The current joint selector reads analytical P50/P95 off the provider's true
distribution. Production does not have that luxury. Swap in the
warm-up + probing machinery from Yiyan's base simulator and re-run to
confirm the design is robust to measurement noise.

### C — Multi-provider per tier

S6/S7/S8 each have exactly one S_Q and one S_A. The real setting has many
providers per tier. Add an S6+ variant with e.g. three S_A providers of
different cost/latency and verify the joint selector still works.

### D — Correlated failures across tiers

If a cloud region goes hot, multiple providers hosted in that region will
degrade together. The current simulator samples each provider independently.
Extending to a shared-latent-event model would let us test whether the
joint design is more fragile or more robust than two-layer under correlated
provider outages.

### E — Phase diagram

Sweep (P50_Q / P50_A, workload / quota) on a grid and draw a heat-map of
`cost_ratio(joint / two_layer)`. The three scenarios above are three cells
in that grid; the whole diagram is paper-worthy and takes ~2 hours to run.

## Bottom-Line Read on the Joint Design

Joint (SLO-anchored + effective cost) matches or beats two-layer in all
three tested scenarios. The gap is dramatic in S6 (the slow-subscription
trap) and small but non-zero in S7/S8 (smooth transition vs. cliff).

The naive P50-band variant (`joint_p50band`) is strictly worse than both
designs in S7/S8 and serves as a cautionary ablation. The paper narrative
should flag this explicitly: the filter choice *is* the design decision;
the rest is accounting.
