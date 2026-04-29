# RouteWise Algorithm Evolution Roadmap

This document is for code readers, coauthors, and AI agents. Its purpose is to
explain which algorithms exist in the repo, why they were introduced, which ones
are historical, and which ones are part of the current paper story.

This is not the paper submission experiment plan. Submission experiments live in
`docs/PAPER_SUBMISSION_EXPERIMENT_PLAN.md`.

## Current Target Story

The current paper story should converge on:

```text
cost router
  -> LP-TTFT-budget body selector
  -> probability-target hedging
  -> Explorer feedback
```

Responsibility split:

- `LP-TTFT-budget` handles normal/body TTFT under a range-normalized
  per-request cost budget.
- `Hedge-ProbTarget` handles individual tail requests after primary dispatch.
- `Explorer` reuses paid hedge traffic as latency-profile feedback.

Important code-status caveat:

- The latest method is implemented mostly in the tiered-capacity sidecar:
  `experiments/tiered_capacity/lp_budget_eval.py`.
- Some older named strategies in `rwsim/policies/composer.py` still point to
  `LP-CDF` or `Hedge-Economic`. Treat that file as a migration map, not as the
  final paper truth until the latest method is promoted.

## Naming Table

Use these names consistently in discussions, code comments, plots, and paper
drafts.

| Name | Meaning | Current status |
| --- | --- | --- |
| `LP-CDF` | Old LP: minimize cost subject to `sum pi_j F_j(SLO) >= rho` | Historical baseline |
| `Residual hedge` | Old additive rule: `elapsed + E[primary residual] + E[backup] > SLO` | Retired |
| `Hedge-Economic` | `P_viol(t) * F_backup(remaining) > C_b / V` | Historical smart hedge baseline |
| `LP-V2` / `P50-band` | V2 selector: Pareto on `(P50, cost)`, then cheapest near-best P50 | Historical routing baseline |
| `LP-TTFT-budget` / `LP-RangeBudget` | Latest body selector: minimize body TTFT under `B_p = c_min + p(c_max - c_min)` | Current target |
| `Hedge-ProbTarget` | Latest hedge: latest safe `t` with `P_success_if_hedge(t) >= rho` | Current target |
| `Explorer` | Hedge-as-probe feedback into latency profiles | Current target |

When someone says "LP v2" in older discussion, verify context. In the current
repo, the V2 route is mostly `V2Router` / P50-band. The later TTFT-budget LP is
the one we want for the paper's current main method.

## Algorithm Timeline

### 1. Phase 3: `LP-CDF`

Original latency-router LP:

```text
minimize    sum_j pi_j * c_j
subject to  sum_j pi_j * F_j(SLO) >= rho
            sum_j pi_j = 1
            pi_j >= 0
```

Meaning:

- Choose the cheapest provider mix that satisfies a TTFT-CDF SLO target.
- Implement routing weights via SWRR.

Code:

- `rwsim/policies/latency_routers/online_lp.py`
- Strategy aliases: `lp_mix`, `lp_hedge`, `lp_explorer`

Status:

- Historical baseline.
- Still useful as a comparison because it shows the older "LP handles tail" design.

### 2. Initial Smart Hedging: `Residual hedge`

Old idea:

```text
hedge if elapsed + E[T_primary - elapsed | T_primary > elapsed]
         + E[T_backup] > SLO
```

Why retired:

- This matches a serial cancel-and-resend model.
- Our execution is parallel racing: keep primary running, dispatch backup, return
  the first response.
- The additive rule is too pessimistic for `min(primary, h + backup)`.

Where to find it:

- Removed from current tree.
- Historical code is visible via:
  `git show 511aca5:experiment/strategies/smart_hedging.py`

Status:

- Retired. Only use as historical explanation or ablation if needed.

### 3. Upgraded Smart Hedging: `Hedge-Economic`

Economic rule:

```text
P_viol(t) * F_backup(L - t - delta) > C_b / V
```

where:

- `P_viol(t) = P(T_primary > L | T_primary > t)`
- `F_backup(...)` is backup success probability within remaining SLO budget.
- `C_b / V` is backup cost divided by SLO-violation penalty.

Why introduced:

- Correctly models parallel racing.
- Adds a cost-benefit interpretation.

Code:

- `rwsim/policies/hedgers/smart_economic.py`
- Strategy aliases still reference this in `rwsim/policies/composer.py`.

Status:

- Historical smart hedge baseline.
- Not the final current paper hedge if we choose probability-target hedging.

### 4. LP-V2 / P50-Band Route Selector

V2 changed provider selection, not the hedge formula.

Selector:

1. Estimate rolling P50.
2. Apply eligibility filters.
3. Pareto prune on `(P50, cost)`.
4. Keep providers within a near-best P50 band.
5. Pick the cheapest provider in the band.

Why introduced:

- P50 looked more stable and predictable than P99.
- This separated body-latency selection from tail hedging.

Code:

- `rwsim/policies/latency_routers/v2.py`
- Strategy aliases: `v2_only`, `v2_p50_hedge`, `v2_explorer`

Status:

- Historical routing baseline.
- Useful for code archaeology and maybe a broader ablation, but not required for
  the submission experiment matrix unless we explicitly want it.

### 5. Latest Body Selector: `LP-TTFT-budget`

Latest LP idea:

```text
minimize    sum_j pi_j * Tbar_j(t)
subject to  sum_j pi_j * c_eff(j) <= B_p(i, t)
            sum_j pi_j = 1
            pi_j >= 0

where       B_p(i, t) = c_min(i, t) + p * (c_max(i, t) - c_min(i, t))
            c_min     = min feasible c_eff(j)
            c_max     = max feasible c_eff(j)
            p in [0, 1]
```

Meaning:

- LP handles the body of TTFT, not the tail.
- `p` is the cost-latency knob:
  - `p = 0` gives the cheapest feasible cost envelope.
  - `p = 1` gives the loosest envelope over the current feasible provider set.
  - intermediate values trace a Pareto frontier between cost and latency.
- This replaces the older `tau * v_hat_i` anchor, which was harder to explain
  and depended on how `v_hat_i` was defined.
- Hedging handles tail SLO rescue after dispatch.

Code:

- `experiments/tiered_capacity/lp_budget_eval.py`
- Current sidecar variants still include older `budget_vhat_t*` runs and
  provider-percentile `budget_body_p*` ablations. Before final paper runs, add
  or promote range-budget variants whose RHS is exactly
  `c_min + p(c_max - c_min)`.

Status:

- Current target body selector.
- Formula updated after Apr 26 discussion with Juncheng: use range interpolation
  over the feasible provider cost envelope, not `tau * v_hat_i`.
- Needs promotion/wrapping as named runnable mainline strategy before final paper
  experiments.

### 6. Latest Hedger: `Hedge-ProbTarget`

Probability-target hedge:

```text
P_success_if_hedge(t)
  = P(not violate | t)
    + P(violate | t) * P(backup succeeds | t)
```

Dispatch rule:

```text
t* = latest t such that P_success_if_hedge(t) >= rho
```

Operational meaning:

- Wait as long as possible while still keeping combined success probability above
  target.
- Avoid earliest-trigger behavior collapsing into immediate replication.

Code:

- `experiments/tiered_capacity/lp_budget_eval.py`
- Key helpers: `_combined_success_probability_after_hedge`,
  `_find_latest_safe_hedge_time_ms`, `_apply_probability_target_hedge`

Status:

- Current target hedger.
- Needs promotion/wrapping as a named hedger if the paper presents it as mainline.

### 7. Explorer / Hedge-As-Probe

Explorer semantics:

- If a hedge fires, record the backup provider's raw TTFT sample.
- Feed that sample into the backup provider's rolling latency profile.
- Keep or separately ablate dedicated background probing.

Important distinction:

- Plain hedge-as-probe should not change backup selection; it only adds feedback.
- Random exploratory backup selection is a separate policy knob. It may improve
  profile freshness, but it can trade short-term cost/tail performance.

Code:

- `experiments/tiered_capacity/lp_budget_eval.py`
- Look for explicit `*_explorer` variants, `explorer_feedback_count`, and
  backup-selection helpers. Plain `*_hedge` variants are hedge-only and do not
  feed backup samples into latency profiles. Explicit `*_randombackup` variants
  enable the separate random backup-selection ablation.

Status:

- Current target full-method feature.
- Must be independently toggleable for clean ablations.

## Code Map

| Path | What it contains | How to treat it |
| --- | --- | --- |
| `rwsim/policies/latency_routers/online_lp.py` | `LP-CDF` and SWRR | Historical baseline / old named strategies |
| `rwsim/policies/latency_routers/v2.py` | `LP-V2` / P50-band selector | Historical routing baseline |
| `rwsim/policies/hedgers/smart_economic.py` | `Hedge-Economic` plus older hedge helpers | Historical smart hedge baseline |
| `experiments/tiered_capacity/lp_budget_eval.py` | `LP-TTFT-budget`, `Hedge-ProbTarget`, Explorer sidecar | Current target implementation source |
| `rwsim/policies/composer.py` | Alias registry for migrated strategies | Migration map; may lag current target story |
| `docs/archive/` | Older design/results notes | Useful for archaeology, not final truth |
| `docs/ALGORITHMS.md` | Broad architecture/canonical notes | Useful, but check against this roadmap before paper edits |

## What New Readers Should Do

1. Read this roadmap first.
2. Use `LP-CDF`, `Hedge-Economic`, and `P50-band` as historical baselines.
3. Treat `LP-TTFT-budget + Hedge-ProbTarget + Explorer` as the target paper
   method.
4. Check whether a runnable strategy is truly latest before generating figures.
5. Do not use `joint_hedge` or `v2_p50_hedge` as evidence for the latest method
   unless the experiment explicitly says it is a historical ablation.

## Cleanup TODO

To make the repo less confusing:

1. Promote `LP-TTFT-budget` from sidecar helper to a named strategy.
2. Promote `Hedge-ProbTarget` from sidecar helper to a named hedger.
3. Keep Explorer as an explicit variant/flag rather than coupling it to every
   hedge run.
4. Update `rwsim/policies/composer.py` with latest strategy aliases.
5. Rename or clearly document `rednote` vs `Enterprise`.
6. Make experiment outputs include the exact algorithm family names above.
