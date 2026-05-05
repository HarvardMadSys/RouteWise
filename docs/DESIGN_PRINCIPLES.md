# RouteWise Simulator: Design Principles

This document captures the design philosophy that the simulator and paper
experiments are organised around. It is a synthesis of the 2026-04-26
design discussion with Juncheng plus the existing structure under
``rwsim/`` and ``experiments/``.

It exists to give code reviewers, future contributors, and the eventual
NSDI artifact evaluator a single place to look when asking "why is this
designed this way?".

---

## 1. Core Philosophy

> "The goal of the simulator is to make sure that we can understand every
> result. Right now we don't understand that."

**Reasoning beats realism.** Simulator design choices should be made for
*explainability*, not for fidelity to a real system. The simulator is a
tool for isolating algorithm behaviour, not for predicting absolute
production numbers — for absolute numbers we use end-to-end OpenRouter
experiments and trace replay.

Three corollaries:

1. **One variable per experiment.** Never let two things change between
   adjacent stages. If a result is unexpected, you must be able to point
   to the single new variable that caused it.
2. **Build complexity gradually.** Start with the simplest distribution
   (``Uniform``), the simplest cost structure (same cost for all),
   the simplest policy (no hedging). Layer on one mechanism at a time.
3. **Don't mimic real providers in the simulator.** The simulator is for
   reasoning. Provider distributions can be calibrated to real traces,
   but only as a later step, and only when the simpler families have
   already established that the algorithm behaves as expected.

---

## 2. The Three Orthogonal Axes

The full simulator grid is generated from three independent axes plus a
workload axis. **Do not invent named scenarios outside this grid** — the
hand-authored ``s6/s7/s8/s9/unified_pool`` YAMLs were dropped on
2026-05-05; the canonical scenarios are S0-S3 from
``EXPERIMENT_LAYOUT.md`` §3.1. Every cell of the grid is identified by
``(stage, distribution, policy, workload)``.

### 2.1 Stage (Provider Setup)

| Stage | Provider Setup | Purpose |
|------:|----------------|---------|
| 0 | All ``S_A`` providers, **same** latency, different cost | Cost router fantasy test — algorithm should always pick the cheapest |
| 1 | All ``S_A`` providers, **same** cost, different latency | Latency router only — algorithm should pick the fastest |
| 2 | All ``S_A`` providers, different cost **and** latency | Cost-latency tradeoff — algorithm should sweep the Pareto frontier as ``p`` varies |
| 3 | Full joint: ``S_A`` + ``S_Q`` (quota) + ``S_C`` (concurrency) | Headline scenario — capacity tiers + subscription pricing on top of Stage 2 |

Stage 0 is the cost-router-only fantasy test (Juncheng's 2026-04-28
directive: "It's the same latency, different costs. We should choose
the cheap one most of the time."). It exists so the cost-routing path
is exercised in isolation before Stage 2 introduces the cost-latency
tradeoff.

Stage 3 intentionally combines two new variables (capacity + subscription
pricing) because Stages 1 and 2 already validated each building block
independently. Stage 3 is the "everything together" headline, not a
fine-grained ablation. If Stage 3 misbehaves, debug by going back to
Stage 2 with smaller perturbations.

### 2.2 Latency Distribution Family

| Family | Class | Purpose |
|--------|-------|---------|
| ``uniform`` | ``rwsim.world.distributions.Uniform`` | Sanity baseline. Bounded support, no tail. Analytical reasoning is trivial. |
| ``normal`` | ``rwsim.world.distributions.Normal`` | Symmetric, light tail. Tests algorithm behaviour without heavy-tail confounders. |
| ``heavy_tail`` | ``rwsim.world.distributions.LogNormal`` (alias ``HeavyTail``) | Heavy-tailed, intended to approximate real provider TTFTs. |

The distribution registry lives at ``rwsim.world.distributions.LATENCY_FAMILIES``.
All three classes implement the same interface:

    sample(rng, size) -> ndarray
    p50() -> float
    p95() -> float
    p99() -> float
    quantile(q) -> float
    mean() -> float
    std() -> float
    cdf(value) -> float

``cdf()`` is required for analytical SLO evaluation in RouteWise hedging;
``p50()``, ``p95()``, ``p99()``, and ``quantile()`` keep provider code
distribution-agnostic so non-LogNormal scenarios can flow through the same
policy implementation without special cases.

**Family default shape.** Each family interprets the same ``shape``
parameter differently. The defaults in
``experiments/simulation/eval_grid._DEFAULT_SHAPE_BY_FAMILY`` are
calibrated so that (a) the tail strictly grows from ``uniform`` to
``normal`` to ``heavy_tail`` and (b) ``Normal``'s left clipping at
``MIN_LATENCY_MS`` only affects a negligible mass of samples for any
P50 used by the grid. Do not change a single number in this table
without revisiting both invariants.

### 2.3 Policy Variant

The simulator exposes six paper-name policy presets:

| Preset | Role |
|--------|------|
| ``greedy_cost`` | Baseline: cheapest available provider |
| ``greedy_latency`` | Baseline: fastest available provider |
| ``random`` | Baseline: random available provider |
| ``ablation_lp_only`` | RouteWise LP body router only |
| ``ablation_lp_hedging`` | LP body router + probability-target hedging |
| ``routewise`` | Full RouteWise: LP + hedging + hedge-as-probe explorer |

The final three presets are the RouteWise method family. They are deliberately
named by paper role, not by older runnable variant ids. The implementation
still has a cost-budget knob ``p`` inside ``RouteWisePolicy`` (default ``p=0.75``), but ``p`` is a
parameter of the policy, not a top-level simulator policy name. A dedicated
``p`` sweep may add explicit presets later if the paper needs a Pareto curve;
the committed paper surface remains the six names above.

**``p`` controls allowed budget, not realised cost.** The LP constraint
is ``Σ π_j × c_eff_j ≤ B_p``, an inequality. Higher ``p`` *permits* more
spending; it does not *force* more spending. In Stage 2 the realised
cost often grows with ``p`` because the providers are designed clean
(latency inversely ordered with cost). In Stage 3 the realised cost is
not guaranteed to be monotone: with quota / concurrency tiers in the mix,
a wider budget can let the LP pick a mix that is simultaneously faster
and cheaper.

6 policies × 4 stages × 3 distributions = **72 simulator configurations**
per workload.

### 2.4 Workload (4th Axis)

Three workloads stress different request profiles:

| Paper id | Runner dataset id | Source | Profile |
|----------|-------------------|--------|---------|
| ``sharegpt_burstgpt`` | ``burstgpt`` | BurstGPT 30-day arrivals/token counts + reused ShareGPT text | Standard chatbot, moderate output lengths |
| ``freeinference``     | ``freeinference`` | FreeInference logs | Multi-model, high variance in I/O |
| ``enterprise``        | ``rednote`` | RedNote enterprise logs | Long output, low variance, high CV in arrivals |

The paper-id column is what appears in figure captions and code that
talks to the paper grid; the runner-dataset-id column is what
``experiments.simulation.lp_budget_eval`` actually loads from disk.
The mapping lives in ``WORKLOAD_DATASET_IDS`` in
``experiments/simulation/eval_grid.py`` — keep it as the single
source of truth so renames stay consistent.

**Workload-as-driver semantics.** The trace contributes only **arrival
timestamps and request/response token counts**. The ``model`` column on
each loaded request is read into ``Request.model`` but is **not** used
by the eval-grid routing pipeline — the providers in
``ProviderSetup.{SAME_COST, COST_LATENCY_TRADEOFF, JOINT_PROVIDER}`` are
deliberately abstract (``S1_fast`` / ``S2_cheap_slow`` / ``S3_quota_chutes``
etc.), have no ``supported_models`` constraint, and price every request
at ``cost_per_token × total_tokens`` regardless of which model the trace
recorded. Concrete consequences:

- For **FreeInference** (12 models) and **Rednote** (8 models),
  per-request costs in the simulator do **not** reflect real per-model
  API prices. They reflect the synthetic Stage-2 / Stage-3 cost ladder.
- All 12 FreeInference models flow through the same providers in
  proportion to their original frequency (Llama-3.3-70B at 65%,
  Llama-4-Scout at 15%, ...). The trace contributes the realistic
  *load shape* (burst structure, token-size variance), not a multi-model
  routing problem.
- **Paper claims drawn from these runs must be relative**, not absolute:
  Pareto frontier shape, the direction of ``p`` sweeps, the marginal
  effect of hedging / explorer. Statements like "X% cost saving on
  FreeInference" require the multi-model OpenRouter end-to-end stage
  in §5, **not** the simulator grid.

This is consistent with the core philosophy ("reasoning beats realism" —
§1): the simulator is a controlled environment for understanding
algorithm behaviour, while absolute multi-model pricing claims belong to
the live OpenRouter experiments.

> "The trace doesn't really matter here, right. We just send a number of
> requests to see what happens."

The trace **content** does not matter for the simulator. What matters is
that we exercise the algorithm under three different request *profiles*
(output-length distribution, arrival pattern). 135 × 3 = **405 runs**
total per seed.

### 2.5 Seed (deferred)

Single-seed iteration during development. **Multi-seed (≥5) + 95% CI
must be added before paper submission** — this is an NSDI requirement,
not a Juncheng directive, but it does not conflict with anything in the
design philosophy.

---

## 3. Invariants

Two layers, deliberately separated.

### 3.1 Structural invariants (config-level, no router execution)

Enforced by ``eval_grid.assert_grid_invariants``. These check the *shape*
of the grid, not its behaviour. They run cheaply and must pass before
any code change lands.

- 12 grid scenarios exist (4 stages × 3 distributions).
- Stage 1 cells: every provider has the same ``cost_per_token``.
- Stage 2 cells: at least 3 distinct ``cost_per_token`` values among providers.
- Stage 3 cells: contain all three tiers (``S_A``, ``S_Q``, ``S_C``) with
  positive quota / concurrency capacity.

### 3.2 Behavioural expectations (require router execution)

These describe how the algorithm *should* behave qualitatively. They are
**not** asserted in unit tests because the actual outcome depends on
shadow-price interaction (``ψ(z)`` versus ``λ(u)``), arrival pattern, and
capacity caps — all of which need the simulator to run end-to-end.

| Stage | Surface | Qualitative expectation |
|------:|---------|--------------------------|
| 1 | any policy | Equal provider cost makes the LP budget constraint non-discriminating; RouteWise should concentrate mass on the fastest provider, matching ``greedy_latency`` on routing when capacity is unconstrained |
| 2 | optional internal ``p=0`` sweep | Mass concentrates on the cheapest provider because the LP cost budget shrinks to ``c_min`` |
| 2 | optional internal ``p=1`` sweep | Mass concentrates on the fastest provider because the cost constraint vanishes |
| 2 | optional internal ``p`` sweep | Mass spreads between cheap and fast providers; **realised cost is monotone in ``p``** because providers are clean (1:2:4 cost / inverse latency) |
| 3 | optional internal ``p=0`` sweep | Mass favours subscription tiers (``S_Q`` and ``S_C`` free at the margin); exact split between ``S_Q`` and ``S_C`` depends on ``ψ(z)`` vs ``λ(u)`` interaction and is workload-dependent |
| 3 | optional internal ``p=1`` sweep | Mass favours fast providers across all tiers, **subject to** capacity caps (``S_C`` ≤ its concurrency limit; spillover to ``S_A`` once full) |
| 3 | any RouteWise policy | Capacity caps are respected (no provider exceeds quota or concurrency limit) **but realised cost is NOT required to be monotone in ``p``** — see §2.3 above |
| All | ``ablation_lp_hedging`` | Hedging materially reduces P99 / SLO violations only when the LP-only baseline is close to or above SLO. When LP-only already meets SLO comfortably, hedge fires near-zero and looks identical to LP-only — **this is correct silence, not a bug** |
| All | ``routewise`` | In stationary scenarios (the eval grid), the explorer is expected to match hedging on body metrics. Validating explorer's positive value requires drift / freshness scenarios, not the base grid |

Encoding these as fail-on-violation tests is premature: an unexpected
result should trigger investigation (calibration? bug? real?), not an
automatic CI failure. They live here as a contract between the design
and the evaluation. Document any deviation from this table in the
experiment writeup.


---

## 4. What NOT to Do

These are explicit "do not" rules drawn from the 2026-04-26 discussion
and from Juncheng's general direction:

1. **Do not show failed attempts as baselines.** Old LP, old hedging,
   ``*_no_probe`` variants are internal ablations, not paper baselines.
   The published paper shows one final method per family, not the
   exploration history.

2. **Do not pile mechanisms into one experiment.** Always run
   ``without_hedging`` and ``with_hedging`` as separate configurations,
   never bundle them.

3. **Do not use AI-generated slides for design discussions.** Slides
   exist to force you to think through every step. Hand-authored slides
   (or this kind of doc) are the canonical artifact.

4. **Do not add load-dependent latency, autocorrelated latency, or a
   discrete-event scheduler to the simulator** unless an experiment
   explicitly requires it. These add reasoning complexity without paying
   the design back. Provider tail dynamics belong in trace replay, not
   in the synthetic simulator.

5. **Do not add new ad-hoc scenarios.** New experimental questions go
   through the grid axes. If a question can't be expressed as a grid
   cell, it's a sign the question itself is under-specified.

6. **Do not delegate self-coupled tasks.** Sub-tasks delegated to
   collaborators must be fully self-contained — full ownership of one
   experiment, not a slice of a shared module.

---

## 5. End-to-End Experiments (Beyond the Simulator)

The simulator grid is the foundation, but the paper's headline numbers
come from end-to-end experiments. These are run **after** the simulator
grid stabilises, in this order:

1. **OpenRouter on-demand only** — single tier (``S_A``), cost-latency
   tradeoff at scale. Validates the algorithm against real-world tail
   behaviour.
2. **Joint live experiment** — three tiers running against real
   providers (Featherless, Chutes, OpenRouter). Limited to models that
   are available across all three (e.g., Minimax).
3. **Multi-model OpenRouter** — sweep across several models (Llama,
   Qwen, GLM, DeepSeek) within the on-demand tier.
4. **FreeInference replay** — large-scale offline replay. Deferred
   until 1–3 land. Do not start this in parallel.

---

## 6. References

- Meeting transcript: ``~/Desktop/output/RouteWise_april26.txt``
- Architecture overview: ``docs/ARCHITECTURE.md``
- Algorithm specification: ``docs/ALGORITHMS.md``
- Distribution layer: ``rwsim/world/distributions.py``
- Eval grid factory: ``experiments/simulation/eval_grid.py``
- Eval grid tests: ``tests/unit/experiments/test_eval_grid.py``
