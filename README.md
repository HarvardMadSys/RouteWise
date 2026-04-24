# RouteWise Synthetic Simulator

This repository contains the synthetic simulator used to evaluate routing
algorithms for RouteWise.

After the 2026-04-22 refactor, the simulator is organized around a single
world model and multiple routing algorithms:

- One shared world model for providers, capacity state, shadow prices,
  workloads, scenarios, and metrics.
- Multiple routing strategies registered as plugins and evaluated on the
  same scenarios.
- Paper experiment code lives under `experiments/`; reusable simulator code
  lives under `rwsim/`.

The old `experiment/` package and compatibility package surfaces have been removed.

## Quick start

From the repository root:

```bash
source .venv/bin/activate
```

Core regression check:

```bash
python tests/golden_capture.py --mode compare
```

Optional pytest wrapper:

```bash
pytest tests/test_golden.py -m slow
```

## What is canonical now

Use `rwsim/` as the canonical package surface for new code:

- `rwsim/world/`: shared world model
- `rwsim/data/`: reusable trace dataset loaders
- `rwsim/offline/`: paper offline/stage simulation primitives
- `rwsim/policies/`: target pipeline-stage policy decomposition
- `rwsim/strategies/`: registered strategy surface
- `rwsim/runner.py`: shared strategy dispatch

Historical paper workflows have canonical homes under `experiments/`.
Reusable primitives live in `rwsim/`.

The target architecture and algorithm decomposition are documented in:

- `docs/ARCHITECTURE.md`
- `docs/ALGORITHMS.md`

Config-driven experiment recipes now live under `experiments/`. The thin
inspection entrypoint is:

```bash
python scripts/run_experiment.py --experiment tiered_capacity --list
```

## Repository layout

```text
RouteWise/
  README.md
  docs/
    ARCHITECTURE.md
    ALGORITHMS.md
  rwsim/
    world/
    data/
    engine/
    offline/
    policies/
    metrics/
    strategies/
    runner.py
    schemas.py
    scenarios.py
    registry.py
  experiments/
    synthetic_latency/
    tiered_capacity/
    estimator_ablation/
    offline_stage/
  scripts/
  tests/
    golden/
    golden_capture.py
    test_golden.py
  outputs/
```

### `rwsim/world`

The shared world model includes:

- `distributions.py`: `LogNormal`
- `providers.py`: unified provider hierarchy
- `capacity.py`: quota and concurrency state
- `scenarios.py`: shared `ScenarioConfig`
- `shadow_price.py`: quota and concurrency shadow pricing
- `workload.py`: request generation
- `metrics.py`: shared `StrategyRun`

### `rwsim/strategies`

Strategies are registered through a shared registry. The current families are:

- `baseline`: `cheapest_fixed`, `fastest_fixed`, `round_robin`,
  `oracle_per_window`
- `lp`: `lp_mix`, `lp_hedge`, `lp_explorer`, `lp_explorer_no_probe`
- `v2`: `v2_only`, `v2_p50_hedge`, `v2_explorer`, `v2_explorer_no_probe`
- `tiered`: `two_layer`, `joint_nohedge`, `joint_hedge`,
  `joint_p50band_nohedge`, `joint_p50band_hedge`

New strategies should be added through the shared registry instead of adding
new ad hoc dispatch logic.

### `rwsim/policies`

The migration target splits routing policy code by pipeline stage:

- `value_estimators/`
- `cost_routers/`
- `latency_routers/`
- `hedgers/`
- `composer.py`

Current strategy aliases are mapped to these stages in
`rwsim/policies/composer.py` and documented in `docs/ALGORITHMS.md`.

### `rwsim/scenarios.py`

`rwsim/scenarios.py` is reserved for generic scenario builders. Concrete S6,
S7, S8, S9, and `unified_pool` definitions live as YAML configs under
`experiments/tiered_capacity/configs/`.

## Supported scenario families

Golden baselines cover five families:

- `latency_synthetic`: S1-S5
- `latency_sanity`: Step 1-Step 5 sanity suites
- `tiered`: S6-S9 plus `unified_pool`
- `calibrated`: MM25 calibrated scenarios
- `stress`: ST1-ST3

All golden artifacts are stored under `tests/golden/`.

## Running experiments

Use `scripts/run_experiment.py` for config-driven runs. Paper-era operational
runners that still orchestrate a full sweep live under `scripts/experiments/`;
they import canonical `rwsim/` and `experiments/` modules and write generated
artifacts under `outputs/`.

### Latency-only synthetic scenarios

```bash
python scripts/experiments/run_synthetic.py
```

Outputs:

- `outputs/synthetic/<scenario>/summary.json`
- latency and provider-selection plots for each scenario

### Sanity-check scenarios

```bash
python scripts/experiments/run_sanity_check.py
```

Outputs:

- `outputs/sanity_check/<step>/summary.json`
- responsiveness plots
- auto-generated ground-truth checks for sweep scenarios

### Tiered scenarios

```bash
python scripts/experiments/run_joint.py
```

Outputs:

- `outputs/joint/<scenario>/summary.json`
- per-scenario plots
- cross-scenario summary figure

### MM25 calibrated scenarios

```bash
python scripts/experiments/run_joint_mm25_baselines.py
```

### Stress scenarios

```bash
python scripts/experiments/run_stress_tests.py
```

Outputs:

- `outputs/stress/<scenario>/summary.json`
- stress-specific plots such as tier-over-time or quota-over-time

## Canonical Python entrypoints

For new code, prefer the `rwsim` namespace:

```python
from rwsim import LATENCY_STRATEGIES, TIERED_STRATEGIES, run_registered_strategy
from rwsim.world import Provider, ScenarioConfig, StrategyRun
```

## Verification discipline

Any structural change to the simulator should satisfy all of the following:

1. `python tests/golden_capture.py --mode compare` stays green.
2. No algorithm decision logic changes unless the change is explicitly a
   research change rather than a refactor.

## Known Algorithm Caveats

These pre-existing behavioral issues were intentionally not mixed into the
structural refactor:

- `two_layer` can select a provider that is unavailable if another provider in
  the same tier is available. Golden baselines preserve the old behavior.
- In `rwsim/policies/latency_routers/tiered_filters.py`, `provider_p95_at()` checks
  `hasattr(provider, "_active_dist")`.
- Stress scenario `st2_s_q_degradation` uses `TieredProvider` with
  `shift_time` and `ttft_dist_after`, not `ShiftingProvider`.
- As a result, request sampling can observe degradation while the
  strategy-side P95 filter still sees the pre-shift profile.
- EMA, histogram, and oracle value estimators assume completed requests have
  non-empty `response_tokens`, even though `Request.response_tokens` is typed
  as optional.

These should be handled as separate behavioral fixes, not as part of the
structural refactor.

## Notes on Layout

- `rwsim/` is the canonical package surface for new development.
- Offline/stage core types, config loading, quota/cost/cache, and simulator
  code now live in `rwsim/offline/` and `experiments/offline_stage/`.
- Offline/stage paper strategies now live in
  `experiments/offline_stage/strategies/`.
- The OpenRouter latency profiling probe now lives at
  `experiments/offline_stage/latency_profiling.py`.
- Full-sweep runner scripts live under `scripts/experiments/`; the repository
  root no longer contains `run_*.py` entrypoints.
