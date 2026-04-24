# RouteWise Simulator Architecture

This document defines the target repository architecture for RouteWise
simulation and paper experiments. The goal is to make the codebase composable:
each module should behave like a small, testable building block with a clear
contract.

The current repository is still in migration, but the root package boundary is
now explicit: `rwsim/` is the simulator package, `experiments/` contains
config-driven experiment recipes, and historical code is quarantined under
`legacy/experiment/`. The target is to make `rwsim/` the single source of
truth.

## Target Layout

```text
RouteWise/
  rwsim/
    data/
      loader.py
    world/
      distributions.py
      providers.py
      workload.py
      quota.py
      concurrency.py
    engine/
      simulator.py
      state.py
      events.py
    policies/
      value_estimators/
      cost_routers/
      latency_routers/
      hedgers/
      base.py
      composer.py
    metrics/
      aggregate.py
      run.py
      slo.py
    scenarios.py
    schemas.py
    registry.py

  experiments/
    synthetic_latency/
      configs/
      experiment.py
      README.md
    tiered_capacity/
      configs/
      experiment.py
      README.md
    estimator_ablation/
      configs/
      experiment.py
      README.md

  scripts/
    run_experiment.py
    plot_experiment.py

  legacy/
    experiment/

  tests/
    unit/
    integration/
    golden/

  outputs/

  docs/
    ARCHITECTURE.md
    ALGORITHMS.md
    LEGACY_AUDIT.md
```

## Layer Responsibilities

### `rwsim/`

`rwsim/` is the core RouteWise simulator package. It contains reusable building
blocks for evaluating routing policies under controlled workloads and provider
constraints.

The expected core contract is:

```text
scenario + policy + seed -> simulation result
```

`rwsim/` should contain:

- Provider, quota, concurrency, workload, and latency distribution models.
- Trace data loaders used by experiments.
- A shared simulation engine and execution state.
- Policy stage interfaces and reusable policy implementations.
- Metric streams, aggregation, and standard result schemas.
- Generic scenario types and config builders.

`rwsim/` should not contain:

- Paper-specific S6/S7/S8/unified-pool scenario definitions.
- Command-line argument parsing.
- Experiment orchestration.
- Paper figure paths.
- Experiment output directory decisions.
- Plotting logic that is only meaningful for a specific paper experiment.
- Experiment-specific analysis modules.

### `rwsim/scenarios.py`

`rwsim/scenarios.py` may define generic scenario types and config builders:

- `build_scenario(config)`
- `load_scenario_config(path)`

Shared schema objects such as `ScenarioConfig`, `ProviderConfig`,
`WorkloadConfig`, and `ShiftEvent` live in `rwsim/schemas.py`.

It must not define concrete paper scenarios such as:

- `make_s6_slow_q_trap()`
- `make_s7_quota_depletion()`
- `make_unified_pool_scenario()`

Concrete paper scenarios belong in experiment configs, for example:

```text
experiments/tiered_capacity/configs/s6_slow_q_trap.yaml
experiments/tiered_capacity/configs/unified_pool.yaml
```

### `rwsim/engine/`

`rwsim/engine/` owns the simulation loop. Policies should make routing
decisions; they should not each run their own request loop, update global
state, or manually assemble result arrays.

Target shape:

```python
class Simulator:
    def run(self, scenario, policy, seed):
        for request in scenario.workload:
            decision = policy.route(request, self.state)
            outcome = self.execute(decision, request)
            self.state.update(outcome)
            self.metrics.record(request, decision, outcome)
        return self.metrics.result()
```

This should replace the current pattern where many `_run_*` functions each
implement their own loop, state updates, and metrics collection.

### `rwsim/policies/`

`rwsim/policies/` should be organized by pipeline stage, not by one strategy
per file.

Target shape:

```text
policies/
  value_estimators/
  cost_routers/
  latency_routers/
  hedgers/
  base.py
  composer.py
```

The intended abstraction is:

```text
value estimation + cost routing + latency control + hedging
```

Strategies such as `lp_explorer`, `lp_explorer_no_probe`, `joint_hedge`, and
`joint_nohedge` should become pipeline configurations where possible, not
separate implementations with duplicated loops.

### `rwsim/schemas.py`

`rwsim/schemas.py` should hold cross-boundary schemas that are shared across
world, engine, policies, metrics, and experiments.

Likely examples:

- `Request`
- `ProviderConfig`
- `ScenarioConfig`
- `RoutingDecision`
- `RoutingOutcome`
- `SimulationResult`

Types that are local to one module should stay local. `schemas.py` must not
become a dumping ground.

During migration, duplicate concepts such as current `ProviderConfig` and
runtime `Provider` representations should be reconciled into one coherent
model.

### `experiments/`

`experiments/` contains reproducible paper experiment recipes. An experiment
combines simulator building blocks with fixed configs, policy pipelines,
seeds, metrics, and output schemas.

Experiment directories may be single-file when the experiment is simple:

```text
experiments/synthetic_latency/
  configs/
  experiment.py
  README.md
```

If an experiment becomes large, it may split into `run.py`, `analyze.py`, and
`plot.py`. That split is optional, not mandatory.

Experiment code may write outputs, but only under `outputs/`.

### `scripts/`

`scripts/` contains thin command-line entrypoints. Scripts should parse
arguments and dispatch into `experiments/`. They should not contain research
logic, simulator logic, or plotting logic beyond selecting an experiment
command.

### `legacy/`

`legacy/` is a temporary compatibility area for old code while the new
architecture is being reproduced.

It exists to preserve baseline comparability during the migration. Some code
inside `legacy/experiment/` is already a wrapper over `rwsim/` or
`experiments/`; the rest is historical offline/stage experiment code. The
detailed inventory lives in `docs/LEGACY_AUDIT.md`.

`legacy/` should be removed once the new architecture reproduces the required
golden baselines and paper experiments.

The rule is:

```text
legacy/ is kill-on-reproduce.
```

### `tests/`

`tests/` proves that both individual building blocks and assembled experiment
paths work.

The intended split is:

- `tests/unit/`: fast tests for individual modules.
- `tests/integration/`: small end-to-end scenarios.
- `tests/golden/`: slow regression baselines that protect refactors from
  changing behavior.

### `outputs/`

`outputs/` contains generated experiment artifacts and is ignored by git.

Typical contents include:

- Config snapshots.
- Metadata such as timestamp, git commit, seed set, and environment.
- Raw per-request or per-seed results.
- Aggregated summaries.
- Figures and tables.
- Logs and debug dumps.

## Dependency Direction

Dependencies must flow downward:

```text
scripts -> experiments -> rwsim
tests   -> rwsim / experiments
legacy  -> isolated compatibility code
```

Forbidden dependencies:

- `rwsim` must not import `experiments`.
- `rwsim` must not import `scripts`.
- `rwsim` must not depend on generated files in `outputs/`.
- `experiments` must not depend on generated files in `outputs/` as source
  inputs unless the dependency is explicit and documented.

## Migration Plan

Migration should be incremental and behavior-preserving.

Current status:

- `rwsim/world/` owns the leaf world primitives: distributions, capacity state,
  providers, workload generation, scenario containers, shadow pricing, and run
  metrics.
- `rwsim/data/` owns reusable trace-data loading helpers used by experiments.
- `legacy/experiment/scripts/simulate/synthetic/_core/` world modules are
  compatibility wrappers.
- `rwsim/strategies/registry.py` owns the canonical strategy registry surface.
- `rwsim/strategies/latency_impl.py` and `rwsim/strategies/tiered_impl.py`
  own the reproduced strategy loops; legacy synthetic runner modules under
  `legacy/experiment/` are wrappers.
- `rwsim/policies/` now owns the migrated latency routers, hedgers, value
  estimators, and initial cost-router selectors. Remaining work is to move
  full request-loop execution into `rwsim/engine/` behind the composer.
- `experiments/tiered_capacity/configs/` owns S6/S7/S8/S9/`unified_pool`
  scenario definitions and can run one registered strategy through
  `scripts/run_experiment.py`.

Recommended order:

1. Keep current golden baselines green.
2. Use this architecture document and `docs/ALGORITHMS.md` as the migration
   contract.
3. Keep existing `rwsim/` facade working while moving implementations behind
   it.
4. Move leaf modules first, starting with distributions and workload.
5. Add unit tests for each moved module.
6. Move provider, quota, concurrency, shadow price, and metric primitives.
7. Extract the shared simulation engine.
8. Move policies behind stage interfaces.
9. Convert concrete paper scenarios into config-driven experiments.
10. Delete legacy code once the corresponding new path has reproduced the
    same behavior.
11. Replace top-level `run_*.py` files with thin scripts.

Each migration step should preserve:

```bash
python tests/golden_capture.py --mode compare
```

Behavior changes should be reviewed as research changes, not hidden inside
structural refactors.

## File-Level Contract

Every non-trivial module should have an explicit contract:

- What are the inputs?
- What are the outputs?
- Is the module deterministic?
- Does it have side effects?
- What unit or integration test proves it works?

This rule is more important than the directory structure. The directory layout
only helps enforce the contract.
