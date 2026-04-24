# Legacy Audit

This document tracks what remains under `legacy/experiment/`.

`legacy/` is not a trash folder and does not mean "already replaced." It is
the quarantine area for code that still matters for reproducibility while the
canonical `rwsim/` and `experiments/` paths catch up.

The rule is:

```text
legacy/ is kill-on-reproduce.
```

That means each legacy module must either have a canonical replacement and a
green regression check, or an explicit reproduction target that lets us delete
it later.

## Current State

### Already Replaced By Canonical Modules

These files should stay thin wrappers only until old imports are no longer
needed:

| Legacy path | Canonical home | Status |
| --- | --- | --- |
| `legacy/experiment/scripts/simulate/synthetic/_core/{capacity,distributions,metrics,providers,scenarios,shadow_price,workload}.py` | `rwsim/world/` | Wrapper |
| `legacy/experiment/scripts/simulate/synthetic/_core/strategies/*.py` | `rwsim/strategies/` | Wrapper |
| `legacy/experiment/scripts/simulate/synthetic/runner.py` | `rwsim/strategies/latency_impl.py` | Wrapper |
| `legacy/experiment/scripts/simulate/synthetic/tiered/strategies.py` | `rwsim/strategies/tiered_impl.py` | Wrapper |
| `legacy/experiment/scripts/simulate/synthetic/{providers,workload}.py` | `rwsim/world/` | Wrapper |
| `legacy/experiment/scripts/simulate/synthetic/tiered/{providers,shadow_price}.py` | `rwsim/world/` | Wrapper |
| `legacy/experiment/scripts/simulate/synthetic/tiered/runner.py` | `rwsim/runner.py` / `run_joint.py` | Compatibility orchestration |
| `legacy/experiment/scripts/simulate/synthetic/scenarios.py` | `experiments/synthetic_latency/configs/` | Wrapper |
| `legacy/experiment/scripts/simulate/synthetic/_core/sanity_check.py` | `experiments/synthetic_latency/sanity.py` | Wrapper |
| `legacy/experiment/scripts/simulate/synthetic/plots.py` | `experiments/synthetic_latency/plots.py` | Wrapper |
| `legacy/experiment/scripts/simulate/synthetic/pareto.py` | `experiments/synthetic_latency/pareto.py` | Wrapper |
| `legacy/experiment/scripts/simulate/synthetic/phase_diagram*.py` | `experiments/synthetic_latency/phase_diagram*.py` | Wrapper |
| `legacy/experiment/scripts/simulate/synthetic/tiered/scenarios.py` | `experiments/tiered_capacity/configs/` | Wrapper |
| `legacy/experiment/scripts/simulate/synthetic/tiered/plots.py` | `experiments/tiered_capacity/plots.py` | Wrapper |
| `legacy/experiment/scripts/simulate/synthetic/tiered/stress_scenarios.py` | `experiments/tiered_capacity/stress.py` | Wrapper |
| `legacy/experiment/scripts/simulate/synthetic/tiered/scenarios_mm25.py` | `experiments/tiered_capacity/minimax_m25.py` | Wrapper |
| `legacy/experiment/scripts/simulate/synthetic/tiered/scenarios_calibrated.py` | `experiments/tiered_capacity/calibrated.py` | Wrapper |
| `legacy/experiment/scripts/simulate/synthetic/tiered/phase_diagram.py` | `experiments/tiered_capacity/phase_diagram.py` | Wrapper |
| `legacy/experiment/scripts/simulate/synthetic/tiered/lp_budget_eval.py` | `experiments/tiered_capacity/lp_budget_eval.py` | Wrapper |
| `legacy/experiment/scripts/simulate/{offline_counterfactual,policies,bootstrap,plot_counterfactual}.py` | `experiments/offline_counterfactual/` | Wrapper |
| `legacy/experiment/data/loader.py` | `rwsim/data/loader.py` | Wrapper |
| `legacy/experiment/predictors/**` | `rwsim/policies/value_estimators/` | Wrapper |
| `legacy/experiment/strategies/online/predictors/**` | `rwsim/policies/value_estimators/` | Wrapper |
| `legacy/experiment/strategies/{online_latency_router,v2_router,smart_hedging}.py` | `rwsim/policies/{latency_routers,hedgers}/` | Wrapper |

Deletion condition: top-level scripts and tests stop importing these paths,
then golden compare stays green.

### Current Experiment Logic Not Fully Replaced Yet

No current synthetic/tiered golden-path module is known to live only under
`legacy/experiment/`. Remaining legacy code is either a compatibility wrapper
for a canonical module or part of the paper-used offline/stage experiment stack
below.

### Paper-Used Offline/Stage Experiments Pending Migration

These files are not on the current synthetic/tiered golden path, but they are
paper-used or paper-era research artifacts. They should not be deleted just
because the new synthetic/tiered runners no longer import them.

The migration rule for this section is stricter than a normal compatibility
wrapper: each workflow needs a canonical replacement under `rwsim/` +
`experiments/`, plus a reproduction check for the relevant paper artifact.

| Legacy path | Canonical migration target |
| --- | --- |
| `config/experiment.yaml` | Move offline/stage experiment config into the owning `experiments/` package or split shared model/provider config from paper-run config. |
| `legacy/experiment/data/schema.py` | Fold offline `ProviderConfig` / `RoutingDecision` models into `rwsim/schemas.py` or a dedicated offline schema module. |
| `legacy/experiment/{config,cost,quota,window_quota,cache,simulator}.py` | Map shared execution, cost, quota, and cache pieces into `rwsim/engine`, `rwsim/world`, and `rwsim/data`. |
| `legacy/experiment/strategies/{all_api,greedy,stage1_*,stage2_*}.py` | Map stage strategies into `rwsim/policies` / `rwsim/strategies` pipeline configs. |
| `legacy/experiment/strategies/online/{base,greedy,learning_augmented,primal_dual}.py` | Map online routing stack into canonical policy stages or strategy aliases. |
| `legacy/experiment/scripts/run_phase{3,4}_simulation.py` | Move Phase 3/4 latency experiment runners into `experiments/` and keep paper-output compatibility. |
| `legacy/experiment/scripts/plot/{common.py,latency/*.py}` | Move Phase 3/4 plotters and paper style helpers into the same experiment package or a paper plotting module. |
| `legacy/experiment/latency_profiling.py` | Live/probe script; keep separate from simulator refactor unless still used. |

Deletion condition: either reproduce the relevant artifact from `rwsim/` +
`experiments/`, or explicitly replace/remove that paper artifact and update the
paper workflow docs together.

## External Callers Still Using Legacy

The main synthetic/tiered runners and `tests/golden_capture.py` now import
canonical `rwsim/` and `experiments/` paths.

There are no known root-level script callers of `legacy.experiment.*` now.
There are also no known canonical `rwsim/` or `experiments/` modules importing
`legacy.experiment.*`.

`tests/test_architecture_scaffold.py` intentionally imports one legacy
predictor wrapper to verify backward compatibility.

## Documentation Debt

The root design notes are paper-era context, not the current package map.
They now carry post-refactor path notes, but any migration work should treat
`docs/ARCHITECTURE.md`, `docs/ALGORITHMS.md`, and this audit as the source of
truth.

## Naming Notes

`mm25` means MiniMax M2.5. In canonical paths, prefer the clearer name
`minimax_m25`.
