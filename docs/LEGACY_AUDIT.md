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
| `legacy/experiment/scripts/simulate/synthetic/scenarios.py` | `experiments/synthetic_latency/configs/` | Wrapper |
| `legacy/experiment/scripts/simulate/synthetic/tiered/scenarios.py` | `experiments/tiered_capacity/configs/` | Wrapper |
| `legacy/experiment/predictors/**` | `rwsim/policies/value_estimators/` | Wrapper |
| `legacy/experiment/strategies/online/predictors/**` | `rwsim/policies/value_estimators/` | Wrapper |
| `legacy/experiment/strategies/{online_latency_router,v2_router,smart_hedging}.py` | `rwsim/policies/{latency_routers,hedgers}/` | Wrapper |

Deletion condition: top-level scripts and tests stop importing these paths,
then golden compare stays green.

### Useful And Not Fully Replaced Yet

These modules still contain real experiment or analysis logic and should not
be deleted until they get canonical homes:

| Legacy path | Intended canonical home | Notes |
| --- | --- | --- |
| `legacy/experiment/scripts/simulate/synthetic/_core/sanity_check.py` | `experiments/synthetic_latency/sanity.py` | Builds Step 1-Step 5 sanity scenarios. |
| `legacy/experiment/scripts/simulate/synthetic/plots.py` | `experiments/synthetic_latency/plots.py` | Plot helpers used by `run_synthetic.py`. |
| `legacy/experiment/scripts/simulate/synthetic/pareto.py` | `experiments/synthetic_latency/pareto.py` | Pareto sweep analysis. |
| `legacy/experiment/scripts/simulate/synthetic/phase_diagram*.py` | `experiments/synthetic_latency/phase_diagram*.py` | LP vs V2 regime maps. |
| `legacy/experiment/scripts/simulate/synthetic/tiered/plots.py` | `experiments/tiered_capacity/plots.py` | Tiered summary plots. |
| `legacy/experiment/scripts/simulate/synthetic/tiered/stress_scenarios.py` | `experiments/tiered_capacity/stress.py` | ST1-ST3 stress scenarios. |
| `legacy/experiment/scripts/simulate/synthetic/tiered/scenarios_mm25.py` | `experiments/tiered_capacity/minimax_m25.py` | MiniMax M2.5 calibrated scenarios. |
| `legacy/experiment/scripts/simulate/synthetic/tiered/scenarios_calibrated.py` | `experiments/tiered_capacity/calibrated.py` | Earlier calibrated tiered scenarios. |
| `legacy/experiment/scripts/simulate/synthetic/tiered/phase_diagram.py` | `experiments/tiered_capacity/phase_diagram.py` | Joint-vs-two-layer regime map. |
| `legacy/experiment/scripts/simulate/synthetic/tiered/lp_budget_eval.py` | `experiments/tiered_capacity/lp_budget_eval.py` or delete after reproduce | Large standalone LP budget evaluation. |

Migration condition: move the implementation to the intended home, leave a
temporary wrapper in legacy, update top-level scripts/tests to import the
canonical module, then rerun golden compare.

### Historical Offline Experiment Stack

These files are still useful only if we need to reproduce the old offline
counterfactual or phase 3/4 latency experiments:

| Legacy path | Decision needed |
| --- | --- |
| `legacy/experiment/data/{schema,loader}.py` | Migrate only if old CSV/data-loader workflow is still active. |
| `legacy/experiment/{config,cost,quota,window_quota,cache,simulator}.py` | Either map into `rwsim/engine`/`rwsim/schemas` or retire with offline stack. |
| `legacy/experiment/strategies/{all_api,greedy,stage1_*,stage2_*}.py` | Historical stage strategy stack; not represented by current pipeline config. |
| `legacy/experiment/strategies/online/{base,greedy,learning_augmented,primal_dual}.py` | Historical online routing stack; not part of current synthetic golden path. |
| `legacy/experiment/scripts/simulate/{offline_counterfactual,policies,bootstrap,plot_counterfactual}.py` | Offline counterfactual workflow. |
| `legacy/experiment/scripts/run_phase{3,4}_simulation.py` | Historical latency experiment runners. |
| `legacy/experiment/scripts/plot/latency/*.py` | Historical phase 3/4 plotters. |
| `legacy/experiment/latency_profiling.py` | Live/probe script; keep separate from simulator refactor unless still used. |

Deletion condition: either reproduce the relevant paper artifact from
`rwsim/` + `experiments/`, or explicitly declare the old experiment out of
scope and remove the corresponding runner/tests/docs together.

## External Callers Still Using Legacy

The root `run_*.py` scripts and `tests/golden_capture.py` still import
`legacy.experiment...` for several paths. That is the next cleanup target.

Until those imports are gone, `legacy/experiment/` cannot be deleted.

## Naming Notes

`mm25` means MiniMax M2.5. In canonical paths, prefer the clearer name
`minimax_m25`.
