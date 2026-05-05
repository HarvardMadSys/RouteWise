# RouteWise Simulator

This repository contains the trace-driven simulator and experiment harness used
to evaluate RouteWise routing policies.

## Current Architecture

The simulator code is organized around one engine and a flat policy interface:

- `rwsim/engine/`: request loop, capacity accounting, in-flight hedge ticks
- `rwsim/world/`: providers, quota/concurrency state, latency distributions
- `rwsim/data/`: trace workload loaders
- `rwsim/policies/`: flat policy presets and implementations
- `rwsim/metrics/`: `SimulationRun` result container and aggregations
- `experiments/`: paper configs, suites, and offline-stage workflows

The old `rwsim/strategies/` layer and stage directories under
`rwsim/policies/` have been removed. Policy presets are:

- `greedy_cost`
- `greedy_latency`
- `random`
- `ablation_lp_only`
- `ablation_lp_hedging`
- `routewise`

## Quick Start

From the repository root:

```bash
source .venv/bin/activate
python -m pip install -e .
```

Prepare trace workload data when needed:

```bash
python3 scripts/prepare_workload.py --days 30
python -m experiments.simulation.dataset_cache build --dataset burstgpt
```

Run one config-driven scenario (paper S0-S3 YAMLs land in
`experiments/simulation/configs/`; until then, use the eval_grid suite):

```bash
routewise run simulation --scenario <S0|S1|S2|S3> --policy routewise --seed 42
```

Run registered suites:

```bash
routewise list --suites
routewise suite simulator_grid
```

## Python Entrypoints

```python
from rwsim import POLICIES, run_policy
from rwsim.metrics import SimulationRun
from rwsim.policies import build_policy
from rwsim.world import Provider, ScenarioConfig
```

## Verification

Fast structural and unit checks:

```bash
pytest -q -m "not slow"
```

Golden comparison remains available for full regression runs:

```bash
python tests/golden_capture.py --mode compare
```

## Documentation

- `docs/RWSIM_REFACTOR_PLAN.md`
- `docs/EXPERIMENT_LAYOUT.md`
- `docs/REPRODUCIBILITY.md`
