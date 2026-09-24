# RouteWise Simulator

This repository contains the trace-driven simulator and experiment harness used
to evaluate RouteWise routing policies.

## Current Architecture

The simulator code is organized around one engine and a flat policy interface:

- `rwsim/engine/`: request loop, capacity accounting, in-flight hedge ticks
- `rwsim/world/`: providers, quota/concurrency state, latency distributions
- `rwsim/data/`: trace workload loaders
- `rwsim/policies/`: flat policy presets and implementations
- `rwsim/metrics/`: `Run` / `PerRequestRecord` result schema and aggregations
- `experiments/`: paper configs, suites, and offline-stage workflows

The old `rwsim/strategies/` layer and stage directories under
`rwsim/policies/` have been removed. Policy presets are:

- `greedy_cost`
- `greedy_latency`
- `random`
- `or_sort_cost`
- `or_sort_latency`
- `ablation_lp_only`
- `ablation_lp_hedging`
- `routewise`

## Quick Start

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
```

Prepare trace workload data when needed:

```bash
python3 scripts/prepare_workload.py --days 30
python -m experiments.simulation.dataset_cache build --dataset burstgpt
```

Run the implemented simulator paper sections. See
`experiments/simulation/README.md` for the target sub-experiment tree:

```bash
routewise simulator list
routewise simulator cost-layer
```

## Python Entrypoints

```python
from rwsim import POLICIES, run_policy
from rwsim.metrics import PerRequestRecord, Run
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

## Reproducibility Notes

The main artifact entrypoints are this README, the simulator CLI, and
`experiments/simulation/README.md`. Generated artifacts should be written under
`outputs/`.
