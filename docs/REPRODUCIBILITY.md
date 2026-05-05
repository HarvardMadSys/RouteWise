# Reproducibility

This document is the operational entrypoint for rerunning RouteWise experiments.
Architecture and algorithm contracts live in `docs/ARCHITECTURE.md` and
`docs/ALGORITHMS.md`.

## Environment

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

If the package is not installed, replace `routewise` below with
`python -m routewise_cli.main`.

## Discover Experiments

```bash
routewise list
routewise list --experiment simulation
routewise list --suites
```

## Run One Paper Section

The simulator is organised one Python file per paper section. See
`experiments/simulation/README.md` for the sub-experiment tree
(§1.1.1 / §1.1.2 / §2.1.1.1 / §3.1 / etc.):

```bash
python -m experiments.simulation.cost_layer
python -m experiments.simulation.latency_layer
python -m experiments.simulation.hedging
python -m experiments.simulation.end_to_end
```

Each runner exposes `--help` for sub-experiment selection (latency family,
overlap, provider pool, hedging on/off).

## Run Full Suites

Full paper sweeps live under `experiments/*/suites/`:

```bash
routewise suite simulator_grid
```

Generated artifacts should go under `outputs/`.

## Regression Checks

The behavior-preserving refactor gate is:

```bash
python tests/golden_capture.py --mode compare
```

Fast structural and policy checks:

```bash
python -m unittest tests/test_architecture_scaffold.py -v
pytest tests/unit/policies -q
```
