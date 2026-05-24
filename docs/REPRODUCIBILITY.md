# Reproducibility

This document is the operational entrypoint for rerunning RouteWise experiments.
Architecture and algorithm contracts live in `docs/ARCHITECTURE.md` and
`docs/ALGORITHMS.md`.

## Environment

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[sim,real-eval,offline,plots,scripts]"
```

If the package is not installed, replace `routewise` below with
`python -m routewise_cli.main`.

The base `pip install -e .` install exposes only the lightweight
`routewise.core` API. Rerunning simulator, live-eval, offline, and plotting
workflows and operational scripts require the extras above. The distribution
name is `routewise`; environments installed under the old
`routewise-simulator` name should be reinstalled.

## Discover Experiments

```bash
routewise list
routewise simulator list
```

## Run One Paper Section

The simulator is organised one Python file per paper section. See
`experiments/simulation/README.md` for the sub-experiment tree
(§1.1.1 / §1.1.2 / §2.1.1.1 / §3.1 / etc.):

```bash
routewise simulator cost-layer
```

Each section runner exposes `--help` for sub-experiment selection. Legacy
full-sweep suites were removed; section commands are the paper-facing surface.

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
