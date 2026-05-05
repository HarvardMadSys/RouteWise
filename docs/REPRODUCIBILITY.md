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

## Validate Configs

```bash
routewise validate simulation --scenario <S0|S1|S2|S3>
```

## Run One Scenario

```bash
routewise run simulation --scenario <S0|S1|S2|S3> --policy routewise --seed 42
```

These commands use `experiments/<name>/experiment.py` and reusable simulator
code under `rwsim/`. Paper-aligned scenario YAMLs (`S0` cost-only / `S1`
latency-only / `S2` cost-latency / `S3` joint tier) live under
`experiments/simulation/configs/`; see `docs/EXPERIMENT_LAYOUT.md` §3.1.

## Run Full Suites

Full paper sweeps live under `experiments/*/suites/`:

```bash
routewise suite simulator_grid
```

Suite-specific arguments are passed after `--`:

```bash
routewise suite simulator_grid -- --scenario S0
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
