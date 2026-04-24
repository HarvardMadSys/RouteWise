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
routewise list --experiment synthetic_latency
routewise list --experiment tiered_capacity
routewise list --suites
```

## Validate Configs

```bash
routewise validate synthetic_latency --scenario s1_dominant
routewise validate tiered_capacity --scenario s6_slow_q_trap
```

## Run One Scenario

```bash
routewise run synthetic_latency --scenario s1_dominant --strategy v2_only --seed 42
routewise run tiered_capacity --scenario s6_slow_q_trap --strategy joint_hedge --seed 42
```

These commands use `experiments/<name>/experiment.py` and reusable simulator
code under `rwsim/`.

## Run Full Suites

Full paper sweeps live under `experiments/*/suites/`:

```bash
routewise suite synthetic
routewise suite sanity
routewise suite joint
routewise suite joint_mm25_baselines
routewise suite stress
```

Suite-specific arguments are passed after `--`:

```bash
routewise suite joint_lp_budget_eval -- --scenario s6_slow_q_trap
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
