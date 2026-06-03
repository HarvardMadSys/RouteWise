# Reproducibility

This is the operational entrypoint for rerunning RouteWise experiments.
Architecture and algorithm contracts live in `docs/ARCHITECTURE.md` and
`docs/ALGORITHMS.md`.

## Environment

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[sim,real-eval,offline,plots,scripts]"
```

The base `pip install -e .` install exposes only the lightweight
`routewise.core` API. Simulator, live-evaluation, offline, plotting, and
operational workflows require the extras above. The package distribution name
is `routewise`; editable environments created under the old
`routewise-simulator` name should be reinstalled.

If the package is not installed, replace `routewise` below with:

```bash
python -m routewise_cli.main
```

## Data

The simulator is trace-driven and does not ship workload traces. Prepare the
local trace and dataset cache before running paper-facing simulator sections:

```bash
python scripts/prepare_workload.py --days 30
python -m experiments.simulation.dataset_cache build --dataset burstgpt
```

Live real-evaluation issues provider requests and requires credentials:

```bash
cp .env.example .env
```

Fill in only the providers used by the run. The pure simulator path does not
require API keys.

## Discover Experiments

Config-driven experiments:

```bash
routewise list
```

Paper-facing simulator sections:

```bash
routewise simulator list
```

## Run One Simulator Section

Each section runner exposes `--help` for scenario, policy, seed, and output
options:

```bash
routewise simulator cost-layer -- --help
routewise simulator cost-layer
```

The simulator is organized one Python file per paper section. See
`experiments/simulation/README.md` for the section tree. Generated artifacts
should go under `outputs/`.

## Regression Checks

Golden comparison for behavior-sensitive simulator outputs:

```bash
python tests/golden_capture.py --mode compare
```

Fast local test suite:

```bash
pytest -q -m "not slow"
```

Live real-evaluation tests are not part of the default reproducibility gate;
they depend on external provider credentials, quota state, and network
conditions.
