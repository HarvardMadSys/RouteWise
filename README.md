# RouteWise

RouteWise core routing algorithms plus the trace-driven simulator and
experiment harnesses used to evaluate multi-provider LLM routing policies.
The lightweight public API is available from `routewise.core`; simulator and
experiment harnesses live under `rwsim` and `experiments`.

<!-- TODO(authors): fill in before public release.
Paper: "<title>", <venue> <year>.
Replace the BibTeX block below with the real entry. Do not ship a
placeholder citation.

```bibtex
@inproceedings{routewise,
  title     = {<paper title>},
  author    = {<authors>},
  booktitle = {<venue>},
  year      = {<year>}
}
```
-->

## Requirements

- Python >= 3.10
- Optional extras in `pyproject.toml` depending on the workflow

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The base install is intentionally lightweight and supports:

```python
from routewise.core import solve_budget_lp, effective_cost
```

Install extras for heavier workflows:

```bash
python -m pip install -e ".[sim]"        # simulator
python -m pip install -e ".[real-eval]"  # live real-evaluation harness
python -m pip install -e ".[offline]"    # offline optimization studies
python -m pip install -e ".[plots]"      # plotting scripts
python -m pip install -e ".[scripts]"    # operational scripts
```

For full local development, install all extras or use `uv sync`. The package
distribution name is now `routewise`; existing editable environments created
under the old `routewise-simulator` name should be reinstalled.

## Data

The simulator is trace-driven and does not ship the workload traces. The
harness expects public LLM-serving traces (BurstGPT, ShareGPT). Generate the
local trace and dataset cache before running experiments:

```bash
python3 scripts/prepare_workload.py --days 30
python -m experiments.simulation.dataset_cache build --dataset burstgpt
```

The real-evaluation harness (the path that issues live provider requests)
additionally needs credentials. Copy the template and fill in only the
providers you use:

```bash
cp .env.example .env
```

The pure simulator path does not require any API keys.

## Running experiments

Install the relevant extra before running a heavier workflow. Simulator section
commands require:

```bash
python -m pip install -e ".[sim]"
```

List the available paper sections and run one:

```bash
routewise simulator list
routewise simulator cost-layer
```

See `experiments/simulation/README.md` for the full sub-experiment tree.

## Python API

Lightweight core API:

```python
from routewise.core import (
    BackupCandidate,
    BudgetLPCandidate,
    combined_success_probability,
    effective_cost,
    hedge_checkpoints_for_slo,
    select_probability_backup,
    solve_budget_lp,
)
```

Simulator API:

```python
from rwsim import POLICIES, run_policy
from rwsim.metrics import PerRequestRecord, Run
from rwsim.policies import build_policy
from rwsim.world import Provider, ScenarioConfig
```

Available policy presets: `greedy_cost`, `greedy_latency`, `random`,
`ablation_lp_only`, `ablation_lp_hedging`, `routewise`.
The simulator does not include OpenRouter native `sort=price` or
`sort=latency` baselines; those remain part of the live real-evaluation
harness only.

## Repository layout

- `rwsim/engine/`: request loop, capacity accounting, in-flight hedge ticks
- `rwsim/world/`: providers, quota and concurrency state, latency distributions
- `rwsim/data/`: trace workload loaders
- `rwsim/policies/`: policy presets and implementations
- `rwsim/metrics/`: `Run` / `PerRequestRecord` result schema and aggregations
- `experiments/`: paper configs, suites, and offline-stage workflows
- `routewise_cli/`: command-line entry point
- `scripts/`: data preparation and profiling utilities

## Testing

Fast structural and unit checks:

```bash
pytest -q -m "not slow"
```

Golden comparison for full regression runs:

```bash
python tests/golden_capture.py --mode compare
```

## Documentation

- `docs/REPRODUCIBILITY.md`: end-to-end steps to reproduce paper results
- `docs/EXPERIMENT_LAYOUT.md`: how experiments, suites, and policies fit together

## License

MIT. See [LICENSE](LICENSE).
