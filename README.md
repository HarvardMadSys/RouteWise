# RouteWise

RouteWise core routing algorithms plus the trace-driven simulator and
experiment harnesses used to evaluate multi-provider LLM routing policies.
The lightweight public API is available from `routewise.core`; the simulator
lives under `routewise.sim` and the experiment harnesses (including the live
real-evaluation runner) under `experiments`.

This repository is organized so the pure routing algorithms can be reused
without pulling in simulator, plotting, or live-provider dependencies. Paper
metadata and citation details will be added after the public manuscript entry
is finalized.

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

After the package rename, simulator trace caches pickled under the old
`rwsim.*` layout are rebuilt automatically on first use. If you later check out
a pre-rename commit, delete the generated `data/*.simcache.pkl` files before
running simulator jobs from that older code.

The real-evaluation replay scripts default to the day0 24h trace and its
idle-compressed variants under `data/real_eval/`. Regenerate those with:

```bash
python3 scripts/prepare_workload.py --start-day 0 --days 1 \
    --output data/real_eval/burstgpt_day0_24h.jsonl
python3 scripts/idle_compress_trace.py \
    --source data/real_eval/burstgpt_day0_24h.jsonl \
    --output data/real_eval/burstgpt_day0_24h_cap10s.jsonl
python3 scripts/idle_compress_trace.py \
    --source data/real_eval/burstgpt_day0_24h.jsonl \
    --output data/real_eval/burstgpt_day0_24h_cap10s_mingap1s.jsonl \
    --min-gap-sec 1
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
    HedgeDispatch,
    RoutingDecision,
    combined_success_probability,
    effective_cost,
    hedge_checkpoints_for_slo,
    select_probability_backup,
    solve_budget_lp,
)
```

Simulator API:

```python
from routewise.sim import POLICIES, run_policy
from routewise.metrics import PerRequestRecord, Run
from routewise.sim.policies import build_policy
from routewise.sim.world import Provider, ScenarioConfig
```

Available policy presets: `greedy_cost`, `greedy_latency`, `random`,
`ablation_lp_only`, `ablation_lp_hedging`, `routewise`.
The simulator does not include OpenRouter native `sort=price` or
`sort=latency` baselines; those remain part of the live real-evaluation
harness only.

## Repository layout

- `routewise/core/`: the RouteWise algorithm (stdlib-only; shared by sim and live)
- `routewise/capacity.py`, `routewise/schemas.py`, `routewise/const.py`: contracts shared by both worlds
- `routewise/metrics/`: `Run` / `PerRequestRecord` result schema and aggregations
- `routewise/sim/engine/`: request loop, capacity accounting, in-flight hedge ticks
- `routewise/sim/world/`: providers, latency distributions, drift schedules
- `routewise/sim/data/`: trace workload loaders
- `routewise/sim/policies/`: policy presets and implementations
- `experiments/`: paper configs, section runners, ablations, offline-stage, and the live real-evaluation harness
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

- `docs/API_PROVIDER_INTERFACE.md`: proposed API-provider-only library interface
- `docs/CORE_API.md`: lightweight `routewise.core` library API and integration guide
- `docs/ARCHITECTURE.md`: simulator architecture and module boundaries
- `docs/ALGORITHMS.md`: algorithm contracts and shared routing semantics
- `docs/REPRODUCIBILITY.md`: end-to-end steps to reproduce paper results

## License

MIT. See [LICENSE](LICENSE).
