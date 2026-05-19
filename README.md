# RouteWise Simulator

Trace-driven simulator and experiment harness for evaluating RouteWise
multi-provider LLM routing policies. It replays request traces against a
model of provider capacity, quota, and latency, then reports per-request
cost and latency outcomes for each routing policy.

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
- The dependencies pinned in `pyproject.toml` (resolved via `uv.lock`)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

This installs the `routewise` command-line entry point.

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

List the available paper sections and run one:

```bash
routewise simulator list
routewise simulator cost-layer
```

See `experiments/simulation/README.md` for the full sub-experiment tree.

## Python API

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
