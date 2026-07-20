# RouteWise

RouteWise is a dependency-free Python library for cost-aware, latency-optimized
routing across multiple LLM API providers. You supply provider prices and
dispatch the returned attempt with your own HTTP or SDK client; RouteWise
learns from the outcomes you report.

The `0.1.0` distribution is an API-provider-only preview. The repository
also contains the simulator and experiment harnesses used by the paper, but
those research packages are deliberately not included in the wheel.

> **Package name:** The PyPI project `routewise` is an unaffiliated,
> incompatible, separate project. For HarvardMadSys RouteWise, install the
> `llm-routewise` distribution and import the `llm_routewise` package.

## Requirements

- Python >= 3.10
- No runtime dependencies for the `llm-routewise` wheel

## Installation

Install the PyPI distribution:

```bash
python -m pip install llm-routewise==0.1.0
```

For library development or paper-artifact workflows, use a source checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Use `uv sync` instead when you also need the full research environment. The
separate PyPI project installed by `pip install routewise` is not compatible
with this project.

The base install exposes the public facade from `llm_routewise`. The canonical
short alias is `rw`:

```python
import llm_routewise as rw

router = rw.Router(
    providers=[
        rw.Provider("fast", price_in=3.0, price_out=15.0),
        rw.Provider("cheap", price_in=0.15, price_out=0.60),
    ],
    alpha=0.25,
)

decision = router.route(input_tokens=800)
response = call_your_provider(decision.provider)
decision.completed(
    ttft_ms=response.ttft_ms,
    output_tokens=response.output_tokens,
)
```

The `0.1.0` `Router` API computes routing decisions but performs no network I/O
and does not read provider API keys. Your application owns provider credentials
and dispatch. For a stateless one-off decision, use `route_once`:

```python
import llm_routewise as rw

result = rw.route_once(
    [
        rw.Candidate("fast", cost_usd=0.008, latency_ms=350),
        rw.Candidate("cheap", cost_usd=0.002, latency_ms=900),
    ],
    alpha=0.25,
)
```

For full repository development, including the research harnesses, use
`uv sync`. The distribution name is `llm-routewise`; editable environments
created from older source checkouts under the `routewise` or
`routewise-simulator` distribution names should be reinstalled.

## Source checkout: data and live evaluation

The following workflows require a source checkout and the development
environment; they are not available from the PyPI wheel. The simulator is
trace-driven and does not ship the workload traces. The harness expects public
LLM-serving traces (BurstGPT, ShareGPT). Generate the local trace and dataset
cache before running experiments:

```bash
python3 scripts/prepare_workload.py --days 30
python -m experiments.simulation.dataset_cache build --dataset burstgpt
```

After the package rename, simulator trace caches pickled under the old
`routewise.*` or `rwsim.*` layouts are rebuilt automatically on first use. If
you later check out a pre-rename commit, delete the generated
`data/*.simcache.pkl` files before running simulator jobs from that older code.

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

The commands below are repository-development workflows and are not part of
the `0.1.0` wheel. Install the development environment with `uv sync` first.

List the available paper sections and run one:

```bash
uv run python -m routewise_cli.main simulator list
uv run python -m routewise_cli.main simulator cost-layer
```

See `experiments/simulation/README.md` for the full sub-experiment tree.

## Python API

Public API-provider facade, using the canonical alias:

```python
import llm_routewise as rw

router = rw.Router([rw.Provider("api", price_in=1.0, price_out=2.0)])
decision = router.route(input_tokens=10)
```

Advanced users may import the pure mathematical primitives from
`llm_routewise.core`.

Research APIs (source checkout only; unavailable in the PyPI wheel):

```python
from llm_routewise.sim import POLICIES, run_policy
from llm_routewise.metrics import PerRequestRecord, Run
from llm_routewise.sim.policies import build_policy
from llm_routewise.sim.world import Provider, ScenarioConfig
```

Available policy presets: `greedy_cost`, `greedy_latency`, `random`,
`ablation_lp_only`, `ablation_lp_hedging`, `routewise`.
The simulator does not include OpenRouter native `sort=price` or
`sort=latency` baselines; those remain part of the live real-evaluation
harness only.

## Source-tree layout

This is the repository layout, not the contents of the published wheel.

- `llm_routewise/`: the public API-provider facade (stdlib-only)
- `llm_routewise/core/`: advanced RouteWise mathematical primitives
- `llm_routewise/capacity.py`, `llm_routewise/schemas.py`, `llm_routewise/const.py`: contracts shared by both worlds
- `llm_routewise/metrics/`: `Run` / `PerRequestRecord` result schema and aggregations
- `llm_routewise/sim/engine/`: request loop, capacity accounting, in-flight hedge ticks
- `llm_routewise/sim/world/`: providers, latency distributions, drift schedules
- `llm_routewise/sim/data/`: trace workload loaders
- `llm_routewise/sim/policies/`: policy presets and implementations
- `experiments/`: paper configs, section runners, ablations, offline-stage, and the live real-evaluation harness
- `routewise_cli/`: command-line entry point
- `scripts/`: data preparation and profiling utilities

## Testing a source checkout

Fast structural and unit checks:

```bash
pytest -q -m "not slow"
```

Golden comparison for full regression runs:

```bash
python tests/golden_capture.py --mode compare
```

## Documentation

- [`docs/API.md`](docs/API.md): API-provider library contract (English)
- [`docs/API.zh-CN.md`](docs/API.zh-CN.md): API-provider library contract (Chinese)
- `docs/CORE_API.md`: lightweight `llm_routewise.core` library API and integration guide
- `docs/ARCHITECTURE.md`: simulator architecture and module boundaries
- `docs/ALGORITHMS.md`: algorithm contracts and shared routing semantics
- `docs/REPRODUCIBILITY.md`: end-to-end steps to reproduce paper results
- `docs/RELEASING.md`: trusted PyPI release setup and release procedure
- `CHANGELOG.md`: published-package changes and migration notes

## License

MIT. See [LICENSE](LICENSE).
