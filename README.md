# RouteWise

RouteWise is a dependency-free Python library for cost-aware, latency-optimized
routing across multiple LLM API providers. You supply provider prices and
dispatch the returned attempt with your own HTTP or SDK client; RouteWise
learns from the outcomes you report.

The planned package `0.3.0` is an API-provider-only preview. The repository
also contains the simulator and experiment harnesses used by the paper, but
those research packages are deliberately not included in the wheel.

> **PyPI namespace notice:** The `routewise` releases currently present on
> PyPI (`0.1.0`--`0.2.0`) were published by an unaffiliated project. They are
> not HarvardMadSys RouteWise releases and are not an earlier version of this
> library. In particular, that package sends credentials to a hosted service
> that HarvardMadSys does not operate. Do not install it for this project or
> provide it with provider API keys. The first official PyPI release is blocked
> until ownership of the distribution name is resolved.

## Requirements

- Python >= 3.10
- No runtime dependencies for the published library

## Installation

Until the PyPI namespace is resolved, install the local routing library from a
trusted checkout of this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Use `uv sync` instead when you also need the full research environment. Do not
use `pip install routewise` until an official release is linked from this
repository.

The base install exposes the public facade directly from `routewise`:

```python
from routewise import Provider, Router

router = Router(
    providers=[
        Provider("fast", price_in=3.0, price_out=15.0),
        Provider("cheap", price_in=0.15, price_out=0.60),
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

`Router` makes the decision but performs no network I/O. For a stateless
one-off decision, use `route_once`:

```python
from routewise import Candidate, route_once

result = route_once(
    [
        Candidate("fast", cost_usd=0.008, latency_ms=350),
        Candidate("cheap", cost_usd=0.002, latency_ms=900),
    ],
    alpha=0.25,
)
```

For full repository development, including the research harnesses, use
`uv sync`. The distribution name is `routewise`; existing editable
environments created under the old `routewise-simulator` name should be
reinstalled.

### PyPI namespace status

There is no supported migration path from PyPI `routewise` `0.1.x`--`0.2.0`:
those releases belong to a different project. HarvardMadSys RouteWise performs
no provider network I/O and never receives your provider credentials. If you
installed the unaffiliated package and supplied provider keys, rotate those
keys before continuing.

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

The commands below are repository-development workflows and are not part of
the `0.3.0` wheel. Install the development environment with `uv sync` first.

List the available paper sections and run one:

```bash
uv run python -m routewise_cli.main simulator list
uv run python -m routewise_cli.main simulator cost-layer
```

See `experiments/simulation/README.md` for the full sub-experiment tree.

## Python API

Public API-provider facade:

```python
from routewise import (
    Attempt,
    Candidate,
    Decision,
    NoProviderError,
    OutcomeError,
    Provider,
    RouteOnceResult,
    Router,
    StatsSnapshot,
    Tuning,
    ValidationError,
    route_once,
)
```

Advanced users may import the pure mathematical primitives from
`routewise.core`.

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

- `routewise/`: the public API-provider facade (stdlib-only)
- `routewise/core/`: advanced RouteWise mathematical primitives
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

- `docs/API_PROVIDER_INTERFACE.md`: API-provider library contract (English)
- `docs/API_PROVIDER_INTERFACE.zh-CN.md`: API-provider library contract (Chinese)
- `docs/CORE_API.md`: lightweight `routewise.core` library API and integration guide
- `docs/ARCHITECTURE.md`: simulator architecture and module boundaries
- `docs/ALGORITHMS.md`: algorithm contracts and shared routing semantics
- `docs/REPRODUCIBILITY.md`: end-to-end steps to reproduce paper results
- `docs/RELEASING.md`: trusted PyPI release setup and release procedure
- `CHANGELOG.md`: published-package changes and migration notes

## License

MIT. See [LICENSE](LICENSE).
