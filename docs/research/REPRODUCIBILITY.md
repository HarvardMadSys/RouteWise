# Reproducibility

This is the operational entrypoint for rerunning RouteWise experiments.
Architecture and algorithm contracts live in
[`ARCHITECTURE.md`](./ARCHITECTURE.md).

## Environment

From the repository root:

```bash
uv sync --frozen
```

The `llm-routewise` `0.2.0` release wheel is the dependency-free API-provider
library.
Simulator, live-evaluation, offline, plotting, and operational workflows are
repository-only and use the development dependency group installed by
`uv sync`.

## Data

The simulator is trace-driven and does not ship workload traces. Prepare the
local trace and dataset cache before running paper-facing simulator sections:

```bash
python scripts/prepare_workload.py --days 30
python -m experiments.simulation.dataset_cache build --dataset burstgpt
```

Live real-evaluation sends provider requests and requires credentials:

```bash
cp .env.example .env
```

Fill in only the providers used by the run. The pure simulator path does not
require API keys.

`experiments/real_evaluation/data/pilot_or_minimax_m3_subscription_or6_true24h.json`
is the paper's true-24h inventory moved to `minimax/minimax-m3`: on 2026-09-14
OpenRouter listed only seven providers for M2.5 (six of the paper's eight pins had
dropped the model) but thirteen for M3, at the same $0.30/$1.20 price point for
most. Its six-provider API pool (Together, GMICloud, Minimax, AtlasCloud, Novita,
StreamLake) came from a ten-minute uniform probe survey of all thirteen. Excluded:
CoreWeave (fastest and cheapest at once, so every cost-aware policy would collapse
onto it), Parasail and DeepInfra (two thirds of probes rate-limited on one key),
Venice (30-40 s stalls), Mara (broken), and SambaNova and ModelRun (2-2.5x the
standard price but no faster, which only inflates `c_max` and flattens the alpha
sweep). The quota emulation pins Minimax on M3 at the same price. The
`single_<Provider>` baseline (for example `single_OR_Together`) pins every request
to one of these metered providers with no hedge and no fallback, as the
"buy one on-demand provider" comparison point.

The live real-evaluation replay defaults to the day-0 24-hour trace and its
idle-compressed variants under `data/real_eval/`. Regenerate those inputs with:

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

## Discover Experiments

The paper-facing simulator sections are the four directly runnable modules
under `experiments/simulation/` — `cost_layer`, `latency_layer`, `hedging`,
and `end_to_end`; each lists its scenarios and policies via `--help`.
Config-driven experiment packages are registered in
`experiments.available_experiments()`.

## Run One Simulator Section

Each section runner exposes `--help` for scenario, policy, seed, and output
options:

```bash
uv run python -m experiments.simulation.cost_layer --help
uv run python -m experiments.simulation.cost_layer
```

The simulator is organized one Python file per paper section. See
`experiments/simulation/README.md` for the section tree. Generated artifacts
should go under `outputs/`.

## Regression Checks

Golden comparison for behavior-sensitive simulator outputs (a same-machine
regression check: the digests are bit-exact and therefore platform-specific
— re-capture them when moving to a new machine, and do not treat cross-
platform digest mismatches as behavior changes):

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
