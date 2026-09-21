# Reproducibility

This is the operational entrypoint for rerunning RouteWise experiments.
Architecture and algorithm contracts live in
[`ARCHITECTURE.md`](./ARCHITECTURE.md).

For the evaluated paper, use the [figure evidence table](FIGURE_MAP.md)
and [clarifications and errata](AE_NOTES.md) alongside these operational
notes. In particular, the Figure 8 archive evaluates an earlier concurrency
variant, and the current simulator retains premature output-length feedback.

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

The artifact includes the de-identified PROD trace (`data/freeinference.jsonl`),
the recorded real-provider runs (`data/real_eval_records/`), the Figure 2/3
measurement exports, and a synthetic smoke fixture. These support the
committed-data commands in the main README without additional data downloads.
Only the public BurstGPT/ShareGPT sources for the composed 30-day workload
must be downloaded. Prepare that workload and its cache with:

```bash
uv run python scripts/prepare_workload.py --days 30
uv run python -m experiments.simulation.dataset_cache build --dataset burstgpt
```

Live real-evaluation sends provider requests and requires credentials:

```bash
cp .env.example .env
```

Fill in only the providers used by the run. The pure simulator path does not
require API keys.

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
