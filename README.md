<h1 align="center">RouteWise — EuroSys '27 Artifact</h1>

<p align="center">
  <strong>Latency–Cost Optimization for Multi-Provider LLM Routing</strong>
  <br>
  <sub>Paper #96 · EuroSys 2027 · Muxin Tian, Haoran Ni, Yiyan Zhai, Yangsun Park, Juncheng Yang</sub>
</p>

<p align="center">
  Developed by the
  <a href="https://juncheng.seas.harvard.edu/" title="Harvard Measurements and Design of Computer Systems Lab">Harvard MadSys Lab</a>
  at <a href="https://seas.harvard.edu/">Harvard SEAS</a>.
</p>

This branch (`eurosys27-ae`) is the evaluated research artifact for the paper.
It is calibrated against one pinned arXiv version of the paper (the exact
version and its PDF SHA-256 will be recorded here once frozen). The legacy
branch `eurosys2027` is not part of the submitted artifact. The `main` branch
hosts the separately released `llm-routewise` library, which evolves
independently — do **not** `pip install llm-routewise` to evaluate this
artifact; everything below runs from this checkout.

## 1. Overview

| Path | Role |
|---|---|
| `llm_routewise/` | Routing core, LP mixture solver, simulator engine, metrics |
| `experiments/simulation/` | Section-driven simulator experiments (paper §3.2–3.5) |
| `experiments/offline_stage/` | Offline/stage configuration and loaders |
| `experiments/real_evaluation/` | Live-provider runner (optional; needs keys, costs money) |
| `plots/` | Figure-generation scripts |
| `data/` | Committed inputs: motivation CSVs, smoke fixture |
| `scripts/` | Workload preparation, kick-the-tires, run helpers |
| `docs/research/REPRODUCIBILITY.md` | Extended operational notes |

## 2. Setup

Requirements: Linux x86-64 or macOS, `git`, and
[uv](https://docs.astral.sh/uv/getting-started/installation/). uv installs
the pinned Python interpreter (`.python-version`) and the exact locked
dependency set; there are no system-level dependencies, no GPU, and no
commercial solver.

```bash
git clone -b eurosys27-ae https://github.com/HarvardMadSys/RouteWise.git
cd RouteWise
uv sync --frozen
```

### Docker alternative (uniform Ubuntu environment)

To evaluate inside a uniform Ubuntu 24.04 container instead of installing
uv on the host:

```bash
docker build -t routewise-ae .
docker run --rm routewise-ae            # runs the kick-the-tires check
docker run --rm -it routewise-ae bash   # shell for every other command below
```

Every command in the following sections works the same inside the
container; add a volume mount (`-v "$PWD/outputs:/artifact/outputs"`) to
keep generated figures on the host.

## 3. Kick the tires (~2 minutes)

```bash
bash scripts/kick_the_tires.sh
```

This replays the committed 120-request synthetic fixture through the
cost-layer simulator section and runs the fast unit tests. It needs **no API
keys and no network access** and ends with `kick-the-tires: PASS`.

## 4. Reproducing the paper's results

Each subsection gives the command, the expected artifacts, and the rough
runtime. Outputs land under `outputs/`; compare the produced figures and
printed statistics against the paper. Figure numbering follows the pinned
arXiv version.

### 4.1 Provider TTFT drift over wall-clock time (~1 minute)

Both source CSVs are committed in `data/drift_source/`.

```bash
uv run python plots/motivation/drift_wall_clock.py \
    --source-dir data/drift_source --output-dir outputs/figures
```

Produces `drift_wall_clock_llama.{pdf,png}` and
`drift_wall_clock_gpt4o.{pdf,png}` in `outputs/figures/`, and prints each
panel's statistics (row count, global P99, max rolling P99) for comparison
with the paper.

### 4.2 Thirty-day simulator studies

The simulator is trace-driven. Prepare the workload once (downloads the
public BurstGPT v2.0 and ShareGPT V3 sources with SHA256-pinned URLs,
roughly a 1 GB download):

```bash
uv run python scripts/prepare_workload.py --days 30
```

Then run the paper-facing simulator sections (one per paper section; each
accepts `--help` for scenario, policy, seed, and output options, writes
`summary.{json,csv}` plus TTFT histograms under `outputs/simulation/<section>/`,
and takes `--jobs N` to parallelize):

| Command | Paper part | Wall time (64-core server, `--jobs 24`) |
|---|---|---|
| `uv run python -m routewise_cli.main simulator cost-layer --jobs 24` | §3.2, quota/concurrency cost tiers | ~17 min (120 cells) |
| `uv run python -m routewise_cli.main simulator latency-layer --jobs 24` | §3.3, latency-band overlap | ~2 min (21 cells) |
| `uv run python -m routewise_cli.main simulator hedging --jobs 24` | §3.4, hedging stress test | ~16 min (8 cells) |
| `uv run python -m routewise_cli.main simulator end-to-end --jobs 24` | §3.5, joint cost+latency routing | ~24 min (39 cells) |

On a laptop, budget roughly 20-40x those times or reduce `--jobs`; every
section also accepts `--max-requests` for a truncated pass.

Rebuild the corresponding figures from the section outputs:

```bash
# End-to-end frontier, SLO violations, TTFT distribution, provider mix:
uv run python -m plots.end_to_end.plot_simulation_frontier \
    --summary-csv outputs/simulation/end_to_end/summary.csv \
    --histograms-json outputs/simulation/end_to_end/ttft_histograms.json \
    --frontier-out outputs/figures/e2e_frontier.pdf \
    --slo-out outputs/figures/e2e_slo.pdf \
    --cdf-out outputs/figures/e2e_ttft_cdf.pdf \
    --provider-mix-out outputs/figures/e2e_provider_mix.pdf \
    --table-out outputs/figures/e2e_table.json

# Output-length misprediction ablation (runs + plot, one command; ~1 h):
uv run python scripts/run_output_length_prediction_ablation.py

# Quota / concurrency effective-cost ablation (runs + plots, one command;
# ~10 min with --jobs 8):
uv run python scripts/run_effective_cost_ablation.py --jobs 8
```

Simulation is deterministic for a given seed within one environment;
across platforms the summary statistics agree to floating-point precision
(the repository's bit-level golden digests are a same-machine developer
tool, not an evaluation check).

### 4.3 24-hour real-provider evaluation (recorded data)

The paper's live-provider numbers cannot be regenerated by calling the
providers again: provider load, pricing, quotas, and rate limits have
changed since the measurement. The paper's numbers derive from the 24-hour
request records captured during the original runs; those records (exported
with an explicit field allowlist) and the analysis scripts that rebuild the
corresponding figures and tables from them will land under
`data/real_eval_records/`.

Optionally, `experiments/real_evaluation/` contains the full live runner to
redo such an experiment with your own provider keys (`cp .env.example .env`).
It **spends real money**, and its results are a new measurement — comparable
in trend, not in exact numbers.

## 5. License and data provenance

The code is MIT-licensed (`LICENSE`). The BurstGPT and ShareGPT source
traces are downloaded from their original public hosts at pinned URLs with
SHA-256 verification and are not redistributed here. The smoke fixture is
synthetic, generated deterministically by
`data/fixtures/generate_smoke_fixture.py`, and contains no text payloads.
