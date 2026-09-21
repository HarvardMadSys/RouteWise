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

This repository is the research artifact for the paper: the routing core,
the trace-driven simulator, the experiment and figure pipelines, and the
instructions to run them.

For the evaluated paper's figure-to-code mapping and the limitations identified
during evaluation, start with the [figure evidence table](docs/research/FIGURE_MAP.md)
and [artifact clarifications and errata](docs/research/AE_NOTES.md). The original
Figure 8 reproduction remains the default; it evaluates the earlier constant-`L`
concurrency variant, which differs from the zero-cost rule in Equation (3).

## 1. Overview

| Path | Role |
|---|---|
| `llm_routewise/` | Routing core, LP mixture solver, simulator engine, metrics |
| `experiments/simulation/` | Trace-driven simulator experiment modules |
| `experiments/offline_stage/` | Offline/stage configuration and loaders |
| `experiments/real_evaluation/` | Live-provider runner (optional; needs keys, costs money) |
| `plots/` | Figure-generation scripts |
| `data/` | Recorded real-world requests, de-identified PROD trace, motivation measurements, smoke fixture |
| `scripts/` | Workload preparation, smoke test, run helpers |
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
docker run --rm routewise-ae            # runs the artifact smoke test
docker run --rm -it routewise-ae bash   # shell for every other command below
```

Every command in the following sections works the same inside the
container; add a volume mount (`-v "$PWD/outputs:/artifact/outputs"`) to
keep generated figures on the host.

If macOS rejects a NumPy compiled library's code signature, use this container
path. At the evaluated commit `ae3a998`, [CI run 34542875011](https://github.com/HarvardMadSys/RouteWise/actions/runs/34542875011)
passed the Ubuntu smoke test, committed-data figure pipelines (including the
Figure 8 replay), and a separate Docker smoke test. Those figure pipelines
ran directly on Ubuntu; only the smoke test ran inside Docker. See the
[recorded environments](docs/research/AE_NOTES.md#environments-and-validation)
for the scope of these checks.

## 3. Getting started (~2 minutes)

```bash
bash scripts/artifact_smoke_test.sh
```

This replays the committed 120-request synthetic fixture through the
cost-layer simulator section and runs the fast unit tests. It needs **no API
keys and no network access** and ends with `artifact smoke test: PASS`.

## 4. Reproducing the paper's results

The paper's evaluation has two arms, and this section mirrors them: the
real-world experiments against live providers (Figures 1, 6, 7) and the
trace-driven simulator experiments (Figure 8 and the 30-day results),
followed by the ablation study (Figure 9) and the background measurement
figures (Figures 2 and 3). Outputs land under `outputs/`; compare the produced
figures and printed statistics against the paper.

### Resource requirements

Around 10 GB of free disk (1 GB of downloads, a 5.6 GB composed workload,
caches and outputs) and roughly 2-4 GB of RAM per simulator worker — with
16 GB of RAM prefer `--jobs 4`; the measured times below used `--jobs 24`
on a 64-core, 500 GB server. No GPU.

### 4.1 Real-world experiments (Figures 1, 6, 7; ~1 minute)

The paper's real-world results come from a 24-hour BurstGPT replay against
live commercial providers. Reproduction analyzes the recorded measurements;
calling providers today would produce a new measurement under different
load, prices, quotas, and rate limits. The request-level records of all ten
policy runs are included in [data/real_eval_records/](data/real_eval_records/).
Run the analysis, plot the six panels, and check the numeric results with:

```bash
uv run python scripts/reproduce_real_world.py
```

Outputs go to `outputs/figures/real_world/`: `figure01_ttft.pdf` (the
cost vs. mean-TTFT frontier of all ten policies) and `figure01_slo.pdf` (the
SLO-violation bars), `figure06a_ttft_distribution.pdf`,
`figure06b_provider_mix.pdf`, `figure07a_provider_latency.pdf`, and
`figure07b_provider_pricing.pdf`, plus a recomputed `real_world_summary.json`
and a LaTeX table. The command fixes the billing window at 86,400 seconds
and the subscription charge at $1.5476333333333334 per non-OpenRouter run.

The check requires exactly 14,233 requests per policy and compares total
cost, mean and P99 TTFT, and SLO-violation rate against
`data/real_eval_records/reference_summary.json`, with relative/absolute
tolerance `1e-9` for floating-point aggregation. It exits unsuccessfully on
a mismatch. For example, RouteWise at α = 0 costs $2.183 with mean TTFT
0.924 s and 0.45% SLO violations; OR-price costs $2.060 with 2.360 s and
20.66%. Figure 7a redraws an **author-reconstructed provider-mean snapshot**
in `paper_minimax_provider_latency.json`; its original profiling logs are
unavailable, so this panel is not independently recomputed from the request
records. Figure 7b uses the recorded inventory prices. See the
[data notes](data/real_eval_records/README.md) for these provenance limits.

Mean/P99 TTFT use successful requests with valid TTFT; the SLO-violation
denominator includes all requests and counts failures as violations. From the
unrounded records, α = 0 improves over OR-auto by 26.48% in total cost,
58.15% in mean TTFT, and 98.01% in SLO-violation rate. See the
[metric definitions and calculation](docs/research/AE_NOTES.md#recorded-metrics-and-headline-percentages).

Optionally, `experiments/real_evaluation/` contains the full live runner to
redo such an experiment with your own provider keys (`cp .env.example .env`).
It **spends real money**, and its results are a new measurement — comparable
in trend, not in exact numbers.

### 4.2 Simulator experiments (Figure 8 and the 30-day results)

The simulator is trace-driven. Fix the seed, workload, scenario, and
predictor when comparing runs; code revisions can also change the routing
decisions. The PROD command below pins the historical simulator revision
and checks the results against the archived reference.

**30-day BurstGPT workload.** Prepare the workload once (downloads the
public BurstGPT v2.0 and ShareGPT V3 sources with SHA256-pinned URLs,
roughly a 1 GB download):

```bash
uv run python scripts/prepare_workload.py --days 30
```

Then replay it through the simulator. The experiment code is organized as
one runnable module per routing mechanism; together they produce the
30-day simulation results (each module lists its scenarios and policies
with `--help`, writes `summary.{json,csv}` plus TTFT histograms under
`outputs/simulation/<module>/`, and takes `--jobs N`):

| Command | Mechanism under test | Wall time (64-core server, `--jobs 24`) |
|---|---|---|
| `uv run python -m experiments.simulation.end_to_end --jobs 24` | joint cost+latency routing | ~24 min (39 cells) |
| `uv run python -m experiments.simulation.cost_layer --jobs 24` | cost tiers: on-demand, quota, concurrency | ~17 min (120 cells) |
| `uv run python -m experiments.simulation.hedging --jobs 24` | request hedging | ~16 min (8 cells) |
| `uv run python -m experiments.simulation.latency_layer --jobs 24` | latency-band overlap | ~2 min (21 cells) |

On a laptop, budget roughly 20-40x those times or reduce `--jobs`; every
module also accepts `--max-requests` for a truncated pass.

The paper (§4.3.1) states the 30-day result qualitatively and attaches no
numbers to it: RouteWise exposes a cost–latency frontier, scales to a much
larger request volume than the 24-hour replay, and reduces SLO violations
compared with cost-oriented baselines. Check those three claims in
`outputs/simulation/end_to_end/summary.csv` on the rows with
`scenario = end_to_end_rw8`, the eight-provider pool of the real-world
experiment: every row has `n_requests = 1813565` (the 30-day
BurstGPT+ShareGPT composition, against 14,233 in the real-world replay);
across `ablation_lp_hedging_alpha0` … `ablation_lp_hedging_alpha100`
(RouteWise with hedging) and likewise across the `ablation_lp_only_*` rows
(LP routing alone), `total_cost_usd` rises and `mean_ttft_ms` falls as α
increases; and every `ablation_lp_hedging_*` row has a lower
`slo_violation_rate` than `greedy_cost` (the LP-only α = 0 point is
cost-first and close to, but not identical to, Greedy-cost). The Linux
evaluation reported $233.473 versus $236.759 and mean TTFT 1545.182 versus
1543.868 ms for these two policies, respectively. The exact values depend
on the simulator revision, so this section is checked for those relations
rather than against archived numbers. The simulator's
[premature output-length feedback](docs/research/AE_NOTES.md#simulator-output-length-feedback)
remains a limitation of these results.

**PROD agentic workload (Figure 8; ~1 minute).** The de-identified trace is
included as [data/freeinference.jsonl](data/freeinference.jsonl), with
timestamps, token and cache counters, measurement metadata, and per-account
pseudonyms. It contains 24,035 raw records; 21,678 valid requests enter the
simulation after filtering failed and zero-token rows. Run all eight paper
policies and rebuild the four panels with:

```bash
uv run python scripts/reproduce_figure8.py --jobs 4
```

This command uses the bundled, unmodified simulator source at commit
[`99c5f3f`](https://github.com/HarvardMadSys/RouteWise/commit/99c5f3fd6504377fb57bd4edecef89e3865b7765),
with seed 42, the `end_to_end_rw8` scenario, a 3 s SLO, `bucket_mean`
prediction, and cache accounting enabled. Later code changes, including
the concurrency shadow-price change, alter the intermediate RouteWise
points. **Figure 8 evaluates the earlier constant-`L` concurrency variant;
it does not validate the zero-cost available-concurrency rule in §3.2.3 and
Equation (3).** Running today's `experiments.simulation.end_to_end` module
directly is therefore a different experiment. Cache accounting applies
trace-observed cache-read tokens to candidate providers with discounted
input rates; it does not reconstruct their cache residency under rerouting.
The bundled source requires no extra download and works in the Docker image
too; see
[source provenance](experiments/simulation/PAPER_FIGURE8.md).

`--jobs 1` is also supported: the wrapper now initializes the dataset cache
inside the extracted source directory before starting either worker path.

The command checks request counts and provider counts exactly, and total
cost, mean/P50/P90/P99 TTFT, SLO-violation rate, and hedge rate with
relative/absolute tolerance `1e-9`, against
[figure8_reference_summary.csv](data/figure8_reference_summary.csv).
For example, the SLO-violation rates are 34.95% for Greedy-cost,
4.18% for RouteWise at α = 0.25, and 0.86% for Greedy-latency.
Any mismatch makes the command fail. The paper's §4.3.2 claims read
directly off `outputs/figure8/simulation/summary.csv`: moving from α = 0 to
α = 0.25 cuts mean TTFT from roughly 3.6 s to 1.0 s (`mean_ttft_ms` 3576 to
1022) and SLO violations from 32.5% to 4.2%; beyond α = 0.5 the frontier
flattens (α = 0.75 and α = 1 coincide with Greedy-latency at 0.86%); and
the `provider_mix` column shows Inceptron's share rising to 73.6% at
α = 0.25 and 91.9% at α = 0.5. Independently generated summaries
and histograms go to `outputs/figure8/simulation/`; panels
`figure08a_freeinference_mean_ttft.pdf`, `figure08b_slo_violations.pdf`,
`figure08c_ttft_distribution.pdf`, and `figure08d_provider_mix.pdf`
go to `outputs/figure8/figures/`.

### 4.3 Ablation study (Figure 9 and the offline analysis)

**Offline analysis.** The cost-layer module includes an `offline` policy,
but its released fixed-start, no-queueing semantics differ from the paper's
time-indexed ILP description. It uses deterministic P50-based durations,
whereas online runs sample durations; some quota settings use a heuristic.
The `offline` rows in `outputs/simulation/cost_layer/summary.json` therefore
do not, by themselves, verify the reported 15.0%, 19.3%, and 6.0% gaps to an
optimum. The matched comparison and original solver-status evidence remain
unresolved; see the [offline analysis limits](docs/research/AE_NOTES.md#offline-comparisons).

```bash
# Fixed output-length bias, Figure 9a pipeline (runs + plot; ~1 h):
uv run python scripts/run_output_length_prediction_ablation.py

# Quota / concurrency effective cost, Figures 9b-9c (runs + plots,
# one command; ~10 min with --jobs 8):
uv run python scripts/run_effective_cost_ablation.py --jobs 8
```

The Figure 9a pipeline scales an oracle's output length by a fixed bias in
each run (−50%, −25%, 0%, +25%, +50%, +100%, +200%). It does not implement
independent uniform random errors or establish robustness of an online
bucket-mean estimator to that noise model. The evaluated paper's §4.4.2
description is corrected in the [artifact errata](docs/research/AE_NOTES.md#figure-9a-experiment-scope).
The successful 10,000-request Figure 9 sweeps reported during evaluation
are functionality checks, not reproduction of the full paper results.

### 4.4 Background measurement figures (Figures 2 and 3, ~1 minute)

Both Figure 2 source CSVs are committed in `data/drift_source/`.

```bash
uv run python plots/motivation/drift_wall_clock.py \
    --source-dir data/drift_source --output-dir outputs/figures
```

Produces `drift_wall_clock_llama.{pdf,png}` and
`drift_wall_clock_gpt4o.{pdf,png}` in `outputs/figures/`, and prints each
panel's statistics (row count, global P99, max rolling P99) for comparison
with the paper.

The released timestamps predominantly form close pairs about an hour apart,
with gaps; they do not show a regular five-minute cadence. The rolling
window is 100 samples, spanning a median 52.29 hours for Llama and 50.72
hours for GPT-4o-mini. See the [Figure 2 sampling clarification](docs/research/AE_NOTES.md#figure-2-sampling).

Figure 3 draws the two TTFT background panels from the sanitized production
request export committed in `data/motivation/ttft_duration/`:

```bash
uv run python -m plots.motivation.plot_ttft_background_panels \
    --output-dir outputs/figures
```

Produces `figure03_ttft_background_panels.{pdf,png}` in `outputs/figures/`
together with a `.summary.json` of the plotted bucket statistics.

## 5. Troubleshooting

- **`Disk quota exceeded` from `pulp/mps_lp.py`** — the LP solver writes
  scratch files to the system temp directory; point `TMPDIR` at a volume
  with space (`export TMPDIR=/path/with/space`).
- **Workers killed / machine unresponsive** — each simulator worker holds
  the full 1.8M-request trace; lower `--jobs` (see resource requirements).
- **First section run is slow to start** — the workload is being pickled
  into a cache on first load; later runs start in seconds.
- **A figure script fails with `No module named 'experiments'`** — run it
  from the repository root, and run `plots.end_to_end.plot_simulation_frontier`
  via `python -m` (as documented), not by file path.

## 6. Citation

```bibtex
@inproceedings{routewise-eurosys27,
  title     = {RouteWise: Latency--Cost Optimization for Multi-Provider LLM Routing},
  author    = {Tian, Muxin and Ni, Haoran and Zhai, Yiyan and Park, Yangsun and Yang, Juncheng},
  booktitle = {Proceedings of the Twenty-Second European Conference on Computer Systems (EuroSys '27)},
  year      = {2027}
}
```

## 7. License and data provenance

The code is MIT-licensed (`LICENSE`). The BurstGPT and ShareGPT source
traces are downloaded from their original public hosts at pinned URLs with
SHA-256 verification and are not redistributed here. The smoke fixture is
synthetic, generated deterministically by
`data/fixtures/generate_smoke_fixture.py`, and contains no text payloads.
