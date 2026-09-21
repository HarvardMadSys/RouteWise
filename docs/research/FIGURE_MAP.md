# Evaluated-paper figure evidence map

Figure numbers refer to the 14-page PDF identified in [AE notes](AE_NOTES.md).
`AE` below means the evaluated artifact `ae3a9988b5df6fd3396a41fd4ab53d1b7af1fadd`;
the follow-up leaves its algorithms and data unchanged. `F8` means the
unmodified historical source `99c5f3fd6504377fb57bd4edecef89e3865b7765`
bundled in `experiments/simulation/paper_figure8_source.tar.gz`. The Figure 8
wrapper's cache initialization is the execution-only fix.

Commands below run from the repository root after `uv sync --frozen`.
Paths in the table are repository-relative. A successful plotting command
does not by itself verify the paper's interpretation; limitations are
part of each mapping.

| Figure/claim | Source and preprocessing | Command | Revision | Expected output and validation | Limits |
|---|---|---|---|---|---|
| 1 | `data/real_eval_records/*/requests.csv`; aggregate 10 × 14,233 requests, 24-hour fixed subscription charge | R | AE | `outputs/figures/real_world/figure01_{ttft,slo}.pdf`, `real_world_summary.json`; 10 reference checks at `1e-9` | Reanalysis of recorded runs; success-only TTFT, all-request SLO rate; separate run windows and incomplete-charge limits in AE notes |
| 2 | `data/drift_source/*.csv`; input length 10, valid positive latency, timestamp order, rolling 100 observations | D | AE | `outputs/figures/drift_wall_clock_{llama,gpt4o}.{pdf,png}`; printed row counts, global and rolling P99 | 2,419/2,442 valid samples; irregular cadence, approximately 52.29/50.72-hour median window spans, not five-minute sampling |
| 3 | `data/motivation/ttft_duration/`; successful requests with valid positive timings/tokens, latency ≥ TTFT, agentic filter; see data README | T | AE | `outputs/figures/figure03_ttft_background_panels.{pdf,png}` and `.summary.json` | Aggregation of a sanitized production export, not new measurements |
| 4 | Author-drawn system architecture | No experiment command | Evaluated paper | Illustrative diagram in the paper | No independent numeric result; implementation and scope described in AE notes |
| 5 | Author-drawn hedging timeline | No experiment command | Evaluated paper | Illustrative diagram in the paper | Schematic probability target; independence/calibration assumptions apply |
| 6a–b | Same request records as Figure 1; successful-request TTFT distributions and provider mix | R | AE | `outputs/figures/real_world/figure06{a_ttft_distribution,b_provider_mix}.pdf` | Same recorded-data/accounting limits as Figure 1 |
| 7a | `plots/end_to_end/paper_minimax_provider_latency.json` | R | AE | `outputs/figures/real_world/figure07a_provider_latency.pdf` | Redraw of an author-reconstructed provider-mean snapshot; original profiling logs unavailable |
| 7b | Inventory identified by `data/real_eval_records/*/args.json` | R | AE | `outputs/figures/real_world/figure07b_provider_pricing.pdf` | Uses recorded inventory prices, not current prices |
| 8a–d | `data/freeinference.jsonl`; 24,035 raw rows → 21,678 after failure/zero-token filtering; seed 42, RW8, SLO 3 s, bucket-mean, cache enabled | P | F8; AE plotter | `outputs/figure8/simulation/summary.csv`, `source_revision.json`, four `outputs/figure8/figures/figure08*.pdf`; 8 policy checks at `1e-9`, exact request/provider counts | Earlier constant-`L` concurrency variant; text specifies zero; premature feedback and trace-cache assumptions remain |
| 9a | Composed BurstGPT/ShareGPT workload; oracle output length scaled by seven fixed biases; α = 0, 0.25, 0.5, LP-only and LP+hedging | O | AE wrapper; historical fixed-bias default from `b6ec95b` | `outputs/ablations/output_length_prediction/manifest.json`, summaries and `figures/output_length_prediction_cost_delta_lines.{pdf,png}` | Systematic bias, not uniform random errors; exact original run manifest not recovered; prefix execution is only a functionality check |
| 9b–c | Composed workload; quota limits/concurrency counts crossed with five effective-cost curves | E | AE | `outputs/ablations/effective_cost/manifest.json`, `quota/` and `concurrency/` summaries, metadata and figures | Evaluation verified 25/40 prefix configurations; full numeric reproduction and claimed offline gaps are not established |
| §4.3.1, 30-day RW8 | Public BurstGPT/ShareGPT composition; 1,813,565 requests per policy | W, then S | AE | `outputs/simulation/end_to_end/summary.csv`; 13 RW8 policies, monotonic cost/mean-TTFT sweeps and lower hedged SLO rates than Greedy-cost | Qualitative checks; LP-only α = 0 is close to, not equal to, Greedy-cost; feedback-timing limitation remains |
| §4.4.1, offline gaps | Current cost-layer scenarios and offline baselines | C is an available diagnostic, not verification of the claimed gaps | AE | `outputs/simulation/cost_layer/summary.{json,csv}` | Original configuration/solver-status mapping unresolved; deterministic offline versus sampled online durations and heuristic quota cases preclude asserting the reported 15.0%/19.3%/6.0% as verified bounds |

## Committed-data commands

```bash
# R — recorded real-world figures and reference checks
uv run python scripts/reproduce_real_world.py

# D — drift measurements
uv run python plots/motivation/drift_wall_clock.py \
  --source-dir data/drift_source --output-dir outputs/figures

# T — TTFT background panels
uv run python -m plots.motivation.plot_ttft_background_panels \
  --output-dir outputs/figures

# P — archived Figure 8, including fresh-simulation reference checks
uv run python scripts/reproduce_figure8.py --jobs 4
# Single-worker alternative (same experiment and references):
uv run python scripts/reproduce_figure8.py --jobs 1
```

R checks against `data/real_eval_records/reference_summary.json`; P checks
against `data/figure8_reference_summary.csv`. Those references are read
after generating the new aggregates and simulation results, respectively.
The tolerance is for numeric aggregates, not byte-identical PDF rendering.
The data checksums and source-archive checksum remain unchanged.

## Downloaded-workload commands

```bash
# W — download pinned public sources and construct the 30-day workload
uv run python scripts/prepare_workload.py --days 30

# S — the RW8 scenario for the qualitative 30-day comparison
uv run python -m experiments.simulation.end_to_end --scenario end_to_end_rw8 --jobs 4

# O — the released fixed-bias Figure 9a pipeline
uv run python scripts/run_output_length_prediction_ablation.py

# E — quota and concurrency effective-cost sweeps
uv run python scripts/run_effective_cost_ablation.py --jobs 8

# C — cost-layer scenarios, including the available offline baselines
uv run python -m experiments.simulation.cost_layer --jobs 4
```

Use each command's `--help` for configuration and `--max-requests` for a
functionality check on a shorter prefix. The Figure 9 wrappers also accept
`--dry-run` to print the planned configuration without executing the sweep.
The full Figure 9 defaults are not accompanied by an archived numeric
acceptance table; the AE notes identify what remains unverified.
