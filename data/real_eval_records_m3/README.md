# 24-hour real-provider replay records on MiniMax-M3

Request-level records of a second 24-hour real-world replay, run on
2026-09-15 (UTC) for the paper revision. The same BurstGPT day-0 segment
(14,233 requests) as the paper's run in
[`../real_eval_records/`](../real_eval_records/) was replayed once per policy
against MiniMax-M3 served by the MiniMax Plus quota emulation
(OpenRouter-pinned Minimax), the concurrency tier `Featherless_SC` (two
concurrent slots per policy process), and six OpenRouter on-demand providers
(Together, GMICloud, Minimax, AtlasCloud, Novita, StreamLake). The inventory
is `experiments/real_evaluation/data/pilot_or_minimax_m3_subscription_or6_true24h.json`;
its notes and `docs/research/REPRODUCIBILITY.md` explain the move from M2.5
and the provider selection.

## Layout

Same as the paper's records: one directory per policy holding `requests.csv`
and `args.json`, plus `reference_summary.json` and `SHA256SUMS`.

| Directory | Name |
|---|---|
| `budget_range_alpha0_hedge` … `budget_range_alpha100_hedge` | RouteWise at α = 0, 0.25, 0.5, 0.75, 1 (hedging on) |
| `greedy_cost`, `greedy_latency` | Greedy-cost, Greedy-latency |
| `or_auto`, `or_sort_latency`, `or_sort_cost` | OR-auto, OR-latency, OR-price |
| `single_OR_Together` | One on-demand provider (Together), no hedge, no fallback |

All eleven policy processes ran concurrently in one 24-hour window, launched
one minute apart, with a shared profiling prober and the SLO at 3,000 ms.
Quota windows were anchored at each process's start. Every replayed prompt
carried a unique prefix so that provider prefix caches warmed by one process
did not discount another's input tokens, and prefix-cache-aware routing was
off. A hedge race's losing leg was allowed to reach its first token before it
was canceled.

## Fields

The columns are the paper records' release columns (see
[`../real_eval_records/README.md`](../real_eval_records/README.md)). `ts` is
the row-write time at request completion, Unix seconds; the dispatch time is
`ts - e2e_ms / 1000`. The run's full logs also carried the per-decision
router state (candidate set, latency objectives, effective costs, quota and
concurrency occupancy, predicted output length) and per-leg generation ids;
those columns are not part of this export.

## Aggregates

The subscription fixed cost is prorated over the 24-hour window at $1.50 per
non-OpenRouter policy (MiniMax Plus $20 and Featherless Premium $25 per
30-day month), the value the run recorded; OpenRouter-only policies and
`single_OR_Together` carry none. `reference_summary.json` holds the
per-policy aggregates recomputed from these files. Regenerate the six
panels (cost frontiers, TTFT distribution, provider mix, provider TTFT,
provider pricing) and check the aggregates with:

```bash
uv run python scripts/reproduce_real_world_m3.py
```

Outputs go to `outputs/figures/real_world_m3/`. Unlike the paper's Figure 7a,
the provider-TTFT panel here is aggregated from these request records.

Regenerated from the run archive with
`scripts/export_ae_data.py real-eval-records --inventory experiments/real_evaluation/data/pilot_or_minimax_m3_subscription_or6_true24h.json`,
which keeps the release columns and writes the repository inventory path
into each `args.json`.
