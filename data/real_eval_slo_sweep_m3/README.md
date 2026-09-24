# Real-provider SLO sweep

One RouteWise operating point, alpha = 0.5 with hedging, replayed the same
24-hour BurstGPT day-0 segment (14,233 requests) four times, once per SLO
target. Everything else is held fixed: the same provider pool as
[`../real_eval_records_m3/`](../real_eval_records_m3/), the same trace, the
same profiling configuration, and quota windows anchored at each run's start.
The four processes ran concurrently in one 24-hour window, launched a minute
apart, on 2026-09-16.

| Directory | Router SLO |
|---|---|
| `budget_range_alpha50_hedge__slo2000` | 2,000 ms |
| `budget_range_alpha50_hedge__slo3000` | 3,000 ms |
| `budget_range_alpha50_hedge__slo4000` | 4,000 ms |
| `budget_range_alpha50_hedge__slo5000` | 5,000 ms |

The SLO is the router's own target: it sets the deadline the LP plans
against and the checkpoints at which hedging fires. Each run is therefore
scored against its own target, not a common one.

## Layout and fields

One directory per SLO, each holding `requests.csv` and `args.json`, with the
same release columns as the paper's records; see
[`../real_eval_records/README.md`](../real_eval_records/README.md) for the
column meanings. `args.json` records the provider inventory and that run's
SLO. `reference_summary.json` holds the per-run aggregates and `SHA256SUMS`
covers every file.

## Reproducing

```bash
uv run python -m scripts.reproduce_hedging_tables
```

That recomputes the aggregates from the CSVs, writes this sweep as
`table2_slo_sweep.tex` plus two panels under
`outputs/figures/hedging_tables/`, and checks the result against
`reference_summary.json`. The same command also produces the per-alpha
hedging table from [`../real_eval_records_m3/`](../real_eval_records_m3/).
Subscription fixed cost is prorated over the 24-hour window at $1.50 per run,
the value the runs recorded.

A sweep like this runs in a single launcher pass: a `POLICY_LIST` entry may
carry a `__slo<ms>` suffix, which names the process and overrides the SLO for
it alone, so one policy can appear several times at different targets.

```bash
POLICY_LIST="budget_range_alpha50_hedge__slo2000 budget_range_alpha50_hedge__slo3000" \
    bash scripts/run_real_eval_8h_policy_processes.sh
```

## What the runs show

Loosening the target from 2 s to 5 s cuts violations from 8.51% to 0.41%
while total cost stays within one cent, because the router pays for the
tighter target in hedging rather than in money: the hedged share falls from
14.6% to 4.8% as the deadline relaxes, and the backup wins more often when it
does fire. Mean TTFT is lowest at the 3 s target; the 2 s runs hedge hardest
and still miss most often, which is the point at which the pool cannot meet
the target at this operating point.
