# 24-hour real-provider replay records

Request-level records of the paper's real-world evaluation: a 24-hour
segment of the BurstGPT workload (14,233 requests) replayed once per policy
against MiniMax-M2.5 served by the MiniMax token plan (quota provider), the
Featherless concurrency subscription, and eight OpenRouter on-demand
providers. Figures 1 and 6 and the real-world policy-level summary metrics
are computed from these files. Figure 7 uses the separate sources described
below.

## Layout

One directory per policy, each holding `requests.csv` (one row per request)
and `args.json` (the provider inventory the run used and the SLO):

| Directory | Paper name |
|---|---|
| `budget_range_alpha0_hedge` … `budget_range_alpha100_hedge` | RouteWise at α = 0, 0.25, 0.5, 0.75, 1 (hedging on) |
| `greedy_cost`, `greedy_latency` | Greedy-cost, Greedy-latency |
| `or_auto`, `or_sort_latency`, `or_sort_cost` | OR-auto, OR-latency, OR-price |

`reference_summary.json` holds the per-policy aggregates the paper reports
(total cost, mean and P99 TTFT, SLO-violation rate), for checking a rerun of
the analysis. `SHA256SUMS` covers the CSVs, per-policy arguments, and reference
summary.

The five baselines ran concurrently in one 24-hour window and the five
RouteWise operating points in a later window, against the same provider pool
and profiling configuration (paper §4.2).

## Fields

| Column | Meaning |
|---|---|
| `ts` | Request dispatch time, Unix seconds |
| `policy` | Routing policy (directory name) |
| `req_id` | Request identifier within the replay |
| `prompt_tokens`, `max_tokens` | Prompt length and output cap of the replayed request |
| `primary_provider`, `backup_provider` | Provider chosen for the primary and (if hedged) backup request |
| `actual_provider`, `tier` | Provider that served the request and its tier (`api`, `quota`, `concurrency`) |
| `status` | `success`, or the failure class |
| `ttft_ms`, `e2e_ms` | Time to first token and end-to-end latency, milliseconds |
| `billed_cost_usd` | Sum of recorded primary and backup billed costs (zero marginal charge on subscription tiers) |
| `physical_cost_usd` | Sum of recorded primary and backup provider costs, including reported subscription-provider usage charges |
| `hedge_triggered`, `hedge_winner` | Whether a backup request was issued, and which leg returned the first token |
| `rate_limited` | Whether the request hit HTTP 429 |

These are the columns the analysis script reads. The original run logs also
carried per-leg timestamps, LP weights, and free-text notes; those are not
part of the artifact. Provider names and prices come from the inventory
referenced in `args.json`
(`experiments/real_evaluation/data/pilot_or_minimax_subscription_or8_true24h.json`).

## Reproducing the figures

Subscription fixed costs are prorated over the 24-hour replay window; the
prorated fixed cost of the two subscriptions for one non-OpenRouter policy
run is $1.5476333333333334 (the value recorded with the paper run). The
`uv run python scripts/reproduce_real_world.py` command passes these two
parameters, recomputes the aggregates, and checks all ten policies against
`reference_summary.json` (exact counts; relative/absolute numeric tolerance
`1e-9`). Calling the underlying plot module with its defaults would prorate
over 8 hours and change total costs.

## Metrics and cost accounting

Mean and percentile TTFT use successful requests with nonnegative, present
TTFT. SLO-violation rate uses all 14,233 requests as the denominator; failures
and successful requests with TTFT above 3,000 ms count as violations. These
denominators intentionally differ. The unrounded α = 0 reductions relative
to OR-auto are 26.48% (total cost), 58.15% (mean TTFT), and 98.01% (SLO
violations). These differ slightly from the paper's 26.6%, 58.4%, and 98.2%.
Calculate the reductions from the regenerated summary with:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
rows = json.loads(Path("outputs/figures/real_world/real_world_summary.json").read_text())
by_policy = {row["policy"]: row for row in rows}
rw, baseline = by_policy["budget_range_alpha0_hedge"], by_policy["or_auto"]
for key in ("total_cost_usd", "ttft_mean_ms", "slo_violation_rate"):
    reduction = 100 * (baseline[key] - rw[key]) / baseline[key]
    print(f"{key}: {reduction:.2f}% relative reduction")
PY
```

Reported totals add fixed subscriptions to `billed_cost_usd`; the recorder
sums both primary and backup legs. Profiling/probe expense is tracked
separately by the runner and is excluded from these totals. For a canceled
leg whose usage is not returned and whose billing continues, the transport
can record zero for an unmeasured charge. That value is not evidence of a
free request. The released export omits the per-leg cost-source annotations
and probe ledger, so it cannot quantify or bound those missing charges.
`physical_cost_usd` is a separately recorded measure, not an additional
term to add again to the billed total. These totals reproduce the exported
accounting records, not a complete invoice reconciliation.

## Provider diagnostics and provenance

Figure 7a uses `plots/end_to_end/paper_minimax_provider_latency.json`, an
author-reconstructed snapshot of the paper's provider mean TTFT values
(seconds). Its original profiling logs are unavailable. This redraw is
not an independent reconstruction from the released request records, whose
per-provider aggregate means can differ. Figure 7b uses the prices in the
inventory referenced by `args.json`; it does not query current provider
prices.

Regenerated from the private run archive with
`scripts/export_ae_data.py real-eval-records`, which keeps the columns above
and maps the run's internal policy names (`budget_range_p25_hedge`) onto the
names the analysis scripts use (`budget_range_alpha25_hedge`).
