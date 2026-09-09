# 24-hour real-provider replay records

Request-level records of the paper's real-world evaluation: a 24-hour
segment of the BurstGPT workload (14,233 requests) replayed once per policy
against MiniMax-M2.5 served by the MiniMax token plan (quota provider), the
Featherless concurrency subscription, and eight OpenRouter on-demand
providers. Figures 1, 6, and 7 and the real-world numbers in the paper are
computed from these files.

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
the analysis. `SHA256SUMS` covers every file.

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
| `billed_cost_usd` | Metered on-demand charge for the request (zero on subscription tiers) |
| `physical_cost_usd` | Charge including any losing hedge leg |
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
README's section 4.1 command passes these two parameters; running the
script with its defaults would prorate over 8 hours and change total costs.

Regenerated from the private run archive with
`scripts/export_ae_data.py real-eval-records`, which keeps the columns above
and maps the run's internal policy names (`budget_range_p25_hedge`) onto the
names the analysis scripts use (`budget_range_alpha25_hedge`).
