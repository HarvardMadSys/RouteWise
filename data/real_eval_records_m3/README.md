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

## Hedging legs

The five `budget_range_alpha*_hedge` directories carry an extra
`hedge_legs.csv`, keyed by `req_id`, holding what a hedging analysis needs and
the release columns do not: each leg's observed time to first token, the hedge
delay, and the losing leg's charge.

| Column | Meaning |
|---|---|
| `hedge_triggered`, `hedge_winner`, `hedge_delay_ms` | Whether a backup was dispatched, which leg answered first, and how long after the primary |
| `primary_ttft_ms`, `backup_ttft_ms` | Each leg's own time to first token, blank when that leg produced none |
| `loser_billed_cost_usd`, `loser_physical_cost_usd` | The charge recorded for the leg that lost the race |

These support the counterfactual behind the per-alpha hedging table: for a
hedged request, the primary's own time to first token is what the request
would have seen had no backup been dispatched. The runs let a losing leg
reach its first token before cancelling it, so `primary_ttft_ms` is present
for 97% of hedged requests; the rest produced no token at all and are left
out of every "without hedge" statistic. Those excluded requests are the ones
where the primary was slowest, so the comparison understates what hedging is
worth.

### What hedging costs, and why the recorded figure is only a floor

Hedging fires on 5 to 8% of requests, so a first guess is that it adds a
similar share to the bill. The recorded figure is far lower, about 0.1 to
0.5% of metered spend, for two reasons that pull in the same direction.

About half of all losing legs land on the quota or concurrency tier, which
bills nothing per request, so those hedges are genuinely free. Of the losing
legs that do land on a metered provider, 94 to 97% report no usage at all:
the leg was cancelled once the winner answered, and the provider returned no
token counts for it. Four of the six metered providers in this inventory are
marked `stream_cancel_billing: continues`, meaning they keep generating and
charging after a cancel, so most of those zeros are a reporting gap rather
than a real saving.

The table therefore reports an estimate built from each provider's own
cancellation rule, with the floor and a ceiling kept beside it in
`hedging_reference_summary.json`.

A losing leg on a `stops` provider is charged for the prompt it had already
processed and nothing more, priced from the inventory using the leg's own
cached-token count. A leg on a `continues` provider is charged for a whole
request. Pricing that whole request uses the router's per-request cost
estimate, which reads about 1.8x high: comparing it with the real charge on
the unhedged metered requests of the same run, where both numbers are known,
gives a correction of roughly 0.55, and the estimate applies that correction.
Free-tier losing legs stay at zero.

That puts the cost of hedging at 5 to 12% of metered spend, falling as alpha
rises because fewer requests are hedged. It sits close to the hedge rate,
which is what one would expect once the free-tier legs and the discount for a
cancelled leg are accounted for. The ceiling, which charges every metered
losing leg an uncorrected full request, is 8 to 18%.

Every figure is expressed against the run's metered spend: the subscription
tiers cost the same whether or not a request is hedged, so that is the
denominator hedging can move. Settling the question exactly needs the per-leg
charges the provider reports after the fact. The runs record each leg's
generation id for that purpose, but those ids are not part of this export.

`hedging_reference_summary.json` holds the per-alpha aggregates computed from
these files.

## Router state

The five `budget_range_alpha*_hedge` directories also carry
`router_state.csv.gz`, keyed by `req_id`: what the RouteWise LP saw at each
decision. The baselines keep no such state. Per-provider quantities are one
column per provider, `latency_objective_ms:<provider>`, `c_eff:<provider>` and
`weight:<provider>`, blank for a provider the LP was not offered.

| Column | Meaning |
|---|---|
| `predicted_output_tokens` | The router's own output-length estimate for the request, from its online predictor |
| `quota_fraction_used` | Fraction `z` of the quota tier consumed in its tightest window at decision time |
| `concurrency_in_flight` | Requests occupying the concurrency tier's slots at decision time |
| `unavailable` | Providers the LP was not offered (no free slot, quota exhausted, or cooling down after an error), `|`-separated |
| `latency_objective_ms:<provider>` | The rolling mean time to first token the LP minimized, whole milliseconds; `1e9` is the penalty a provider carries right after a failure |
| `c_eff:<provider>` | The effective request cost the LP charged: the metered price for an on-demand provider, the shadow price `psi(z) = L (U/L)^z` for the quota tier, zero for the concurrency tier |
| `c_min_usd`, `c_max_usd`, `budget_usd` | The cheapest and dearest effective cost among the candidates and the budget `(1 - alpha) c_min + alpha c_max` the LP was held to |
| `weight:<provider>` | The LP's dispatch probabilities; the primary was sampled from them |

The cost envelope of the run was `L = $0.000143808`, `U = $0.00084768`, the
P10 and P90 of the cheapest on-demand price over the trace.

These columns feed four mechanism measurements of the revision: whether the
concurrency slot was offered whenever free and taken whenever fastest, quota
consumption over the day against Greedy-cost and an offline "quota whenever
available" replay, what a tight budget reserves quota for, and how the LP
rebalances traffic as provider latency moves.

### What the budget reserves quota for

RouteWise never exhausts a quota window, and its quota use *rises* with alpha
(1,458 requests at alpha = 0 against 3,113 at alpha = 1) although a higher
alpha is the less cost-sensitive setting. The resolution is that a tight budget
does not use less quota indiscriminately, it spends quota on different
requests: the long responses, which are dear on a metered provider and free of
marginal charge on quota.

The figure shows this as a composition. One 100% stacked bar per operating
point holds the length mix of the requests that went to quota, over two
reference bars. Both references are needed because the three distributions
differ. `All requests` is the trace itself, where responses over 50 tokens are
24.9% of the workload. `Contested` is the pool of decisions in which the budget
actually had to weigh quota against a metered price, meaning the concurrency
slot was busy; there they are only 3.1%, because the free slot takes most long
requests before that comparison ever happens. Against that pool, quota traffic
at alpha = 0 is enriched in long responses more than fivefold, 16.5% against
3.1%, and the enrichment falls away to 2.4% at alpha = 1.

Note that the 11-50 token band is a third of the trace but only an eighth of
quota traffic at every alpha. That is a routing effect rather than a property
of the workload: at those lengths the prompt dominates the request's metered
price, so an 11-50 token response costs about what a 1-10 token one does and
the budget has little reason to prefer quota for it.

### The LP rebalancing traffic

The last figure puts the LP's input and its output on one time axis for a
single operating point, alpha = 0.5, over hours 10 to 24; the first ten hours
of the trace hold 3% of its requests, too few to measure a traffic share in.
The upper panel is each provider's rolling mean time to first token, which is
the quantity the LP minimizes, and the lower panel is the dispatch
distribution it solved for, averaged over 20-minute bins. A slot holding fewer
than 15 decisions is merged with the one after it rather than dropped, so a
handful of bars are wider and the timeline has no holes; each bar covers
exactly the period it summarizes, and the trace never goes quiet for a whole
slot inside this window.

Read together, the bands narrow as the lines rise. GMICloud is the clearest
case: its rolling latency goes from 1.47x the best alternative before hour 16
to 2.53x after it, and its share of traffic falls from 14.7% to 0.4%. That is
the LP moving traffic, not the provider dropping out, and the `offered_share`
column of `lp_rebalancing` in the summary is what rules the second reading
out: GMICloud was a candidate in 100% of decisions, was never rate-limited and
never failed. The concurrency slot is the one provider whose availability does
move, at 41%, because its two slots are often full.

Across the bins each provider's traffic share runs against its latency, with
Spearman correlations of -0.54 for the Featherless slot, -0.53 for the Minimax
API endpoint, -0.50 for Together and -0.38 for GMICloud. The quota tier is the
exception at -0.19, which is what one would expect: its share answers to the
shadow price and the budget as much as to latency.

The point of the panel is the dashed line in the upper half. Individual
providers swing by more than a factor of two over the day, and GMICloud ends
it at about 1.9 s, while the time to first token the policy actually achieves
stays between roughly 0.4 and 1.0 s throughout.

Regenerate the four figures and the table rows, and check the aggregates
against `mechanisms_reference_summary.json`, with:

```bash
uv run python -m scripts.reproduce_real_world_mechanisms
```

Outputs go to `outputs/figures/real_world_m3/`:
`mechanism_concurrency_rows.tex`, `mechanism_quota_over_time.pdf`,
`mechanism_quota_by_length.pdf` and `mechanism_lp_rebalancing.pdf`.

`mechanisms_reference_summary.json` is that run's `mechanisms_summary.json`
copied here. `SHA256SUMS` covers it, and the checksums are written by the
export commands below, so re-run `scripts/export_ae_data.py router-state`
after replacing the reference or the checksum for it goes stale.

## Aggregates

The subscription fixed cost is prorated over the 24-hour window at $1.50 per
non-OpenRouter policy (MiniMax Plus $20 and Featherless Premium $25 per
30-day month), the value the run recorded; OpenRouter-only policies and
`single_OR_Together` carry none. `reference_summary.json` holds the
per-policy aggregates recomputed from these files. Regenerate the six
panels (cost frontiers, TTFT distribution, provider mix, provider TTFT,
provider pricing) and check the aggregates with:

```bash
uv run python -m scripts.reproduce_real_world_m3
```

Outputs go to `outputs/figures/real_world_m3/`. Unlike the paper's Figure 7a,
the provider-TTFT panel here is aggregated from these request records.

Regenerated from the run archive with
`scripts/export_ae_data.py real-eval-records --inventory experiments/real_evaluation/data/pilot_or_minimax_m3_subscription_or6_true24h.json`,
which keeps the release columns and writes the repository inventory path
into each `args.json`; `scripts/export_ae_data.py hedge-legs` and
`scripts/export_ae_data.py router-state` add the two per-policy extras.
