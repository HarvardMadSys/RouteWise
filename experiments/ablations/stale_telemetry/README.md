# Stale-Telemetry Robustness Experiment

Evaluates whether RouteWise's probability-target hedging can mitigate a
sudden provider-side latency increase **before the latency profiler has been
updated**, by sending a backup request to a provider that is not experiencing
the same load spike. Such conditions are hard to control against live
providers, so the experiment runs in the trace-driven simulator.

## Environment model

The §2.2 same-cost real-world hedging scenario (`hedging_real_world_rw3` by
default: three empirical OpenRouter TTFT profiles, identical prices, SLO 2 s)
plus a recurring spike train on the provider the router prefers at baseline
(lowest baseline mean TTFT; `api_fast` in RW3). Every `period` minutes that
provider's TTFT distribution becomes `ScaledDistribution(base, magnitude)`
(mean x`magnitude`, shape preserved) for `duration` minutes and is then
restored, via `Provider.ttft_shift_schedule`. The other providers stay
stationary, so a backup sent to them is never spiked. The first spike starts
after a `warmup` so rolling profiles are full; repeating the spike gives every
policy hundreds of episodes to average over the diurnal trace.

## Telemetry regimes

Each of LP-only and LP+hedging RouteWise runs under three beliefs about
provider latency:

| Regime | Belief source | Reaction to the spike |
|---|---|---|
| `stale` | priors frozen at the pre-spike baseline distribution (`StaleBeliefRouteWisePolicy`) | never — the profiler is not updated |
| `observed` | production rolling profile, `W` = 15 min window (`latency_profile_mode="observed"`, `explorer=True`) | lags by up to `W`; hedges also feed backup samples |
| `fresh` | `configured`-mode oracle reading the true current distribution | instantaneous |

`stale` and `fresh` bracket profiler staleness. The paper question is read
off the `stale` pair: with identical, never-updated beliefs, hedging's only
new information is that the in-flight request has not returned yet. The
conditional success probability at each checkpoint then falls below the
target and the router dispatches a backup to a non-spiked provider.

## Grid

- Scenarios: magnitude ∈ {2, 3, 5} x spike duration ∈ {10} min, period 60 min,
  warmup 30 min (`--magnitude` / `--spike-duration-min` repeat to sweep).
- Policies: `routewise_{lp,hedge}_{stale,observed15m,fresh}`.

## Run

```bash
uv run python scripts/run_stale_telemetry_experiment.py --jobs 18
```

Runs the 18-cell grid on the full 30-day BurstGPT replay and plots; about
10 minutes of wall time with 18 workers on a 64-core server (each worker
holds the trace, roughly 2.5 GB). For a shorter pass, truncate the trace
(`--duration-sec 259200` for three days, about one minute) or use
`--max-requests`. `--pool rw8` swaps in the eight-provider pool;
`--spiked-provider` picks a different victim.

## Outputs

Under `outputs/ablations/stale_telemetry/`:

- `summary.json` / `summary.csv`: one row per (scenario, policy). Whole-run
  columns as in other sections, plus per-phase columns `baseline_*`,
  `spike_*`, `post_spike_*` (`n_requests`, `mean_ttft_ms`, `p50/p90/p99_ms`,
  `slo_violation_rate`, `hedge_rate`, `mean_cost_usd`,
  `spiked_provider_{primary,final}_share`) and spike-phase deltas against the
  no-mitigation `routewise_lp_stale` row
  (`spike_slo_violation_delta_vs_stale_lp_pp`,
  `spike_p99_reduction_vs_stale_lp_pct`, `spike_cost_multiplier_vs_stale_lp`).
  `post_spike` is the `duration`-long window right after each spike ends;
  `baseline` is everything else.
- `spike_timeseries.csv`: 1-minute bins aligned on spike onset
  (`minutes_since_onset` from `-pre_onset` to `period - pre_onset`), pooled
  over all episodes: `n`, `slo_violation_rate`, `mean_ttft_ms`, `hedge_rate`,
  `spiked_provider_final_share`.
- `stale_telemetry_spike_summary.csv` and `figures/`: spike-phase table,
  onset-aligned time series per scenario, and spike-phase metrics against
  magnitude (from `plots/ablations/plot_stale_telemetry.py`).

## Reading the results

- `routewise_lp_stale` is the no-mitigation reference: it keeps sending
  every request to the spiked provider for the whole spike
  (`spike_spiked_provider_final_share` = 1).
- `routewise_hedge_stale` isolates hedging's contribution under identical,
  never-updated beliefs; `spike_cost_multiplier_vs_stale_lp` is the price of
  that mitigation.
- Probability-target hedging only dispatches when some backup lifts the
  conditional success probability to the 0.99 target. Once the beliefs about
  the spiked primary are fresh or partially updated, no backup can reach the
  target, so the `fresh` and `observed` rows barely hedge during the spike
  and their mitigation comes from LP re-routing instead.
- `post_spike_*` shows the profiler hangover: the rolling profile keeps
  avoiding the recovered provider for up to `W` minutes.

## Caveats

- Providers share one price, so routing is latency-only and the LP budget
  knob `alpha` (kept at the §2.2 value 0.75) has no effect.
- The spike is a multiplicative scale of the empirical profile, the same
  model as the profile-window ablation; it does not model queueing
  saturation or error responses.
- In `observed` mode an empty rolling window falls back to the true (fresh)
  distribution. Backup providers that receive no traffic therefore look
  fresh; `profile_mean_fallback_rate` / `profile_cdf_fallback_rate` report
  how often that happened.
- Percentile columns for phases are averaged across seeds (seeds only change
  provider sampling; the trace is shared), unlike the whole-run histogram
  percentiles.
