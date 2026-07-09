# Latency-Profile Window Ablation

Sweeps the RouteWise online latency-profile window length (`profile_window_sec`,
default 15 min from `routewise.core.latency_profile.DEFAULT_PROFILE_WINDOW_SEC`)
against environments whose provider TTFT distributions change periodically.
The window controls how quickly the router reacts to provider latency changes:
long windows are stable but slow to adapt; short windows adapt fast but are
noisy and often empty on sparse traffic.

## Environment model

Staggered square wave on top of the §3 `end_to_end_rw8` scenario: every
provider alternates between its baseline TTFT distribution and a degraded one
(`ScaledDistribution`, mean x`magnitude`, shape preserved) every `period`
minutes, with phase offsets staggered across providers so the identity of the
lowest-latency provider genuinely swaps over time. `period=0` denotes the
static environment. Implemented with `Provider.ttft_shift_schedule`.

## Grid

- Scenarios: change period P ∈ {static, 60, 30, 10, 5} min at magnitude 3.
- Policies: window W ∈ {1, 2, 5, 15, 30, 60} min × {LP-only, LP+hedging} in
  `latency_profile_mode="observed"` (with `explorer=True`), plus a
  `configured`-mode oracle per family that reacts to every change instantly.

## Run

```bash
uv run python scripts/run_profile_window_ablation.py --jobs 8
```

Outputs land in `outputs/ablations/profile_window/`: `summary.csv` (enriched
with `window_min`, `shift_period_min`, `profile_fallback_rate`),
`profile_window_delta_summary.csv` (regret vs the same-environment oracle),
and `figures/`.

## Caveats

- `ObservedRollingLatencyProfileStrategy` falls back to the provider's true
  mean when a window is empty — an oracle leak that flatters very small
  windows on sparse trace stretches. `profile_fallback_rate` in the summary
  quantifies how often this fired; interpret small-W cells against it.
- LP-only at alpha=0 keeps only min-effective-cost providers feasible, so
  latency estimates matter mainly for tie-breaks between the subscription
  providers; the hedging family exercises the profile (mean + CDF) much more.
