# Effective Cost Calibration Ablation

This package runs the follow-up sensitivity study for the `L` and `U`
envelope used by quota/concurrency shadow prices.

The main effective-cost ablation compares curve shapes while holding the
envelope fixed:

```text
psi(z) = L * (U / L)^z
```

This follow-up holds the curve shape fixed, using the curve-shape ablation
result (`exp_lu`), and varies only the envelope calibration.

## Fixed Config

- Workload: defaults to `burstgpt`, matching the existing curve-shape result
  under `outputs/ablations/effective_cost_phaseA_*`.
- Quota setting: `q* = 16`
- Quota latency family: `heavy_tail`
- Seed: `42`
- Curve shape: `exp_lu`
- LP budget knob: `p = 0.5`

Use `--workload sharegpt_burstgpt` if the paper run needs the combined trace.

## Candidate Calibration Axes

API reference used to compute request value:

- `cheapest_api`: current default outside option
- `median_api`: sensitivity check
- `mean_api`: sensitivity check
- `max_api`: conservative upper reference

Percentile envelope:

- `p10_p90`: current default
- `p05_p95`: wider, more tail-sensitive
- `p25_p75`: narrower, smaller dynamic range
- `min_max`: sanity check; likely unstable under long-tail requests

## Default Run

The default command runs the two one-dimensional sweeps and de-duplicates the
default point:

```bash
uv run routewise ablation effective-cost-calibration \
  --seed 42 \
  --jobs 8 \
  --output-dir outputs/ablations/effective_cost_calibration
```

This produces seven policies:

- `cheapest_api` with `p05_p95`, `p10_p90`, `p25_p75`, and `min_max`
- `cheapest_api`, `median_api`, `mean_api`, and `max_api` at `p10_p90`

Run the full cross-product only if the one-dimensional sweeps show strong
interactions:

```bash
uv run routewise ablation effective-cost-calibration \
  --sweep cross-product \
  --seed 42 \
  --jobs 8 \
  --output-dir outputs/ablations/effective_cost_calibration_cross
```

## Questions To Answer

- Does `cheapest_api + p10_p90` remain competitive against nearby calibration
  choices?
- Does widening the envelope make `exp_lu` too aggressive early and too steep
  late?
- Does narrowing the envelope under-protect quota when quota is binding?
- Do `median_api` or `mean_api` references change the conclusion of the curve
  ablation, or only shift absolute shadow-price scale?
- Is `max_api` too conservative, causing quota underuse and higher API spend?

Keep this separate from the curve-shape ablation to avoid mixing two design
variables in the first paper-facing result.
