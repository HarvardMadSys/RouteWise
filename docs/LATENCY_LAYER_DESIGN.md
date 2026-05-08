# Latency Layer Design

Last updated: 2026-05-07.

This document defines the synthetic latency-layer scenarios for simulator
section 2.1. It resolves the overlap-parameterisation open item in
`docs/EXPERIMENT_LAYOUT.md`.

---

## 1. Goal

The latency layer isolates one paper question:

> When all providers have the same cost, does RouteWise route toward the
> fastest provider?

Cost is held fixed across providers. The only controlled difference is the
configured latency distribution.

This section is not a hedging experiment and is not an end-to-end
cost-latency tradeoff experiment. Hedging lives in `experiments/simulation/hedging.py`.

---

## 2. Provider Setup

Each synthetic scenario uses three on-demand API providers:

| Provider | Mean TTFT |
|---|---:|
| `fast` | `100ms` |
| `medium` | `300ms` |
| `slow` | `1000ms` |

All three providers use the same input-token and output-token prices. The
providers differ only in their latency distributions.

The default RouteWise budget knob for this section is `p = 0.75`. Add a
`p` sweep only if a paper figure explicitly needs it.

---

## 3. Scenario Grid

The first grid is:

| Family | Scenarios |
|---|---|
| `uniform` | `latency_layer_uniform_no_overlap`, `latency_layer_uniform_half_overlap` |
| `normal` | `latency_layer_normal_no_overlap`, `latency_layer_normal_half_overlap` |
| `heavy_tail` | `latency_layer_heavy_tail_no_overlap`, `latency_layer_heavy_tail_half_overlap` |
| `real_world` | `latency_layer_real_world` |

The `heavy_tail` family is the simulator's `LogNormal` distribution.

Real-world latency profiles report the same realised band-coverage metrics,
but they are not calibrated by the synthetic overlap construction and do not
carry an overlap label.

---

## 4. Quantile And Band Definitions

Let `d` denote one provider's latency distribution. Let `L_d` denote one
latency sample drawn from distribution `d`, measured in milliseconds.

For a probability level `alpha` in `(0, 1)`, define the `alpha` quantile
`q_alpha(d)` as the latency value satisfying:

```text
P(L_d <= q_alpha(d)) = alpha
```

The two quantiles used in this design are:

```text
q10(d) = q_0.10(d)
q90(d) = q_0.90(d)
```

So `q10(d)` is the 10th-percentile latency, and `q90(d)` is the
90th-percentile latency.

Define the central latency band `B(d)` as:

```text
B(d) = [q10(d), q90(d)]
```

This band contains the central 80% of the distribution by probability mass.
It deliberately ignores the lowest 10% and highest 10% tails.

For an interval `I = [a, b]`, define its length as:

```text
|I| = max(b - a, 0)
```

For two intervals `I = [a, b]` and `J = [c, d]`, define their intersection
length as:

```text
|I intersect J| = max(min(b, d) - max(a, c), 0)
```

---

## 5. Construction Metric

Synthetic overlap scenarios are constructed using directional Q10-Q90 band
coverage.

For two provider distributions `d_a` and `d_b`, define:

```text
coverage(d_a -> d_b) = |B(d_a) intersect B(d_b)| / |B(d_a)|
```

This metric asks:

> What fraction of provider `a`'s central latency band is covered by provider
> `b`'s central latency band?

It is directional. In general:

```text
coverage(d_a -> d_b) != coverage(d_b -> d_a)
```

The construction anchor is always the fast-to-medium pair:

```text
coverage(d_fast -> d_medium)
```

The slow provider is still included in every scenario, but the construction
target does not try to force `medium -> slow` or `fast -> slow` to match the
same number. Those pairwise overlaps are measured and reported post hoc.

---

## 6. Overlap Regimes

The two synthetic overlap regimes are:

| Regime | Construction target |
|---|---|
| `no_overlap` | `coverage(d_fast -> d_medium) = 0` |
| `half_overlap` | `coverage(d_fast -> d_medium) = 0.5` |

Therefore, `half_overlap` means:

> Half of the fast provider's Q10-Q90 latency band is covered by the medium
> provider's Q10-Q90 latency band.

It does not mean that the full probability distributions overlap by 50%.

Example:

```text
B(d_fast)   = [70ms, 130ms]
B(d_medium) = [100ms, 460ms]
```

The overlap interval is:

```text
[100ms, 130ms]
```

Its length is `30ms`. The fast band length is `130ms - 70ms = 60ms`. Thus:

```text
coverage(d_fast -> d_medium) = 30ms / 60ms = 0.5
```

This is a valid `half_overlap` construction example.

---

## 7. Family Parameterisation

Within one synthetic scenario, each family uses one shared shape parameter
across the three providers. The shape is relative to the real-space mean TTFT,
so fast, medium, and slow keep the same distributional form at different
latency scales. Mean anchoring matches the RouteWise LP objective, which uses
expected TTFT.

Let:

```text
m_fast = 100ms
m_medium = 300ms
m_slow = 1000ms
R = m_medium / m_fast = 3
```

### Uniform

Uniform latency uses a relative half-width `r`:

```text
d_m = Uniform[m * (1 - r), m * (1 + r)]
```

where `m` is the provider mean TTFT. The implementation must keep `r < 1` so the
lower support remains non-negative.

The Q10-Q90 band is:

```text
B(d_m) = [m * (1 - 0.8r), m * (1 + 0.8r)]
```

### Normal

Normal latency uses a relative standard deviation `c`:

```text
d_m = Normal(mean_ms = m, sigma = c * m)
```

The simulator clips sampled Normal latencies at `1ms`. Scenario parameters
should keep the Q10-Q90 band positive; plots that show Q1-Q99 whiskers should
clip the displayed lower whisker to the same floor.

Let:

```text
z90 = inverse_standard_normal_cdf(0.90) ~= 1.28155
```

The Q10-Q90 band is:

```text
B(d_m) = [m * (1 - z90*c), m * (1 + z90*c)]
```

### Heavy Tail

Heavy-tail latency uses the simulator's `LogNormal` distribution with shared
log-space standard deviation `sigma_log`:

```text
d_m = LogNormal(mu = ln(m) - 0.5 * sigma_log^2, sigma = sigma_log)
```

The mean is `m`; the P50 is lower than `m` whenever `sigma_log > 0`.

Let:

```text
A = exp(z90 * sigma_log)
```

The Q10-Q90 band is:

```text
B(d_m) = [m / A, m * A]
```

---

## 8. Exact Half-Overlap Construction

The implementation should solve each family-specific shape parameter so that:

```text
coverage(d_fast -> d_medium) = 0.5
```

For families whose Q10-Q90 bands can be written as:

```text
B(d_m) = [m * (1 - a), m * (1 + a)]
```

the fast-to-medium coverage with `R = 3` is:

```text
coverage(d_fast -> d_medium) = 2 - 1/a
```

Setting the coverage to `0.5` gives:

```text
a = 2/3
```

This yields:

| Family | Half-overlap parameter |
|---|---:|
| Uniform | `r = (2/3) / 0.8 = 5/6` |
| Normal | `c = (2/3) / z90 ~= 0.520` |

For LogNormal, the fast-to-medium coverage is:

```text
coverage(d_fast -> d_medium) = (A^2 - R) / (A^2 - 1)
```

With `R = 3` and target coverage `0.5`:

```text
A^2 = 5
sigma_log = ln(sqrt(5)) / z90 ~= 0.628
```

The no-overlap regime does not need a unique closed-form parameter because
any shape with disjoint fast and medium Q10-Q90 bands has target coverage `0`.
The implementation should choose a deterministic shape and verify the realised
fast-to-medium band coverage is `0`.

---

## 9. Reporting Metrics

Every scenario summary should include both construction metadata and realised
overlap measurements.

Construction metadata:

```text
overlap_regime
overlap_construction_metric = q10_q90_directional_band_coverage
target_anchor_pair = fast_medium
target_band_coverage_fast_medium
```

Realised band-coverage metrics:

```text
realised_band_coverage_fast_medium
realised_band_coverage_medium_slow
realised_band_coverage_fast_slow
```

The realised metrics should be computed from the configured distributions,
not inferred from scenario names.

---

## 10. Expected Results

In `no_overlap` scenarios:

- `greedy_latency` and RouteWise should concentrate on the fast provider.
- `random` should split roughly evenly across fast, medium, and slow.
- RouteWise should have much lower mean, P50, and tail latency than `random`.

In `half_overlap` scenarios:

- RouteWise should still prefer the fast provider.
- Medium-provider share may rise because fast and medium are harder to
  distinguish from observed latency samples.
- Slow-provider share should remain low.

If RouteWise routes heavily to the slow provider in synthetic no-hedging
latency-layer scenarios, the latency layer implementation should be inspected
before running larger grids.

---

## 11. Implementation Boundaries

Experiment-specific overlap helpers belong under:

```text
experiments/simulation/latency_overlap.py
```

The section runner belongs in:

```text
experiments/simulation/latency_layer.py
```

Do not add construction-specific concepts such as "half overlap" to
`rwsim/world/distributions.py`. The world distributions should remain generic:
`Uniform`, `Normal`, `LogNormal`, empirical distributions, and the shared
latency distribution protocol.

The latency-layer implementation should add tests for:

- scenario names and grid size;
- equal-cost provider invariant;
- mean TTFT ladder `100ms / 300ms / 1000ms`;
- half-overlap fast-to-medium band coverage;
- no-overlap fast-to-medium band coverage;
- presence of realised overlap metrics in scenario/run summaries.
