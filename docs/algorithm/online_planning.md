# Online Implementation Plan (Hybrid Inference)

This document describes a concrete, reproducible implementation plan for the online algorithms in
`docs/algorithm/online.tex`, with an initial focus on Stage 1 (quota-only) and an optional extension
to Stage 2 (quota + concurrency).

## Scope and Milestones

**Milestone A (Stage 1, required):** Learning-Augmented Primal-Dual (LA-PD) for `S_Q` using a
quantile predictor for output length and an explicit safeguard/fallback path.

**Milestone B (Stage 2, optional):** Extend LA-PD to `S_C` by predicting a conservative duration
upper bound `p_t^{UCB}` (via `TTFT` and `TPS` quantiles) and integrating it into CAPQ (value density,
opportunity cost, feasibility).

## Code Layout (Proposed)

- `experiment/predictors/`
  - `base.py`: Interfaces and shared dataclasses (quantiles, prediction context).
  - `histogram.py`: Dependency-free streaming quantile baseline (histogram + backoff).
  - `ema.py`: Simple EMA baseline for output tokens (and optionally TTFT/TPS).
  - `lightgbm_quantile.py` (optional): Quantile regression model if `lightgbm` is available.
- `experiment/strategies/online/learning_augmented.py`
  - `LearningAugmentedPrimalDualStrategy`: Stage 1 LA-PD strategy using `v_t^{LCB}` + fallback.
  - `LearningAugmentedUnifiedStrategy` (optional): Stage 2 unified router using `v_t^{LCB}` and
    `p_t^{UCB}`.
- `experiment/strategies/online/__init__.py`
  - Export new strategies.
- `experiment/scripts/run_online_experiments.py`
  - Add LA-PD baselines and evaluation metrics output (savings ratio + calibration metrics).

## Predictor Design

### Interface

At routing time, the predictor must only use features observable at request arrival.
Ground-truth `response_tokens` / `latency_ms` may only be used in post-decision updates.

Minimum interface (Stage 1):
- `predict_output_tokens_quantiles(request) -> (q10, q50, q90)`
- `update_after_completion(request)` (uses `request.response_tokens` if available)

Optional interface (Stage 2):
- `predict_duration_ucb_seconds(request) -> float`
- `update_sc_stats_after_completion(request)` (uses `ttft_ms`, `latency_ms`, `response_tokens`)

### Features (Arrival-Time Observable)

Use only features available in `experiment.data.schema.Request`:
- `request.request_tokens` (input token count)
- `request.model` (categorical)
- `request.timestamp` (derive hour-of-day / day-of-week)

If additional fields exist in a specific dataset, they can be added behind a feature gate, but the
default implementation must not assume access to prompt content.

### Baseline: Dependency-Free Quantile Predictor

Implement a streaming histogram estimator with a backoff chain:
- Keyed stats (most specific): `(model, input_bin, hour_bin)`
- Backoff: `(model, input_bin)` -> `(model)` -> `(global)`

Use log-spaced bins for output tokens (and, for Stage 2, for `TTFT` and `TPS`).
Quantile queries return the bin midpoint at the target cumulative mass.

This baseline is:
- Deterministic and fast (no training pipeline required).
- Fully reproducible in the sandbox (no extra packages).

### Optional: LightGBM Quantile Regression

If `lightgbm` is available, implement a quantile model:
- Train separate models for `q10/q50/q90`, or use one model per quantile.
- Use a strict time-based split (see below) to avoid leakage.
- Persist model artifacts under `experiment/cache/` with a config hash key.

This should be treated as an optional improvement over the dependency-free baseline.

## LA-PD Strategy (Stage 1)

### Decision Rule

For request `r_t`:
1. Predict output token quantiles `(L10, L50, L90)`.
2. Compute conservative value:
   - `v_t^{LCB} = Cost_API(n_in, L10)`
3. Compute shadow price threshold `theta_t = psi(z_t)` where `z_t = used/Q`.
4. Route to subscription iff:
   - `used < Q` and `v_t^{LCB} > theta_t`

### Safeguard / Fallback

To avoid catastrophic behavior under adversarial underestimation:
- If the predictor is not warmed up (insufficient samples), or
- If `v_t^{LCB} < epsilon` (suspiciously low),

then fall back to a non-learning estimator (EMA-based `hat{n}^{out}`) and compute `v_t` from it.

The fallback must be implemented in code (not only described in the paper).

### State Updates

After a request completes, update the predictor with realized `response_tokens`.
In offline replay, this happens immediately after `route()` returns, but the strategy must treat the
update as “post-decision” and never use realized outputs for the current decision.

## Evaluation Protocol

### Splits (No Leakage)

Use time-based splits by day:
- Train: earliest days
- Validation: middle block (for quantile choice and `epsilon`)
- Test: latest days (report numbers here)

If using an online-updating predictor, “training” can be treated as a warmup period and excluded
from reported metrics.

### Baselines

Stage 1:
- Greedy
- Primal-Dual (EMA value estimate)
- LA-PD (P10 default), plus ablations P20/P50
- Offline Optimal (Stage 1 oracle)

Stage 2 (optional):
- Greedy priority `S_Q -> S_C -> S_A`
- Unified Primal-Dual (CAPQ)
- LA-PD + CAPQ using `p_t^{UCB}`
- Offline ILP Optimal (if solver available)

### Metrics

Routing:
- Total Savings (USD)
- Savings Ratio: `Savings_online / Savings_optimal`
- Quota Utilization

Prediction calibration:
- Quantile Coverage for each quantile (e.g., P10 coverage should be close to 10%)
- Pinball loss (optional)
- MAE for median prediction (optional)

## Implementation Checkpoints

1. Sanity: P10 coverage is not degenerate (e.g., not 0% or 100%).
2. No leakage: decisions do not read `response_tokens` / `latency_ms` for the current request.
3. Regression: LA-PD (P10) is between Primal-Dual (EMA) and Offline Optimal on savings ratio.

## Suggested Execution Commands

- Stage 1 experiments (existing runner):
  - `python experiment/scripts/run_online_experiments.py --stage 1`
- Stage 2 experiments (existing runner):
  - `python experiment/scripts/run_online_experiments.py --stage 2`

If new strategies are added to the runner, ensure results are written to `experiment/results/online/`
with a stable JSON schema compatible with existing plotting scripts.
