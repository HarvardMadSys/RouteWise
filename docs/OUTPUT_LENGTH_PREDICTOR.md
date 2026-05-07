# Output-Length Predictor Option

Existing simulator runners gain an optional `--predictor` flag. Default
behavior is unchanged — only the LP routing cost on S_A providers can be
swapped to use a predicted output length instead of the trace truth.
Service time, capacity reservation, and final billing always use truth.

```bash
routewise simulator cost-layer    --predictor ema        # streaming EMA
routewise simulator latency-layer --predictor histogram  # streaming histogram
routewise simulator hedging       --predictor constant_p99
```

## Predictor catalog

| Flag value | Behavior |
|---|---|
| omitted / `none` | Truth (original behavior) |
| `oracle` | Truth, routed through the predictor path |
| `histogram` | Streaming histogram with hierarchical backoff |
| `ema` | Per-model + global EMA, normal-approx quantiles |
| `constant_mean`, `constant_p1` … `constant_p99` | Workload-calibrated constant |
| `fixed:<N>` | Literal token count |

Use `--predictor-quantile {q10,q50,q90}` (default `q50`) to pick which quantile of the
prediction feeds the LP cost.

## Files changed

| File | Change |
|---|---|
| `rwsim/policies/routewise.py` | Added `OutputPredictor` Protocol + optional `output_predictor` / `output_predictor_quantile` fields on `RouteWisePolicy`. When set, S_A LP cost uses the predictor; `observe()` calls `predictor.update()`. |
| `experiments/simulation/common.py` | Predictor spec normalization, instantiation, preset materialization. Extended `make_routewise_presets()` with two kwargs. Wired into `run_policy()`. |
| `experiments/simulation/{cost_layer,latency_layer,hedging}.py` | `--predictor` and `--predictor-quantile` CLI flags. Hedging's `make_policy_presets()` accepts the same kwargs. |
| `experiments/offline_stage/value_estimators/constant.py` | New: `ConstantOutputPredictor` + `workload_constant_value()`. Re-exported from the package `__init__`. |

No new policy classes, no new presets, no engine changes. `default` and `--predictor oracle` produce output.

## Caveats

- **Cost envelope** L/U is still computed from truth `response_tokens`, so a
  biased predictor sees per-request cost on a different basis than the LP
  budget axis. By design — keeps L/U comparable across predictors.
- **Capacity reservation** still uses truth (the engine never sees the
  predictor). If you want predicted output to also affect concurrency
  reservation, that's a follow-up.
- **Hedging backup-selection** ignores the predictor. Only the LP
  cost-router uses it.
- **`DEFAULT_PRESETS`** in `rwsim/policies/__init__.py` does not surface the
  predictor — only the section CLIs do. Direct `RouteWisePolicy(...)`
  callers can pass `output_predictor=` themselves.
