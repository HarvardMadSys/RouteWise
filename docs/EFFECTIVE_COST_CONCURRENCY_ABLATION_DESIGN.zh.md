# Effective Cost Concurrency-Only 消融实验设计

> 目标：利用 §1.3 Featherless weighted-concurrency infrastructure，判断
> concurrency resource 是否应该复用 quota 的 exponential scarcity price，还是保留
> 当前 concurrency-specific pricing。

最后更新：2026-05-07。

---

## 1. 实验问题

Quota-only q-sweep 已经说明：`exp_lu` 在中等 quota tightness 下表现最好，
但在极端稀缺或 abundant regime 下，constant baseline 可能单点更便宜。

Concurrency-only 的问题不同：

```text
quota 是一次性消耗资源；
concurrency 是请求结束后可复用资源。
```

所以 Phase B 不应该默认把 quota 的公式搬到 concurrency。它要回答：

```text
在 S_C + S_A 的 concurrency-only 场景里，
current legacy concurrency price 是否已经足够？
exp_lu / linear_lu 这类 unified candidate 是否真的更好？
```

这一步是 Phase C joint ablation 的前置条件。不能跳过它直接做 joint。

---

## 2. 已有 §1.3 Baseline

现有主结果：

```text
outputs/simulation/cost_layer_1_3_featherless_main/summary.csv
```

已知配置：

```text
scenario = concurrency
plan = featherless_premium
model = sharegpt
model_class = ge_70b
model_concurrency_cost = 4
workload = burstgpt
trace = full 30d
seed = 42
jobs = 8
```

现有 best points 约为：

| Policy | Best n | Total cost |
|---|---:|---:|
| `offline` | 11 | `$444` |
| `greedy_cost` | 12 | `$480` |
| `ablation_lp_only_p0` | 12 | `$549` |

这说明当前 RouteWise p=0 在 concurrency-only 下明显弱于 `greedy_cost`。
Phase B 首先要解释这个 gap，而不是只比较 unified formula。

---

## 3. 固定配置

所有 Phase B cells 固定：

```text
phase = concurrency
concurrency_plan = featherless_premium
model = sharegpt
workload = burstgpt
trace = full 30d
seed = 42
p = 0
hedging = off
explorer = off
policy = LPOnlyAblationPolicy with rolling latency profile
```

Provider set:

```text
S_C = featherless_premium_concurrency
S_A = fixed cheap / mid / expensive API fallback ladder
```

No S_Q provider in Phase B.

---

## 4. Sweep Grid

Concurrency account count:

```text
n ∈ {6, 8, 10, 11, 12, 13, 14, 16}
```

Reason:

- §1.3 offline optimum is near `n=11`.
- §1.3 greedy / current p0 optimum is near `n=12`.
- Lower `n={6,8}` gives scarce concurrency.
- Higher `n={14,16}` checks over-provisioned behavior.

Curves:

| Curve | Formula | Interpretation |
|---|---|---|
| `legacy_linear_u` | `U * u` | 旧版 RouteWise concurrency behavior |
| `exp_lu` | `L * (U/L)^u` | Unified exponential candidate |
| `linear_lu` | `L + u(U-L)` | Unified linear alternative |
| `constant_l` | `L` | Selected RouteWise concurrency default |
| `constant_u` | `U` | Conservative sanity baseline |

Here `u` is weighted concurrency utilization:

```text
u = used_concurrency_cost / total_concurrency_capacity
```

Total main grid:

```text
8 n values × 5 curves × 1 seed = 40 cells
```

Expected qualitative result:

- `constant_l` should be close to `greedy_cost`, because both treat available
  concurrency as almost free at routing time. This is a strong mechanism
  hypothesis, not just a sanity check.
- If `constant_l ≈ greedy_cost`, it confirms that the current p0 gap is caused
  by legacy concurrency pricing being less aggressive than greedy admission.
- If `exp_lu` also approaches greedy/offline without pathological saturation,
  it is a viable unified formula candidate.
- If only `constant_l` approaches greedy/offline, concurrency likely needs a
  different reservation logic than quota; the unified formula claim should be
  weakened or rejected.
- `constant_u` should underuse S_C and prefer S_A, especially at larger n where
  paying fixed fees but avoiding concurrency is clearly wasteful.

Optional sensitivity, only after the main grid is understood:

```text
model = qwen3-coder-30b
model_class = 24_34b
model_concurrency_cost = 2
```

Do not include this in the first main run.

---

## 5. Primary Metrics

Use existing `summary.csv` fields:

```text
total_cost_usd_per_run
api_cost_usd_per_run
subscription_fixed_cost_usd_per_run
tier_mix
provider_mix
mean_ttft_ms
p99_ms
slo_violation_rate
peak_used_concurrency_cost
mean_concurrency_utilization
concurrency_saturated_in_trace
trace_paper_grade
```

Cost is primary. TTFT / P99 / SLO are sanity checks because latency
distributions are intentionally held fixed.

Mechanism metrics:

- `tier_mix["concurrency"]`: how much traffic uses S_C.
- `mean_concurrency_utilization`: whether a curve chronically overfills S_C.
- `peak_used_concurrency_cost`: whether peak demand actually reaches capacity.
- `concurrency_saturated_in_trace`: whether the trace can saturate the chosen n.

---

## 6. Main Figures

### Figure A. Percent Delta Heatmap vs `legacy_linear_u`

Rows:

```text
constant_l
exp_lu
linear_lu
legacy_linear_u
constant_u
```

Columns:

```text
n = 6, 8, 10, 11, 12, 13, 14, 16
```

Cell value:

```text
delta_pct(curve, n) =
  (total_cost(curve, n) - total_cost(legacy_linear_u, n))
  / total_cost(legacy_linear_u, n) * 100
```

Interpretation:

- `0%`: equal to current RouteWise concurrency formula.
- negative/blue: better than current formula.
- positive/red: worse than current formula.

Use percent delta so the plot is comparable across n despite changing fixed
fees.

### Figure B. Total Cost Curves

Line plot:

```text
x = n
y = total_cost_usd_per_run
lines = offline, greedy_cost, legacy_linear_u, exp_lu, linear_lu, constant_l, constant_u
```

Each line must mark its own argmin n with a star:

```text
label: (n*, total*)
```

Add horizontal reference lines for:

```text
offline best total
greedy_cost best total
```

Purpose:

```text
Does any effective-cost curve close the current p0 gap to greedy/offline?
Does it select the same n region as §1.3?
```

This distinction is critical:

```text
Figure A answers: at the same purchased n, which curve is cheaper?
Figure B answers: if each curve can choose its own best n, which formula is
overall cheapest?
```

Add a lower subplot sharing the same x-axis:

```text
x = n
y = mean_concurrency_utilization
lines = the same effective-cost curves
```

This plays the same explanatory role as the binding-day bar in the quota
q-sweep: utilization should fall as n grows, and curve choice should matter
less in low-utilization regimes.

### Optional Appendix Figure

Concurrency usage heatmap:

```text
rows = curve
cols = n
value = tier_mix["concurrency"]
```

Use only if the main heatmap needs mechanism support.

---

## 7. Implementation Plan

Do not modify production `RouteWisePolicy`.

Extend Method A ablation code locally:

```text
experiments/ablations/effective_cost/
  harness.py    # add Phase B scenario generation
  presets.py    # allow concurrency_curve sweep with fixed quota_curve placeholder
  policy.py     # already supports concurrency providers via concurrency_curve
```

Key point:

```text
LPOnlyAblationPolicy already prices S_C using scarcity_price(
    concurrency_curve,
    provider.concurrency.utilization(now),
    cost_envelope,
)
```

So Phase B should mostly be harness/preset work, not a new policy.

CLI target:

```bash
routewise ablation effective-cost \
  --phase concurrency \
  --concurrency-plan featherless_premium \
  --model sharegpt \
  --concurrency-count 6 --concurrency-count 8 \
  --concurrency-count 10 --concurrency-count 11 \
  --concurrency-count 12 --concurrency-count 13 \
  --concurrency-count 14 --concurrency-count 16 \
  --concurrency-curve legacy_linear_u \
  --concurrency-curve exp_lu \
  --concurrency-curve linear_lu \
  --concurrency-curve constant_l \
  --concurrency-curve constant_u \
  --workload burstgpt \
  --p 0 \
  --seed 42 \
  --jobs 8 \
  --output-dir outputs/ablations/effective_cost_phaseB_concurrency_main
```

Parser semantics to lock:

- repeated `--concurrency-count` expands scenarios;
- repeated `--concurrency-curve` expands policies;
- `--p 0` means only `p=0`, not the default p sweep;
- Phase A quota defaults must remain unchanged.

Required tests:

- `make_scenarios(phase="concurrency", concurrency_count=(...))` emits one
  scenario per n.
- CLI with 8 n values × 5 curves × 1 seed builds 40 cells.
- The Phase B CLI test must also assert that `--p 0` produces only p0 policies,
  not the default p sweep. This repeats the Phase A protection on the
  concurrency path.
- Phase B presets must vary `concurrency_curve` while keeping quota curve
  irrelevant for S_C-only runs.
- Non-positive or duplicate concurrency counts fail clearly.

---

## 8. Dispatch Plan

Split by curve group:

```text
freeinference-gpu:
  n={6,8,10,11,12,13,14,16}
  curves={legacy_linear_u, exp_lu, linear_lu}
  24 cells

freeinference-gpu1:
  n={6,8,10,11,12,13,14,16}
  curves={constant_l, constant_u}
  16 cells
```

This split keeps the paper-relevant curves together and puts sanity baselines
on the second server.

Expected runtime:

```text
§1.3 full-trace reference: roughly 16 cells in about 30 minutes.
Phase B has 40 cells total.
gpu  runs 24 cells: expected wall time roughly 75 minutes.
gpu1 runs 16 cells: expected wall time roughly 50 minutes.
```

Actual runtime can drift because q/n=16-like high-capacity cells tend to run
longer. Monitor with tqdm progress and final `summary.csv` row counts rather
than assuming both servers finish together.

Expected row counts:

```text
gpu:  24 rows + header
gpu1: 16 rows + header
merged: 40 rows + header
```

---

## 9. Sanity Checks Before Plotting

After rsync, verify:

1. Combined grid has exactly 40 rows.
2. All rows use:
   - `public_scenario = concurrency`
   - `concurrency_plan = featherless_premium`
   - `model = sharegpt`
   - `model_class = ge_70b`
   - `model_concurrency_cost = 4`
   - `workload_dataset = burstgpt` in metadata
   - `seed = 42`
   - `p = 0`
3. For each n:

```text
subscription_fixed_cost_usd_per_run ≈ 25 * n * trace_days / 30
```

4. `constant_l` should usually use more S_C than `constant_u`.
5. `constant_u` should usually fall back to S_A more often.
6. TTFT / P99 should be close across curves for the same n.
7. If a curve beats `legacy_linear_u`, confirm it is not doing so by causing
   pathological saturation or latency artifacts.

---

## 10. Decision Rule

Do not claim a unified formula from one n.

Use the full n-sweep:

- If `exp_lu` closes the gap to greedy/offline and does not over-saturate S_C,
  it becomes a serious unified candidate for Phase C.
- If `legacy_linear_u` is consistently best or close to best, keep the current
  split formula: quota can use `exp_lu`, concurrency can keep legacy linear.
- If `constant_l` wins only by saturating S_C and pushing latency/utilization
  into bad territory, treat it as an invalid aggressive baseline.
- If `constant_u` wins at small n, interpret it as evidence that scarce
  concurrency should be priced conservatively; do not generalize to abundant n.

The Phase B output should decide which concurrency curve(s) enter Phase C:

```text
current_paper: quota=exp_lu, concurrency=legacy_linear_u
unified_exp:   quota=exp_lu, concurrency=exp_lu
best_split:    quota=best Phase A, concurrency=best Phase B
```

---

## 11. Files To Produce

Raw outputs:

```text
outputs/ablations/effective_cost_phaseB_concurrency_core/
outputs/ablations/effective_cost_phaseB_concurrency_sanity/
```

Merged analysis artifact:

```text
outputs/ablations/effective_cost_phaseB_concurrency_merged/
  summary.csv
  effective_cost_concurrency_percent_delta.csv
```

Plot script:

```text
plots/ablations/effective_cost/plot_concurrency_heatmap.py
```

Figures:

```text
outputs/ablations/effective_cost_phaseB_concurrency_merged/figures/
  effective_cost_concurrency_percent_delta_heatmap.pdf
  effective_cost_concurrency_percent_delta_heatmap.png
  effective_cost_concurrency_total_cost_curves.pdf
  effective_cost_concurrency_total_cost_curves.png
  effective_cost_concurrency_usage_heatmap.pdf   # optional
  effective_cost_concurrency_usage_heatmap.png   # optional
```
