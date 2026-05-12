# Output Length Prediction Ablation Design

> 目标：把 output length prediction 从一个临时 sensitivity check 整理成可复现、可解释、能回应 Juncheng 质疑的 paper ablation。

## 1. 背景

当前主线 simulator 的默认假设是：

- 每个 request 的真实 output length 来自 trace，例如 `response_tokens` / `num_decode_tokens`。
- 这个 output length 在所有 provider、policy、latency distribution 下固定。
- RouteWise 在 routing decision 时默认提前知道这个真实 output length。

这不是“所有请求 output length 都一样”。它是“每个请求有自己的真实长度，但 router 在决策时可以偷看 truth”。

这个假设方便先验证 cost layer、latency layer、hedging、joint routing 本身，但它不是线上真实情况。线上 routing 时只能看到 input tokens、prompt/context、历史统计等，不能知道这次 generation 最终会输出多少 tokens。

因此需要一个专门的 value-estimation / output-length-prediction ablation 来回答：

> 如果 RouteWise 不再提前知道真实 output length，而只能用预测值，routing cost 和 provider mix 会变差多少？

## 2. 原始版本做了什么

从会议记录看，早期版本主要是 misprediction robustness / sensitivity check：

1. 比较真实 output length、EMA / histogram 等预测方法、oracle / offline 结果。
2. 观察到 cost gap 大约只有 2%-3%，于是形成了“output length prediction 不敏感”的结论。
3. 后续又做过一些人工扰动：
   - `predicted = true_output * k`
   - lognormal / random multiplicative noise
   - input-correlated cases，例如 long underestimate、short overestimate、tail underestimate

这个版本的问题不是方向错，而是结论表达太强，而且证据不够完整：

- 主要看 total cost，容易掩盖 provider mix / quota usage 的变化。
- 缺少 extreme-value sanity check，无法确认 predictor 是否真的打到 routing critical path。
- 结论偏二元：“敏感 / 不敏感”，没有展示 error 逐步变大时的趋势。
- 没有把 predictor / noise 参数写入 artifact，复现实验不稳。

如果原版确实只看到约 2%-3% gap，需要先把这个现象解释成待验证的候选原因，而不是直接写成结论：

- global multiplicative scaling 可能没有改变 request 之间的 value ranking，只改变整体 API cost level。
- 场景可能不在 quota / concurrency 的决策边界附近，因此 prediction error 没有机会改变 admission。
- input token cost 可能主导了 pay-per-token API cost，output token 的扰动被稀释。
- workload 的 output length 分布可能本来就窄，简单 predictor 已经足够。
- 当前 shadow-price / budget setting 可能让 `S_A`、`S_Q`、`S_C` 的相对排序很稳定。
- 旧的 `lambda(u) = U * u` concurrency 公式会让 `S_C` effective cost 随 utilization 平滑升降，predictor 在 API vs `S_C` 决策点的误差可能被 `u` 自身的动态吸收；Phase D 直接测试这个假设。

新版 ablation 要做的是区分这些解释，而不是只复现“gap 很小”这个数字。

## 3. Juncheng 为什么觉得反直觉

RouteWise 的 on-demand API estimated cost 是：

```text
estimated_api_cost =
  input_price * input_tokens
  + output_price * predicted_output_tokens
```

quota provider 通常按 request 消耗 quota，不按 output token 消耗 quota。因此 output length prediction 直接影响一个 request 的 estimated value：

- 如果预测输出很长，这个 request 走 API 会很贵，更应该用 quota / subscription。
- 如果预测输出很短，这个 request 走 API 不贵，quota 应该留给更高价值的 request。

所以合理预期是：

```text
output prediction changes
  -> estimated API cost changes
  -> request ranking / quota admission changes
  -> provider mix and API cost change
```

如果实验显示无论怎样扰动 output prediction 都几乎不变，可能有两种解释：

1. 系统确实鲁棒，output length 在该 workload 下不是主导因素。
2. 实验没有打到关键路径，或者场景设计不敏感，或者实现没有真正使用 prediction。

因此 Juncheng 要求先做 sanity check：用极端常数或极端 scale 验证 routing 会不会明显变化。如果极端设置都不改变结果，优先怀疑实现或实验设计。

## 4. 为什么不照搬 cost layer 的 1.1 / 1.2 / 1.3 全网格

Notion 里的主 simulation 结构是：

- 1.1 on-demand only：同 latency，不同 API cost
- 1.2 add quota provider
- 1.3 add concurrency provider
- 每节覆盖 uniform / normal / heavy-tail / real-world latency distribution

这套结构适合验证 cost layer / latency layer 本身，但不适合作为 output-length ablation 的主网格。

### 4.1 1.1 on-demand-only 对 output prediction 不敏感

在 1.1 中，所有 provider 都是 on-demand API，价格比例固定，例如 cheap / mid / expensive。output length prediction 改变后，三个 provider 的 cost 大多仍保持同样排序：

```text
cheap < mid < expensive
```

没有 quota / subscription admission 问题，也没有“哪些 request 更值得消耗 scarce capacity”的排序问题。因此 1.1 很难检验 output prediction 的关键作用。

1.1 可用于 smoke check，但不应该是 paper ablation 主体。

### 4.2 四种 latency distribution 不是核心变量

output length prediction 影响的是 estimated API cost，而不是 TTFT distribution。uniform / normal / heavy-tail / real-world latency family 主要用于验证 latency router 和 hedging。

如果在 output-length ablation 中再乘上四种 latency distribution，会产生大量重复格子，并引入 latency sampling noise，使结论更难解释。

除非我们想证明结果对 latency family 也鲁棒，否则主实验应固定一个代表性 latency setup，把变量集中在 prediction error 上。

### 4.3 真正关键的是 subscription/API 边界

output prediction 最应该影响的是：

- `S_Q + S_A`：quota request slot vs pay-per-token API
- `S_C + S_A`：concurrency slot vs pay-per-token API
- `S_Q + S_C + S_A`：joint scarce-capacity routing

尤其要选择 quota / concurrency / subscription count 在决策边界附近的 setting，例如 §1.2 中 q* 附近，或 §1.3 中 concurrency 刚开始饱和的 count 附近。

如果 quota 过多，几乎所有 request 都能进 quota，prediction 不重要。
如果 quota 过少，只有极少数 request 能进 quota，prediction 也可能不明显。
最敏感的是中间区域：一些 request 该进 quota，一些 request 该走 API。

## 5. 完整实验设计

完整 ablation 应分成四个阶段：sanity、noise trend、real predictor comparison、S_C shadow-price sensitivity。

### 5.0 Scenario Matrix

先把 scenario x phase 固定下来，避免实际跑实验时临时拼 grid。

除非另行标注，所有 phase 共用：BurstGPT 30d、`seed=42`、`p=0`、`latency_family=heavy_tail`（LogNormal-like）、单一 latency family、`S_C` shadow-price baseline 为 `conc_constant_l`。

| Scenario | 角色 | Phase A: sanity | Phase B: noise trend | Phase C: real predictors | Phase D: S_C shadow price |
|---|---|---:|---:|---:|---:|
| §1.2 `S_Q + S_A` | 主线 1：quota admission dynamics | yes | yes | yes | no |
| §1.3 `S_C + S_A` | 主线 2：concurrency threshold，最干净 | yes | yes | yes | yes |
| §3 `S_Q + S_C + S_A` | 主线 3：joint routing，paper-facing | yes | no by default | yes | no |

默认不在 §3 joint 上跑 Phase B。原因是 Phase B 的 per-request noise sweep 会迅速放大 grid，而大部分趋势应先从 §1.2 和 §1.3 的 bilateral 曲线里解释。joint 只跑 Phase A + Phase C：

- Phase A 用来抓 wiring / scenario bug。如果极端 predictor 在 joint 里也不改变 tier mix，说明 joint setting 可能离决策边界太远。
- Phase C 用来给 paper 主图：real online predictors 在真实部署 setting 下相对 oracle 损失多少。

如果出现 bilateral 敏感但 joint 不敏感，或 bilateral 不敏感但 joint 敏感，再回头给 §3 补一小段 Phase B。否则默认不补。

建议的最小 grid：

| Phase | Scenarios | Predictor / curve settings | Runs |
|---|---:|---:|---:|
| A: sanity | §1.2 + §1.3 + §3 | 6 | 18 |
| B: noise trend | §1.2 + §1.3 | about 12 | about 24 |
| C: real predictors | §1.2 + §1.3 + §3 | 6 primary predictors | 18 |
| D: S_C shadow price | §1.3 only | 4 shadow curves x 3 predictor settings | 12 |
| Total | | | about 72 |

`z` / utilization stratification 是后处理切片，不需要重新跑实验。

### Phase A: Extreme Sanity Check

目标：确认 output prediction 确实接入 RouteWise routing。

推荐场景：

- workload：BurstGPT / ShareGPT trace
- scenario：见 §5.0，覆盖 §1.2、§1.3、§3
- quota count：§1.2 选 q* 附近，例如 q = 8, 12, 14, 16
- concurrency count：§1.3 选刚开始饱和的 count，例如 selected C* 附近
- joint count：§3 使用已选定的 paper main setting，或在 q* / C* 附近各取一个组合
- policy：RouteWise LP-only，优先 `p=0`；必要时加 `p=0.25`
- latency：固定一个代表性 distribution，例如 heavy-tail 或当前 §1.2 默认

predictor settings：

| Setting | 含义 | 预期 |
|---|---|---|
| `oracle` / truth | 使用真实 output length | baseline |
| `fixed:very_small` | 所有请求预测为很短 | 更多请求看起来便宜，API fraction 应上升 |
| `fixed:mean` | 所有请求预测为平均值 | request ranking 被抹平 |
| `fixed:very_large` | 所有请求预测为很长 | API 看起来很贵，quota fraction 应上升 |
| `scale:0.1` | `pred = 0.1 * truth` | 明显 under-predict |
| `scale:10` | `pred = 10 * truth` | 明显 over-predict |

通过标准：

- 极端 settings 必须让 `provider_mix` / `tier_mix` / `api_cost_usd` 明显变化。
- 如果极端 settings 也不变，先修实现或场景，不进入正式 ablation。

### Phase B: Noise / Scale Trend

目标：不是判断“敏感 / 不敏感”，而是展示 per-request prediction error 逐渐增大时 RouteWise 如何退化。

推荐场景：只跑 §1.2 `S_Q + S_A` 和 §1.3 `S_C + S_A`。默认不跑 §3 joint。

主 sweep 应该是 per-request noise，因为它会改变 request 之间的 value ranking：

```text
per-request uniform noise:
  predicted_i = true_i * Uniform(1-rho, 1+rho)

per-request lognormal noise:
  predicted_i = true_i * LogNormal(mu=-sigma^2/2, sigma=sigma)
```

推荐 `rho` / `sigma`：

```text
rho in {0.1, 0.2, 0.5, 1.0}
sigma in {0.1, 0.25, 0.5, 1.0}
```

global multiplicative scale 仍然保留，但降级为 sanity / calibration curve：

```text
scale k in {0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0}
predicted_i = k * true_i
```

biased cases：

- always under-predict：`predicted_i = 0.5 * true_i`
- always over-predict：`predicted_i = 2.0 * true_i`
- tail underestimate：若 `true_i > p90(true_output)`，则 `predicted_i = 0.5 * true_i`；否则使用 oracle。
- long-under / short-over：按 input length 分两段，长 input 段 `predicted_i = 0.5 * true_i`，短 input 段 `predicted_i = 2.0 * true_i`。

后处理增加 stratified analysis，不重跑实验：

- quota progress bucket：按 `z = consumed_quota / quota_capacity` 切成 early / middle / late。
- concurrency utilization bucket：按 `u = used_concurrency / concurrency_capacity` 切成 idle / moderate / saturated。
- boundary bucket：按 `abs(estimated_api_cost - shadow_price)` 切分，重点看接近 threshold 的 requests。
- output-length bucket：按真实 output length quantile 切分，检查误差是否主要伤害 long-output tail。

输出：

- x-axis：noise scale / multiplicative factor
- y-axis：
  - `relative_cost_vs_oracle`
  - `api_cost_usd`
  - `quota_fraction`
  - `api_fraction`
  - optional：routing disagreement rate vs oracle

核心 takeaway 应该是趋势：

> 小误差范围内 cost gap 很小；当 error 达到某个阈值后，provider mix 开始偏移，API cost 上升。

### Phase C: Real Predictor Comparison

目标：回答 paper 里的现实问题：不用 oracle truth，简单 online estimator 是否足够接近 oracle。

推荐场景：覆盖 §1.2、§1.3、§3。§3 是 paper-facing 主图，不能省。

primary predictors：

| Predictor | 说明 |
|---|---|
| `oracle` | truth output length，上限 |
| `constant_mean` | workload-level mean |
| `constant_p50` | workload median |
| `constant_p90` | conservative constant |
| `ema` | streaming EMA |
| `histogram:q50` | median histogram |

optional predictors：

| Predictor | 说明 |
|---|---|
| `histogram:q10` | aggressive low estimate |
| `histogram:q90` | high estimate |

如果资源允许，可增加 workload：

- BurstGPT / ShareGPT：chat workload，output variance 可能较大
- FreeInference：更混合的 API workload
- RedNote / enterprise：agentic or coding-assistant workload，input length 更长，可能更不敏感

这部分用于解释为什么某些 workload 可能确实不敏感：

- input token cost dominates
- output length distribution narrow
- quota not near decision boundary
- p / LP budget setting makes provider choice insensitive
- predictor bias acts like conservative gating

### Phase D: S_C Shadow-Price Sensitivity

目标：确认 output-length predictor 的结论不是当前 concurrency shadow-price 公式的偶然产物。

推荐场景：只跑 §1.3 `S_C + S_A`。不要和 §3 joint 混在一起，否则会把 quota shadow price、concurrency shadow price、output predictor 三个变量混成一团。

shadow-price curves：

| Curve id | Formula | 目的 |
|---|---|---|
| `conc_constant_l` | `lambda(u) = L` | 当前 code / paper baseline，predictor sensitivity 最干净 |
| `conc_legacy_linear_u` | `lambda(u) = U * u` | 旧 baseline；现有 effective-cost code 中的 raw curve 是 `util_linear_u`；用于解释原版约 3% 不敏感是否来自 shadow-price absorption |
| `conc_linear_lu` | `lambda(u) = L + u * (U - L)` | 与 quota envelope 对齐的 linear curve |
| `conc_exp_lu` | `lambda(u) = L * (U / L)^u` | 与 quota 同形的指数曲线，检查更强 shadow-price absorption |

predictor settings：

- `oracle`
- representative noisy predictor，例如 `lognormal:sigma=0.5`
- best real predictor from Phase C，例如 `ema` 或 `histogram:q50`

这部分不是为了替代 effective-cost ablation，而是为了给 output-length ablation 加一个条件边界：如果结论只在某个 S_C curve 下成立，paper 里必须写成 conditional statement。

## 6. Metrics 和 Artifact Schema

不能只看 total cost。至少输出：

### Cost metrics

- `api_cost_usd`
- `subscription_fixed_cost_usd`
- `total_cost_usd`
- `relative_cost_vs_oracle`
- `relative_cost_vs_offline`
- `oracle_gap_pct`

### Routing metrics

- `provider_mix`
- `tier_mix`
- `quota_fraction`
- `concurrency_fraction`
- `api_fraction`
- `routing_disagreement_vs_oracle`（可选）

### Predictor and stratification metrics

- `predictor_kind`
- `predictor_quantile`
- `noise_model`
- `noise_scale`
- `bias_mode`
- `fixed_value`
- `prediction_mae`
- `prediction_mape`
- `prediction_p50_error`
- `prediction_p90_error`
- `q10_coverage` / `q50_coverage` / `q90_coverage`（对 histogram / EMA）
- `shadow_price_curve`
- `quota_z_bucket`（后处理）
- `concurrency_u_bucket`（后处理）
- `near_boundary_bucket`（后处理）

### Reproducibility metadata

每一行 summary 必须包含：

- `workload`
- `scenario`
- `subscription_count`
- `concurrency_count`
- `p`
- `seed`
- `latency_family`
- `shadow_price_curve`
- `predictor_kind`
- `predictor_config`
- `noise_model`
- `noise_seed`

如果 predictor 信息只出现在 output directory 名字里，artifact 不够 paper-grade。

## 7. 建议代码结构

建议新建独立 ablation package，而不是把所有逻辑塞进 `experiments/simulation/cost_layer.py`：

```text
experiments/ablations/output_length_prediction/
  __init__.py
  predictors.py      # experiment-local fixed / scaled / noisy predictors
  presets.py         # scenario and sweep definitions
  harness.py         # CLI runner and artifact writer
  README.md          # runbook
```

CLI：

```bash
routewise ablation output-length --phase sanity
routewise ablation output-length --phase scale-sweep
routewise ablation output-length --phase predictors
routewise ablation output-length --phase sc-shadow-price
```

PR #1 里 Yiyan 做的 `--predictor` hook 可以作为底层能力复用，但完整 ablation 需要更高层的 harness 来定义 sweep、metadata 和 plots。

## 8. 和 Yiyan PR #1 的关系

Yiyan PR #1 已经补了关键入口：

- `RouteWisePolicy` 可以接收 `output_predictor`
- routing-time `S_A` effective cost 可以用 predicted output length
- `cost-layer` / `latency-layer` / `hedging` runner 增加 `--predictor`
- 支持 `oracle`、`ema`、`histogram`、`constant_pXX`、`fixed:N`

但它还不是完整 ablation：

- 没有 noise sweep。
- 没有 extreme sanity phase。
- 没有固定 experiment matrix。
- 没有 §1.3 / §3 的 scenario x phase 覆盖计划。
- 没有 S_C shadow-price sensitivity。
- summary / metadata 没有记录 predictor config。
- 没有 provider mix / quota usage 的专门对比图。
- `latency-layer` / `hedging` 上的 predictor flag 多数情况下不会显著改变结果，因为 same-cost providers 下 cost 排序不依赖 output length。

因此 PR #1 应被视为“wiring layer”，后续还需要 ablation harness 和 artifact schema。

## 9. 验收标准

完整实验至少应满足：

1. `oracle` / default truth 结果一致。
2. `fixed:very_small` 和 `fixed:very_large` 在 quota-sensitive scenario 中显著改变 `provider_mix` 或 `api_cost_usd`。
3. `fixed:very_small` 和 `fixed:very_large` 在 concurrency-sensitive scenario 中显著改变 `concurrency_fraction` 或 `api_cost_usd`。
4. per-request noise sweep 呈现连续趋势，而不是只给一个 2%-3% 的单点结论。
5. §3 joint 的 Phase A 能确认 predictor 在三层 routing 中仍然打到 critical path。
6. §3 joint 的 Phase C 给出 paper-facing real-predictor-vs-oracle 结果。
7. Phase D 说明 S_C 场景下的结论是否依赖当前 `lambda(u)` 公式。
8. EMA / histogram / constant 的比较能说明：
   - 是否接近 oracle；
   - 哪个 predictor 更稳；
   - predictor error 主要通过哪类 request 影响 cost。
9. 每个 artifact 行都能独立说明 predictor / noise / scenario / seed。
10. 至少一张 paper-facing figure 展示：
   - cost gap vs prediction error；
   - provider / tier mix vs prediction error；
   - real predictors vs oracle。

## 10. 推荐最终结论形式

最终 paper 里不要写：

> Output length prediction does not matter.

更稳的表述是：

> RouteWise is robust to moderate output-length prediction error in the tested quota-sensitive regimes. Extreme sanity checks confirm the predictor is on the routing critical path; as prediction error grows, routing mix shifts and cost increases. In realistic online predictors, EMA / histogram remains close to oracle because the workload's cost ranking is stable under moderate error.

如果 §1.3 / Phase D 显示结论依赖 concurrency shadow-price curve，需要把表述改成：

> Under the current `conc_constant_l` concurrency shadow-price baseline, RouteWise remains robust to moderate output-length prediction error. Sensitivity checks over legacy `U * u`, linear-envelope, and exponential-envelope S_C curves show whether this robustness is a property of the workload or of the current congestion-pricing formula.

这样既回应了 Juncheng 的反直觉质疑，也避免过度声明。

## 11. 参考依据

- Notion: `RouteWise -> Evaluation -> Simulation`
- Notion: `RouteWise -> Design`
- Meeting transcript: `~/Desktop/output/routewise_Feb27.txt`
- Meeting transcript: `~/Desktop/output/routewise_april17_named.txt`
- Meeting transcript: `~/Desktop/output/routewise_april20.txt`
- Meeting transcript: `~/Desktop/output/RouteWise_May5.txt`
- Meeting transcript: `~/Desktop/output/juncheng_zoom_may4.zoom.txt`
