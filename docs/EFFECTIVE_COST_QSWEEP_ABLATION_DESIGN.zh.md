# Effective Cost Q-Sweep 消融实验设计

> 目标：验证 quota 稀缺曲线在不同 quota 稀缺度下的行为，而不是只在
> `q=16` 这个相对充裕的单点上比较成本。

最后更新：2026-05-07。

---

## 1. 实验问题

当前 `q=16` 的结果：

```text
constant_l < exp_lu < linear_lu << constant_u
```

但 `q=16` 对 Chutes/BurstGPT 30d 来说偏充裕（abundant），很多天的 quota
根本用不完。在充裕区间下，`constant_l` 永远把 quota 定价为 `L`，会更激进
地塞 quota；它略便宜不能说明它是合理的 online 稀缺定价。

本实验要回答：

```text
随着 quota 从稀缺（scarce）到充裕（abundant），
exp_lu 是否比 linear_lu / constant_l / constant_u 更稳定、更合理？
```

这才对应论文里 online knapsack / shadow-price 的 claim。

---

## 2. 固定配置

所有 cell 固定使用以下配置：

```text
phase = quota
subscription_plan = chutes
workload = burstgpt
trace = full 30d
latency_family = heavy_tail
seed = 42
p = 0
hedging = off
explorer = off
policy = LPOnlyAblationPolicy with rolling latency profile
```

不跑 `p` sweep。`p` sweep 只作为 sanity check，不进入本实验主结论。

---

## 3. Sweep 网格

Quota 数量：

```text
q ∈ {2, 4, 8, 12, 16}
```

曲线：

| 曲线 | 公式 | 含义 |
|---|---|---|
| `exp_lu` | `L * (U/L)^z` | 论文公式 |
| `linear_lu` | `L + z(U-L)` | 主要的线性对手 |
| `constant_l` | `L` | FCFS 风格 sanity baseline，quota 永远便宜 |
| `constant_u` | `U` | 保守 sanity baseline，quota 永远昂贵 |

总 cell 数：

```text
5 个 q × 4 条曲线 × 1 个 seed = 20 cells
```

---

## 4. 各 q 区间的预期行为

以 q 作为稀缺度坐标轴：

| q | 预期区间 | 为什么重要 |
|---:|---|---|
| 2 | 严重稀缺 | 应该惩罚过早烧掉 quota 的策略 |
| 4 | 稀缺 | 对 shadow-price 行为最强的检验 |
| 8 | 中等稀缺 | 排序变化最可能在这里出现 |
| 12 | 弱稀缺 | 接近拐点 |
| 16 | 偏充裕 | 之前的单点对照 |

预期定性结果：

- `constant_u`：跨整个网格都应该差，因为它没用足已经付了费的 quota。
- `constant_l`：在大 q 下可能看起来不错；但在小 q 下应该变脆，因为
  它会过早烧光稀缺的 quota。
- `linear_lu`：在中高 `z` 区间应该比 `exp_lu` 更保守，把更多请求推到
  API 回退。
- `exp_lu`：如果它能接近最低成本同时避免"用不足"和"过早过用"两种失败
  模式，那就是最 defensible 的曲线。

---

## 5. 主要指标

直接用 `summary.csv` 现有字段：

```text
total_cost_usd_per_run
api_cost_usd_per_run
subscription_fixed_cost_usd_per_run
quota_request_fraction
tier_mix
provider_mix
mean_ttft_ms
p99_ms
slo_violation_rate
trace_paper_grade
quota_fits_in_trace
```

成本是主结果。TTFT / P99 / SLO 是 sanity check，因为 provider 的延迟分布
是被有意固定的。

---

## 6. 衍生稀缺度指标

对每个 q，计算 binding-day fraction：

```text
cap_per_day(q) = q * 5000
binding_day_fraction(q) =
    request_count_day > cap_per_day(q) 的天数 / 30
```

这个量应该在跑完后由 plotting / 分析脚本算，**不要**写进 simulator 的
summary schema。

用途：

```text
binding_day_fraction ↑  =>  quota 越紧
                       =>  曲线选择的差异应该越明显
```

---

## 7. 主图

### 图 A：相对 `exp_lu` 的百分比差值热图

行：

```text
constant_l
exp_lu
linear_lu
constant_u
```

列：

```text
q = 2, 4, 8, 12, 16
```

每格的值：

```text
delta_pct(curve, q) =
  (total_cost(curve, q) - total_cost(exp_lu, q))
  / total_cost(exp_lu, q) * 100
```

含义：

- `0%`：跟论文公式持平。
- 正数 / 红色：比 `exp_lu` 贵。
- 负数 / 蓝色：比 `exp_lu` 便宜。

用百分比差值而不是原始美元，是因为 q 会改变固定费的尺度。这样可以避免
被 `constant_u` 这种大 outlier 把色条撑爆，从而掩盖中间格那些小但重要
的差异。

### 图 B：Binding-day Fraction 条形图

直接放在热图下方，跟热图共享 x 轴上的 q 值：

```text
x = q
y = binding_day_fraction(q)
```

它解释机制：

```text
quota 越紧 -> binding 的天数越多 -> 曲线选择的影响越显著
```

### 可选附录图

Quota 使用率热图：

```text
行 = curve
列 = q
值 = quota_request_fraction
```

只在主图的机制解释不够时才用。

---

## 8. 执行计划

首选实现：让 harness CLI 支持重复的 `--qstar`。

期望的 CLI：

```bash
routewise ablation effective-cost \
  --phase quota \
  --curve exp_lu \
  --curve linear_lu \
  --qstar 2 --qstar 4 --qstar 8 --qstar 12 --qstar 16 \
  --latency-family heavy_tail \
  --workload burstgpt \
  --p 0 \
  --seed 42 \
  --jobs 8 \
  --output-dir outputs/ablations/effective_cost_phaseA_qsweep_exp_linear
```

Parser 必须保证的语义：

- `--qstar` 必须可重复。重复出现时应该展开成多个 scenario，**不能**把
  整个列表当成一个 q 值。
- `--p 0` 必须只跑 `p=0`。Parser 应该保留当前的 fallback 语义：用户提供
  了 p 列表就严格用用户给的；没提供才用 `DEFAULT_P_VALUES`。`--p 0` 这种
  调用绝对不能意外触发默认 p sweep。

任务拆分：

```text
freeinference-gpu:
  q = {2,4,8,12,16} × curves = {exp_lu, linear_lu}

freeinference-gpu1:
  q = {2,4,8,12,16} × curves = {constant_l, constant_u}
```

每台 server 各跑 10 cells。`--qstar` 支持重复后，`--jobs 8` 才能真正在
这 10 个 cell 之间并行。

不要用 bash 在 server 端串行跑 5 个 q：每个 q 只有 2 cells，并发能力
被浪费。

必须配套的 harness 测试：

- 加一个 unit test 验证：重复传入 `--qstar` 会展开成每个 q 一个 scenario。
- 这个 test 要保护最终网格的 cell 数：
  `len(q_values) × len(curves) × len(seeds)`。
- 防止"q 列表被当成单个标量、最终只跑了 4 cells 而不是 20 cells"
  这种静默失败。

---

## 9. 画图前的 Sanity Checks

Rsync 回本地后，验证：

1. 每台 server 各产出 10 行 summary（不算 header）。
2. 合并后整张网格正好 20 行。
3. 所有行都满足：
   - metadata 里 `workload_dataset = burstgpt`
   - `seed = 42`
   - `latency_family = heavy_tail`
   - `p = 0`
4. 每个 q 上，`subscription_fixed_cost_usd_per_run` 应该约等于：

```text
20 * q * trace_days / 30
```

5. `constant_u` 的 quota 占比应该明显低于其他曲线。
6. `constant_l` 的 quota 占比应该明显高于其他曲线。
7. 同一个 q 下，不同曲线的 TTFT / P99 应该接近；如果偏差很大，说明
   routing 或 latency profile 出了 artifact，要先排查再画图。

---

## 10. 决策规则

**不要**用 `q=16` 这一个单点的结果决定哪条曲线胜出。

要看完整的 q-sweep：

- 如果 `exp_lu` 在 abundant q 下接近最优、在 scarce q 下更优 → 保留
  `exp_lu` 作为论文公式。
- 如果 `constant_l` 只在 q=16 赢，但在 q=2/4/8 上明显落后 → 把 q=16 的
  小赢解读成"abundant 区间下的 artifact"。
- 如果 `linear_lu` 在所有 q 下都比 `exp_lu` 贵 → 拒绝它作为 quota 主曲线。
- 如果某条非 exp 的曲线在所有 q 下都 dominant → 暂停结论，先去看 per-request
  的 quota 分配，再决定是否修改论文 claim。

论文的主 claim 应该是"跨稀缺度 robust"，而不是某一个充裕配置下的单点
比较。

---

## 11. 产物文件

原始输出：

```text
outputs/ablations/effective_cost_phaseA_qsweep_exp_linear/
outputs/ablations/effective_cost_phaseA_qsweep_constants/
```

合并后的分析 artifact：

```text
outputs/ablations/effective_cost_phaseA_qsweep_merged/
  summary.csv
  effective_cost_qsweep_percent_delta.csv
  effective_cost_qsweep_binding_days.csv
```

`effective_cost_qsweep_binding_days.csv` 的 schema：

```csv
q,binding_day_fraction
```

Plot 脚本：

```text
plots/ablations/effective_cost/plot_qsweep_heatmap.py
```

输出图：

```text
outputs/ablations/effective_cost_phaseA_qsweep_merged/figures/
  effective_cost_qsweep_percent_delta_heatmap.pdf
  effective_cost_qsweep_percent_delta_heatmap.png
  effective_cost_qsweep_quota_fraction_heatmap.pdf   # 可选
  effective_cost_qsweep_quota_fraction_heatmap.png   # 可选
```
