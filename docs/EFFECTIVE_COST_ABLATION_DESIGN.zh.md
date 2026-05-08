# Effective Cost 消融实验设计

> 这是 RouteWise simulator 里用于验证 effective cost 背后 shadow-price
> 公式的消融实验设计文档。本文假设 cost-layer 1.1、1.2、1.3 都作为
> plan-backed simulator section 存在，并定义下一步实验 harness；这个
> harness 不改变主论文 cost-layer figure 的代码路径。

最后更新: 2026-05-07。

---

## 1. TL;DR

这个消融实验需要回答两个相互独立的问题：

1. **Quota 曲线选择。** 对于可消耗的 subscription quota，shadow price
   应该用论文里的 exponential 曲线，还是 linear 曲线？
2. **统一稀缺性公式。** Quota 和 concurrency 能否共用一个 scarcity-price
   函数，还是 reusable concurrency 需要一套单独的公式？

实现应放在一个集中的 ablation package 里，复用现有 simulator
orchestration，但不改变 production `RouteWisePolicy` 的接口。采用
**Method A**：ablation 自己拥有一个很小的 LP-only policy，只实现公式
sweep 所需的 cost-router 决策。

```text
experiments/
  simulation/
    cost_layer.py                  # 主论文 cost-layer 路径，不改

  ablations/
    effective_cost/
      curves.py                    # ablation 候选公式
      policy.py                    # LPOnlyAblationPolicy，无 hedging/explorer
      presets.py                   # curve/p sweep -> section-local presets
      harness.py                   # scenario/policy listing, CLI, run_cell
      oracle.py                    # 后续补 Stage Q / Stage QC adapter
      README.md
```

不要重新引入 additive effective cost。Provider tier 的语义继续保持
piecewise：

```text
S_A: c_eff = real API marginal cost
S_Q: c_eff = quota scarcity price
S_C: c_eff = concurrency scarcity price
```

第一步实现目标是 **Phase A quota-only**，使用现有 1.2 配置：

```text
q* = 16
latency_family = heavy_tail
workload = burstgpt
seed = 42
```

第一轮实验应先比较不同公式，并检查 routing behavior。Stage Q / Stage QC
oracle 对最终 regret 数字仍然重要，但不应该阻塞第一轮 formula sweep。

---

## 2. 研究问题

### Q1. 什么是合适的 quota scarcity curve？

Quota 是可消耗资源。如果 online router 在低价值请求上过早花掉 quota，
就会损失未来的 savings。当前论文公式使用 exponential shadow price：

```text
psi_exp(z; L, U) = L * (U / L)^z
```

其中：

- `z` 是当前 active quota window 中已经消耗的 quota 比例。
- `L` 是 API-equivalent request cost envelope 的下界。
- `U` 是 API-equivalent request cost envelope 的上界。

消融实验至少需要和一个 linear alternative 对比：

```text
psi_linear_lu(z; L, U) = L + z * (U - L)
```

判断标准不是哪条曲线看起来更漂亮。第一轮 sweep 的标准是：在固定 1.2
配置下，哪条曲线产生更合理的 cost、latency 和 quota-allocation
behavior。最终论文结果需要进一步用 quota oracle 的 regret 来支撑，并且
保持相同 latency distribution 和相同 purchased subscription plan。

### Q2. Quota 和 concurrency 能否共用同一个公式？

Quota 和 concurrency 不是同一类问题：

- Quota 在一个 window 内会被消耗掉。
- Concurrency 在每个请求完成后会被释放并复用。

因此，统一公式只是一个 hypothesis，不是先验假设。这个问题需要 joint
test，因为 isolated quota-only 和 concurrency-only 运行无法回答：当两类
资源同时存在时，它们之间的 ranking 是否仍然稳定。

Joint 问题可以写成：

```text
scarcity_price(curve, x, L, U)
```

其中：

```text
S_Q: x = z, quota fraction used
S_C: x = u, weighted concurrency utilization
```

如果同一个 `curve` 在 joint runs 里对两个 tier 都有效，我们可以 claim
一个统一 effective-cost rule。否则，论文应该分别报告 quota 和
concurrency 的曲线。

---

## 3. 当前准备状态

### 已准备好

当前 simulator 已经有构建 ablation 所需的核心组件：

- Cost-layer 1.1 有 API-only scenarios，并使用 workload-level cost envelope。
- Cost-layer 1.2 有 plan-backed quota scenarios，包括 Chutes 和
  MiniMax-style multi-window quota。
- Cost-layer 1.3 有 plan-backed Featherless weighted concurrency 的设计，
  但最终 concurrency 配置和结果还在进行中。
- `RouteWisePolicy` 现在使用显式 workload-level `(L, U)`，而不是
  per-request calibration。
- `effective_cost()` 按 provider tier piecewise 计算，并和论文结构一致。
- Summary rows 已经覆盖第一轮 quota-only sweep 所需的主要 cost/latency
  输出。

### Final Ablation 前还没准备好的部分

在 headline ablation results 之前，有三个 gap：

1. **Oracle gap.** `experiments/simulation/cost_layer.py` 里有一个本地
   `offline` runner，会做 greedy quota selection 和 first-fit concurrency
   packing。它可以作为 smoke test baseline，但不是 Stage Q / Stage QC lower
   bound。
2. **Concurrency gap.** 1.3 concurrency configuration 还没有完全 settle，
   所以 Phase B 和 Phase C 应该等 selected concurrency setup 可复现后再做。
3. **Preset gap.** 全局 `rwsim.policies.DEFAULT_PRESETS` 仍然暴露没有
   `cost_envelope` 的 RouteWise presets。Section-local simulator harness 会注入
   envelope，但 generic `build_policy("routewise")` 现在会失败。如果 ablation
   harness 使用 section-local presets，这不是 blocker；但在依赖 generic
   runner 前应该修好。

---

## 4. 实验设计

### Phase 0. Method A Harness Sanity

在比较公式前，先构建最小 Method A harness：它能在单个 quota-only scenario
上运行 curve-specific LP-only policy。

使用：

- `experiments/ablations/effective_cost/policy.py`
- `experiments/ablations/effective_cost/presets.py`
- `experiments/ablations/effective_cost/harness.py`

Ablation policy 不应该 subclass 或修改 `RouteWisePolicy`。它只实现：

1. 用 `scarcity_price()` 计算每个 provider 的 `c_eff`；
2. 计算 `B_p = c_min + p * (c_max - c_min)`；
3. 求解同样的 LP-only routing decision；
4. 用给定 seed sample primary provider。

Solver 实现锁定：复制当前 `RouteWisePolicy` 里的手写 LP enumerator，而不是
使用 `scipy.optimize.linprog`。这部分重复对 ablation 是可接受的，因为它能保留
production cost-router 的 tie-break、normalization 和 sampling 语义。实验目标是
测 curve 差异，不是测 solver 差异。

需要用 production formula 做 sanity test：当 ablation curve 是 `exp_lu` 时，
S_Q effective cost 应该和当前 RouteWise quota formula 一致。这样可以避免
sweep 实际测到的是不同 router，而不是不同 curve。

### Phase A. Quota-Only Curve Ablation

目标：回答 Q1。

配置：

- Scenario family：cost-layer quota only，复用成熟的 1.2 builder。
- Primary plan：`chutes`。
- Optional sensitivity：`minimax_subscription_plus`。
- Headline count：`q* = 16`。
- Latency：`heavy_tail`。
- Dataset：`burstgpt`，也就是 §1.2 Chutes main run 使用的 BurstGPT 30-day
  trace。
- Seed：`42`。
- Development smoke：相同配置加 `--max-requests`。

曲线：

| Curve id | Formula | Purpose |
|---|---|---|
| `quota_exp_lu` | `L * (U / L)^z` | Paper formula |
| `quota_linear_lu` | `L + z * (U - L)` | Main alternative |
| `quota_constant_l` | `L` | Sanity baseline：总是把 quota 看得很便宜 |
| `quota_constant_u` | `U` | Sanity baseline：总是把 quota 看得很贵 |

主要指标：

- `total_cost_usd_per_run`
- `api_cost_usd_per_run`
- `subscription_fixed_cost_usd_per_run`
- `tier_mix`
- `quota_fits_in_trace`
- quota/API routing split
- quota exhaustion time，如果 quota 被耗尽
- quota utilization over time
- API fallback 在 quota scarcity 前后的集中程度
- mean / p95 / p99 latency

派生分析：

- route 到 S_Q 的请求的平均 API-equivalent value
- route 到 S_A 的请求的平均 API-equivalent value
- S_Q-routed requests 在整体 request value distribution 里的 percentile rank
- quota depleted 后被迫走 S_A 的 high-value requests 数量

实现决策：选择 option (b)。这些 derived analyses 是基于正常 simulator outputs
和 per-request records 的 post-run script 或 notebook，不是第一版 harness 新增的
`summary.csv` 字段。不要为了第一轮 sweep 扩展 shared summary schema，也不要改
`experiments/simulation/common.py`。如果 records 不足以支持某个 derived view，
就把该 view 作为定性分析或下一步工作，第一版 paper-facing comparison 仍以
上面的 primary metrics 为准。

成功标准：

对于第一轮 quota-only result，如果 exponential curve 相比 linear 和 constant
baselines 能降低 API fallback / total cost，同时展示合理的 quota trajectory，
就可以认为它有初步合理性。它不能靠大量 unused quota 取胜，也不能靠前期把
quota 烧在 low-value requests 上、导致后期 high-value requests 被推到 S_A
来取胜。Tail latency 的恶化也不能大到抵消 cost 论点。

### Phase B. Concurrency-Only Curve Ablation

目标：在测试统一公式前，理解当前 concurrency formula 是否足够。

状态：等待 1.3 concurrency configuration 可复现。

配置：

- Scenario family：cost-layer concurrency only。
- Primary plan：`featherless_premium`。
- Counts：configured `subscription_counts`。
- Primary model：`sharegpt` mapped to `ge_70b`，因为它会完整占用一个
  Premium account 的 weighted capacity。
- Optional sensitivity：`qwen3-coder-30b` mapped to `24_34b`。
- Latency：保持现有 cost-layer `heavy_tail` default。

曲线：

| Curve id | Formula | Purpose |
|---|---|---|
| `conc_legacy_linear_u` | `U * u` | 旧版 RouteWise concurrency baseline |
| `conc_linear_lu` | `L + u * (U - L)` | 和 quota 相同的 linear shape |
| `conc_exp_lu` | `L * (U / L)^u` | Exponential unified candidate |
| `conc_constant_l` | `L` | Sanity baseline：过度使用 concurrency |

主要指标：

- `total_cost_usd_per_run`
- `api_cost_usd_per_run`
- `oracle_gap_pct`
- `tier_mix`
- `peak_used_concurrency_cost`
- `mean_concurrency_utilization`
- `concurrency_saturated_in_trace`
- selected concurrency count under each curve

成功标准：

选出的曲线应降低 API fallback cost，同时不能把 router 推入明显 saturated 的
concurrency 行为。因为 concurrency 是 reusable 资源，Phase B 只能作为证据，
不是 Q2 的最终答案。

### Phase C. Joint Quota + Concurrency Ablation

目标：回答 Q2。

状态：等待 Phase A 和 Phase B 都有稳定配置。

这个阶段是必须的。统一公式无法只靠 isolated quota-only 或 concurrency-only
runs 验证，因为真正的问题是：当 S_Q、S_C、S_A 都 feasible 时，router 是否
能正确给它们排序。

配置：

- 一个来自 Phase A candidate set 的 S_Q provider。
- 一个来自 Phase B candidate set 的 S_C provider。
- 相同 fixed cheap/mid/expensive S_A fallback ladder。
- 所有 tiers 使用相同 latency。
- 在 Phase A 和 Phase B 独立最优设置附近 sweep quota counts 和 concurrency
  counts。

候选 policies：

| Policy id | Quota curve | Concurrency curve | Question |
|---|---|---|---|
| `separate_best` | best Phase A curve | best Phase B curve | Online formulas 的上界参考 |
| `unified_exp_lu` | exponential | exponential | 一个 exponential curve 是否可行？ |
| `unified_linear_lu` | linear LU | linear LU | 一个 linear curve 是否可行？ |
| `current_paper` | exponential | legacy linear U | 当前 RouteWise 行为 |

主要指标：

- against Stage QC 的 `oracle_gap_pct`。
- `total_cost_usd_per_run`。
- tier mix：S_Q vs S_C vs S_A。
- selected `(quota_count, concurrency_count)`。
- 两类 scarce resources 的 utilization diagnostics。

成功标准：

只有当 unified formula 的 joint oracle gap 接近 `separate_best`，并且保持相同
selected capacity region 时，才可以认为统一公式可接受。如果它改变
selected plan/count，并导致 regret 上升，就不应该 claim unified formula。

---

## 5. 代码设计

### 5.1 Experiment-scoped curve helpers

添加一个小模块：

```text
experiments/ablations/effective_cost/curves.py
```

建议接口：

```python
ScarcityCurve = Literal[
    "exp_lu",
    "linear_lu",
    "legacy_linear_u",
    "constant_l",
    "constant_u",
]

def scarcity_price(curve: ScarcityCurve, x: float, *, L: float, U: float) -> float:
    ...
```

这个模块应该 deterministic 且容易 unit test。它不能 import provider、policy
或 simulator engine types。Candidate curves 在实验结果证明应该改变 stable
RouteWise formula 前，都留在 ablation package 里。

### 5.2 Method A policy boundary

不要把每个 candidate curve 都加入 `rwsim.policies` 作为 core policy surface。
Stable `rwsim` implementation 应保持 paper-current formula：

```text
S_Q: exp_lu
S_C: legacy_linear_u
```

不要给 `RouteWisePolicy` 添加 `effective_cost_fn` field、subclass hook 或
ablation-specific branch。Ablation policy 是一个独立的 cost-layer-only 工具，
应该留在 `experiments/ablations/effective_cost/` 内。

添加：

```text
experiments/ablations/effective_cost/policy.py
```

建议接口：

```python
@dataclass
class LPOnlyAblationPolicy:
    quota_curve: ScarcityCurve
    concurrency_curve: ScarcityCurve
    p: float
    cost_envelope: tuple[float, float]
    seed: int = 0
    profile_window_sec: float = 15 * 60

    def route(self, request, state):
        ...
```

这个 policy 有意省略 hedging 和 explorer feedback。它应保留与 production
LP-only RouteWise 一致的 rolling latency-profile objective，这样在 provider
configured latency distribution 相同的场景下，`p` sweep 仍然有意义。它只重复
clean formula ablation 所需的小段 LP-only cost-router 和 profile logic。

Policy construction 走 ablation-local 路径。`presets.py` 可以输出 curve/p
metadata，但 `harness.py` 应该在 materialize workload cost envelope 后，通过一个
小的 local builder 实例化 `LPOnlyAblationPolicy`。不要把
`LPOnlyAblationPolicy` 注册到 `rwsim.policies.DEFAULT_PRESETS`，也不要走 generic
`build_policy()` 路径。

不要在 `cost_layer.py` 里添加 ablation-specific branching。

### 5.3 Ablation harness

添加两个小模块和 harness：

```text
experiments/ablations/effective_cost/presets.py
experiments/ablations/effective_cost/harness.py
```

职责：

- 用现有 plan-backed builders 构建 Phase A quota-only scenarios。
- 优先使用 public `cost_layer.make_scenarios()` / `make_scenario()` API。如果
  不得不使用 private quota builder，在 harness 里包一层，不要让 private
  imports 到处扩散。
- 构建 curve-specific LP-only ablation presets。
- 调用 shared `run_section()` helper。
- 用和其他 simulator sections 相同的 shape 写出 `metadata.json`、
  `summary.csv`、`summary.json` 和 histograms。
- Stage Q / Stage QC oracle attachment 留给后续 `oracle.py` 步骤。
- 实现时可以在 `routewise_cli/main.py` 里添加一个很薄的 `ablation`
  subcommand group，并让它 delegate 到这个 harness。这是 user-facing CLI
  entry point，不是 production `rwsim` policy path 的改动。

建议 CLI：

```bash
routewise ablation effective-cost \
  --phase quota \
  --curve exp_lu \
  --curve linear_lu \
  --qstar 16 \
  --latency-family heavy_tail \
  --workload burstgpt \
  --p 0.5 \
  --seed 42 \
  --max-requests 1000

routewise ablation effective-cost \
  --phase joint \
  --quota-plan chutes \
  --concurrency-plan featherless_premium \
  --model sharegpt
```

### 5.4 Tests

在跑 full traces 前先加 unit tests：

```text
tests/unit/ablations/test_effective_cost_curves.py
tests/unit/ablations/test_effective_cost_policy.py
tests/unit/ablations/test_effective_cost_harness.py
```

最低覆盖：

- `scarcity_price()` 在 `x = 0`、`x = 0.5` 和 near exhaustion 时返回预期值。
- Current-curve ablation 的 S_Q effective cost 匹配 production
  `quota_shadow_price()`。
- `p` 只改变 LP budget，不改变 workload cost envelope。
- Curve-specific presets 显式传入 workload cost envelopes。
- 第一版 harness 的 Phase A scenario construction 只使用 S_Q + S_A。
- Headline Phase A config 固定为 `q*=16`、`heavy_tail`、`burstgpt`、
  `seed=42`。
- 测试不需要修改 `rwsim/policies/routewise.py` 或
  `experiments/simulation/common.py`。

---

## 6. 输出目录

使用独立 output directory：

```text
outputs/ablations/effective_cost/
  quota/
  concurrency/
  joint/
```

每个 phase 应输出：

```text
metadata.json
summary.csv
summary.json
ttft_histograms.json
ttft_histograms_by_seed.json
```

Plot code 放在：

```text
plots/ablations/effective_cost/
```

在 ablation result 稳定、并明确它会进入哪张 paper figure 或 appendix figure
之前，不要把这些图混进 `plots/cost_layer/simulator/`。

---

## 7. Yangsun Branch 处理

Yangsun 的 c1-c4 concurrency comparison 可以作为 Phase B context 保留，但
不应作为 implementation source of truth。

保留：

- scenario intuition
- 能暴露 “linear looks good at c4 because of implicit load balancing” 观察的
  generated results
- first-pass concurrency comparison 的 credit

废弃：

- 绕过论文 piecewise effective-cost structure 的 formula code
- additive effective cost
- 嵌在 `cost_layer.py` 里的 section-local one-off curve logic

如果有用，可以在新的 ablation harness 里重建他的 c1-c4 sweep，作为 legacy
sensitivity run。

---

## 8. 不在本次范围内

这个 ablation 不应包括：

- hedging
- explorer feedback
- live OpenRouter calls
- service-time-aware concurrency pricing
- S_C queueing policies
- real invoice reconciliation

Service-time-aware concurrency 是后续值得做的方向，但它需要 predicted
duration，因此会触碰 latency/value-estimation 轴线。这会混淆当前
effective-cost formula ablation。

---

## 9. Open Questions

1. Oracle objective 应该是 cost-only，还是 value-aware with predicted request
   savings？默认这次 ablation 应该用 cost-only，除非论文叙述明确转向
   value-aware routing。
2. Phase A headline grid 应包含哪些 quota plans：只用 Chutes，还是 Chutes
   加 MiniMax Plus？
3. Phase B 主表是否同时包含 `sharegpt` 和 `qwen3-coder-30b`，还是把第二个
   model 作为 sensitivity？
4. 接受 unified formula 时，什么 tolerance 算 “close to separate_best”：absolute
   cost delta、relative oracle gap，还是 selected count stability？
5. 全局 `DEFAULT_PRESETS` 是否应该有一个 safe default `cost_envelope`，还是强制
   generic runners 显式传入？
6. Phase A 是否应加入 May 5 讨论里提到的 square/root-style curve，还是第一版
   paper-facing grid 只保留 exp/linear/constants？

---

## 10. 建议实现顺序

1. 保留 `experiments/ablations/effective_cost/curves.py` 和 unit tests 作为
   formula surface。
2. 添加 `policy.py`，实现自包含的 Method A LP-only ablation policy。
3. 添加 `presets.py`，用于 curve/p sweep preset generation。
4. 添加 `harness.py`，生成 Phase A scenarios/presets，固定 `q*=16`、
   `heavy_tail`、`burstgpt`、`seed=42`。
5. 先用 `--max-requests` 跑 Phase A smoke，再跑完整 Phase A trace。
6. 分析 cost、quota trajectory、request-value allocation 和 latency tails。
7. 当 formula sweep 可复现后，添加带 Stage Q adapter 的 `oracle.py`。
8. 等 1.3 settle 后，添加 Phase B scenario/preset generation。
9. 添加 concurrency-only oracle adapter。
10. 跑 Phase B smoke 和完整 trace。
11. 添加 Phase C joint scenario generation 和 Stage QC oracle adapter。
12. 跑 Phase C smoke。
13. 最后再跑完整 joint grid，并产出 paper/appx plots。
