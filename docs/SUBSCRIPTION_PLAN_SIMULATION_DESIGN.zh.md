# Subscription Plan Simulation Design 中文版

> RouteWise simulator cost-layer §1.2 和 §1.3 的设计文档，覆盖
> quota-style 和 concurrency-style subscription plans。这个文档说明如何在
> simulator 里模拟 Chutes、MiniMax Starter / Plus / Max、Featherless 等订阅计划，
> 同时避免把“路由时的边际成本”和“论文里报告的账单成本”混在一起。

最后更新：2026-05-06。

---

## 1. TL;DR

Cost-layer §1.2 和 §1.3 应该模拟**具体的 subscription plan**，而不是匿名的
`quota_q1..q4` 或 `concurrency_c1..c4` 资源。

目标文件结构：

```text
experiments/
  subscription_plans.yaml          # 共享的 quota / concurrency plan facts
  subscriptions.py                 # 实验代码共享的 loader / dataclass

experiments/simulation/
  cost_layer.py                    # 生成参数化 plan runs
  common.py                        # provider builder + cost summary helpers
```

公开 CLI 应该保留一个 quota scenario 和一个 concurrency scenario，把产品 plan 和订阅数量作为显式参数：

```bash
routewise simulator cost-layer \
  --scenario quota \
  --subscription-plan chutes \
  --subscription-count 2

routewise simulator cost-layer \
  --scenario concurrency \
  --concurrency-plan featherless_premium \
  --concurrency-count 1
```

输出 artifact 可以包含展开后的参数，例如：

```text
quota__plan=chutes__n=2
```

但这只是输出 metadata / 文件名，不是 public scenario name。

核心设计规则：

```text
S_Q 路由成本：
  quota 内单个 request 的 marginal request cost = 0
  effective cost = quota shadow price ψ(z)

S_C 路由成本：
  capacity 可用时 marginal request cost = 0
  effective cost = concurrency shadow price λ(u_weighted) * concurrency_cost(model)

实验报告成本：
  total_cost_usd = api_cost_usd + subscription_fixed_cost_usd
```

所以 Chutes 和 Featherless 在购买后对路由来说仍然是 zero marginal cost，但最终 cost bar /
table 必须包含按 trace 时长 prorate 的 fixed subscription fee。

---

## 2. Problem

当前 cost-layer quota 和 concurrency scenarios 是泛化名字：

```text
cost_layer_quota_q1
cost_layer_quota_q2
cost_layer_quota_q3
cost_layer_quota_q4
cost_layer_concurrency_c1
cost_layer_concurrency_c2
cost_layer_concurrency_c3
cost_layer_concurrency_c4
```

它们的 capacity knobs 写死在 `experiments/simulation/cost_layer.py`。现在这个名字太含糊。

另一个极端也不好：把每个 product/count 组合展开成 public scenario name，例如：

```text
cost_layer_quota_chutes_q1
```

这样会把 `plan × n` 伪装成很多不同 experiment type，重新变成我们刚删掉的
grid-style API。实际上 `plan` 和 `n` 只是同一个 plan-backed experiment 的参数。

我们要回答的 paper question 是：

> 给定一个真实 subscription plan，RouteWise 应该买几个 subscription/account，
> 又应该怎么把 quota 分配给 workload 里的 requests？

对 §1.3，对应的问题是：

> 给定一个真实 concurrency subscription plan，RouteWise 应该买几个 subscription/account，
> 又应该让哪些 in-flight requests 消耗稀缺 concurrency capacity？

这个问题依赖 plan facts：

- quota size 和 reset window
- concurrency allotment 和 per-model concurrency cost
- monthly fee
- model/provider identity
- 这个 fee 是否已经确认到足以做 dollar-cost claim

这些 facts 应该只在一个小的 experiment config file 里声明一次，然后由
`cost_layer.py` 使用。

---

## 3. Conceptual Boundary

### 3.1 路由 marginal cost 不是账单 cost

如果一个 subscription plan 已经为当前实验窗口买好了，那么 quota 内多服务一个
request 不会产生额外 API invoice。因此 provider 的 marginal token price 仍然是 0：

```python
TieredProvider(
    tier=ProviderTier.S_Q,
    input_cost_per_token=0.0,
    output_cost_per_token=0.0,
    quota=QuotaState(...),
)
```

这对 dispatch 是正确的。

RouteWise 在路由时真正应该比较的是 quota 的**机会成本**，也就是 shadow price：

```text
c_eff(S_Q, request, z) = ψ(z; L, U)
```

其中：

- `z` 是 quota fraction used
- `L/U` 是 workload-level cost-envelope calibration values

### 3.2 报告成本必须包含 fixed fee

论文里的 cost bar / table 不能假装 subscriptions 是免费的。应该报告：

```text
api_cost_usd
subscription_fixed_cost_usd
total_cost_usd = api_cost_usd + subscription_fixed_cost_usd
```

对月费 subscription：

```text
billing_period_days = 30
subscription_fixed_cost_usd =
  monthly_fee_usd * num_subscriptions * (trace_days / billing_period_days)
```

这笔 fee 不管 RouteWise 有没有用完 quota 都要付。

### 3.3 为什么不把 subscription fee amortize 到每个 request？

不要在路由时给每个 Chutes request 加：

```text
$20 / (30 * 5000)
```

这样会混淆两个不同决策：

1. **Purchase decision**：我们应该买几个 subscription？
2. **Dispatch decision after purchase**：买完以后，哪些 request 值得用稀缺 quota？

在这个 simulator 里，`q1..q4` sweep 回答 purchase decision。固定某个 `q`
以后，RouteWise 的 dispatch decision 应该由 quota scarcity 决定，而不是给已经付过钱的 plan
再额外加一个假的 per-request fee。

---

## 4. Plan Configuration

新增：

```text
experiments/subscription_plans.yaml
```

这个文件放在 `experiments/` 下，而不是 `experiments/simulation/` 下，因为 simulator、
real-evaluation、旧 offline-stage experiments 都需要同一份 subscription facts。

它不应该放进 `rwsim/`，因为 `rwsim` 是 generic engine，不应该知道 Chutes/MiniMax
这种产品事实。它也不应该放进 `experiments/simulation/latency_profiles/`，因为这些是
billing/capacity facts，不是 latency samples。

已有的 subscription facts，例如
`experiments/offline_stage/configs/experiment.yaml` 里的 Chutes `$20/month` 和
`5000/day`，不能继续作为独立 source of truth。

迁移目标：

```text
experiments/subscription_plans.yaml        # canonical plan facts
experiments/subscriptions.py               # canonical loader/dataclass
experiments/offline_stage/...              # import canonical facts
experiments/real_evaluation/...            # import canonical facts
```

迁移期间，旧 YAML 可以暂时保留本地 experiment settings，但重复的 plan facts 必须在引入
canonical file 的同一个 PR 里移除。validator 只能作为移除重复字段时的临时 guardrail，
不能替代 single source of truth。

### 4.1 Schema

```yaml
plans:
  chutes:
    display_name: "Chutes"
    tier: quota
    billing_mode: subscription
    monthly_fee_usd: 20.0
    quota_windows:
      - {name: daily, quota_requests: 5000, quota_window_sec: 86400}
    subscription_counts: [1, 2, 3, 4, 5, 6, 8]
    eligible_sections: [cost_layer_quota, end_to_end]
    cost_claim_allowed: true
    source: "experiments/offline_stage/configs/experiment.yaml subscriptions.chutes"
    notes: "Chutes public plan: $20/mo for 5000 requests/day."

  minimax_subscription_starter:
    display_name: "MiniMax Starter"
    tier: quota
    billing_mode: subscription
    monthly_fee_usd: 10.0
    quota_windows:
      - {name: five_hour, quota_requests: 1500, quota_window_sec: 18000}
      - {name: weekly_allowance, quota_requests: 15000, quota_window_sec: 604800}
    subscription_counts: [1, 2, 3, 4]
    eligible_sections: [cost_layer_quota]
    cost_claim_allowed: true
    source: "User-provided MiniMax pricing screenshot, 2026-05-06"
    notes: "$10/mo; 1500 model requests / 5h; weekly allowance is 10x the 5-hour quota."

  minimax_subscription_plus:
    display_name: "MiniMax Plus"
    tier: quota
    billing_mode: subscription
    monthly_fee_usd: 20.0
    quota_windows:
      - {name: five_hour, quota_requests: 4500, quota_window_sec: 18000}
      - {name: weekly_allowance, quota_requests: 45000, quota_window_sec: 604800}
    subscription_counts: [1, 2, 3, 4]
    eligible_sections: [cost_layer_quota]
    cost_claim_allowed: true
    source: "User-provided MiniMax pricing screenshot, 2026-05-06"
    notes: "$20/mo; 4500 model requests / 5h; weekly allowance is 10x the 5-hour quota."

  minimax_subscription_max:
    display_name: "MiniMax Max"
    tier: quota
    billing_mode: subscription
    monthly_fee_usd: 50.0
    quota_windows:
      - {name: five_hour, quota_requests: 15000, quota_window_sec: 18000}
      - {name: weekly_allowance, quota_requests: 150000, quota_window_sec: 604800}
    subscription_counts: [1, 2]
    eligible_sections: [cost_layer_quota]
    cost_claim_allowed: true
    source: "User-provided MiniMax pricing screenshot, 2026-05-06"
    notes: "$50/mo; 15000 model requests / 5h; weekly allowance is 10x the 5-hour quota."

  featherless_premium:
    display_name: "Featherless Premium"
    tier: concurrency
    billing_mode: subscription
    monthly_fee_usd: 25.0
    concurrency_allotment: 4
    model_concurrency_costs_by_class:
      le_15b: 1
      24_34b: 2
      ge_70b: 4
    default_model_class: ge_70b
    model_class_overrides:
      llama-3.3-70b-instruct: ge_70b
      qwen3-coder-30b: 24_34b
      llama-4-scout: le_15b
    subscription_counts: [1, 2, 3, 4]
    eligible_sections: [cost_layer_concurrency, end_to_end]
    cost_claim_allowed: true
    source: "Featherless docs: Plans and Concurrency Limits, checked 2026-05-06"
    notes: "Concurrency 是 weighted capacity，不是 request count。Premium allotment=4，所以一个 cost=4 的 70B request 会占满这个 plan。"

```

### 4.2 字段语义

| Field | Meaning |
|---|---|
| `monthly_fee_usd` | 固定 subscription fee。`null` 表示可以模拟 routing/utilization，但不能做 total-cost dollar claim。 |
| `quota_windows` | 一个或多个 quota constraints。request 只有在所有 window 都有剩余额度时才能使用该 plan。Chutes 有 daily window；MiniMax 普通订阅有 5-hour quota 和 weekly allowance 两层约束。 |
| `concurrency_allotment` | 一个 subscription/account 的 weighted concurrency capacity。Featherless Premium 的 allotment 是 `4`，这不是四个任意 requests。 |
| `model_concurrency_costs_by_class` | 按 model-size class 定义 weighted capacity cost。Featherless 文档里的 class 包括 `le_15b=1`、`24_34b=2`、`ge_70b=4`。 |
| `default_model_class` | trace model ID 没有匹配到 override 时使用的 fallback class。§1.3 paper smoke 应该使用保守默认值，例如 `ge_70b`，不要静默把 request 从 S_C drop 掉。 |
| `model_class_overrides` | 可选的 workload/provider-specific mapping，把 trace model IDs 映射到 model-size classes。OpenRouter 风格 ID 和 trace aliases 不应该进入 core capacity state。 |
| `subscription_counts` | 论文里允许 sweep 的 subscription/account 数量。不要跑会让 quota 完全不稀缺的 count。 |
| `eligible_sections` | 这个 plan 可以被哪些 experiment section 使用。§1.2 主图使用 tagged `cost_layer_quota` 的 plans；§1.3 使用 tagged `cost_layer_concurrency` 的 plans。 |
| `cost_claim_allowed` | 这个 plan 是否允许在 paper figure 里报告 `total_cost_usd`。 |
| `source` | 人可以 audit 的来源。 |

---

## 5. Scenario Design

### 5.1 一个 scenario，显式参数

保持 public scenario surface 小而清楚：

```text
--scenario quota
--subscription-plan <plan_id>
--subscription-count <n>
```

例子：

```bash
routewise simulator cost-layer --scenario quota --subscription-plan chutes --subscription-count 1
routewise simulator cost-layer --scenario quota --subscription-plan chutes --subscription-count 4
routewise simulator cost-layer --scenario quota --subscription-plan minimax_subscription_plus --subscription-count 2
```

这比两种旧设计都好：

- `cost_layer_quota_q1` 隐藏了到底在测哪个产品 plan。
- `cost_layer_quota_chutes_q1` 把参数 sweep 展开成很多 scenario name，重新制造 grid-style API。

run output 仍然应该把 resolved 参数写入 metadata 和文件名，例如：

```text
scenario = "quota"
subscription_plan = "chutes"
subscription_count = 2
artifact_label = "quota__plan=chutes__n=2"
```

### 5.2 Subscription count sweep

对每个 plan：

```text
q1 = 1 subscription/account
q2 = 2 subscriptions/accounts
q3 = 3 subscriptions/accounts
q4 = 4 subscriptions/accounts
```

每个 reset window 的 quota capacity：

```text
quota_requests_per_window = quota_window.quota_requests * q
quota_window_sec = quota_window.quota_window_sec
```

这比现在 generic `_QUOTA_SIZE_PER_PROVIDER` 更真实。比如：

- Chutes `q1` = `5000/day`
- MiniMax Starter `q1` = `1500/5h` 且 `15000/week`
- MiniMax Plus `q1` = `4500/5h` 且 `45000/week`
- MiniMax Max `q1` = `15000/5h` 且 `150000/week`

quota 会按每个 configured quota window reset。比如 30-day Chutes run 里，`q=1`
大约等于 30 个连续 window，每个 window 有 5000 requests。simulator 的
quota state 负责 deterministic reset。MiniMax 普通订阅有两层约束：5-hour quota
和 weekly allowance，二者都必须有剩余额度，request 才能走这个 subscription plan。

跑 paper figure 之前，section 应该按 quota window 对 requests 分桶，判断每个 window
里 quota 是否都足够。不要只看一个月总容量，因为 bursty trace 可能某一天爆掉，即使整个月总容量看起来够。

```text
window_id(request, quota_window) =
  floor((request.timestamp - trace_start) / quota_window.quota_window_sec)
request_count_by_window = count requests per (quota_window, window_id)
quota_capacity_per_window = quota_window.quota_requests * q
```

设置：

```text
quota_saturated_in_trace =
  all(
    request_count_by_window[quota_window, w] <= quota_capacity_per_window(quota_window)
    for every quota_window and every window w
  )
```

如果 `quota_saturated_in_trace=true`，说明这个 run 没有真正测试 quota scarcity：
RouteWise 可以在每个 reset window 里几乎把所有 request 都发到 quota。这样的结果不应该进主 paper
q-sweep 图。

这也是为什么每个 plan 有自己的 `subscription_counts`。Chutes 可以用 `[1, 2, 3, 4, 5, 6, 8]`，
某些 MiniMax tier 可能更少的 count 就已经 saturate。

CLI 支持单值：

```bash
--subscription-count 2
```

也支持 sweep：

```bash
--subscription-counts 1,2,3,4,5,6,8
```

full paper run 也可以接受多个 plan：

```bash
--subscription-plans chutes,minimax_subscription_plus
```

这些都是 cost-layer section 的参数，不需要变成通用 `rwsim` engine 概念。

### 5.3 每个 scenario 的 provider set

所有 §1.2 quota scenarios 都使用 `latency_family = heavy_tail`，也就是
simulator 里的 LogNormal latency family。§1.2 是 cost-layer quota scarcity
实验，所以 distribution robustness 是主 subscription-count sweep 之后的
follow-up check，不放进主实验 grid。

每个 quota run 应该包含：

```text
1 aggregate S_Q provider，代表选中 plan 的 q 个 subscription/accounts
一组固定的 S_A fallback providers，在所有 q sweep 里保持不变
```

默认形态：

```text
所有 q 值：1 aggregate S_Q provider + 同一组 S_A fallback providers
```

默认不要把 q 个相同 subscription accounts 建模成 q 个独立 providers。等价 providers 会让
provider fractions 变得任意且 noisy，也会让 goldens 更不稳定。

对 §1.2，`q` 表示 aggregate purchased quota：

```text
single-window plan:
  QuotaState(size = q * quota_window.quota_requests, window_sec = quota_window.quota_window_sec)

multi-window plan:
  one aggregate quota counter per quota window
```

对有多个 quota windows 的 plan，使用 composite quota state：每个 window 一个 aggregate counter。
一个 request 会同时消耗所有 configured windows 的 1 个额度。如果 simulator 目前只支持 single-window
`QuotaState`，MiniMax Starter / Plus / Max 必须先 reject，直到 composite quota support 落地。

如果之后某个 ablation 真的要研究 per-account behavior，再显式加。

aggregate 模型假设多个 accounts 同步 reset。对 Chutes，这符合当前 simulator setup：
同一时间买同一种 plan，同一个 daily reset boundary。错峰账号属于 out of scope。如果未来要做 staggered
reset ablation，再把账号建模成多个独立 `QuotaState`。

§1.2 里的 aggregate-q abstraction 对 cost analysis 是合理的。real evaluation 里如果选中
`n=k`，部署上会体现成 k 个真实 provider accounts 或 API keys。这是 deployment detail，不是
routing-model change。

fallback API providers 使用 §1.1 已经定义的 cost-layer API price ladder，除非某个 section
明确要测试 real-world API price set。

### 5.4 §1.2 不做 concurrency

§1.2 是 quota-only。它只把明确写在 plan 里的 request quota window 当作 binding
constraint。

这样 §1.2 聚焦在 paper question：

```text
给定 subscription quota 和 monthly fee，RouteWise 应该买几个 subscription，
又应该把稀缺 quota 分给哪些 requests？
```

Concurrency 由下面的 §1.3 处理，不应该作为隐藏的额外约束混进 §1.2 quota sweep。

### 5.5 Concurrency subscription §1.3

§1.3 是 §1.2 的 concurrency-plan 对应版本。它不应该继续复用旧 public scenario names
`cost_layer_concurrency_c1..c4`。public shape 应该是：

```text
--scenario concurrency
--concurrency-plan <plan_id>
--concurrency-count <n>
```

例子：

```bash
routewise simulator cost-layer --scenario concurrency --concurrency-plan featherless_premium --concurrency-count 1
routewise simulator cost-layer --scenario concurrency --concurrency-plan featherless_premium --concurrency-count 4
```

run output 仍然应该写入 resolved 参数：

```text
scenario = "concurrency"
concurrency_plan = "featherless_premium"
concurrency_count = 1
artifact_label = "concurrency__plan=featherless_premium__n=1"
```

每个 §1.3 run 包含：

```text
1 aggregate S_C provider，代表 n 个 selected plan subscriptions/accounts
每个 n 使用同一组 S_A fallback providers
```

aggregate S_C capacity 是 weighted capacity：

```text
concurrency_capacity = plan.concurrency_allotment * n
request_model_class = resolve_model_class(request.model, plan.model_class_overrides, plan.default_model_class)
request_concurrency_cost = plan.model_concurrency_costs_by_class[request_model_class]
used_concurrency_cost = sum(active_request.concurrency_cost)
available iff used_concurrency_cost + request_concurrency_cost <= concurrency_capacity
```

这是 §1.3 最重要的 modeling point。Featherless `concurrency_allotment=4`
不表示任意 model 都能同时跑四个 request。一个 `concurrency_cost=4` 的 70B request
会独占一个 Premium subscription。四个 cost-1 small-model requests 也可以同时跑。
mixed requests 只有在 summed `concurrency_cost` 不超过 allotment 时才可行。

Routing 使用和论文一致的 piecewise effective-cost 语义：

```text
c_eff(S_A, r) = API token cost
c_eff(S_Q, r, z) = ψ(z)                    if quota is available, else ∞
c_eff(S_C, r, u_weighted) = λ(u_weighted) * concurrency_cost(r.model)
                  if weighted capacity is available, else ∞

u_weighted = used_concurrency_cost / concurrency_capacity
```

这些项是不同 providers 的 alternatives，不是加在同一个 provider 上的 additive penalties。
在 concurrency-only §1.3 run 里，RouteWise 比较 S_C effective cost 和 S_A API cost。
在后续 joint end-to-end run 里，S_Q 和 S_C 仍然是分开的 candidate providers；
不要计算 `API cost + quota shadow price + concurrency shadow price`。

online S_C effective cost 不乘 predicted request duration。simulator 仍然需要 observed 或
sampled service time 来知道 capacity 何时释放，offline scheduling 也仍然有 processing time。
这些是 state-evolution facts，不是 online effective-cost factor。

实现应该放在 `rwsim`，不是 `experiments/simulation`。`rwsim/world/capacity.py`
应该负责 weighted concurrency state：

```text
capacity_units
used_concurrency_cost
active requests keyed by finish time
admit(request_model, finish_time) -> bool
release_finished(current_time)
```

`experiments/simulation` 只负责 resolve plan，构造
`TieredProvider(tier=S_C, concurrency=...)`，然后启动 sweep。section runner
应该拒绝缺少 `concurrency_allotment` 的 concurrency plan、拒绝没有
`concurrency_cost` 的 resolved class。未知 trace model IDs 应该 resolve 到
`default_model_class` 并写 warning metadata；真正 incompatible 的 model classes 才应该拒绝进入 S_C。

第一版 §1.3 只做 cost-layer S_C provider 的 zero-queue / immediate-admission。
queueing policy 放到后面的 end-to-end experiments，因为那时 SLO 和 latency behavior
才是 paper question 的一部分。

§1.3 主图只应该使用 `featherless_premium`。不要把
`experiments/offline_stage/configs/experiment.yaml` 里的 legacy `featherless_scale`
迁移到 `experiments/subscription_plans.yaml`，也不要给它挂
`eligible_sections: [cost_layer_concurrency]`：它同时混用了一个当前公开页面不存在的
`$75 / C=8` plan，以及 70B-class `concurrency_cost=2`；而 Featherless docs 里
70B-class model cost 仍然是 `4`。更高 effective concurrency 已经被 Premium count sweep 覆盖：

| Setting | Weighted capacity | 70B cost | Effective 70B concurrency |
|---|---:|---:|---:|
| `featherless_premium`, `n=1` | 4 | 4 | 1 |
| `featherless_premium`, `n=2` | 8 | 4 | 2 |
| `featherless_premium`, `n=3` | 12 | 4 | 3 |
| `featherless_premium`, `n=4` | 16 | 4 | 4 |


---

## 6. Metrics and Artifacts

### 6.0 Metric ownership

分清三层：

```text
PerRequestRecord  -> per-request engine facts，不包含 fixed subscription fee
Run               -> engine aggregate over records，不包含 fixed subscription fee
SectionSummary    -> paper-facing row，包含 fixed subscription fee
```

`subscription_fixed_cost_usd` 不是单个 request 的属性。它是某个 purchased plan 在某个 trace span
上的 paper/section aggregate。因此它应该放在 section summary / artifact layer，而不是塞进每个
`PerRequestRecord`。

### 6.1 必须输出的 cost fields

section output 应该暴露：

```text
api_cost_usd
subscription_fixed_cost_usd
total_cost_usd
subscription_cost_known
trace_paper_grade
quota_saturated_in_trace
```

规则：

- `api_cost_usd` 是所有走 on-demand `S_A` API 的 request cost 总和。
- `subscription_fixed_cost_usd` 是 prorated fixed fee。
- `total_cost_usd` 只有在所有 active plans 都有 `cost_claim_allowed=true` 时才能做 paper claim。
- `subscription_cost_known=false` 用于未来 monthly fee 尚未确认的 plan；MiniMax Starter / Plus / Max 当前按截图价格视为 known。
- `trace_paper_grade=false` 表示这个 smoke run 太短，不适合做 fixed-fee conclusion。
- `quota_saturated_in_trace=true` 表示主 paper q-sweep plots 应该排除该 run，因为 quota scarcity 没被测试到。

对 concurrency-plan runs，额外输出：

```text
concurrency_capacity_units       # integer weighted capacity units
peak_used_concurrency_cost       # integer weighted capacity units
mean_concurrency_utilization     # ratio in [0, 1]
concurrency_saturated_in_trace   # boolean
```

所有 concurrency metrics 都基于 weighted capacity units，不是 raw in-flight request count。

### 6.2 现有 `cost_usd` 字段

engine-level `Run` 可以保留内部 `cost_usd` 字段。paper-facing section summary 不应该暴露含糊的
`cost_usd` 列。

```text
engine Run.cost_usd:
  internal simulator API cost

section summary:
  api_cost_usd
  subscription_fixed_cost_usd
  total_cost_usd
```

不要把 `cost_usd` 悄悄改成包含 fixed fee。如果现有 section summary 当前写 `cost_usd`，
subscription-plan migration 应该做 schema-breaking change：把它重命名为 `api_cost_usd`，
并重新生成相关 goldens。

### 6.3 Trace duration

fixed fee 要按 workload span prorate：

```text
trace_days = (last_request_timestamp - first_request_timestamp) / 86400
```

如果使用 `--max-requests` 跑 smoke，就报告该 smoke trace 的真实跨度。不要把 100k laptop smoke
假装成 full one-month experiment。

full paper run 应该在大机器上跑完整 BurstGPT month，不加 `--max-requests`。

现实里 monthly fee 是完整支付的。simulator 用 `trace_days / billing_period_days` prorate，只是为了让不同
trace span 之间可比。一个完整 30-day BurstGPT run 会付完整 monthly fee；5-day smoke 只报告
`5/30` prorated cost，并且应该设置 `trace_paper_grade=false`。

建议 paper-grade check：

```text
trace_paper_grade =
  trace_days >= 5 * min(quota_window.quota_window_sec for quota_window in plan.quota_windows) / 86400
```

这里的 `5` 是 heuristic：至少跨 5 个 reset windows，让 shadow price 经历多个 capacity cycles，
避免只报告单个 window 的 startup artifact。主 paper number 仍然应该用完整 BurstGPT month。

如果 smoke run 短于某个 plan quota window，它会看到该 window 的 quota 都立即可用。
simulator 不模拟 within-window rate pacing。smoke 只用来 sanity check routing decision，不用于
capacity-pacing claims。

---

## 7. Offline Baseline

Offline 仍然是 routing baseline，不是一个“没有 subscription 成本”的世界。

对包含 subscription plans 的 scenarios：

- Offline 可以知道完整 workload，并选择哪些 requests 使用 quota。
- Offline 必须为同一个 `q` scenario 支付和 online policies 相同的 fixed subscription fee。
- Offline 应该比较：
  - quota allocation quality
  - API fallback cost
  - 包含 fixed plan fee 的 total cost

这样比较才公平：

```text
same purchased capacity, different routing/allocation policy
```

fixed fee 跟 utilization 无关。不允许把 subscription fee 乘以 “fraction of quota used”。
没用满 subscription 本身就是 purchase-count sweep 要惩罚的事情。

未来 joint scenarios 可能组合多个 subscription plans。那时每个 plan 的 fixed fee 独立计算并求和：

```text
subscription_fixed_cost_usd =
  Σ_plan monthly_fee_usd(plan) * count(plan) * (trace_days / billing_period_days)
```

---

## 8. Implementation Plan

### Phase 0：前置条件，workload-level cost envelope

不要在 per-request `L/U` calibration 上实现 §1.2。

RouteWise 必须接收 workload-level cost envelope：

```text
L, U = P10/P90 of cheapest-API request cost over the workload
```

并且同一个 run 内所有 requests 使用同一组 `L/U`。这保证 quota shadow price `ψ(z)` 表达的是
“quota scarcity”，而不是“当前 request 的 cost scale”。

这就是 effective-cost fix：把 per-request `L/U` 替换成整个 workload 上 cheapest-API request cost 的
`(P10, P90)`，同一个 run 里的每个 request 都看到相同 envelope。

### Phase 1：Plan config 和 provider construction

1. 新增 `experiments/subscription_plans.yaml`。
2. 新增 shared loader，例如：

```python
def load_subscription_plans(path: Path | None = None) -> dict[str, SubscriptionPlan]:
    ...
```

3. 在 `experiments/subscriptions.py` 新增 shared dataclass：

```python
@dataclass(frozen=True)
class SubscriptionPlan:
    plan_id: str
    display_name: str
    tier: Literal["quota", "concurrency"]
    monthly_fee_usd: float | None
    quota_windows: tuple[QuotaWindow, ...]
    concurrency_allotment: int | None
    model_concurrency_costs_by_class: Mapping[str, int]
    default_model_class: str | None
    model_class_overrides: Mapping[str, str]
    subscription_counts: tuple[int, ...]
    eligible_sections: tuple[str, ...]
    cost_claim_allowed: bool
    source: str
    notes: str = ""
```

4. 更新 `make_quota_provider()`，让它可以接收 plan：

```python
def make_quota_provider(
    name: str,
    *,
    quota_size: int | None = None,
    plan: SubscriptionPlan | None = None,
    subscription_count: int = 1,
    ...
) -> TieredProvider:
    ...
```

如果同时传 `plan` 和 `quota_size`，抛 `ValueError`。`quota_size` 是非 plan 实验的 legacy/manual path；
plan-backed §1.2 runs 必须从 selected `SubscriptionPlan` 推导 quota size。

此阶段 `TieredProvider` 仍然保持 generic。plan facts 可以放在 provider metadata 或 section-level scenario
metadata 里。不要把 Chutes-specific fields 加进 `rwsim`，除非真的出现 generic need。

`subscription_count` 应该 aggregate capacity 成一个 provider：

```text
single-window plan:
  QuotaState(size = subscription_count * quota_window.quota_requests)

multi-window plan:
  one aggregate quota counter per quota window
```

它不是 provider index。

### Phase 2：Parameterized quota scenario

更新 `cost_layer.py`：

```python
def _make_quota_scenario_for_plan(plan_id: str, subscription_count: int) -> ScenarioConfig:
    plan = load_subscription_plans()[plan_id]
    providers = [
        make_quota_provider(
            f"{plan_id}_quota",
            plan=plan,
            subscription_count=subscription_count,
        )
    ]
    ...
```

新增 section CLI flags：

```text
--scenario quota
--subscription-plan chutes
--subscription-plans chutes,minimax_subscription_plus
--subscription-count 2
--subscription-counts 1,2,3,4,5,6,8
```

CLI 内部把这些参数列表展开成 run cells，但不暴露生成的 scenario catalogue。

requested counts 必须属于 `plan.subscription_counts`。如果超出范围，直接报错，并提示 allowed set。

这个 path 落地时，同一个 PR 删除 generic public `cost_layer_quota_q1..q4` scenarios。除非 golden migration 真的需要短期 internal test-only path，否则不要保留 public legacy alias。

### Phase 3：Cost summary fields

在 section summary path 里加入 fixed-fee accounting。

推荐实现位置是 `experiments/simulation/common.py`，因为每个 section runner 都会经过它。函数应该拿到：

```text
scenario
run parameters (subscription_plan, subscription_count)
policy
seed
trace_start_ts
trace_end_ts
Run summary
```

并追加：

```text
api_cost_usd
subscription_fixed_cost_usd
total_cost_usd
subscription_cost_known
trace_paper_grade
quota_saturated_in_trace
```

这应该在 shared section runner 里做一次，不要每个 policy 各自实现。

### Phase 4：Smoke 和 full runs

laptop smoke：

```bash
uv run routewise simulator cost-layer \
  --scenario quota \
  --subscription-plan chutes \
  --subscription-counts 1,2,4 \
  --policy greedy_cost \
  --policy offline \
  --policy ablation_lp_only_p0 \
  --policy ablation_lp_only_p25 \
  --policy ablation_lp_only_p50 \
  --workload burstgpt \
  --max-requests 100000 \
  --seed 42 \
  --jobs 8 \
  --output-dir outputs/simulation/cost_layer_1_2_chutes_smoke
```

full paper run：

```text
Run on gpu1/gpu2, full BurstGPT month, no --max-requests.
```

full run 只在 smoke 满足以下条件后启动：

- quota allocation favors high-value requests
- `api_cost_usd` 随 subscription count 增加而下降
- `subscription_fixed_cost_usd` 随 subscription count 增加而上升
- `total_cost_usd = api_cost_usd + subscription_fixed_cost_usd`
- offline 对同一个 `q` 支付和 RouteWise 一样的 fixed fee

### Phase 5：Parameterized concurrency scenario (§1.3)

§1.2 落地后再加 §1.3：

1. 扩展 `experiments/subscription_plans.yaml` 和 `experiments/subscriptions.py`，
   加入一个窄版 `featherless_premium` concurrency plan：`concurrency_allotment`、
   `model_concurrency_costs_by_class`、`default_model_class` 和可选 `model_class_overrides`。
2. 在 `rwsim/world/capacity.py` 增加 weighted concurrency state；不要在 experiment scripts
   里实现 Featherless-specific logic。
3. 在 `experiments/simulation/common.py` 增加 provider builder，把 concurrency plan/count
   转成一个 aggregate S_C provider。
4. 在 `experiments/simulation/cost_layer.py` 增加 `--scenario concurrency`、
   `--concurrency-plan` 和 `--concurrency-count(s)`。
5. 在同一个 PR 删除旧 public `cost_layer_concurrency_c1..c4` scenarios。如果 golden migration
   需要 alias，只保留 internal test-only alias。
6. fixed-fee accounting 仍然只放在 section-summary layer，和 quota plans 一样。

第一版 paper-grade smoke 可以把所有 §1.3 workload models 都映射到保守的 `ge_70b` class。
把 class resolver 扩展成完整 provider-specific model catalogue 是 follow-up，不阻塞最初的
Featherless Premium result。

---

## 9. Expected Results

对 Chutes：

- 增加 `q` 应该降低 API fallback cost。
- 增加 `q` 也会增加 fixed subscription cost。
- 因此 total cost 应该存在一个 optimum，不一定在 `q4`。
- RouteWise 应该比 greedy quota-first 更倾向于把稀缺 quota 分配给更大的、高价值 requests。
- Offline 应该是在同样 purchased capacity 下的 lower bound。

对 MiniMax Starter / Plus / Max：

- monthly fee 和 quota 都来自用户提供的 pricing screenshots，因此可以和 Chutes 一样进入 cost-layer sweep。
- Starter 和 Plus 可以先跑 `[1, 2, 3, 4]`；Max 的 quota 大很多，先跑 `[1, 2]`，避免高 count 直接让 workload saturate。
- 任何 `quota_saturated_in_trace=true` 的 tier/count 都不应该进入主 q-sweep 图。

对 Featherless-style concurrency：

- 增加 `n` 应该降低因为 S_C capacity saturate 导致的 API fallback cost。
- 增加 `n` 也会增加 fixed subscription cost。
- 因此 total cost 应该存在 optimum，不一定在最大的 `n`。
- utilization 必须用 weighted capacity units 报告。例如一个 cost=4 的 70B in-flight request
  对一个 Premium subscription 是 100% utilization，不是 25%。
- 主 §1.3 sweep 应该报告 `featherless_premium × n`，不是 `featherless_scale`；
  Premium `n=1..4` 已经用当前公开 docs 对得上的 facts 覆盖了 70B effective concurrency `1..4`。

---

## 10. Selection Rule for Later Experiments

May-4 讨论里确定了一个 workflow：

```text
cost-layer §1.2 先找出 subscription plan/count；
cost-layer §1.3 先找出 concurrency plan/count；
后面的 end-to-end experiments 复用这个 selected setting。
```

所以 §1.2 有两个任务：

1. 报告 cost-layer subscription-count sweep 本身。
2. 为后续 sections 选择 canonical `subscription_plan`、`subscription_count` 和 workload window。

§1.3 对 concurrency plans 使用同样 workflow。

### 10.1 选什么

selected setting 应该记录成 metadata，不要留在人的脑子里：

```yaml
selected_subscription_setting:
  source_experiment: cost_layer_quota
  workload: burstgpt
  workload_window: full_month
  subscription_plan: chutes
  subscription_count: 2
  selection_metric: total_cost_usd
  tie_breaks: [smaller_subscription_count, higher_quota_utilization]

selected_concurrency_setting:
  source_experiment: cost_layer_concurrency
  workload: burstgpt
  workload_window: full_month
  concurrency_plan: featherless_premium
  concurrency_count: 1
  selection_metric: total_cost_usd
  tie_breaks: [smaller_concurrency_count, higher_weighted_utilization]
```

一开始可以存在 cost-layer output metadata 里。如果稳定了，再复制到一个 checked-in config，供 end-to-end runs 使用。

### 10.2 Selection metric

对 monthly fee 已知的 plans，选择：

```text
argmin_n total_cost_usd(plan, n)
```

约束：

- `subscription_cost_known = true`
- `trace_paper_grade = true`
- quota 没有在整个 workload 中 saturate
- RouteWise 真的把 quota 分配给更高价值 requests

对 §1.3，把 quota-specific constraints 换成：

- weighted concurrency capacity 真的被 exercised
- S_C saturate 或 model incompatible 时，RouteWise 会 spill 到 API
- utilization 用 `used_concurrency_cost / concurrency_capacity` 计算

Tie-breaks：

1. 选择更小的 `n`
2. 对 §1.2 选择 quota utilization 更高的 setting；对 §1.3 选择 weighted concurrency utilization 更高的 setting
3. 选择 latency distribution 更适合 end-to-end 的 provider

对未来 monthly fee 未知的 plans，不要选为主 dollar-cost setting。
它仍然可以作为 allocation/utilization robustness run 报告。

对 `cost_claim_allowed=false` 的 quota plans，使用：

```text
selection metric = argmax_n quota_utilization
```

约束：

- `quota_saturated_in_trace = false`
- run 仍然报告 `api_cost_usd`
- paper text 不出现 `total_cost_usd` claim

对 `cost_claim_allowed=false` 的 concurrency plans，使用：

```text
selection metric = argmax_n mean_concurrency_utilization
```

同样不能做 dollar-cost claim。

### 10.3 Workload window

默认 paper-grade setting 是 full one-month workload。对 §1.2 来说，这是最干净的，因为 subscription economics
需要足够的请求量。

May-4 讨论里也提到可以找一个只需要一个 subscription 的 period，让 end-to-end story 更容易解释。
这可以做，但必须有提前声明的 selection rule。不能看完结果后手挑一个方便的 day。

可接受的 window rules：

```text
full_month
representative_7d_window = median-request-volume contiguous 7-day window
representative_1d_window = median-request-volume day
```

representative windows 的 tie-break：如果多个 windows 距离 median volume 一样，选择最早的 start timestamp。

主 cost-layer claim 优先使用 `full_month`。对 end-to-end，如果 full-month 太大，或者 one-subscription story
被规模掩盖，可以用 representative smaller window。chosen window 必须写进 metadata。

### 10.4 后续 sections 怎么用

End-to-end 不应该重新打开 q-sweep，除非 paper question 明确是 purchase count。它应该导入 selected setting：

```text
selected quota plan/count from §1.2
selected concurrency plan/count from §1.3
fixed RW3/RW8 on-demand pools
```

这样 end-to-end experiment 聚焦在完整 RouteWise routing stack，而不是重新做 subscription purchase search。

---

## 11. Tests

### 11.1 Unit tests

需要加 tests：

- YAML loader 拒绝缺少 quota size 的 plan。
- YAML loader 拒绝 `cost_claim_allowed=true` 且 `monthly_fee_usd=null`。
- Chutes plan 读出来是 `$20/month`、`5000/day`。
- `make_quota_provider(plan=chutes, subscription_count=2)` 生成
  `QuotaState(size=10000, window_sec=86400)`。
- saturated count 会生成 warning / metadata flag，不会静默进入 paper q-sweep。
- Featherless Premium 读出来是 `$25/month`、`concurrency_allotment=4`。
- `make_concurrency_provider(plan=featherless_premium, concurrency_count=2)`
  生成 weighted capacity `8`。
- `concurrency_cost=4` 的 model 会消耗一个 Premium subscription 的全部 capacity。
- `concurrency_cost=1` 的 model 在一个 Premium subscription 上可以同时 admit 四个 request。
- unknown trace model ID 会 resolve 到 `default_model_class` 并写 warning metadata，不会静默掉出 S_C。
- 真正 incompatible 的 resolved class 不能 admit 到 S_C。

### 11.2 Cost accounting tests

用一个 timestamp 已知的小 synthetic workload：

```text
trace span = 1 day
monthly_fee_usd = $30
q = 2
billing_period_days = 30
expected fixed fee =
  monthly_fee_usd * q * (trace_days / billing_period_days)
  = 30 * 2 * (1 / 30) = $2
```

断言：

```text
subscription_fixed_cost_usd == 2.0
total_cost_usd == api_cost_usd + 2.0
```

还要断言 offline 在同一个 scenario 里支付同一笔 fixed fee。

对 concurrency plan 重复同样的 fixed-fee accounting test：

```text
monthly_fee_usd = $30
concurrency_count = 2
trace span = 1 day
expected fixed fee = 30 * 2 * (1 / 30) = $2
```

### 11.3 Behavior smoke

对 Chutes `q1`，100k BurstGPT smoke：

```text
mean cheapest-API cost of quota-routed requests >
mean cheapest-API cost of API-routed requests
```

这个保护 paper 的核心 claim：稀缺 quota 应该留给更高价值 requests。

对 Featherless Premium `n=1`，synthetic behavior smoke：

```text
concurrency_allotment = 4
one active request with concurrency_cost=4 blocks another cost-1 request
four active requests with concurrency_cost=1 block the fifth cost-1 request
after finish_time passes, capacity is released
```

这个保护 §1.3 的核心 claim：concurrency 是 weighted capacity，不是 raw request count。

---

## 12. Out of Scope

第一版不做这些：

| Item | Reason |
|---|---|
| Per-token subscription quotas | 当前 paper plan 使用 request quotas。不同单位需要新公式。 |
| Buying decision optimizer | 目前 `q1..q4` sweep 就是 optimizer。 |
| Chutes live calls | 这里只做 simulator。 |
| Featherless live calls | §1.3 只做 simulator。Real evaluation 后面可以复用 selected setting。 |
| Legacy `featherless_scale` | 旧 offline-stage entry 不是当前公开 Featherless plan，并且对 70B-class models 使用了 `concurrency_cost=2`。不要放进 §1.3 主图。 |
| S_C queueing policy | 第一版 §1.3 只做 immediate-admission；queueing 放到 end-to-end SLO experiments。 |
| MiniMax High-Speed / Ultra plans | 第一版 §1.2 不做。它们很可能需要独立 latency profiles，而且会把 pricing 和 model-speed 变化混在一起。 |
| Full end-to-end joint setup | 等 cost-layer §1.2 和 §1.3 都稳定后再做。 |
| Joint S_Q + S_C purchase optimizer | 先让 §1.2 和 §1.3 独立选择 quota/concurrency settings。 |
| Tier-upgrade pricing | `q`/`n` 表示多个独立 accounts/subscriptions，不表示升级到更高 plan tier。 |

---

## 13. Open Questions

1. 后续 end-to-end 使用 full-month selected setting，还是使用一个按预声明规则选出的 representative smaller window？

Chutes 和 MiniMax Starter / Plus / Max 的实现 blocker 只有一个：MiniMax 的 5-hour + weekly allowance
需要 composite quota support。Chutes 可以直接用现有 single-window quota state 跑。
