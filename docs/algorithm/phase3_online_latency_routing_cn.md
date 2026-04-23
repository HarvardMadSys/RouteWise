# Phase 3：在线 Latency-Aware Provider Routing（中文说明）

## 1. 概览

### 1.1 目标

实现一个**在线的 latency-aware 路由模块**：在多 provider 平台（例如 OpenRouter）中选择 provider，以**最小化成本**并满足一个**尾部约束**（tail constraint），例如：

> 对给定 deadline \(L\)，要求混合分布在 \(L\) 处的 CDF 满足：\(\sum_j \pi_j \hat{F}_j(L) \ge 0.99\)。

### 1.2 关键设计原则

1. **Latency 是独立维度**：不干扰已有 cost 优化（Stage 1/2）。当系统选择走 API（或 multi-provider API fallback）时，再在 provider 层面做 latency-aware 选择。
2. **LP 最优混合（LP-mix）**：使用线性规划求解最优的 provider 混合概率 \(\pi\)，而不是启发式 Pareto。由于只有一个尾部约束 + 概率和约束，LP 的最优基本可行解最多包含 2 个非零 provider（simplex 结构性质）。
3. **在线 profiling + 时间窗**：用 probing requests 维护每个 provider 的延迟分布，采用时间窗捕捉 non-stationarity。
4. **Reliability-aware**：将 timeout / 429 / 5xx 等失败事件纳入 tail 约束与（可选的）软惩罚项，避免“低延迟但高失败率”的假优势。

### 1.3 范围（Scope）

- **Phase 3 目标内**：在线 LP-mix routing、基于 probing 的 profiling、BurstGPT 回放评估
- **Phase 4 再做**：Smart hedging、\(\kappa\) 等高级 cost penalty 的系统化调参、E2E 长输出场景的完整建模

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Online Latency Router                            │
│                                                                          │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │  Probing Module  │───>│  Profile Store   │───>│     LP Solver     │  │
│  │ (45s/provider)   │    │ (short + long)   │    │ (scipy.linprog)   │  │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘  │
│           │                       │                       │             │
│           │                       ▼                       ▼             │
│           │              ┌──────────────────┐    ┌──────────────────┐  │
│           │              │    Pre-filter    │    │   SWRR Sampler    │  │
│           │              │ (error/latency)  │    │ (quota-based)     │  │
│           │              └──────────────────┘    └──────────────────┘  │
│           │                                               │             │
│           ▼                                               ▼             │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    Request Router                                 │  │
│  │                    route(request) -> provider                     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心组件

### 3.1 Moving Window Profiler（时间窗分布估计）

我们维护每个 provider 的 TTFT 分布，采用**时间窗（time-based window）**而不是样本数窗，避免 probe 频率变化导致窗口语义漂移。

由于 short window 的样本量可能很小（例如 15 min、45s/probe 时约 20 个样本），直接估计 P99 会高度不稳定。因此，我们不在 short window 上“直接估 P99”，而是构造用于 LP 的 **Mixed-Window CDF**（在固定的 \(L\) 上估 \(\hat{F}(L)\)）。

#### 3.1.1 Mixed-Window CDF（带自适应 shrinkage）

对固定 deadline \(L\)，定义：

\[
\hat{F}_j(L) = \beta_j \cdot \hat{F}^{\text{short}}_j(L) + (1-\beta_j)\cdot \hat{F}^{\text{long}}_j(L)
\]

其中 \(\beta_j\) 不是常数，而是根据 short window 的有效样本量自适应：

\[
\beta_j = \frac{N^{\text{short}}_{\text{eff},j}}{N^{\text{short}}_{\text{eff},j} + \lambda}
\]

- \(N^{\text{short}}_{\text{eff},j}\)：short window 内该 provider 的**成功 probe**数量（或你们定义的有效样本量）
- \(\lambda\)：**prior strength（先验强度）**，表示 long window 相当于提供了 \(\lambda\) 个“伪样本”的稳定基底。该参数建议做敏感性分析（例如 \(\lambda\in\{10,20,50\}\)）

#### 3.1.2 Failure 事件如何进入 \(\hat{F}(L)\)

在 Phase 3 的主线语义中，tail 约束指的是“请求在 \(L\) 内成功返回”的概率，因此 timeout/429/5xx 等失败应视为**miss deadline**：

- 对任何有限 \(L\)，失败样本对 \(\hat{F}(L)\) 的贡献为 0（等价于 latency = \(\infty\)）

这使得 tail 约束天然具备 reliability 约束的效果。

#### 3.1.3 P99 的位置

- short window：不直接用于估 P99（只用于估 \(\hat{F}^{\text{short}}(L)\)）
- long window：P99（以及 P90/P95）用于报告与可视化（例如展示 drift / heavy-tail），但不作为 LP 的直接输入统计量

---

### 3.2 Pre-filter（硬过滤）

在求 LP 前，先用硬规则过滤明显不可用的 provider（减少 LP 无解概率，也避免被坏 provider 污染决策）。

推荐规则：

1. `total_error_rate > 5%` → 直接下线（明显不可用）
2. `F_hat(L_min) < 0.80` → 直接下线（连基本 SLO 都达不到）

其中 \(L_{\min}\) 是一个“基本可用性”的 deadline，例如 1s（可调）。

注意：这里用的是混合 CDF \(\hat{F}\)（与 LP 输入一致）。

---

### 3.3 LP Solver（在线 LP-mix）

对通过 pre-filter 的候选 providers，解下面的 LP：

\[
\min_{\pi}\ \sum_j \pi_j \cdot c_j \cdot (1+\kappa e_j)
\]
subject to
\[
\sum_j \pi_j \cdot \hat{F}_j(L) \ge 0.99,\quad \sum_j \pi_j = 1,\quad \pi_j \ge 0
\]

符号解释：

- \(c_j\)：provider \(j\) 的单位请求成本（由 token pricing 与固定 request shape 计算得到）
- \(\hat{F}_j(L)\)：provider \(j\) 在 deadline \(L\) 处的 mixed-window CDF
- \(e_j\)：短窗或固定窗口内的错误率（可细分 timeout_rate 与 other_error_rate）
- \(\kappa\)：软惩罚系数

#### 3.3.1 关于 \(\kappa\)（避免 double-count）

由于 failures 已经通过 \(\hat{F}_j(L)\) 进入了约束项，\(\kappa\) 应被视为“二级偏好（secondary preference）”，用于在可行集内更偏好可靠 provider。

推荐：

- 默认 \(\kappa=0\) 作为主线（避免 double-count）
- 在评估中做 ablation：\(\kappa\in\{0,5,10\}\)

#### 3.3.2 LP 无解（Infeasible）时的 fallback

当 SLO 太严或整体网络波动导致 LP 无解时，建议采用分层 fallback：

1. **Relaxed retry**：尝试将 \(L\) 放宽为 `relaxation_factor * L`（例如 1.2、1.5、2.0 逐级尝试，或对 \(L\) 做网格/二分搜索找最小可行 \(L^\*\)）
2. **Best-effort**：若仍无解，选择 \(\arg\max_j \hat{F}_j(L)\) 的 provider，并记录为 “Infeasible Regime”

#### 3.3.3 为什么 LP 最多 2 个非零 provider？

该 LP 有：

- \(n\) 个变量（\(\pi_1,\dots,\pi_n\)）
- 1 个不等式约束（tail constraint）
- 1 个等式约束（\(\sum \pi_j = 1\)）

最优基本可行解至多有 `m=2` 个非零变量，因此最多混合 2 个 provider（这是 LP 结构性质，不是一般 Pareto 的性质）。

---

### 3.4 SWRR Sampler（平滑加权轮询采样）

LP 输出混合概率 \(\pi\)，在线执行时每个请求仍只选一个 provider。为了降低纯随机采样的短期方差，使用 **Smooth Weighted Round-Robin (SWRR)** 做确定性交织：

- 长期频率收敛到 \(\pi\)
- 短期更平滑，减少 burst

#### 3.4.1 权重更新的平滑与节流

由于 \(\hat{F}(L)\) 会随时间波动，LP 解可能跳变，因此需要两层稳健性：

1. **\(\pi\) 平滑**：\(\pi_{\text{new}} = \alpha \pi_{\text{LP}} + (1-\alpha)\pi_{\text{old}}\)，默认 \(\alpha=0.3\)
2. **LP 更新节流**：不要每条 probe 都重解 LP，建议设置最小更新间隔（例如 60s）或累计 \(k\) 条新 probe 后更新一次，避免高频震荡

此外，SWRR 维护内部状态 `current_weights`，权重更新后可进行轻度归一化/soft reset 以防数值漂移（细节属于实现层的稳定性工程）。

---

## 4. 参数建议

### 4.1 SLO（deadline）集合

建议评测：`L ∈ {1s, 2s, 3s, 5s, 10s}`
其中 3s 往往是从“几乎只能选最快”过渡到“开始出现 2-provider mix”的拐点。

### 4.2 Profiling 配置

| 参数 | 建议值 | 说明 |
|---|---:|---|
| Probe interval | 45s/provider | 与 Phase 1 baseline 对齐 |
| Probe request | 极短 prompt + 小输出 | 主要测 TTFT（queueing/prefill） |
| Short window | 15 min | 自适应 drift |
| Long window | 3 h | 稳定基底 |
| Latency metric | TTFT | Phase3 主线指标 |
| LP update | Throttled（如 60s） | 防震荡 |
| prior strength | \(\lambda\in\{10,20,50\}\) | 做敏感性分析 |
| \(\kappa\) | 默认 0；做 ablation | 避免 double-count |
| relaxation factor | 参数化或网格 | 避免 magic number |

---

## 5. 评估计划（MVP）

### 5.1 Simulation setup（无泄漏因果）

1. **Profile source**：Phase 1 probing 日志（例如 `latency_llama70b_24h.csv`）
   - 决策时刻 \(t\)：只允许使用 timestamp < \(t\) 的历史 probe 更新 profile
   - 评测打分：可以用 nearest-neighbor 作为“后验 ground truth”近似（但**不能**回灌到 profile，避免 data leakage）
   - 只用 base workload（固定 request shape）

2. **Workload**：BurstGPT trace（200–500 requests）
   - 回放到达过程（arrival time）
   - 成本按固定 request shape 或统一 token 假设计算（保持与 probing 一致）

3. **Baselines**
   - Single-Best：固定选“最可靠/最快”的单一 provider（例如最大 \(\hat{F}(L)\) 或最低 long-window P99）
   - Cheapest：固定选最便宜 provider
   - Random：在 eligible 集合中均匀随机
   - LP-Mix（Ours）：online profiling + LP + SWRR

### 5.2 指标

- Cost：\$/request 或总成本
- Latency：P50/P90/P99（报告用 long window 或评测样本）
- SLO violation rate：\(\Pr[T > L]\)（在固定 \(L\) 上）
- Reliability：timeout/429/5xx 比例（单独报告）

---

## 6. 实现路径（不包含 Phase 4 hedging）

建议实现顺序：

1. MovingWindowProfiler（时间窗 + mixed-window \(\hat{F}(L)\)）
2. Pre-filter（硬过滤）
3. LP Solver（含 infeasible fallback）
4. SWRR Sampler（含 \(\pi\) 平滑、LP 更新节流）
5. Simulation harness（Phase1 日志回放 + BurstGPT 200–500）
6. 输出主图：cost vs P99 / violation

---

## 7. Phase 4 再做的内容

1. Smart hedging（残余寿命/生存函数触发）
2. 长输出场景：E2E / TPS 建模
3. \(\kappa\) 与 relaxation 策略的系统化调参
4. 真实 OpenRouter live eval（Phase5）
