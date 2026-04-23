# EMA vs Histogram Predictor 详解

## 1. 为什么需要 Predictor？

在 online routing 中，router 在每个 request 到达时必须立刻做出决策：发给 S_Q（免费 quota）还是 S_A（付费 API）。

决策依据是 request 的 **value**（省下来的 API 费用）：

```
value = (input_tokens × price_in + output_tokens × price_out) / 1M
```

问题是：**output_tokens 在决策时是未知的**（request 还没执行，不知道 LLM 会输出多少 token）。

而 output price 通常是 input price 的 3-10 倍，所以 value 的主要成分是 output tokens。

因此，我们需要一个 predictor 来**预测 output token 数量**，进而估算 request value，再和 threshold 比较做出路由决策。

---

## 2. EMA Predictor（指数移动平均）

### 2.1 核心思想

最简单的方法：用过去请求的 output token 数量的**滑动平均值**来预测下一个请求的 output token 数量。

### 2.2 公式

```
L̂_{t+1} = α × L_t^actual + (1 - α) × L̂_t
```

- `L̂_t` = 第 t 步的预测值（estimated output length）
- `L_t^actual` = 第 t 个请求完成后观测到的真实 output token 数
- `α = 0.1` = 平滑系数（越大越重视最近的观测值）

直白地说：新预测 = 10% 的最新观测 + 90% 的旧预测。

### 2.3 Quantile 估计

EMA 本身只给一个点估计（mean）。为了得到 P10/P50/P90，用**正态近似**：

```
q10 = mean - 1.28 × std
q50 = mean
q90 = mean + 1.28 × std
```

其中 std 也是通过 EMA 在线更新的：
```
variance = (1 - α) × (variance + α × (actual - mean)²)
std = sqrt(variance)
```

### 2.4 分层策略

EMA predictor 维护两层状态：

```
Level 1: per-model EMA    （需要 10+ 样本才启用）
Level 2: global EMA       （需要 20+ 样本才启用）
```

预测时优先用 per-model 的统计值。如果某个 model 样本不够（比如刚出现的新 model），退回到 global。

### 2.5 特点总结

| 特点 | 说明 |
|------|------|
| 输入特征 | **只用 model 名称**，不用 input length、时段等 |
| Warmup 速度 | 快，~10-20 个样本就能收敛 |
| 适应速度 | 快，α=0.1 意味着 ~10 个样本就能跟上分布变化 |
| Quantile 质量 | 差，正态近似不准（实际 output 分布通常是 heavy-tail） |
| Calibration | q50 coverage = 72%（理想应为 50%，说明系统性高估） |
| 实现复杂度 | 极低，O(1) 预测和更新 |

### 2.6 代码对应

```python
class EMAOutputPredictor:
    def predict(self, request):
        # 找到对应的 EMA state（per-model 或 global）
        state = self.model_states[model]  # or self.global_state
        mean = state.mean
        std = state.std
        return QuantilePrediction(
            q10 = mean - 1.28 * std,
            q50 = mean,
            q90 = mean + 1.28 * std,
        )

    def update(self, request):
        # request 完成后，用真实值更新 EMA
        actual = request.response_tokens
        state.mean += alpha * (actual - state.mean)
```

---

## 3. Histogram Predictor（直方图分位数估计器）

### 3.1 核心思想

不同的请求条件（model、input 长度、时段）对应不同的 output 分布。
Histogram predictor 为每种条件组合维护一个独立的 output token 分布直方图，
查询 quantile 时直接从对应直方图中读取。

### 3.2 分层 Backoff 架构

Histogram 使用 4 级分层 key，从细到粗：

```
Level 0 (最细): key = (model, input_bin, hour_bin)    需要 10+ 样本
Level 1:        key = (model, input_bin)               需要 20+ 样本
Level 2:        key = (model)                          需要 50+ 样本
Level 3 (最粗): key = global                           需要 100+ 样本
```

- **input_bin**: input token 数量的 log2 分桶
  - bin 0: [1, 2) tokens
  - bin 1: [2, 4) tokens
  - bin 10: [1024, 2048) tokens
  - bin 15: [32768, 65536) tokens
  - 共 20 个 bin

- **hour_bin**: 每 4 小时一个 bin，一天 6 段
  - bin 0: 0:00-3:59
  - bin 1: 4:00-7:59
  - ...
  - bin 5: 20:00-23:59

预测时：先尝试最细粒度的 Level 0，如果该 key 样本不够（<10），退回 Level 1，
依此类推，直到找到一个样本够多的层级。

### 3.3 Streaming Histogram 原理

每个 key 对应一个 **50 bin 的 log-spaced 直方图**，范围 [1, 100000]：

```
Bin 0:  [1.0,  1.26)
Bin 1:  [1.26, 1.58)
...
Bin 25: [316, 398)      ← 大多数 output 落在这个范围
...
Bin 49: [79433, 100000)
```

用 log-spaced 是因为 output token 分布跨越好几个数量级（有的请求输出 10 token，有的输出 10000 token）。

**添加样本**：request 完成后，把真实 output token 数量放进对应 bin 的 count +1。

**查询 quantile**：遍历 bin，累加 count，直到累计达到目标比例。

```
查询 P10：找到累积 count 达到 total × 10% 的 bin
  → 返回该 bin 的 lower bound（保守估计，确保 LCB 语义）

查询 P50：累积 count 达到 total × 50% 的 bin
  → 返回该 bin 的 midpoint

查询 P90：累积 count 达到 total × 90% 的 bin
  → 返回该 bin 的 upper bound（保守估计，确保 UCB 语义）
```

### 3.4 一个具体例子

假设我们在预测 `llama-3.3-70b, input 1500 tokens, 下午 2 点` 的请求。

```
Step 1: 计算 key
  model = "llama-3.3-70b"
  input_bin = log2(1500) ≈ 10   → bin 10 对应 [1024, 2048)
  hour_bin = 14 // 4 = 3        → bin 3 对应 12:00-15:59

Step 2: 查找最细粒度的直方图
  key = ("llama-3.3-70b", 10, 3)
  → 如果这个 key 有 15 个样本（≥10），用它
  → 如果不够，退回 ("llama-3.3-70b", 10)
  → 如果还不够，退回 ("llama-3.3-70b")
  → 最后退回 global

Step 3: 从选中的直方图查询 quantile
  假设用了 Level 1 的直方图，里面有 50 个样本：

  Bin [50, 63):   ████ 8 个样本
  Bin [63, 79):   ██ 4 个样本
  Bin [79, 100):  █████ 10 个样本
  Bin [100, 126): ████████ 16 个样本
  Bin [126, 158): ████ 8 个样本
  Bin [158, 200): ██ 4 个样本

  P10: 累积到 50 × 0.1 = 5 → 落在第一个 bin → 返回 50（bin lower bound）
  P50: 累积到 50 × 0.5 = 25 → 落在 [100, 126) bin → 返回 113（midpoint）
  P90: 累积到 50 × 0.9 = 45 → 落在 [158, 200) bin → 返回 200（bin upper bound）
```

### 3.5 特点总结

| 特点 | 说明 |
|------|------|
| 输入特征 | **model + input_bin + hour_bin**（3 维条件） |
| Warmup 速度 | 慢，Level 0 每个 key 需要 10+ 样本，key 空间很大 |
| 适应速度 | 慢，直方图累积历史数据，不像 EMA 有遗忘机制 |
| Quantile 质量 | 好，直接从经验分布读取，不依赖分布假设 |
| Calibration | q50 coverage = 56%（接近理想的 50%） |
| 实现复杂度 | 中等，O(num_bins) 预测，O(1) 更新 |

### 3.6 代码对应

```python
class HistogramOutputPredictor:
    def predict(self, request):
        ctx = PredictionContext.from_request(request)
        hour_bin = ctx.hour_of_day // 4

        # 从分层统计中获取 quantile（自动 backoff）
        q10 = self.stats.quantile(ctx.model, ctx.input_bin, hour_bin, 0.10)
        q50 = self.stats.quantile(ctx.model, ctx.input_bin, hour_bin, 0.50)
        q90 = self.stats.quantile(ctx.model, ctx.input_bin, hour_bin, 0.90)
        return QuantilePrediction(q10, q50, q90)

    def update(self, request):
        # request 完成后，把真实值加入所有层级的直方图
        actual = request.response_tokens
        self.stats.add(model, input_bin, hour_bin, actual)
        # 同时更新 Level 0, 1, 2, global 四个直方图
```

---

## 4. PD 与 LA 在决策上的核心区别

当 Predictor（EMA 或 Histogram）算出了 P10、P50、P90 这些分布数据后，最终的“拍板权”在 PD 或 LA 手里。

由于 **输出长度是未知的**，我们在计算请求的“预计省钱价值（Value）”时，就会面临**风险（Risk）**：如果你以为它会输出很长（价值很高）从而用掉了宝贵的配额，结果 LLM 只回了一个 "OK"（价值极低），那你就血亏了一个配额。

### 4.1 PD (普通原始对偶) —— “风险中性”的赌徒
```python
value = cost_api(input_tokens, predictor.q50)   # ← 使用 P50（中位数）
if value >= threshold(z) and quota_remaining > 0:
    route → S_Q (subscription)
else:
    route → S_A (API)
```
- **逻辑：** PD 选择相信**中位数 (P50)**。它认为：“只要你的平均期望长度算出来的价值，超过了我的门槛，我就赌一把，把配额给你。”
- **缺点：** 容易被“骗”。如果一个请求方差很大（有一半概率输出 1000 个字，但也有一半概率只输出 10 个字），PD 看到 P50 = 500 觉得挺高就放行了。结果真跑的时候它可能只输出了 10 个字，白白浪费了宝贵的配额。

### 4.2 LA (Learning-Augmented 学习增强) —— “极度厌恶风险”的保守派
```python
value = cost_api(input_tokens, predictor.q10)   # ← 使用 P10（保守下界 / LCB）
if value >= threshold(z) and quota_remaining > 0:
    route → S_Q (subscription)
else:
    route → S_A (API)
```
- **逻辑：** LA 根本不看你的平均表现，它只看**下限 (Lower Confidence Bound, P10)**。
- **什么是 LA？** Learning-Augmented 在这个语境下，特指结合了**分位数预测（Quantile Prediction）**来对抗不确定性（Uncertainty）的算法增强。
- **为什么用 P10？** 用 P10 算出的价值意味着：**“我有 90% 的绝对把握，这个请求最终实际省下的钱【至少】有这么多。”**
- **LA 的优势：** 只有当这个“最惨情况的保底价值（P10）”都大于系统的影子价格 `threshold(z)` 时，LA 才舍得动用配额。这是一种极致的防守策略（保守估计），完美杜绝了配额被“高方差、雷声大雨点小”的请求白白骗走。

---

## 5. 2×2 实验结果

|  | PD (用 P50) | LA (用 P10) |
|---|:---:|:---:|
| **EMA** | **CR 1.18** (最优) | CR 1.19 |
| **Histogram** | CR 1.23 | CR 1.22 |

CR = Competitive Ratio = 在线策略成本 / 离线最优成本，越低越好。

### 5.1 为什么 EMA 比 Histogram 好？

**原因 1: Warmup 速度**

EMA 只需 ~20 个样本就能给出合理预测。Histogram 的 Level 0 需要每个 (model, input_bin, hour_bin) key 至少 10 个样本。假设有 12 个 model × 20 个 input_bin × 6 个 hour_bin = 1440 个 key，很多 key 长期凑不够样本，只能退回到粗粒度，实际和 per-model 的 EMA 差不多。

**原因 2: 适应速度**

EMA 的指数衰减意味着旧数据自然淡出（半衰期 ≈ log(0.5)/log(0.9) ≈ 7 个样本）。Histogram 没有遗忘机制，所有历史数据权重相同。当 output 分布变化时（比如用户从写代码切换到写文档），EMA 能在 ~10 个请求内跟上，Histogram 需要更多样本才能"稀释"旧数据。

**原因 3: 高估的"意外好处"**

EMA 的 q50 coverage 是 72%（系统性高估 output length），这意味着它倾向于高估 request value → 更激进地使用 quota。当 quota 充裕时，多用 quota 不亏（没用完的 quota 就浪费了），所以高估反而有利。

### 5.2 为什么 PD 比 LA 好？

LA 用 P10（保守）估值，会过度保守，导致很多"中等价值"的请求本可以用 quota 却被发去了 API。在 quota 充裕的日子里，这些 quota 就白白浪费了。

PD 用 P50（中性）估值，更平衡，在 quota 充裕和紧张的场景之间取得了更好的 tradeoff。

### 5.3 核心 Takeaway

> **在 online routing 中，predictor 的响应速度（responsiveness）比统计精度（calibration）更重要。**

Histogram 在统计学意义上更"正确"（calibration 更好），但 EMA 在实际 routing 成本上更优。这是因为 routing 是一个**决策问题**而非**预测问题**——我们不需要精确知道 output 有多长，只需要快速判断这个 request 是否"值得"用 quota。

---

## 6. 与 Slide 的对应关系

| Slide 页面 | 内容 |
|-----------|------|
| Experimental Setup | 2×2 策略矩阵定义 |
| Online Results: Cost Comparison | PD-EMA（最优）vs 其他策略的成本对比 |
| Ablation Study | 2×2 heatmap，说明 predictor > decision rule |
| Predictor Calibration Analysis | EMA vs Histogram 的 calibration 对比 |
| Hyperparameter Sensitivity | α 值的影响（α=0.1 最优） |
