# Oracle Experiment: Output Length Prediction Sensitivity Analysis

## 1. Background: Juncheng's Key Insight

在讨论 embedding-based output length prediction 方法时，Juncheng 提出了一个关键观点：

> "We don't necessarily need a better solution. What we want to understand is: **if we know
> the output length, how does it compare with just using EMA and Histogram?** If the difference
> is small, then just using heuristics is sufficient. If the difference is large, then we have
> to think about better ways to predict length."

进一步，他指出：

> "Not the output length prediction accuracy, but rather the **total cost**. When you
> mispredict output length, how much impact will you have on the end-to-end result?
> Because it's possible that you mispredict -- if output token is 1 but you mispredict
> it as 10, even though 10x larger, it doesn't matter much. But if output length is 1K
> and you mispredict it as 5K, although it's only 5x, the impact could be very large."

> "You can compare: run the same system using predicted length and known length...
> We want to compare the predicted output length and true length to see the **impact
> on e2e cost**. Even if they're close, we don't necessarily need a fancier method."

## 2. Experiment Design

### 2.1 Core Question

**Output length prediction 的误差对 routing 的 end-to-end cost 影响有多大？**

这个实验把两个因素解耦：
- **在线决策的损失**（不知道未来请求）→ Optimal vs PD-Oracle 的 gap
- **预测误差的损失**（不知道 output length）→ PD-Oracle vs PD-EMA 的 gap

```
                        知道未来请求？
                     Yes              No
                 ┌──────────┬──────────────────┐
   知道 output   │ Optimal  │  PD-Oracle       │
   length？      │ (离线)    │  (在线+真实长度)  │
   Yes           │          │                  │
                 ├──────────┼──────────────────┤
   No            │   N/A    │  PD-EMA          │
                 │          │  PD-Histogram    │
                 └──────────┴──────────────────┘

Gap 分解:
  Total Gap = Cost(PD-EMA) - Cost(Optimal)
            = [Cost(PD-EMA) - Cost(PD-Oracle)]  ← 预测误差的影响
            + [Cost(PD-Oracle) - Cost(Optimal)]  ← 在线决策的影响
```

### 2.2 Strategy Variants

| Strategy | Output Length | Future Requests | Decision Rule |
|----------|:---:|:---:|------|
| **Optimal** (offline) | True | Known | Sort by cost, pick top-Q per day |
| **PD-Oracle** | True | Unknown | PD threshold on true value |
| **LA-Oracle** | True | Unknown | LA threshold on true value (P10=true) |
| **PD-EMA** | EMA predicted | Unknown | PD threshold on predicted value |
| **PD-Histogram** | Hist predicted | Unknown | PD threshold on predicted value |
| **LA-EMA** | EMA predicted | Unknown | LA threshold on predicted LCB |
| **LA-Histogram** | Hist predicted | Unknown | LA threshold on predicted LCB |
| **Greedy** | N/A | Unknown | FCFS (first come first serve) |

### 2.3 Metrics

- **Primary metric**: Total API cost (lower = better)
- **Competitive Ratio (CR)**: Cost / Optimal Cost (closer to 1.0 = better)
- **Prediction Gap**: CR(PD-EMA) - CR(PD-Oracle) — 预测误差带来的额外 cost

### 2.4 Datasets

在所有 4 个 dataset 上跑，覆盖不同 workload 特征：

| Dataset | Requests | Days | Models | Output Length Distribution |
|---------|:--------:|:----:|:------:|--------------------------|
| ShareGPT | 201K | 7 | 1 | High variance (chat) |
| FreeInference | 371K | 90 | 12 | Multi-model, skewed traffic |
| Enterprise | 55K | 84 | 8 | Business workload |
| BurstGPT | 1.4M | 61 | 1 | Large scale |

### 2.5 Quota Levels

每个 dataset 用 3 个 quota level（和现有实验一致）：
- Low: quota ≈ 10% of daily requests
- Medium: quota ≈ 30% of daily requests
- High: quota ≈ 50% of daily requests

## 3. Implementation Plan

### 3.1 Step 1: Create OracleOutputPredictor

创建一个新的 predictor，在 `predict()` 时直接返回真实的 `response_tokens`。

文件: `experiment/strategies/online/predictors/oracle.py`

```python
class OracleOutputPredictor(OutputTokenPredictor):
    """Oracle predictor that uses true output length from trace replay.

    This predictor "cheats" by reading the ground-truth response_tokens
    from the request object. It serves as an upper bound on what any
    predictor can achieve, isolating the impact of prediction error
    on end-to-end routing cost.
    """

    def predict(self, request: Request) -> QuantilePrediction:
        true_len = float(request.response_tokens)
        # Oracle knows the exact length, so all quantiles = true length
        return QuantilePrediction(
            q10=true_len,
            q50=true_len,
            q90=true_len,
            is_warmed_up=True,
        )

    def update(self, request: Request) -> None:
        pass  # No state to update

    def reset(self) -> None:
        pass  # No state to reset

    @property
    def is_warmed_up(self) -> bool:
        return True  # Always warmed up
```

### 3.2 Step 2: Add Oracle Variants to Experiment Runner

在 `scripts/run_online_experiments.py` 中加入 PD-Oracle 和 LA-Oracle：

```python
strategies = [
    # ... existing strategies ...
    ("PD-Oracle", PrimalDualOnlineStrategy(
        ..., output_predictor=OracleOutputPredictor())),
    ("LA-Oracle", LearningAugmentedPrimalDualStrategy(
        ..., output_predictor=OracleOutputPredictor())),
]
```

### 3.3 Step 3: Run Experiments

```bash
source .venv/bin/activate
python -m experiment.scripts.run_online_experiments \
    --datasets burstgpt freeinference sharegpt enterprise \
    --stages stage1 \
    --output experiment/results/oracle/
```

### 3.4 Step 4: Analyze Results

生成对比表格和可视化：

```
Expected output format:

Dataset: BurstGPT (Q=5000)
┌──────────────┬────────┬──────┬─────────────────┐
│ Strategy     │ Cost   │ CR   │ Gap from Oracle │
├──────────────┼────────┼──────┼─────────────────┤
│ Optimal      │ $532   │ 1.00 │ --              │
│ PD-Oracle    │ $???   │ ???  │ 0 (baseline)    │
│ PD-EMA       │ $629   │ 1.18 │ ???             │
│ PD-Histogram │ $???   │ ???  │ ???             │
│ LA-Oracle    │ $???   │ ???  │ 0 (baseline)    │
│ LA-EMA       │ $633   │ 1.19 │ ???             │
│ Greedy       │ $691   │ 1.30 │ --              │
│ All-API      │ $907   │ 1.70 │ --              │
└──────────────┴────────┴──────┴─────────────────┘

Key analysis:
  Online decision gap = CR(PD-Oracle) - CR(Optimal)
  Prediction gap      = CR(PD-EMA)   - CR(PD-Oracle)
  Total gap           = CR(PD-EMA)   - CR(Optimal) = 0.18
```

### 3.5 Step 5: Visualization

生成一张 stacked bar chart：

```
CR breakdown for each strategy:

  PD-EMA:  |████ Optimal ████|█ Online gap █|█ Pred gap █|
           1.0               ???            ???          1.18

如果 Prediction gap 很小 → "Simple heuristics are sufficient"
如果 Prediction gap 很大 → "Need better prediction methods"
```

## 4. Expected Outcomes

### Scenario A: Prediction gap is small (expected)

```
PD-Oracle CR ≈ 1.15,  PD-EMA CR = 1.18
Prediction gap = 0.03 (仅 17% of total gap)
```

**结论**: Output length prediction 的误差对 routing cost 影响很小。
EMA/Histogram 这种简单方法足够。不需要 embedding-based 方法。

**原因**:
- Routing 是 binary threshold decision，不需要精确长度
- 误判短请求（1→10 tokens）影响很小，因为 value 差距也小
- EMA 的快速适应性补偿了统计精度的不足

### Scenario B: Prediction gap is large

```
PD-Oracle CR ≈ 1.05,  PD-EMA CR = 1.18
Prediction gap = 0.13 (72% of total gap)
```

**结论**: 需要更好的预测方法。但 embedding method 仍然不可用（时序不兼容）。
可以考虑：
- Prompt feature engineering (category B)
- 更好的在线统计方法（weighted histogram, contextual bandits）

### 以往实验的间接证据

从现有 ablation 结果来看，Scenario A 更可能：
- EMA (CR 1.18) vs Histogram (CR 1.23): predictor 选择只影响 0.05 CR
- PD vs LA: decision rule 只影响 0.01 CR
- 这说明系统对 prediction quality 不是很敏感

## 5. Paper Contribution

无论哪个 scenario，这个实验都有 paper value：

- **Scenario A → 正面贡献**: 证明简单方法 sufficient，系统设计不需要复杂预测
  > "Oracle analysis shows that perfect output length prediction would only reduce
  > CR by X%, confirming that simple online estimators are near-optimal for routing."

- **Scenario B → 指明方向**: 量化了 prediction quality 的 upper bound improvement
  > "Oracle analysis reveals Y% potential improvement from better prediction,
  > motivating future work on lightweight pre-routing length estimation."

## 6. Timeline

| Step | Task | Estimated Time |
|:----:|------|:--------------:|
| 1 | Create OracleOutputPredictor | 30 min |
| 2 | Update experiment runner | 30 min |
| 3 | Run experiments (4 datasets × 3 quota levels) | 1-2 hours |
| 4 | Analyze results + visualization | 1 hour |
| 5 | Update paper / literature survey | 1 hour |
| **Total** | | **~4-5 hours** |
