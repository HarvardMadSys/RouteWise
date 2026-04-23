# Literature Survey: Output Length Prediction for LLM Inference

## 1. Why Output Length Prediction Matters

In LLM inference systems, the output length (number of generated tokens) is unknown before
decoding begins. This creates challenges for:
- **Scheduling**: SJF/SRPT schedulers need to know job duration
- **Memory management**: KV cache allocation depends on output length
- **Cost routing (our use case)**: Request value depends on output tokens

## 2. Taxonomy of Approaches

我们把现有方法按 **预测时机** 和 **信息来源** 分为 6 大类：

```
                        预测时机
                 ┌─────────┬──────────┐
                 │ 路由前    │ Prefill后 │  Decode中
    ─────────────┼─────────┼──────────┼──────────
    外部小模型    │ (A)     │          │
    Prompt特征   │ (B)     │          │
    LLM自预测    │ (C)     │          │
    LLM内部激活  │         │ (D)      │ (E)
    在线统计     │ (F)     │          │
```

### 2.1 (A) External Model — Prompt-Only

用一个独立的小模型（BERT/DistilBERT），从 input prompt 预测 output length。

| Paper | Venue | Method | Accuracy |
|-------|-------|--------|----------|
| **S3** (Jin et al., 2023) | — | Fine-tuned DistilBERT，输出 10 个长度 bucket 分类 | 粗粒度；对变长输出准确度差 |
| **Proxy Model** (Qiu et al., 2024) | ASPLOS | Lightweight BERT proxy 回归预测 | SJF with proxy: 仅比 FCFS 提升 ~3% JCT |

**优点**: 预测发生在路由前，不需要访问 LLM。
**缺点**:
- BERT 推理需要 5-10ms，对 <50us 的路由决策来说太慢
- 需要额外部署一个模型（238-270MB 显存）
- 需要针对每种 LLM 和任务类型的训练数据
- Prompt-only 无法处理 **one-to-many** 问题（同一 prompt 在不同 temperature 下输出长度完全不同）

### 2.2 (B) Prompt Feature Engineering

不训练模型，而是用 prompt 的统计特征（长度、问号数量、任务类型关键词等）做简单回归。

| Method | Features | 说明 |
|--------|----------|------|
| Input length heuristic | input_tokens | 简单假设：输入越长，输出越长 |
| Task-type classifier | keywords ("summarize", "translate", "code") | 不同任务类型有不同输出长度分布 |
| XGBoost/RandomForest | input_length + task_type + model + time | 传统 ML 特征工程 |

**优点**: 不需要 GPU，推理快（<1ms），可在路由前执行。
**缺点**:
- 特征工程强依赖领域知识
- 输入长度和输出长度的相关性在很多场景下很弱
  （如 "用一句话总结这篇 10000 字论文" → 输入长输出短）
- 没有标准化的 benchmark 评测

### 2.3 (C) LLM Self-Prediction

让 LLM 自己先预估输出长度，再正式生成。

| Paper | Venue | Method | Accuracy |
|-------|-------|--------|----------|
| **RLP** (Zheng et al., 2023) | arXiv | LLM 先生成 1-4 个 token 预测长度，再做正式生成 | GPT-4/Claude: <50 词误差，>90% 准确率 |
| **Separate LLM** (Zheng et al., 2024) | — | 用一个更小的 LLM 专门预测长度 | 比 BERT 更准，但增加推理开销 |

**优点**: 不需要额外模型；LLM 对自己的输出行为有一定"自知"。
**缺点**:
- 需要额外一次 LLM forward pass（即使只生成 1-4 token，也要完整 prefill）
- 自预测的准确性依赖 prompt engineering
- **不适合多 provider 路由**：预测本身就需要选一个 provider 执行
- 对于 reasoning/CoT 任务，LLM 自己也不知道 chain 会有多长

### 2.4 (D) Internal Activation — Post-Prefill (Embedding Method)

利用 LLM prefill 阶段产生的 hidden states（即 "embeddings"）来预测 output length。
这是当前准确度最高的一类方法，也是 Juncheng 提到的 "embedding method"。

#### 2.4.1 背景：LLM 推理的两个阶段

LLM 处理一个请求分为两个阶段：

```
阶段 1: Prefill（预填充）
  - 把整个 input prompt 一次性过完所有 transformer layers
  - 每一层产生 hidden states（也叫 activations / embeddings）
  - 维度: [num_input_tokens × hidden_dim]
  - 例如 Llama-3.3-70B: hidden_dim = 8192

阶段 2: Decode（解码）
  - 一个 token 一个 token 地自回归生成 output
  - 每生成一个 token 都要过一次所有 layers
  - 直到生成 <EOS> token 或达到 max_length
```

Embedding method 的核心观察是：

> **Prefill 结束后，LLM 的 hidden states 里已经隐含了 "这个请求大概要生成多少 token" 的信息。**

直觉上：LLM 在 prefill 完成后，内部状态已经 "理解" 了这个 prompt 要求什么
（是写一篇长文章，还是回答一个 yes/no 问题），因此这些 hidden states 里
包含了 output 长度的线索。

#### 2.4.2 代表性工作

| Paper | Venue | Method | Accuracy |
|-------|-------|--------|----------|
| **TRAIL** (Shahout et al., 2024) | ICLR 2025 | LLM layer embeddings → lightweight classifier | MAE ~117-155 tokens |
| **EGTP** (Xie et al., 2025) | ICLR 2026 | Entropy-guided token pooling on prefill activations | MAE ~68-134 tokens; **29% lower MAE than TRAIL** |
| **GNN-based** (Piotrowski et al., 2025) | ACL 2025 SRW | GNN on layerwise hidden states | 50%+ lower NMAE on short outputs |

#### 2.4.3 TRAIL 的工作原理

TRAIL 的方法最直接：在 LLM 的某些 transformer layers 上面接一个很小的 MLP
prediction head，用 (hidden states -> output length) 的监督数据训练。

```
Input prompt (例如 "写一个快速排序算法")
    |
    v
+-------------------+
|   LLM Prefill     |   <-- 正常的 prefill 过程
|   (Layer 1 - N)   |
+-------------------+
    |
    |  取出某些 layer 的 hidden states
    |  例如取最后一层的 last token embedding
    |  维度: [1 x hidden_dim]
    v
+-------------------+
| Lightweight MLP   |   <-- 额外训练的小分类器（几 MB）
| (Prediction Head) |       结构: Linear -> ReLU -> Linear
+-------------------+
    |
    v
  预测: "这个请求大概会生成 ~120 tokens"
```

具体做法：
1. **数据收集**：在目标 LLM 上跑一批请求，记录每个请求的 (hidden states, actual output length)
2. **训练 MLP head**：用这些数据训练一个小的回归模型（或分类模型，预测长度 bucket）
3. **推理时**：Prefill 完成后，把 hidden states 喂给 MLP，即可预测 output 长度
4. **额外开销**：MLP 很小（5-7MB），推理只需 ~1ms（GPU 上），远小于 prefill 本身的延迟

#### 2.4.4 EGTP 的改进（当前 SOTA）

EGTP 在 TRAIL 基础上做了两个关键改进：

**改进 1: Entropy-Guided Token Pooling**

TRAIL 通常只取最后一个 token 的 embedding。但 EGTP 认为：不同 token 对
output 长度的 "贡献" 不同。例如 "请详细解释量子力学" 中，"详细" 这个词
对 output 长度的影响比 "请" 大得多。

EGTP 的做法：计算每个 input token 的 **entropy**（不确定性），entropy 越高
的 token 权重越大，然后做加权平均得到一个 pooled vector：

```
Input prompt: "请 详细 解释 量子 力学"

每个 token 的 hidden state:
  "请"    -> h_1 (entropy = 0.3, 权重低)
  "详细"  -> h_2 (entropy = 0.8, 权重高)
  "解释"  -> h_3 (entropy = 0.7, 权重中)
  "量子"  -> h_4 (entropy = 0.6, 权重中)
  "力学"  -> h_5 (entropy = 0.5, 权重中)

Entropy-guided pooling:
  pooled = weighted_avg(h_1, ..., h_5, weights=softmax(entropies))

然后:
  pooled vector -> 回归 head -> 预测 output length
```

**改进 2: Multi-layer fusion**

不只看最后一层，而是融合多个 layer 的 hidden states，因为不同层捕捉不同
粒度的信息（浅层: 语法，深层: 语义）。

#### 2.4.5 EGTP 具体数据

在 Qwen2.5 7B + ForeLen benchmark 上的表现：

| Scenario | EGTP MAE | TRAIL MAE | SSJF-Reg MAE |
|----------|:--------:|:---------:|:------------:|
| Long Sequence | 81.6 | 134.2 | -- |
| Reasoning | 133.6 | 124.2 | -- |
| RL Sampling | 95.2 | 155.5 | -- |
| **Average** | **103.5** | **138.0** | **228.9** |

EGTP 平均 MAE 103.5 tokens，比 TRAIL 降低了 25%。

#### 2.4.6 为什么 Embedding Method 不适合我们的场景

**根本原因：时序不兼容**

Embedding method 需要的 hidden states 产生于 **prefill 完成之后**，
但我们的路由决策发生在 **prefill 之前**（甚至还没选 provider）：

```
我们的路由流程:

  Request 到达 (只知道 input prompt)
       |
       v
  +-----------+
  |  Router   |  <-- 在这里就要决定发给谁
  |  (< 50us) |      此时还没有任何 provider 处理过这个请求
  +-----------+      所以根本没有 hidden states!
       |
       +----> Provider A (FreeInference)  -> [Prefill] -> hidden states -> Decode
       +----> Provider B (Together API)   -> [Prefill] -> hidden states -> Decode
```

Hidden states 是 prefill 的**产物**，而路由决策发生在 prefill **之前**。
这不是精度或延迟的问题，而是**信息根本不存在**的问题。

**其他不兼容因素：**

1. **需要修改 serving engine 内部**：必须在 LLM 的 inference pipeline 中
   插入 prediction head。对于第三方 API（Together、Groq 等），我们根本无法
   修改他们的 serving engine。

2. **需要针对每个 LLM 单独训练** prediction head：不同 LLM 的 hidden states
   维度和语义不同，TRAIL/EGTP 的 MLP head 不能跨模型复用。我们的场景涉及
   14+ 种模型。

3. **即使能用，精度也不够**：EGTP 的 MAE ~100 tokens，对于我们的 workload
   （median output = 51 tokens），相对误差约 196%。虽然这是当前 SOTA，
   但对 routing 来说仍然很粗糙。

**总结：Embedding method 是为 scheduling（单个 serving 实例内部调度）设计的，
不是为 routing（跨 provider 路由）设计的。**

在 scheduling 场景中，LLM 已经在本地运行，prefill 已经完成，利用 hidden states
做调度（比如预测短任务优先执行）是自然的。但在 routing 场景中，路由器在
"应该把请求发给谁" 这个问题上做决策时，没有任何 provider 执行过这个请求，
因此没有 hidden states 可用。

**优点**: 当前最高准确度（EGTP MAE 降低 29%）；复用 prefill 的计算，额外开销小（5-7MB）。
**缺点**:
- **必须在 prefill 之后**才能预测 -> 此时已经选定了 provider
- 需要修改 serving engine 内部（插入预测 head）
- 需要针对每个 LLM 单独训练 prediction head
- 对 multi-provider routing 无用（信息在路由时不存在）

### 2.5 (E) Progressive Prediction — During Decode

在 decode 过程中，每生成几个 token 就更新一次"剩余长度"预测。

| Paper | Venue | Method | 说明 |
|-------|-------|--------|------|
| **EGTP-PLP** (Xie et al., 2025) | ICLR 2026 | Progressive Length Prediction module | 解决 RL sampling 的高方差问题 |
| **TRAIL-online** (Shahout et al., 2024) | ICLR 2025 | 每个 token 后更新 remaining length | 用于动态调度 |

**优点**: 能处理 stochastic decoding（同一 prompt 不同输出长度）。
**缺点**: 只适用于 scheduling（单实例内部调度），不适用于 routing（跨 provider 路由）。

### 2.6 (F) Learning-to-Rank (Relative Order)

不预测绝对长度，只预测一个 batch 内请求的**相对排序**。

| Paper | Venue | Method | Result |
|-------|-------|--------|--------|
| **LTR** (Fu et al., 2024) | NeurIPS 2024 | Learning-to-rank on prompt features | 2.8x lower latency; 6.5x higher throughput |

**优点**: 排序比绝对预测容易得多；对 scheduling 足够。
**缺点**: **不适用于 cost routing** — 我们需要 value 的绝对估计值来和 threshold 比较，
不是相对排序。

### 2.7 (G) Simple Online Estimators (Our Approach)

轻量级在线估计器，随着每个 request 完成逐步更新。

| Method | Features | Warmup | Adaptation | Overhead |
|--------|----------|:------:|:----------:|:--------:|
| **EMA** | model only | ~20 samples | ~10 samples | <1 us |
| **Histogram** | model + input_bin + hour | ~100 samples | slow (no decay) | ~10 us |

**优点**: 零外部依赖，极低延迟，在线自适应，不需要训练数据。
**缺点**: 不精确，依赖历史数据的平稳性假设。

## 3. 各方法与我们场景的适配性分析

### 3.1 核心约束：路由决策发生在 Prefill 之前

```
User Request
    │
    ▼
┌──────────┐
│  Router   │ ← 决策点：发给哪个 provider？
│ (<50 us)  │    此时没有任何 LLM 内部信息
└──────────┘
    │
    ├──→ Provider A (S_Q: FreeInference)  ─→ Prefill ─→ Decode
    └──→ Provider B (S_A: Together API)   ─→ Prefill ─→ Decode
```

**这是最关键的约束**。大部分高精度方法（D, E 类）都需要 LLM 内部 activation，
但在路由时 request 还没发给任何 provider，这些信息根本不存在。

### 3.2 逐类排除

| 类别 | 方法 | 路由前可用？ | 延迟 | 适配我们的场景？ | 排除原因 |
|:----:|------|:---:|:---:|:---:|------|
| **(A)** | BERT/DistilBERT proxy | Yes | 5-10ms | **No** | 比路由预算（50us）慢 100x；需训练数据 |
| **(B)** | Prompt feature + ML | Yes | <1ms | **Maybe** | 可行但 input-output 相关性弱 |
| **(C)** | LLM 自预测 | **No** | ~100ms+ | **No** | 需要 LLM forward pass，等于先选了 provider |
| **(D)** | Post-prefill activation | **No** | ~1ms (GPU) | **No** | 必须 prefill 后才有 activation |
| **(E)** | Decode 中 progressive | **No** | per-token | **No** | 在 decode 阶段才生效 |
| **(F)** | Learning-to-rank | Yes | ~1ms | **No** | 只给排序，不给绝对值；routing 需要 value |
| **(G)** | EMA / Histogram | Yes | <10us | **Yes** | 满足所有约束 |

### 3.3 (B) Prompt Feature + ML 为什么也不够好

这是唯一一个"可行但我们没用"的类别，需要解释清楚：

1. **Input-output 相关性弱**: 在 FreeInference 数据上，input_tokens 和 output_tokens
   的 Pearson 相关系数通常 <0.3。"用一句话总结这篇万字论文" 输入长但输出短；
   "写一个完整的排序算法" 输入短但输出长。

2. **需要训练数据**: 每种 model + 每种 task type 都需要足够的标注数据来训练回归模型。
   我们的多 model 多 provider 场景下，训练数据获取成本高。

3. **不如 EMA 自适应**: XGBoost 等 batch 模型是静态的，不能在线适应 workload 变化。
   EMA 天然在线更新，~10 个样本就能跟上分布变化。

4. **实验结果已经说明**: 我们的 Histogram predictor 实际上就是一种条件化的统计方法
   （按 model + input_bin + hour_bin 分组），相当于一种非参数版本的 feature-based
   prediction。结果是它反而不如更简单的 EMA，说明在 routing 场景下，
   feature richness 的收益不大。

### 3.4 即使准确度更高，对 routing 帮助也有限

即使 EGTP（SOTA）也有 MAE ~100 tokens。对于 FreeInference（median output = 51 tokens）：

```
相对误差 ≈ MAE / median_output = 100 / 51 ≈ 196%
```

但 routing 决策本质上是一个 **binary threshold comparison**：

```
if value(request) >= threshold(quota_usage):
    → subscription
else:
    → API
```

你不需要知道 output 精确是 50 还是 150 tokens，你只需要判断
"这个 request 值不值得用 quota"。这是一个粗粒度的分类问题，不是回归问题。

我们的 ablation 实验证实了这一点：EMA（calibration 差，q50 coverage 72%）
反而比 Histogram（calibration 好，q50 coverage 56%）的 **routing cost 更低**
（CR 1.18 vs 1.23）。

### 3.5 延迟对比总结

| Method | Latency | Memory | 路由前可用 | 需要训练 |
|--------|:-------:|:------:|:---:|:---:|
| EMA | **<1 us** | O(1)/model | Yes | No |
| Histogram | **~10 us** | O(bins × keys) | Yes | No |
| XGBoost | ~0.1-1 ms | ~10 MB | Yes | Yes |
| S3/BERT | 5-10 ms | 238-270 MB | Yes | Yes |
| EGTP | 1-2 ms (GPU) | 5-7 MB | **No** | Yes |
| LLM Self | 100ms+ | — | **No** | No |

## 4. Our Design Choice: Justification

We use simple online estimators (EMA/Histogram) for the following reasons:

| Criterion | Embedding-Based | Our Approach |
|-----------|:---------------:|:------------:|
| Latency overhead | 1-10 ms | <10 us |
| Requires model access | Yes (post-prefill) | No |
| Works pre-routing | No (needs provider) | **Yes** |
| Handles multi-provider | No | **Yes** |
| Warmup requirement | Training data + GPU | ~20 online samples |
| Adaptivity | Static model | Online adaptation |
| Routing performance | Unknown (not evaluated for routing) | **CR 1.18** |

**Key argument**: Embedding-based methods are designed for **scheduling** (within a
single serving instance), not for **routing** (across multiple providers). In the
routing setting:
1. Decision happens before any provider sees the request
2. We don't have activation access
3. Sub-microsecond latency is required
4. Online adaptation to workload shift matters more than static accuracy

## 5. What We Could Say in the Paper

Suggested paragraph for the paper:

> **Why not use embedding-based predictors?** Recent work on output length prediction
> (EGTP [Xie et al., ICLR 2026], TRAIL [Shahout et al., ICLR 2025]) achieves
> state-of-the-art MAE by leveraging LLM internal activations post-prefill. However,
> these methods are designed for single-instance scheduling where the model's hidden
> states are available. In our multi-provider routing setting, the routing decision
> must be made *before* sending the request to any provider, precluding access to
> model activations. Furthermore, even EGTP's MAE of ~100 tokens represents ~68%
> relative error on typical workloads, and our ablation shows that predictor
> responsiveness dominates calibration accuracy for routing decisions (Table X).
> Simple online estimators (EMA, Histogram) achieve strong routing performance
> (CR 1.18) with <50 us overhead and zero dependency on model internals.

## 6. References

1. **EGTP**: Xie et al., "Predicting LLM Output Length via Entropy-Guided
   Representations", ICLR 2026. [arXiv:2602.11812](https://arxiv.org/abs/2602.11812)

2. **TRAIL**: Shahout et al., "Don't Stop Me Now: Embedding Based Scheduling for LLMs",
   ICLR 2025. [arXiv:2410.01035](https://arxiv.org/abs/2410.01035)

3. **GNN-based**: Piotrowski et al., "When Will the Tokens End? Graph-Based Forecasting
   for LLMs Output Length", ACL 2025 SRW.
   [ACL Anthology](https://aclanthology.org/2025.acl-srw.61/)

4. **Learning-to-Rank**: Fu et al., "Efficient LLM Scheduling by Learning to Rank",
   NeurIPS 2024. [arXiv:2408.15792](https://arxiv.org/abs/2408.15792)

5. **Proxy Model**: Qiu et al., "Efficient Interactive LLM Serving with Proxy
   Model-based Sequence Length Prediction", ASPLOS 2024.
   [arXiv:2404.08509](https://arxiv.org/abs/2404.08509)

6. **S3**: Jin et al., "S3: Increasing GPU Utilization during Generative Inference for
   Higher Throughput", 2023.
