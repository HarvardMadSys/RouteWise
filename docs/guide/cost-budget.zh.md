# 成本预算

`alpha` 是在花费和延迟之间做取舍的唯一旋钮。本页解释这个旋钮怎么起作用；下面几个
公式的权威定义在 [API 参考](../reference/api.md)。

## 估计成本

对每个可选供应商：

```text
(非缓存输入 * price_in + 缓存输入 * 缓存价格
 + 预测输出 * price_out) / 1,000,000
```

其中预测输出项：你传了 `estimated_output_tokens` 就用你的，否则用 RouteWise
内部的在线估计。

## 前缀缓存 {#prefix-cache}

缓存输入指 prompt 中可以由供应商前缀缓存直接提供的部分，它按该供应商的缓存价
进入成本估计：

- `Provider(..., price_cached=0.30)` 设定缓存价。不设时缓存输入按 `price_in` 计费。
- `route(estimated_cached_tokens=...)` 表示你预计命中缓存的 prompt token 数——
  可以是对所有供应商生效的一个整数，命中率不同时也可以按供应商名给 mapping。
  超过 `input_tokens` 的值会被截断。
- 请求完成后回报实际的 `cached_tokens`，让计费反映实际发生的，而不是估计值。

```python
decision = router.route(
    input_tokens=8_000,
    estimated_cached_tokens={"fast": 7_500, "cheap": 0},
)
```

缓存预期只移动决策中的成本一侧；延迟一侧来自你回报的结果。

## 预算

设可选供应商的成本极值为 `C_min` 和 `C_max`，预算为：

```text
budget = C_min + alpha * (C_max - C_min)
```

RouteWise 随后采样一个供应商混合，使其主尝试的期望成本落在该预算内，同时倾向
选择学习到的延迟更低的供应商。

| `alpha` | 行为 |
| ---: | --- |
| `0.0` | 最小成本预算 |
| `0.25` | 默认值。以成本为主，留出一些延迟余量 |
| `1.0` | 整个成本区间都可用于延迟优化 |

## 预算不是什么

!!! warning "这是期望混合预算，不是单请求上限"

    预算约束的是采样混合下主尝试的期望成本，不是任何单次请求的硬上限；
    并且一次派发的[对冲](hedging.md)是额外的尝试，可能增加花费。

## 单次覆盖

`alpha` 设在 router 上，也可以对单次调用覆盖：

```python
decision = router.route(input_tokens=800, alpha=0.0)
```

## 怎么选值

从 `alpha=0.0` 开始，测量尾延迟，然后逐步调高直到尾延迟满足目标。调高 `alpha`
会扩大路由器可以付费选择的供应商集合，它不会降低花费。
