# 核心概念

## 路由闭环

RouteWise 位于应用和供应商之间，本身不碰网络：它返回一个名字，其余由你完成。

1. 描述每个供应商及其价格。
2. 请求一次决策。
3. 由你自己派发请求。
4. 回报发生了什么。

第 4 步决定了第 2 步能否变好。没有结果回报，路由器就没有可学习的东西。

## Provider

```python
rw.Provider(name, price_in, price_out, price_cached=None)
```

不可变的供应商定义。价格是有限的非负数，单位为每百万 token 的美元。名称非空，
且在同一个 router 内唯一。未提供 `price_cached` 时，缓存输入按 `price_in` 计费。

## Router

```python
rw.Router(
    providers,
    *,
    alpha=0.25,
    slo_ms=None,
    seed=None,
    cold_start="explore",
    clock=None,
    tuning=None,
)
```

`alpha` 设定[成本预算](cost-budget.md)。`slo_ms` 启用[对冲](hedging.md)检查点。
`seed` 让采样可复现。

## Decision

`Router.route()` 返回一个 `Decision`，其中给出应使用的供应商：

```python
decision = router.route(input_tokens=800)
```

如果应用已经能预测生成长度，把这个点估计传进来。省略则使用内部在线估计。

```python
decision = router.route(
    input_tokens=800,
    estimated_output_tokens=predict_output_tokens(prompt),
)
```

该估计只影响路由和对冲的成本计算，不是实际用量。

## 结果回报

请求完成后，回报被采纳尝试的实际 `output_tokens`，或显式的 `cost_usd`，用于计费：

```python
decision.completed(ttft_ms=420.0, output_tokens=180)
```

正的实际输出 token 数还会更新内部的输出长度估计器。只有被采纳、已完成且输出
token 为正的尝试才会参与这项学习。

## 冷启动

默认的 `cold_start="explore"` 保留未建立画像的供应商的可选资格，并为选中的探索
目标持有一段租约。严格模式 `cold_start="require_observations"` 会排除未建立画像的
供应商，因此需要先播种：

```python
router.observe("provider-a", ttft_ms=240.0)
router.observe("provider-b", ttft_ms=310.0)
```

连续健康失败达到 `Tuning.cooldown_after` 次后触发冷却。一次 TTFT 成功即可清除。

## 统计

`router.stats()` 返回不可变的 `StatsSnapshot`，覆盖各供应商的选中次数、TTFT 分位数、
按健康与请求拆分的错误计数、冷却状态和花费。

!!! warning "不要重复计算对冲花费"

    对冲花费是供应商花费的一个横截面。把 `hedges.actual_spend_usd` 加到供应商
    合计上会重复计算。

## 进程作用域

观测、冷却、租约、估计、随机状态和计数器都存在于当前 Python 进程中，不做持久化，
也不跨进程共享。
