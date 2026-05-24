# Real-Eval Prefix Cache Routing Plan

> 目标：把 prefix cache 加进 real-eval 的 **routing decision model**，只影响 RouteWise 的 API marginal cost / effective cost 计算；最终 cost comparison 仍然使用 OpenRouter/provider 返回的实际 `usage.cost`。

最后更新：2026-05-11。

## 1. Decision

Real-eval 里 prefix cache 分成两套口径：

1. **Routing estimated cost**
   - 用于 `RouteWisePolicy.route()` 里的 `request_costs`、`c_eff`、LP budget、hedge backup selection。
   - cache hit token 数直接来自 trace（freeinference 的 `cache_read_tokens`），不是我们自己构造的 provider-local prefix model：
     ```text
     cached_tokens = min(prompt_tokens, trace_cached_input_tokens)
     estimated_api_cost =
       (prompt_tokens - cached_tokens) * input_price
       + cached_tokens * cached_input_price
       + completion_tokens * output_price
     ```
   - 如果 trace 没给这个字段（或值为空/None），保守地视为 cold miss（`cached_tokens = 0`）。
   - 这个 cost 只决定“选哪个 provider”。

2. **Billed cost**
   - 用于 CSV / summary 里的 `billed_cost_usd`、`primary_cost_usd`、`backup_cost_usd`、`total_cost_usd`。
   - 继续优先使用 OpenRouter/provider 返回的 `usage.cost`。
   - 不用我们的 cache model 覆盖最终账单。

这避免了一个危险解释：我们自己假设 cache 能省钱，然后 summary 也按假设省钱。real-eval 的 cost headline 必须是实际 API 返回账单。

## 2. Why Not Provider-Local Prefix Model

旧设计在 policy 里维护

```python
provider_prefix_cache: dict[str, dict[str, int]]
```

并按 `prompt_tokens + completion_tokens` 累计 previous context length。问题：

- 它假设“同一个 session 下一轮 prompt 会复用全部上文”，对真实流量过于乐观，会高估 cache hit 概率，进而把 API provider 的 effective cost 拉得过低，导致 routing 决策不公平。
- 它和 freeinference trace 自带的真实 `cache_read_tokens` 重复，且总是和后者不一致。
- 它在 hedged dispatch / canceled loser / 429 fallback 之后还要 fork 出多个 update 路径，复杂且容易和 billing 错配。

因此我们改成：**trace field 是唯一 truth；缺失即 cold**。Simulator 和 real-eval 走同一套语义。

## 3. Non-Goals

第一版不做这些：

- 不改变 latency profile、TTFT CDF、hedge success probability。
- 不改变 quota / concurrency capacity accounting。
- 不重新引入 provider-local prefix model 作为缺失字段的 fallback。
- 不对 OpenRouter native `sort_latency` / `sort_price` 做 pre-dispatch cache-aware routing，因为 native sort 最终选哪个 sub-provider 是 OpenRouter 决定的。
- 不把 prefix-cache estimated savings 直接写进 final billed cost。
- 不做 text LCP、semantic similarity、tokenizer-level exact prefix matching。

## 4. Data Model

### 4.1 TraceRequest

`TraceRequest` 增加：

```python
prefix_id: str | None = None
trace_cached_input_tokens: int | None = None
```

`load_trace_jsonl()` 从 trace row 里：

- `prefix_id`: 按 `prefix_id` → `sharegpt_conversation_id` → `session_id` 顺序取第一个非空值；都没有就 `None`。仍保留 prefix id 主要是为了 diagnostics / 后续 sensitivity，不直接参与 cost 计算。
- `trace_cached_input_tokens`: 按 `cache_read_tokens` → `cached_input_tokens` 顺序取第一个非空数值；都没有就 `None`，下游按 cold miss 处理。和 simulator 接受的 key 保持一致。

### 4.2 RequestContext

`RequestContext` 增加：

```python
prefix_id: str | None = None
trace_cached_input_tokens: int | None = None
```

`_prepare_dispatch()` 把 `TraceRequest.trace_cached_input_tokens` 直接拷贝进 context。Policy route 只需要 token length、prefix namespace 和这个 trace 字段，不需要 prompt text。

### 4.3 ProviderSpec / TransportConfig

Real provider 不使用 synthetic 20%。`ProviderSpec.cached_input_price_per_m: float | None` 仍然是 routing discount 的门槛：

```text
None  -> 不应用 cache discount，即使 trace 有 cache_read_tokens 也按 input_price 计费
float -> 应用 cache discount
```

`refresh_inventory.py` 仍负责从 OpenRouter endpoints `pricing.input_cache_read` 填这个字段。

## 5. Policy State

Policy 不再保存任何 prefix-cache 状态。`BasePolicy` 只持有：

- `states[provider]` (latency profile + capacity)
- `prefix_cache_routing: bool` 作为 master switch

`provider_prefix_cache`、`record_prefix_cache_dispatch` 已删除。Runner 也不再调用 post-dispatch cache update。

## 6. Routing Cost Helper

`experiments/real_evaluation/prefix_cache.py` 暴露两个函数：

```python
def cached_input_tokens(
    *,
    prompt_tokens: int,
    trace_cached_input_tokens: int | None,
) -> int:
    if trace_cached_input_tokens is None:
        return 0
    return min(max(prompt_tokens, 0), max(trace_cached_input_tokens, 0))


def cache_aware_request_cost_usd(
    spec: ProviderSpec,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    trace_cached_input_tokens: int | None,
    enabled: bool,
) -> float:
    if spec.tier != "api":
        return 0.0
    if not enabled or spec.cached_input_price_per_m is None:
        return compute_request_cost_usd(
            prompt_tokens,
            completion_tokens,
            spec.input_price_per_m,
            spec.output_price_per_m,
        )
    cached = cached_input_tokens(
        prompt_tokens=prompt_tokens,
        trace_cached_input_tokens=trace_cached_input_tokens,
    )
    uncached = max(prompt_tokens - cached, 0)
    return (
        spec.input_price_per_m * uncached
        + spec.cached_input_price_per_m * cached
        + spec.output_price_per_m * max(completion_tokens, 0)
    ) / 1_000_000.0
```

注意：这两个函数都不依赖 `provider_name` 或任何 policy state；同一个 trace request 在所有候选 provider 上看到的 `cached_tokens` 是一样的。这是有意的：trace 自带的 `cache_read_tokens` 描述的是一次真实历史 dispatch 在某 provider 上的命中量，我们用它作为 baseline locality 信号，不再 per-provider 假设。

## 7. RouteWise Integration

### 7.1 BasePolicy

`request_cost_for_state(state, ctx)` 直接调用 `cache_aware_request_cost_usd(state.spec, ..., trace_cached_input_tokens=ctx.trace_cached_input_tokens, enabled=self.prefix_cache_routing)`。

`routing_cache_diagnostics(provider, ctx)` 返回 `(cached_tokens, estimated_cost)`，仅用于 recorder 写诊断列。

### 7.2 BudgetRangePolicy.route()

`request_costs = {s.spec.name: self.request_cost_for_state(s, ctx) for s in feasible}`。`c_eff` 里的 S_A cost 是 cache-aware；S_Q/S_C shadow price 不变。

### 7.3 Hedge Backup Selection

```python
select_checkpoint_backup(
    ...,
    cost_fn=lambda state: policy.request_cost_for_state(state, ctx),
)
```

hedging module 不需要直接知道 cache 字段。

## 8. Dispatch Ordering

Cache state 只是 read-only 的 trace 字段；不再需要 post-dispatch update。Pipeline 简化为：

1. `_prepare_dispatch()` 构造 `RequestContext`（含 `trace_cached_input_tokens`），调用 `policy.route()`。
2. `policy.charge_capacity(primary, ...)`。
3. `_execute_prepared_dispatch()` 真实发送 request。
4. 收到 `SingleRequestResult` 后：
   - feedback latency profile
   - account billed cost
   - record row
   - **不再** 更新 policy prefix cache

Hedged / 429 fallback / canceled loser 同样不需要 cache update。

## 9. Billing And Recorder

### 9.1 Final Cost

`Recorder.write_request()` 和 `write_hedged()` 继续使用 `result.billed_cost_usd`，来自：

```text
OpenRouter usage.cost if present
else local fallback estimate
```

不要用 routing estimated cost 替换。

### 9.2 Observed Cache Tokens

Transport 解析 provider response 中的实际 cache-hit token 数，写进 `SingleRequestResult.cache_read_tokens_observed`，仅用于诊断（不参与账单覆盖）。需要兼容的 usage 字段：

```text
OpenAI-style:   usage.prompt_tokens_details.cached_tokens
Anthropic-style: usage.cache_read_input_tokens
OpenRouter:     上面两类字段的 passthrough
```

### 9.3 Diagnostic Fields

Recorder 写出：

```text
primary_cached_input_tokens             # route-time, from trace field
backup_cached_input_tokens
primary_observed_cached_input_tokens    # response-time, from provider usage
backup_observed_cached_input_tokens
primary_routing_estimated_cost_usd
backup_routing_estimated_cost_usd
cost_source                             # reported | estimated | missing
```

可以直接对比 `cached_input_tokens` (trace-predicted) 与 `observed_cached_input_tokens` (provider-reported)，验证 trace 字段在 cross-provider replay 下的代表性。

Summary cost comparison 仍然只聚合 `billed_cost_usd`。

### 9.4 Cache Savings Metric

不要从单次 run 的 `routing_estimated_cost_usd - billed_cost_usd` 推导 savings。真实 cache-aware routing savings 只能来自 paired runs：

```text
cache_savings_usd =
  total_cost_usd(RouteWise prefix_cache_routing=off)
  - total_cost_usd(RouteWise prefix_cache_routing=on)
```

paired runs 必须使用同一 trace、同一 inventory、同一 SLO、同一 policy family。OpenRouter native baselines 作为外部对照，不参与这个内部差分。

### 9.5 Paper-Grade Cost Caveat

如果某次 run 中 `estimated` / `missing` 占比很高，不能把它当强 paper-grade cost claim。尤其 hedged cancellation loser 可能拿不到 final usage/cost。

## 10. OpenRouter Native Baselines

`openrouter_auto`、`sort_latency`、`sort_price` 是 sentinel policy；它们不做 cache-aware pre-dispatch decisions。最终 billed cost 仍然用 OpenRouter 返回的 `usage.cost`，如果 OpenRouter/provider 真给了 cache discount，它自然会反映到 baseline cost 里。

## 11. CLI / Inventory

### 11.1 Inventory

API provider entry 支持：

```json
{
  "name": "OR_Friendli",
  "tier": "api",
  "input_price_per_m": 0.118,
  "cached_input_price_per_m": 0.0236,
  "output_price_per_m": 0.99
}
```

没有 `cached_input_price_per_m` 就不做 routing discount，即使 trace 提供了 `cache_read_tokens` 也按 input price 计费。

### 11.2 CLI

```text
--prefix-cache-routing
```

默认 off。

```text
on:  RouteWise decision cost 应用 trace-reported cache discount（provider 必须有 cached_input_price_per_m）
off: RouteWise decision cost 全部按 uncached input price
```

没有 `--cached-input-price-fraction`。real-eval 用 inventory 真实价格。

## 12. Tests

最小测试集：

1. `load_trace_jsonl()` 能从 `cache_read_tokens` / `cached_input_tokens` 字段填充 `TraceRequest.trace_cached_input_tokens`，缺失即 `None`。
2. `cached_input_tokens(...)` 在 `trace_cached_input_tokens is None` 时返回 0；非空时按 `min(prompt_tokens, value)` 截断。
3. `BudgetRangePolicy.route()` 在 trace 给出 cache hit 后使用更低 S_A request cost；trace 没给时退回 uncached price。
4. `select_checkpoint_backup()` 通过 `cost_fn` 复用 cache-aware cost function。
5. Recorder billed cost 仍来自 `SingleRequestResult.billed_cost_usd`，不会被 routing estimated cost 覆盖；`primary_cached_input_tokens` 写出 route-time trace 值，`primary_observed_cached_input_tokens` 写出 provider observed 值。
6. Transport 能解析 `usage.prompt_tokens_details.cached_tokens` 和 `usage.cache_read_input_tokens`。
7. `refresh_inventory.py` 能把 OpenRouter `pricing.input_cache_read` 写成 `cached_input_price_per_m`。
8. `--prefix-cache-routing` 默认 off 时，现有 real-eval unit tests 输出保持不变。

## 13. Rollout

### Phase A: Routing-Only Cache

- TraceRequest / RequestContext 新增 `trace_cached_input_tokens`。
- `prefix_cache.py` 只保留 trace-only helper。
- 只改 RouteWise cost calculation。
- Final cost 仍用 reported billed cost。
- 加 diagnostic fields。
- 跑 small smoke trace：100-500 requests，覆盖至少 3 个 provider，并包含至少一个 multi-turn conversation。目标是验证 routing path、observed-vs-trace cache token parsing，不追求 statistical significance。

### Phase B: Live Pilot

用同一份 freeinference trace 跑 paired policies：

```text
RouteWise prefix_cache_routing=off
RouteWise prefix_cache_routing=on
OpenRouter auto
OpenRouter sort_latency
OpenRouter sort_price
```

关注：

- RouteWise provider mix 是否因为 trace-reported cache hit 改变。
- `usage.cost` 的 actual total cost 是否下降。
- `cost_source=reported` 占比是否足够高。
- predicted (`primary_cached_input_tokens`) 与 observed (`primary_observed_cached_input_tokens`) 的差距分布。
- hedging backup fire / winner 行为是否变化。

## 14. Expected Paper Wording

可以这样写：

```text
In the live runner, prefix caching affects only RouteWise's routing-time
cost model. The cached-input-token count for each request is taken directly
from the trace's reported cache_read_tokens field; when that field is
absent we treat the request as a cold miss. The cached portion is priced
at the provider's recorded cached-input rate for effective-cost routing.
Reported cost metrics remain based on OpenRouter/provider-returned
usage.cost, not on our cache estimator.
```

中文解释：

```text
我们只让 trace-reported cache 改变 RouteWise 的"选择"，不让 cache 模型
直接改最终账单。最终 cost 是 provider 实际返回的 usage.cost，因此 cache
是否真的省钱由 live bill 验证。
```
