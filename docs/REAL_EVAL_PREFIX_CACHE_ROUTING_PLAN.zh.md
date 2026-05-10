# Real-Eval Prefix Cache Routing Plan

> 目标：把 prefix cache 加进 real-eval 的 **routing decision model**，只影响 RouteWise 的 API marginal cost / effective cost 计算；最终 cost comparison 仍然使用 OpenRouter/provider 返回的实际 `usage.cost`。

## 1. Decision

Real-eval 里 prefix cache 分成两套口径：

1. **Routing estimated cost**
   - 用于 `RouteWisePolicy.route()` 里的 `request_costs`、`c_eff`、LP budget、hedge backup selection。
   - 可以使用 length-based prefix-cache model：
     ```text
     cached_tokens = min(prompt_tokens, previous_context_tokens)
     estimated_api_cost =
       (prompt_tokens - cached_tokens) * input_price
       + cached_tokens * cached_input_price
       + completion_tokens * output_price
     ```
   - 这个 cost 只决定“选哪个 provider”。

2. **Billed cost**
   - 用于 CSV / summary 里的 `billed_cost_usd`、`primary_cost_usd`、`backup_cost_usd`、`total_cost_usd`。
   - 继续优先使用 OpenRouter/provider 返回的 `usage.cost`。
   - 不用我们的 cache model 覆盖最终账单。

这避免了一个危险解释：我们自己假设 cache 能省钱，然后 summary 也按假设省钱。real-eval 的 cost headline 必须是实际 API 返回账单。

## 2. Non-Goals

第一版不做这些：

- 不改变 latency profile、TTFT CDF、hedge success probability。
- 不改变 quota / concurrency capacity accounting。
- 不把 simulator 的 `20% input price` synthetic assumption 自动套到 real provider。
- 不对 OpenRouter native `sort_latency` / `sort_price` 做 pre-dispatch cache-aware routing，因为 native sort 最终选哪个 sub-provider 是 OpenRouter 决定的。
- 不把 prefix-cache estimated savings 直接写进 final billed cost。
- 不做 text LCP、semantic similarity、多候选 context、TTL/LRU。

## 3. Current Code Shape

相关入口：

- `experiments/real_evaluation/runner.py`
  - `TraceRequest` 目前只有 `arrival_time_sec`、`prompt`、`prompt_tokens`、`max_tokens`。
  - `_prepare_dispatch()` 构造 `RequestContext(prompt_tokens, completion_tokens_budget)`，调用 `policy.route()`。
  - `_execute_prepared_dispatch()` / `_execute_coalesced_group()` 负责真实 dispatch、feedback、accounting、recording。

- `experiments/real_evaluation/policies.py`
  - `RequestContext` 目前没有 `prefix_id`。
  - `request_cost_for_spec()` 调 `request_marginal_cost()`。
  - `BudgetRangePolicy.route()` 用 `request_costs` 和 `effective_cost()` 计算 `c_eff`。
  - `select_safe_cheapest_backup()` 目前按 uncached request cost 选 backup。

- `experiments/real_evaluation/shadow_price.py`
  - `request_marginal_cost()` 已经有 `TODO(routewise-cache)`。
  - `calibrate_envelopes()` 也有 cache TODO。

- `experiments/real_evaluation/transports.py`
  - `SingleRequestResult.billed_cost_usd` 是最终 recorder 使用的账单。
  - `OpenAICompatStreamingTransport` 优先用 `usage.cost`，没有 reported cost 时才 fallback 到 `_estimate_cost()`。
  - 目前还没有把 provider-reported cached input tokens 解析成结构化字段；Phase A 需要补上，用于 predicted-vs-observed cache calibration。

## 4. Data Model

### 4.1 TraceRequest

给 `TraceRequest` 增加：

```python
prefix_id: str | None = None
```

`load_trace_jsonl()` 从 trace row 里按下面优先级取：

```text
prefix_id
sharegpt_conversation_id
session_id
```

不默认用 `user_id`。原因是 real-eval 当前主要跑 ShareGPT/BurstGPT 风格 conversation chain；`user_id` 会把多个不相关对话合并成一个 cache namespace，第一版先不这么做。

### 4.2 RequestContext

给 `RequestContext` 增加：

```python
prefix_id: str | None = None
```

Policy route 时只需要 token length 和 prefix namespace，不需要 prompt text。

### 4.3 ProviderSpec / TransportConfig

Real provider 不使用 synthetic 20%。新增可选真实价格字段：

```python
cached_input_price_per_m: float | None = None
```

Inventory JSON 中只有明确写了 `cached_input_price_per_m` 的 API provider 才启用 cache-aware routing discount。没有该字段表示：

```text
provider does not support modeled cache discount
```

或者：

```text
we do not trust a cached input price for this provider yet
```

## 5. Policy State

每个 policy 独立维护 cache state，和 latency profile / capacity state 一样隔离：

```python
provider_prefix_cache: dict[str, dict[str, int]]
```

含义：

```text
provider_prefix_cache[provider_name][prefix_id] = previous_context_tokens
```

`previous_context_tokens` 是这个 policy 上一次 dispatch 到该 provider 后，该 session 在 provider 侧可复用的 context length 近似：

```text
prompt_tokens + completion_tokens
```

这是 Juncheng 说的 “significant overlap in context length with user's previous request” 的最小实现。

## 6. Routing Cost Helper

建议在 real-eval 新增一个小 helper，例如：

```text
experiments/real_evaluation/prefix_cache.py
```

核心函数：

```python
def cached_input_tokens(
    *,
    provider_name: str,
    prefix_id: str | None,
    prompt_tokens: int,
    provider_prefix_cache: dict[str, dict[str, int]],
) -> int:
    if prefix_id is None:
        return 0
    previous_context = provider_prefix_cache.get(provider_name, {}).get(prefix_id, 0)
    return min(prompt_tokens, previous_context)
```

```python
def cache_aware_request_cost_usd(
    spec: ProviderSpec,
    ctx: RequestContext,
    provider_prefix_cache: dict[str, dict[str, int]],
) -> float:
    if spec.tier != "api":
        return 0.0
    if spec.cached_input_price_per_m is None:
        return compute_request_cost_usd(...)

    cached = cached_input_tokens(...)
    uncached = max(ctx.prompt_tokens - cached, 0)
    return (
        spec.input_price_per_m * uncached
        + spec.cached_input_price_per_m * cached
        + spec.output_price_per_m * ctx.completion_tokens_budget
    ) / 1_000_000.0
```

```python
def record_prefix_cache_dispatch(
    *,
    provider_name: str,
    prefix_id: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    provider_prefix_cache: dict[str, dict[str, int]],
) -> None:
    if prefix_id is None:
        return
    provider_prefix_cache.setdefault(provider_name, {})[prefix_id] = (
        max(prompt_tokens, 0) + max(completion_tokens, 0)
    )
```

## 7. RouteWise Integration

### 7.1 BasePolicy

`BasePolicy.__init__()` 增加：

```python
self.provider_prefix_cache = {}
```

并提供两个 thread-safe 方法：

```python
def request_cost_for_state(self, state: ProviderState, ctx: RequestContext) -> float
def record_prefix_cache_dispatch(...)
```

这些方法复用现有 `self._lock`，避免 replay worker 并发修改 cache state。

### 7.2 BudgetRangePolicy.route()

把当前：

```python
L, U = calibrate_envelopes(self.specs, ctx.prompt_tokens, ctx.completion_tokens_budget)
request_costs = {
    s.spec.name: request_cost_for_spec(s.spec, ctx) for s in feasible
}
```

改成：

```python
L, U = calibrate_envelopes(..., provider_prefix_cache=self.provider_prefix_cache, ctx=ctx)
request_costs = {
    s.spec.name: self.request_cost_for_state(s, ctx) for s in feasible
}
```

这样 `c_eff` 里的 S_A cost 是 cache-aware；S_Q/S_C shadow price 不变。

### 7.3 Hedge Backup Selection

`select_safe_cheapest_backup()` 现在用 `request_cost_for_spec()` 排 safe backup。第一版要么：

1. 增加可选 `cost_fn`，由 policy 传入 cache-aware cost；推荐。
2. 或让它接收 `provider_prefix_cache` 和 `ctx.prefix_id`。

推荐方案：

```python
select_safe_cheapest_backup(..., cost_fn=lambda state: policy.request_cost_for_state(state, ctx))
```

这样 hedging module 不需要直接知道 cache state。

## 8. Dispatch Ordering

Cache state 必须是 pre-dispatch lookup、post-dispatch update。

### 8.1 Single Request

顺序：

1. `_prepare_dispatch()` 调 `policy.route()`，route 使用 dispatch 前 cache state。
2. `policy.charge_capacity(primary, ...)` 保持现状。
3. `_execute_prepared_dispatch()` 真实发送 request。
4. 收到 `SingleRequestResult` 后：
   - feedback latency profile
   - account billed cost
   - record row
   - update policy prefix cache

第 4 步里 cache update 可以放在 record 前或 record 后，但必须使用实际返回的 completion token count：

```text
completion_tokens = result.completion_tokens
```

如果 provider 没返回 completion tokens，可以 fallback 到 `req.max_tokens` 或 0。建议第一版 fallback 到 result 里的值；没有就 0，避免夸大 cache state。

### 8.2 Hedged Request

Primary：

- primary 一定 dispatch，所以成功/失败/取消都可以更新 provider cache 吗？
- 第一版建议只在 `result.prompt_tokens > 0 or result.completion_tokens > 0` 时更新。

Backup：

- 只有 `hedged.backup_result is not None` 时更新。
- 如果 backup 没 fire，不更新。

Canceled loser：

- 如果 cancellation 后没有 final usage，`completion_tokens` 可能是 0。
- 这时最多记录 `prompt_tokens + observed_completion_tokens`。
- 不要凭 trace `max_tokens` 假设 canceled loser 完整生成。

## 9. Billing And Recorder

### 9.1 Final Cost

`Recorder.write_request()` 和 `write_hedged()` 继续使用：

```python
primary_result.billed_cost_usd
backup_result.billed_cost_usd
```

这些值来自 transport：

```text
OpenRouter usage.cost if present
else local fallback estimate
```

不要用 routing estimated cost 替换这些字段。

### 9.2 Observed Cache Tokens

Transport 需要解析 provider/OpenRouter response 中的实际 cache-hit token 数，但这些字段只用于诊断，不参与账单覆盖。

建议给 `SingleRequestResult` 增加：

```python
cache_read_tokens_observed: int | None = None
cost_source: str = "missing"  # "reported" | "estimated" | "missing"
```

需要兼容的 usage 字段：

```text
OpenAI-style:
  usage.prompt_tokens_details.cached_tokens

Anthropic-style:
  usage.cache_read_input_tokens

OpenRouter passthrough:
  优先解析上面两类字段；如果 OpenRouter 后续暴露别名，再加 alias。
```

这些 observed 字段的用途：

- 对比 routing predicted cache hit 和 provider observed cache hit。
- 统计 OpenRouter native baselines 自己是否也命中了 provider cache。
- 在 appendix 里说明 cache model 的预测误差。

它们不改变：

```text
billed_cost_usd
primary_cost_usd
backup_cost_usd
total_cost_usd
```

### 9.3 Diagnostic Fields

为了 debug 和 paper appendix，可以增加非账单字段：

```text
primary_cached_input_tokens
backup_cached_input_tokens
primary_observed_cached_input_tokens
backup_observed_cached_input_tokens
primary_routing_estimated_cost_usd
backup_routing_estimated_cost_usd
cost_source
```

其中：

- `*_cached_input_tokens` 来自 route-time cache lookup。
- `*_observed_cached_input_tokens` 来自 provider response usage 字段。
- `*_routing_estimated_cost_usd` 只解释 policy decision。
- `cost_source` 建议来自 transport：
  - `reported`：使用 OpenRouter/provider `usage.cost`
  - `estimated`：没有 reported cost，使用本地 price fallback
  - `missing`：没有 cost

Summary cost comparison 仍然只聚合 `billed_cost_usd`。

### 9.4 Cache Savings Metric

不要从单次 run 的 `routing_estimated_cost_usd - billed_cost_usd` 推导 savings；那只是 model-vs-bill 差异。

真实 cache-aware routing savings 只能来自 paired runs：

```text
cache_savings_usd =
  total_cost_usd(RouteWise no-cache-routing)
  - total_cost_usd(RouteWise cache-routing)
```

paired runs 必须使用同一 trace、同一 inventory、同一 SLO、同一 policy family。OpenRouter native baselines 作为外部对照，不参与这个内部差分。

### 9.5 Paper-Grade Cost Caveat

如果某次 run 中 `estimated` / `missing` 占比很高，不能把它当强 paper-grade cost claim。尤其 hedged cancellation loser 可能拿不到 final usage/cost。

## 10. OpenRouter Native Baselines

`openrouter_auto`、`sort_latency`、`sort_price` 是 sentinel policy：

```text
policy returns sentinel -> runner asks OpenRouter native router
```

第一版不让这些 baselines 做 cache-aware pre-dispatch decisions。理由：

- route 前不知道 OpenRouter 最终选哪个 sub-provider；
- provider-local prefix cache state 没有明确 key；
- native sort 是 baseline，不需要我们模拟它的 internal cost model。

但最终 billed cost 仍然用 OpenRouter 返回的 `usage.cost`。如果 OpenRouter/provider 真给了 cache discount，它自然会反映到 baseline cost 里。

## 11. CLI / Inventory

### 11.1 Inventory

给 API provider entry 支持：

```json
{
  "name": "OR_Friendli",
  "tier": "api",
  "input_price_per_m": 0.118,
  "cached_input_price_per_m": 0.0236,
  "output_price_per_m": 0.99
}
```

没有 `cached_input_price_per_m` 就不做 routing discount。

`refresh_inventory.py` 也要同步维护这个字段。OpenRouter endpoints API 里如果返回：

```text
ep["pricing"]["input_cache_read"]
```

则写入：

```text
cached_input_price_per_m = input_cache_read * 1_000_000
```

如果字段缺失、为 `None`、或值 `<= 0`，写成 `null` / 省略该字段，表示该 provider 不参与 modeled cache discount。这样 real-eval 不需要人工给每个 OpenRouter provider 填 cached input price。

### 11.2 CLI

建议加一个明确开关：

```text
--prefix-cache-routing
```

默认 off，保证历史 real-eval 行为不变。

开关含义：

```text
on: RouteWise decision cost uses prefix cache where provider has cached_input_price_per_m.
off: RouteWise decision cost uses uncached input price.
```

不需要 `--cached-input-price-fraction`。real-eval 用 inventory 真实价格，不用 synthetic fraction。

## 12. Tests

最小测试集：

1. `load_trace_jsonl()` 能读 `sharegpt_conversation_id` / `session_id` 为 `prefix_id`。
2. 缺少 prefix id 时 cache hit 为 0。
3. Provider A 的 cache state 不影响 Provider B。
4. Policy A 的 cache state 不影响 Policy B。
5. `BudgetRangePolicy.route()` 在 cache hit 后使用更低 S_A request cost。
6. `select_safe_cheapest_backup()` 使用 cache-aware cost function。
7. Single dispatch 后更新 primary provider cache。
8. Hedged backup fired 后更新 backup provider cache；未 fired 不更新。
9. Recorder billed cost 仍来自 `SingleRequestResult.billed_cost_usd`，不会被 routing estimated cost 覆盖。
10. Transport 能解析 `usage.prompt_tokens_details.cached_tokens` 和 `usage.cache_read_input_tokens`。
11. `refresh_inventory.py` 能把 OpenRouter `pricing.input_cache_read` 写成 `cached_input_price_per_m`。
12. `--prefix-cache-routing` 默认 off 时，现有 real-eval unit tests 输出保持不变。

## 13. Rollout

建议分两步：

### Phase A: Routing-Only Cache

- 加 data model / helper / policy state。
- 只改 RouteWise cost calculation。
- Final cost 仍用 reported billed cost。
- 加 diagnostic fields。
- 跑 small smoke trace：100-500 requests，覆盖至少 3 个 provider，并包含至少一个 multi-turn conversation。目标是验证 routing path、dispatch order、cache state isolation、observed cache token parsing，不追求 statistical significance。

### Phase B: Live Pilot

用同一份 trace 跑 paired policies：

```text
RouteWise no-cache-routing
RouteWise cache-routing
OpenRouter auto
OpenRouter sort_latency
OpenRouter sort_price
```

关注：

- RouteWise provider mix 是否因为 cache hit 改变。
- `usage.cost` 的 actual total cost 是否下降。
- `cost_source=reported` 占比是否足够高。
- latency 是否因为更多 API routing 改变。
- hedging backup fire / winner 行为是否变化。

## 14. Expected Paper Wording

可以这样写：

```text
In the live runner, prefix caching affects only RouteWise's routing-time
cost model. For each policy and provider, we maintain a per-session
length-based cache state; if a later request in the same session is routed
to the same provider, the cached portion of input tokens is priced at the
provider's recorded cached-input rate for the purpose of effective-cost
routing. Reported cost metrics remain based on OpenRouter/provider returned
usage.cost, not on our cache estimator.
```

中文解释：

```text
我们只让 cache 改变 RouteWise 的“选择”，不让 cache 模型直接改最终账单。
最终 cost 是 provider 实际返回的 usage.cost，因此 cache 是否真的省钱由 live bill 验证。
```
