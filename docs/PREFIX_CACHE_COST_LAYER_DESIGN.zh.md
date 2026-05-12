# Prefix Cache Cost Layer Design

> 目标：把 prefix cache 作为简单、可解释的 cost-layer feature 加进 simulator。第一版只改 API input-token billing，不改 latency、queueing、capacity accounting、hedging success probability。

最后更新：2026-05-11。

---

## 1. TL;DR

第一版采用 **trace-driven** 的 cache-hit 信号：

```text
same request in the freeinference trace already reports cache_read_tokens
=> use that value (capped by prompt length) as cached input tokens
=> discount API input-token cost
```

如果 trace 没给这个字段（或值为空/None），保守地视为 cold miss，不做 discount。

核心字段是 request metadata 里 trace loader 透传的：

```text
request.metadata["cache_read_tokens"]      # freeinference 主字段
request.metadata["cached_input_tokens"]    # 兼容别名
```

route 时，对每个 S_A provider 计算：

```python
trace_cached = trace_observed_cached_input_tokens(request)  # int | None
if trace_cached is None or provider.tier != S_A or provider.cached_input_cost_per_token is None:
    cached_input_tokens = 0
else:
    cached_input_tokens = min(request.request_tokens, trace_cached)

uncached_input_tokens = request.request_tokens - cached_input_tokens
```

API cost：

```text
cost =
  uncached_input_tokens * input_price
  + cached_input_tokens * cached_input_price
  + output_tokens * output_price
```

simulator **不再** 维护任何 provider-local prefix cache state，也不在 dispatch 后做 cache update。

---

## 2. Why Not Provider-Local Prefix Model

旧设计在 `SimulationState` 里维护

```python
provider_prefix_cache: dict[str, dict[str, int]]
```

并按 `request_tokens + response_tokens` 累加 previous context length，dispatch 后更新；route 时 `cached_input_tokens = min(prompt_tokens, previous_context_tokens)`。问题：

- 它假设“同一个 session 下一轮 prompt 一定复用全部上文”，对真实流量过于乐观，会高估 cache hit 概率。
- freeinference 这种真实 API trace 自带 `cache_read_tokens`，它才是 ground truth。我们自己构造的 length-based proxy 通常和 ground truth 不一致。
- post-dispatch update 把 cache state 和 hedged/canceled/billing 路径耦合，复杂且容易错配。

因此 simulator 改成：**trace 字段是唯一信号源；缺失就是 cold miss**。Real-eval 走同一套语义，保证两边一致。

---

## 3. Non-Goals

第一版明确不做：

- 不维护 provider-local prefix cache state。
- 不在缺失 trace 字段时合成 cache hit。
- 不做 prompt text longest-common-prefix。
- 不做 tokenizer-level exact prefix matching。
- 不做 embedding / semantic similarity。
- 不做 provider-specific cache hit probability。
- 不改变 TTFT distribution。
- 不改变 queueing / capacity accounting / service duration。
- 不改变 quota / concurrency shadow-price 公式。
- 不在 cache-enabled ablation 中 claim existing exact offline oracle 仍是 strict optimum。

---

## 4. Request Metadata

### 4.1 Trace Cached Input Tokens

`Request.metadata` 透传 trace 原始字段：

```text
request.metadata["cache_read_tokens"]    # freeinference 主键，优先
request.metadata["cached_input_tokens"]  # 简化别名
```

helper `trace_observed_cached_input_tokens(request)` 按上面顺序返回第一个非空的整数；都没有就 `None`。

Loader 不再要求生成 `prefix_id`；conversation/session id 只在 diagnostics / 后续 sensitivity 里可选使用。

### 4.2 Cap By Prompt Length

helper `cached_input_tokens(provider, request, state)` 把 trace 值按 `min(prompt_tokens, value)` 截断，避免 trace 把 0 prompt 之外的 token 也算成 hit。如果 trace 给的是 0，则按 cold miss 处理（这正是 "authoritative zero" 行为：trace 明确告诉我们没有 cache hit）。

---

## 5. Provider Pricing

Provider 增加可选字段：

```python
cached_input_cost_per_token: float | None = None
```

语义：

```text
None  -> provider 不支持 cached-input discount；即使 trace 报告 cache hit 也按 input_price 计费
float -> cached input token 单价
```

如果没有真实 cached-input price，可以在 scenario factory 里用 fraction 生成：

```text
cached_input_cost_per_token = input_cost_per_token * cached_input_price_fraction
```

Phase A synthetic 实验固定 `cached_input_price_fraction = 0.2`，即对 synthetic API price `{1, 2, 4}`，cache-read input price 为 `{0.2, 0.4, 0.8}`。这是固定 design choice，不是主实验 knob。

不同 tier 的处理：

```text
S_A: use cache-aware API token billing
S_Q: keep quota shadow price unchanged
S_C: keep concurrency shadow price unchanged
```

如果 provider 只有 legacy blended `cost_per_token`，cache discount 默认 disabled。

---

## 6. Runtime State

`SimulationState` **不** 再持有 `provider_prefix_cache` 字段。

唯一保留的 master switch 是 scenario metadata：

```python
scenario.metadata["prefix_cache_enabled"] = True | False
```

`SimulationState.metadata` 把它复制下来；helper 用它作为 cache-aware billing 的总开关。

---

## 7. Helper API

`rwsim/policies/prefix_cache.py` 暴露：

```python
def trace_observed_cached_input_tokens(request: Request) -> int | None:
    """读 request.metadata 的 cache_read_tokens / cached_input_tokens。"""

def cached_input_tokens(
    provider: Provider,
    request: Request,
    state: SimulationState,
    *,
    cache_enabled: bool = True,
) -> int:
    """trace 字段缺失 → 0；否则按 min(request_tokens, value) 截断。"""

def cache_aware_marginal_cost(
    provider: Provider,
    request: Request,
    state: SimulationState,
    *,
    now: float,
    cache_enabled: bool = True,
) -> float:
    """走 cached_input_tokens(...) → provider.marginal_cost_for_request(...) 的薄封装。"""

def response_tokens_for_request(request: Request) -> float:
    """billing-like 计算时使用的 output token 数。"""
```

`request_prefix_id` 和 `record_prefix_cache_dispatch` 已删除。

`Provider.marginal_cost_for_request(request, now, *, cached_input_tokens=0)` 接口不变；旧调用默认 `cached_input_tokens=0`。

---

## 8. Cost Computation

### 8.1 Cached Tokens

```python
if not cache_enabled or not state.metadata.get("prefix_cache_enabled"):
    return 0
if provider.tier != ProviderTier.S_A:
    return 0
if provider.cached_input_cost_per_token is None:
    return 0

trace_value = trace_observed_cached_input_tokens(request)
if trace_value is None:
    return 0

return min(max(request.request_tokens, 0), max(trace_value, 0))
```

### 8.2 API Cost

对 `S_A` provider:

```python
uncached_input_tokens = max(request.request_tokens - cached_input_tokens, 0)

cost = (
    uncached_input_tokens * provider.input_cost_per_token
    + cached_input_tokens * provider.cached_input_cost_per_token
    + response_tokens * provider.output_cost_per_token
)
```

如果 provider 没有 split input/output price，直接走旧的 `marginal_cost_for_request()`。

### 8.3 No Post-Dispatch Update

dispatch 完成后 simulator **不** 写任何 cache state。trace 字段就是 ground truth；不需要事后回写。

Rejected / hedged / canceled 路径也都没有 cache update 逻辑。

---

## 9. Integration Points

### 9.1 RouteWise Effective Cost

`S_A` 分支使用 `cache_aware_marginal_cost(provider, request, state)`；`S_Q` / `S_C` shadow price 不变。Policy 接口不变。

### 9.2 Baselines

`greedy_cost` 必须使用同一个 cache-aware API cost，否则会不公平。`greedy_latency` 的 primary key 仍然是 latency，cost tie-break 也用 cache-aware cost。

### 9.3 Hedging

backup provider selection 如果看 marginal cost，使用 cache-aware cost。primary 与 backup 看到的 trace cache 字段相同，因此 backup 也会得到相同的 cache discount；这反映了一个事实：trace 报告的是真实历史 dispatch 在某 provider 上的 cache hit，我们用作 coarse locality 信号，不再 per-provider 假设。

### 9.4 Billing Order

简化为：

1. Policy route 使用 trace 字段决定 cache-aware cost。
2. Primary cost 用同一字段计费。
3. Primary dispatch 后无需 cache update。
4. Backup 如果实际 dispatch，用同一字段计费。
5. Backup dispatch 后无需 cache update。

---

## 10. Offline Oracle

Cache 仍然让 cost 取决于“是否选了能命中 cache 的 provider”，但在 trace-only model 下每条 request 的 cache token 是固定的（不依赖之前的 routing assignment）。因此：

- 旧 path-independent offline exact oracle 仍能在 cache-enabled run 里运行；
- 但它的 cost 比较仍然要使用 cache-aware billing 公式，否则不公平。

实操上推荐：

```text
cache ablation compares online policies on cache-aware billing
exact offline oracle can be reported alongside, but using the same
cache-aware billing function as the online policies.
```

---

## 11. Experiment Plan

### 11.1 Dataset

Phase A 优先用 **freeinference** 风格 trace，因为它自带 `cache_read_tokens`：

```text
freeinference: 直接用 metadata["cache_read_tokens"]
ShareGPT / BurstGPT: 通常没有 cache_read_tokens，整批按 cold miss 处理
                     （需要 cache 实验时可以单独跑 dataset_cache 生成的 mock 字段）
```

### 11.2 Policies

至少比较：

- RouteWise
- greedy_cost
- greedy_latency
- random

### 11.3 Knobs

Phase A 不做 cached-price sweep。默认：

```text
cache_enabled in {false, true}
cached_input_price_fraction = 0.2     # synthetic API only
```

`cache_enabled` 通过 helper 层 / `prefix_cache_enabled` scenario metadata 实现，不修改 provider config。control run 和 treatment run 复用同一份 provider config；`cache_enabled=False` 时 `cached_input_tokens(...)` 直接返回 0。

### 11.4 Metrics

新增 cache-specific metrics：

```text
cache_hit_rate                # trace 报告的 cache hit 占请求比例
cached_input_token_fraction
cached_input_tokens_total
uncached_input_tokens_total
provider_mix
total_cost_usd
api_cost_usd
cache_savings_usd
```

`cache_savings_usd` 需要 paired runs：

```text
same scenario / same seed / cache_enabled=false
vs
same scenario / same seed / cache_enabled=true
```

Latency metrics 仍然保留，但预期变化只来自 provider mix 改变，而不是 cache 直接改变 TTFT distribution。

---

## 12. Tests

需要覆盖：

1. **Cold cache.** 没有 trace cache 字段、`prefix_cache_enabled=False`、或 provider 没有 cached price 时，cost 与旧路径一致。
2. **Trace-driven hit.** trace 报告 `cache_read_tokens=k` → `cached_input_tokens = min(prompt, k)`，并按 cached-input price 计费。
3. **Zero is authoritative.** trace 显式给 0 时按 cold miss 处理，不再回退到 length-based proxy。
4. **State update absent.** dispatch 后 `SimulationState` 没有任何 cache-related mutation。
5. **RouteWise routing.** trace 报告 cache hit 的 API provider 在 effective cost 下被偏好。
6. **Baseline fairness.** greedy_cost 使用 cache-aware cost。
7. **Latency unchanged.** cache hit 不改变 sampled TTFT distribution。
8. **No-cache compatibility.** cache disabled 或所有 provider `cached_input_cost_per_token=None` 时，golden behavior 不变。

---

## 13. Implementation Phases

### Phase A: Trace-Driven Cost Model

- 保留 provider schema 里的 `cached_input_cost_per_token`。
- 在 helper / `Request.metadata` 中支持 `cache_read_tokens` / `cached_input_tokens` 透传。
- `S_A` cost cache-aware（RouteWise、baselines、hedging backup selection、simulator billing）。
- 删除 `provider_prefix_cache` 字段和 `record_prefix_cache_dispatch` 调用路径。
- 单元测试覆盖 §12。

### Phase B: Scenario / Metrics

- 在 freeinference 风格 trace 上加 prefix-cache ablation scenario。
- 加 cache metrics 到 simulation outputs。
- paired no-cache vs cache-enabled comparison script。

### Phase C: Optional Sensitivity

- prompt text LCP / tokenizer-level prefix matching（针对没有 `cache_read_tokens` 的 trace）。
- prompt_hash exact-repeat matching。
- provider-specific hit probability。
- min overlap threshold。
- cached-input price fraction sweep。
- cache-aware greedy oracle。

These should not block Phase A.

---

## 14. Paper Wording

Safe wording:

```text
We model provider-side prefix caching as a trace-driven, per-request
input-cost discount. For each request we read the cached-input-token count
from the trace's reported cache_read_tokens field (capped by prompt
length); when the trace does not report a value we treat the request as a
cold miss. Cached tokens are billed at the provider's cached-input price.
Cache affects only API billing; latency distributions are unchanged.
```

Assumption footnote:

```text
This is an intentionally coarse model. We rely on the trace's reported
cache-read tokens as the cache-hit signal and apply the same value across
candidate providers at routing time, which makes our cost model
provider-agnostic for the cached fraction. Exact text-prefix matching and
provider-specific hit probabilities are left to sensitivity analysis.
```

Oracle caveat:

```text
Under the trace-driven cache model, per-request cached-token counts do not
depend on previous routing assignments, so the existing path-independent
offline oracle remains valid as long as it uses the same cache-aware
billing function as the online policies.
```
