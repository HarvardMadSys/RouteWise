# Prefix Cache Cost Layer Design

> 目标：按 May 4 / May 5 会议里 Juncheng 说的 "easy way"，把 prefix cache 作为一个简单、可解释的 cost-layer feature 加进 simulator。第一版只改 API input-token billing，不改 latency、queueing、capacity accounting、hedging success probability。

最后更新: 2026-05-10。

---

## 1. TL;DR

第一版采用 length-based approximation：

```text
same provider
+ same conversation/session
+ previous context length overlaps current prompt length
=> assume prefix cache hit
=> discount API input-token cost
```

核心 state 是：

```python
provider_prefix_cache: dict[str, dict[str, int]]
```

含义：

```text
provider_name -> session_id/prefix_id -> previous_context_tokens
```

route 时，对每个 provider 估计：

```python
previous_context_tokens = provider_prefix_cache[provider].get(prefix_id, 0)
cached_input_tokens = min(request.request_tokens, previous_context_tokens)
uncached_input_tokens = request.request_tokens - cached_input_tokens
```

API cost：

```text
cost =
  uncached_input_tokens * input_price
  + cached_input_tokens * cached_input_price
  + output_tokens * output_price
```

dispatch 后更新：

```python
provider_prefix_cache[chosen_provider][prefix_id] =
    request.request_tokens + response_tokens
```

第一版不比较 prompt 文本，不做 LCP，不做 semantic similarity，不做 provider hit-rate probability，不做 TTL/LRU。

---

## 2. Meeting Basis

Juncheng 在 May 4 会议里的原话：

```text
The missing part, now is prefix cache. That's a big part we ignored.
```

他的 easy way：

```text
If you see there is a significant overlap in context length with the user's
previous request, you know there's a prefix cache hit. When you know the prefix
cache hit, you can estimate the cost. So basically, you reflect that in cost.
```

他也明确说第一版不要把它放进 latency：

```text
For latency, let's ignore it for now...
we don't consider prefix cache in the latency part.
```

原因：

1. TTFT / tail latency 主要由 queueing latency 主导。
2. 真实 provider 可能 load balance 到不同 node，即使逻辑上有 prefix overlap，也不保证实际 node-level hit。
3. 他不想引入 provider-specific cache hit-rate complexity；第一版可以直接假设 predicted cache 会 hit。

因此 simulator 第一版只在 cost layer 体现 prefix cache。

---

## 3. Non-Goals

第一版明确不做：

- 不做 prompt text longest-common-prefix。
- 不做 tokenizer-level exact prefix matching。
- 不做 embedding / semantic similarity。
- 不做 multi-context per user matching。
- 不用 `user_id` 作为默认 cache key。
- 不做 provider-specific cache hit probability。
- 不做 TTL / LRU eviction。
- 不使用 `cache_read_tokens` 作为 routing truth。
- 不改变 TTFT distribution。
- 不改变 queueing / capacity accounting / service duration。
- 不改变 quota / concurrency shadow-price 公式。
- 不在 cache-enabled ablation 中 claim existing exact offline oracle 仍是 strict optimum。

这些可以作为 future work 或 sensitivity，不进 Phase A。

---

## 4. Request Metadata

### 4.1 Prefix ID

`prefix_id` 表示同一条可复用上下文链。Phase A 只使用 conversation/session 级别 id：

```text
request.metadata["prefix_id"]
or request.metadata["sharegpt_conversation_id"]
or request.metadata["session_id"]
or None
```

如果 `prefix_id is None`，该 request 视为 cold，不参与 prefix-cache discount。

这不是在设计 multi-user router；它只是回答 Juncheng 说的 "user's previous request" 在 trace 里应该对应哪条历史上下文。对 ShareGPT/BurstGPT 这类 conversation trace，`conversation_id/session_id` 就是这个边界。

Phase A 不默认使用 `user_id`。原因是一个 user 可以有多条 unrelated tasks，把 `user_id` 当成单一上下文链会夸大 cache hit。FreeInference 这种没有 session id 的 trace，先不作为 Phase A 主实验；后续可以单独做 `user_id as session_id` sensitivity。

### 4.2 Context Length Overlap

第一版不比较 prompt 文本。它只保存 provider 在某个 session 上一次 dispatch 后看到的上下文 token 数：

```text
previous_context_tokens = previous_prompt_tokens + previous_response_tokens
```

当前请求如果考虑发给同一个 provider：

```python
cached_input_tokens = min(
    current_request_tokens,
    previous_context_tokens,
)
```

这对应 Juncheng 的 "significant overlap in context length"：如果当前 prompt 长度不超过 provider 已经见过的上一轮上下文长度，就把这部分视为 cache hit；如果当前 prompt 更长，则最多命中上一轮上下文长度。

Phase A 不设额外 overlap threshold。原因是多轮 chat 的下一轮 prompt 通常包含之前上下文；这个 coarse model 的目标是让 cost layer 看到 provider-local cache locality，而不是精确判定真实 token prefix。

### 4.3 Loader Rule

Phase A 需要 loader 保留或生成：

```text
prefix_id
request_tokens
response_tokens / estimated_response_tokens
```

推荐规则：

```text
ShareGPT:
  prefix_id = sharegpt_conversation_id

Burst/session trace:
  prefix_id = session_id

No conversation/session id:
  prefix_id = None
  cache disabled for that request
```

Phase A 不需要 loader 保留 `prompt_text` / `response_text` 来计算 cache hit。文本字段可以继续保留给 future exact-prefix sensitivity，但不是核心实现依赖。

---

## 5. Provider Pricing

Provider 增加可选字段：

```python
cached_input_cost_per_token: float | None = None
```

语义：

```text
None:
  provider does not support cached-input discount

float:
  cached input token price for this provider
```

如果没有真实 cached-input price，可以在 scenario factory 里用 fraction 生成：

```text
cached_input_cost_per_token =
  input_cost_per_token * cached_input_price_fraction
```

不同 tier 的处理：

```text
S_A:
  use cache-aware API token billing

S_Q:
  keep quota shadow price unchanged

S_C:
  keep concurrency shadow price unchanged
```

原因：`S_Q` / `S_C` 在 RouteWise effective cost 里是 opportunity cost，不是 provider API token billing。

如果 provider 只有 legacy blended `cost_per_token`，没有拆开的 `input_cost_per_token` / `output_cost_per_token`，cache discount 默认 disabled，避免错误地给 blended price 打折。

---

## 6. Runtime State

在 `SimulationState` 里增加：

```python
provider_prefix_cache: dict[str, dict[str, int]]
```

含义：

```text
provider_name -> prefix_id -> previous_context_tokens
```

例子：

```text
session A turn 1 routed to provider P:
  prompt = 1000 tokens
  response = 500 tokens

after dispatch:
  provider_prefix_cache[P][A] = 1500

session A turn 2 considers provider P:
  prompt = 1800 tokens
  cached_input_tokens = min(1800, 1500) = 1500

session A turn 2 considers provider Q:
  provider_prefix_cache[Q][A] missing
  cached_input_tokens = 0
```

第一版采用 infinite cache：

- no TTL
- no LRU
- no node-level miss
- one integer per `(provider, prefix_id)`

空间复杂度：

```text
O(num_providers_that_saw_prefixes * num_prefix_ids_seen_by_provider)
```

状态非常小，不需要存 prompt text。

---

## 7. Helper API

建议新增 helper，例如 `rwsim/policies/prefix_cache.py`：

```python
def request_prefix_id(request: Request) -> str | None:
    ...

def cached_input_tokens(
    provider: Provider,
    request: Request,
    state: SimulationState,
    *,
    cache_enabled: bool = True,
) -> int:
    ...

def cache_aware_marginal_cost(
    provider: Provider,
    request: Request,
    state: SimulationState,
    *,
    now: float,
    cache_enabled: bool = True,
) -> float:
    ...

def record_prefix_cache_dispatch(
    provider: Provider,
    request: Request,
    state: SimulationState,
    response_tokens: int,
    *,
    cache_enabled: bool = True,
) -> None:
    ...
```

`response_tokens` 应使用 simulator 对这次 dispatch 记账时采用的输出 token 数。如果调用点只能使用 `estimated_response_tokens`，需要在调用点保持 billing 和 cache update 使用同一个值。

`Provider.marginal_cost_for_request()` 可以增加可选参数：

```python
def marginal_cost_for_request(
    self,
    request,
    now: float,
    *,
    cached_input_tokens: int = 0,
) -> float:
    ...
```

旧调用默认 `cached_input_tokens=0`，行为不变。

---

## 8. Cost Computation

### 8.1 Cached Tokens

逻辑：

```python
if not cache_enabled:
    return 0

if provider.cached_input_cost_per_token is None:
    return 0

prefix_id = request_prefix_id(request)
if prefix_id is None:
    return 0

previous_context_tokens = state.provider_prefix_cache[provider.name].get(
    prefix_id, 0
)

return min(request.request_tokens, previous_context_tokens)
```

### 8.2 API Cost

对 `S_A` provider:

```python
uncached_input_tokens = max(
    request.request_tokens - cached_input_tokens,
    0,
)

cost = (
    uncached_input_tokens * provider.input_cost_per_token
    + cached_input_tokens * provider.cached_input_cost_per_token
    + response_tokens * provider.output_cost_per_token
)
```

如果 provider 没有 split input/output price，直接走旧的 `marginal_cost_for_request()`。

### 8.3 Cache Update

dispatch 成功后：

```python
state.provider_prefix_cache[provider.name][prefix_id] = (
    request.request_tokens + response_tokens
)
```

Rejected request 不更新 cache。

如果 hedge backup 真的 dispatch 了，backup provider 也要更新自己的 cache，因为它确实收到了 prompt。

---

## 9. Integration Points

### 9.1 RouteWise Effective Cost

RouteWise 的 `S_A` 分支应使用 cache-aware marginal cost：

```text
S_A:
  cache_aware_marginal_cost(provider, request, state)

S_Q:
  quota shadow price unchanged

S_C:
  concurrency shadow price unchanged
```

当前 simulator 的 `Policy.route(request, state)` 已经把 `SimulationState` 传给 policy，因此 Phase A 不需要改 base policy 接口。

### 9.2 Baselines

`greedy_cost` 必须使用同一个 cache-aware API cost，否则会不公平。

`greedy_latency` 的 primary key 仍然是 latency；如果需要 cost tie-break，也应使用 cache-aware cost。

### 9.3 Hedging

backup provider selection 如果看 marginal cost，也应使用 cache-aware cost。

primary 和 backup 的 cache update 都在实际 dispatch 后发生。primary provider 的 cache update 不会让同一 request 的 backup 获得 discount，因为 provider-local cache 不共享。

### 9.4 Billing Order

保持 pre-dispatch / post-dispatch ordering：

1. Policy route 使用 dispatch 前 cache state。
2. Primary cost 使用 dispatch 前 cache state。
3. Primary dispatch 后更新 primary provider cache。
4. Backup 如果实际 dispatch，backup cost 使用 backup dispatch 前 cache state。
5. Backup dispatch 后更新 backup provider cache。

---

## 10. Offline Oracle

Prefix cache 让 cost 依赖之前 routing assignment：

```text
cost(i, provider) depends on whether earlier requests in the same session
were assigned to that provider.
```

因此现有 path-independent offline exact oracle 不能继续在 cache-enabled run 里 claim strict optimum。

Phase A 建议：

```text
cache ablation compares online policies only
and does not report exact offline optimum.
```

如果需要参考线，可以实现并明确命名：

```text
cache-aware greedy oracle / heuristic oracle
```

不要叫 exact oracle。

---

## 11. Experiment Plan

### 11.1 Dataset

Phase A 优先用干净 conversation/session trace：

```text
ShareGPT:
  prefix_id = sharegpt_conversation_id

BurstGPT/session trace:
  prefix_id = session_id
```

FreeInference / RedNote 不作为 Phase A 主实验，因为它们是真实 API logs，不一定有 clean session chain：

- FreeInference sampled rows 里 `session_id` 为空。
- RedNote 有部分 `session_id`，但不是所有行都有。
- 两者都有 `user_id` 和 prompt text，但用 `user_id` 当单一 chain 会把 unrelated tasks 混在一起。

后续可以单独做 sensitivity：

```text
user_id as session_id
text-prefix matching
prompt_hash exact-repeat matching
cache_read_tokens calibration
```

这些不进最小实现。

### 11.2 Policies

至少比较：

- RouteWise
- greedy_cost
- greedy_latency
- random

### 11.3 Knobs

第一轮 sweep：

```text
cached_input_price_fraction in {0.0, 0.1, 0.25, 0.5}
cache_enabled in {false, true}
```

`0.0` 表示 cached input 免费，是 sensitivity 极端；`0.5` 更接近 OpenAI / Anthropic 这类 cached-input price roughly half-price 的配置。真实 provider price 可用时，优先使用 per-provider `cached_input_cost_per_token`。

`cache_enabled` 应在 helper 层实现，不通过 mutate provider config 实现。这样 control run 和 treatment run 可以复用同一份 provider config；`cache_enabled=False` 时 `cached_input_tokens(...)` 直接返回 0。

### 11.4 Metrics

新增 cache-specific metrics：

```text
cache_hit_rate
cached_input_token_fraction
cached_input_tokens_total
uncached_input_tokens_total
provider_cache_hit_rate
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

单次 run 只记录 cached/uncached token counts 和实际 billed cost。

Latency metrics 仍然保留，但预期变化只来自 provider mix 改变，而不是 cache 直接改变 TTFT distribution。

---

## 12. Tests

需要覆盖：

1. **Provider pricing.** cached input tokens 用 cached input price，output tokens 不打折。
2. **Cold cache.** no prefix id / no provider cache / provider no cached price 时 cost 与旧路径一致。
3. **Provider-locality.** provider A 见过 session 不会让 provider B 获得 discount。
4. **State update.** dispatch 后 provider cache 记录 `request_tokens + response_tokens`。
5. **RouteWise routing.** cached API provider 的 effective cost 降低，能改变选择。
6. **Baseline fairness.** greedy_cost 使用 cache-aware cost。
7. **Hedging.** backup dispatch 会更新 backup provider cache。
8. **Latency unchanged.** cache hit 不改变 sampled TTFT distribution。
9. **No-cache compatibility.** cache disabled 或所有 provider `cached_input_cost_per_token=None` 时，golden behavior 不变。
10. **Offline oracle compatibility.** cache-disabled no-cache scenarios 继续复现 existing offline oracle summary，确保 Phase A 不误伤主实验路径。

---

## 13. Implementation Phases

### Phase A: Minimal Length-Based Cost Model

- Add `cached_input_cost_per_token` to provider schema/config.
- Add `provider_prefix_cache: dict[str, dict[str, int]]` to `SimulationState`.
- Add prefix-cache helper module.
- Make `S_A` cost cache-aware in RouteWise, baselines, hedging backup selection, and simulator billing.
- Add unit tests.

### Phase B: Scenario / Metrics

- Add ShareGPT/BurstGPT prefix-cache ablation scenarios.
- Add provider cached-price configs or cached-price fraction.
- Add cache metrics to simulation outputs.
- Add paired no-cache vs cache-enabled comparison script.

### Phase C: Optional Sensitivity

- `user_id as session_id` for FreeInference / RedNote.
- prompt text LCP / tokenizer-level prefix matching.
- prompt_hash exact-repeat matching.
- provider-specific hit probability.
- TTL/LRU eviction.
- min overlap threshold.
- `cache_read_tokens` calibration for logs that report it.
- cache-aware greedy oracle.

These should not block Phase A.

---

## 14. Paper Wording

Safe wording:

```text
We model provider-side prefix caching as a per-provider, per-session
length-based input-cost discount. When a request arrives in a session that has
previously been routed to the same provider, we treat up to the previous
request-plus-response token count as cached input tokens, billed at the
provider's cached-input price. Cache affects only API billing; latency
distributions are unchanged.
```

Assumption footnote:

```text
This is an intentionally coarse model following our design goal of isolating
the cost-layer effect. We use an infinite provider-local cache and assume
predicted cache hits are billed at the provider's cached-input price. Exact
text-prefix matching, eviction, and provider-specific hit probabilities are
left to sensitivity analysis.
```

Oracle caveat:

```text
Because cache-aware costs depend on previous routing assignments, the existing
path-independent offline oracle is not an exact optimum for cache-enabled runs.
We therefore report cache ablations without the exact oracle, or use a clearly
labeled cache-aware greedy heuristic.
```
