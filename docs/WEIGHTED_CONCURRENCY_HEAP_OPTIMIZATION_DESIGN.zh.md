# WeightedConcurrencyState Heap 优化设计

> 目标：在不改变 simulator 语义、不改实验 harness 的前提下，优化
> `WeightedConcurrencyState` 的 active request release / utilization 热点，为
> Phase B concurrency-only 消融降运行时间。

最后更新：2026-05-07。

---

## 1. 背景

Phase B concurrency-only 实验会跑：

```text
40 cells × full BurstGPT 30d trace
≈ 40 × 1,813,565 requests
≈ 72.5M request-level simulation steps
```

当前 bottleneck 之一在 `WeightedConcurrencyState`：

```python
active: list[tuple[finish_time, model_class, cost]]

used_concurrency_cost(now):
    release_finished(now)     # list filter
    return sum(cost for active)
```

在 simulator 路径中，这些调用非常频繁：

```text
provider.is_available()
provider.concurrency.can_admit_interval(...)
provider.account_request(...)
policy.effective_cost_for_provider(... utilization(now) ...)
summary/record metadata collection
```

当前实现每次 query 都可能扫一遍 active list。对 full trace 和多 cells 来说，
这个开销会累计。

---

## 2. 非目标

这次不是实验 fast path。

不做：

- 不绕过 `Simulator`;
- 不改 `LPOnlyAblationPolicy`;
- 不改 cost-layer / ablation harness;
- 不改 summary schema;
- 不改 routing semantics;
- 不跑 full Phase B experiment。

这次只优化 shared primitive：

```text
rwsim/world/capacity.py::WeightedConcurrencyState
```

---

## 3. 当前语义必须保持

这些行为不能变：

- `release_finished(now)` 移除 `finish_time <= now` 的 active intervals。
- `used_concurrency_cost(now)` 先 release，再返回当前占用。
- release 是不可逆的；如果先查 `now=30`，再查 `now=5`，已 release 的请求不会复活。
- `utilization(now)` 返回 `min(used / capacity_units, 1.0)`。
- `can_admit(model_class, now)` 对未知 model class 返回 `False`。
- `can_admit(model_class, now)` 会按 `now` release 后判断。
- `can_admit_interval(start, end, model_class)` 对 weighted concurrency 仍然忽略 `end`，只按 `start` release/check。
- `admit(model_class, finish_time, now=...)` 容量不足时返回 `False`，成功时返回 `True`。
- `admit(..., now=...)` 仍要求 `finish_time > now`。
- `admit_interval(now, service_time_sec, model_class)` 仍调用 `admit(..., finish_time=now+service_time_sec, now=now)`。
- `peak_used_concurrency_cost` 仍记录历史峰值。
- `total_capacity_unit_seconds_used` 仍只在 `now is not None` 的 successful admit 上累计：

```text
cost * (finish_time - now)
```

- `reset()` 清空 active state，并把:

```text
active == []
peak_used_concurrency_cost == 0
total_capacity_unit_seconds_used == 0.0
```

- `limit` property 不变。

---

## 4. Data Structure

保留 dataclass public field 名：

```python
active
```

但把它定义为 internal heap：

```python
active: list[tuple[float, int, str, int]]
# (finish_time, sequence_number, model_class, cost)
```

新增 internal fields：

```python
_current_used_cost: int = 0
_sequence: int = 0
```

保留 `active` 字段名的原因：

- 它现在是 dataclass public field；
- `rg` 看起来生产代码没有直接依赖 tuple shape；
- 保留字段名比改成 `_active_heap` API 破坏更小；
- `reset()` 后仍保证 `active == []`。

不提供 legacy tuple shape compatibility。除非发现现有测试直接构造：

```python
WeightedConcurrencyState(active=[(finish, model_class, cost)])
```

否则不兼容旧 tuple shape。

---

## 5. Algorithm

Release:

```python
def release_finished(now):
    while active and active[0][0] <= now:
        _, _, _, cost = heapq.heappop(active)
        _current_used_cost -= cost
```

Query:

```python
def used_concurrency_cost(now=None):
    if now is not None:
        release_finished(now)
    return _current_used_cost

def utilization(now=None):
    return min(used_concurrency_cost(now) / capacity_units, 1.0)

def can_admit(model_class, now=None):
    cost = concurrency_cost(model_class)
    if cost is None:
        return False
    if now is not None:
        release_finished(now)
    return _current_used_cost + cost <= capacity_units
```

Admit:

```python
def admit(model_class, finish_time, now=None):
    if now is not None and finish_time <= now:
        raise ValueError(...)
    cost = concurrency_cost(model_class)
    if cost is None:
        return False
    if now is not None:
        release_finished(now)
    if _current_used_cost + cost > capacity_units:
        return False

    heapq.heappush(active, (finish_time, _sequence, model_class, cost))
    _sequence += 1
    _current_used_cost += cost
    peak_used_concurrency_cost = max(peak_used_concurrency_cost, _current_used_cost)
    if now is not None:
        total_capacity_unit_seconds_used += cost * (finish_time - now)
    return True
```

Do not update peak by calling `used_concurrency_cost()` after admit. Use the
running counter directly to avoid extra release/query side effects.

---

## 6. Complexity

Current:

```text
release/check/sum: O(active_count) per query
```

After heap:

```text
release expired requests: O(k log active_count)
used_concurrency_cost: O(1) after release
utilization: O(1) after release
can_admit: O(1) after release
admit: O(log active_count)
```

Here `k` is the number of requests that actually finish at or before the queried
time. Since simulator time advances monotonically through the trace, each active
request is popped once.

---

## 7. Test Plan

### Unit Tests

Add/update capacity tests, preferably in:

```text
tests/test_world_capacity.py
```

Required tests:

1. `test_weighted_concurrency_heap_release_semantics`

```text
admit finish at 10/20/30
used_concurrency_cost(0)  keeps all
used_concurrency_cost(10) releases finish_time <= 10
used_concurrency_cost(20) releases finish_time <= 20
used_concurrency_cost(30) releases finish_time <= 30
```

2. `test_weighted_concurrency_running_used_cost_and_utilization`

```text
multiple admits increase running used cost
release decreases it
utilization matches used / capacity
```

3. `test_weighted_concurrency_rejects_when_capacity_full`

```text
full capacity -> can_admit False, admit False
after release -> can_admit True
```

4. `test_weighted_concurrency_peak_and_capacity_seconds_preserved`

```text
peak_used_concurrency_cost records max running used cost
total_capacity_unit_seconds_used += cost * (finish - now)
```

5. `test_weighted_concurrency_reset_clears_heap_and_counters`

```text
active == []
used == 0
peak == 0
total_capacity_unit_seconds_used == 0.0
```

6. `test_weighted_concurrency_interval_helpers`

Cover simulator path:

```text
can_admit_interval(start, end) ignores end but releases/checks at start
admit_interval(now, service_time_sec) admits until now + service_time_sec
admit_interval still accumulates capacity seconds
```

7. Randomized equivalence test

Use a small list-based reference model inside the test.

Important constraint:

```text
now must be nondecreasing
```

Reason: release is irreversible in the current implementation too. Randomized
tests with decreasing time would be invalid unless the reference model also
simulates irreversible release.

Operations:

```text
nondecreasing now
random can_admit
random admit
random used_concurrency_cost
random utilization
compare heap implementation vs list reference
```

### Existing Tests To Run

Run:

```bash
uv run pytest tests/test_world_capacity.py
uv run pytest tests/unit/simulation/test_cost_layer.py
uv run pytest tests/unit/ablations/test_effective_cost_policy.py
uv run pytest tests/unit/ablations/test_effective_cost_harness.py
```

Also run the existing ablation curve tests if files touched by imports change:

```bash
uv run pytest tests/unit/ablations/test_effective_cost_curves.py
```

### Smoke

Run a small simulator smoke, not full Phase B:

```bash
uv run routewise ablation effective-cost \
  --phase concurrency \
  --concurrency-count 6 --concurrency-count 8 \
  --concurrency-curve util_linear_u \
  --p 0 \
  --seed 42 \
  --workload burstgpt \
  --max-requests 1000 \
  --jobs 2 \
  --output-dir /tmp/routewise_weighted_concurrency_heap_smoke
```

Expected:

```text
2 rows complete
summary.csv exists
no metric/schema changes
```

---

## 8. Benchmark Plan

Optional local microbenchmark, not required for correctness:

```text
100k to 1M synthetic operations
compare old list reference vs new heap state
measure used_concurrency_cost / utilization / can_admit loops
```

This benchmark should not be used as a correctness proof. It only checks that
the intended hotspot was actually improved.

---

## 9. Rollback Boundary

If tests fail in ways that suggest semantic drift, rollback only the
`WeightedConcurrencyState` internals. Do not compensate by changing:

```text
Simulator
Provider
LPOnlyAblationPolicy
effective-cost harness
summary aggregation
```

The point of this patch is to preserve all external experiment semantics while
making S_C state queries cheaper.

---

## 10. Review Checklist

Before running Phase B full experiment, confirm:

- `active` is documented as internal heap;
- `reset()` leaves `active == []`;
- no production code depends on old `active` tuple shape;
- randomized equivalence test uses nondecreasing time;
- interval helper tests cover provider/simulator call path;
- `peak_used_concurrency_cost` is updated from `_current_used_cost`, not by
  calling `used_concurrency_cost()`;
- all listed tests pass;
- small Phase B smoke passes.
