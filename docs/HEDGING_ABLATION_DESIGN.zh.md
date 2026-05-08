# Hedging Ablation Design

更新时间：2026-05-08

本文记录 latency layer §2.2 hedging 的 ablation 方案。目标是后续开发时有一个稳定的工程和实验蓝图，避免把主算法、ablation knob、explorer/probing 语义混在一起。

## 1. 背景

当前 §2.2 主实验已经注册为：

```text
routewise simulator hedging
```

主实验比较：

```text
ablation_lp_only_p75
ablation_lp_hedging_p75
```

主算法语义是：

```text
probability-target hedging
dispatch timing = latest_safe
backup selection = probability
explorer = false
target success probability = 0.99
```

也就是说，RouteWise 先按 LP 选择 primary provider；如果 primary 在 checkpoint 还没有返回，则计算当前发送 backup 后满足 SLO 的概率是否能达到 0.99。backup provider 从非 primary 的 available providers 中选择。主算法不做 random explorer，也不把 backup request 当作 probing/exploration 机制。

## 2. 要回答的问题

Hedging ablation 主要回答两个独立问题。

第一个问题是 when to hedge：

```text
earliest_safe:
  第一个能让 success probability >= 0.99 的 checkpoint 就 dispatch backup。

latest_safe:
  如果未来 checkpoint 仍然有 backup 能维持 success probability >= 0.99，
  当前 checkpoint 先等；只有再等可能失去 0.99 guarantee 时才 dispatch。
```

第二个问题是 which backup to send：

```text
probability:
  从 feasible backup 中选择最便宜的；同成本时选 success probability 更高的；
  再同分时选 true mean latency 更低的。

random_feasible:
  trigger 和 timing 仍然由 probability-target 决定；
  但真正 dispatch 时，从当前 feasible 的非 primary available providers 中随机选择 backup。
```

注意：`random_feasible` 的完整语义是 `random_among_feasible_non_primary`，它不是 explorer。它只改变 feasible backup 之间的 selection，不改变 trigger，也不启用 backup-result learning。这里刻意不从 all non-primary 中随机选，因为那会把“backup 是否 feasible”也混进 ablation，导致 hedge_rate / SLO safety 同时变化。

## 3. 推荐工程结构

采用 hybrid 方案：

```text
rwsim/policies/hedging.py
  放稳定、可复用、可单测的 hedging primitives。

rwsim/policies/routewise.py
  继续实现 production RouteWisePolicy。
  使用 hedging primitives。
  不暴露 public ablation knobs。
  只提供 protected hooks 给 ablation subclass 覆盖。

experiments/ablations/hedging/
  放 hedging ablation policy、presets、harness、CLI glue。
```

这个设计的原则是：

```text
主算法代码保持干净
ablation knobs 放在 experiments/ablations/
hedging 公式不复制
production 和 ablation 不 drift
```

## 4. rwsim 层：抽 reusable hedging primitives

新增：

```text
rwsim/policies/hedging.py
```

建议包含：

```python
@dataclass(frozen=True)
class BackupCandidate:
    provider: Provider
    success_probability: float
    marginal_cost: float
    true_mean_ms: float

    @property
    def feasible(self) -> bool:
        return self.success_probability >= HEDGE_SUCCESS_TARGET - EPS
```

核心 helper：

```python
def combined_success_probability(
    cdf_ms: Callable[[Provider, float], float],
    primary: Provider,
    backup: Provider,
    *,
    elapsed_ms: float,
    slo_ms: float,
    dispatch_overhead_ms: float,
) -> float:
    ...
```

公式语义：

```text
P(success if hedge now)
= P(primary meets SLO | primary has not finished by t)
  + P(primary misses SLO | primary has not finished by t)
    * P(backup finishes within remaining time)
```

其中：

```text
t = elapsed_ms
remaining time = slo_ms - elapsed_ms - dispatch_overhead_ms
```

再提供：

```python
def collect_backup_candidates(...):
    """Return scored non-primary available backup candidates."""

def select_probability_backup(candidates: Sequence[BackupCandidate]) -> BackupCandidate | None:
    """Production backup selection."""

def best_backup_success_probability(candidates: Sequence[BackupCandidate]) -> float:
    """Used by latest_safe future check."""
```

这些函数应该尽量是 pure helper。它们不应该知道 experiment name、policy name、summary csv 或 ablation mode。

## 5. RouteWisePolicy 层：保留 production 默认行为

`RouteWisePolicy` 的 public API 不新增 ablation 参数。生产默认仍然是：

```text
latest_safe + probability backup
```

将当前 `tick()` 中 inline 的 hedging block 重构成清晰的内部步骤：

```python
def tick(...):
    candidates = self._collect_backup_candidates(...)
    selected = self._select_backup_candidate(candidates)
    if selected is None:
        return None
    if not self._should_dispatch_now(selected, future_checkpoints, ...):
        return None
    return HedgeDispatch(...)
```

Protected hooks：

```python
def _select_backup_candidate(
    self,
    candidates: Sequence[BackupCandidate],
) -> BackupCandidate | None:
    return select_probability_backup(candidates)

def _should_dispatch_now(
    self,
    selected: BackupCandidate,
    *,
    future_feasible: bool,
) -> bool:
    return selected.feasible and not future_feasible
```

默认行为解释：

```text
selected.feasible == True:
  当前 checkpoint hedge 可以达到 0.99。

future_feasible == True:
  未来 checkpoint 仍然存在 backup 能达到 0.99，因此现在不 dispatch。

默认 latest_safe:
  当前 feasible 且 future 不再 feasible 时 dispatch。
```

这个 refactor 应该是行为不变的纯重构。

## 6. Ablation 层：experiments/ablations/hedging

新增目录：

```text
experiments/ablations/hedging/
  __init__.py
  policy.py
  presets.py
  harness.py
```

### 6.1 policy.py

`HedgingAblationPolicy` 继承 `RouteWisePolicy`：

```python
class HedgingAblationPolicy(RouteWisePolicy):
    def __init__(
        self,
        *,
        dispatch_timing: Literal["latest_safe", "earliest_safe"],
        backup_selection: Literal["probability", "random_feasible"],
        **kwargs,
    ):
        super().__init__(
            hedging="probability_target",
            explorer=False,
            **kwargs,
        )
        self.dispatch_timing = dispatch_timing
        self.backup_selection = backup_selection
```

Override `_select_backup_candidate`：

```text
probability:
  使用 production select_probability_backup。

random_feasible:
  从 feasible candidates 中随机选一个。
  如果没有 feasible candidate，则不 dispatch。
```

Override `_should_dispatch_now`：

```text
earliest_safe:
  selected.feasible 时立刻 dispatch。

latest_safe:
  复用 production 行为。
```

Primary route、LP、effective cost、rolling latency profile、observe 逻辑都继承 production `RouteWisePolicy`。ablation policy 只改 hedging dispatch timing 和 backup selection 两个维度。

### 6.2 presets.py

稳定 policy names：

```text
hedging__dispatch=latest_safe__backup=probability__p75
hedging__dispatch=earliest_safe__backup=probability__p75
hedging__dispatch=latest_safe__backup=random_feasible__p75
```

可选完整矩阵：

```text
hedging__dispatch=earliest_safe__backup=random_feasible__p75
```

第一版建议默认只跑前三个。原因是前三个分别回答：

```text
production baseline: latest_safe + probability
timing ablation: earliest_safe + probability
backup selection ablation: latest_safe + random_feasible
```

### 6.3 harness.py

CLI：

```text
routewise ablation hedging
```

复用 §2.2 scenarios：

```text
hedging_heavy_tail
hedging_real_world_rw3
```

默认 policies：

```text
hedging__dispatch=latest_safe__backup=probability__p75
hedging__dispatch=earliest_safe__backup=probability__p75
hedging__dispatch=latest_safe__backup=random_feasible__p75
```

输出目录：

```text
outputs/ablations/hedging/
```

输出字段至少包含：

```text
scenario
policy
dispatch_timing
backup_selection
routewise_p
slo_ms
target_success_probability
hedge_rate
p50_ms
mean_ttft_ms
p99_ms
slo_violation_rate
mean_api_cost_usd
mean_total_cost_usd
cost_multiplier_vs_production
p99_delta_vs_production_ms
hedge_rate_delta_vs_production
```

其中 production baseline 明确是：

```text
production_baseline_policy = hedging__dispatch=latest_safe__backup=probability__p75
```

`*_vs_production` 字段都相对这行计算，不相对 LP-only。

如果 run schema 里能拿到 backup winner，也建议输出：

```text
backup_win_rate
primary_win_rate
mean_dispatch_checkpoint
mean_hedge_wait_ms
```

这些字段不是第一版 blocker，但对分析 earliest/latest 的差异很有用。

## 7. 实验矩阵

主 §2.2 不变：

```text
LP-only
production RouteWise hedging = latest_safe + probability
```

Hedging ablation：

```text
latest_safe + probability
earliest_safe + probability
latest_safe + random_feasible
```

Scenarios：

```text
hedging_heavy_tail
hedging_real_world_rw3
```

未来如果 §2.2 扩展到 8-provider：

```text
hedging_heavy_tail_8
hedging_real_world_rw8
```

但第一版 ablation 不应该阻塞在 RW8 上。

## 8. 预期结果与解释

Timing ablation 的预期：

```text
earliest_safe:
  hedge_rate 更高
  cost multiplier 更高
  p99 可能更低

latest_safe:
  hedge_rate 更低
  cost multiplier 更低
  p99 接近 earliest_safe
```

如果 latest_safe 的 p99 接近 earliest_safe，同时 hedge_rate 明显更低，就支持主算法：

```text
latest-safe preserves most tail-latency gains while avoiding unnecessary backup requests.
```

Backup selection ablation 的预期：

```text
probability:
  更稳定满足 SLO target
  p99 / violation rate 更好

random_feasible:
  trigger/timing 和 probability 一致，因此 hedge_rate 应该接近 probability
  但 p99 / violation rate 更差，因为 backup 可能选到慢 provider
```

这能回答：

```text
probability-aware backup selection 是否真的必要？
```

## 9. 必须有的测试

### 9.1 Pure helper tests

测试 `combined_success_probability`：

```text
primary 已经几乎必然 SLO 内完成时，backup 影响很小。
primary 几乎必然 miss 时，combined success 接近 backup success。
remaining time <= 0 时，backup_success = 0。
```

测试 `select_probability_backup`：

```text
只从 feasible candidates 中选。
先按 marginal_cost 最低。
同 cost 时 success_probability 高者优先。
再同分时 true_mean_ms 低者优先。
```

### 9.2 Production refactor tests

锁住 production 默认行为：

```text
RouteWisePolicy 默认 tick 行为和重构前一致。
```

更实际的测试方式：

```text
构造 primary + two backups 的固定 scenario。
确认 default RouteWisePolicy 仍然选择原来的 backup。
确认 latest_safe 仍然会等待 future feasible checkpoint。
```

### 9.3 Ablation equivalence tests

最关键：

```text
HedgingAblationPolicy(
  dispatch_timing="latest_safe",
  backup_selection="probability",
)

等价于 production RouteWisePolicy。
```

这条保证 ablation baseline 就是主算法。

### 9.4 Ablation behavior tests

```text
earliest_safe 在第一个 feasible checkpoint dispatch。
latest_safe 在 future feasible 时不 dispatch。
random_feasible 不选 primary provider。
random_feasible 只从 feasible candidates 中抽样。
random_feasible 不启用 explorer learning。
```

### 9.5 Harness tests

```text
routewise ablation hedging 注册成功。
policy names 可 parse。
smoke run 写 summary.json 和 summary.csv。
summary 中包含 dispatch_timing / backup_selection metadata。
```

## 10. 开发步骤

建议拆成两个 commit 或两个 PR。

### Step 1: pure refactor

```text
新增 rwsim/policies/hedging.py
移动 combined_success_probability 等 helper
RouteWisePolicy.tick 改用 helper
加入 protected hooks
不新增 ablation CLI
不改变 production 行为
```

验证：

```text
uv run pytest tests/unit/policies/test_flat_policies.py -q
uv run pytest tests/unit/simulation/test_hedging.py -q
```

### Step 2: hedging ablation

```text
新增 experiments/ablations/hedging/
实现 HedgingAblationPolicy
实现 presets / harness
注册 routewise ablation hedging
加入 unit tests 和 smoke test
```

验证：

```text
uv run pytest tests/unit/ablations/test_hedging_ablation.py -q
uv run routewise ablation hedging --scenario hedging_heavy_tail --seed 42 --max-requests 20 --output-dir /tmp/routewise_hedging_ablation_smoke
```

## 11. 不做的事情

第一版不做：

```text
不把 dispatch_timing / backup_selection 加到 RouteWisePolicy public __init__。
不把 random backup 称为 explorer。
不从 all non-primary 中随机选 backup；第一版 random ablation 只在 feasible backups 中随机。
不让 backup result learning 进入这个 ablation。
不重写 LP primary routing。
不复制 hedging success probability 公式。
```

这些边界能保持主代码干净，也能让 ablation 结果更容易解释。
