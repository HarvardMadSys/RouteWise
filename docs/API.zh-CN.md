# llm-routewise 0.1.0 中文 API 参考

[English version](API.md)

`llm-routewise` 是一个无运行时依赖的 LLM API provider 路由库。它负责根据价格、
延迟观测和预算偏好选择 provider，但不发送 HTTP 请求，也不读取或保存 provider API
key。调用方使用自己的 HTTP 或 SDK client 执行请求，再把结果报告给 RouteWise。

本文档只描述 `llm-routewise==0.1.0` wheel 中已经实现并测试的公开 API。

## 安装与导入

安装指定的 preview 版本：

```bash
python -m pip install llm-routewise==0.1.0
```

distribution 名使用连字符，Python import 包名使用下划线：

```python
import llm_routewise as rw
```

PyPI distribution `routewise` 是另一个项目，不是本库的旧版本或兼容版本。此项目应安装
`llm-routewise`，并从 `llm_routewise` 导入。

要求 Python 3.10 或更高版本。

## 最小示例

`Provider` 的价格单位是每一百万 token 的美元价格。`Router.route()` 只返回路由决定；
`call_your_provider()` 代表调用方自己的执行代码。

```python
import llm_routewise as rw

router = rw.Router(
    [
        rw.Provider("fast", price_in=3.00, price_out=15.00),
        rw.Provider("cheap", price_in=0.15, price_out=0.60),
    ],
    alpha=0.25,
    seed=7,
)

decision = router.route(input_tokens=800)
response = call_your_provider(decision.provider)

decision.completed(
    ttft_ms=response.ttft_ms,
    output_tokens=response.output_tokens,
    cached_tokens=response.cached_tokens,
    cost_usd=response.cost_usd,
)
```

如果 provider 没有直接返回实际费用，可以省略 `cost_usd`；RouteWise 会在 token 用量足够时
按照 `Provider` 的价格计算费用。

## 使用 OpenAI SDK 选择 OpenRouter provider

OpenRouter 可以把同一个固定 model 路由到不同 provider endpoint。为了让 RouteWise 记录的
provider 与实际服务请求的 provider family 一致，应使用 OpenRouter provider slug 作为
`Provider.name`，保持 `MODEL` 不变，只允许 `decision.provider`，并关闭 OpenRouter
fallback。`openai` SDK 和 API key 由调用方管理，不是 `llm-routewise` 的依赖。

上游请求格式见 OpenRouter 的 [provider routing](https://openrouter.ai/docs/guides/routing/provider-selection)
和 [OpenAI SDK integration](https://openrouter.ai/docs/guides/community/openai-sdk) 文档。

```python
import os
import time

from openai import OpenAI

import llm_routewise as rw

MODEL = "openai/gpt-5-mini"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

router = rw.Router(
    [
        # 以下是美元/百万 token 的示意值；请替换成当前 endpoint 价格。
        rw.Provider("openai", price_in=1.0, price_out=4.0),
        rw.Provider("azure", price_in=1.2, price_out=3.8),
    ],
    alpha=0.25,
)

decision = router.route(input_tokens=120)
started = time.monotonic()
try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Explain cache-aware routing."}],
        extra_body={
            "provider": {
                "only": [decision.provider],
                "allow_fallbacks": False,
            }
        },
    )
except Exception as exc:
    kind, code = classify_provider_error(exc)
    decision.failed(kind=kind, code=code)
    raise

usage = response.usage
decision.completed(
    ttft_ms=(time.monotonic() - started) * 1000.0,
    output_tokens=None if usage is None else usage.completion_tokens,
)
```

配置前应确认每个 provider slug 都能服务固定的 `MODEL`。base slug 可能匹配同一 provider
的多个区域或特殊 endpoint；如果归因或价格需要精确到 endpoint，应从 model 页面复制完整
slug，并填写当前的美元/百万 token 价格。单 provider allowlist 加上关闭 fallback，可避免
OpenRouter 在 provider 不可用时静默切换，导致 RouteWise 归因错误；此时应把失败报告给
`decision.failed(...)`。非 streaming 请求可把完整响应延迟作为 `ttft_ms`；usage 或账单
稍后到达时可用 `decision.settle(...)` 补报。

## 顶层公开 API

`0.1.0` 顶层公开 13 个符号：有状态 API `Provider`、`Router`、`Decision`、
`Attempt`、`Tuning`、`StatsSnapshot`；无状态 API `Candidate`、`RouteOnceResult`、
`route_once`；异常 `RouteWiseError`、`ValidationError`、`NoProviderError`、
`OutcomeError`。

`Decision`、`Attempt`、`StatsSnapshot` 和 `RouteOnceResult` 应通过相应方法取得，不应直接
构造。

## `Provider`

```python
rw.Provider(
    name: str,
    price_in: float,
    price_out: float,
    price_cached: float | None = None,
)
```

- `name`：非空 provider 名称；同一个 `Router` 中必须唯一。
- `price_in`：每一百万输入 token 的价格。
- `price_out`：每一百万输出 token 的价格。
- `price_cached`：每一百万缓存命中输入 token 的价格；为 `None` 时按 `price_in` 计价。

价格必须是有限、非负实数。`Provider` 创建后不可变。

## `Tuning`

| 参数与默认值 | 含义 |
| --- | --- |
| `hedge_target=0.99` | 触发 backup 时要求达到的成功概率，范围 `(0, 1]` |
| `penalty_ms=60000.0` | health failure 在尚无首 token 时写入 profile 的惩罚值 |
| `window_min=15.0` | 延迟 profile 的滑动窗口分钟数 |
| `cooldown_sec=30.0` | 达到连续 health failure 阈值后的 cooldown 秒数 |
| `cooldown_after=3` | 进入 cooldown 前允许的连续 health failure 次数 |
| `hedge_min_samples=5` | primary 和 backup 参与 hedging 前各自需要的成功 TTFT 样本数 |
| `exploration_lease_sec=60.0` | 避免重复并发探索未建档 provider 的最长秒数 |

`hedge_target`、`penalty_ms`、`window_min` 和 `exploration_lease_sec` 必须为正数；
`cooldown_sec` 必须非负；两个计数参数必须为正整数。`Tuning` 创建后不可变。

## `Router`

### 构造

```python
rw.Router(
    providers,
    *,
    alpha: float = 0.25,
    slo_ms: float | None = None,
    seed: int | None = None,
    cold_start: str = "explore",
    clock=None,
    tuning: rw.Tuning | None = None,
)
```

- `providers`：非空 `Provider` iterable，名称不得重复。
- `alpha`：默认成本偏好，范围 `[0, 1]`。`0` 只允许最低预测成本；`1` 允许到当前
  eligible 集合的最高预测成本，并在该预算内优化延迟。约束针对随机 mixture 的预测
  期望成本，不是每次采样的费用上限，也不是实际账单保证。
- `slo_ms`：正数时启用 hedging；`None` 表示禁用。
- `seed`：用于可复现的 provider 抽样；必须是 `int` 或 `None`。
- `cold_start`：`"explore"` 或 `"require_observations"`。
- `clock`：返回秒数的无参数 callable；默认使用 `time.monotonic`。
- `tuning`：`Tuning` 实例；省略时使用上述默认值。

### `route()`

```python
decision = router.route(
    *,
    input_tokens: int,
    estimated_cached_tokens: int | Mapping[str, int] = 0,
    alpha: float | None = None,
    exclude: Iterable[str] = (),
) -> rw.Decision
```

- `input_tokens`：本次请求的非负整数输入 token 数。
- `estimated_cached_tokens`：一个非负整数时应用于所有 provider；mapping 时按 provider
  指定，缺失名称按 `0`，未知名称报错。超过 `input_tokens` 的值按 `input_tokens`
  计算。
- `alpha`：仅覆盖本次请求；`None` 使用构造时的默认值。
- `exclude`：本次请求排除的 provider 名称。未知名称报错；成本上下界会在剩余 eligible
  provider 上重新计算。

RouteWise 使用当前输出长度估计和 provider 价格计算候选费用。一个被采用且成功完成的
attempt 在报告正数 `output_tokens` 后会更新后续请求的输出长度估计。返回的
`expected_cost_usd` 是 mixture 的预测期望费用；实际费用以调用方报告为准。

### `observe()`

```python
router.observe(
    provider: str,
    *,
    ttft_ms: float | None = None,
    kind: str | None = None,
    code: str | None = None,
) -> None
```

每次调用必须且只能提供 `ttft_ms` 或 `kind` 其中一个：

- `ttft_ms`：外部成功观测的有限、非负首 token 延迟；写入该 provider 的 profile，并
  重置 health failure streak 和 cooldown。
- `kind="health"`：provider 健康问题；记录错误和延迟 penalty，并推进 failure streak。
- `kind="request"`：请求或配置问题；只记录错误，不写入延迟 penalty 或 cooldown。
- `code`：仅可与 `kind` 一起提供。内建标签包括 `rate_limited`、`timeout`、
  `server_error`、`connection`、`bad_request`、`auth` 和 `unsupported`；其他字符串在
  stats 中归入 `"other"`。

`observe()` 只记录调用当下的观测，不接受历史时间参数。

### 冷启动

- `cold_start="explore"`（默认）：未建立 profile 的 provider 可以在预测预算允许的
  mixture 中被探索。只要 mixture 包含未建档 provider，
  `Decision.expected_latency_ms` 就是 `None`。正在进行的探索会暂时阻止同一 provider
  被重复并发探索；没有可用于 profile 的结果时，限制会在 `exploration_lease_sec`
  后到期。
- `cold_start="require_observations"`：未建立 profile 的 provider 不 eligible。首次
  `route()` 前可用 `router.observe(provider, ttft_ms=...)` 提供样本；如果没有任何
  eligible provider，`route()` 抛出 `NoProviderError`。

profile 中的样本会在 `window_min` 窗口后过期，因此严格冷启动模式下的 provider 可能
重新变为未建档状态。

### Clock

默认 clock 是 `time.monotonic`；也可传入返回秒数的无参数 callable 进行确定性测试。
clock 必须返回有限数值。值减小时 Router 继续使用已见过的最大值；观测 API 不接受调用方
提供的时间戳。clock 返回非法值或抛出异常时，公开操作会抛出 `ValidationError`。
`hedge_now()` 使用调用方提供的 `elapsed_ms`，而不是由 Router clock 推导 elapsed time。

Router、Decision 和 Attempt 操作支持并发调用；hedge slot 分配、lifecycle transition、
adoption 和统计更新会保持一致结果。

### `stats()`

`router.stats() -> rw.StatsSnapshot`；快照及其嵌套 mapping 都不可变，包含：

- `providers[name]`：`primary_selections`、当前窗口的 `ttft_p50_ms` 和
  `ttft_p95_ms`、按 `health`/`request` 分类的 `errors`、
  `cooldown_remaining_sec`、`actual_spend_usd`、`calculated_spend_usd` 和
  `unsettled_attempts`。
- `hedges`：`offered`、`declined`、`won`、`actual_spend_usd` 和
  `calculated_spend_usd`。hedge spend 是 provider spend 的交叉视图，不应与
  provider 总额相加。
- `exploration`：`decisions` 和 `target_selected`。
- `decisions_without_adoption`：多个 attempt 全部结束但没有声明采用关系，且至少一个
  attempt 完成的 decision 数。

## `Decision`

`Router.route()` 返回一个逻辑请求对应的 `Decision`。以下属性只读：

| 属性 | 类型与含义 |
| --- | --- |
| `provider` | 抽样得到的 primary provider 名称 |
| `weights` | provider 到抽样权重的不可变 mapping |
| `expected_cost_usd` | 当前 mixture 的预测期望成本 |
| `expected_latency_ms` | 当前 mixture 的预测 TTFT；含未建档 provider 时为 `None` |
| `checkpoints_ms` | 启用 hedging 时建议检查的 elapsed-time checkpoints |
| `trace` | 包含预算、成本上下界、权重、选择原因和排除原因的不可变诊断信息 |
| `state` | 逻辑请求状态 |
| `primary` | primary `Attempt` |
| `backups` | 已提供 backup `Attempt` 的 tuple |

`decision.explain()` 返回一行可读的路由说明。

以下方法代理到 primary attempt：`first_token(*, ttft_ms, adopted=None)`、
`completed(*, output_tokens=None, ttft_ms=None, cached_tokens=None, cost_usd=None,
adopted=None)`、`failed(*, kind, code=None)`、`cancelled()`、`declined()` 和
`settle(*, output_tokens=None, cached_tokens=None, cost_usd=None)`。

## `Attempt` 生命周期与结算

`Attempt` 表示一次具体 provider 派发。公开属性是：

- `provider: str`
- `state: str`
- `billing_state: str`

状态从 `pending` 经 `first_token()` 进入 `streaming`；`pending` 或 `streaming` 可终结为
`completed`、`failed` 或 `cancelled`，而 `declined` 只能从 `pending` 进入。

- `first_token(ttft_ms=..., adopted=...)`：记录 TTFT 并进入 `streaming`。
- `completed(...)`：正常完成。非 streaming 调用可以在这里提供 `ttft_ms`。
- `failed(kind=..., code=...)`：`kind` 必须是 `"health"` 或 `"request"`。首 token
  之前的 health failure 会写入一次 latency penalty；首 token 之后失败会保留已有
  TTFT，不再写第二条延迟记录。
- `cancelled()`：attempt 已派发，随后被中止。
- `declined()`：attempt 从未派发；只能从 `pending` 调用。

完全相同的 `first_token` 或 terminal 上报是幂等的；与既有状态或字段冲突时抛出
`OutcomeError`。

### Adoption 与逻辑请求状态

`adopted=True` 表示调用方采用了该 attempt 的响应；只接受 `True` 或 `None`，同一个
decision 最多采用一个 attempt。

- 只有 primary 且从未提供 backup 时，primary 完成会自动成为 adopted attempt。
- 存在多个 attempt 时，调用方应显式标记采用的 attempt。
- streaming attempt 必须在 `first_token(..., adopted=True)` 时声明 adoption，不能等到
  `completed()` 再补。
- adopted attempt 终结后，`Decision.state` 跟随它的 terminal state。
- 没有 adoption 且所有 attempt 已终结时：有 completed attempt 则 decision 为
  `unresolved`；否则依次按 `failed`、`cancelled`、`declined` 归结。

### Billing

`completed()` 可以提交 `output_tokens`、`cached_tokens` 和 `cost_usd`。这些字段都是
非负且只写一次；重复相同值是 no-op，提供不同值会抛出 `OutcomeError`。

如果用量或账单稍后才到达，在除 `declined` 外的 terminal attempt 上调用：

```python
attempt.settle(
    output_tokens=actual_output_tokens,
    cached_tokens=actual_cached_tokens,
    cost_usd=actual_cost_usd,
)
```

`settle()` 不能在 terminal 之前调用，也不能用于 `declined` attempt。`failed()` 和
`cancelled()` 不接收 billing 参数；需要时先报告 terminal outcome，再调用 `settle()`。

`billing_state` 的取值：

- `"unknown"`：没有显式费用，也没有足够用量计算费用。
- `"calculated"`：已知输出用量、未提供显式费用，按创建 attempt 时的 provider 价格
  计算。
- `"actual"`：已提供显式 `cost_usd`。如果实际费用晚于 calculated 费用到达，统计会
  从 calculated 原子迁移到 actual，不会重复计费。

## Hedging

构造 `Router` 时提供正数 `slo_ms` 才会启用 hedging：

```python
router = rw.Router(providers, slo_ms=3000.0)
decision = router.route(input_tokens=500)

for elapsed_ms in decision.checkpoints_ms:
    backup = decision.hedge_now(elapsed_ms=elapsed_ms)
    if backup is not None:
        dispatch_backup(backup.provider)
        break
```

`hedge_now()` 返回一个 backup `Attempt` 或 `None`。返回 `None` 是正常结果，可能表示当前
时点应继续等待、样本不足、没有满足成功概率目标的 backup、primary 已开始 streaming，
或 hedge slot 已关闭。每个 decision 最多提供一个 backup。

RouteWise 不会派发 backup。调用方实际发送后，使用 backup `Attempt` 的 lifecycle 方法
报告结果；如果决定不发送已提供的 backup，调用 `backup.declined()`；已发送后中止则调用
`backup.cancelled()`。当 primary 与 backup 竞速时，用 `adopted=True` 明确标记真正采用的
响应。

## 无状态路由：`route_once()`

已经自行计算每个候选项成本和延迟的调用方可以使用无状态 API。

```python
result = rw.route_once(
    [
        rw.Candidate("fast", cost_usd=0.008, latency_ms=350.0),
        rw.Candidate("cheap", cost_usd=0.002, latency_ms=900.0),
    ],
    alpha=0.25,
    seed=7,
)
```

```python
rw.route_once(
    candidates,
    *,
    alpha: float,
    seed: int | None = None,
    rng=None,
) -> rw.RouteOnceResult
```

`Candidate(name: str, cost_usd: float, latency_ms: float)` 不可变；名称必须是非空字符串，
成本和延迟必须是有限、非负实数，同一次调用中的名称必须唯一。

- `alpha` 必填，范围 `[0, 1]`。
- `seed` 和 `rng` 互斥。
- `rng` 必须提供 `random()`，并返回 `[0, 1)` 内的有限实数。
- `route_once()` 不保留观测、profile 或统计状态。

固定 seed 加相同输入会重放同一次抽样。连续调用需要不同随机 draw 时，应传入一个可复用
的 RNG。

`RouteOnceResult` 不可变，包含：

- `provider: str`：抽样结果。
- `weights: Mapping[str, float]`：不可变的 provider 权重，和为 `1`。
- `budget_usd: float`：`c_min + alpha * (c_max - c_min)`。

## 异常

所有公开库异常都继承 `RouteWiseError`：

```text
RouteWiseError
├── ValidationError  # 同时也是 ValueError
├── NoProviderError
└── OutcomeError
```

- `ValidationError`：类型、数值范围、provider 名称或参数组合无效。数值参数通常要求有限
  且非负；token 数必须是非负整数，布尔值不视为数字。
- `NoProviderError`：所有 provider 当前都被排除、处于 cooldown、尚未具备所需观测，
  或冷启动探索仍在进行。
- `OutcomeError`：重复上报的值冲突、非法状态转换、重复 adoption，或非法 settlement。

建议只捕获能够在当前调用层明确处理的具体异常；不要把 `RouteWiseError` 统一忽略。

## `0.1.0` preview 限制

- 只支持按输入、输出和 cached token 计价的 API provider。
- 不包含 provider SDK、HTTP client、托管服务或 credential 管理；所有网络 I/O 由调用方
  完成。
- RouteWise 只选择已配置的 provider 名称；endpoint 和 model 映射由调用方管理，本 API
  不执行 model selection。
- 不包含 quota、并发限制、reserved capacity 或 subscription pricing。
- wheel 不包含研究 simulator、offline 工具、metrics 子包、实验脚本、数据集或研究 CLI。
- Router 的学习状态和统计保存在当前进程内；没有持久化或跨进程状态同步 API。
- 成本约束基于预测期望费用，不保证单次抽样费用或最终 provider 账单。实际费用统计只取决
  于调用方报告的 usage 和 `cost_usd`。
