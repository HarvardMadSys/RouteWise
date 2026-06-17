# RouteWise 库接口

> 状态:面向 v1 公开库的设计草案,范围限定为 API provider。本文是
> [API_PROVIDER_INTERFACE.md](API_PROVIDER_INTERFACE.md) 的中文对照版,以英文版为准。
> 文中的 `Router`、`Decision`、`Client` 尚未实现;它们依赖的底层数学原语已经存在,
> 记录在 [CORE_API.md](CORE_API.md)。

## RouteWise 是什么

RouteWise 是一个 Python 库,服务于"同一个模型要经过多个 API provider 调用"的应用。
DeepSeek-V4、GLM-5.1 这类开源权重模型由许多 provider 售卖,价格各不相同,延迟在
provider 之间有差异,而且随时间漂移。对每一个请求,RouteWise 决定该用哪个 provider:
在成本预算之内,选期望延迟最低的那个,预算由一个旋钮控制。当一个响应有错过截止时间
的风险时,RouteWise 可以向第二个 provider 补发一个延迟的备份请求。

RouteWise 不是通用 LLM 客户端(它不维护各家 provider 的 SDK),不是模型选择器
(模型固定,只换 provider,所以绝不用质量换成本),也不是托管服务(你的 API key
始终留在你自己的进程里)。

## 安装

```bash
pip install routewise              # 决策内核,无第三方依赖
pip install "routewise[client]"    # 额外装上内置执行客户端(httpx)
pip install "routewise[litellm]"   # 额外装上 LiteLLM 路由策略插件
```

方括号后缀是 pip 的 extra,即一组可选依赖。基础包只含决策逻辑,不引入任何标准库以外的
东西。需要 RouteWise 替你发请求时装 `client`,想把算法接进现成的 LiteLLM router 时装
`litellm`。

## 完整接口

一个 router 服务一个模型。你问它请求该发给谁,你用自己的代码把请求发出去,再把结果回报
给它,让下一次决策更准。

```python
from routewise import Provider, Router

router = Router(
    [Provider("fireworks", price_in=0.27, price_out=1.10),
     Provider("together",  price_in=0.18, price_out=0.88)],
    alpha=0.25,                          # 唯一旋钮:0=最便宜,1=最快
)

decision = router.route(input_tokens=1800)          # 问:发给哪个 provider?
response = send(decision.provider, request)          # 你自己的执行层
decision.report(ttft_ms=response.ttft_ms,            # 把结果告诉它
                output_tokens=response.output_tokens)
```

这就是成本-延迟路由主路径的接口表面:三个名词(`Provider`、`Router`、`Decision`)、两个
动词(`route`、`report`)、一个旋钮(`alpha`)。如果用 `slo_ms` 开启尾延迟对冲,唯一新增
的动词是 `hedge_now`。价格单位是每百万 token 美元。`Provider.name` 是一个不透明标签,
由你的执行层去解析;`Router` 本身不发起任何网络调用。

一次决策里调用方可能还想要的其它东西,都是只读属性,因此不增加任何学习负担:

```python
decision.provider            # 采样选出的 provider,把请求发给它
decision.weights             # 底层混合,例如 {"together": 0.7, "fireworks": 0.3}
decision.expected_cost_usd   # 本次决策的期望成本
decision.expected_latency_ms # 本次决策的期望 TTFT
decision.explain()           # 一行人类可读的选择说明
```

### 回报结果

`route()` 返回一个 `Decision`,它是一个句柄:已经握着这次请求的身份信息(provider、输入
长度、缓存长度),所以 `report()` 只需带新信息。成功时回报延迟和输出长度,失败时改为回报
错误:

```python
decision.report(ttft_ms=312.0, output_tokens=540)   # 成功
decision.report(error="rate_limited")                # 429、超时、5xx……
```

回报的错误会作为一个罚时样本进入该 provider 的延迟画像;若反复失败,该 provider 会被移出
候选集并进入一段短暂冷却。这些效果你都不用自己算,你只回报事实,router 自己反应。

## 尾延迟对冲

router 优化的是延迟分布的"主体",对冲负责保护"尾巴"。给 router 一个 SLO(service-level
objective,即首 token 时间的截止线),每个决策就会带上若干检查点时刻,在这些时刻补发一个
备份请求可能划算:

```python
router = Router([...], alpha=0.25, slo_ms=3000)

decision = router.route(input_tokens=1800)
# 主请求在飞。在 decision.checkpoints_ms 的每个时刻,如果首 token 还没回来:
backup = decision.hedge_now(elapsed_ms=1500)         # None,或又一个句柄
if backup is not None:
    bresp = send(backup.provider, request)           # 主请求和备份赛跑
    backup.report(ttft_ms=bresp.ttft_ms)             # 每个句柄各报各的
```

`hedge_now()` 返回的是又一个 `Decision`,与主请求对称:每派发一次就产生一个句柄,每个句柄
回报自己的结果。当此刻还不值得补发时,它返回 `None`。在内部,router 会把备份一直拖到"主
请求加备份合起来仍能满足 SLO 的成功概率刚好压住目标"的最晚检查点,所以补发很罕见,而且每
一次都有可量化的理由。开启对冲只需加 `slo_ms` 这一个改动,新增的方法只有 `hedge_now`。

## 托管执行:Client

`Router` 只做决策,执行由你负责。如果你更希望 RouteWise 连执行也一起做,`Client` 会把
"每个模型一个 router"包在一个 OpenAI 兼容的接口后面。这里模型是字典的 key(所以
`Provider` 仍然不带 model 字段),每个 provider 提供 client 调用它所需的 `base_url` 和
`api_key`:

```python
import asyncio
from routewise import Client, Provider

client = Client(
    {
        "deepseek-v4": [
            Provider("fireworks", price_in=0.27, price_out=1.10,
                     base_url="https://api.fireworks.ai/inference/v1", api_key="..."),
            Provider("together", price_in=0.18, price_out=0.88,
                     base_url="https://api.together.xyz/v1", api_key="..."),
        ],
    },
    alpha=0.25,
    slo_ms=3000,
)

async def main():
    response = await client.chat.completions.create(
        model="deepseek-v4",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(response.choices[0].message.content)
    print(response.routewise.provider, response.routewise.cost_usd)

asyncio.run(main())
```

`create()` 与 OpenAI SDK 签名一致,包括 `stream=True`。任何 OpenAI 兼容端点都能作为后端,
今天这覆盖了大多数商业 provider。token 计数、结果回报、对冲赛跑、败者取消都发生在 client
内部,你不用写任何计时代码。对于想服务多个模型的 `Router` 用户,同样的模式只是他自己的一行
代码:`routers = {m: Router(pool[m], alpha=0.25) for m in pool}`。

## 一次决策是怎么做出来的

### Alpha 旋钮

每次决策时,router 估算这个请求在每个 provider 上的成本,取最便宜的 `c_min` 和最贵的
`c_max`,把本次请求的预算定为:

```text
budget = c_min + alpha * (c_max - c_min)
```

`alpha = 0` 把预算钉在最便宜的 provider 上;`alpha = 1` 在能降延迟时允许用最贵的 provider;
中间的值连续地用钱换延迟。因为预算会相对当前价格区间自校准,`alpha` 是无量纲的:接口里任何
地方都不出现"每请求多少美元"的绝对阈值,同一个值可以跨部署、跨模型迁移。给 `route()` 传
`alpha=` 可以为单个请求覆盖默认值,这样同一个 router 能服务有不同成本-延迟取向的租户。

### 混合

在预算绑定时,延迟最优的策略是对最多两个 provider 的概率混合(例如 70% Together、30%
Fireworks),由一个小型线性规划求出。按这个比例发流量,正是"把平均成本压在预算上、同时把
平均延迟最小化"的做法,所以 `route()` 从混合里采样一个 provider,而不是返回一个固定的赢家。
完整的混合保留在 `decision.weights` 里可见。

每次调用都重新采样,所以预算保证来自多次请求累积出的 `decision.provider` 分布,而不是任何
静态顺序。让网关保持纯执行器:接 OpenRouter 时,每个请求只把采样出的那一个 provider 发过去:

```python
payload["provider"] = {"only": [decision.provider]}
```

给网关一个带 fallback 的有序列表,会变成按固定优先级路由,还会绕过 RouteWise 自己的失败路径:
`report(error=...)`、冷却、以及下一次请求重新采样。

### 学习闭环

这里的"延迟"指首 token 时间(TTFT)。router 不需要任何延迟配置:每条回报的结果喂给两个内部
估计器,一个是每 provider 的滚动 TTFT 画像(跟踪漂移),一个是输出长度估计器(在生成开始前
就给请求定价)。还没有任何观测的 provider 会被优先安排一次,所以一个全新的 router 会先把每个
provider 都探索一遍,再开始优化。

## 可观测性

`router.stats()` 返回每 provider 的流量份额、TTFT 各分位、错误计数、实际花费,以及整体的对冲
触发率和命中率。`decision.explain()` 给出单次选择的一行说明,`decision.trace` 以结构化字典
返回同样的内容:

```python
print(decision.explain())
# budget=$0.000912 (alpha=0.25, c_min=$0.000718@deepinfra, c_max=$0.001494@fireworks)
# mix: deepinfra 0.71, fireworks 0.29 -> sampled: deepinfra
# excluded: together (cooldown, 18s left)
```

## Tuning(逃生舱)

算法常数留在主签名之外,因为它们是标定好的默认值,而不是逐部署的选择:对冲成功目标、失败罚时、
画像窗口、冷却策略。如果调用方测到了改动它们的理由,就传一个 `Tuning` 对象,它同时也承载测试用
的随机种子:

```python
from routewise import Router, Tuning

router = Router([...], alpha=0.25, slo_ms=3000,
                tuning=Tuning(hedge_target=0.95, window_min=30, seed=7))
```

大多数用户从不构造 `Tuning`。

## API 参考

### Provider

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `name` | `str` | 由你的执行层解析的标识符 |
| `price_in` | `float` | 输入价,每百万 token 美元 |
| `price_out` | `float` | 输出价,每百万 token 美元 |
| `price_cached` | `float \| None` | 命中缓存前缀 token 的折扣价(provider 提供时填) |
| `base_url`、`api_key` | `str \| None` | 仅 `Client` 需要 |

### Router

```python
Router(providers, *, alpha=0.25, slo_ms=None, tuning=None)

router.route(*, input_tokens, cached_tokens=0, alpha=None) -> Decision
router.stats() -> Stats
```

当所有 provider 都在冷却中时,`route()` 抛 `NoProviderError`。`cached_tokens` 用于缓存感知的
成本估算;在多轮 agent 工作负载下,缓存折扣对"谁最便宜"的影响可能盖过标价差异。

### Decision

由 `route()`(以及 `hedge_now()`)返回的句柄。

| 成员 | 类别 | 含义 |
| --- | --- | --- |
| `provider` | 属性 | 采样出的 provider,把请求发给它 |
| `weights` | 属性 | 底层混合(至多两项非零) |
| `expected_cost_usd` | 属性 | 本次决策的期望成本 |
| `expected_latency_ms` | 属性 | 本次决策的期望 TTFT |
| `checkpoints_ms` | 属性 | 对冲再评估时刻;无 SLO 时为空 |
| `report(*, ttft_ms=None, output_tokens=None, error=None)` | 方法 | 回报结果 |
| `hedge_now(*, elapsed_ms)` | 方法 | 返回一个备份 `Decision`,或 `None` |
| `explain()`、`trace` | 方法、属性 | 人类可读与结构化的说明 |

### Tuning

```python
Tuning(*, hedge_target=0.99, penalty_ms=10000.0, window_min=15,
       cooldown_sec=30.0, cooldown_after=3, seed=None)
```

### Client

```python
Client(providers_by_model, *, alpha=0.25, slo_ms=None, tuning=None)
await client.chat.completions.create(model=..., messages=..., stream=False,
                                     alpha=None, ...)
client.routers   # 内部每模型 Router 组成的字典(可取 stats 等)
```

`providers_by_model` 把模型名映射到它的 provider 列表。响应带一个 `routewise` 属性,含
`provider`、`cost_usd`、`ttft_ms`、`hedged`、`hedge_won`。在流式模式下,client 在第一个内容
chunk 处测 TTFT,让对冲的两个流赛跑,并取消败者。

### 无状态函数

```python
from routewise import Candidate, route_once

decision = route_once(
    [Candidate("fireworks", cost_usd=0.0012, latency_ms=240.0),
     Candidate("together",  cost_usd=0.0008, latency_ms=410.0)],
    alpha=0.25, seed=7,
)
```

`route_once()` 是给"自己已经在跟踪成本和延迟"的调用方用的纯函数:它不定价、不学习,只解预算
LP 并采样一个 provider。`Router.route()` 在内部就是这个函数加上那两个估计器。复现论文结果的
研究用户从这里入手。

## 设计原则

决策与执行分离。内核不打开任何连接,因此零依赖,且与任何 HTTP 客户端或异步框架正交;内置的
`Client` 只是一种执行层,不是唯一的。成本控制只用一个无量纲旋钮,所以接口里不出现任何逐部署的
美元阈值。决策是一个句柄,所以回报结果时不重复任何请求身份。库引入的每一笔额外成本(对冲、探索)
都浮现在 `stats()` 里,而不是悄悄混进账单。

## 范围与路线图

订阅型 provider(预付请求配额、预留并发槽)是 RouteWise 算法的另一半,也是计划中的 v2 扩展。
它们会作为同一个 `Router` 上的新 provider 类型加入,本文描述的接口形状不变。模型选择永久不做,
因为 RouteWise 路由的是固定模型跨 provider。跨进程持久化或共享已学到的状态(一对
`export_state` / `state=`,或一条用于回放日志结果的离线 `observe()` 路径)推迟到 v1.1;v1 在
进程内学习,各个独立副本各自从自己的流量里学。其余路线图项是 LiteLLM 策略插件,以及一个端到端
延迟目标(v1 优化的是 TTFT)。

## 实现备注

本节面向 RouteWise 开发者,库用户读到这里即可停下。接口到 `routewise.core` 的映射如下:
`route_once()` 在 `solve_budget_lp()` 外面包上 alpha 到 budget 的换算和权重采样;`Router` 再
加上滚动延迟画像和分桶均值输出长度估计器(二者已在 `rwsim` 和生产网关里原型化);`hedge_now()`
包住 `hedge_checkpoints_for_slo()`、`combined_success_probability()` 和
`select_probability_backup()`。

有三处行为是契约性的,而非偶然。其一,主选择从 LP 权重采样;改成 argmax 会破坏预算保证。其二,
延迟打平时向更便宜的 provider 倾斜(`cost_tiebroken_objective()`),在内部完成,调用方看不到。
其三,一个 router 绑定一个模型,这把 model 参数从 `route`、`report`、`Provider` 里移除;有多个
模型的调用方用一个 router 字典作 key,正如 `Client` 内部所做。

`L`/`U` 稀缺性区间、配额状态、并发状态在本接口里全不出现。只有 API provider 时,一个请求的
有效成本等于它的估算计费成本,所以配额影子价所需的成本区间(见 [CORE_API.md](CORE_API.md))
根本不会被构造出来。
