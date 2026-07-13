# RouteWise 库接口

> 状态：已实现的设计提案，第 5 次修订（2026-07-12），等待 Juncheng 做最终
> 契约评审。范围仅限 API 提供商；initial release（package `0.2.0`）是论文
> 系统的 **API-only preview**。第 5 次修订直接构建在具备 capacity 意识的
> 第 4 版之上：完整保留该版的 capacity 扩展接缝，并加入 GO/NO-GO 评审轮
> 的各项决定——补全的 attempt 状态机（`declined`、adoption 与结果分离、
> `unresolved` 终态）、单次原子 delta 的 billing 迁移、仅支持“现在”且基于
> 可注入 monotonic clock 的 observation、`kind`/`code` 错误模型、冻结的
> 公开签名与校验，以及 release-gate 表。下文的 `Router`、`Decision`、
> `Attempt` 和 `route_once` 接口现已在 `codex/api-provider-library-v1` 上完成
> 原型实现，包括表 A/表 B。公共 API 冻结与 `0.2.0` 发布，以其余开放问题
> 全部关闭且 release gates 通过为准。它们所依赖的数学原语目前已经存在，
> 并记录在 [CORE_API.md](CORE_API.md) 中。本文档将该接口
> 命名为“API v1”；首次交付它的 package 版本是 `0.2.0`，关于 `1.0.0` 的
> 任何讨论，都要等冻结后的 API 经过公开使用检验再说。

## 相比第 1 版的变化（截至第 4 版）

1. `report()` 的关键字参数被类型化生命周期方法
   （`first_token`、`completed`、`failed`、`cancelled`）以及用于处理
   延迟到达账单的只写一次 `settle()` 步骤取代。被取消的 hedge loser
   不会再被误认为失败，终态之后才到达的账单也仍可记录。
2. `Decision` 表示逻辑请求并拥有其所有尝试；decision 上的结果方法作用于
   primary attempt，而 `hedge_now()` 返回 backup `Attempt`。一个逻辑请求
   最多有一个 winner，也可能没有。
3. 新增 `router.observe()` 入口和两种冷启动模式。仅靠结果上报会让从未被
   选中的提供商得不到数据。内建探索会把未建档提供商混入预算可行的 mixture，
   因而每个 decision（无论是否探索）的预测期望成本都保持在预算内。凡经由
   exploration mixture（`q > 0`）路由的 decision 都会打标，而不只是随机
   draw 落在 target 上的那些。
4. 概率 hedging 受 profile 深度（`hedge_min_samples`）约束，且
   `hedge_now()` 具有状态机：每个 decision 只有一个 backup slot，并发下
   原子操作；primary 的 first token、adoption 和终止都会将其关闭。
5. `route()` 新增 `exclude=`。请求成本边界依据当前请求的 eligible 集合
   计算，并将重试的后果（排除最便宜的提供商会抬高 `alpha=0` 的预算）
   明确写入契约。
6. 错误会被分类：health failure 会受到 penalty 并推进 cooldown；request
   failure 只计入统计。`first_token` 之后发生失败会保留 TTFT 样本，不再
   增加第二条 latency penalty。
7. 明确定义 cooldown 语义：连续 health failure、到期、成功后重置。
   手动 `exclude=` 永远不会改变提供商状态。
8. 缓存状态输入接受按提供商给出的值；每个提供商的 cache 独立预热。
9. `seed` 从 `Tuning` 移到 `Router` 构造函数；`Tuning.penalty_ms`
   从 10000 修正为 60000，与已交付语义（`DEFAULT_ERROR_PENALTY_MS`）一致。
10. Alpha 被定义为对预测期望成本的约束，而非单请求上限，也不是对实际账单的
    保证。实际支出只汇总已上报的值；路由时估计永远不会计入其中。
11. `stats()` 收缩为 router 能够如实度量的 schema：使用
    `primary_selections` 和 `hedges.offered`，而非“requests”，因为 router
    只能观察 selection、offer 和 report，永远观察不到 dispatch。
12. `route_once()` 返回不可变的 `RouteOnceResult`，而非可上报的
    `Decision`，并明确给出 `Candidate` 契约。
13. 明确 packaging 契约：wheel 仅交付库本身，包含 `py.typed` 和固定的
    顶层导出列表。
14. `Client` 和 LiteLLM plugin 移至后续 release；initial release
    （`0.2.0`）只交付无依赖的 `Router`。等待 Juncheng 签字确认。
15. v2 subscription 路线图不再承诺接口保持不变；capacity reservation
    会增加交互。`routewise.core` 仍是高级扩展接缝。
16. 披露 output-length estimator 的精确 fallback cascade（bucket 样本达到
    5 条后使用 bucket mean，global 样本达到 20 条后使用 global mean，
    二者均未达到前使用 500 tokens），因为它要在任何结果到达之前为请求定价。
17. initial implementation 会预留一个内部 capacity transaction 接缝，
    `0.2.0` 中仅由 no-op API controller 实现。Capacity admission 与 L/U
    scarcity pricing 分离，也与 `routewise.core` 消费的纯 `ProviderView`
    分离；目前不导出任何 capacity API。

## 第 5 版的变化

1. attempt 生命周期新增 `declined()`：表示一个从未被派发的 attempt，通常
   是你选择不发送的 offered backup。`cancelled()` 现在只表示“已派发、随后
   中止”。二者都不写入 penalty。capacity 接缝同步对齐：因 commit 失败而被
   取代的 attempt 以 `declined` 关闭，而不是 `cancelled`。
2. Adoption 与结果分离并改名：`adopted=True`（原 `selected=`）标记你实际
   采用其响应的 attempt。被采纳的 attempt 仍可能失败或被取消；此时逻辑
   请求随之进入相同结果，且没有 winner。**winner = adopted 且
   completed。** 只有 winner 训练 output-length estimator；`hedges.won`
   只统计 adopted 且 completed 的 backup。adoption 不能事后补报：它只随
   `first_token` 或 `completed` 传入，不存在 `adopt()` 方法，`settle()`
   也不携带 adoption 标志。
3. 逻辑请求具备完备的归结规则，覆盖此前未定义的情形：存在 adoption 时，
   decision 与被采纳 attempt 的终态一致；没有 adoption 时，若有任何
   attempt completed 则归结为 `unresolved`（计入
   `decisions_without_adoption`），否则有失败则为 `failed`，否则有取消
   则为 `cancelled`，否则为 `declined`。
4. 路由时的缓存输入改名为 `estimated_cached_tokens`；settlement 中上报的
   `cached_tokens` 才是账单真值。每个 attempt 在创建时固化一份 price
   snapshot，calculated cost 一律用该 snapshot 计算。
5. billing 是每 attempt 一台三状态机，状态集合为 `{unknown, calculated,
   actual}`——显式 `cost_usd` 先到时会跳过 calculated——并采用单次 delta
   规则：每次上报调用先整体校验，计算出该 attempt 唯一的目标状态，再以
   一次原子 delta 调平聚合值。迟到的显式 `cost_usd` 会
   扣除该 attempt 的 calculated 贡献并加入 actual 金额。
   `unsettled_attempts` 拥有显式的增加与减少规则。per-provider 支出包含
   backup attempt；`hedges.*_spend_usd` 是同一笔钱的交叉切片，绝不能与
   前者相加。
6. `observe()` 在 `0.2.0` 中去掉 `at=`：所有 observation 都由 router 的
   可注入 monotonic clock（`Router(clock=)`）盖“现在”戳。历史 bootstrap
   将成为以后独立的 API，而不是把两套时钟塞进同一个参数。
7. `failed(kind=..., code=...)` 取代字符串 taxonomy。行为只由 `kind`
   决定（`"health"` 或 `"request"`）；`code` 只是标签，未知 code 在
   stats 中聚合为 `"other"`，从而保证 metrics 基数有界。key 过期算不算
   provider-health 事件由调用方判断。
8. 内建探索只承诺 body-routing 自启动。一个 window event 即结束冷启动，
   而 hedging 需要 `hedge_min_samples` 条成功样本，因此低流量 provider
   可能在具备 hedge 资格之前很久就已可路由；router 不会为弥合该差距而
   强行追加探索。
9. cooldown streak 的“成功”定义为任何进入该 provider 窗口的 TTFT 样本，
   无论它来自 `first_token`、`completed` 还是一次成功的 observation。
10. 类型与校验在契约表 D 中冻结：provider 身份参数与输出一律为 `str`
    名称，`Router` 构造函数是唯一接受 `Provider` 对象的签名；缓存映射
    缺失某个名称按 0 处理，未知名称则报错；公开
    mapping 深度不可变；所有 token、成本和延迟参数都校验为有限且非负；
    参数校验失败和 outcome 冲突不会提交业务状态；capacity 全部失败时也不
    提交 selection、lease、counter、spend 或 RNG 变更，但窗口 bookkeeping
    可以推进到本次操作捕获的 `now`，不会回滚。
11. `route_once()` 接受可复用的 `rng=`，与 `seed=` 互斥。固定 seed 下，
    相同输入与权重会重放同一次抽样，破坏长期经验 mixture；单次调用的
    LP 解与 budget 不受影响。契约已明确警告这一点。
12. Hedging 定义 survival-zero fallback：当 primary 的经验分布认为它早该
    完成、而现实中它仍未出 token 时，primary 的剩余机会按零计，联合概率
    退化为 backup 单独的成功概率。当前 core 在该情形直接返回 0.0，公开
    hedging 前必须由 facade 兜底或修复 core。
13. router 不持有任何未终结 handle 的强引用；attempt 状态存放在 handle
    上，router 侧残留以 exploration lease 超时加上任何未关闭的 capacity
    reservation（`0.2.0` 中为 no-op）为上界。
14. `0.2.0` 定名为 API-only preview。论文的主要结果包含 quota 与
    concurrency 路由，因此本 release 不声称复现完整论文。

## RouteWise 是什么

RouteWise 是一个 Python 库，面向通过多个 API 提供商调用同一模型的应用。
DeepSeek-V4 等开放权重模型由许多提供商销售，价格各不相同，延迟也会因提供商而异
并随时间漂移。对于每个请求，RouteWise 都会决定使用哪个提供商：在成本符合预算的
选项中选择期望延迟最低者，并由一个旋钮控制。当响应有可能错过 deadline 时，
RouteWise 可以向第二个提供商延迟派发 backup 请求。router 会从你告知的数据中学习：
包括自身 decision 的结果，以及你从外部送入的任何测量值。

RouteWise 不是通用 LLM client（不交付任何提供商专用 SDK），不是 model selector
（模型固定，只改变提供商，因此绝不会为了成本牺牲响应质量），也不是托管服务
（你的 API key 始终留在自己的进程中）。

## 安装

```bash
pip install routewise
```

initial release（package `0.2.0`）是 API-only preview：只包含决策库本身，
且不会导入标准库以外的任何内容。论文的完整系统还包含 quota 与 concurrency
subscription 的定价；它们属于后续代际，因此本 release 不声称复现完整论文
结果。execution client（`routewise[client]`，基于 httpx）和 LiteLLM
routing-strategy plugin（`routewise[litellm]`）计划在后续 release 中作为
可选 extra 提供；`0.2.0` 不定义任何 extra。参见“范围与路线图”。

## 完整接口

一个 router 服务于一个模型。你向它询问请求应发往何处，用自己的代码发送请求，
再把发生的结果告诉 decision，让下一个 decision 获得更充分的信息。

```python
from routewise import Provider, Router

router = Router(
    [Provider("fireworks", price_in=0.27, price_out=1.10),
     Provider("together",  price_in=0.18, price_out=0.88)],
    alpha=0.25,          # 唯一旋钮：0 = 最便宜，1 = 最快
    seed=42,             # sampling 是随机的；设置 seed 以便复现
)

decision = router.route(input_tokens=1800)           # 询问：使用哪个提供商？
response = send(decision.provider, request)          # 你的执行层
decision.first_token(ttft_ms=response.ttft_ms)       # stream 已开始
decision.completed(output_tokens=response.output_tokens)   # 并已结束
```

常见路径由三个名词（`Provider`、`Router`、`Decision`）、两个时刻（先用
`route()` 询问，再告诉 decision 发生了什么）和一个旋钮（`alpha`）组成。
这个简写描述的是常见路径，而不是完整系统：完整的 v1 接口还增加了 `observe()`，
用于接收并非由 router 发起的测量；启用带 `slo_ms` 的 hedging 时，还增加了
`hedge_now()` 这一询问。价格单位为每百万 token 的美元数。`Provider.name`
是一个不透明 label，由你的执行层解析；`Router` 本身不会发起网络调用。

调用者还可能希望从 decision 获取的其他所有内容，都是只读属性，因此无需额外学习：

```python
decision.provider            # 本次 sampling 得到、应使用的提供商
decision.weights             # 底层 mixture，例如 {"together": 0.7, "fireworks": 0.3}
decision.expected_cost_usd   # 该 decision 的预测期望成本
decision.expected_latency_ms # 期望 TTFT；探索未建档提供商时为 None
decision.explain()           # 用一行人类可读文字解释这次选择
```

## 上报结果

`route()` 返回一个 `Decision`，即逻辑请求。其背后有一次或多次派发，每次都由
一个 `Attempt` handle 表示；primary attempt 随 decision 一同创建，
`hedge_now()` 则添加 backup。decision 已经持有请求身份信息（提供商、输入长度、
缓存长度），所以结果调用只携带新增信息。attempt 的各种结束方式含义不同，因此结果
使用类型化方法，而不是一个带关键字参数的 `report()`：

```python
decision.first_token(ttft_ms=312.0)      # stream 已开始：一条 TTFT 样本
decision.completed(output_tokens=540)    # 已完成；记录已知 usage
decision.settle(cost_usd=0.00092)        # 账单稍后到达

# 用以下终态结果代替 completed()：
# decision.failed(kind="health", code="rate_limited")  # penalty + cooldown
# decision.cancelled()                   # 已派发后由你中止：无 penalty
# backup.declined()                      # 被 offer 但从未派发
```

`Decision` 上的结果方法作用于其 primary attempt；每个 backup `Attempt`
都使用相同的方法上报自身结果。

**生命周期。** `first_token` 最多调用一次，且必须在任何终态调用之前；它会在
提供商的 latency profile 中记录一条 TTFT observation，并作为成功信号重置该
提供商的 failure streak。每个 attempt 只允许进入一个终态（`completed`、
`failed`、`cancelled` 或 `declined`）；`declined` 只能从 `pending` 进入，因为
已经开始的 stream 按定义就是派发过的。重复完全相同的调用是 no-op；与先前调用
矛盾的调用（不同的终态，或为已知字段提供不同值）会抛出 `OutcomeError`。
`first_token` 之后发生失败会保留 TTFT 样本，不会再写入第二条 latency
penalty：一次 attempt 永远不会记两条 latency entry。该失败的 `kind` 仍照常
生效，因此 health failure 会推进 cooldown，而 request failure 只计入 stats。
从不上报的 handle 不会让 router 学到任何信息；该 attempt 已经花掉的钱也不会被
记录，因此 `stats()` 会少算。对你实际派发的每个 attempt 都应上报其终态，并在
billing 字段变为已知时完成结算。完整的状态转换表见契约表 A。

**Settlement 只写一次，且以单次 delta 施加。** billing 真值经常在终态之后
才到达，例如提供商的 usage record、hedge loser 的部分账单或 cache-hit
count。三个 billing 字段（`output_tokens`、`cached_tokens`、`cost_usd`）在
每个 attempt 上都只能提供一次：可以随终态调用提供，也可以稍后通过
`settle()` 填入仍未知的字段。此处的 `cached_tokens` 是账单真值，与路由时的
`estimated_cached_tokens` 相区别。`settle()` 在除 `declined` 外的任何终态下
合法：declined 的 attempt 从未被派发，无账可结，对它 settle 会抛出
`OutcomeError`。再次提供相同值是 no-op；提供不同值会抛出
`OutcomeError`。不存在 overwrite 路径，也不需要：路由时估计从不进入实际
支出，因此 settle 的值始终是第一个实际值，而不是对某个值的修正。每次上报
调用都先整体校验，然后把该 attempt 的 billing 状态重算到唯一目标（已知
`cost_usd` 则为 `actual`，否则可计算则为 `calculated`，否则为 `unknown`），
再按契约表 B 以一次原子 delta 从先前状态调平聚合值。因此，一次同时给出
`output_tokens` 和 `cost_usd` 的调用会直接落到 `actual`，绝不会向 calculated
支出贡献任何金额。所以 `completed()` 的 `output_tokens` 是可选的；如果 usage
稍后才到达，就先省略它，之后再 settle。`failed`、`cancelled` 和 `declined`
自身不携带任何 billing 参数；金额总是通过 `completed` 或 `settle` 流入。
`cancelled()` 默认刻意不上报任何内容，因为 attempt 被中止时，其 usage 通常
仍未知；零也是一种明确声明，所以除非确知为零，否则不要写入。若提供商将来会
修订已经开出的账单，应在后续版本中设计显式 revision 机制，而不是在这里允许
overwrite。

`completed(output_tokens=None, ttft_ms=None, cached_tokens=None,
cost_usd=None, adopted=None)` 标记正常完成，并记录所提供的任何 usage 或 billing
字段。仅调用它并不能保证金额成本已经明确。非 streaming 调用者跳过
`first_token`，改在这里传入 `ttft_ms`。

`failed(kind=..., code=None)` 把影响与标签分开。router 的全部行为只由
`kind` 决定：

| `kind` | 影响 |
| --- | --- |
| `"health"` | 一条 penalty 样本（默认 60 s；若已记录 `first_token` 则不再写入），并推进 cooldown |
| `"request"` | 计入 `stats()`；无 penalty、无 cooldown |

`code` 是用于可观测性的自由标签。推荐 code：health 类为 `rate_limited`、
`timeout`、`server_error`、`connection`；request 类为 `bad_request`、
`auth`、`unsupported`。推荐清单之外的 code 在 `stats()` 中聚合为
`"other"`，因此 metrics 基数有界。`kind` 由调用方判断：key 过期期间你在修
配置时可以报 `kind="request"`；若你希望该提供商被搁置，也可以报
`kind="health"`。推荐 code 清单见“开放问题”。

**Adoption 与 winner。** adoption 与结果是两个独立事实。在你真正采用某个
attempt 的响应的那一刻，用 `adopted=True` 标记它；router 无法根据完成顺序
推断采用关系，因为在 streaming race 中，primary 可能已经开始向你的用户输出，
而 backup 却先生成完毕。

- 当 decision 只有 primary attempt（从未 offer 过 backup）时，`completed`
  的 primary 自动被采纳。简单路径永远不写 `adopted=`。
- 一旦创建过 backup attempt，adoption 就必须显式：在 `first_token` 上
  （streaming，在你采用该 stream 的时刻）或在 `completed` 上（非
  streaming）传入 `adopted=True`。存在多个 attempt 却没有显式 adoption
  时，没有任何 attempt 被采纳：completion 仍会记录支出，output-length
  estimator 不会接受任何训练，`hedges.won` 也不会变化。
- 将两个 attempt 标记为 adopted 会抛出 `OutcomeError`。adoption 不能事后
  补报：不存在 `adopt()` 方法，`settle()` 也不携带 adoption 标志。
- **winner = adopted 且 completed。** 被采纳后失败或被取消的 attempt 仍是
  被采纳的，但该请求没有 winner。

逻辑请求的归结方式唯一确定。存在 adoption 时，decision 与被采纳 attempt 的
终态一致：`completed`（存在 winner）、`failed` 或 `cancelled`。没有
adoption 时，待所有 attempt 进入终态后：若有任何 attempt completed，则为
`unresolved`（本可采纳却始终未声明；`decisions_without_adoption` 加一，且
不训练任何 estimator）；否则若有失败则为 `failed`；否则若有取消则为
`cancelled`；否则为 `declined`。

Streaming race，采纳 primary：

```python
decision.first_token(ttft_ms=312.0, adopted=True)
```

或者，在另一个独立的非 streaming race 中，采纳 backup：

```python
backup.completed(output_tokens=540, adopted=True)
```

**记账规则，属于契约的一部分：**

1. 每个 attempt 的 `first_token` 都是对其提供商的一次真实 observation：它会
   送入该提供商的 latency profile，并重置其 failure streak。
2. output-length estimator 只在某个 attempt 是 winner（adopted 且
   `completed`）时训练，在其 `output_tokens` 变为已知的时刻进行，且仅当
   该值为正；零输出的 completion 记账但不训练。其他所有情况——包括某个
   被采纳的 attempt 随后失败并结算了部分 usage——都只更新支出。
3. 被取消或被 declined 的 attempt 永远不会写入 penalty，也永远不会推进
   cooldown。
4. 一个 attempt 最多记一条 latency entry：一条 TTFT 样本或一条 failure penalty，
   二者绝不同时出现。
5. `route()` 上的手动 `exclude=` 不会改变任何提供商状态。
6. router 不持有任何未终结 handle 的强引用：attempt 状态存放在 handle 上，
   router 只保留聚合值、exploration lease 表和任何未关闭的 capacity
   reservation（`0.2.0` 中为 no-op）；被丢弃的 handle 会被垃圾回收，router
   侧残留不超过上述上界。

## 保持 Profile 新鲜

仅靠已上报结果无法让 router 始终准确。selection 会偏向看起来表现良好的提供商，
因此从未被选中的提供商也永远不会产生结果：它无法摆脱糟糕的第一印象，空闲提供商的
漂移也不会被观察到。该库通过冷启动行为和 observation 入口闭合这一反馈循环；生产
部署则通过该入口加入自己的 probe。

**冷启动。** `Router` 构造函数接受 `cold_start=`，共有两种模式。

默认的 `cold_start="explore"` 会以预算可行的 mixture 将未建档提供商混入路由。
如果一个提供商的当前窗口完全没有 event，就算作未建档；只要有一条 health-failure
penalty 就已经算有 event，因此刚刚失败的提供商归 cooldown 管，而非 exploration。
只要至少存在一个 eligible、未持有 lease 的未建档提供商，router 就会从中
选择最便宜者作为 exploration target（同价时按名称排序），
并使用 target `u` 与最便宜 eligible 提供商之间的两点 mixture 来路由请求；在保证
mixture 的预测期望成本不超预算的前提下，为 `u` 分配尽可能大的概率：

```text
q = 1                                   if c_u <= budget
q = (budget - c_min) / (c_u - c_min)    otherwise
```

当 `q` 为零时（`alpha=0` 且任何 target 都贵于 floor），router 会完全跳过探索，
按普通 LP 路由：不打 tag、不增加 counter、不占 lease；探索不能把流量送往预算绝不
可能接纳的地方。当 `q > 0` 时，凡由 exploration mixture 路由的 decision，都会在
`explain()` 和 `trace` 中打上 `reason="cold_start_exploration"` 标签（trace 还会
记录 target、`q` 和 `latency_estimate="unprofiled"`），无论 draw 是否落在 target
上，因为两种情况下请求都已经偏离 LP 最优解。`stats()` 同时统计两项：
`exploration.decisions` 统计经 mixture 路由的 decision，
`exploration.target_selected` 统计命中 target 的 draw。只有 draw 落在 target 上时，
才会原子取得该提供商的 exploration lease，因此取得 lease 的 decision 的 primary
就是 target 本身；draw 落在最便宜提供商上不会消耗 lease。后续请求会跳过已取得
lease 的 target，直至 lease 释放：在 target 的第一个 window event 到达时，或经过
`Tuning.exploration_lease_sec` 后（以先发生者为准），因此从不上报的 exploration
handle 不会永远阻塞该提供商。lease 属于取得它的 attempt；陈旧 attempt 的迟到 event
绝不会释放较新的 lease。当本次请求已不存在 eligible、未持有 lease 的未建档 target 时（每个候选
要么已持有 lease、要么被 exclude、要么处于 cooldown），该请求跳过探索：
在其余 eligible 提供商上走普通 LP；若一个都没有，`route()` 抛出
`NoProviderError`，其消息如实报告清空集合的各项实际原因；只有当集合确由
active exploration lease 单独排空时，才使用冷启动 lease 已占满的文案。探索与
其他所有 decision 一样遵守预算；其成本仍可能高于
不探索时的最优解，而且 non-target 概率份额会被路由到最便宜而非最快的提供商，所以
探索的代价体现在延迟和最优性上，绝不会体现在违反预算上。

内建探索只承诺 body-routing 自启动，别无其他：一个 window event 即结束某个
提供商的冷启动，而 probability hedging 需要 `hedge_min_samples` 条成功样本，
因此低流量提供商可能在具备 hedge 资格之前很久就已经可路由。router 不会为
弥合这一差距强行追加探索；hedge 资格来自自然流量，或来自你通过 `observe()`
执行的 warmup。

严格模式 `cold_start="require_observations"` 会令未建档提供商不具备 eligibility。
如果所有提供商都未建档，`route()` 会抛出 `NoProviderError`，并提示先为 profile
植入数据。这是生产模式：在接收流量前，用 probe 引导每个提供商。

内建探索是一种方便自启动的机制，并非研究 harness 的 warmup 规程。该 harness
最初以 5 秒 cadence 执行 24 轮 probe（约两分钟），并在 replay 前验证每个提供商
至少有 5 条样本；每个实际提交、且耗尽其配置的全部 attempts 后仍失败的 warmup
probe，都会注入一条 synthetic 10 秒样本（由于 probe 仍在 in-flight 而跳过的轮次
不会注入任何内容），因此一个所有 probe 都失败的提供商将只依据 synthetic 样本排序，
validation threshold 并不保证存在真实成功。steady-state probe failure 会被丢弃。
5 秒 probe cadence 和 synthetic 10 秒样本已出现在论文修订后的 profiling
文本中；24 轮 warmup 形态和 5 条样本的 validation threshold 则是 harness
默认值。希望获得 warmup 语义的部署需要运行自己的 probe
loop，并将数据送入 `observe()`；router 不调度 probe，也不运行 timer。论文还
提出过 exploratory hedging（把 hedge 派发用作有机探测）；那是与冷启动探索不同
的另一种机制，不在本 release 范围内，因为冷启动时既没有可 hedge 的流量，也
未必配置了 SLO。

**Observation 入口。** `router.observe()` 接收并非由 router 发起的测量：

```python
router.observe("together", ttft_ms=284.0)                  # 你的 probe 成功了
router.observe("fireworks", kind="health", code="timeout") # 你的 probe 失败了
```

`observe()` 是 measurement 入口，而不是 probe API：你送入的所有内容都按 routed
request 产生的数据同样计入，只是不参与 traffic-share accounting。成功会增加一条
TTFT 样本并重置 failure streak；失败遵循与 `failed()` 相同的 `kind`/`code`
模型，因此 health failure 会增加一条 penalty 样本并推进 cooldown。请自行选择要
上报的内容：production sidecar 会丢弃失败的 probe，而不是上报，因为与真实请求
反馈相比，失败的 probe 证据较弱；希望采用相同策略的 probe loop 只上报成功。
每条 observation 都由 router 的时钟盖“现在”戳；`0.2.0` 中没有 `at=`，因此暂不
支持携带原始时间戳的历史 replay。bootstrap 的方式是把近期测量按当前数据重放；
专门的历史导入 API 可以以后再加，而不必把两套时钟域塞进同一个参数。当前的两类
用途：定期 out-of-band probe（只需几行你自己的调度代码），以及通过重放 peer
replica 的近期测量为新 router 预热。

**Thin profile 与 hedging。** 一条 live 样本足以解除数据饥饿，却不足以构成 latency
distribution。当前窗口内成功样本少于 `Tuning.hedge_min_samples` 的提供商完全不参与
probability hedging，无论充当哪种角色：它们不会被 offer 为 backup；当 primary 的
窗口样本如此稀少时，`hedge_now()` 也会返回 `None`。body routing 可以容忍 thin
profile，tail probability math 则不行。这道 gate 是新的 library policy：默认值 5
借用了 research harness 的 warmup validation threshold，而当前没有 production
hedger 强制执行 per-provider sample minimum，所以该值仍需验证（参见“开放问题”）。
## 尾延迟对冲

路由器优化的是延迟分布的主体，而对冲保护的是分布尾部。为路由器设定一个
SLO（服务级别目标，即首 token 到达期限），每个决策便会携带一组检查点时间；
到达这些时间点时，发送一个备用请求可能值得：

```python
router = Router([...], alpha=0.25, slo_ms=3000, seed=42)

decision = router.route(input_tokens=1800)
# 主请求正在进行中。在 decision.checkpoints_ms 的每个时间点，如果
# 首 token 尚未到达：
backup = decision.hedge_now(elapsed_ms=1500)         # None 或一个 Attempt
if backup is not None:
    dispatch(backup.provider, request)               # 与进行中的主请求竞速
```

当尚不值得发送备用请求时，`hedge_now()` 返回 `None`；否则，它返回这个决策
唯一可能拥有的备用 `Attempt`。其状态机属于契约的一部分：每个决策只有一个备用
槽位；并发调用 `hedge_now()` 时，所有调用合计至多产生一个 `Attempt`（该调用为
原子操作）；槽位一旦使用，后续调用便返回 `None`；一旦主 attempt 记录首 token，
尚未使用的槽位就会立即关闭，因为对冲保护的是首 token 到达时间，而首 token
到达后便已无可保护之物；一旦某个 attempt 被采纳，或逻辑请求进入终态，
`hedge_now()` 同样始终返回 `None`。`elapsed_ms` 由你测量，以主请求派发时刻为
基准；router 每次调用还会读取一次自己的时钟，用于评估当前 profile、cooldown
和 backup eligibility。无论你是否实际派发，返回一个 `Attempt` 都会消耗该
槽位。路由器从不观察派发行为，因此 `stats()` 统计的是 `hedges.offered`，
而不是已派发的数量；如果你决定不发送已提供的备用 attempt，请调用
`declined()` 关闭其句柄，以确保记账真实准确（`hedges.declined` 统计这类
情况）。`hedges.won` 统计被采纳且完成（adopted 且 completed）的备用 attempt。

竞速结束后，请按每个 attempt 的实际结果分别上报：

```python
decision.first_token(ttft_ms=1710.0, adopted=True)   # 采纳主 attempt
decision.completed(output_tokens=540)
backup.cancelled()                                   # 已派发的败者：不施加惩罚
backup.settle(cost_usd=0.0002)                       # 败者的账单，已知后写入
```

如果备用 attempt 获胜，调用方式与之对称。`Attempt` 并不是 `Decision`：备用
attempt 是由对冲算法选出的单次潜在派发，因此它没有混合策略、预算或
自己的检查点，也不能继续对冲。路由器会在内部暂缓提供备用 attempt，
直到最晚的那个检查点——在该时间点，主 attempt 与备用 attempt 的联合 SLO
达标概率仍能超过目标值。这样既能让备用请求保持少见，也能确保每次备用请求都有
量化依据。

有一个边界情形属于契约。当主 attempt 的经验分布对已观测的 elapsed 时间给出
零生存概率（窗口内的所有样本都小于该时间），而该请求在现实中确实还没有首
token 时，以观测事实为准：主 attempt 在 SLO 内的剩余机会按零计，联合概率
退化为备用 attempt 在剩余预算内的成功概率。当前 core 的
`combined_success_probability` 在这一情形下会直接返回 0.0，恰好在主 attempt
看起来最无望的时刻拒绝对冲；在 core 修复之前，facade 必须应用该
fallback（实现差距第 8 条）。要启用对冲，唯一需要做的就是添加 `slo_ms`。

## 决策如何产生

### Alpha 旋钮

每次决策时，路由器都会估算该请求在每个符合条件的 provider 上的成本，并取其中
最便宜和最昂贵的值作为该请求的成本边界 `c_min` 与 `c_max`。该请求的预算为：

```text
budget = c_min + alpha * (c_max - c_min)
```

`alpha = 0` 将预算固定在最便宜的合格 provider 上；`alpha = 1` 允许使用
最昂贵的 provider；介于两者之间的值则以连续方式用金钱换取延迟。合格集合
由你传入的 providers，减去 `exclude=` 中的 provider，减去处于 cooldown 的
provider，再减去持有 active exploration lease 的 unprofiled provider（explore
模式）或全部 unprofiled provider（严格模式）组成；边界就在该集合上计算。因此，`alpha` 表达的是当前可达成本区间内的相对位置，这使它成为
无量纲参数，并能跨部署与模型迁移；接口中不会出现任何绝对的“每请求美元数”阈值。

以下三点属于契约。第一，集合变化时，边界也会变化。排除最便宜的 provider 会
抬高 `c_min`；因此，当某个请求以 `exclude={failed_cheapest}` 重试时，
`alpha=0` 重试的绝对预算就是第二便宜 provider 的价格。第二，预算约束的是决策
混合策略的预测期望成本，而非每个单独请求：把 70% 的同类请求发送给便宜
provider、30% 发送给昂贵 provider，可以让平均成本保持在预算上，尽管每个请求
实际都会落在预算的一侧或另一侧。第三，这项约束基于预测值，使用路由时的输出
长度估计来定价；如果生成内容变长，最终账单可能有所不同，而备用对冲请求的支出
则完全不计入预算。实际支出可在 `stats()` 中查看，绝不会被静默调平。

向 `route()` 传入 `alpha=`，可以只为单个请求覆盖默认值，从而让同一个路由器
服务具有不同成本—延迟目标的租户。对于路由器无法自行获知、且仅与当前请求有关的
不合格条件，请传入 `exclude=`：例如 provider 的上下文窗口容纳不下请求、请求
需要某种能力，或某租户被规定不得使用某一区域。cooldown 负责处理 provider 健康度；
`exclude` 只处理当前请求，绝不改变 provider 状态。

### 混合策略

当预算约束生效时，延迟最优策略是一个最多包含两个 provider 的概率混合策略
（例如 70% Together、30% Fireworks），由一个小型线性规划计算得出。按照这些
比例发送流量，才能在最小化平均延迟的同时将平均成本维持在预算上；因此，
`route()` 会使用路由器自身以 seed 初始化的 RNG，从混合策略中采样一个 provider。
这种采样属于契约：若改为固定取最大概率项，就会破坏预算保证。完整混合策略始终可
通过 `decision.weights` 查看。

每次调用都会重新采样，因此预算保证来自许多请求累计后的
`decision.provider` 采样结果，而不是任何静态顺序。让网关只做纯执行器。
使用 OpenRouter 时，每个请求只发送被采样中的那个 provider：

```python
payload["provider"] = {"only": [decision.provider]}
```

如果向网关传入一个有序回退列表，它将改为按固定优先级路由。这样还会
绕过 RouteWise 自己的失败处理路径：`failed(...)`、cooldown，以及下一次请求时
重新采样。

### 学习循环

这里的延迟指首 token 到达时间（TTFT）。结果与外部观测会写入每个 provider 的
滚动 TTFT profile，以追踪漂移；只有输出长度为正的、已完成的 winner，才会训练
输出长度估算器，而该估算器用于在生成开始前为请求定价。长度估算器按输入长度分桶
取均值，并采用固定的回退级联：当请求所属输入长度桶已有 5 个样本时，使用该桶的
均值定价；当估算器总计已有 20 个样本时，使用全局均值；在达到任一阈值之前，使用
固定的默认值 500 个 output tokens。该默认值决定全新路由器的成本估计，并进而
决定其混合策略。是否允许调用方按请求覆盖该估计（`estimated_output_tokens=`，在已知
`max_tokens` 时很有用）仍是一个待决问题。缓存定价：`estimated_cached_tokens`
按 `price_cached` 为输入打折；`price_cached=None` 的 provider 按 `price_in`
全价计费缓存 token，因此该估计对它没有影响。冷启动和空闲 provider 的
覆盖机制见“保持 Profile 新鲜”一节。

## 可观测性

`stats()` 返回不可变的 `StatsSnapshot`，只包含路由器能够如实测量的数据：它能
观察主 provider 的选择、备用 attempt 的提供以及你的上报，但绝不观察实际派发。
下面的 schema 是待冻结候选稿；一旦签署确认，字段今后只能增加，不能修改或
删除。

每个 provider 包含：`primary_selections`（采样主 provider 为该 provider 的
决策数）、当前窗口内的 `ttft_p50_ms` 和 `ttft_p95_ms`（窗口为空时为 `None`）、
`errors`（先按 `kind` 再按 `code` 计数，未知 code 归入 `"other"`）、
`cooldown_remaining_sec`（健康时为 `0.0`）、`actual_spend_usd`、
`calculated_spend_usd` 和 `unsettled_attempts`。per-provider 支出包含被路由到
该 provider 的 backup attempt。

全局包含：`hedges.offered`、`hedges.declined`、`hedges.won`（adopted 且
completed 的 backup）、`hedges.actual_spend_usd`、
`hedges.calculated_spend_usd`（对 per-provider 支出的交叉切片，绝不能与其
相加）、`exploration.decisions`、`exploration.target_selected` 和
`decisions_without_adoption`。

支出来源遵循契约表 B：显式 `cost_usd` 计入实际支出；没有显式 cost、但最终
用量完整时，计入计算支出；已上报终态但计费数据不足的 attempt 计入
`unsettled_attempts`，且绝不会把未知金额计入任何金额合计。未上报的 attempt
不可见。counter 的生命周期（router 生命周期还是 rolling window）属于开放
问题 7。更丰富的内容（历史记录、百分位曲线、按租户拆分）属于你的遥测系统，
可从 decision trace 中获取。

`decision.explain()` 用一行文字解释单次选择，`decision.trace` 则以不可变
`Mapping` 返回相同内容：

```python
print(decision.explain())
# budget=$0.000912 (alpha=0.25, c_min=$0.000718@deepinfra, c_max=$0.001494@fireworks)
# mix: deepinfra 0.71, fireworks 0.29 -> sampled: deepinfra
# excluded: together (cooldown, 还剩 18s)
```

## 调优（高级配置入口）

算法常量不会出现在主要签名中，因为它们是校准后的默认值，而不是逐部署的选择。
只有在测量后确认有理由修改时，调用方才需要传入 `Tuning` 对象：

```python
from routewise import Router, Tuning

router = Router([...], alpha=0.25, slo_ms=3000, seed=7,
                tuning=Tuning(hedge_target=0.95, window_min=30))
```

大多数用户永远不需要构造 `Tuning`。

`penalty_ms` 是每次健康类失败所记录的合成延迟，无论该失败来自上报还是观测。
默认值 60 秒与生产语义一致；当真实样本逐渐移出窗口时，它可以让正在触发限流或
持续报错的 provider 对 LP 保持低吸引力。连续发生 `cooldown_after` 次健康类失败，
且中间没有成功，会让 provider 进入
`cooldown_sec` 时长的 cooldown：cooldown 期间，该 provider 不具备 route 或充当
对冲备用项的资格；期限届满后恢复资格；任何成功都会重置计数器——这里的成功
指任何进入窗口的 TTFT 样本（来自 `first_token`、`completed` 或一次成功的
observation）。请求类失败既不影响计数器，也不影响 profile。系统没有单独的半开状态：期限届满后，
provider 的窗口中仍保留惩罚样本，因此除非其他选项更差，LP 仍会避开它；它的
第一次成功会开始逐步冲淡这些惩罚。连续失败记录是否也应随窗口过期，仍是一个
待决问题。`exploration_lease_sec` 限制一次冷启动探索持有其 provider 专属租约的
最长时间；它是 monotonic 时钟上的 elapsed-time 期限，并且有意不复用
`penalty_ms`，因为后者是合成延迟值，二者因不同原因而变化。

## 时钟

router 从唯一一个可注入的 monotonic clock 读取时间：`Router(clock=...)`
接受一个零参数、返回浮点秒数的 callable，默认 `time.monotonic`。所有与时间
相关的行为都读取它：profile window、cooldown 到期、exploration lease，以及
observation 的“现在”时间戳。算法中不出现任何墙钟时间，因此系统时钟调整不会
破坏窗口；测试注入 fake clock，而不是 sleep。原始读数只要求有限；router
以 `now = max(previous_now, raw_now)` 推导有效时间，因此即使原始时钟倒退，
有效时间也保持非递减。每个公开操作至多读取一次时钟，并在其全部检查中使用
这唯一一次有效 `now`，因此单次调用绝不会横跨两个时刻。契约表 C 列出了
每个构件读取时钟的时机。

## 错误与校验

构造函数会一次性校验：`alpha` 位于 [0, 1] 范围内；提供 `slo_ms` 时其值为正；
价格有限且非负；provider 名称唯一且非空；至少存在一个 provider。当合格
集合为空（所有 provider 均被排除、处于 cooldown、在严格 cold-start 模式下为
unprofiled，或在 explore 模式下 unprofiled 且探索已在进行中）时，`route()`
会抛出 `NoProviderError`；它绝不会静默路由到
不合格的 provider。非法参数值在传入它的那次调用上抛出 `ValidationError`。
误用结果上报（冲突的终态调用、为已知字段传入不同值、将第二个 attempt 标为
adopted）会抛出 `OutcomeError`。所有库异常都继承自 `RouteWiseError`。参数校验
失败和 outcome 冲突不会提交业务状态；capacity 全部失败的 route 也不会提交
selection、lease、counter、spend 或 RNG 变更。将窗口 bookkeeping 推进到本次
操作捕获的 `now` 不会回滚。逐调用规则见契约表 D。

## 契约表

下面四张表，加上“实现说明”中的 release-gate 表，构成本契约可实现的核心。
其余散文解释意图；散文与表格冲突时，以表格为准。

### 表 A：Attempt 状态机与逻辑归结

状态：`pending` →（`streaming`）→ `completed | failed | cancelled |
declined` 之一。`settle()` 在除 `declined` 外的任何终态下有效且不改变状态。
重复完全相同的调用是 no-op；矛盾的调用抛出 `OutcomeError`。

| 当前态 | 事件 | 新态 | latency profile | failure streak | estimator | 支出 |
| --- | --- | --- | --- | --- | --- | --- |
| pending | `declined()` | declined | — | — | — | — |
| pending | `first_token(ttft_ms)` | streaming | +1 条 TTFT 样本 | 重置 | — | — |
| pending | `completed(...)` | completed | 给了 `ttft_ms` 则 +1 条样本 | 给了 `ttft_ms` 则重置 | 仅 winner，在 `output_tokens` 已知且 > 0 时 | 按表 B |
| pending | `failed(kind="health")` | failed | +1 条 penalty 样本 | +1 | — | 经 `settle` |
| pending | `failed(kind="request")` | failed | — | — | — | 经 `settle` |
| pending | `cancelled()` | cancelled | — | — | — | 经 `settle` |
| streaming | `completed(...)` | completed | —（已记样本） | —（已重置） | 仅 winner，在 `output_tokens` 已知且 > 0 时 | 按表 B |
| streaming | `failed(kind="health")` | failed | —（保留 TTFT，不写 penalty） | +1 | — | 经 `settle` |
| streaming | `failed(kind="request")` | failed | — | — | — | 经 `settle` |
| streaming | `cancelled()` | cancelled | —（保留 TTFT） | — | — | 经 `settle` |
| 除 declined 外的终态 | `settle(...)` | 不变 | — | — | 仅 winner 的首个已知且 > 0 的 `output_tokens` | 按表 B |

Adoption：`adopted=True` 只随 `first_token` 或 `completed` 传入；每个
decision 至多一个被采纳的 attempt；不允许事后补报。从未 offer 过 backup 的
decision 在其 primary completed 时隐式采纳。

逻辑归结（完备）：存在 adoption 时，decision 与被采纳 attempt 的终态一致
（`completed` = 有 winner、`failed`、`cancelled`）。没有 adoption 时，待所有
attempt 进入终态：有 completion → `unresolved`（`decisions_without_adoption`
加一）；否则有失败 → `failed`；否则有取消 → `cancelled`；否则 →
`declined`。hedge slot 在 primary 首 token、adoption 或逻辑终结三者最早时刻
关闭。

### 表 B：Billing 状态与聚合迁移

每个 attempt 有 `billing_state ∈ {unknown, calculated, actual}`；创建时固化
price snapshot（其 provider 的 `price_in`、`price_out`、`price_cached`），
calculated cost 一律按 snapshot 计算。每次上报调用先整体校验，计算唯一目标
状态（已知 `cost_usd` 则为 `actual`，否则已知 `output_tokens` 则为
`calculated`，否则为 `unknown`），再以一次原子 delta 从先前状态调平聚合值；
同时携带 usage 和 `cost_usd` 的调用直接落到 `actual`。

| 迁移 | 触发 | 原子聚合 delta |
| --- | --- | --- |
| unknown → unknown | 进入 `declined` 以外的终态且 billing 仍未知（无任何字段，或只有 `cached_tokens`） | `unsettled_attempts` +1，每个 attempt 恰好一次 |
| unknown → unknown | 之后的 `settle` 只补 `cached_tokens` | 保存该字段；聚合值不变 |
| unknown → calculated | `output_tokens` 变为已知；`cached_tokens` 有账单真值用真值，否则用路由时估计 | 按 snapshot 计价加入 `calculated_spend_usd`；若此前已计数则 `unsettled_attempts` −1 |
| unknown → actual | 显式 `cost_usd` 已知 | 加入 `actual_spend_usd`；若此前已计数则 `unsettled_attempts` −1 |
| calculated → calculated | `settle` 填入原先未知的 usage 字段 | 用重新计算的贡献替换该 attempt 原有的派生贡献 |
| calculated → actual | 显式 `cost_usd` 迟到 | 从 `calculated_spend_usd` 扣除该 attempt 的派生贡献，把 `cost_usd` 加入 `actual_spend_usd` |
| actual → actual | 重复相同 `cost_usd` | no-op；不同值抛出 `OutcomeError` |
| actual → actual | 显式 cost 之后迟到的 `output_tokens` 或 `cached_tokens` | 保存字段；spend delta 为零（显式 cost 保持权威）；winner 的首个已知且为正的 `output_tokens` 仍训练 estimator |

规则：billing 字段只能从 `None` 写入一次；任何地方都不存在 overwrite
（calculated 金额是派生值，从不写入字段，因此迟到的显式成本仍是该字段的首次
写入）。账单上的 `cached_tokens` 大于 `input_tokens` 时按上报值原样保存，但
派生成本使用 `min(cached_tokens, input_tokens)`，与 core 的 cached-token
上限一致。`declined` 的 attempt 永不进入 billing 记账。未知金额永不作为钱数
求和；由 `unsettled_attempts` 计数。per-provider 支出包含 backup attempt；
`hedges.*_spend_usd` 是对同一批 attempt 的交叉切片，绝不能与 provider 支出
相加。

### 表 C：时钟与错误语义

| 构件 | 时间来源 | 语义 |
| --- | --- | --- |
| profile window（`window_min`） | 当前公开操作的唯一 `now` | 超窗样本退出 mean/CDF |
| cooldown（`cooldown_sec`） | 当前公开操作的唯一 `now` | 到期恢复资格；成功重置 streak |
| exploration lease（`exploration_lease_sec`） | 当前公开操作的唯一 `now` | 由 target 的首个 window event 或到期释放，先到先生效 |
| `observe()` 时间戳 | 当前公开操作的唯一 `now` | 一律“现在”；`0.2.0` 无 `at=` |
| `hedge_now(elapsed_ms=)` | 每次调用读取一次，供其全部检查共用 | `elapsed_ms` 由调用方以 primary 派发时刻为基准测量；该操作的 `now` 用于 profile、cooldown 和 eligibility 查询 |

| 路径 | profile | failure streak | stats |
| --- | --- | --- | --- |
| `first_token` / `completed(ttft_ms=...)` / `observe(ttft_ms=...)` | +1 条 TTFT 样本 | 重置 | 窗口分位数变化 |
| `failed(kind="health")`，first token 前 | +1 条 penalty 样本（`penalty_ms`） | +1 | `errors["health"][code]` |
| `failed(kind="health")`，first token 后 | 无（TTFT 已入账） | +1 | `errors["health"][code]` |
| `failed(kind="request")` 任意时刻 | 无 | 无 | `errors["request"][code]` |
| `observe(kind="health", code=...)` | +1 条 penalty 样本 | +1 | `errors["health"][code]` |
| `observe(kind="request", code=...)` | 无 | 无 | `errors["request"][code]` |
| `cancelled()` / `declined()` | 无 | 无 | 相应 hedge 计数 |
| 未知 `code` | 按其 `kind` | 按其 `kind` | 聚合入 `"other"` |

### 表 D：公开签名与校验

provider 身份参数与输出一律为 `str` 名称；`Router` 构造函数是唯一接受
`Provider` 对象的签名。公开 mapping（`weights`、
`trace`、stats 结构）深度不可变。所有 token 计数为非负整数；所有价格、成本
和延迟为有限非负浮点数。除注明者外，违规抛出 `ValidationError`；抛出异常的
参数校验或 outcome 冲突不会提交业务状态；capacity 全部失败时遵循上文的
分阶段事务规则。

| 调用 | 规则 | 异常 |
| --- | --- | --- |
| `Provider(name, price_in, price_out, price_cached=None)` | 名称非空；价格有限非负 | `ValidationError` |
| `Router(providers, *, alpha=0.25, slo_ms=None, seed=None, cold_start="explore", clock=None, tuning=None)` | ≥1 个 provider；名称唯一；`alpha ∈ [0,1]`；给出 `slo_ms` 时为正；`cold_start ∈ {"explore","require_observations"}`；`clock` 为零参 callable、返回有限浮点秒（有效时间被钳制为非递减）；`seed` 为 int 或 None | `ValidationError` |
| `router.route(*, input_tokens, estimated_cached_tokens=0, alpha=None, exclude=())` | `input_tokens ≥ 0`；逐调用 `alpha ∈ [0,1]`；映射形式：缺失名称按 0，未知名称报错，每个值钳制到 `input_tokens`（与 core 的 cached-token 上限一致）；`exclude` 中的名称必须存在；合格集合为空则报错 | `ValidationError`；`NoProviderError` |
| `router.observe(provider, *, ttft_ms=None, kind=None, code=None)` | provider 已知；`ttft_ms` 与 `kind` 恰取其一；`code` 只能伴随 `kind`；`kind ∈ {"health","request"}` | `ValidationError` |
| `attempt.first_token(*, ttft_ms, adopted=None)` | 至多一次，且在终态前；`adopted ∈ {True, None}` | `OutcomeError` |
| `attempt.completed(*, output_tokens=None, ttft_ms=None, cached_tokens=None, cost_usd=None, adopted=None)` | 每个 attempt 一个终态；每个 decision 一个 adopted；不允许事后 adoption；`adopted ∈ {True, None}` | `OutcomeError` |
| `attempt.failed(*, kind, code=None)` | `kind` 必填且合法 | `ValidationError`；`OutcomeError` |
| `attempt.cancelled()` | 终态，且只进入一次 | `OutcomeError` |
| `attempt.declined()` | 只能从 `pending` 进入 | `OutcomeError` |
| `attempt.settle(*, output_tokens=None, cached_tokens=None, cost_usd=None)` | 除 `declined` 外的任何终态；逐字段只写一次；不带 adoption 标志 | `OutcomeError` |
| `decision.hedge_now(*, elapsed_ms)` | `elapsed_ms` 有限且 ≥ 0；返回 `None` 或唯一的 backup `Attempt` | `ValidationError` |
| `Tuning(hedge_target=0.99, penalty_ms=60000.0, window_min=15, cooldown_sec=30.0, cooldown_after=3, hedge_min_samples=5, exploration_lease_sec=60.0)` | `hedge_target ∈ (0,1]`；`penalty_ms > 0`；`window_min > 0`；`cooldown_sec ≥ 0`；`cooldown_after ≥ 1` 整数；`hedge_min_samples ≥ 1` 整数；`exploration_lease_sec > 0` | `ValidationError` |
| `Candidate(name, cost_usd, latency_ms)` | 名称非空；数值有限非负 | `ValidationError` |
| `route_once(candidates, *, alpha, seed=None, rng=None)` | candidates 非空且名称唯一；`alpha ∈ [0,1]`；`seed` 与 `rng` 互斥；`rng` 暴露 `random() -> float`，取值范围 [0,1) | `ValidationError` |

`route_once()` 的警告属于契约：固定 `seed=` 时，相同输入会重放完全相同的
抽样，因此在服务循环里传入常量 seed 会使 mixture 无法在时间上兑现，静默
失去长期的预算混合保证；单次调用的 LP 解与预算本身不受影响。长期使用请
传入可复用的 `rng=random.Random(...)`，或将该函数当作一次性工具。

## API Reference

### Provider

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `name` | `str` | 由你的执行层解析的标识符；在每个 router 内唯一 |
| `price_in` | `float` | 输入价格，单位为美元/百万 token |
| `price_out` | `float` | 输出价格，单位为美元/百万 token |
| `price_cached` | `float \| None` | 缓存前缀 token 的折扣价格；`None` 表示无折扣（缓存 token 按 `price_in` 计费） |

### Router

```python
Router(providers, *, alpha=0.25, slo_ms=None, seed=None,
       cold_start="explore", clock=None, tuning=None)

router.route(*, input_tokens, estimated_cached_tokens=0,
             alpha=None, exclude=()) -> Decision
router.observe(provider, *, ttft_ms=None, kind=None, code=None) -> None
router.stats() -> StatsSnapshot
```

`estimated_cached_tokens` 可以是一个应用于所有 provider 的 `int`，也可以在各
provider 的缓存状态不同时使用映射 `{provider_name: tokens}`；实际通常正是
后者：每个 provider 的前缀缓存会独立预热，而在多轮 agent 工作负载中，缓存
折扣对最便宜 provider 的判定影响可能超过标价差异。映射中缺失的名称按 0
处理；未知名称抛出 `ValidationError`；每个值都会钳制到 `input_tokens`。
`Router` 是线程安全的：可以从任意线程调用 `route()`、结果上报方法、
`observe()` 和 `stats()`。

### Attempt

一次潜在派发的句柄。由 `hedge_now()` 返回。

| 成员 | 类别 | 含义 |
| --- | --- | --- |
| `provider` | 属性 | 如果接受此 attempt，应将其派发到哪里 |
| `first_token(*, ttft_ms, adopted=None)` | 方法 | stream 已开始；记录一个 TTFT 样本；`adopted=True` 表示采纳此 attempt 的响应 |
| `completed(*, output_tokens=None, ttft_ms=None, cached_tokens=None, cost_usd=None, adopted=None)` | 方法 | 已完成；billing 字段可选，可以稍后 settle；winner 的首个已知且为正的 `output_tokens` 训练 length estimator |
| `failed(*, kind, code=None)` | 方法 | `kind="health"`：写入 penalty 并推进 cooldown；`kind="request"`：只计入 stats；若已调用 `first_token`，不再写入第二条 latency 记录 |
| `cancelled()` | 方法 | 已派发后由你中止；不写入 penalty，也不申报 spend |
| `declined()` | 方法 | 从未派发；关闭句柄；只能从 `pending` 进入 |
| `settle(*, output_tokens=None, cached_tokens=None, cost_usd=None)` | 方法 | 在除 `declined` 外的任意 terminal state 之后填入仍未知的 billing 字段，每个字段恰好写入一次 |

### Decision

由 `route()` 返回的逻辑请求。它暴露相同的结果上报方法，这些方法作用于其
primary attempt；此外还暴露 decision context。

| 成员 | 类别 | 含义 |
| --- | --- | --- |
| `provider` | 属性 | 抽样选中的 primary provider；将请求发送到这里 |
| `weights` | 属性 | 底层 mixture（非零项最多两个） |
| `expected_cost_usd` | 属性 | 此 decision 的预测期望成本 |
| `expected_latency_ms` | 属性 | `float \| None`：期望 TTFT；当 mixture 包含 unprofiled provider 时为 `None` |
| `checkpoints_ms` | 属性 | hedge 重新评估的时间点；没有 SLO 时为空 |
| `hedge_now(*, elapsed_ms)` | 方法 | 返回此 decision 唯一的 backup `Attempt`，或返回 `None` |
| `explain()`, `trace` | 方法、属性 | 人类可读和结构化的解释 |

### Tuning

```python
Tuning(*, hedge_target=0.99, penalty_ms=60000.0, window_min=15,
       cooldown_sec=30.0, cooldown_after=3, hedge_min_samples=5,
       exploration_lease_sec=60.0)
```

### StatsSnapshot

“可观测性”一节所述计数器和支出来源字段的不可变快照，遵循表 B 的迁移规则。
计数器的生命周期仍属于开放问题 7；该问题签署确认后，这个运行时类型本身将
成为公开、在顶层导出且稳定的类型。

### 无状态函数

```python
from routewise import Candidate, route_once

result = route_once(
    [Candidate("fireworks", cost_usd=0.0012, latency_ms=240.0),
     Candidate("together",  cost_usd=0.0008, latency_ms=410.0)],
    alpha=0.25, rng=my_rng,
)
result.provider     # 抽样选中的 provider
result.weights      # 抽样所依据的 mixture
result.budget_usd   # LP 求解时使用的请求预算
```

`route_once()` 是一个无状态函数，供已经自行跟踪成本和 latency 的调用方使用。
它既不计算价格，也不学习，只求解 budget LP 并抽样选择一个 provider。它返回
一个不可变的 `RouteOnceResult`，且恰好包含上述三个字段，而不是 `Decision`：
它背后没有 router 状态，因此也没有可上报结果的对象。`Candidate(name,
cost_usd, latency_ms)` 是不可变值类型；由于 candidate 由调用方提供，此处的
eligibility 过滤由调用方负责。确定性来自 `seed=`；长期使用的调用方应改传
可复用的 `rng=`（参见表 D 的警告）。`Router.route()` 在内部等价于此函数
加上前文所述的 estimator 和状态。研究用户从这里开始，但请注意：API-only
preview 覆盖的是论文的 on-demand 路由，不包含 quota 与 concurrency 结果。

## 托管执行：Client（计划用于后续版本）

`Router` 负责决策，由你负责执行。对于希望 RouteWise 也负责执行的调用方，
后续版本计划提供 `Client`：它通过兼容 OpenAI 的接口，为每个 model 配置一个
router。该设计沿用 revision 1，但目前暂缓，以便 initial release 保持零依赖；
同时，正确地让对冲 stream 竞速并取消落败者确有实质工程难度，不应阻碍
decision core 发布。待 Juncheng 签署确认。

```python
client = Client(
    {"deepseek-v4": [Provider("fireworks", ..., base_url=..., api_key=...),
                     Provider("together",  ..., base_url=..., api_key=...)]},
    alpha=0.25, slo_ms=3000,
)
response = await client.chat.completions.create(model="deepseek-v4", messages=[...])
print(response.routewise.provider, response.routewise.cost_usd)
```

`create()` 与 OpenAI SDK 的函数签名一致，包括 `stream=True`。token 计数、结果
上报、adoption、hedge 竞速和 loser 取消均在 client 内部完成。`Provider`
上的 `base_url` 和 `api_key` 仅供 `Client` 使用。对于想使用多个 model 的
`Router` 用户，相同模式只需自己写一行代码：
`routers = {m: Router(pool[m], alpha=0.25) for m in pool}`。

## 设计原则

决策与执行保持分离。core 不打开任何连接，因此可以保持零依赖，并与任何 HTTP
client 或 async framework 相互独立。成本控制只使用一个无量纲旋钮，因此接口
中不会出现特定于某一部署的美元阈值。decision 本身就是句柄，因此上报结果时
无需再次提供请求身份。结果采用类型化设计，是因为 attempt 的不同结束
方式含义不同：health failure 归因于 provider，request failure 归因于请求，而
cancellation 与 declination 不归因于任何一方；如果把它们压缩成一个 error
flag，一次误调用就可能写入 60 秒的 penalty。影响由 `kind` 决定、标签由
`code` 承载，因此判断权留在调用方手中，metrics 基数也保持有界。lifecycle 与
settlement 分离，是因为 timing truth 和 billing truth 会在不同时刻到达；
settlement 采用 write-once，是因为估计值从不进入实际 spend，所以永远不存在
需要修正的实际值。adoption 与结果分离，是因为“采用某个响应”和“该响应成功”
是两个不同的事实。router 只承诺统计它确实能够观察到的内容：selection、
offer 和你的上报，绝不声称观察到 dispatch。library 引入的每一项额外成本或
风险（hedge offer、exploration decision、未声明 adoption 的请求）都会呈现在
`stats()` 中，而不会隐没在账单里。router 的线程安全由一把粗粒度锁保障，不
持有任何未终结 handle 的强引用，并只从一个可注入的 monotonic clock 读取
时间。

## 范围与路线图

订阅式 provider（预付费请求配额、预留并发 slot）是 RouteWise 算法的另一半，
也是计划中的 v2 扩展；论文的主要结果依赖它们，这正是 `0.2.0` 定名为
API-only preview 而非论文 artifact 的原因。相关定价数学已经存在于
`routewise.core` 中（quota shadow
price 及其 L/U scarcity calibration）。v2 无法回避的是 capacity lifecycle：
quota slot 会被消耗，concurrency slot 则会被占用并释放；因此接口将新增 v1 中
没有对应项的 reservation 交互（在返回可 dispatch 的 attempt 前 reserve，
dispatch 开始时 commit，并在结果产生时 close 或 release）。v1 接口的设计可以
通过增量方式承接这种增长（新增 `Provider`
构造方式和新的可选交互，而非修改既有签名），但本 revision 撤回 revision 1
中“接口形状不会改变”的承诺。`routewise.core` 的 `ProviderView` 仍然是高级集成
的衔接点；hybridInference 已经基于这些原语，在生产环境中运行 quota 和
concurrency capacity。

### Capacity 扩展接缝

Capacity 改变的是 routing decision 周围的 transaction，而不是 LP 本身。
capacity-aware facade 会构造绑定请求的 candidate snapshot，让纯 core 求解并
抽样，然后原子地尝试 reserve 被抽中的 provider。reservation 失败表示 snapshot
已经过时或 capacity 发生争用：facade 会在本次 routing transaction 内排除该
candidate，重新计算 eligible-set cost bounds 和 budget，并在有限重试次数内再次
求解。它不会上报 provider error、写入 latency penalty 或推进 cooldown。返回的
decision 中，weights、budget 和 trace 描述的是 sampled provider 最终成功
reserve 的那次求解；trace 还会记录 capacity exclusion 和 replan 次数。

Primary 与 hedge attempt 各自拥有独立 reservation。在未来支持 capacity 的
release 中，`Attempt.started()`（或 execution adapter 中等价的 hook）会在网络
dispatch 前原子执行 commit-if-still-owned。如果 reservation 已过期或被 fencing
排除，commit 就会失败，任何 I/O 都不得开始，execution layer 必须启动一次新的
routing transaction；managed adapter 会自动处理。被取代的 attempt 以
`declined` 关闭——它从未被派发——既不贡献 latency sample，也不计作
provider-health failure；手工 API 的精确形态和异常类型则留到未来 capacity
release 决定。终态 `completed`、`failed` 或 `cancelled` 会关闭已 commit 的
reservation，`declined()` 则释放未 commit 的 reservation。API provider 使用
no-op controller。quota
reservation 在 commit 时消耗额度，close 时不会返还；concurrency reservation
会一直持有 slot，直至 close 或 lease expiry。quota reservation 还要绑定
quota-window epoch，从而明确 reset race 应归属哪个窗口。分布式 controller 还需要
idempotency key、lease 或续租，以及 fencing，以防 stale process 在过期后错误释放
较新的 reservation。

hedge reserve failure 会在同一次 `hedge_now()` 调用内处理：facade 临时排除发生
争用的 backup，在剩余 eligible candidate 上重新执行 backup selection，并以有界
次数尝试 reservation。只有某次 reservation 成功并返回 `Attempt` 后，one-backup
slot 才会被消耗。如果全部失败，`hedge_now()` 返回 `None`，以一份新的不可变
`decision.trace` snapshot 发布 capacity exclusion，且不消耗 slot。

initial `0.2.0` public surface 仍然仅支持 API，不暴露
`CapacityController`、`Reservation` 或 `Attempt.started()`。但它会通过
`_NoopCapacityController` 走同一套私有 orchestration，使后续 capacity release
可以增加实现和 dispatch 交互，而无需重写 `Router`。当 capacity-backed provider
成为 public API 后，这类 attempt 必须在 I/O 前调用 `Attempt.started()`，managed
execution adapter 会自动调用；对 API provider 而言它仍是 no-op，因此既有
quickstart 保持不变。

推迟到后续版本的项目包括：`Client` 执行层、LiteLLM strategy plugin、历史
observation 导入（`observe(at=)` 的后继方案），以及跨进程 state（一个
`export_state` / `state=` 对）。initial release 在进程内
学习；各独立 replica 分别从自己的 traffic 中学习，而通过 `observe()` 重放
peer 的测量值，是当前的临时共享机制。端到端 latency objective 属于路线图内容
（v1 优化 TTFT）。model selection 永久不在范围内，因为 RouteWise 所做的是将
一个固定 model 路由到不同 provider。

## 开放问题

以下问题会阻止契约冻结。

1. `Client` 是随 initial release（`0.2.0`）发布，还是推迟到后续版本？当前
   草案选择推迟。如果移入 `0.2.0`，契约还必须把 `base_url` 和
   `api_key` 加入 `Provider`，在顶层导出 `Client`，并在安装接口
   中定义 HTTP 可选依赖。（Juncheng）
2. `failed()`/`observe()` 的推荐 `code` 清单（行为已由 `kind` 定死；该清单
   只用于约束 metrics 标签的基数）。
3. Cooldown 细节：failure streak 是否应随 profile window 一同过期？自然形成的
   half-open 机制（penalty 会让刚恢复的 provider 缺乏吸引力）是否足够，还是
   需要显式的 trial state？
4. `cold_start` 应是 Router 参数（如草案所示）还是 `Tuning` 字段？对于可能在
   首次部署时就承接生产流量的 library，`"explore"` 是否是正确的默认值？
5. 探索节奏：草案规定每当 window 为空时重新 arm；每个 provider
   有一个 exploration lease，由该 provider 的首个 window event 释放，或在
   `exploration_lease_sec`（默认值 60）后释放。对于反复 flap 的 provider，
   re-exploration 是否需要频率上限？60 秒是否是合适的 lease 时限？
6. `decision.primary` 和 `decision.backups` 是公开属性，还是由 outcome method
   静默代理（如 draft 所示）？
7. 上述 `StatsSnapshot` schema 是 freeze candidate；请确认字段列表和
   counter 的生命周期（整个 router 生命周期还是 rolling window）。支出来源
   阶梯与 calculated→actual 的迁移已由表 B 定死。
8. `hedge_min_samples=5` 借用了 research harness 的 warmup validation threshold；
   per-provider hedging gate 本身是新的 library policy，在生产中没有对应先例。
   请确认该机制及其数值。
9. `route()` 是否应接受 `estimated_output_tokens=`，以便已知 `max_tokens` 的
   调用方覆盖 length estimator；后者的 fallback 默认值为 500 token。
10. wheel 是否保留研究子包（`routewise.sim`、`routewise.offline`、
    `routewise.metrics`，以及共享研究契约 `routewise.capacity` 与
    `routewise.schemas`），还是只交付 facade 与 core（表 E 起草了窄
    allowlist）？排除它们能守住“只装库本身”的承诺；保留它们会增大体积，
    但能让研究用户免于 source checkout。保留还会击穿“`0.2.0` 零 extras”
    的立场：研究子包会引入科学计算依赖，因此需要一个 `[sim]` 式的 extra，
    或重新定义依赖策略。

## 实现说明

本节面向 RouteWise 开发者；library 用户读到上面即可。

RouteWise 中会出现两种不同的成本范围，二者绝不能共用一个名称。**request cost
bounds** `c_min`/`c_max` 按请求、在合格 provider 范围内计算；每种部署都有
这两个值，包括仅使用 API provider 的部署。**L/U scarcity calibration** 为
quota 与 concurrency 的稀缺性定价；仅使用 API 的 provider fleet 永远不会
构造它。core 的 `RouteWiseRouter.cost_envelope` 参数指的是 L/U 对，而不是
request bounds；facade 在 v1 中绝不暴露它。

Capacity admission 与 scarcity pricing 是彼此分离的内部职责。
`_ScarcityCalibrator` 维护 workload-level L/U 输入与 effective-cost state；
capacity controller 只暴露 admission state 和原子 reservation。
`ProviderView` 继续保持为纯粹、绑定请求的 snapshot，不加入任何会产生副作用的
reserve 方法。初始私有接缝有意保持精简：

```python
class _CapacityController(Protocol):
    def snapshot(
        self, *, resource_key: str, now: float
    ) -> CapacitySnapshot: ...
    def try_reserve(
        self, *, resource_key: str, attempt_id: str,
        snapshot: CapacitySnapshot
    ) -> _Reservation | None: ...

class _Reservation(Protocol):
    def commit(self) -> bool: ...
    def release(self) -> None: ...
```

`resource_key` 标识原子 capacity domain；它可以对应一个 provider，也可以对应由
多个 provider endpoint 共享的 pool。snapshot 的作用域限定在该 key 下，attempt ID
则全局唯一，因此 replan 不会含糊地寻址到为另一个 candidate 创建的 reservation。

reservation 状态机如下：

```text
RESERVED --commit(success)--> COMMITTED --release/finish/expire--> CLOSED
    |
    +--release/expire/commit-lost-------------------------------> CLOSED
```

所有状态转换都必须幂等，并以 attempt identity 为键。只有调用方仍拥有有效
reservation 时，`commit()` 才会原子成功；过期或失去 fencing ownership 后返回
`False`。commit 前，`release()` 会取消并归还 reservation；commit 后，其含义由
controller 决定：concurrency 会归还 slot，而 quota 只关闭持久 reservation record，
绝不会返还已消耗的额度。为了从进程崩溃中恢复，已 commit 的 concurrency lease
也可以通过 expiry 进入 `CLOSED`。分布式实现需要 lease generation 或 fencing
token；可选的 renewal 能力可以为长请求延长该 lease。

facade 负责包围纯 core 的有副作用 orchestration：

```text
构造 candidate snapshot
        -> 求解 LP 并抽样
        -> try_reserve（抽中的 provider）
        -> 失败：exclude，重算 bounds/budget，并重新求解
        -> 成功：挂接 reservation，返回 Decision/Attempt
        -> dispatch 开始：commit-if-owned
        -> commit ownership 丢失：以 `declined` 关闭（不影响 health/latency），
           不执行 I/O，并启动新的 routing transaction
        -> terminal outcome：release/finish
```

primary 重试循环有明确上限，并且一次只 reserve 一个 sampled candidate，因此
求解时不会持有跨 provider 的锁。`hedge_now()` 使用上文所述的独立有界
exclude/reselect 循环，并拥有独立 backup reservation。no-op API controller
总会在第一次尝试时成功。

该接口与 `routewise.core` 的对应关系如下：`route_once()` 封装
`solve_budget_lp()`，加入 alpha-to-budget 转换和基于 seed 的 weight sampling；
`Router` 再加入 rolling latency profile（基于 `RollingLatencyProfile` 的
`LatencyBeliefs`）、bucket-mean output-length estimator、cooldown、cold-start
处理和 attempt bookkeeping；`hedge_now()` 封装
`hedge_checkpoints_for_slo()`、`combined_success_probability()` 和
`select_probability_backup()`。

本 contract 与当前 `routewise.core` 之间的差距如下，除最后一条外均位于
facade 层：

1. Core 的默认 sampler 是 `argmax_weight_sampler`。facade 使用其带 seed 的 RNG
   按 LP weight 抽样；core 的默认行为保持不变，因为 research harness 会注入
   自己的 RNG discipline。
2. Cooldown、`kind`/`code` 错误模型、cold-start mode 和 exploration mixture
   在 core 中并不存在；在那里，可用性是 adapter 的职责。这些功能由 facade
   负责。
3. bucket-mean estimator 位于
   `experiments/offline_stage/value_estimators/bucket_mean.py`，且会导入
   实验类型。facade 需要在 package 内实现一个零依赖的等价版本。
4. `RouteWiseRouter` 在没有 L/U 对时会拒绝路由，但仅使用 API 的 provider fleet
   永远不会用到它（`effective_cost("api")` 会原样返回 request cost）。只有存在
   非 API tier 时，core 才应要求提供该参数。
5. `LatencyBeliefs` 不是线程安全的；profile query 会修改 window bookkeeping。
   facade 通过自己的锁串行化所有访问。
6. Attempt bookkeeping（adoption、带表 B 迁移规则的 write-once settlement、
   每个 attempt 的 price snapshot、one-latency-entry 规则、hedge slot 及其
   在 primary first token 到达时关闭的行为，以及带 generation token 的
   exploration lease——这样 stale attempt 的迟到事件就无法释放更新的
   lease）在 core 中没有对应实现；它是由 handle 自身持有的新 facade state，
   因此 router 不保留任何强引用。
7. 私有 capacity seam、no-op controller、reservation 状态机和有界
   reserve/replan 循环都是新的 facade orchestration。它们不会进入
   `routewise.core`，这些 capacity Protocol 也不属于 `0.2.0` compatibility
   surface。
8. 当前实现分支中的 `combined_success_probability` 已应用规定的
   survival-zero fallback（只按 backup 的成功概率计算），并有 core
   回归测试保护该边界。

### 表 E：Wheel Allowlist 与发布门

wheel allowlist（freeze candidate，以 installed-wheel import test 为最终
仲裁）：`routewise` 顶层（facade 模块、`py.typed`）、`routewise.core`、
`routewise.const`，以及零依赖 estimator 模块。排除：`experiments/`、
`plots/`、`scripts/`、全部数据文件，以及研究子包 `routewise.sim`、
`routewise.offline`、`routewise.metrics`。开放问题 10 尚未关闭，因此当前
preview build 暂时保留两个仅依赖标准库的共享契约 `routewise.capacity` 与
`routewise.schemas`。
`[project.scripts]` 的 CLI entry point 不进入 `0.2.0`：当前
`routewise_cli.main` 在模块顶层导入 `experiments`，因此在出现 library-only
CLI 之前，console command 从 wheel 中移除。`0.2.0` 不定义任何 pip extra；
`[client]` 与 `[litellm]` 随其功能一同到来。

| 发布门 | 要求 | 当前状态 |
| --- | --- | --- |
| wheel 内容 | 按上述 allowlist；不含 experiments、CLI 或数据 | 本地通过：25 个条目，精确 member 检查通过 |
| console script | `0.2.0` 不携带 | 通过：entry point 已移除，source CLI 仅供仓库使用 |
| 类型标记 | wheel 内含 `py.typed` | 通过 |
| 顶层导出 | 恰好导出 `Provider`、`Router`、`Decision`、`Attempt`、`Tuning`、`Candidate`、`route_once`、`RouteOnceResult`、`StatsSnapshot`、`RouteWiseError`、`ValidationError`、`NoProviderError`、`OutcomeError` | 通过 |
| 安装测试 | 每次 release 在干净环境执行 install + import + `route_once` smoke，并纳入 CI | Python 3.10–3.14 本地通过，已加入 CI workflow |
| 测试套件 | clean checkout 上快速测试全绿 | 本地通过：640 passed；缺少可选 BurstGPT 数据时 12 项明确 skipped；3 项 slow tests deselected |
| CI | 3.10–3.14 矩阵（pyproject 声明 `>=3.10` 且无上限，3.14 已是当前 feature series）、lint、wheel build | workflow 已加入，拆分 dependency-free 与 research-compatibility jobs，等待首次远端运行 |
| 元数据 | `version = 0.2.0`；库定位 description；README library-first；`[project.urls]`、classifiers、SPDX license；arXiv 引用 | 除 arXiv 引用外均已完成 |
| PyPI 名称 | 以一次真实上传确认 `routewise` 注册成功（项目页当前 404） | 未确认 |
| 工作区 | 从 `origin/main` 建干净 worktree 实现，不用分叉的本地 `main` | 通过：独立 worktree 上的 `codex/api-provider-library-v1` |

以下行为属于契约，而非偶然的实现细节：primary selection 按 LP weight
抽样（使用 argmax 会破坏 budget guarantee）；latency 相同时选择更便宜的
provider（`cost_tiebroken_objective()`，在内部应用，因此调用方永远看不到）；
一个 router 绑定一个 model，因此 `route` 和 `Provider` 中不需要 model 参数；
cancelled 或 declined 的 attempt 永远不会写入 penalty；每个 attempt 最多记入
一条 latency 记录；billing 字段采用 write-once，且估计值从不计入实际 spend；
每次上报调用恰好施加一次原子聚合 delta；output-length estimator 只使用
winner（adopted 且 completed）训练；存在多个 attempt 时，adoption 只能显式
声明；hedge slot 在 primary first token、adoption 或终结时关闭；每个
decision 的预测期望成本都遵守 request budget，无论是否处于 exploration；当
`q > 0` 时，exploration-mixture decision 一律打标，而 `q = 0` 的请求按普通
LP 路由且不打标；当 mixture 包含 unprofiled provider 时，
`expected_latency_ms` 为 `None` 而非 sentinel（绝不使用 core 的 `1e9`
avoidance value）；survival-zero fallback 只按 backup 的成功概率对冲；
request cost bounds 根据当前请求的 eligible set 计算。L/U scarcity
calibration、quota state 和 concurrency state 均不会出现在此接口中。仅使用
API provider 时，一个请求的 effective cost 等于其 estimated metered cost，
因此 quota shadow price 所需的 calibration（参见
[CORE_API.md](CORE_API.md)）永远不会被构造。capacity reservation failure
属于 feasibility event，而不是 provider-health failure：它只会触发不带
latency penalty 或 cooldown 的有界 replan，最终 decision snapshot 则反映
成功的那次求解。
