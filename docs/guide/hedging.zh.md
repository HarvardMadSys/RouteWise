# 对冲

当第一次尝试看起来很可能错过延迟目标时，对冲会再发一次请求。它用成本换尾延迟，
因此只在必要时触发。

## 启用检查点

对冲需要在 router 上设定延迟目标：

```python
router = rw.Router(providers, alpha=0.25, slo_ms=1_500.0)
```

此后 `Decision.checkpoints_ms` 会给出应用应当询问是否对冲的各个耗时点。

## 请求备份尝试

```python
backup = decision.hedge_now(elapsed_ms=decision.checkpoints_ms[-1])
if backup is not None:
    dispatch_backup(backup.provider)
    decision.first_token(ttft_ms=420.0, adopted=True)  # 主尝试胜出。
    decision.completed(output_tokens=180)
    cancel_backup_request()
    backup.cancelled()
    backup.settle(cost_usd=0.00011)
```

没有有用的备份时 `hedge_now` 返回 `None`。否则返回一个 `Attempt`，并占用那唯一的
备份槽位。

!!! important "拿到句柄不等于已经派发"

    如果不打算发出这个备份，调用 `backup.declined()`。如果发出了，就独立回报并结算它。

对冲还要求主尝试有足够的当前样本（由 `Tuning.hedge_min_samples` 控制），以及一个
可选的备份供应商。

## 备份胜出时

把备份标记为采纳，并终止主尝试。对冲胜出要求备份既被采纳又已完成。

## 时间

`hedge_now` 使用你传入的耗时毫秒数，而不是 router 时钟。其余与时间相关的操作读取
router 时钟，默认是单调时钟。

## Tuning

`Tuning` 里有两个字段直接改变对冲行为：

- `hedge_target` 是对冲要达到的组合成功概率。调低它，RouteWise 就不那么急于对冲。
- `hedge_min_samples` 是评估对冲前主尝试必须具备的当前样本数。如果早期的噪声测量
  触发了你不想要的备份，就调高它。

```python
router = rw.Router(
    providers,
    slo_ms=1_500.0,
    tuning=rw.Tuning(hedge_min_samples=10),
)
```

其余字段管的是冷却、观测窗口和探索租约。它们的默认值和校验规则见
[API 参考](../reference/api.md#tuning)。
