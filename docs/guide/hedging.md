# Hedging

Hedging sends a second attempt when the first one looks likely to miss your
latency objective. It trades cost for tail latency, so it fires selectively.

## Enabling checkpoints

Hedging requires a latency objective on the router:

```python
router = rw.Router(providers, alpha=0.25, slo_ms=1_500.0)
```

`Decision.checkpoints_ms` then carries the elapsed times at which your
application should ask whether to hedge.

## Asking for a backup

```python
backup = decision.hedge_now(elapsed_ms=decision.checkpoints_ms[-1])
if backup is not None:
    dispatch_backup(backup.provider)
    decision.first_token(ttft_ms=420.0, adopted=True)  # Primary wins.
    decision.completed(output_tokens=180)
    cancel_backup_request()
    backup.cancelled()
    backup.settle(cost_usd=0.00011)
```

`hedge_now` returns `None` when no useful backup is available. Otherwise it
returns an `Attempt` and consumes the single backup slot.

!!! important "Returning the handle does not dispatch it"

    If you will not send the backup, call `backup.declined()`. If you do send
    it, report and settle it independently.

Hedging needs enough current samples for the primary, controlled by
`Tuning.hedge_min_samples`, plus an eligible backup.

## When the backup wins

Mark the backup adopted and terminate the primary. A hedge win requires the
backup to be both adopted and completed.

## Timing

`hedge_now` uses the elapsed milliseconds you pass in, not the router clock.
Everything else that is time-aware reads the router clock, which is monotonic by
default.

## Tuning

Two `Tuning` fields change how hedging behaves:

- `hedge_target` is the combined success probability hedging aims for. Lower it
  and RouteWise hedges less eagerly.
- `hedge_min_samples` is how many current primary samples must exist before
  hedging is evaluated at all. Raise it if early, noisy measurements are
  triggering backups you do not want.

```python
router = rw.Router(
    providers,
    slo_ms=1_500.0,
    tuning=rw.Tuning(hedge_min_samples=10),
)
```

The remaining fields govern cooldowns, the observation window, and exploration
leases. Their defaults and validation rules are in the
[reference](../reference/api.md#tuning).
