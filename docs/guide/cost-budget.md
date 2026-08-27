# Cost budget

`alpha` is the single knob that trades spend against latency.

## Estimated cost

For each eligible provider:

```text
(non-cached input * price_in + cached input * cached price
 + predicted output * price_out) / 1,000,000
```

The predicted output term uses your `estimated_output_tokens` when you supply
one, and RouteWise's internal online estimate otherwise.

## The budget

With eligible cost extremes `C_min` and `C_max`, the budget is:

```text
budget = C_min + alpha * (C_max - C_min)
```

RouteWise then samples a provider mixture whose expected primary cost fits this
budget while preferring lower learned latency.

| `alpha` | Behaviour |
| ---: | --- |
| `0.0` | Minimum-cost budget |
| `0.25` | Default. Mostly cost-driven, some latency headroom |
| `1.0` | Full cost range available to latency optimization |

## What the budget is not

!!! warning "Expected-mixture budget, not a per-request cap"

    The budget constrains the expected cost of the primary attempt across the
    sampled mixture. It is not a hard ceiling on any single request, and a
    dispatched [hedge](hedging.md) is an additional attempt that may add spend.

## Per-request override

`alpha` is set on the router and can be overridden for one call:

```python
decision = router.route(input_tokens=800, alpha=0.0)
```

## Choosing a value

Start at `alpha=0.0`, measure your tail latency, then raise it until the tail
meets your objective. Raising `alpha` widens the set of providers the router
may pay for; it never reduces spend.
