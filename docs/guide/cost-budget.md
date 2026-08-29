# Cost budget

`alpha` is the single knob that trades spend against latency. This page explains
how the knob works; the [reference](../reference/api.md) is the authoritative
definition of the formulas below.

## Estimated cost

For each eligible provider:

```text
(non-cached input * price_in + cached input * cached price
 + predicted output * price_out) / 1,000,000
```

The predicted output term uses your `estimated_output_tokens` when you supply
one, and RouteWise's internal online estimate otherwise.

## Prefix cache {#prefix-cache}

Cached input is the part of the prompt a provider can serve from its prefix
cache, and it enters the estimate at that provider's cached rate:

- `Provider(..., price_cached=0.30)` sets the cached rate. Omit it and cached
  input is billed at `price_in`.
- `route(estimated_cached_tokens=...)` says how many prompt tokens you expect
  to hit cache — one integer for every provider, or a mapping by provider name
  when hit rates differ. Values above `input_tokens` are clamped.
- On completion, report the actual `cached_tokens` so billing reflects what
  happened rather than the estimate.

```python
decision = router.route(
    input_tokens=8_000,
    estimated_cached_tokens={"fast": 7_500, "cheap": 0},
)
```

Cache expectations move the cost side of the decision only; the latency side
comes from the outcomes you report.

## Learned cache-locality {#learned-cache-locality}

When `affinity_key` is supplied, RouteWise can learn destination-local cache
evidence from actual completion observations. On `decision.completed(...)` with
positive `cached_tokens`, RouteWise records evidence associating the affinity
identity with the selected provider. Subsequent requests with the same
`affinity_key` incorporate this learned evidence into routing.

Key properties:

- **Evidence is probabilistic**: Observations decay over time and with
  negative observations (misses). It is not authoritative cache state.
- **Caller estimates take precedence**: Explicit `estimated_cached_tokens`
  always wins over learned evidence for the providers it covers.
- **Provider is the finest granularity**: RouteWise preserves locality at the
  `provider.name` level. If a provider hides multiple replicas behind an
  internal load balancer, RouteWise cannot preserve replica-local state unless
  replicas are individually addressable.
- **Optional**: Cache-locality learning is disabled when `affinity_key` is not
  supplied. Existing callers are unaffected.

The learned value becomes `Decision._estimated_cached_tokens`, which affects
both routing cost and calculated billing fallback. Report actual
`cached_tokens` whenever possible to ensure accurate billing.

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
