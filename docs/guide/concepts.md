# Core concepts

## The routing loop

RouteWise sits between your application and your providers. It never touches
the network: it returns a name, and you do the rest.

1. Describe each provider and its prices.
2. Ask for a decision.
3. Dispatch the request yourself.
4. Report what happened.

Step 4 is what makes step 2 improve. Without outcome reports the router has
nothing to learn from.

## Provider

A provider is one endpoint you can send a request to, described by what it
charges. Prices are per million tokens because that is the unit providers
publish.

```python
rw.Provider("cheap", price_in=0.15, price_out=0.60)
```

`price_cached` is optional; omit it and cached input is billed at `price_in`.

Providers are immutable, and the name you give one is the name RouteWise hands
back to you. See the [reference](../reference/api.md#provider) for the
validation rules.

## Router

The router holds your providers and the policy that chooses among them. Three
constructor arguments carry most of the behaviour you will care about:

- `alpha` sets the [cost budget](cost-budget.md).
- `slo_ms` sets a latency objective and enables [hedging](hedging.md)
  checkpoints.
- `seed` makes sampling reproducible.

```python
router = rw.Router(providers, alpha=0.25, slo_ms=1_500.0)
```

`cold_start` is described below. `clock` and `tuning` are for callers who need
to control time or the policy constants; both are in the
[reference](../reference/api.md#router).

## Decision

`Router.route()` returns a `Decision` naming the provider to use:

```python
decision = router.route(input_tokens=800)
```

If your application already predicts completion length, pass that point
estimate. Omit it to use the internal online estimate.

```python
decision = router.route(
    input_tokens=800,
    estimated_output_tokens=predict_output_tokens(prompt),
)
```

The estimate affects route and hedge cost calculations only. It is not actual
usage.

## Outcome feedback

On completion, report the adopted attempt's actual `output_tokens`, or an
explicit `cost_usd`, for billing:

```python
decision.completed(ttft_ms=420.0, output_tokens=180)
```

Positive actual output tokens also update the internal output-length estimator.
Only an adopted, completed attempt with positive output tokens contributes to
that learning.

## Cold start

The default `cold_start="explore"` keeps unprofiled providers eligible and
leases a selected exploration target. Strict `cold_start="require_observations"`
excludes unprofiled providers, so seed them first:

```python
router.observe("provider-a", ttft_ms=240.0)
router.observe("provider-b", ttft_ms=310.0)
```

Consecutive health failures trigger a cooldown after `Tuning.cooldown_after`
occurrences. A single TTFT success clears it.

## Statistics

`router.stats()` returns an immutable `StatsSnapshot` covering per-provider
selections, TTFT percentiles, error counts split by health and request,
cooldown state, and spend.

!!! warning "Do not double-count hedge spend"

    Hedge spend is a cross-section of provider spend. Adding
    `hedges.actual_spend_usd` to the provider totals counts it twice.

## Process scope

Observations, cooldowns, leases, estimates, random state, and counters live in
the current Python process. Nothing is persisted or shared across processes.
