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

```python
rw.Provider(name, price_in, price_out, price_cached=None)
```

An immutable provider definition. Prices are finite, non-negative numbers in
USD per million tokens. Names must be non-empty and unique within one router.
When `price_cached` is absent, cached input is billed at `price_in`.

## Router

```python
rw.Router(
    providers,
    *,
    alpha=0.25,
    slo_ms=None,
    seed=None,
    cold_start="explore",
    clock=None,
    tuning=None,
)
```

`alpha` sets the [cost budget](cost-budget.md). `slo_ms` enables
[hedging](hedging.md) checkpoints. `seed` makes sampling reproducible.

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
