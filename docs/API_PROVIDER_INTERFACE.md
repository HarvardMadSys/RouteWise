# RouteWise Library Interface

> Status: design proposal for the v1 public library, scoped to API providers
> only. It supersedes the earlier sketch in this file. The `Router`, `Decision`,
> and `Client` surfaces below are not implemented yet; the math primitives they
> build on exist today and are documented in [CORE_API.md](CORE_API.md).

## What RouteWise Is

RouteWise is a Python library for applications that call the same model through
multiple API providers. Open-weight models such as DeepSeek-V4 or GLM-5.1 are
sold by many providers at different prices, with latency that varies across
providers and drifts over time. For each request, RouteWise decides which
provider to use: the lowest expected latency whose cost fits a budget,
controlled by a single knob. When a response risks missing its deadline,
RouteWise can dispatch a late backup request to a second provider.

RouteWise is not a general LLM client (it ships no per-provider SDKs), not a
model selector (the model is fixed; only the provider changes, so response
quality is never traded for cost), and not a hosted service (your API keys stay
in your process).

## Installation

```bash
pip install routewise              # decision core, no third-party dependencies
pip install "routewise[client]"    # adds the bundled execution client (httpx)
pip install "routewise[litellm]"   # adds the LiteLLM routing-strategy plugin
```

The bracket suffix is a pip extra, an optional dependency group. The base
package contains only decision logic and imports nothing outside the standard
library. Install the `client` extra when RouteWise should execute requests for
you, or the `litellm` extra to plug the algorithm into an existing LiteLLM
router.

## The Whole Interface

One router serves one model. You ask it where to send a request, you send the
request with your own code, and you report the outcome back so the next decision
is better informed.

```python
from routewise import Provider, Router

router = Router(
    [Provider("fireworks", price_in=0.27, price_out=1.10),
     Provider("together",  price_in=0.18, price_out=0.88)],
    alpha=0.25,                          # the one knob: 0 = cheapest, 1 = fastest
)

decision = router.route(input_tokens=1800)          # ask: which provider?
response = send(decision.provider, request)          # your execution layer
decision.report(ttft_ms=response.ttft_ms,            # tell it what happened
                output_tokens=response.output_tokens)
```

That is the main surface for cost-latency routing: three nouns (`Provider`,
`Router`, `Decision`), two verbs (`route`, `report`), and one knob (`alpha`).
When tail-latency hedging is enabled with `slo_ms`, the only additional verb is
`hedge_now`. Prices are USD per million tokens. `Provider.name` is an opaque
label your execution layer resolves; the `Router` itself makes no network calls.

Everything else a caller might want from a decision is a read-only attribute, so
it adds nothing to learn:

```python
decision.provider            # the sampled provider to use
decision.weights             # the underlying mixture, e.g. {"together": 0.7, "fireworks": 0.3}
decision.expected_cost_usd   # expected cost of this decision
decision.expected_latency_ms # expected TTFT of this decision
decision.explain()           # one human-readable line explaining the choice
```

### Reporting Outcomes

`route()` returns a `Decision`, which is a handle: it already holds the request
identity (provider, input length, cached length), so `report()` carries only new
information. On success, report latency and output length. On failure, report
the error instead:

```python
decision.report(ttft_ms=312.0, output_tokens=540)   # success
decision.report(error="rate_limited")                # 429, timeout, 5xx, ...
```

A reported error enters the provider's latency profile as a penalty sample and,
after repeated failures, drops the provider from the candidate set for a short
cooldown. You never compute these effects; you report the fact and the router
reacts.

## Tail-Latency Hedging

The router optimizes the body of the latency distribution. Hedging protects the
tail. Give the router an SLO (service-level objective, the time-to-first-token
deadline), and each decision carries checkpoint times at which a backup may be
worth sending:

```python
router = Router([...], alpha=0.25, slo_ms=3000)

decision = router.route(input_tokens=1800)
# The primary is in flight. At each time in decision.checkpoints_ms where no
# first token has arrived yet:
backup = decision.hedge_now(elapsed_ms=1500)         # None, or another handle
if backup is not None:
    bresp = send(backup.provider, request)           # race primary and backup
    backup.report(ttft_ms=bresp.ttft_ms)             # each handle reports itself
```

`hedge_now()` returns another `Decision`, symmetric with the primary: every
dispatch produces a handle, and every handle reports its own outcome. It returns
`None` when no backup is worth sending yet. Internally the router withholds the
backup until the latest checkpoint at which the combined primary-plus-backup
chance of meeting the SLO still clears the target, so backups stay rare and each
one has a quantified reason. Adding `slo_ms` is the only change needed; the one
new method is `hedge_now`.

## Managed Execution: Client

`Router` decides; you execute. If you would rather RouteWise execute too,
`Client` wraps a router per model behind an OpenAI-compatible surface. Here the
model is the dictionary key (so `Provider` still carries no model field), and
each provider supplies the `base_url` and `api_key` the client needs to call it:

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

`create()` mirrors the OpenAI SDK signature, including `stream=True`. Any
OpenAI-compatible endpoint works as a backend, which today covers most
commercial providers. Token counting, outcome reporting, hedge racing, and loser
cancellation happen inside the client; you write no timing code. For a `Router`
user who wants several models, the same pattern is one line of their own code:
`routers = {m: Router(pool[m], alpha=0.25) for m in pool}`.

## How a Decision Is Made

### The Alpha Knob

At each decision the router estimates what the request would cost at every
provider, takes the cheapest value `c_min` and the dearest value `c_max`, and
sets this request's budget to

```text
budget = c_min + alpha * (c_max - c_min)
```

`alpha = 0` pins the budget to the cheapest provider; `alpha = 1` admits the
dearest provider when it lowers latency; values between buy latency with money,
continuously. Because the budget recalibrates against the live price range,
`alpha` is dimensionless: no absolute dollars-per-request threshold appears
anywhere in the interface, and one value transfers across deployments and
models. Pass `alpha=` to `route()` to override the default for a single request,
which lets one router serve tenants with different cost-latency targets.

### Mixtures

Under a binding budget, the latency-optimal policy is a probability mixture over
at most two providers (for example 70% Together, 30% Fireworks), computed by a
small linear program. Sending traffic in those proportions is what holds average
cost at the budget while minimizing average latency, so `route()` samples one
provider from the mixture rather than returning a fixed winner. The full mixture
stays visible in `decision.weights`.

Each call samples afresh, so the budget guarantee comes from the sampled
`decision.provider` accumulated over many requests, not from any static order.
Keep the gateway a pure executor. With OpenRouter, send the one sampled provider
per request:

```python
payload["provider"] = {"only": [decision.provider]}
```

Handing a gateway an ordered fallback list would route by fixed priority instead.
It would also bypass RouteWise's own failure path: `report(error=...)`, cooldown,
and re-sampling on the next request.

### The Learning Loop

Latency here means time to first token (TTFT). The router needs no latency
configuration: each reported outcome feeds two internal estimators, a rolling
TTFT profile per provider (which tracks drift) and an output-length estimator
(which prices a request before generation starts). A provider with no
observations yet is scheduled once with priority, so a fresh router explores
every provider before it optimizes.

## Observability

`router.stats()` returns per-provider traffic share, TTFT percentiles, error
counts, and realized spend, plus overall hedge trigger and win rates.
`decision.explain()` gives a one-line account of a single choice, and
`decision.trace` returns the same content as a structured dict:

```python
print(decision.explain())
# budget=$0.000912 (alpha=0.25, c_min=$0.000718@deepinfra, c_max=$0.001494@fireworks)
# mix: deepinfra 0.71, fireworks 0.29 -> sampled: deepinfra
# excluded: together (cooldown, 18s left)
```

## Tuning (Escape Hatch)

The algorithm constants stay out of the main signatures because they are
calibrated defaults, not per-deployment choices: the hedge success target, the
failure penalty, the profile window, and the cooldown policy. A caller who has
measured a reason to change them passes a `Tuning` object, which also carries the
test seed:

```python
from routewise import Router, Tuning

router = Router([...], alpha=0.25, slo_ms=3000,
                tuning=Tuning(hedge_target=0.95, window_min=30, seed=7))
```

Most users never construct a `Tuning`.

## API Reference

### Provider

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | `str` | identifier your execution layer resolves |
| `price_in` | `float` | input price, USD per million tokens |
| `price_out` | `float` | output price, USD per million tokens |
| `price_cached` | `float \| None` | discounted price for cached prefix tokens, when offered |
| `base_url`, `api_key` | `str \| None` | required by `Client` only |

### Router

```python
Router(providers, *, alpha=0.25, slo_ms=None, tuning=None)

router.route(*, input_tokens, cached_tokens=0, alpha=None) -> Decision
router.stats() -> Stats
```

`route()` raises `NoProviderError` when every provider is in cooldown.
`cached_tokens` feeds cache-aware cost estimation; on multi-turn agent
workloads, cache discounts can outweigh list-price differences in deciding which
provider is cheapest.

### Decision

A handle returned by `route()` (and by `hedge_now()`).

| Member | Kind | Meaning |
| --- | --- | --- |
| `provider` | attribute | the sampled provider; send the request here |
| `weights` | attribute | the underlying mixture (at most two nonzero entries) |
| `expected_cost_usd` | attribute | expected cost of this decision |
| `expected_latency_ms` | attribute | expected TTFT of this decision |
| `checkpoints_ms` | attribute | hedge re-evaluation times; empty without an SLO |
| `report(*, ttft_ms=None, output_tokens=None, error=None)` | method | feed the outcome back |
| `hedge_now(*, elapsed_ms)` | method | returns a backup `Decision`, or `None` |
| `explain()`, `trace` | method, attribute | human-readable and structured explanations |

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
client.routers   # dict of the inner per-model Routers (stats, ...)
```

`providers_by_model` maps a model name to its provider list. Responses carry a
`routewise` attribute with `provider`, `cost_usd`, `ttft_ms`, `hedged`, and
`hedge_won`. In streaming mode the client measures TTFT at the first content
chunk, races hedged streams, and cancels the loser.

### Stateless Function

```python
from routewise import Candidate, route_once

decision = route_once(
    [Candidate("fireworks", cost_usd=0.0012, latency_ms=240.0),
     Candidate("together",  cost_usd=0.0008, latency_ms=410.0)],
    alpha=0.25, seed=7,
)
```

`route_once()` is a pure function for callers that already track cost and
latency: it prices nothing and learns nothing, solving only the budget LP and
sampling a provider. `Router.route()` is internally this function plus the two
estimators. Research users reproducing paper results start here.

## Design Principles

Decisions stay separate from execution. The core opens no connection, which
keeps it dependency-free and orthogonal to any HTTP client or async framework;
the bundled `Client` is one execution layer, not the only one. Cost control uses
one dimensionless knob, so no deployment-specific dollar threshold appears in the
interface. A decision is a handle, so reporting an outcome repeats no request
identity. Every extra cost the library introduces (hedges, exploration) surfaces
in `stats()` rather than disappearing into the bill.

## Scope and Roadmap

Subscription-style providers (prepaid request quotas, reserved concurrency slots)
are the second half of the RouteWise algorithm and the planned v2 extension. They
arrive as new provider types on the same `Router`, with no change to the
interface shape here. Model selection stays out permanently, since RouteWise
routes a fixed model across providers. Persisting or sharing learned state across
processes (an `export_state` / `state=` pair, or an offline `observe()` path for
replaying logged outcomes) is deferred to v1.1; v1 learns in-process, and
independent replicas each learn from their own traffic. Remaining roadmap items
are the LiteLLM strategy plugin and an end-to-end latency objective (v1 optimizes
TTFT).

## Implementation Notes

This section addresses RouteWise developers; library users can stop above. The
interface maps onto `routewise.core` as follows: `route_once()` wraps
`solve_budget_lp()` with the alpha-to-budget conversion and weight sampling;
`Router` adds the rolling latency profile and the bucket-mean output-length
estimator already prototyped in `routewise.sim` and the production gateway; `hedge_now()`
wraps `hedge_checkpoints_for_slo()`, `combined_success_probability()`, and
`select_probability_backup()`.

Three behaviors are contractual, not incidental. Primary selection samples from
the LP weights; an argmax would break the budget guarantee. Latency ties break
toward the cheaper provider (`cost_tiebroken_objective()`), applied internally so
callers never see it. One router binds to one model, which removes the model
argument from `route`, `report`, and `Provider`; callers with several models key
a dictionary of routers, exactly as `Client` does internally.

The `L`/`U` scarcity envelope, quota state, and concurrency state appear nowhere
in this interface. With API providers only, the effective cost of a request
equals its estimated metered cost, so the cost envelope that the quota shadow
price needs (see [CORE_API.md](CORE_API.md)) is never constructed.
