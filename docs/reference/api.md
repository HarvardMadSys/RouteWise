# llm-routewise Python API

This is the public Python API for `llm-routewise`. The library selects provider
names and records request outcomes. The application owns HTTP calls, client
objects, credentials, retries, and response handling.

## Install

Python 3.10 or later is required. The package has no runtime dependencies.

```bash
python -m pip install llm-routewise
```

The distribution name contains a hyphen; the import contains an underscore:

```python
import llm_routewise as rw
```

The top-level exports are `Attempt`, `Candidate`, `Decision`,
`NoProviderError`, `OutcomeError`, `Provider`, `RouteOnceResult`,
`RouteWiseError`, `Router`, `StatsSnapshot`, `Tuning`, `ValidationError`, and
`route_once`.

## Minimal example

Provider prices are USD per one million tokens.

```python
import llm_routewise as rw

router = rw.Router(
    [
        rw.Provider("provider-a", price_in=0.20, price_out=0.80),
        rw.Provider("provider-b", price_in=0.50, price_out=1.50),
    ],
    alpha=0.25,
    seed=7,
)

decision = router.route(input_tokens=800)
response = send_to_provider(decision.provider)  # Application-owned I/O.
decision.completed(
    ttft_ms=response.ttft_ms,
    output_tokens=response.output_tokens,
    cached_tokens=response.cached_tokens,
    cost_usd=response.cost_usd,
)
```

On dispatch failure, call `decision.failed(...)` instead of `completed(...)`.
Outcome reports drive learning, cooldowns, lifecycle state, and spend metrics.

## OpenRouter provider selection with the OpenAI SDK

OpenRouter can route one fixed model across multiple provider endpoints. To
keep RouteWise's selected provider aligned with the provider family that
serves the request, use OpenRouter provider slugs as `Provider.name`, keep
`MODEL` fixed, allow only `decision.provider`, and disable OpenRouter fallback.
The `openai` package and API key belong to the application; they are not
`llm-routewise` dependencies.

See OpenRouter's [provider routing](https://openrouter.ai/docs/guides/routing/provider-selection)
and [OpenAI SDK integration](https://openrouter.ai/docs/guides/community/openai-sdk)
documentation for the upstream request contract.

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
        # Illustrative USD per 1M tokens; use current endpoint prices.
        rw.Provider("openai", price_in=1.0, price_out=4.0),
        rw.Provider("azure", price_in=1.2, price_out=3.8),
    ],
    alpha=0.25,
)


def complete_with_openrouter(messages, input_tokens):
    decision = router.route(input_tokens=input_tokens)
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            extra_body={
                "provider": {
                    "only": [decision.provider],
                    "allow_fallbacks": False,
                }
            },
        )
    except Exception as exc:
        # Client-specific application code returns "health" or "request"
        # plus an optional error code.
        kind, code = classify_provider_error(exc)
        decision.failed(kind=kind, code=code)
        raise

    usage = response.usage
    decision.completed(
        ttft_ms=(time.perf_counter() - started) * 1_000.0,
        output_tokens=None if usage is None else usage.completion_tokens,
    )
    return response
```

Verify that every configured provider slug serves the fixed model. A base slug
can match multiple regional or specialized endpoints; use the full slug copied
from the model page when endpoint-level attribution or pricing matters, and
replace the illustrative prices with current USD-per-million-token prices.
With one allowed provider and fallback disabled, an unavailable selected
provider fails instead of silently changing RouteWise's provider attribution;
report that failure on the `Decision`. For non-streaming calls, measured
response latency may be used as `ttft_ms`. Pass `cost_usd` when billed cost is
available; otherwise RouteWise calculates spend after `output_tokens` is
reported.

## Cost model

For each eligible provider, estimated cost is:

```text
(non-cached input * price_in + cached input * cached price
 + predicted output * price_out) / 1,000,000
```

When `price_cached` is absent, cached input uses `price_in`. With eligible cost
extremes `C_min` and `C_max`, the budget is
`C_min + alpha * (C_max - C_min)`. RouteWise samples a provider mixture whose
expected primary cost fits this budget while preferring lower learned latency.

The predicted output term uses the caller's `estimated_output_tokens` when
provided. Otherwise RouteWise uses its internal online output-length estimate.

`alpha=0` uses the minimum-cost budget; `alpha=1` allows the full cost range.
This is an expected-mixture budget, not a hard per-request cap. A dispatched
hedge is an additional attempt and may add spend.

## Stateful API

### `Provider`

```python
rw.Provider(name: str, price_in: float, price_out: float,
            price_cached: float | None = None)
```

An immutable provider definition. `name` must be non-empty. Prices must be
finite, non-negative numbers in USD per million tokens. Provider names in one
router must be unique. Invalid values raise `ValidationError`.

### `Tuning`

```python
rw.Tuning(
    *,
    hedge_target=0.99,
    penalty_ms=60_000.0,
    window_min=15.0,
    cooldown_sec=30.0,
    cooldown_after=3,
    hedge_min_samples=5,
    exploration_lease_sec=60.0,
)
```

`Tuning` is immutable. `hedge_target` is the required combined success
probability and must be in `(0, 1]`. `penalty_ms` is the pre-first-token health
failure penalty; `window_min` is the observation window; `cooldown_sec` and
`cooldown_after` control cooldowns; `hedge_min_samples` gates hedge evaluation;
and `exploration_lease_sec` limits an in-flight cold-start lease.

Durations are finite and non-negative, except `penalty_ms`, `window_min`, and
`exploration_lease_sec`, which must be positive. Counts are positive integers.
Invalid values raise `ValidationError`.

### `Router`

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

- `providers`: non-empty iterable of uniquely named `Provider` objects.
- `alpha`: default tradeoff in `[0, 1]`.
- `slo_ms`: optional positive latency objective; enables hedge checkpoints.
- `seed`: optional integer sampling seed.
- `cold_start`: `"explore"` or `"require_observations"`.
- `clock`: optional zero-argument callable returning finite numeric seconds.
- `tuning`: optional `Tuning` instance.

Invalid constructor arguments raise `ValidationError`.

#### `Router.route`

```python
router.route(
    *,
    input_tokens: int,
    estimated_output_tokens: float | None = None,
    estimated_cached_tokens: int | Mapping[str, int] = 0,
    alpha: float | None = None,
    exclude: Iterable[str] = (),
) -> rw.Decision
```

`input_tokens` is a non-negative integer. `estimated_output_tokens` is an
optional finite, non-negative point estimate supplied by the caller. When it
is `None`, RouteWise uses its internal online output-length estimate.
`estimated_cached_tokens` is either one non-negative integer for all providers
or a mapping by provider name; missing mapping entries default to zero, and
values above `input_tokens` are clamped. `alpha` overrides the router default.
`exclude` applies only to this request, and all names must be known.

The output estimate affects routing and hedge cost calculations for this
request. It is not a generation limit or actual usage. Settle the adopted
attempt with its actual `output_tokens` or an explicit `cost_usd`; positive
actual output tokens also train the internal estimator.

Returns a `Decision` without dispatching any request. Raises `ValidationError`
for invalid values or clock output. Raises `NoProviderError` when all providers
are excluded, cooling down, unprofiled under strict cold start, or leased for
cold-start exploration.

#### `Router.observe`

```python
router.observe(provider: str, *, ttft_ms: float | None = None,
               kind: str | None = None, code: str | None = None) -> None
```

Records a measurement obtained outside a `Decision`. Supply exactly one of a
finite non-negative `ttft_ms`, or failure `kind="health"` / `kind="request"`.
`code` is an optional string allowed only with `kind`.

Health failures before first token affect latency learning; health failures
also advance the cooldown streak. Request failures are counted separately and
do not affect provider health. A TTFT success resets the health streak and
cooldown. Recognized codes are `rate_limited`, `timeout`, `server_error`,
`connection`, `bad_request`, `auth`, and `unsupported`; other values aggregate
as `other`. Events use the current router clock; historical timestamps are not
accepted. Invalid arguments raise `ValidationError`.

#### `Router.stats`

```python
router.stats() -> rw.StatsSnapshot
```

Returns a deeply immutable snapshot. Invalid clock output raises
`ValidationError`.

### `Decision`

`Decision` is returned by `route`; applications do not construct it. Its
read-only properties are:

- `provider: str`: selected primary provider.
- `weights: Mapping[str, float]`: immutable mixture weights.
- `expected_cost_usd: float`: expected primary-attempt cost.
- `expected_latency_ms: float | None`: expected latency when available.
- `checkpoints_ms: tuple[float, ...]`: hedge evaluation times.
- `state: str`: logical request state.
- `primary: Attempt` and `backups: tuple[Attempt, ...]`: attempt handles.
- `trace: Mapping[str, object]`: immutable routing metadata.

Primary-attempt reporting methods delegate to `decision.primary`:

```python
decision.first_token(*, ttft_ms, adopted=None) -> None
decision.completed(*, output_tokens=None, ttft_ms=None, cached_tokens=None,
                   cost_usd=None, adopted=None) -> None
decision.failed(*, kind, code=None) -> None
decision.cancelled() -> None
decision.declined() -> None
decision.settle(*, output_tokens=None, cached_tokens=None, cost_usd=None) -> None
```

Other methods:

```python
decision.hedge_now(*, elapsed_ms: float) -> rw.Attempt | None
decision.explain() -> str
```

`elapsed_ms` must be finite and non-negative. `explain()` summarizes the
budget, mixture, sampled provider, and exclusions.

### `Attempt` and lifecycle

The primary handle is `decision.primary`; `hedge_now` may return one backup.
Read-only properties are `provider`, `state`, and `billing_state`.

```python
attempt.first_token(*, ttft_ms, adopted=None) -> None
attempt.completed(*, output_tokens=None, ttft_ms=None, cached_tokens=None,
                  cost_usd=None, adopted=None) -> None
attempt.failed(*, kind, code=None) -> None
attempt.cancelled() -> None
attempt.declined() -> None
attempt.settle(*, output_tokens=None, cached_tokens=None, cost_usd=None) -> None
```

Attempt states follow:

```text
pending -> streaming -> completed | failed | cancelled
pending -------------> completed | failed | cancelled | declined
```

Token counts are non-negative integers; times and costs are finite and
non-negative. `first_token` records TTFT and enters `streaming`. `completed`
may include TTFT for non-streaming calls, but it cannot contradict an earlier
TTFT. `failed` takes `kind="health"` or `kind="request"` and an optional string
code. `declined` is valid only while pending and means no dispatch occurred;
use `cancelled` for an aborted dispatch. `settle` is valid only after a terminal
outcome other than `declined`.

Only one terminal outcome is accepted. Repeating an identical first-token,
terminal, or settlement report is a no-op. Contradictory state or write-once
billing data raises `OutcomeError`; invalid values raise `ValidationError`.

When no backup was offered, completing the primary automatically adopts it.
After a backup is offered, identify the response the application uses:

- For streaming, pass `adopted=True` to that attempt's `first_token`.
- For non-streaming, pass `adopted=True` to its `completed`.
- Omit the argument for a non-adopted attempt; `adopted=False` is invalid.
- Only one attempt may be adopted. Streaming adoption cannot be added later.

If every attempt is terminal, at least one completed, and none adopted, the
decision becomes `unresolved`. Decision states are `pending`, `streaming`,
`completed`, `failed`, `cancelled`, `declined`, and `unresolved`.

### Billing

`Attempt.billing_state` is `actual` when `cost_usd` is supplied, `calculated`
when output tokens permit price-based calculation, and otherwise `unknown`.
Use `settle(...)` after a terminal outcome to add billing data that was not yet
available. A terminal dispatched attempt with unknown billing increments
`unsettled_attempts` until settled. Declined attempts have no settlement.

Only an adopted, completed attempt with positive output tokens contributes to
output-length learning.

## Cold start and hedging

The default `cold_start="explore"` keeps unprofiled providers eligible and
leases a selected exploration target for `exploration_lease_sec`. Strict
`cold_start="require_observations"` excludes unprofiled providers; seed them
before routing:

```python
router.observe("provider-a", ttft_ms=240.0)
router.observe("provider-b", ttft_ms=310.0)
```

Consecutive health failures trigger cooldown after `cooldown_after`; a TTFT
success clears it.

With `slo_ms` configured, the application uses `decision.checkpoints_ms` and
calls `hedge_now(elapsed_ms=...)`. Hedging requires enough current samples for
the primary and an eligible backup. It returns `None` if no useful backup is
available; otherwise it returns an `Attempt` and consumes the single backup
slot. Returning the handle does not dispatch it. Call `backup.declined()` if it
will not be sent, or independently report and settle it if dispatched.

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

If the backup wins, mark the backup adopted and terminate the primary. A hedge
win requires the backup to be both adopted and completed.

## Clock and concurrency

The default clock is monotonic. A custom clock returns finite numeric seconds.
A time-aware public operation takes one reading, and a value below the previous
reading is clamped so router time does not move backward. A bad value or raised
clock exception becomes `ValidationError`. `hedge_now` uses caller-supplied
elapsed milliseconds, not the router clock.

Router, decision, and attempt operations support concurrent use. Hedge-slot
allocation, lifecycle transitions, adoption, and statistics retain a single
consistent outcome.

## `StatsSnapshot`

The snapshot and all nested mappings are immutable. Fields are:

- `providers`: per-provider `primary_selections`, `ttft_p50_ms`,
  `ttft_p95_ms`, `errors` split by health/request, `cooldown_remaining_sec`,
  `actual_spend_usd`, `calculated_spend_usd`, and `unsettled_attempts`.
- `hedges`: `offered`, `declined`, `won`, `actual_spend_usd`, and
  `calculated_spend_usd`. Hedge spend is a cross-section of provider spend and
  must not be added to the provider totals.
- `exploration`: `decisions` and `target_selected`.
- `decisions_without_adoption`: number of unresolved decisions.

TTFT percentiles are `None` when no current samples exist.

## Stateless API

Use this API when the application already has cost and latency estimates and
does not want observations retained between calls.

```python
rw.Candidate(name: str, cost_usd: float, latency_ms: float)

rw.route_once(
    candidates: Iterable[rw.Candidate],
    *,
    alpha: float,
    seed: int | None = None,
    rng=None,
) -> rw.RouteOnceResult
```

`Candidate` is immutable. Its name is non-empty, and cost and latency are
finite and non-negative. Candidate names passed to `route_once` are unique and
the iterable is non-empty. The caller owns eligibility filtering and cost
calculation. `alpha` is in `[0, 1]` and uses the same budget formula as Router.

Pass either an integer `seed` or an object with callable `random()`, not both.
`random()` returns a finite real in `[0, 1)`. The same seed and inputs repeat a
draw; reuse one random-number object for a sequence of draws.

`RouteOnceResult(provider, weights, budget_usd)` is immutable. `weights` is an
immutable non-negative mapping that sums to one, and the selected provider has
positive weight. Invalid values raise `ValidationError`; an unexpected lack of
a feasible result raises `RouteWiseError`.

## Exceptions

```text
RouteWiseError
├── ValidationError (also ValueError)
├── NoProviderError
└── OutcomeError
```

- `ValidationError`: invalid argument, clock result, or constructed value.
- `NoProviderError`: no eligible provider for a stateful route.
- `OutcomeError`: conflicting lifecycle, adoption, or settlement report.

## API boundaries

- Only on-demand, metered per-token provider prices are represented.
- There is no general LLM client, provider SDK adapter, network transport,
  authentication, or API-key management.
- RouteWise selects configured provider names; endpoint and model mapping stays
  with the application. It does not perform model selection.
- Quotas, concurrency limits, reserved capacity, and subscription pricing are
  not part of this API.
- Observations, cooldowns, leases, estimates, random state, and counters live in
  the current Python process and are not persisted or shared across processes.
- `observe` cannot ingest historical timestamps.
- Paper-specific simulator and experiment tooling are maintained on the
  [`eurosys27-ae`](https://github.com/HarvardMadSys/RouteWise/tree/eurosys27-ae)
  artifact branch and are not part of this library API.
