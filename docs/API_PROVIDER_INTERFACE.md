# RouteWise Library Interface

> Status: implemented design proposal, revision 5 (2026-07-12), awaiting
> Juncheng's final contract review. Scoped to API providers only; the initial release
> (package `0.3.0`) is an **API-only preview** of the paper's system. Revision
> 5 is built directly on the capacity-aware revision 4: it keeps that
> revision's capacity extension seam intact and adds the decisions from the
> GO/NO-GO review round: a completed attempt state machine (`declined`,
> adoption separated from outcome, an `unresolved` terminal), a single-delta
> billing migration, "now"-only observations on an injectable monotonic
> clock, the `kind`/`code` error model, frozen public signatures and
> validation, and the release-gate table. The `Router`, `Decision`,
> `Attempt`, and `route_once` surfaces below now have a prototype
> implementation on `codex/api-provider-library-v1`, including Tables A and B.
> The public API freezes, and `0.3.0` ships, only after the remaining open
> questions close and the release gates pass. The math primitives they build on are
> documented in [CORE_API.md](CORE_API.md). This document names the interface
> "API v1"; the package that first ships it is `0.3.0`, and any `1.0.0`
> discussion waits until the frozen API has baked in public.

## Changes from Revision 1 (through Revision 4)

1. `report()` kwargs are replaced by typed lifecycle methods
   (`first_token`, `completed`, `failed`, `cancelled`) plus a write-once
   `settle()` step for billing that arrives late. A cancelled hedge loser
   can no longer be mistaken for a failure, and a bill that lands after the
   terminal state can still be recorded.
2. A `Decision` is the logical request and owns its attempts; outcome
   methods on the decision apply to the primary attempt, and `hedge_now()`
   returns a backup `Attempt`. A logical request has at most one winner,
   possibly none.
3. New `router.observe()` inlet plus two cold-start modes. Outcome reporting
   alone starves never-selected providers. Built-in exploration blends the
   unprofiled provider into a budget-feasible mixture, so the predicted
   expected cost of every decision, exploring or not, stays within the
   budget. Every decision routed by the exploration mixture (`q > 0`) is
   tagged, not only the ones whose draw lands on the target.
4. Probability hedging is gated on profile depth (`hedge_min_samples`), and
   `hedge_now()` has a state machine: one backup slot per decision, atomic
   under concurrency, closed by the primary's first token, by adoption, and
   by termination.
5. `route()` gains `exclude=`. The request cost bounds are computed over the
   eligible set of the current request, and the retry consequence (excluding
   the cheapest provider raises the `alpha=0` budget) is stated as contract.
6. Errors are classified: health failures earn penalties and advance
   cooldown; request failures count in stats only. A failure after
   `first_token` keeps the TTFT sample and adds no second latency penalty.
7. Cooldown semantics are defined: consecutive health failures, expiry,
   reset on success. Manual `exclude=` never changes provider state.
8. Cache-state input accepts per-provider values; each provider's cache
   warms independently.
9. `seed` moves from `Tuning` to the `Router` constructor;
   `Tuning.penalty_ms` is corrected from 10000 to 60000, matching the
   shipped semantics (`DEFAULT_ERROR_PENALTY_MS`).
10. Alpha is documented as a constraint on predicted expected cost, not a
    per-request cap and not a realized-bill guarantee. Realized spend sums
    only reported values; route-time estimates never enter it.
11. `stats()` shrinks to a schema the router can honestly measure:
    `primary_selections` and `hedges.offered` rather than "requests",
    because the router observes selections, offers, and reports, never
    dispatch.
12. `route_once()` returns an immutable `RouteOnceResult`, not a reportable
    `Decision`, and the `Candidate` contract is stated.
13. The packaging contract is stated: the wheel ships the library alone,
    with `py.typed` and a fixed top-level export list.
14. `Client` and the LiteLLM plugin move to a later release; the initial
    release (`0.3.0`) ships the dependency-free `Router` only. Pending
    Juncheng's sign-off.
15. The v2 subscription roadmap no longer promises an unchanged interface;
    capacity reservation will add interactions. `routewise.core` remains the
    advanced seam.
16. The output-length estimator's exact fallback cascade (bucket mean after
    5 bucket samples, global mean after 20 global samples, 500 tokens
    before either) is disclosed, since it prices requests before any
    outcome arrives.
17. The initial implementation reserves an internal capacity-transaction
    seam, backed only by a no-op API controller in `0.3.0`. Capacity admission
    stays separate from L/U scarcity pricing and from the pure `ProviderView`
    consumed by `routewise.core`; no capacity API is exported yet.

## Revision 5 Changes

1. The attempt lifecycle gains `declined()`: an attempt that was never
   dispatched, typically an offered backup you chose not to send.
   `cancelled()` now means only "dispatched, then aborted". Neither writes a
   penalty. The capacity seam aligns: an attempt superseded by a failed
   commit closes as `declined`, not `cancelled`.
2. Adoption is separated from outcome and renamed: `adopted=True` (was
   `selected=`) marks the attempt whose response you took. An adopted
   attempt can still fail or be cancelled; the logical request then shares
   that outcome and has no winner. **Winner = adopted and completed.** Only
   winners train the output-length estimator; `hedges.won` counts only
   adopted-and-completed backups. Adoption cannot be added after the fact:
   it rides only on `first_token` or `completed`, there is no `adopt()`
   method, and `settle()` carries no adoption flag.
3. The logical request has a total resolution rule, including the previously
   undefined cases: it mirrors the adopted attempt's terminal state when
   adoption exists; without adoption it resolves `unresolved` if any attempt
   completed (counted in `decisions_without_adoption`), else `failed` if any
   failed, else `cancelled` if any was cancelled, else `declined`.
4. Route-time cache input is renamed `estimated_cached_tokens`; the
   `cached_tokens` reported at settlement is billed truth. Every attempt
   freezes a price snapshot at creation, and calculated costs use it.
5. Billing is a three-state machine per attempt over `{unknown, calculated,
   actual}` — `calculated` is skipped when an explicit `cost_usd` arrives
   first — with a single-delta rule: each reporting call is validated
   whole, the attempt's unique target state is computed, and one atomic
   delta reconciles the aggregates. A late explicit `cost_usd` subtracts the
   attempt's calculated contribution and adds the actual amount.
   `unsettled_attempts` has explicit increment and decrement rules.
   Per-provider spend includes backup attempts; `hedges.*_spend_usd` is a
   cross-slice of the same money and must not be added to it.
6. `observe()` loses `at=` in `0.3.0`: every observation is stamped "now" by
   the router's injectable monotonic clock (`Router(clock=)`). Historical
   bootstrap becomes a separate later API rather than a second clock inside
   one parameter.
7. `failed(kind=..., code=...)` replaces the string taxonomy. Behavior is
   decided solely by `kind` (`"health"` or `"request"`); `code` is a label,
   and unknown codes aggregate as `"other"` so metrics cardinality stays
   bounded. The caller judges whether an expired key is a provider-health
   event.
8. Built-in exploration promises body-routing self-start only. One window
   event ends cold start, while hedging needs `hedge_min_samples`
   successes, so a low-traffic provider may be routable long before it is
   hedge-ready; the router does not force extra exploration to close that
   gap.
9. A "success" for the cooldown streak is any TTFT sample entering the
   provider's window, whether from `first_token`, from `completed`, or from
   a successful observation.
10. Types and validation freeze in Contract Table D: provider identity
    parameters and outputs are `str` names, with the `Router` constructor as
    the one signature that takes `Provider` objects; a cache mapping missing
    a provider means 0 and an unknown name raises; public mappings are deeply immutable; every token,
    cost, and latency argument is validated finite and non-negative; validation
    and outcome-conflict failures commit no business state.
11. `route_once()` accepts a reusable `rng=`, mutually exclusive with
    `seed=`. With a fixed seed, identical inputs and weights replay the
    identical draw, breaking the long-run empirical mixture; each single
    call's LP solution and budget are unaffected. The contract warns about
    this.
12. Hedging defines a survival-zero fallback: when the primary's empirical
    distribution says it should already have finished but it observably has
    not, the primary's remaining chance is taken as zero and the combined
    probability reduces to the backup's. Today's core returns 0.0 there and
    must be wrapped or fixed before hedging goes public.
13. The router holds no strong references to outstanding handles; attempt
    state lives on the handles, and router-side residue is bounded by the
    exploration lease timeout plus any open capacity reservations (no-op in
    `0.3.0`).
14. `0.3.0` is named an API-only preview. The paper's headline results
    include quota and concurrency routing, so this release does not claim
    to reproduce the full paper.

## What RouteWise Is

RouteWise is a Python library for applications that call the same model through
multiple API providers. Open-weight models such as DeepSeek-V4 are sold by many
providers at different prices, with latency that varies across providers and
drifts over time. For each request, RouteWise decides which provider to use:
the lowest expected latency whose cost fits a budget, controlled by a single
knob. When a response risks missing its deadline, RouteWise can dispatch a late
backup request to a second provider. The router learns from what you tell it:
outcomes of its own decisions, and any measurements you feed it from outside.

RouteWise is not a general LLM client (it ships no per-provider SDKs), not a
model selector (the model is fixed; only the provider changes, so response
quality is never traded for cost), and not a hosted service (your API keys stay
in your process).

## Installation

```bash
pip install "routewise>=0.3,<0.4"
```

The initial release (package `0.3.0`) is an API-only preview: it contains the
decision library alone and imports nothing outside the standard library. The
version lower bound avoids the incompatible hosted-service SDK published as
`routewise` through `0.2.0`; that SDK's `RouteWiseClient` is not part of this
interface. The
paper's full system also prices quota and concurrency subscriptions; those
arrive in a later generation, so this release does not claim to reproduce the
full paper results. An execution client (`routewise[client]`, httpx-based) and
a LiteLLM routing-strategy plugin (`routewise[litellm]`) are planned as
optional extras for a later release; `0.3.0` defines no extras. See Scope and
Roadmap.

## The Whole Interface

One router serves one model. You ask it where to send a request, you send the
request with your own code, and you tell the decision what happened so the next
decision is better informed.

```python
from routewise import Provider, Router

router = Router(
    [Provider("fireworks", price_in=0.27, price_out=1.10),
     Provider("together",  price_in=0.18, price_out=0.88)],
    alpha=0.25,          # the one knob: 0 = cheapest, 1 = fastest
    seed=42,             # sampling is random; seed it for reproducibility
)

decision = router.route(input_tokens=1800)           # ask: which provider?
response = send(decision.provider, request)          # your execution layer
decision.first_token(ttft_ms=response.ttft_ms)       # the stream started
decision.completed(output_tokens=response.output_tokens)   # and finished
```

The common path is three nouns (`Provider`, `Router`, `Decision`), two moments
(ask with `route()`, then tell the decision what happened), and one knob
(`alpha`). That shorthand names the common path, not the whole system: the full
v1 surface adds `observe()` for measurements the router did not initiate, and,
when hedging is enabled with `slo_ms`, the question `hedge_now()`. Prices are
USD per million tokens. `Provider.name` is an opaque label your execution layer
resolves; the `Router` itself makes no network calls.

Everything else a caller might want from a decision is a read-only attribute,
so it adds nothing to learn:

```python
decision.provider            # the sampled provider to use
decision.weights             # the underlying mixture, e.g. {"together": 0.7, "fireworks": 0.3}
decision.expected_cost_usd   # predicted expected cost of this decision
decision.expected_latency_ms # expected TTFT; None while exploring an unprofiled provider
decision.explain()           # one human-readable line explaining the choice
```

## Reporting Outcomes

`route()` returns a `Decision`: the logical request. Behind it sit one or more
dispatches, each represented by an `Attempt` handle; the primary attempt is
created with the decision, and `hedge_now()` adds a backup. The decision
already holds the request identity (provider, input length, estimated cached
length), so outcome calls carry only new information. The ways an attempt can
end mean different things, so the outcomes are typed methods rather than one
`report()` with keyword arguments:

```python
decision.first_token(ttft_ms=312.0)      # the stream started: one TTFT sample
decision.completed(output_tokens=540)    # it finished; record known usage
decision.settle(cost_usd=0.00092)        # the bill arrived later

# Alternative terminal outcomes instead of completed():
# decision.failed(kind="health", code="rate_limited")  # penalty + cooldown
# decision.cancelled()                   # dispatched, then aborted: no penalty
# backup.declined()                      # offered, never dispatched
```

Outcome methods on a `Decision` apply to its primary attempt; each backup
`Attempt` reports itself with the same methods.

**Lifecycle.** `first_token` may be called at most once, before any terminal
call; it records one TTFT observation in the provider's latency profile and,
as a success signal, resets the provider's failure streak. One terminal state
(`completed`, `failed`, `cancelled`, or `declined`) is allowed per attempt;
`declined` is legal only from `pending`, because a stream that has started
was dispatched by definition. Repeating an identical call is a no-op; a call
that contradicts an earlier one (a different terminal state, or a different
value for an already-known field) raises `OutcomeError`. A failure after
`first_token` keeps the TTFT sample and writes no second latency penalty: one
attempt never books two latency entries. The failure's `kind` still applies,
so a health failure advances cooldown while a request failure counts in stats
only. A handle you never report teaches the router nothing, and any money the
attempt did spend goes unrecorded, so `stats()` undercounts. Terminally
report every attempt you dispatched, then settle its billing fields when they
become known. The full transition table is Contract Table A.

**Settlement is write-once, applied as one delta.** Billing truth often
arrives after the terminal state: the provider's usage record, the hedge
loser's partial bill, a cache-hit count. The three billing fields
(`output_tokens`, `cached_tokens`, `cost_usd`) may each be supplied exactly
once per attempt, either with the terminal call or later through `settle()`,
which fills fields that are still unknown. `cached_tokens` here is billed
truth, as distinct from the route-time `estimated_cached_tokens`. `settle()`
is legal in every terminal state except `declined`: a declined attempt was
never dispatched, so it has nothing to bill, and settling it raises
`OutcomeError`. Supplying
the same value again is a no-op; supplying a different value raises
`OutcomeError`. There is no overwrite path, and none is needed: route-time
estimates never enter realized spend, so a settled value is always the first
actual value, never a correction of one. Each reporting call is validated as
a whole before anything is applied; then the attempt's billing state is
recomputed to its unique target (`actual` if `cost_usd` is known, else
`calculated` if computable, else `unknown`); then one atomic delta reconciles
the aggregates from the prior state to the target, per Contract Table B. A
call that supplies `output_tokens` and `cost_usd` together therefore lands
directly on `actual` and never contributes to calculated spend.
`completed()`'s `output_tokens` is optional; leave it out when usage arrives
later, and settle it. `failed`, `cancelled`, and `declined` carry no billing
parameters of their own; money always flows through `completed` or `settle`.
`cancelled()` deliberately reports nothing by default, because an aborted
attempt's usage is usually unknown at abort time; zero is a claim, so do not
write it unless you know it. If providers someday revise issued bills, that
becomes an explicit revision mechanism in a later version, not an overwrite
here.

`completed(output_tokens=None, ttft_ms=None, cached_tokens=None,
cost_usd=None, adopted=None)` marks normal completion and records any supplied
usage or billing fields. It does not by itself guarantee that a monetary cost
is known. Non-streaming callers skip `first_token` and pass `ttft_ms` here.

`failed(kind=..., code=None)` separates effect from label. `kind` decides
everything the router does:

| `kind` | Effect |
| --- | --- |
| `"health"` | one penalty sample (60 s default; none if `first_token` already recorded) plus cooldown progress |
| `"request"` | counted in `stats()`; no penalty, no cooldown |

`code` is a free-form label for observability. Recommended codes:
`rate_limited`, `timeout`, `server_error`, `connection` for health;
`bad_request`, `auth`, `unsupported` for request. Codes outside the
recommended list aggregate as `"other"` in `stats()`, so metrics cardinality
stays bounded. The caller judges the kind: an expired key can be
`kind="request"` while you fix your configuration, or `kind="health"` if you
want the provider benched. The recommended code list is an open question.

**Adoption and winners.** Adoption and outcome are separate facts. Mark the
attempt whose response you actually took with `adopted=True`, at the moment
you take it; the router cannot infer adoption from completion order, because
in a streaming race the primary may already be streaming to your user while
the backup finishes generating first.

- While the decision has only its primary attempt (no backup was ever
  offered), a `completed` primary is adopted automatically. The simple path
  never writes `adopted=`.
- Once a backup attempt has been created, adoption must be explicit: pass
  `adopted=True` on `first_token` (streaming, at the moment you adopt the
  stream) or on `completed` (non-streaming). With multiple attempts and no
  explicit adoption, no attempt is adopted: completions still record spend,
  nothing trains the output-length estimator, and `hedges.won` does not
  move.
- Marking two attempts adopted raises `OutcomeError`. Adoption cannot be
  added after the fact: no `adopt()` method exists and `settle()` carries no
  adoption flag.
- **Winner = adopted and completed.** An adopted attempt that then fails or
  is cancelled stays adopted, but the request has no winner.

The logical request resolves exactly one way. With an adoption, the decision
mirrors the adopted attempt's terminal state: `completed` (a winner exists),
`failed`, or `cancelled`. Without an adoption, once every attempt is
terminal: `unresolved` if any attempt completed (an adoption was possible
but never declared; `decisions_without_adoption` increments and nothing
trains), else `failed` if any attempt failed, else `cancelled` if any was
cancelled, else `declined`.

Streaming race, with the primary adopted:

```python
decision.first_token(ttft_ms=312.0, adopted=True)
```

Or, in a separate non-streaming race, with the backup adopted:

```python
backup.completed(output_tokens=540, adopted=True)
```

**Bookkeeping rules, part of the contract:**

1. Every attempt's `first_token` is a genuine observation of its provider:
   it feeds that provider's latency profile and resets its failure streak.
2. The output-length estimator trains only on an attempt that is the winner
   (adopted and `completed`), at the moment its `output_tokens` becomes
   known, and only when that value is positive; a zero-output completion
   records spend but trains nothing. Everything else, including an adopted
   attempt that later failed and settled partial usage, updates spend only.
3. A cancelled or declined attempt never writes a penalty and never advances
   cooldown.
4. One attempt books at most one latency entry: a TTFT sample or one failure
   penalty, never both.
5. Manual `exclude=` on `route()` changes no provider state.
6. The router holds no strong reference to any outstanding handle: attempt
   state lives on the handle, the router keeps only aggregates, the
   exploration lease table, and any open capacity reservations (no-op in
   `0.3.0`), and a dropped handle is garbage-collected with no router-side
   residue beyond those bounds.

## Keeping Profiles Fresh

Reported outcomes alone cannot keep a router honest. Selection favors providers
that look good, so a provider that is never chosen never produces an outcome:
it cannot recover from a bad first impression, and an idle provider's drift
goes unseen. The library closes this loop with a cold-start behavior and an
observation inlet; production deployments add their own probes through the
inlet.

**Cold start.** The `Router` constructor takes `cold_start=`, with two modes.

The default, `cold_start="explore"`, blends unprofiled providers into routing
as budget-feasible mixtures. A provider counts as unprofiled while its current
window holds no event at all; one health-failure penalty already counts as an
event, so a provider that just failed is cooldown's business, not
exploration's. When at least one eligible, unleased, unprofiled provider
exists, the router picks the cheapest such provider as exploration target
(ties by name) and routes the request by a two-point
mixture between the target `u` and the cheapest eligible provider, giving `u`
the largest probability that keeps the mixture's predicted expected cost
within the budget:

```text
q = 1                                   if c_u <= budget
q = (budget - c_min) / (c_u - c_min)    otherwise
```

When `q` comes out zero (at `alpha=0`, for any target dearer than the floor),
the router skips exploration entirely and routes by the plain LP: no tag, no
counter, no lease; exploration cannot move traffic that the budget could
never admit. When `q > 0`, every decision routed by the exploration mixture
is tagged `reason="cold_start_exploration"` in `explain()` and `trace` (the
trace also records the target, `q`, and `latency_estimate="unprofiled"`),
whether or not the draw lands on the target, because either way the request
deviated from the LP optimum. `stats()` counts both: `exploration.decisions`
for mixture-routed decisions and `exploration.target_selected` for draws that
hit the target. The per-provider exploration lease is taken atomically only
when the draw lands on the target, so the leased decision's primary is the
target itself; a draw that lands on the cheapest provider consumes no lease.
A leased target is skipped by later requests until the lease releases: at the
target's first window event or after `Tuning.exploration_lease_sec`,
whichever comes first, so an exploration handle that is never reported cannot
block the provider forever. A lease belongs to the attempt that took it; a
stale attempt's late event never releases a newer lease. When no eligible,
unleased, unprofiled target remains for this request (each candidate is
leased, excluded, or cooling), the request skips exploration: it routes by
the plain LP over the remaining eligible providers, and if none exist,
`route()` raises `NoProviderError` whose message reports the actual reasons
that emptied the set, naming the leased-out cold start only when active
exploration leases alone did so. Exploration respects
the same budget as every other decision; it may still cost more than the
non-exploring optimum would have, and it routes the non-target mass to the
cheapest provider rather than the fastest, so its price is paid in latency
and in optimality, never in a budget violation.

Built-in exploration promises body-routing self-start and nothing more: one
window event ends a provider's cold start, while probability hedging needs
`hedge_min_samples` successes, so a low-traffic provider may be routable long
before it is hedge-ready. The router does not force further exploration to
close that gap; hedging readiness comes from natural traffic or from warmup
you run through `observe()`.

The strict mode, `cold_start="require_observations"`, makes unprofiled
providers ineligible. If every provider is unprofiled, `route()` raises
`NoProviderError` with advice to seed profiles first. This is the production
mode: bootstrap each provider from probes before taking traffic.

Built-in exploration is a self-start convenience, not the warmup discipline of
the research harness. That harness starts with 24 probe rounds at a 5-second
cadence (roughly two minutes) and validates at least 5 samples per provider
before replay; each submitted warmup probe that exhausts its configured
attempts in failure injects one synthetic 10-second sample (rounds skipped
because a probe is still in flight inject nothing), so a provider that failed
every probe is ranked from synthetic samples alone, and the validation
threshold does not guarantee real successes. Steady-state probe failures are
dropped. The 5-second probe cadence and the synthetic 10-second sample appear
in the paper's revised profiling text; the 24-round warmup shape and the
5-sample validation threshold are harness defaults. A
deployment that wants warmup semantics runs its own probe loop and feeds
`observe()`; the router schedules no probes and runs no timers. The paper
also sketches exploratory hedging (hedge dispatches as organic probes); that
is a different mechanism from cold-start exploration and is not in this
release, because cold start has no traffic to hedge and may have no SLO
configured.

**The observation inlet.** `router.observe()` feeds measurements the router
did not initiate:

```python
router.observe("together", ttft_ms=284.0)                  # your probe succeeded
router.observe("fireworks", kind="health", code="timeout") # your probe failed
```

`observe()` is a measurement inlet, not a probe API: everything you feed it
counts exactly as if a routed request had produced it, minus traffic-share
accounting. A success adds a TTFT sample and resets the failure streak; a
failure follows the same `kind`/`code` model as `failed()`, so a health
failure adds a penalty sample and advances cooldown. Choose what you report:
the production sidecar drops failed probes rather than reporting them,
because a failed probe is weak evidence next to real-request feedback, and a
probe loop that wants the same policy reports successes only. Every
observation is stamped "now" by the router's clock; there is no `at=` in
`0.3.0`, so replaying history with original timestamps is not supported yet.
Bootstrap by replaying recent measurements as current ones; a dedicated
history-import API can come later without putting two clock domains inside
one parameter. Two uses today: periodic out-of-band probes (a few lines of
your own scheduling code), and warm-starting a fresh router by replaying a
peer replica's recent measurements.

**Thin profiles and hedging.** One live sample relieves starvation, but it is
not a latency distribution. Providers with fewer than
`Tuning.hedge_min_samples` successful samples in the current window take no
part in probability hedging, in either role: they are not offered as backups,
and `hedge_now()` returns `None` while the primary's window is that thin. Body
routing tolerates a thin profile; tail probability math does not. The gate is
new library policy: the default of 5 borrows the research harness's warmup
validation threshold, and no production hedger currently enforces a
per-provider sample minimum, so the value needs validation (see Open
Questions).

## Tail-Latency Hedging

The router optimizes the body of the latency distribution. Hedging protects the
tail. Give the router an SLO (service-level objective, the time-to-first-token
deadline), and each decision carries checkpoint times at which a backup may be
worth sending:

```python
router = Router([...], alpha=0.25, slo_ms=3000, seed=42)

decision = router.route(input_tokens=1800)
# The primary is in flight. At each time in decision.checkpoints_ms where no
# first token has arrived yet:
backup = decision.hedge_now(elapsed_ms=1500)         # None, or an Attempt
if backup is not None:
    dispatch(backup.provider, request)               # races the in-flight primary
```

`hedge_now()` returns `None` while no backup is worth sending, or an `Attempt`
for the one backup this decision may have. Its state machine is part of the
contract: each decision has one backup slot; concurrent `hedge_now()` calls
produce at most one `Attempt` between them (the call is atomic); once the slot
is used, later calls return `None`; the unused slot closes as soon as the
primary records its first token, because hedging protects time to first token
and there is nothing left to protect after it arrives; and after an attempt
is adopted or the logical request reaches a terminal state, `hedge_now()`
returns `None` regardless. `elapsed_ms` is measured by you, relative to the
primary's dispatch; the router additionally reads its own clock once per call
to evaluate current profiles, cooldowns, and backup eligibility. Returning an
`Attempt` consumes the slot whether or not you dispatch it. The router never
observes dispatch, which is why `stats()` counts `hedges.offered`, not
dispatched; if you decide not to send an offered backup, close its handle
with `declined()` so accounting stays honest (`hedges.declined` counts
those). `hedges.won` counts backups that were adopted and completed.

When the race resolves, report each attempt for what happened to it:

```python
decision.first_token(ttft_ms=1710.0, adopted=True)   # primary adopted
decision.completed(output_tokens=540)
backup.cancelled()                                   # dispatched loser: no penalty
backup.settle(cost_usd=0.0002)                       # loser's bill, once known
```

If the backup wins, the calls mirror. An `Attempt` is not a `Decision`: a
backup is one dispatch chosen by the hedging math, so it carries no mixture,
no budget, and no checkpoints of its own, and it cannot itself hedge.
Internally the router withholds the backup until the latest checkpoint at
which the combined primary-plus-backup chance of meeting the SLO still clears
the target, so backups stay rare and each one has a quantified reason.

One boundary case is contractual. When the primary's empirical distribution
assigns zero survival probability to the observed elapsed time (every window
sample sits below it) yet the request observably has no first token, the
observation wins: the primary's remaining chance within the SLO is taken as
zero and the combined probability reduces to the backup's chance within the
remaining budget. Today's core `combined_success_probability` returns 0.0 in
exactly that situation, which would refuse to hedge precisely when the
primary looks most lost; the facade must apply the fallback until core is
fixed (implementation gap 8). Adding `slo_ms` is the only change needed to
enable hedging.

## How a Decision Is Made

### The Alpha Knob

At each decision the router estimates what the request would cost at every
eligible provider and takes the cheapest and dearest values as the request
cost bounds `c_min` and `c_max`. This request's budget is

```text
budget = c_min + alpha * (c_max - c_min)
```

`alpha = 0` pins the budget to the cheapest eligible provider; `alpha = 1`
admits the dearest; values between buy latency with money, continuously. The
eligible set is the providers you passed, minus `exclude=`, minus any in
cooldown, minus unprofiled providers holding an active exploration lease (in
explore mode) or all unprofiled providers (in strict mode), and the bounds
are computed over that set. `alpha` therefore expresses a position
within the currently reachable cost range, which is what makes it
dimensionless and transferable across deployments and models; no absolute
dollars-per-request threshold appears anywhere in the interface.

Three consequences are contractual. First, the bounds move when the set moves.
Excluding the cheapest provider raises `c_min`, so when a request is retried
with `exclude={failed_cheapest}`, the absolute budget of an `alpha=0` retry is
the price of the second-cheapest provider. Second, the budget constrains the
predicted expected cost of the decision's mixture, not each individual
request: sending 70% of identical requests to a cheap provider and 30% to a
dear one holds the average at the budget even though each request lands on one
side or the other. Third, the constraint is a prediction, priced with the
output-length estimate at route time; the realized bill can differ when
generation runs long, and hedge backups spend outside the budget entirely.
Realized spend is visible in `stats()`, never silently reconciled.

Pass `alpha=` to `route()` to override the default for a single request, which
lets one router serve tenants with different cost-latency targets. Pass
`exclude=` for request-bound ineligibility the router cannot know about: a
context window the provider cannot fit, a capability the request needs, a
tenant pinned away from a region. Cooldown handles provider health; `exclude`
handles this request only, and never changes provider state.

### Mixtures

Under a binding budget, the latency-optimal policy is a probability mixture
over at most two providers (for example 70% Together, 30% Fireworks), computed
by a small linear program. Sending traffic in those proportions is what holds
average cost at the budget while minimizing average latency, so `route()`
samples one provider from the mixture using the router's own seeded RNG. The
sampling is contractual: replacing it with a fixed argmax would break the
budget guarantee. The full mixture stays visible in `decision.weights`.

Each call samples afresh, so the budget guarantee comes from the sampled
`decision.provider` accumulated over many requests, not from any static order.
Keep the gateway a pure executor. With OpenRouter, send the one sampled
provider per request:

```python
payload["provider"] = {"only": [decision.provider]}
```

Handing a gateway an ordered fallback list would route by fixed priority
instead. It would also bypass RouteWise's own failure path: `failed(...)`,
cooldown, and re-sampling on the next request.

### The Learning Loop

Latency here means time to first token (TTFT). Outcomes and external
observations feed a rolling TTFT profile per provider, which tracks drift;
only a completed winner with a positive output length feeds the output-length
estimator that prices requests before generation starts. The length
estimator is a bucket mean over input length with a fixed fallback cascade: a
request is priced by its input-length bucket's mean once that bucket has 5
samples, by the global mean once the estimator has 20 samples overall, and by
a fixed default of 500 output tokens before either threshold is met. That
default sets the cost estimates, and with them the mixtures, of a fresh
router. Whether callers may override the estimate per request
(`estimated_output_tokens=`, useful when `max_tokens` is known) is an open
question. Cache pricing: `estimated_cached_tokens` discounts the input at
`price_cached`; a provider with `price_cached=None` bills cached tokens at
`price_in`, so the estimate changes nothing there. Cold start and coverage of
idle providers are described in Keeping Profiles Fresh.

## Observability

`stats()` returns an immutable `StatsSnapshot` of what the router can
honestly measure: it observes primary selections, backup offers, and your
reports, never actual dispatch. The schema below is the freeze candidate;
once signed off, fields are added over time, never changed or removed.

Per provider: `primary_selections` (decisions whose sampled primary was this
provider), `ttft_p50_ms` and `ttft_p95_ms` over the current window (`None`
while empty), `errors` (counts keyed by `kind` then `code`, unknown codes
under `"other"`), `cooldown_remaining_sec` (`0.0` when healthy),
`actual_spend_usd`, `calculated_spend_usd`, and `unsettled_attempts`.
Per-provider spend includes backup attempts routed to that provider.

Globally: `hedges.offered`, `hedges.declined`, `hedges.won` (adopted and
completed backups), `hedges.actual_spend_usd`,
`hedges.calculated_spend_usd` (a cross-slice of the per-provider spend,
never additive with it), `exploration.decisions`,
`exploration.target_selected`, and `decisions_without_adoption`.

Spend provenance follows Contract Table B: explicit `cost_usd` contributes to
actual spend; complete final usage without an explicit cost contributes to
calculated spend; a terminally reported attempt with insufficient billing
data counts in `unsettled_attempts` and never contributes an unknown amount
to a money total. Unreported attempts are invisible. Counter lifetimes
(router lifetime versus rolling window) are part of Open Question 7.
Anything richer (histories, percentile curves, per-tenant splits) is your
telemetry's job, fed from decision traces.

`decision.explain()` gives a one-line account of a single choice, and
`decision.trace` returns the same content as an immutable `Mapping`:

```python
print(decision.explain())
# budget=$0.000912 (alpha=0.25, c_min=$0.000718@deepinfra, c_max=$0.001494@fireworks)
# mix: deepinfra 0.71, fireworks 0.29 -> sampled: deepinfra
# excluded: together (cooldown, 18s left)
```

## Tuning (Escape Hatch)

The algorithm constants stay out of the main signatures because they are
calibrated defaults, not per-deployment choices. A caller who has measured a
reason to change them passes a `Tuning` object:

```python
from routewise import Router, Tuning

router = Router([...], alpha=0.25, slo_ms=3000, seed=7,
                tuning=Tuning(hedge_target=0.95, window_min=30))
```

Most users never construct a `Tuning`.

`penalty_ms` is the synthetic latency recorded per health failure, whether
reported or observed. The 60-second default matches the production semantics;
it keeps a provider that is rate-limiting or erroring unattractive to the LP
while real samples age out. `cooldown_after` consecutive health failures with
no intervening success put the provider in cooldown for `cooldown_sec`: while
cooling it is ineligible for routing and for hedge backups, expiry restores
eligibility, and any success resets the counter, where a success is any TTFT
sample entering the window (`first_token`, `completed`, or a successful
observation). Request-kind failures touch neither the counter nor the
profile. There is no separate half-open state: after expiry the provider's
window still carries its penalty samples, so the LP avoids it unless the
alternatives are worse, and its first success starts washing the penalties
out. Whether a failure streak should also expire with the window is an open
question. `exploration_lease_sec` bounds how long a cold-start exploration
may hold its per-provider lease; it is an elapsed-time deadline on the
monotonic clock and deliberately not `penalty_ms`, which is a synthetic
latency value, because the two move for different reasons.

## Clock

The router reads time from one injectable monotonic clock:
`Router(clock=...)` accepts a zero-argument callable returning seconds as a
`float`, defaulting to `time.monotonic`. Every time-dependent behavior reads
it: profile windows, cooldown expiry, exploration leases, and the "now" stamp
on observations. Wall-clock time appears nowhere in the algorithm, so clock
adjustments cannot corrupt windows, and tests inject a fake clock instead of
sleeping. Raw readings must be finite; the router derives its effective time
as `now = max(previous_now, raw_now)`, so effective time is non-decreasing
even when the raw clock steps backward. Every public operation reads the
clock at most once and uses that single effective `now` for all of its
checks, so one call never straddles two instants. Contract Table C lists which construct reads
the clock and when.

## Errors and Validation

The constructor validates once: `alpha` within [0, 1], `slo_ms` positive when
given, prices finite and non-negative, provider names unique and non-empty, at
least one provider. `route()` raises `NoProviderError` when the eligible set is
empty (every provider excluded, cooling, unprofiled in strict cold-start
mode, or unprofiled with an exploration already in flight in explore mode);
it never silently routes to an ineligible provider. Invalid argument
values raise `ValidationError` at the call that supplied them. Outcome misuse
(a conflicting terminal call, a different value for an already-known field, a
second adopted attempt) raises `OutcomeError`. All library exceptions subclass
`RouteWiseError`. Argument-validation and outcome-conflict failures commit no
business state. A capacity-exhausted route commits no selection, lease,
counter, spend, or RNG change; advancing rolling-window housekeeping to the
operation's captured `now` is not rolled back. Contract Table D gives the
per-callable rules.

## Contract Tables

The four tables below, plus the release-gate table in Implementation Notes,
are the implementable core of this contract. Prose elsewhere explains intent;
where prose and table disagree, the table wins.

### Table A: Attempt State Machine and Logical Resolution

States: `pending` → (`streaming`) → one of `completed | failed | cancelled |
declined`. `settle()` is valid in any terminal state except `declined` and
does not change state. Identical repeated calls are no-ops; contradictions
raise `OutcomeError`.

| From | Event | To | Latency profile | Failure streak | Estimator | Spend |
| --- | --- | --- | --- | --- | --- | --- |
| pending | `declined()` | declined | — | — | — | — |
| pending | `first_token(ttft_ms)` | streaming | +1 TTFT sample | reset | — | — |
| pending | `completed(...)` | completed | +1 TTFT sample if `ttft_ms` given | reset if `ttft_ms` given | winner only, when `output_tokens` known and > 0 | per Table B |
| pending | `failed(kind="health")` | failed | +1 penalty sample | +1 | — | via `settle` |
| pending | `failed(kind="request")` | failed | — | — | — | via `settle` |
| pending | `cancelled()` | cancelled | — | — | — | via `settle` |
| streaming | `completed(...)` | completed | — (already sampled) | — (already reset) | winner only, when `output_tokens` known and > 0 | per Table B |
| streaming | `failed(kind="health")` | failed | — (keeps TTFT; no penalty) | +1 | — | via `settle` |
| streaming | `failed(kind="request")` | failed | — | — | — | via `settle` |
| streaming | `cancelled()` | cancelled | — (keeps TTFT) | — | — | via `settle` |
| terminal except declined | `settle(...)` | unchanged | — | — | winner's first known `output_tokens` only, when > 0 | per Table B |

Adoption: `adopted=True` rides only on `first_token` or `completed`; at most
one adopted attempt per decision; no post-hoc adoption. A decision that never
offered a backup adopts its primary implicitly on completion.

Logical resolution (total): with an adoption, the decision mirrors the
adopted attempt's terminal state (`completed` = winner, `failed`,
`cancelled`). Without one, once all attempts are terminal: any completion →
`unresolved` (increments `decisions_without_adoption`); else any failure →
`failed`; else any cancellation → `cancelled`; else → `declined`. The hedge
slot closes at the primary's first token, at adoption, or at logical
termination, whichever comes first.

### Table B: Billing States and Aggregate Migration

Per attempt, `billing_state ∈ {unknown, calculated, actual}`. Each attempt
freezes a price snapshot (its provider's `price_in`, `price_out`,
`price_cached`) at creation; calculated costs always use the snapshot. Every
reporting call is validated whole, the unique target state is computed
(`actual` if `cost_usd` known, else `calculated` if `output_tokens` known,
else `unknown`), and one atomic delta moves the aggregates from the prior
state to the target; a call carrying both usage and `cost_usd` lands directly
on `actual`.

| Transition | Trigger | Atomic aggregate delta |
| --- | --- | --- |
| unknown → unknown | entering a terminal state other than `declined` with billing still unknown (no fields, or `cached_tokens` only) | `unsettled_attempts` +1, exactly once per attempt |
| unknown → unknown | a later `settle` supplies `cached_tokens` only | store the field; no aggregate change |
| unknown → calculated | `output_tokens` becomes known; `cached_tokens` uses billed truth if known, else the route-time estimate | add snapshot-priced cost to `calculated_spend_usd`; `unsettled_attempts` −1 if previously counted |
| unknown → actual | explicit `cost_usd` known | add to `actual_spend_usd`; `unsettled_attempts` −1 if previously counted |
| calculated → calculated | a still-unknown usage field is filled by `settle` | replace the attempt's derived contribution with the recomputed one |
| calculated → actual | explicit `cost_usd` arrives late | subtract the attempt's derived contribution from `calculated_spend_usd`, add `cost_usd` to `actual_spend_usd` |
| actual → actual | identical `cost_usd` repeated | no-op; a different value raises `OutcomeError` |
| actual → actual | late `output_tokens` or `cached_tokens` after an explicit cost | store the fields; spend delta zero (the explicit cost stays authoritative); the winner's first known positive `output_tokens` still trains the estimator |

Rules: billing fields are write-once from `None`; there is no overwrite
anywhere (calculated amounts are derived, never stored in a field, so a late
explicit cost is still the field's first write). A billed `cached_tokens`
greater than `input_tokens` is stored as reported, but derived costs use
`min(cached_tokens, input_tokens)`, matching core's cached-token cap. A
`declined` attempt never enters billing accounting. Unknown amounts are never summed as money;
`unsettled_attempts` counts them. Per-provider spend includes backup
attempts; `hedges.*_spend_usd` re-slices the same attempts and must never be
added to provider spend.

### Table C: Clock and Error Semantics

| Construct | Time source | Semantics |
| --- | --- | --- |
| Profile window (`window_min`) | the current operation's single `now` | samples older than the window fall out of mean/CDF |
| Cooldown (`cooldown_sec`) | the current operation's single `now` | expiry restores eligibility; success resets the streak |
| Exploration lease (`exploration_lease_sec`) | the current operation's single `now` | released by the target's first window event or expiry, whichever first |
| `observe()` stamp | the current operation's single `now` | always "now"; no `at=` in `0.3.0` |
| `hedge_now(elapsed_ms=...)` | one reading per call, shared by all its checks | `elapsed_ms` is caller-measured from primary dispatch; the operation's `now` timestamps the profile, cooldown, and eligibility lookups |

| Path | Profile | Failure streak | Stats |
| --- | --- | --- | --- |
| `first_token` / `completed(ttft_ms=...)` / `observe(ttft_ms=...)` | +1 TTFT sample | reset | window percentiles move |
| `failed(kind="health")` before first token | +1 penalty sample (`penalty_ms`) | +1 | `errors["health"][code]` |
| `failed(kind="health")` after first token | none (TTFT already booked) | +1 | `errors["health"][code]` |
| `failed(kind="request")` any time | none | none | `errors["request"][code]` |
| `observe(kind="health", code=...)` | +1 penalty sample | +1 | `errors["health"][code]` |
| `observe(kind="request", code=...)` | none | none | `errors["request"][code]` |
| `cancelled()` / `declined()` | none | none | hedge counters where applicable |
| unknown `code` | per its `kind` | per its `kind` | aggregated under `"other"` |

### Table D: Public Signatures and Validation

Provider identity parameters and outputs are `str` names everywhere; the
`Router` constructor is the one signature that takes `Provider` objects.
Public mappings (`weights`, `trace`, stats structures) are deeply immutable. All
token counts are non-negative integers; all prices, costs, and latencies are
finite non-negative floats. Violations raise `ValidationError` unless noted,
and validation or outcome-conflict failures commit no business state. Capacity
exhaustion follows the staged-transaction rule described above.

| Callable | Rule | Error |
| --- | --- | --- |
| `Provider(name, price_in, price_out, price_cached=None)` | non-empty name; finite non-negative prices | `ValidationError` |
| `Router(providers, *, alpha=0.25, slo_ms=None, seed=None, cold_start="explore", clock=None, tuning=None)` | ≥1 provider; unique names; `alpha ∈ [0,1]`; `slo_ms > 0` when given; `cold_start ∈ {"explore","require_observations"}`; `clock` a zero-arg callable returning finite float seconds (effective time is clamped non-decreasing); `seed` int or None | `ValidationError` |
| `router.route(*, input_tokens, estimated_cached_tokens=0, alpha=None, exclude=())` | `input_tokens ≥ 0`; per-call `alpha ∈ [0,1]`; mapping form: missing name → 0, unknown name → error, each value clamped to `input_tokens` (matching core's cached-token cap); `exclude` names must exist; empty eligible set → error | `ValidationError`; `NoProviderError` |
| `router.observe(provider, *, ttft_ms=None, kind=None, code=None)` | known provider; exactly one of `ttft_ms` / `kind`; `code` only with `kind`; `kind ∈ {"health","request"}` | `ValidationError` |
| `attempt.first_token(*, ttft_ms, adopted=None)` | once, before terminal; `adopted ∈ {True, None}` | `OutcomeError` |
| `attempt.completed(*, output_tokens=None, ttft_ms=None, cached_tokens=None, cost_usd=None, adopted=None)` | one terminal per attempt; one adopted per decision; no post-hoc adoption; `adopted ∈ {True, None}` | `OutcomeError` |
| `attempt.failed(*, kind, code=None)` | `kind` required and valid | `ValidationError`; `OutcomeError` |
| `attempt.cancelled()` | terminal once | `OutcomeError` |
| `attempt.declined()` | legal only from `pending` | `OutcomeError` |
| `attempt.settle(*, output_tokens=None, cached_tokens=None, cost_usd=None)` | any terminal state except `declined`; write-once per field; no adoption flag | `OutcomeError` |
| `decision.hedge_now(*, elapsed_ms)` | `elapsed_ms` finite, ≥ 0; returns `None` or the single backup `Attempt` | `ValidationError` |
| `Tuning(hedge_target=0.99, penalty_ms=60000.0, window_min=15, cooldown_sec=30.0, cooldown_after=3, hedge_min_samples=5, exploration_lease_sec=60.0)` | `hedge_target ∈ (0,1]`; `penalty_ms > 0`; `window_min > 0`; `cooldown_sec ≥ 0`; `cooldown_after ≥ 1` int; `hedge_min_samples ≥ 1` int; `exploration_lease_sec > 0` | `ValidationError` |
| `Candidate(name, cost_usd, latency_ms)` | non-empty name; finite non-negative values | `ValidationError` |
| `route_once(candidates, *, alpha, seed=None, rng=None)` | candidates non-empty with unique names; `alpha ∈ [0,1]`; `seed` and `rng` mutually exclusive; `rng` exposes `random() -> float` in [0,1) | `ValidationError` |

`route_once()` warning, part of the contract: with a fixed `seed=`, identical
inputs replay the identical draw, so a service loop that passes a constant
seed stops realizing the mixture over time and silently loses the long-run
budget-mix guarantee; each single call's LP solution and budget are
unaffected. Pass a reusable `rng=random.Random(...)` for long-lived use, or
treat the function as a one-shot tool.

## API Reference

### Provider

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | `str` | identifier your execution layer resolves; unique per router |
| `price_in` | `float` | input price, USD per million tokens |
| `price_out` | `float` | output price, USD per million tokens |
| `price_cached` | `float \| None` | discounted price for cached prefix tokens; `None` means no discount (cached tokens bill at `price_in`) |

### Router

```python
Router(providers, *, alpha=0.25, slo_ms=None, seed=None,
       cold_start="explore", clock=None, tuning=None)

router.route(*, input_tokens, estimated_cached_tokens=0,
             alpha=None, exclude=()) -> Decision
router.observe(provider, *, ttft_ms=None, kind=None, code=None) -> None
router.stats() -> StatsSnapshot
```

`estimated_cached_tokens` is one `int` applied to every provider, or a
mapping `{provider_name: tokens}` when cache state differs per provider,
which it usually does: each provider's prefix cache warms independently, and
on multi-turn agent workloads cache discounts can outweigh list-price
differences in deciding which provider is cheapest. A name missing from the
mapping means 0; an unknown name raises `ValidationError`; values are clamped
to `input_tokens`. `Router` is thread-safe: `route()`, outcome methods,
`observe()`, and `stats()` may be called from any thread.

### Attempt

A handle for a potential dispatch. Returned by `hedge_now()`.

| Member | Kind | Meaning |
| --- | --- | --- |
| `provider` | attribute | where to dispatch this attempt, if you accept it |
| `first_token(*, ttft_ms, adopted=None)` | method | the stream started; one TTFT sample; `adopted=True` takes this attempt's response |
| `completed(*, output_tokens=None, ttft_ms=None, cached_tokens=None, cost_usd=None, adopted=None)` | method | finished; billing fields optional, settle later; the winner's first known positive `output_tokens` trains the length estimator |
| `failed(*, kind, code=None)` | method | `kind="health"`: penalty and cooldown; `kind="request"`: stats only; after `first_token`, no second latency entry |
| `cancelled()` | method | dispatched, then aborted; no penalty, no spend claim |
| `declined()` | method | never dispatched; closes the handle; legal only from `pending` |
| `settle(*, output_tokens=None, cached_tokens=None, cost_usd=None)` | method | fill still-unknown billing fields after any terminal state except `declined`, each exactly once |

### Decision

The logical request, returned by `route()`. Exposes the same outcome methods,
which apply to its primary attempt, plus the decision context.

| Member | Kind | Meaning |
| --- | --- | --- |
| `provider` | attribute | the sampled primary provider; send the request here |
| `weights` | attribute | the underlying mixture (at most two nonzero entries) |
| `expected_cost_usd` | attribute | predicted expected cost of this decision |
| `expected_latency_ms` | attribute | `float \| None`: expected TTFT, `None` when the mixture includes an unprofiled provider |
| `checkpoints_ms` | attribute | hedge re-evaluation times; empty without an SLO |
| `hedge_now(*, elapsed_ms)` | method | returns the decision's one backup `Attempt`, or `None` |
| `explain()`, `trace` | method, attribute | human-readable and structured explanations |

### Tuning

```python
Tuning(*, hedge_target=0.99, penalty_ms=60000.0, window_min=15,
       cooldown_sec=30.0, cooldown_after=3, hedge_min_samples=5,
       exploration_lease_sec=60.0)
```

### StatsSnapshot

An immutable snapshot of the counters and spend-provenance fields described
in Observability, following the Table B migration. Counter lifetimes remain
part of Open Question 7; the runtime type itself is public, exported at the
top level, and stable once that question is signed off.

### Stateless Function

```python
from routewise import Candidate, route_once

result = route_once(
    [Candidate("fireworks", cost_usd=0.0012, latency_ms=240.0),
     Candidate("together",  cost_usd=0.0008, latency_ms=410.0)],
    alpha=0.25, rng=my_rng,
)
result.provider     # the sampled provider
result.weights      # the mixture it was drawn from
result.budget_usd   # the request budget the LP ran under
```

`route_once()` is a stateless function for callers that already track cost
and latency: it prices nothing and learns nothing, solving only the budget LP
and sampling a provider. It returns an immutable `RouteOnceResult` with
exactly the three fields above, not a `Decision`: there is no router state
behind it, so there is nothing to report outcomes to. `Candidate(name,
cost_usd, latency_ms)` is an immutable value type; eligibility filtering is
the caller's job here, since the caller supplies the candidates. Determinism
comes from `seed=`; long-lived callers pass a reusable `rng=` instead (see
the Table D warning). `Router.route()` is internally this function plus the
estimators and state described above. Research users start here, with the
caveat that the API-only preview covers the paper's on-demand routing, not
its quota and concurrency results.

## Managed Execution: Client (Planned, Later Release)

`Router` decides; you execute. For callers who would rather RouteWise execute
too, a later release plans `Client`: a router per model behind an
OpenAI-compatible surface. The design is retained from revision 1; it is
deferred so that the initial release ships dependency-free, and because
racing hedged streams and cancelling losers correctly is real engineering
that should not gate the decision core. Pending Juncheng's sign-off.

```python
client = Client(
    {"deepseek-v4": [Provider("fireworks", ..., base_url=..., api_key=...),
                     Provider("together",  ..., base_url=..., api_key=...)]},
    alpha=0.25, slo_ms=3000,
)
response = await client.chat.completions.create(model="deepseek-v4", messages=[...])
print(response.routewise.provider, response.routewise.cost_usd)
```

`create()` mirrors the OpenAI SDK signature, including `stream=True`. Token
counting, outcome reporting, adoption, hedge racing, and loser cancellation
happen inside the client. `base_url` and `api_key` on `Provider` are used by
`Client` only. For a `Router` user who wants several models, the same pattern
is one line of their own code:
`routers = {m: Router(pool[m], alpha=0.25) for m in pool}`.

## Design Principles

Decisions stay separate from execution. The core opens no connection, which
keeps it dependency-free and orthogonal to any HTTP client or async framework.
Cost control uses one dimensionless knob, so no deployment-specific dollar
threshold appears in the interface. A decision is a handle, so reporting an
outcome repeats no request identity. Outcomes are typed because the ways an
attempt can end mean different things: a health failure indicts the provider,
a request failure indicts the request, and a cancellation or declination
indicts nobody; collapsing them into one error flag would let one mistaken
call write a 60-second penalty. Effects are decided by `kind`, labels by
`code`, so judgment stays with the caller and metrics stay bounded. Lifecycle
and settlement are separate because timing truth and billing truth arrive at
different moments, and settlement is write-once because estimates never enter
realized spend, so there is never an actual value to correct. Adoption and
outcome are separate because taking a response and that response succeeding
are different facts. The router claims only what it can observe: selections,
offers, and your reports, never dispatch. Every extra cost or risk the
library introduces (hedge offers, exploration decisions, unresolved
adoptions) surfaces in `stats()` rather than disappearing into the bill. The
router is thread-safe behind one coarse lock, holds no strong references to
outstanding handles, and reads one injectable monotonic clock.

## Scope and Roadmap

Subscription-style providers (prepaid request quotas, reserved concurrency
slots) are the second half of the RouteWise algorithm and the planned v2
extension; the paper's headline results depend on them, which is why `0.3.0`
is named an API-only preview rather than a paper artifact. Their pricing
mathematics already exists in `routewise.core` (the quota shadow price and
its L/U scarcity calibration). What v2 cannot avoid is capacity lifecycle: a
quota slot is spent, a concurrency slot is held and released, so the
interface will grow reservation interactions (reserve before a dispatchable
attempt is returned, commit when dispatch starts, and close or release at
outcome) that have no v1 counterpart. The v1 surface is designed to survive
that growth additively (new `Provider` constructors and new optional
interactions rather than changed signatures), but this revision withdraws
revision 1's promise that the interface shape would not change.
`routewise.core`'s `ProviderView` remains the seam for advanced integrations;
hybridInference already runs quota and concurrency capacity in production
against those primitives.

### Capacity Extension Seam

Capacity changes the transaction around a routing decision, not the LP itself.
A capacity-aware facade constructs request-bound candidate snapshots, asks the
pure core to solve and sample, then atomically tries to reserve the sampled
provider. A failed reservation means the snapshot was stale or capacity was
contended: for this routing transaction, the facade excludes that candidate,
recomputes the eligible-set cost bounds and budget, and solves again with a
bounded retry count. It does not report a provider error, write a latency
penalty, or advance cooldown.
The returned decision's weights, budget, and trace describe the final solve
whose sampled provider was successfully reserved; the trace also records
capacity exclusions and the replan count.

Primary and hedge attempts own independent reservations. In a future
capacity-backed release, `Attempt.started()` (or the execution adapter's
equivalent hook) performs an atomic commit-if-still-owned immediately before
network dispatch. If the reservation expired or was fenced out, commit fails,
no I/O may begin, and the execution layer starts a fresh routing transaction;
a managed adapter does this automatically. The superseded attempt closes as
`declined` — it was never dispatched — and contributes neither a latency
sample nor a provider-health failure; the exact manual API and exception
belong to the future capacity release. A terminal `completed`, `failed`, or
`cancelled` call closes a committed reservation, and `declined()` releases an
uncommitted one. For API providers the controller is a no-op. A quota
reservation consumes its unit at commit and does not refund it on close; a
concurrency reservation holds a slot until close or lease expiry. A quota
reservation also binds to a quota-window epoch so reset races have a defined
owner. Distributed controllers additionally need idempotency keys, leases or
renewal, and fencing so a stale process cannot release a newer reservation
after expiry.

Hedge reserve failure is handled inside the same `hedge_now()` call: the facade
temporarily excludes the contended backup, re-runs backup selection over the
remaining eligible candidates, and tries a bounded number of reservations.
The one-backup slot is consumed only after one reservation succeeds and an
`Attempt` is returned. If none succeeds, `hedge_now()` returns `None`,
publishes the capacity exclusions in a fresh immutable `decision.trace`
snapshot, and consumes no slot.

The initial `0.3.0` public surface remains API-only and does not expose
`CapacityController`, `Reservation`, or `Attempt.started()`. It nevertheless
uses the same private orchestration with `_NoopCapacityController`, so a later
capacity release can add implementations and dispatch interactions instead of
rewriting `Router`. When capacity-backed providers become public,
`Attempt.started()` is required before I/O for those attempts and managed
execution adapters call it automatically; keeping it a no-op for API
providers preserves the existing quickstart.

Deferred to a later release: the `Client` execution layer, the LiteLLM
strategy plugin, historical observation import (the successor to
`observe(at=)`), and cross-process state (an `export_state` / `state=`
pair). The initial release learns in-process; independent replicas each learn
from their own traffic, and replaying a peer's recent measurements through
`observe()` is the interim sharing mechanism. An end-to-end latency objective
is roadmap (v1 optimizes TTFT). Model selection stays out permanently, since
RouteWise routes a fixed model across providers.

## Open Questions

These block the contract freeze.

1. Does `Client` ship in the initial release (`0.3.0`) or a later one? The
   draft defers it. If it moves into `0.3.0`, the contract must also add
   `base_url` and `api_key` to `Provider`, export `Client` at the top level,
   and define the HTTP optional dependency in the installation surface.
   (Juncheng)
2. The recommended `code` list for `failed()`/`observe()` (behavior is fixed
   by `kind`; the list only bounds metrics labels).
3. Cooldown details: should a failure streak expire with the profile window,
   and is the organic half-open mechanism (penalties keep a returning
   provider unattractive) enough, or is an explicit trial state needed?
4. `cold_start`: Router argument (as drafted) or `Tuning` field, and is
   `"explore"` the right default for a library that may take production
   traffic on first deploy?
5. Exploration cadence: re-arming whenever the window is empty is drafted,
   with one exploration lease per provider, released by the provider's first
   window event or after `exploration_lease_sec` (default 60). Does a
   flapping provider need a rate cap on re-exploration, and is 60 seconds
   the right lease horizon?
6. Are `decision.primary` and `decision.backups` public attributes, or do
   outcome methods proxy silently (as drafted)?
7. The `StatsSnapshot` schema above is the freeze candidate; confirm the
   field list and counter lifetimes (router lifetime versus rolling window).
   The spend provenance ladder and the calculated→actual migration are
   settled in Table B.
8. `hedge_min_samples=5` borrows the research harness's warmup validation
   threshold; the per-provider hedging gate itself is new library policy with
   no production counterpart. Confirm both the mechanism and the value.
9. Should `route()` accept `estimated_output_tokens=` so callers with a
   known `max_tokens` can override the length estimator, whose fallback
   default is 500 tokens?
10. Does the wheel keep the research subpackages (`routewise.sim`,
    `routewise.offline`, `routewise.metrics`, plus the shared research
    contracts `routewise.capacity` and `routewise.schemas`) or ship the
    facade and core alone (Table E drafts the narrow allowlist)? Excluding
    them keeps the "library alone" promise; keeping them costs size but
    spares research users a source checkout. Keeping them also breaks the
    "`0.3.0` defines no extras" stance: the research subpackages pull
    scientific dependencies, so they would need a `[sim]`-style extra or a
    redefined dependency policy.

## Implementation Notes

This section addresses RouteWise developers; library users can stop above.

Two different cost ranges appear in RouteWise and must not share a name. The
**request cost bounds** `c_min`/`c_max` are computed per request over the
eligible providers; every deployment has them, API-only included. The **L/U
scarcity calibration** prices quota and concurrency scarcity; API-only fleets
never construct it. Core's `RouteWiseRouter.cost_envelope` parameter is the
L/U pair, not the request bounds; the facade never exposes it in v1.

Capacity admission and scarcity pricing are separate internal concerns.
`_ScarcityCalibrator` maintains workload-level L/U inputs and effective-cost
state; a capacity controller exposes only admission state and atomic
reservation. `ProviderView` remains a pure, request-bound snapshot with no
side-effecting reserve method. The initial private seam is intentionally small:

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

`resource_key` names the atomic capacity domain, which may be one provider or
a pool shared by several provider endpoints. Snapshots are scoped to that key,
and attempt IDs are globally unique, so a replan cannot ambiguously address a
reservation created for another candidate.

The reservation state machine is:

```text
RESERVED --commit(success)--> COMMITTED --release/finish/expire--> CLOSED
    |
    +--release/expire/commit-lost-------------------------------> CLOSED
```

Every transition is idempotent and keyed by the attempt identity. `commit()`
atomically succeeds only while the caller still owns the live reservation and
returns `False` after expiry or fencing loss. Before commit, `release()`
cancels and returns the reservation. After commit, its meaning is
controller-specific: concurrency returns the slot, while quota only closes
the durable reservation record and never refunds the consumed unit. A
committed concurrency lease may also expire to `CLOSED` for crash recovery.
Distributed implementations need a lease generation or fencing token; an
optional renewal capability may extend that lease for a long request.

The facade owns the side-effecting orchestration around the pure core:

```text
build candidate snapshots
        -> solve LP and sample
        -> try_reserve(sampled provider)
        -> on failure: exclude, recompute bounds/budget, and re-solve
        -> on success: attach reservation and return Decision/Attempt
        -> at dispatch start: commit-if-owned
        -> on commit loss: close as declined (no health/latency impact),
           do no I/O, and start a fresh routing transaction
        -> at terminal outcome: release/finish
```

The primary retry loop is bounded and reserves only one sampled candidate at a
time, so it holds no cross-provider lock while solving. `hedge_now()` uses its
own bounded exclude/reselect loop and independently owned backup reservation,
as described above. The no-op API controller always succeeds on the first
attempt.

The interface maps onto `routewise.core` as follows: `route_once()` wraps
`solve_budget_lp()` with the alpha-to-budget conversion and seeded weight
sampling; `Router` adds the rolling latency profile (`LatencyBeliefs` over
`RollingLatencyProfile`), the bucket-mean output-length estimator, cooldown,
cold-start handling, and the attempt bookkeeping; `hedge_now()` wraps
`hedge_checkpoints_for_slo()`, `combined_success_probability()`, and
`select_probability_backup()`.

Gaps between this contract and today's `routewise.core`, facade-level except
the last:

1. Core's default sampler is `argmax_weight_sampler`. The facade samples from
   the LP weights with its seeded RNG; the core default stays as is, because
   the research harnesses inject their own RNG discipline.
2. Cooldown, the `kind`/`code` error model, cold-start modes, and the
   exploration mixture do not exist in core, where availability is adapter
   business. The facade owns them.
3. The bucket-mean estimator lives in
   `experiments/offline_stage/value_estimators/bucket_mean.py` and imports
   experiment types. The facade needs the dependency-free equivalent inside
   the package.
4. `RouteWiseRouter` refuses to route without the L/U pair, but an API-only
   fleet never uses it (`effective_cost("api")` returns the request cost
   unchanged). Core should require it only when a non-API tier is present.
5. `LatencyBeliefs` is not thread-safe; profile queries mutate window
   bookkeeping. The facade serializes all access behind its lock.
6. Attempt bookkeeping (adoption, write-once settlement with the Table B
   migration, the price snapshot per attempt, the one-latency-entry rule,
   the hedge slot and its closure on the primary's first token, and the
   exploration lease with a generation token so a stale attempt's late event
   cannot release a newer lease) has no core counterpart; it is new facade
   state held by the handles themselves, so the router keeps no strong
   references.
7. The private capacity seam, no-op controller, reservation state machine, and
   bounded reserve/replan loop are new facade orchestration. They stay out of
   `routewise.core`, and the capacity Protocols are not part of the `0.3.0`
   compatibility surface.
8. `combined_success_probability` on this implementation branch now applies
   the specified survival-zero fallback (backup-only probability); a focused
   core regression test protects the boundary.

### Table E: Wheel Allowlist and Release Gates

Wheel allowlist (freeze candidate, confirmed by the installed-wheel import
test): `routewise` top level (facade modules, `py.typed`), `routewise.core`,
`routewise.const`, and the dependency-free estimator module. The current
preview build provisionally retains the stdlib-only shared contracts
`routewise.capacity` and `routewise.schemas` while Open Question 10 remains
open. It excludes `experiments/`, `plots/`, `scripts/`, all data files, and
the research subpackages `routewise.sim`, `routewise.offline`, and
`routewise.metrics`. The `[project.scripts]` CLI entry point does not ship in `0.3.0`:
the current `routewise_cli.main` imports `experiments` at module top, so the
console command is removed from the wheel until a library-only CLI exists.
`0.3.0` defines no pip extras; `[client]` and `[litellm]` arrive with their
features.

| Release gate | Requirement | Status today |
| --- | --- | --- |
| Wheel contents | allowlist above; no experiments, CLI, or data | passes locally: 25 entries; exact-member checker green |
| Console script | absent from `0.3.0` | passes: entry point removed; source CLI is repository-only |
| Type marker | `py.typed` in the wheel | passes |
| Exports | top level exports exactly `Provider`, `Router`, `Decision`, `Attempt`, `Tuning`, `Candidate`, `route_once`, `RouteOnceResult`, `StatsSnapshot`, `RouteWiseError`, `ValidationError`, `NoProviderError`, `OutcomeError` | passes |
| Install test | clean-environment install + import + `route_once` smoke, in CI, per release | passes locally on Python 3.10–3.14; CI workflow added |
| Test suite | fast tests green on a clean checkout | passes locally: 640 passed, 12 explicitly skipped without optional BurstGPT data, 3 slow tests deselected |
| CI | 3.10–3.14 matrix (pyproject declares `>=3.10` with no upper bound and 3.14 is the current feature series), lint, wheel build | passes on PR #13 with separate dependency-free and research-compatibility jobs |
| PyPI release | published GitHub Release, exact `v<version>` tag, protected `pypi` environment, OIDC Trusted Publishing | workflow added; PyPI publisher and GitHub environment still require one-time configuration |
| Metadata | `version = 0.3.0`; library description; README library-first; `[project.urls]`, classifiers, SPDX license; arXiv citation resolved | partial: everything except the arXiv citation is present |
| PyPI project/version | existing team-owned `routewise` project; unused `0.3.0`; Trusted Publisher configured | project contains the legacy hosted SDK through `0.2.0`; `0.3.0` is selected because uploaded versions cannot be reused; ownership and publisher setup still require confirmation |
| Working tree | implement from a clean worktree off `origin/main`, not the diverged local `main` | passes: `codex/api-provider-library-v1` in an isolated worktree |

Behaviors that are contractual, not incidental: primary selection samples from
the LP weights (an argmax would break the budget guarantee); latency ties
break toward the cheaper provider (`cost_tiebroken_objective()`, applied
internally so callers never see it); one router binds to one model, which
removes the model argument from `route` and `Provider`; a cancelled or
declined attempt never writes a penalty; one attempt books at most one latency
entry; billing fields are write-once and estimates never enter realized
spend; each reporting call applies exactly one atomic aggregate delta; the
output-length estimator trains only on winners (adopted and completed); with
multiple attempts, adoption exists only by explicit declaration; the hedge
slot closes at the primary's first token, at adoption, or at termination;
every decision's predicted expected cost, exploring or not, respects the
request budget; exploration-mixture decisions are tagged whenever `q > 0`,
and a `q = 0` request routes by the plain LP untagged; `expected_latency_ms`
is `None` rather than a sentinel (never core's `1e9` avoidance value) when
the mixture includes an unprofiled provider; the survival-zero fallback
hedges on the backup's probability alone; and the request cost bounds are
computed over the eligible set of the current request. The L/U scarcity
calibration, quota state, and concurrency state appear nowhere in this
interface. With API providers only, the effective cost of a request equals
its estimated metered cost, so the calibration that the quota shadow price
needs (see [CORE_API.md](CORE_API.md)) is never constructed. Capacity
reservation failures are feasibility events rather than provider-health
failures: they trigger bounded replan without latency penalty or cooldown,
and the final decision snapshot reflects the successful solve.
