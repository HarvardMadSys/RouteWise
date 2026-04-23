# Phase 6 Dispatch De-duplication Plan

Date: 2026-04-23

## 1. Motivation

The phase 6 shared-prober coordinator sends **one real API call per
(policy, trace request)** pair. With 11 policies this is a constant 11x
multiplier on real provider load relative to the trace's logical
request rate. In our smoke runs on 2026-04-23 this caused two concrete
problems:

1. **Self-induced 429 rate limits on heavily-selected providers.** More
   than half of the client-decided policies pick Friendli as primary
   (lp_mix, smart_hedge, fastest_fixed, budget_vhat_t75_hedge,
   budget_vhat_t75_hedge_explorer). Friendli received 5-7x the real load
   per trace request, tripping its per-API-key rate limit on qwen3-235b
   and deepseek-v3.2 traces.
2. **Unnecessary cost.** OpenRouter billed for every duplicate call
   even when all callers were going to observe the same outcome.

De-duplicating dispatch across policies that pick the same provider
eliminates both problems without compromising paired fairness.

## 2. Two Classes of Policies

The de-duplication is valid only when the provider selection is known
to the client before the API call. This splits our 11 policies into
two groups.

### 2.1 Client-routed (deduplicable)

| Policy | Primary selector |
|---|---|
| `lp_mix` | Client LP over `phase5.router.profiles` |
| `smart_hedge` | Client LP + hedge |
| `budget_vhat_t75` | Client budget-constrained LP |
| `budget_vhat_t75_hedge` | Budget LP + hedge |
| `budget_vhat_t75_hedge_explorer` | Budget LP + hedge + explorer feedback |
| `cheapest_fixed` | Client constant (min cost per request) |
| `fastest_fixed` | Client constant (min p50 profile) |

For client-routed policies the primary provider is decided in the
process before any network call. Multiple policies whose primary is
the same provider can share a single real request.

### 2.2 Server-routed (non-deduplicable)

| Policy | Primary selector |
|---|---|
| `openrouter_auto` | OpenRouter server-side load balancer |
| `sort_price` | OpenRouter server-side, sorted by price |
| `sort_throughput` | OpenRouter server-side, sorted by throughput |
| `sort_latency` | OpenRouter server-side, sorted by latency |

For server-routed policies the client sends the request with a
`provider: {sort: ...}` hint but does not know which concrete provider
OpenRouter will pick. The chosen provider is observed only from the
response's `actual_provider` field. These policies each issue one real
request per trace request and cannot be de-duplicated.

## 3. Dispatch Phases per Trace Request

Given a trace request `r_t`:

**Phase 1 - gather client decisions.** For each client-routed policy,
compute the primary provider via the policy's selector without sending
any network call.

**Phase 2 - de-duplicated client primary dispatch.** Group
client-routed policies by primary provider. For each unique primary
provider, spawn one thread that issues exactly one real API call.

**Phase 3 - server-routed dispatch.** For each server-routed policy,
spawn one independent thread that issues one real API call.

**Phase 4 - per-policy hedge.** Each hedge-capable policy
(`smart_hedge`, `budget_vhat_t75_hedge`,
`budget_vhat_t75_hedge_explorer`) evaluates its own hedge trigger
against its shared primary result. If triggered, the policy spawns its
own backup request thread. Backup is not de-duplicated because:

- backup providers differ per policy (different profiles, different
  selectors)
- hedge rate is low (~10-20% in practice)
- backup dedup would require cross-policy coordination that complicates
  the race logic for marginal gain

## 4. Shared Result Broadcast

When a client-routed primary is shared across `k` policies, the real
API call produces one `SingleRequestResult`. This result is broadcast
as follows:

- `ttft_ms`, `e2e_ms`, `status`, `cost_usd`, `error_message`,
  `actual_provider`, `prompt_tokens`, `completion_tokens` are copied
  unchanged.
- Each policy's per-policy accounting (`policies[p].results`,
  `total_cost`, `error_count`) still appends the shared result as its
  own logical record. The `real_cost_count` and `estimated_cost_count`
  per policy remain accurate because only one policy's `total_cost`
  owns the real billing; the other `k-1` policies record a logical
  cost but not a second OpenRouter charge. Accounting note: we record
  the same `cost_usd` for all `k` policies as their reported cost
  (this is the cost they would pay in production), but only the
  leader's thread actually increments `observation_logger`-tracked
  billing.
- Per-policy router state update (`_update_router_from_result`) is
  called once per policy, so policy-local profiles still learn from
  the shared observation.
- `evaluation_log.csv` in each policy's directory still contains one
  row per trace request, as expected by downstream analysis scripts.

## 5. Fairness Property

Paired comparison is preserved and strengthened. Before dedup, two
policies that picked the same provider observed independent samples
from that provider's latency distribution and could disagree on the
TTFT for `r_t` purely by variance. After dedup, they observe the
exact same TTFT for `r_t`. Systematic differences between policies
remain: policies that picked different providers still observe
independent samples. The difference of interest -- which policy's
selection produced a better outcome -- is unchanged.

## 6. New Per-Request Logging Fields

Each row in `evaluation_log.csv` gains:

- `is_dedup_leader` (bool): true if this policy's thread executed the
  real API call.
- `dedup_primary_share` (int): number of client-routed policies that
  chose this same primary for `r_t` (1 if not shared).
- `dedup_broadcast_source` (string | empty): name of the leader policy
  if this row is a broadcast recipient, empty otherwise.

`shared_observation_log.csv` is unchanged (it already records only
warmup and probing events, not trace requests).

## 7. Paper Framing for Section 5 Setup

Proposed text for the evaluation setup when we describe the paired
online measurement:

> Within a single-model run, all policies are dispatched in parallel on
> the same trace request, observing identical provider state at the
> time of decision. Server-routed policies (OpenRouter Auto, sort=*)
> each issue one real request per trace request because the provider
> selection is performed server-side. Client-routed policies (LP-mix,
> Smart Hedge, Budget LP, Cheapest/Fastest Fixed) that select the same
> provider for a given trace request share a single real API call,
> with the observed TTFT, status, and cost broadcast to all sharing
> policies. This dispatch de-duplication eliminates redundant load on
> heavily-selected providers (e.g., Friendli) without compromising
> paired fairness, because policies that chose the same provider would
> -- in a no-dedup harness -- have each sampled independently from the
> same provider's latency distribution. Hedge backups, when triggered,
> are dispatched per policy.

## 8. Scope and Explicit Non-Goals

In scope:

- Client-routed primary de-duplication.
- Cost and observation accounting that preserves paired comparison.
- Paper-facing logging fields for transparency.

Out of scope for this change:

- De-duplicating hedge backups across policies.
- De-duplicating across models (models already run sequentially).
- Any change to warmup or probing (already shared).
- Any change to server-routed policies or the SDK layer.

## 9. Expected Quantitative Impact

Based on smoke observations (16 req per model):

| Metric | Before dedup | After dedup (estimate) |
|---|---|---|
| Real API calls per trace req | 11 | 6-8 |
| Friendli load per trace req | 5-7x | 1-2x (client) + 1-2x (server) |
| Redundant cost fraction | ~40-50% | 0% |

These numbers grow more favorable on longer traces, where the same
heavy-provider skew compounds.

## 10. Implementation Sketch

Primary changes land in `experiment/scripts/phase6_sa_online_evaluation.py`:

1. Split `SUPPORTED_POLICIES` into `_CLIENT_ROUTED_POLICIES` and
   `_SERVER_ROUTED_POLICIES` sets.
2. Add `_select_client_primary(policy, now, req)` that returns the
   primary provider for client-routed policies without dispatching.
3. Refactor `run_trace_replay_shared_probing` into the four phases
   above.
4. Extend `_log_result` to accept the three new logging fields.
5. Update `save_combined_summary` header to include the new fields so
   downstream analysis has column compatibility.
