# Schema Unification (H6)

> Decision document. Defines the delta from the current `PerRequestRecord`
> to a schema that SIM, REAL-EVAL, and PROD all populate. Scoped to record
> contract only — capacity/concurrency model (H5) is handled by prior commits
> and out of scope here.

Last updated: 2026-05-21.

---

## 1. TL;DR

SIM and REAL-EVAL already share a `PerRequestRecord` and a `Run` aggregate:

- `rwsim/metrics/record.py:20` — `PerRequestRecord` dataclass
- `rwsim/metrics/run.py:103` — `Run.records: list[PerRequestRecord]`
- `experiments/real_evaluation/recorder.py:381` — REAL-EVAL already constructs `PerRequestRecord`

What is missing for full three-way alignment:

1. New identity / LP / hedging fields on `PerRequestRecord`.
2. `total_cost_usd` / `primary_cost_usd` / `backup_cost_usd` keep their role
   as the **canonical accounting cost** read by default metrics; two optional
   debug/ops cost lines (`routing_estimated_*`, `physical_*`) sit alongside.
3. Promoting existing real-eval `metadata["real_*"]` LP / physical-cost keys to canonical.
4. PROD writes the same routing detail into `api_logs.metadata["routewise"]`,
   without schema migration.

H5 (simulator capacity and concurrency model) is already resolved by
commits `c82d11f`, `e8729f0`, `5d0b708`, `3400ead`. The earlier
"core purity / CapacitySnapshot" framing is dropped.

---

## 2. Current State

### 2.1 SIM and REAL-EVAL share `PerRequestRecord`

`rwsim/metrics/record.py:20–58` defines 26 fields (identity, workload, routing,
latency, cost, hedging, status, metadata). REAL-EVAL constructs the same
dataclass at `experiments/real_evaluation/recorder.py:381`, with the extra
real-eval-only diagnostics tucked under `metadata["real_*"]`.

Today the record carries one cost line: `total_cost_usd`, `primary_cost_usd`,
`backup_cost_usd`. REAL-EVAL fills these with billed cost
(`recorder.py:402: total_cost_usd=billed_cost`). SIM fills them with its
simulated accounting cost. The schema below keeps this line as the canonical
accounting cost — each source is responsible for filling it with its most
authoritative "what this request cost" number — and adds two optional lines
for LP debug (`routing_estimated_*`) and provider reconciliation
(`physical_*`).

### 2.2 PROD has none of this

`apps/backend/serving/storage/database.py:73–110` (plus the `ALTER TABLE`
migrations at 196–204) defines `api_logs`. It has `cost_usd`,
`upstream_cost_usd`, and a `metadata JSONB` column. It has no routing-level
fields: no `policy`, no `primary_provider`/`backup_provider` split, no
`hedge_*`, no `lp_*`, no `slo_*`.

### 2.3 What needs to change

1. Extend `PerRequestRecord` with the missing canonical fields.
2. Promote real-eval's existing `metadata["real_*"]` LP / cost fields to
   those canonical fields.
3. PROD writes a `routewise` namespace into `api_logs.metadata` carrying the
   same canonical field set.

---

## 3. `PerRequestRecord` Delta

### 3.1 Add identity, LP, and hedging fields

```python
# Identity extension
model: str | None = None
source: Literal["sim", "real_eval", "prod"] | None = None
timestamp_sec: float | None = None        # wall-clock epoch, request received

# LP state at routing time
lp_weights: dict[str, float] | None = None
lp_budget_usd: float | None = None
lp_status: Literal[
    "feasible",
    "single_candidate",
    "all_over_budget",
    "no_candidates",
] | None = None

# Hedging algorithm + schedule
hedge_algorithm: Literal[
    "probability_target",   # canonical RouteWise hedge algorithm
    "disabled",
] | None = None
hedge_schedule: Literal[
    "slo_relative_checkpoints",  # hedge_checkpoints_for_slo()
] | None = None
```

`source` is already tracked on `Run` (`run.py:112`); it goes on the record so
merged cross-source datasets stay unambiguous.

### 3.2 Cost lines: one canonical accounting, two optional debug/ops

```python
# Canonical accounting cost
# Used by default metrics, paper plots, and dashboards.
# total = primary + backup when backup exists.
total_cost_usd: float = 0.0
primary_cost_usd: float = 0.0
backup_cost_usd: float | None = None

# RouteWise decision-time cost estimate (optional)
# Used for LP / routing debug, not default metrics.
routing_estimated_cost_usd: float | None = None
primary_routing_estimated_cost_usd: float | None = None
backup_routing_estimated_cost_usd: float | None = None

# Upstream/provider-side physical cost (optional)
# Used for ops, margin, provider reconciliation.
physical_cost_usd: float | None = None
primary_physical_cost_usd: float | None = None
backup_physical_cost_usd: float | None = None
```

Three cost lines, one default. Each source defines `total_cost_usd` as its
most authoritative "what this request cost" number — there is no ambiguity
because the meaning is fixed per source:

| Source | `total_cost_usd` | `routing_estimated_cost_usd` | `physical_cost_usd` |
|---|---|---|---|
| SIM | simulated accounting cost | decision-time LP estimate (may equal `total_cost_usd`) | `None` |
| REAL-EVAL | `billed_cost_usd` (user-billed) | `primary_routing_estimated + backup_routing_estimated` | observed upstream/physical cost |
| PROD | `api_logs.cost_usd` | `metadata["routewise"]["routing_estimated_cost_usd"]` | `api_logs.upstream_cost_usd` |

`Run` aggregate methods (`mean_cost_usd`, `total_cost_usd`, `cost_by_tier`)
read `total_cost_usd` directly. No `cost_basis` parameter; cross-source
plots compare the same field unchanged. Debug code reads
`routing_estimated_*` to ask "what did the LP think it would cost?"
Operations / margin code reads `physical_*` to reconcile with provider
bills.

### 3.3 Field semantics

- **`timestamp_sec`**: wall-clock epoch seconds of request received. SIM may omit.
- **`elapsed_sec`** (already exists): run/job/trace-relative offset of request received.
  Never first-token-byte or any post-decision instant. PROD records may use `0.0` or a job-relative offset.
- **`lp_status`**: four-value enum. Adding a value later is non-breaking (string enum).
- **`hedge_algorithm`** + **`hedge_schedule`** are independent axes. `hedge_delay_ms=750`
  is ambiguous without both: the decision rule (algorithm) and the time-evaluation
  strategy (schedule) together determine the meaning.
- **`hedge_winner`** (already exists): `None` when `hedge_triggered=False`.

---

## 4. Real-eval `metadata["real_*"]` Promotions

The recorder at `experiments/real_evaluation/recorder.py:410–448` currently
stuffs LP / cost / cache fields into `metadata["real_*"]`. After this change:

| Current `metadata` key | New canonical field |
|---|---|
| `real_lp_weights` | `lp_weights` |
| `real_lp_status` | `lp_status` |
| `real_budget_usd` | `lp_budget_usd` |
| `real_physical_cost_usd` | `physical_cost_usd` |
| `real_primary_physical_cost_usd` | `primary_physical_cost_usd` |
| `real_backup_physical_cost_usd` | `backup_physical_cost_usd` |
| `real_primary_routing_estimated_cost_usd` | `primary_routing_estimated_cost_usd` |
| `real_backup_routing_estimated_cost_usd` | `backup_routing_estimated_cost_usd` |

Set `source = "real_eval"`. Hedge metadata depends on the policy: a
hedge-enabled policy (`policy.use_hedge=True`) writes
`hedge_algorithm = "probability_target"` and
`hedge_schedule = "slo_relative_checkpoints"`; a body-only policy
(`policy.use_hedge=False`) writes `hedge_algorithm = "disabled"` and
`hedge_schedule = None`. The runner passes these explicitly per request.

`total_cost_usd` keeps its existing meaning (currently
`recorder.py:402: total_cost_usd=billed_cost`); it is the canonical accounting
cost. The recorder additionally populates `routing_estimated_cost_usd =
primary_routing_estimated_cost_usd + backup_routing_estimated_cost_usd` for
LP debug parity with SIM.

Keep in `metadata` (transport / cache / timing diagnostics, not routing semantics):

- `real_retry_count`, `real_rate_limit_count`, `real_http_status`
- `real_transport`, `real_retry_sleep_ms`, `real_status`, `real_cost_source`
- `real_*_cached_input_tokens`, `real_*_observed_cached_input_tokens`
- `real_hedge_checkpoint_ts`, `real_primary_start_ts`, `real_backup_dispatch_ts`,
  `real_backup_start_ts`, `real_primary_first_token_ts`, `real_backup_first_token_ts`
- `real_actual_dispatch_overhead_ms`, `real_checkpoint_dispatch_overhead_ms`,
  `real_backup_dispatch_overhead_ms`
- `real_tier_mix`, `real_reference_cost_usd` (LP-side diagnostics — promote later if needed)

Drop `real_wall_clock_ts` (now redundant with the canonical `timestamp_sec`).

---

## 5. PROD Stage 1 — `metadata["routewise"]`

PROD does NOT alter `api_logs`. It writes routing detail into the existing
`metadata JSONB` column under a `"routewise"` namespace:

```json
{
  "routewise": {
    "policy": "routewise",
    "primary_provider": "chutes",
    "primary_tier": "quota",
    "backup_provider": "openrouter",
    "backup_tier": "api",
    "final_provider": "openrouter",
    "final_tier": "api",
    "hedge_triggered": true,
    "hedge_delay_ms": 1250.0,
    "hedge_winner": "backup",
    "hedge_algorithm": "probability_target",
    "hedge_schedule": "slo_relative_checkpoints",
    "lp_weights": {"chutes": 0.7, "openrouter": 0.3},
    "lp_budget_usd": 0.0012,
    "lp_status": "feasible",
    "routing_estimated_cost_usd": 0.0008,
    "primary_routing_estimated_cost_usd": 0.0,
    "backup_routing_estimated_cost_usd": 0.0008,
    "slo_ms": 3000,
    "slo_violated": false
  }
}
```

Existing `api_logs` columns map to canonical fields without renaming:

| `api_logs` column | Canonical field |
|---|---|
| `request_id` | `request_id` |
| `timestamp` | `timestamp_sec` |
| `model_id` | `model` |
| `provider` | `final_provider` (keep column name; do NOT rename) |
| `ttft_ms` | `ttft_ms` |
| `latency_ms` | `e2e_ms` |
| `prompt_tokens` | `prompt_tokens` |
| `completion_tokens` | `completion_tokens_actual` |
| `cost_usd` | `total_cost_usd` (canonical accounting) |
| `upstream_cost_usd` | `physical_cost_usd` |
| `status_code` | `status` (canonicalize) |
| `error` | `error_class` |

Set `source = "prod"`. Once PROD migrates to the canonical RouteWise hedger,
write `hedge_algorithm = "probability_target"` and
`hedge_schedule = "slo_relative_checkpoints"`. Pre-migration legacy PROD
hedging is not part of the canonical enum; if it must be logged before the
migration, keep it in PROD-specific metadata instead of `hedge_algorithm`.

**Stage 1 explicitly does NOT**:
- Add columns to `api_logs`
- Rename the `provider` column
- Change semantics of `cost_usd` / `upstream_cost_usd`

Column promotion of high-frequency JSONB fields (`policy`, `primary_provider`,
`hedge_triggered`, `lp_status`, `slo_violated`) is a Stage 2 decision driven
by actual query frequency, not part of this doc.

---

## 6. Migration Steps

| Step | Owner | Notes |
|---|---|---|
| 1. Extend `PerRequestRecord` with new identity/LP/hedging fields and six optional cost fields (`routing_estimated_*`, `physical_*`) | SIM core | All new fields default to `None`; existing `total_cost_usd` etc. unchanged |
| 2. SIM populates new fields where applicable (`lp_weights`, `lp_budget_usd`, `hedge_algorithm`, `hedge_schedule`, `source="sim"`); `routing_estimated_*` and `lp_status` remain `None` until the policy emits explicit decision-time estimates on `RoutingDecision.metadata` | SIM | Engine already has `lp_weights` / `lp_budget` from the policy — wire those through; the rest is a policy-side follow-up |
| 3. REAL-EVAL recorder promotes metadata `real_*` per §4 table | REAL | Direct edit of `recorder.py:381–448` |
| 4. PROD router writes `api_logs.metadata["routewise"]` per §5 | PROD | Only logging path changes; `api_logs` schema unchanged |
| 5. Cross-source parity test reads sim/real/prod records into the same metric function | All | `Run.mean_cost_usd()` reads `total_cost_usd` unchanged; no `cost_basis` param needed |

Each step is independently shippable. Step 4 (PROD) can land before step 2
or 3 — they share only the schema definition.

---

## 7. Non-Goals

- H5 (simulator capacity/concurrency model). Handled by prior commits.
- Extracting a shared `routewise-core` Python package. Separate future decision.
- PROD column promotion. Stage 2, data-driven.
- Renaming `total_cost_usd` / `primary_cost_usd` / `backup_cost_usd`.
  They stay as the canonical accounting cost line.
- Merging the offline (`rwsim/offline/`) cost-oracle records into
  `PerRequestRecord`. Deferred per `RWSIM_REFACTOR_PLAN.md`.

---

## 8. Sign-Off

This document is the source of truth for the H6 schema work if we agree:

1. Add new identity / LP / hedging fields: `model`, `source`, `timestamp_sec`,
   `lp_weights`, `lp_budget_usd`, `lp_status`, `hedge_algorithm`,
   `hedge_schedule`.
2. Add six optional cost fields: `routing_estimated_cost_usd` plus
   primary/backup splits, and `physical_cost_usd` plus primary/backup splits.
3. Keep `total_cost_usd` / `primary_cost_usd` / `backup_cost_usd` as the
   canonical accounting cost line; each source fills it with its most
   authoritative "what this request cost" number.
4. `Run` aggregate methods read `total_cost_usd` directly; no `cost_basis`
   parameter.
5. Real-eval promotes eight `metadata["real_*"]` keys to canonical fields.
6. PROD writes routing detail to `api_logs.metadata["routewise"]` without
   schema migration.

Disagreements should name the section and propose the smallest alternative
that preserves three-way schema alignment.
