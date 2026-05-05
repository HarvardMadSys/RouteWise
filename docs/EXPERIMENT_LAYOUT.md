# RouteWise Experiment Layout

> Decision document. Sign off so we stop relitigating naming and structure.

Last updated: 2026-05-04. Replaces ad-hoc decisions scattered across Slack
threads, the 5/1 NSDI meeting, the 5/4 1:1 with Juncheng, and the recent
`tiered_capacity` → `simulation` rename.

---

## 1. Mental model — three layers

```
Layer 1   rwsim/                       simulator engine (library)
Layer 2   experiments/<name>/          experiment harness (uses rwsim)
Layer 3   plots/, outputs/, docs/      artifacts (consume layer 2 output)
```

Rules:

- `rwsim/` is paper-agnostic. Provider models, distributions, policies, LP
  solver, hedging logic, metrics primitives.
- `experiments/<name>/` decides *which* scenarios, datasets, seeds, sweeps to
  run. Imports from `rwsim/`. Never re-implements engine logic.
- `plots/` consumes results, produces figures and tables. Organised by paper
  section / concept (cost_layer/, latency_layer/, end_to_end/, ablations/,
  per-experiment subdirs). Shared style/palettes/helpers live at
  `plots/{style,palettes,helpers}.py`. Never owns experiment definitions.

---

## 2. Active experiment subsystems

| Subsystem | What it does | Data source | Paper role |
|---|---|---|---|
| `simulation/` | RouteWise online evaluation on synthetic scenarios (cost / latency / joint / hedging) | synthetic generator + cached traces | **All synthetic figures** |
| `real_evaluation/` | RouteWise driving live OpenRouter calls in real time | live OpenRouter API | **All real-world figures** |
| `offline_stage/` | Offline cost-only oracle (greedy + ILP) | trace | Lower-bound baseline |
| `estimator_ablation/` | Compare EMA / histogram / oracle value estimators | trace | Appendix ablation |
| `plots/` (top-level) | Render paper figures and tables | results from above | Final artifact |

---

## 3. Naming conventions

### 3.1 Two independent "Stage" systems — never mix

**Cost Oracle Stages** (in `offline_stage/`, the baseline / lower bound):

- **Stage Q** — quota-only oracle (S_Q + S_A). Greedy by API cost.
- **Stage QC** — quota + concurrency oracle (S_Q + S_C + S_A). MILP.

**Synthetic Scenarios** (in `simulation/`, the RouteWise evaluation):

- **S0** — same latency, different cost (3 × S_A)
- **S1** — same cost, different latency (3 × S_A)
- **S2** — cost-latency tradeoff (3 × S_A)
- **S3** — full joint tier (S_A + S_Q + S_C)

**Rule:** never say "Stage 1" or "Stage 2" without prefix. Use
"Cost Oracle Stage Q" or "Synthetic S0". Most past confusion comes from this
collision.

### 3.2 Pending directory renames

These two still carry legacy names from before the cleanup. Proposed:

| Current | Proposed | Why |
|---|---|---|
| `real_evaluation/` | `live/` | Symmetric with `simulation/`; "eval" overloaded with "model eval" |
| `offline_stage/` | `cost_oracle/` | Says what it is; "stage" was an internal name |

Renames defer until naming is signed off. Same procedure as the
`tiered_capacity → simulation` rename.

---

## 4. The four-layer experiment plan (5/4 meeting)

Per Juncheng's 5/4 framing, paper experiments split into four layers, each
isolating one decision axis. Within `simulation/`, scenarios are organised
along this axis:

| Layer | Goal | Setup | Key metrics |
|---|---|---|---|
| **Cost layer** | Show RouteWise minimises cost when latency is held equal | same latency / different cost; ShareGPT 1-month workload; subscription count optimisation (1/2/3/4) | per-provider cost, request fraction, TTFT distribution |
| **Latency layer** | Show RouteWise picks fast providers when cost is held equal | same cost / different latency; 4 distributions (uniform / normal / lognormal / real-world); distribution overlap as ablation knob | mean / P50 / P90 / P99 TTFT |
| **Hedging** | Show Hedging-Explorer cuts P99 with bounded cost overhead | inside latency layer; probe + moving-average online profile; backup = random non-primary (Explorer style); evaluate trigger at P25 / P50 / P75 / P90 | P99 reduction, hedge trigger rate, cost multiplier |
| **End-to-end** | Show RouteWise wins on real-world workload | real-world distribution; 3-provider and 8-provider configs | cost vs latency Pareto, SLO violation, tier mix |

---

## 5. Locked technical decisions

These are settled. Don't reopen unless evidence changes.

- **LP budget knob** — `B_p(t) = c_min(t) + p · (c_max(t) − c_min(t))`,
  `p ∈ [0, 1]`. Self-calibrates to feasible-provider cost envelope. (Paper
  §3.4.2 already updated.)
- **Hedging-Explorer trigger** — probability-targeted: dispatch backup at
  the *latest* `t*` such that combined `P_succ(t*) ≥ p*`, default
  `p* = 0.99`. (Paper §3.4 already aligned.)
- **Backup-provider selection** — random non-primary (Explorer style),
  *not* fastest. Enables free probing of stale provider profiles.
- **Trigger re-evaluation** — `P_succ(t)` is computed at multiple
  checkpoints (e.g. P25, P50, P75, P90 of remaining SLO), not once at
  dispatch time. Required because queue depth changes fast.
- **Latency profile maintenance** — bootstrap from 1-hour pre-experiment
  probe, then online moving-average update.
- **Canonical simulator real-world pools** — use one fixed model and two fixed
  provider sets for all simulator real-world-distribution experiments. Source
  model is Qwen3-235B from the cached 24-hour OpenRouter run. The 3-provider
  pool is `RW3 = [WandB, DeepInfra, Novita]`. The 8-provider pool is
  `RW8 = [WandB, DeepInfra, Google, Alibaba, Novita, Cerebras, SiliconFlow,
  AtlasCloud]`. `RW3` is intentionally a subset of `RW8`; do not swap providers
  per figure. Other models are robustness/appendix only.
- **Prefix cache** — modelled in **cost only**, not in latency. Assume
  100% hit when routed to the same provider as the user's previous
  request. Per-provider hit-rate variance not modelled (acknowledged
  limitation).
- **Simulation uses ground-truth** — value estimator runs in `simulation/`
  are off by default. Estimator effect is its own ablation
  (`estimator_ablation/`).
- **Single simulator** — `rwsim/` is the only simulator. No parallel
  offline simulator. (`tiered_capacity → simulation` rename did most of
  this.)
- **Metrics module is a top-level boundary** — `rwsim/metrics/` (not
  `rwsim/world/metrics.py`) owns the simulation result schema and
  aggregation. `world/` only owns provider / quota / concurrency /
  distribution / scenario. `Run` and per-request output protocol belong in
  `rwsim/metrics/run.py`; `rwsim/world/metrics.py` is removed.

---

## 6. Open items

These need a decision or an owner before the relevant experiment can run.

| Item | Owner | Blocker |
|---|---|---|
| Queueing-vs-input-length measurement data for motivation figure | Juncheng | He said "added in the last few days" — need export |
| Subscription count answer (1 vs 2/3/4) under 1-month ShareGPT | Murphy | Can run once cost-layer harness is ready |
| Distribution-overlap parameterisation (KL? area? σ ratio?) | Murphy + Haoran | Pick one before latency layer |
| Hedging probing frequency in `live/` runs | Murphy | Soft default = 1 hour bootstrap |
| Per-provider prefix-hit-rate modelling for `live/` | (defer) | Locked: assume 100% in `simulation/`; revisit only if `live/` results need it |
| End-to-end: keep no-hedge column? | Juncheng | Soft default = yes for ablation |
| Real-world experiment data — migrate from `~/Desktop/NSDI2027_RouteWise/` to `results/cached/`? | Murphy | 493 MB to copy, then 2.1 GB to delete |

---

## 7. Cached real-world data we already have

Paper should reuse, not rerun:

| Source | Model | Duration | Requests | Providers |
|---|---|---|---|---|
| `phase5_online_7d` (Mar 21–28) | Llama 3.3 70B | 7 days | 232 230 | 14 OR |
| `phase5_qwen3_7d_clean` (Apr 11–12) | Qwen3 235B | 24 hours | 564 419 | 13 OR |
| `phase5_minimax_m25_24h` (Apr 12) | MiniMax M2.5 | 3 hours | 201 097 | 17 OR |
| `phase6_joint_1h_hour9_v3` (Apr 23) | MiniMax M2.5 (joint) | 1 hour | 1 345 | 4 quota/concurrency + 17 OR |

These live under `~/Desktop/NSDI2027_RouteWise/experiment/results/`. Plan
is to move them under `results/cached/` so paper figures are reproducible
without paying OpenRouter again.

---

## 8. Sign-off checklist

If you agree with all of the following, this doc is the source of truth:

- [ ] §1–§3 naming and mental model
- [ ] §4 four-layer experiment plan
- [ ] §5 locked technical decisions
- [ ] §6 open-item owners and defaults

Reply on Slack with disagreements; otherwise this is the spec we execute
against.
