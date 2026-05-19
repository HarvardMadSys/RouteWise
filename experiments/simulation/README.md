# Simulation Experiment

Paper-facing simulator harness. Source of truth for experiment structure is
the Notion page **RouteWise → Evaluation → Simulation** (last synced
2026-05-04). This README mirrors that page; if the two diverge, Notion wins
and this file should be re-synced.

The high-level mental model lives in [`docs/EXPERIMENT_LAYOUT.md`](../../docs/EXPERIMENT_LAYOUT.md).

## Common setup

**Latency families** — `rwsim/world/distributions.py` registry plus the empirical
latency profiles in [`latency_profiles/`](latency_profiles/):

- `uniform`           — bounded, no tail
- `normal`            — symmetric, light tail
- `heavy_tail`        — `LogNormal`, heavy tail
- `real_world` (RW3)  — three real OpenRouter providers, less-overlapping latency
- `real_world` (RW8)  — eight real OpenRouter providers, realistic overlap

Real-world pools are locked. See [`latency_profiles/pools.yaml`](latency_profiles/pools.yaml):

- `RW3 = [WandB, DeepInfra, Novita]`
- `RW8 = [WandB, DeepInfra, Google, Alibaba, Novita, Cerebras, SiliconFlow, AtlasCloud]`
- `minimax_m25_rw8 = [Inceptron, Friendli, DeepInfra, SambaNova, Venice, AtlasCloud, Chutes, SiliconFlow]`
- `rw8_pooled` — RW8 samples concatenated; used wherever latency must be held
  constant across providers (cost layer real-world case).

**Baselines** — every layer compares against:

- `greedy_cost`      — cheapest feasible; zero-cost ties break `S_C` → `S_Q` → `S_A`
- `greedy_latency`   — always lowest expected TTFT
- `random`           — uniform over feasible providers
- `offline`          — cost-only oracle (greedy / ILP, see `experiments/offline_stage/`)

OpenRouter-native `sort=price` / `sort=latency` baselines are not simulated;
they remain live real-evaluation policies only.

**Dataset** — ShareGPT one-month trace.

**Metrics** — every run reports:

- per-system TTFT distribution (CDF)
- cost (bar + table)
- per-provider request fraction (stacked bar)
- mean / P50 / P99 (headline)
- P10 / P25 / P75 / P90 (tail breakdown)

The materialised metric record lives in `rwsim/metrics/run.py` (`Run`).

Routing assumes the output token length is known at decision time
(value-estimator effect is its own ablation, see
`experiments/estimator_ablation/`).

---

## 1. Cost layer (`cost_layer.py`)

Same latency / different cost. Cost layer rolls in scarce-capacity tiers
incrementally so the contribution of each capacity model is isolatable.

### 1.1 On-demand only — three providers, three cost points

Sample setup: cost = `$1 / $2 / $4 per M tokens`, all synthetic mean TTFT =
300 ms.

- 1.1.1 `uniform` family
- 1.1.2 `normal` family
- 1.1.3 `heavy_tail` family
- 1.1.4 `real_world` family (uses `rw8_pooled` so all three providers share
  one empirical distribution)

### 1.2 Add quota provider

One subscription distribution (typically `lognormal`-shaped). Sweep the
number of subscriptions to find the cost-optimal count.

### 1.3 Add concurrency provider

One subscription distribution. Same subscription-count sweep.

---

## 2. Latency layer (`latency_layer.py`)

Same cost / different latency profiles. Cost is held constant so the router
only chooses on latency. The ablation knob is **distribution overlap**:

- `no_overlap`   — provider distributions are clearly separated
- `half_overlap` — provider distributions share roughly half their support

### 2.1 No hedging (three providers)

- 2.1.1 `uniform`     — no_overlap, half_overlap
- 2.1.2 `normal`      — no_overlap, half_overlap
- 2.1.3 `heavy_tail`  — no_overlap, half_overlap
- 2.1.4 `real_world`  — RW3

### 2.2 Hedging (`hedging.py`)

Same setup as 2.1 plus `Hedging-Explorer`. Run on three-provider and
eight-provider configurations:

- 2.2.1 `heavy_tail`   — three providers and eight providers
- 2.2.2 `real_world`   — RW3 and RW8

Headline metrics: hedge trigger fraction (bar), P99 (bar), mean and P50.

---

## 3. End-to-end (`end_to_end.py`)

Real-world cost and real-world latency, multi-tier deployments. Subscription
counts come from §1.2 / §1.3 results.

### 3.1 Hedging (Hedging-Explorer on)

- Three-provider config: 1 on-demand + 1 quota + 1 concurrency
- Controlled cost-tier config: 3 on-demand providers with §1 synthetic API
  prices (`$1/$5`, `$2/$10`, `$4/$20` per million input/output tokens) and
  dispersed real-world latency profiles, plus 1 quota + 1 concurrency
- RW8+capacity config: 8 MiniMax M2.5 OpenRouter on-demand providers
  (`minimax_m25_rw8`) + 1 quota + 1 concurrency
  (10 providers total)

### 3.2 Varying `p` (LP budget knob)

Same two configs as §3.1; sweep `p ∈ [0, 1]` for the cost-vs-latency Pareto.
HTTP `429` rates belong to live real-evaluation runs, not simulator runs.

---

## Code layout

The implementation follows the Notion structure one-file-per-section
(see [`docs/SIMULATION_SECTION_REFACTOR.md`](../../docs/SIMULATION_SECTION_REFACTOR.md)):

```text
experiments/simulation/
  cost_layer.py        # §1
  latency_layer.py     # §2.1
  hedging.py           # §2.2
  end_to_end.py        # §3
  common.py            # provider builders, workload loading, summary helpers
  latency_profiles/    # real-world latency artifacts (RW3 / RW8 / rw8_pooled)
```

Implemented section runners are invoked through the CLI:

```bash
routewise simulator list
routewise simulator cost-layer
```
