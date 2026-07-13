# Simulation Experiments

Paper-facing simulator harness. Each section is implemented as a dedicated
Python module and exposed through the repository-only CLI module:
`uv run python -m routewise_cli.main simulator <section>`.

## Common Setup

Latency families come from `routewise/sim/world/distributions.py` plus empirical
latency profiles in [`latency_profiles/`](latency_profiles/):

- `uniform`: bounded, no tail.
- `normal`: symmetric, light tail.
- `heavy_tail`: lognormal-style heavy tail.
- `real_world` (RW3): three empirical OpenRouter provider profiles.
- `real_world` (RW8): eight empirical OpenRouter provider profiles.

Real-world pools are locked in
[`latency_profiles/pools.yaml`](latency_profiles/pools.yaml):

- `RW3 = [WandB, DeepInfra, Novita]`
- `RW8 = [WandB, DeepInfra, Google, Alibaba, Novita, Cerebras, SiliconFlow, AtlasCloud]`
- `minimax_m25_rw8 = [Inceptron, Friendli, DeepInfra, SambaNova, Venice, AtlasCloud, Chutes, SiliconFlow]`
- `rw8_pooled`: RW8 samples concatenated for cases where latency must be held
  constant across providers.

Common simulator baselines:

- `greedy_cost`: cheapest feasible provider.
- `greedy_latency`: lowest expected TTFT.
- `random`: uniform over feasible providers.
- `offline`: cost-only oracle, implemented under `experiments/offline_stage/`.

OpenRouter-native `sort=price` and `sort=latency` are live real-evaluation
baselines only.

The default simulator dataset is a one-month ShareGPT trace. Routing assumes
the output token length is known at decision time; output-prediction error is
handled by its own ablation.

## 1. Cost Layer (`cost_layer.py`)

Same latency, different cost. Cost-layer experiments incrementally add scarce
capacity tiers so each capacity model can be isolated.

### 1.1 On-demand only

Three API providers with costs `$1`, `$2`, and `$4` per million input tokens.
All providers share the same latency construction for a given run:

- `uniform`
- `normal`
- `heavy_tail`
- `real_world` using `rw8_pooled`

### 1.2 Add quota provider

Adds one quota tier and sweeps the number of subscriptions.

### 1.3 Add concurrency provider

Adds one concurrency tier and sweeps the number of subscriptions.

## 2. Latency Layer (`latency_layer.py`)

Same cost, different latency profiles. Cost is held constant so the router
only chooses on latency. The main ablation knob is profile overlap:

- `no_overlap`: provider distributions are clearly separated.
- `half_overlap`: provider distributions share part of their support.

Scenarios cover `uniform`, `normal`, `heavy_tail`, and `real_world` profiles.

## 3. Hedging (`hedging.py`)

Probability-target hedging on top of the latency-layer setup. The router
checks a canonical SLO checkpoint grid and dispatches the latest backup that
can still meet the target combined success probability.

Headline metrics include hedge trigger fraction, P99 TTFT, mean TTFT, P50
TTFT, and cost multiplier.

## 4. End-to-End (`end_to_end.py`)

Real-world cost and real-world latency, including multi-tier deployments:

- Three-provider config: one API tier, one quota tier, one concurrency tier.
- Controlled cost-tier config: three API providers with fixed paper-facing
  costs plus one quota tier and one concurrency tier.
- RW8+capacity config: eight MiniMax M2.5 API providers plus one quota tier
  and one concurrency tier.

End-to-end scenarios evaluate hedging and the `p` budget knob used for
cost-vs-latency Pareto sweeps.

## Code Layout

```text
experiments/simulation/
  cost_layer.py        # cost-layer section
  latency_layer.py     # latency-layer section
  hedging.py           # hedging section
  end_to_end.py        # end-to-end section
  common.py            # provider builders, workload loading, summary helpers
  latency_profiles/    # empirical latency artifacts
```

## CLI

List sections:

```bash
uv run python -m routewise_cli.main simulator list
```

Run a section:

```bash
uv run python -m routewise_cli.main simulator cost-layer
uv run python -m routewise_cli.main simulator latency-layer
uv run python -m routewise_cli.main simulator hedging
uv run python -m routewise_cli.main simulator end-to-end
```
