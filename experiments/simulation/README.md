# Simulation Experiments

Paper-facing simulator harness. Each section is implemented as a dedicated,
directly runnable Python module:
`uv run python -m experiments.simulation.<section>`.

See the [figure reproduction guide](../../docs/research/FIGURE_MAP.md) for
the data, commands, and source revisions used by each paper figure.

## Common Setup

Latency families come from `llm_routewise/sim/world/distributions.py` plus empirical
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
- `offline`: cost-only offline baseline in `offline_oracle.py`; exactness
  depends on the scenario, and its duration model differs from online runs.

OpenRouter-native `sort=price` and `sort=latency` are live real-evaluation
baselines only.

The default workload is the 30-day BurstGPT/ShareGPT composition prepared by
`scripts/prepare_workload.py`. The common routing predictor defaults to
`bucket_mean`; the Figure 9a wrapper explicitly selects `oracle` and applies
fixed multiplicative biases. The simulator currently updates bucket-mean
feedback in request-loop order without waiting for simulated completion.
Known output lengths are also used to calculate realized request outcomes;
they should not be confused with the prediction available to the router.

## Model assumptions

### Output-length feedback

The simulator computes a request outcome and immediately calls
`policy.observe` before routing the next arrival. The output-length estimator
is updated without waiting for simulated response completion. With overlapping
arrivals, later decisions can therefore use lengths that would not yet be
observable online. This applies to both the current simulator and the
historical Figure 8 source; the full effect of completion-gated feedback
has not been quantified.

### Quota threshold

The exponential quota threshold is an empirically evaluated heuristic.
Theorem 3.3's competitive-ratio guarantee does not hold for this threshold;
no replacement guarantee is claimed. Empirical P10/P90 estimates are not
bounds on every request's value.

### Offline baselines

`offline_oracle.py` assigns requests at their fixed arrival times without
queueing, reordering, or preemption. Online concurrency occupancy uses
sampled service durations, while the offline baseline uses P50 TTFT and
throughput. Some supported settings are solved exactly under those
assumptions; multi-window quota uses a value-greedy heuristic, and joint
exact solving is restricted by request count.

These baselines do not supply a matched-duration comparison or the original
solver-status/configuration mapping for the paper's 15.0%, 19.3%, and 6.0%
gaps. Their output should not be interpreted as verification of those
distances to an optimum.

### Output-length ablation

The Figure 9a wrapper uses
`predicted_tokens = actual_tokens * (1 + error_pct / 100)`, with one fixed
bias per run at −50%, −25%, 0%, +25%, +50%, +100%, or +200%. Its baseline
is an oracle. The default sweep contains 42 configurations per seed:
seven biases, three α values, and LP-only/LP+hedging policies.

This measures systematic bias rather than the independent uniform random
errors described in §4.4.2, and does not establish bucket-mean robustness
to that noise model. The wrapper emits a manifest for new runs; a complete
original Figure 9a run manifest is not included.

### Cache, cost, and latency

Trace-observed cache-read tokens are applied to candidate providers offering
cached-input rates without reconstructing provider-local cache residency
under rerouting. See the [trace data notes](../../data/README.md#prod-trace-freeinferencejsonl).
The LP budget constrains expected primary effective cost; it does not cap
the final bill including hedges. Hedging probabilities rely on independence
assumptions for primary and backup latencies and represent targets, not
guaranteed SLOs. The existence of a sparse optimal LP solution does not
exclude dense optima under ties, and uniform α spacing need not produce
uniformly spaced achieved cost/latency points.

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

## Running Sections

Each section prints its scenarios, policies, and options with `--help`.
Run a section:

```bash
uv run python -m experiments.simulation.cost_layer
uv run python -m experiments.simulation.latency_layer
uv run python -m experiments.simulation.hedging
uv run python -m experiments.simulation.end_to_end
```
