# Real-World Latency Profiles

This directory stores compact empirical TTFT artifacts used by simulator
experiments that need real-world provider latency distributions.

## Artifact

- `qwen3_24h.npz`: per-provider TTFT samples for Qwen3-235B from the cached
  24-hour OpenRouter run.
- `qwen3_24h.json`: provenance and per-provider summary statistics.
- `pools.yaml`: canonical RW3/RW8 provider pools, the MiniMax M2.5 RW8
  provider pool, and the pooled RW8 profile.
- `minimax_m25_subscriptions.npz`: empirical TTFT samples for Minimax M2.5
  subscription providers used by cost-layer quota/concurrency experiments.
- `minimax_m25_subscriptions.json`: provenance and per-provider summary
  statistics for the subscription-provider profile.
- `minimax_m25_openrouter_24h.npz`: empirical TTFT samples for MiniMax M2.5
  OpenRouter providers from the cached `phase5_minimax_m25_24h` run.
- `minimax_m25_openrouter_24h.json`: provenance, observed-duration warning,
  per-provider summary statistics, and OpenRouter price snapshot metadata.

The Qwen3 artifact was prepared from an internal cached 24-hour OpenRouter
evaluation log. The raw CSV is not committed to this repository.

The Minimax M2.5 subscription artifact keeps successful TTFT samples for Chutes
direct subscription, Featherless direct subscription, and MiniMax native
subscription. Chutes samples are per-request records from an internal joint
online run filtered to direct Chutes transport. Featherless and MiniMax native
samples are hourly direct-probe snapshots with one request per snapshot.

The MiniMax M2.5 OpenRouter artifact keeps successful TTFT samples from the
cached run directory named `phase5_minimax_m25_24h`. The available CSV covers
2.82 observed hours, not a complete 24-hour measurement; the metadata sidecar
records `run_stats.is_full_day_observation=false`.

## Refresh

Regenerate the artifact with:

```bash
python -m scripts.prepare_latency_profiles
python -m scripts.prepare_minimax_m25_openrouter_profile --endpoints-json /tmp/openrouter_minimax_m25_endpoints.json
```

The profile preparation script drops providers with fewer than 1,000 valid TTFT
samples and caps each provider at 50,000 samples using a fixed subsample seed.
The cap keeps the committed artifact small while preserving the empirical
latency shape for bootstrap sampling.

## Pool Semantics

- `rw3` and `rw8` keep provider-specific Qwen3 empirical distributions. Use
  them when provider latency differences are the point of the experiment.
- `minimax_m25_rw8` keeps the selected eight MiniMax M2.5 OpenRouter providers:
  Inceptron, Friendli, DeepInfra, SambaNova, Venice, AtlasCloud, Chutes, and
  SiliconFlow. This is the §3 end-to-end OpenRouter API pool.
- `rw8_pooled` concatenates all RW8 provider samples into one anonymous
  distribution. Use it when latency must be held constant and the experiment
  varies another axis, such as cost.

Costs are not defined in this directory. Each simulator section owns its own
cost model.
