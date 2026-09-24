# Real-World Latency Profiles

This directory stores compact empirical TTFT artifacts used by simulator
experiments that need real-world provider latency distributions.

## Artifact

- `qwen3_24h.npz`: per-provider TTFT samples for Qwen3-235B from the cached
  24-hour OpenRouter run.
- `qwen3_24h.json`: provenance and per-provider summary statistics.
- `pools.yaml`: profile registry plus canonical RW3/RW8 provider pools, the
  MiniMax M2.5 RW8 provider pool, pricing source policy, and the pooled RW8
  profile.
- `minimax_m25_subscriptions.npz`: empirical TTFT samples for Minimax M2.5
  subscription providers used by cost-layer quota/concurrency experiments.
- `minimax_m25_subscriptions.json`: provenance and per-provider summary
  statistics for the subscription-provider profile.
- `minimax_m25_openrouter_24h.npz`: empirical TTFT samples for MiniMax M2.5
  OpenRouter providers from the cached `phase5_minimax_m25_24h` run.
- `minimax_m25_openrouter_24h.json`: provenance, observed-duration warning,
  per-provider summary statistics, and OpenRouter price snapshot metadata.
- `minimax_m3_shared_profile_24h.npz`: per-provider TTFT samples for MiniMax M3
  from the 24-hour real-eval rerun of 2026-09-15, covering the six metered
  OpenRouter providers and both subscription tiers.
- `minimax_m3_shared_profile_24h.json`: provenance, per-provider summary
  statistics, and prices carried over from the run's inventory.

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

The MiniMax M3 artifact comes from the shared-profile events of the 24-hour
rerun whose request records are in `data/real_eval_records_m3/`: 202,143 TTFT
samples over 24.2 observed hours for eight providers, natural traffic and
profiling probes together. Unlike the M2.5 subscription artifact, whose quota
and concurrency samples are hourly single-request probes from a separate
campaign, every tier here is measured in one run, so the M3 end-to-end scenario
takes all three tiers from this one file.

## Refresh

Regenerate the artifact with:

```bash
# Qwen3 235B
python -m scripts.prepare_latency_profile \
    --model qwen3_235b \
    --source-log /path/to/qwen3/evaluation_log.csv \
    --out-npz experiments/simulation/latency_profiles/qwen3_24h.npz

# MiniMax M2.5 (OpenRouter)
python -m scripts.prepare_latency_profile \
    --model minimax_m25 --model-family minimax-m2.5 \
    --source-log /path/to/minimax/evaluation_log.csv \
    --out-npz experiments/simulation/latency_profiles/minimax_m25_openrouter_24h.npz \
    --endpoints-json /tmp/openrouter_minimax_m25_endpoints.json \
    --price-source https://openrouter.ai/api/v1/models/minimax/minimax-m2.5/endpoints \
    --run-label phase5_minimax_m25_24h

# MiniMax M3 (real-eval shared-profile events)
python -m scripts.prepare_latency_profile \
    --model minimax_m3 --model-family minimax-m3 \
    --source-log /path/to/m3_run/shared_profile_events.jsonl \
    --source-format shared_profile_events \
    --inventory-json experiments/real_evaluation/data/pilot_or_minimax_m3_subscription_or6_true24h.json \
    --out-npz experiments/simulation/latency_profiles/minimax_m3_shared_profile_24h.npz \
    --tier mixed --run-label real_eval_minimax_m3_burstgpt_day0_24h
```

The profile preparation script drops providers with fewer than 1,000 valid TTFT
samples and caps each provider at 50,000 samples using a fixed subsample seed.
The cap keeps the committed artifact small while preserving the empirical
latency shape for bootstrap sampling.

## Pool Semantics

- `rw3` and `rw8` keep provider-specific Qwen3 empirical distributions and
  static committed price metadata for simulator sections that need API costs.
  Use them when provider latency differences are the point of the experiment.
- `minimax_m25_rw8` keeps the selected eight MiniMax M2.5 OpenRouter providers:
  Inceptron, Friendli, DeepInfra, SambaNova, Venice, AtlasCloud, Chutes, and
  SiliconFlow. This is the §3 end-to-end OpenRouter API pool. Its prices are
  resolved from `minimax_m25_openrouter_24h.json` rather than hardcoded in the
  scenario.
- `minimax_m3_rw6` keeps the six metered MiniMax M3 providers of the 2026-09-15
  rerun: Minimax, Together, GMICloud, AtlasCloud, Novita, and StreamLake. Prices
  resolve from `minimax_m3_shared_profile_24h.json`, which carries them from the
  run's inventory. The `end_to_end_m3_rw6` scenario pairs this pool with the
  MiniMax Plus quota tier and the Featherless Premium concurrency tier, both
  drawn from the same profile.
- `rw8_pooled` concatenates all RW8 provider samples into one anonymous
  distribution. Use it when latency must be held constant and the experiment
  varies another axis, such as cost.

Costs are not defined in this directory. Each simulator section owns its own
cost model.
