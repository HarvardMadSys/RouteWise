# Real-World Latency Profiles

This directory stores compact empirical TTFT artifacts used by simulator
experiments that need real-world OpenRouter latency distributions.

## Artifact

- `qwen3_24h.npz`: per-provider TTFT samples for Qwen3-235B from the cached
  24-hour OpenRouter run.
- `qwen3_24h.json`: provenance and per-provider summary statistics.
- `pools.yaml`: canonical RW3/RW8 provider pools and the pooled RW8 profile.

The raw CSV is not committed to this repository. The current artifact was
prepared from:

```text
/Users/realtmxi/Desktop/NSDI2027_RouteWise/experiment/results/phase5_qwen3_7d_clean/run_20260410_171625/evaluation_log.csv
```

## Refresh

Regenerate the artifact with:

```bash
python -m scripts.prepare_latency_profiles
```

The profile preparation script drops providers with fewer than 1,000 valid TTFT
samples and caps each provider at 50,000 samples using a fixed subsample seed.
The cap keeps the committed artifact small while preserving the empirical
latency shape for bootstrap sampling.

## Pool Semantics

- `rw3` and `rw8` keep provider-specific empirical distributions. Use them
  when provider latency differences are the point of the experiment.
- `rw8_pooled` concatenates all RW8 provider samples into one anonymous
  distribution. Use it when latency must be held constant and the experiment
  varies another axis, such as cost.

Costs are not defined in this directory. Each simulator section owns its own
cost model.
