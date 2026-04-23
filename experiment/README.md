# Experiment Configuration

This document explains the `experiment.yaml` configuration file for Stage 2 dual-subscription routing experiments.

## Overview

The routing optimization considers three options for each request:
- **S_Q**: Daily quota subscription (e.g., Chutes)
- **S_C**: Concurrency-limited subscription (e.g., Featherless AI or Local GPU)
- **API**: Pay-as-you-go API (fallback, uses per-model pricing)

## Configuration Sections

### `model_pricing`

API pricing per 1M tokens for each model. When a request is routed to API, the cost is calculated using the pricing for that specific model.

```yaml
model_pricing:
  llama-3.3-70b-instruct:
    input: 0.25   # $/1M input tokens
    output: 0.75  # $/1M output tokens
```

**Important**: Every model in your workload must have pricing defined here. The system will raise an error if a model is missing (no silent fallback).

### `subscriptions`

#### S_Q: Daily Quota (Chutes)

```yaml
chutes:
  type: daily_quota
  monthly_fee: 20.0      # $/month
  daily_quota: 5000      # requests/day
  supported_models: [...]  # Models available on this subscription
```

#### S_C: Concurrency-Limited Options

Three pre-configured options:

| Config Name | Provider | Monthly Fee | Concurrency | Use Case |
|-------------|----------|-------------|-------------|----------|
| `local_gpu` | Local GPU | $0 (sunk cost) | 8 | Self-hosted inference |
| `featherless_premium` | Featherless AI | $25 | 4 | Cloud inference (budget) |
| `featherless_scale` | Featherless AI | $75 | 8 | Cloud inference (scale) |

Each S_C config specifies:
- `monthly_fee`: Monthly subscription cost
- `concurrency_limit`: Max concurrent requests
- `supported_models`: Models with their multipliers

**Multiplier**: Larger models consume more concurrency slots.
- 70B models: multiplier=4 (uses 4 slots)
- 30B models: multiplier=2 (uses 2 slots)
- 7B models: multiplier=1 (uses 1 slot)

Example: With `concurrency_limit=8` and a 70B model (multiplier=4), effective concurrency is 8/4=2.

## Usage

### Run Stage 2 Experiments

```bash
# Using local GPU (default)
python -m experiment.scripts.run_stage2_experiments \
    --data freeinference \
    --sc-config local_gpu

# Using Featherless Premium
python -m experiment.scripts.run_stage2_experiments \
    --data freeinference \
    --sc-config featherless_premium

# Using Featherless Scale
python -m experiment.scripts.run_stage2_experiments \
    --data freeinference \
    --sc-config featherless_scale
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data` | freeinference | Dataset: freeinference, rednote, sharegpt |
| `--sc-config` | local_gpu | S_C configuration to use |
| `--daily-quota` | 5000 | Daily quota for S_Q |
| `--delta` | 300.0 | Time slot size in seconds for ILP |
| `--solver` | gurobi | ILP solver: gurobi (fast) or cbc (free) |
| `--latency-slo` | 0 | Max queueing delay (0=zero-wait system) |

### Single Model Comparison

For ablation studies comparing different model sizes:

```bash
python -m experiment.scripts.run_single_model_comparison \
    --data freeinference \
    --sc-config local_gpu
```

## Adding New Models

1. Add API pricing to `model_pricing`:
```yaml
model_pricing:
  new-model-name:
    input: 0.10
    output: 0.30
```

2. Add to S_Q supported models (if applicable):
```yaml
chutes:
  supported_models:
    - new-model-name
```

3. Add to S_C supported models with multiplier:
```yaml
local_gpu:
  supported_models:
    new-model-name:
      multiplier: 2  # Based on model size
```

## Featherless AI Model Compatibility

Not all models are supported by Featherless AI. The `featherless_premium` and `featherless_scale` configs only list actually supported models:
- llama-3.3-70b-instruct
- glm-4.6
- deepseek-r1

For full model support, use `local_gpu` config which assumes you can run any model locally.

See: https://featherless.ai/docs/model-compatibility
