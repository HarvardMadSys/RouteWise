# Offline Stage

This package owns the paper offline/stage experiment configuration.

- `configs/experiment.yaml` is the canonical Stage 1 / Stage 2 config.
- `config.py` parses the YAML into offline provider and experiment objects.
- `latency_profiling.py` is the OpenRouter probe used to collect TTFT/E2E
  samples for latency-stage analysis.
- `config/experiment.yaml` at the repository root is a compatibility symlink.

Reusable offline primitives live in `rwsim/offline/`. Strategy implementations
live under `strategies/` so `legacy/experiment/strategies/` can remain
compatibility wrappers only.
