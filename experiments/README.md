# Experiments

This directory is the target home for reproducible paper experiments.

Config-driven experiment packages should combine:

- configs under `configs/`,
- a thin experiment runner,
- fixed seed sets,
- metric definitions,
- output schema decisions.

The simulator paper line is section-driven instead of config-driven. Its
entrypoints live directly under `experiments/simulation/`.

Each simulator section is a directly runnable module (`uv run python -m
experiments.simulation.<section> --help` lists its scenarios and policies);
config-driven packages are registered in
`experiments.available_experiments()`.

Earlier latency-phase replay packages were retired; the current paper-facing
simulator method lives under `simulation/`.

The agentic benchmark has a separate, heavier dependency group and currently
uses Python 3.13 because its upstream stack does not yet publish all Python
3.14 wheels:

```bash
uv sync --python 3.13 --only-group agent-benchmark
```

`offline_stage/` owns the paper offline/stage configuration and config loader.
The reusable offline simulator primitives live in `llm_routewise/offline/`; the
remaining stage strategy implementations are being migrated behind
compatibility wrappers.
