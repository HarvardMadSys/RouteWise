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

Use the CLI to inspect available entrypoints:

```bash
routewise list
routewise simulator list
```

Earlier latency-phase replay packages were retired; the current paper-facing
simulator method lives under `simulation/`.

`offline_stage/` owns the paper offline/stage configuration and config loader.
The reusable offline simulator primitives live in `rwsim/offline/`; the
remaining stage strategy implementations are being migrated behind
compatibility wrappers.
