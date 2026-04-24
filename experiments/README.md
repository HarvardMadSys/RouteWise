# Experiments

This directory is the target home for reproducible paper experiments.

Each experiment should combine:

- configs under `configs/`,
- a thin experiment runner,
- fixed seed sets,
- metric definitions,
- output schema decisions.

Core simulation logic belongs in `rwsim/`, not here. The application CLI lives
in `routewise_cli/` and dispatches into this package.

Use the CLI to inspect config-driven entrypoints:

```bash
routewise list --experiment tiered_capacity
routewise list --suites
```

Paper-used workflows that are not yet config-driven also live here during the
migration, including `offline_counterfactual/`, `latency_phase3/`, and
`latency_phase4/`.

Full-sweep paper runners live under `experiments/*/suites/`. They are allowed
to orchestrate grids, plots, and output paths, but they should not own reusable
simulator logic.

`offline_stage/` owns the paper offline/stage configuration and config loader.
The reusable offline simulator primitives live in `rwsim/offline/`; the
remaining stage strategy implementations are being migrated behind
compatibility wrappers.
