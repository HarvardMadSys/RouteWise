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

Legacy workflows from earlier paper iterations also live here during the
migration, including `offline_counterfactual/`, `latency_phase3/`, and
`latency_phase4/`. Treat Phase 3/4 latency replay results as deprecated
historical baselines unless they are rerun after auditing the replay
interpolation choice; the current paper-facing joint method lives under
`tiered_capacity/`.

Full-sweep paper runners live under `experiments/*/suites/`. They are allowed
to orchestrate grids, plots, and output paths, but they should not own reusable
simulator logic.

`offline_stage/` owns the paper offline/stage configuration and config loader.
The reusable offline simulator primitives live in `rwsim/offline/`; the
remaining stage strategy implementations are being migrated behind
compatibility wrappers.
