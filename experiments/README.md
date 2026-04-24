# Experiments

This directory is the target home for reproducible paper experiments.

Each experiment should combine:

- configs under `configs/`,
- a thin experiment runner,
- fixed seed sets,
- metric definitions,
- output schema decisions.

Core simulation logic belongs in `rwsim/`, not here.

Use `scripts/run_experiment.py --experiment tiered_capacity --list` to inspect
the config-driven entry points.

Paper-used workflows that are not yet config-driven also live here during the
migration, including `offline_counterfactual/`, `latency_phase3/`, and
`latency_phase4/`.

`offline_stage/` owns the paper offline/stage configuration and config loader.
The reusable offline simulator primitives live in `rwsim/offline/`; the
remaining stage strategy implementations are being migrated behind
compatibility wrappers.
