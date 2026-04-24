# Tiered Capacity Experiment

Target home for S6/S7/S8/S9, `unified_pool`, calibrated, and stress capacity
experiments.

Concrete scenarios should be YAML configs under `configs/`; policy behavior
should be assembled from `rwsim.policies` pipeline aliases.

Smoke example:

```bash
routewise run tiered_capacity --scenario s6_slow_q_trap --strategy joint_hedge
```

Full-sweep tiered runners live under `suites/` and are exposed as
`routewise suite joint`, `routewise suite joint_mm25_baselines`, and
`routewise suite stress`.
