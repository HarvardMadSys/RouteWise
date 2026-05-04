# Simulation Experiment

Target home for the paper-facing simulator experiment harness: S6/S7/S8/S9,
`unified_pool`, calibrated scenarios, stress scenarios, and the eval-grid
simulation sweeps.

Concrete scenarios should be YAML configs under `configs/`; policy behavior
should be assembled from `rwsim.policies` pipeline aliases.

Smoke example:

```bash
routewise run simulation --scenario s6_slow_q_trap --strategy joint_hedge
```

Full-sweep simulation runners live under `suites/` and are exposed as
`routewise suite simulator_grid`, `routewise suite mm25_baselines`, and
`routewise suite stress`.
