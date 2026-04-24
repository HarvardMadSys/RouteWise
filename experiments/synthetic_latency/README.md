# Synthetic Latency Experiment

Target home for S1-S5 latency-only scenarios and related ablations.

Concrete scenarios should be YAML configs under `configs/`; shared mechanics
should live in `rwsim/`.

Smoke example:

```bash
routewise run synthetic_latency --scenario s1_dominant --strategy v2_only
```

Full-sweep synthetic latency runners live under `suites/` and are exposed as
`routewise suite synthetic`, `routewise suite sanity`, and related suite names.
