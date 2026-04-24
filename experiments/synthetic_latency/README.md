# Synthetic Latency Experiment

Target home for S1-S5 latency-only scenarios and related ablations.

Concrete scenarios should be YAML configs under `configs/`; shared mechanics
should live in `rwsim/`.

Smoke example:

```bash
python scripts/run_experiment.py --experiment synthetic_latency \
  --scenario s1_dominant --strategy v2_only --run
```
