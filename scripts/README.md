# Scripts

Target home for thin command-line entrypoints.

Scripts should parse arguments and dispatch into `experiments/`; they should
not contain simulator, policy, plotting, or paper-specific research logic.

Current target entry point:

- `run_experiment.py`: inspect, validate, and summarize config-driven experiments.

Example smoke run:

```bash
python scripts/run_experiment.py --experiment tiered_capacity \
  --scenario s6_slow_q_trap --strategy joint_nohedge --run
```
