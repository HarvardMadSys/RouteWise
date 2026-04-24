# Scripts

Target home for command-line entrypoints.

Scripts should parse arguments and dispatch into `experiments/`; they should
not contain simulator, policy, plotting, or paper-specific research logic.
`scripts/experiments/` contains preserved full-sweep paper runners during the
engine migration. They write generated artifacts under `outputs/`.

Current target entry points:

- `run_experiment.py`: inspect, validate, and summarize config-driven experiments.
- `experiments/run_synthetic.py`: latency synthetic sweep.
- `experiments/run_sanity_check.py`: sanity-check sweep.
- `experiments/run_joint.py`: tiered joint-vs-two-layer sweep.
- `experiments/run_stress_tests.py`: tiered stress sweep.

Example smoke run:

```bash
python scripts/run_experiment.py --experiment tiered_capacity \
  --scenario s6_slow_q_trap --strategy joint_nohedge --run
```
