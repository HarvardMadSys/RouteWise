# Scripts

Target home for thin command-line entrypoints.

Scripts should parse arguments and dispatch into `experiments/`; they should
not contain simulator, policy, plotting, or paper-specific research logic.

Current target entry point:

- `run_experiment.py`: inspect, validate, and summarize config-driven experiments.
