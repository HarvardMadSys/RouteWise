# Phase 3 Latency Routing Experiment

Paper-used workflow for replaying Phase 1 latency profiling data and evaluating
LP-Mix against latency routing baselines.

Run:

```bash
python -m experiments.latency_phase3.experiment --slo 3.0
```

Plot:

```bash
python -m experiments.latency_phase3.plots --results-dir experiment/results/latency_phase3
```
