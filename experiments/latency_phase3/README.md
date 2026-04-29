# DEPRECATED — Phase 3 Latency Routing Experiment

Historical workflow for replaying Phase 1 latency profiling data and evaluating
LP-Mix against latency routing baselines. This is an early LP-CDF / LP-Mix
iteration, not the current paper-facing method.

Do not use Phase 3 results as final paper claims unless the replay interpolation
choice has been audited and the experiment has been rerun. The policy profiles
are time-causal, but realized latency uses nearest-neighbor trace replay, which
can choose future samples for evaluation. The current paper-facing joint method
is the tiered-capacity range-budget sidecar.

Run:

```bash
python -m experiments.latency_phase3.experiment --slo 3.0
```

Plot:

```bash
python -m experiments.latency_phase3.plots --results-dir experiment/results/latency_phase3
```
