# DEPRECATED — Phase 4 Smart Hedging Experiment

Historical workflow for replaying Phase 1 latency profiling data and evaluating
old smart hedging strategies on top of Phase 3 LP-Mix routing. This is an early
LP-CDF + smart-hedging iteration, not the current paper-facing method.

Do not use Phase 4 results as final paper claims unless the Phase 3 replay
interpolation choice has been audited and the experiment has been rerun. The
current paper-facing joint method is the tiered-capacity range-budget sidecar
with Hedge-ProbTarget / Explorer.

Run:

```bash
python -m experiments.latency_phase4.experiment --slo 3.0
```

Plot:

```bash
python -m experiments.latency_phase4.plots --results-dir experiment/results/latency_phase4
```
