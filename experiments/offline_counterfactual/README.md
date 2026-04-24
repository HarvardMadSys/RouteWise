# Offline Counterfactual Experiment

Paper-used workflow for replaying OpenRouter evaluation logs while excluding a
dominant provider. The implementation lives here; legacy
`legacy.experiment.scripts.simulate.*` modules are compatibility wrappers.

Run:

```bash
python -m experiments.offline_counterfactual.experiment \
  --csv sample_data/evaluation_log_sample.csv \
  --exclude WandB \
  --slo-ms 2000 \
  --n-trials 1 \
  --output-dir /tmp/rwsim-counterfactual-smoke
```

Plot:

```bash
python -m experiments.offline_counterfactual.plots \
  --results-dir /tmp/rwsim-counterfactual-smoke
```
