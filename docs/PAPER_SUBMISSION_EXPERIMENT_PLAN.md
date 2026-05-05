# Paper Submission Experiment Plan

This plan now relies on the current simulator architecture documented in
`docs/ARCHITECTURE.md` and `docs/EXPERIMENT_LAYOUT.md`.

## Simulator Policies

Run the paper simulator with:

- `greedy_cost`
- `greedy_latency`
- `random`
- `ablation_lp_only`
- `ablation_lp_hedging`
- `routewise`

The RouteWise line is the complete LP + hedging + explorer policy. The two
ablation presets isolate LP-only and LP + hedging behavior.

## Workloads

Main simulator results use trace-driven workloads through
`experiments.simulation.lp_budget_eval.generate_scenario_workload()`.

## Suites

Use:

```bash
routewise suite simulator_grid
routewise suite mm25_baselines
routewise suite stress
```

Single scenario smoke run:

```bash
routewise run simulation --scenario s6_slow_q_trap --policy routewise --seed 42
```
