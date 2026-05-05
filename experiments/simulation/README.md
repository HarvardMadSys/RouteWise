# Simulation Experiment

Target home for the paper-facing simulator experiment harness. Per
`docs/EXPERIMENT_LAYOUT.md` §3.1 the canonical scenarios are S0-S3:

- **S0** — same latency, different cost (3 × S_A)
- **S1** — same cost, different latency (3 × S_A)
- **S2** — cost-latency tradeoff (3 × S_A)
- **S3** — full joint tier (S_A + S_Q + S_C)

Concrete scenarios live as YAML configs under `configs/`. Policy behavior
should be selected from the flat paper-name presets in `rwsim.policies`
(`greedy_cost`, `greedy_latency`, `random`, `ablation_lp_only`,
`ablation_lp_hedging`, `routewise`).

The full-sweep simulation runner lives under `suites/` and is exposed as
`routewise suite simulator_grid`.
