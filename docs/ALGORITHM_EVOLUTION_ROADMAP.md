# Algorithm Evolution Roadmap

This roadmap is superseded by the flat policy architecture in
`docs/RWSIM_REFACTOR_PLAN.md`.

Current simulator policy work should use these names:

- `greedy_cost`
- `greedy_latency`
- `random`
- `ablation_lp_only`
- `ablation_lp_hedging`
- `routewise`

New algorithm changes should land in `rwsim/policies/routewise.py` unless a
second real policy needs the same helper. Do not recreate the historical stage
directory layout.
