# `experiments/simulation/` Section Refactor — Implementation Plan

> Phased implementation doc. Goal: replace the universal `eval_grid` /
> `simulator_grid` infrastructure with one Python file per paper section,
> directly reflecting the four-layer experiment plan in
> `docs/EXPERIMENT_LAYOUT.md` §4 while keeping every intermediate commit
> runnable.

Last updated: 2026-05-05.

This document is self-contained. Do not require reading
`docs/RWSIM_REFACTOR_PLAN.md` or `docs/EXPERIMENT_LAYOUT.md` to execute it.

---

## 1. TL;DR

Replace these legacy infrastructure files:

```text
experiments/simulation/eval_grid.py            # universal "scenario × variant × dataset" grid factory
experiments/simulation/lp_budget_eval.py       # main grid runner glue
experiments/simulation/suites/run_simulator_grid.py   # full-sweep CLI suite
experiments/simulation/scenarios.py            # generic dict→ScenarioConfig builder
experiments/simulation/experiment.py           # CLI registry shim
experiments/simulation/materialize.py          # YAML→ScenarioConfig materializer
experiments/simulation/runner.py               # thin run_policy wrapper
experiments/simulation/configs/                # empty .gitkeep dir
experiments/suites.py                          # registry of "suites"
```

with this final shape:

```text
experiments/simulation/
  __init__.py
  common.py                # provider builders, workload loading, summary helpers
  cost_layer.py            # paper §3.2 — same latency / different cost
  latency_layer.py         # paper §3.3 — same cost / different latency (incl. overlap)
  hedging.py               # paper §3.3.4 — Hedging-Explorer
  end_to_end.py            # paper §5  — file present, NOT CLI-registered yet
  dataset_cache.py         # trace-cache helpers (kept; cross-section)
  presets.yaml             # policy preset definitions (kept)
```

Final CLI shape:

```bash
routewise simulator list
routewise simulator cost-layer
routewise simulator latency-layer
routewise simulator hedging
# end-to-end registers in a later phase, after empirical-distribution wire
```

**Implementation is phased, not a single big-bang commit.** Each phase
ships independently, runs, and is verified before the next starts. The
legacy infrastructure (`eval_grid.py` / `lp_budget_eval.py` /
`run_simulator_grid.py` / suite registry) stays alive as a parallel path
until the new sections demonstrably cover its functionality, then is
deleted in the final phase. See §8 for the phase plan.

Two invariants the phased migration preserves:

1. **Every commit on main is runnable.** `routewise simulator list` and
   `routewise suite simulator_grid` may both work for a few phases
   (during the parallel window). Tests stay green throughout.
2. **No public `NotImplementedError`-raising stubs.** If a section is
   not ready, its module file may exist (with scenario factories and
   helpers) but it does NOT define `main()` and it is NOT registered in
   the CLI. The grace period is "file present, no public surface" —
   never "public command that crashes".

This is **not a return to a 405-cell universal grid**. Notion's
`5 × 3 × 2 × 3 × 4 = 405` framing is an early factorial view of the
same experiments. The implementation stays per-section: each section
file declares its own internal sweeps (over distribution / overlap /
`p` / workload as applicable to that paper question). There is no
shared flat-grid runner.

---

## 2. Final shape

### 2.1 File tree after refactor

```text
experiments/simulation/
  __init__.py              # exports the four section modules
  common.py                # ~200-300 lines, see §4
  cost_layer.py            # paper-section module — CLI-registered
  latency_layer.py         # paper-section module — CLI-registered
  hedging.py               # paper-section module — CLI-registered
  end_to_end.py            # paper-section module — file present, NOT CLI-registered
  dataset_cache.py         # unchanged
  presets.yaml             # unchanged

routewise_cli/
  main.py                  # `simulator` subcommand replaces `suite`

tests/
  golden/
    cost_layer/scenarios.json
    latency_layer/<scenario>.json
    hedging/<scenario>.json
    # end_to_end/ added in a follow-up commit when scenarios are real
  unit/
    simulation/
      test_common.py
      test_cost_layer.py
      test_latency_layer.py
      test_hedging.py
      # test_end_to_end.py added when end_to_end ships
```

### 2.2 What gets deleted

| Path | Reason |
|---|---|
| `experiments/simulation/eval_grid.py` | universal grid replaced by per-section factories |
| `experiments/simulation/lp_budget_eval.py` | runner glue subsumed by per-section `main()` |
| `experiments/simulation/suites/` (whole dir) | no more "suite" concept |
| `experiments/simulation/scenarios.py` | YAML dict builder; no YAML scenarios |
| `experiments/simulation/experiment.py` | CLI registry shim subsumed by per-section CLI |
| `experiments/simulation/materialize.py` | YAML→ScenarioConfig; no YAML scenarios |
| `experiments/simulation/runner.py` | re-exports `rwsim.runner.run_policy`; callers use `rwsim.runner` directly |
| `experiments/simulation/configs/` | empty dir |
| `experiments/suites.py` | no suite registry |
| `experiments/_configs.py` | only used by `experiment.py` shim |
| `tests/golden/simulator_grid/` (if any) | legacy goldens |
| `tests/golden/calibrated/` | MM25 simulator goldens already off paper path |

Existing references to these paths in `routewise_cli/main.py`, `README.md`,
`docs/EXPERIMENT_LAYOUT.md`, `docs/RWSIM_REFACTOR_PLAN.md`,
`docs/REPRODUCIBILITY.md` all update to the new shape.

### 2.3 What stays

| Path | Note |
|---|---|
| `experiments/simulation/dataset_cache.py` | trace caching, cross-section utility |
| `experiments/simulation/presets.yaml` | policy preset definitions, loaded by `rwsim.policies.build_policy` |
| `experiments/real_evaluation/` | separate domain, untouched |
| `experiments/offline_stage/` | separate domain, untouched |
| `experiments/estimator_ablation/` | separate domain, untouched |

---

## 3. Section module contract

Every CLI-registered `experiments/simulation/<section>.py` exposes the
same runnable surface:

```python
# Runnable section module surface

SECTION_NAME: str = "cost_layer"          # CLI subcommand name (kebab-case)

def list_scenarios() -> tuple[str, ...]:
    """Return scenario names this section produces."""

def make_scenarios() -> dict[str, ScenarioConfig]:
    """Return all scenarios for this section keyed by name."""

def policies_for_section() -> tuple[str, ...]:
    """Return policy presets relevant to this section.

    Default = all six paper-name presets (greedy_cost, greedy_latency,
    random, ablation_lp_only, ablation_lp_hedging, routewise). Sections
    may narrow this set if some baselines are not meaningful (e.g. cost-
    layer can drop greedy_latency since latency is held constant).
    """

def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``routewise simulator <section>``.

    Parses args, dispatches scenarios × policies × seeds, writes per-section
    output under ``outputs/simulation/<section>/`` and prints a summary
    table to stdout. Returns process exit code.
    """
```

The CLI is wired by `routewise_cli/main.py` looking up
`experiments.simulation.<section>.main` based on the subcommand.
Planned-but-unregistered modules such as `end_to_end.py` may expose only
`SECTION_NAME`, constants, and scenario factory placeholders; they must
not expose `main()` until they are runnable and CLI-registered in the same
phase.

### 3.1 Per-section content

Numbers below are sourced from Notion (Simulation / Evaluation pages,
2026-05-04). The doc-internal name for the universal `p` sweep is
`P_SWEEP = (0.0, 0.25, 0.5, 0.75, 1.0)`.

**Simulator-wide invariant: explorer is OFF.** The Notion plan is
explicit: "ignore explorer in simulator, keep it in real experiment."
This means the simulator never runs the `routewise` preset
(`hedging=probability_target, explorer=true`). Instead it runs:

- `ablation_lp_only` — LP body, no hedge
- `ablation_lp_hedging` — LP body + hedge, no explorer feedback

Plus the relevant baselines (`greedy_cost`, `greedy_latency`, `random`)
filtered per section. The full `routewise` preset is exercised only in
`experiments/real_evaluation/`.

#### `cost_layer.py`

Paper question: "When latency is held equal, does RouteWise minimize cost?"

Constants:
- All providers share identical TTFT distribution with **P50 = 300ms**.
- API input cost ratio: **`$1 / $2 / $4`** per million input tokens
  (provider A / B / C).
- API output cost ratio: **`$5 / $10 / $20`** per million output tokens
  (5× the input price for each provider).
- Workload: `sharegpt_burstgpt` (the canonical 30-day combined trace).
- `p` sweep: `P_SWEEP = (0.0, 0.25, 0.5, 0.75, 1.0)`.
- Policy filter: drop `greedy_latency` (no latency signal across providers).
  Include `offline`, the offline cost-only baseline with full trace
  knowledge. It is section-local, not an online `Policy`.

Scenarios:

| Scenario | Provider setup |
|---|---|
| `cost_layer_uniform` | 3 × S_A, all `Uniform(0.5×P50, 1.5×P50)` = `[150, 450]ms`, input `$1/$2/$4`, output `$5/$10/$20` |
| `cost_layer_normal` | 3 × S_A, all `Normal(P50, 0.3×P50)` = `Normal(300, 90)`, input `$1/$2/$4`, output `$5/$10/$20` |
| `cost_layer_heavy_tail` | 3 × S_A, all `LogNormal(ln(P50), 0.5)`, input `$1/$2/$4`, output `$5/$10/$20` |
| `cost_layer_real_world` | 3 × S_A, all use the same `rw8_pooled` empirical Qwen3/OpenRouter TTFT distribution, input `$1/$2/$4`, output `$5/$10/$20` |
| `cost_layer_quota_q1` | 1 × S_Q + 2 × S_A, base distribution = `LogNormal` |
| `cost_layer_quota_q2` | 2 × S_Q + 1 × S_A |
| `cost_layer_quota_q3` | 3 × S_Q + 1 × S_A |
| `cost_layer_quota_q4` | 4 × S_Q + 1 × S_A |
| `cost_layer_concurrency_c1` | 1 × S_C + 2 × S_A |
| `cost_layer_concurrency_c2` | 2 × S_C + 1 × S_A |
| `cost_layer_concurrency_c3` | 3 × S_C + 1 × S_A |
| `cost_layer_concurrency_c4` | 4 × S_C + 1 × S_A |

The `_q1..q4` and `_c1..c4` scenarios answer the subscription-count
optimization question. `cost_layer_real_world` is the real-world counterpart
for §1.1.4; it keeps the cost-layer invariant by using one pooled empirical
latency distribution for all three S_A providers.

#### `latency_layer.py`

Paper question: "When cost is held equal, does RouteWise pick the fast
provider? How does it degrade as the latency distributions become harder
to distinguish?"

Constants:
- All 3 providers same cost (any value; arbitrary because not used).
- Provider P50 ladder: **`100ms / 300ms / 1000ms`** (10× geometric
  ratio; provider names `fast` / `medium` / `slow`).
- Workload: `sharegpt_burstgpt`.
- Policy filter: drop `greedy_cost` (no cost signal).
- This section runs **without hedging**; hedging story lives in `hedging.py`.

Two axes:
- `family`: `uniform` / `normal` / `heavy_tail` (LogNormal). `real_world`
  is added in the EmpiricalDistribution wire follow-up.
- `overlap`: **`no_overlap` / `half_overlap`** (only two, per Notion §2.1).

Construction (per Notion's family rules):

| family | no_overlap | half_overlap |
|---|---|---|
| uniform | `Uniform(0.5×P50, 1.5×P50)` per provider — supports `[50,150] / [150,450] / [500,1500]` are non-overlapping in pairs | adjacent supports overlap by half of the smaller window. e.g. fast `[50,150]`, medium `[100,200]` — half of fast's window is in medium |
| normal | `Normal(P50, 0.3×P50)` per provider; gap = 3σ → `Normal(100,30) / Normal(300,90) / Normal(1000,300)` already gives near-zero density overlap because gap ≫ 3σ for the 100→300 step | tighten the σ relationship until the 100/300 pair has substantial mass overlap; document the σ choice in `latency_layer.py` |
| heavy_tail | `LogNormal(ln(P50), 0.5)` per provider; the 100/300/1000 P50 ladder gives tails that brush each other but bodies stay distinct | tighten σ_log so the 100/300 bodies overlap |

`OverlapLevel` is qualitative across families. The label is
paper-storytelling, not a numerically comparable metric.

The overlap construction helper lives **in `latency_layer.py`**, not
`common.py`. Function name: `_construct_overlap_distributions(family,
overlap)` (module-private). It is the only place that knows the
family-specific σ / window-width formulas. If a future section needs the
same construction (e.g. `hedging.py` reusing latency_layer scenarios),
it imports from `latency_layer`, not from `common`.

3 families × 2 overlap = **6 scenarios** named
`latency_layer_<family>_<overlap>` (e.g. `latency_layer_uniform_no_overlap`).

`p` sweep: optional. The dominant signal in latency layer is provider
selection / TTFT, not the cost-budget knob. Default to `p = 0.75` and
add a sweep only if a paper figure asks for it.

#### `hedging.py`

Paper question: "Does hedging cut P99 with bounded cost overhead?"

Constants:
- Reuses `latency_layer`'s `heavy_tail` family with `half_overlap`. Two
  scenarios planned (per Notion §2.2):
  - `hedging_heavy_tail` — synthetic heavy-tail (3 providers, P50
    100/300/1000ms, LogNormal, half_overlap)
  - `hedging_real_world` — added with EmpiricalDistribution wire
- Workload: `sharegpt_burstgpt`.
- Policy comparison: **`ablation_lp_only` vs `ablation_lp_hedging`**
  (LP-without-hedge vs LP-with-hedge). `routewise` (full explorer)
  intentionally not run — explorer off in simulator.
- `p` sweep: `P_SWEEP` so cost-overhead-vs-P99-reduction tradeoff is
  visible across the cost-budget axis.

Section-specific output:
- per-scenario hedge_rate, P99 / SLO reduction vs `ablation_lp_only`
  baseline, cost multiplier vs `ablation_lp_only` baseline.

#### `end_to_end.py`

Paper question: "Does RouteWise win on the real-world workload?"

**This module ships in earlier phases as a planning file with constants
and comments for the intended scenarios. It does NOT define `main()` and the
`routewise_cli/main.py` does NOT register the `end-to-end` subcommand
until the empirical-distribution wire phase lands.**

Concretely: `end_to_end.py` exposes `SECTION_NAME` plus documented
constants for the planned RW3 / RW8 scenarios. Until empirical profiles
are wired, it may omit `make_scenarios()` entirely or return an empty
mapping, but it does **not** export a `main` symbol. There is no
`routewise simulator end-to-end` until the phase that lands real form.

Planned scenarios (for the follow-up commit that lands real form):

| Scenario | Provider pool | Workload |
|---|---|---|
| `end_to_end_rw3_with_hedging` | RW3 = [WandB, DeepInfra, Novita] (one S_A, one S_Q, one S_C constructed from Qwen3-235B 24h profiles) | sharegpt_burstgpt |
| `end_to_end_rw3_no_hedge` | same providers; ablation column for paper Table | sharegpt_burstgpt |
| `end_to_end_rw8_with_hedging` | RW8 = 8 OR providers + 2 (one S_Q, one S_C) | sharegpt_burstgpt |
| `end_to_end_rw8_no_hedge` | same; ablation | sharegpt_burstgpt |

The no-hedge column is included per Notion's intended Table-1-style
ablation (LP-only vs LP+hedge across real-world pools).

Policy comparison: `ablation_lp_only` vs `ablation_lp_hedging` (still
no explorer). Plus baselines `greedy_cost`, `greedy_latency`, `random`.

Dependencies blocking the real form:
1. EmpiricalDistribution wired to provider config.
2. Qwen3-235B 24h parquet samples under
   `experiments/simulation/profiles/`.
3. `pools.yaml` defining RW3 / RW8 → per-provider sample slicing.

Until all three land, `end_to_end.py` is shape-only documentation; do
not plumb it into the CLI.

---

## 4. `common.py` contract

The shared module exposes:

```python
# experiments/simulation/common.py

from rwsim.metrics import PerRequestRecord, Run
from rwsim.policies import build_policy
from rwsim.runner import run_policy
from rwsim.schemas import ProviderTier, Request, ScenarioConfig
from rwsim.world.providers import TieredProvider
from rwsim.world.distributions import LogNormal, Normal, Uniform


# ---- Provider builders -----------------------------------------------------

def make_api_provider(
    name: str,
    *,
    cost_per_token: float,
    p50_ms: float,
    p99_ms: float,
    family: str = "lognormal",
    family_params: dict | None = None,
) -> TieredProvider:
    """Build a pay-per-token (S_A) provider with the requested distribution."""

def make_quota_provider(
    name: str,
    *,
    quota_size: int,
    quota_window_sec: float,
    p50_ms: float,
    p99_ms: float,
    family: str = "lognormal",
) -> TieredProvider:
    """Build an S_Q provider with quota and zero marginal cost."""

def make_concurrency_provider(
    name: str,
    *,
    concurrency_limit: int,
    p50_ms: float,
    p99_ms: float,
    family: str = "lognormal",
) -> TieredProvider:
    """Build an S_C provider with a concurrency cap and zero marginal cost."""


# ---- Distribution helpers --------------------------------------------------

P_SWEEP: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
"""Canonical p-knob sweep used by every section that varies p."""

P50_LADDER_MS: tuple[float, float, float] = (100.0, 300.0, 1000.0)
"""Canonical fast / medium / slow P50 used by latency_layer and any
section that needs a multi-provider latency ladder."""

COST_RATIO_PER_MILLION: tuple[float, float, float] = (1.0, 2.0, 4.0)
"""Canonical input-token price ratio for the 3-S_A cost-layer scenarios."""

OUTPUT_COST_MULTIPLIER: float = 5.0
"""Output-token price is 5× input-token price in cost-layer scenarios."""

# NOTE: `construct_overlap_distributions` lives in `latency_layer.py`,
# not here. Overlap construction is a paper-section-specific concept;
# common.py stays cross-section-neutral.


# ---- Workload loading ------------------------------------------------------

def load_sharegpt_burstgpt_workload(
    *,
    duration_sec: float | None = None,
    seed: int = 0,
) -> list[Request]:
    """Load the canonical paper workload, optionally truncated."""


# ---- Run dispatch ----------------------------------------------------------

def run_section(
    scenarios: dict[str, ScenarioConfig],
    policies: tuple[str, ...],
    seeds: tuple[int, ...] = (42, 43, 44),
    *,
    workload_loader=load_sharegpt_burstgpt_workload,
    output_dir: Path,
) -> dict[str, dict[str, list[Run]]]:
    """Cartesian product of scenarios × policies × seeds. Writes per-cell
    Run records to disk and returns the grouped result for summary."""


# ---- Section CLI helper ----------------------------------------------------

def section_main(
    *,
    section_name: str,
    list_scenarios,
    make_scenarios,
    policies_for_section,
    extra_summary_columns: tuple[str, ...] = (),
) -> Callable[[list[str] | None], int]:
    """Build a uniform ``main()`` for a section module.

    Each section just calls ``main = section_main(section_name="cost_layer",
    list_scenarios=list_scenarios, make_scenarios=make_scenarios,
    policies_for_section=policies_for_section)`` and exposes the resulting
    function. Argparse, dispatch, output paths, and summary printing all
    live inside ``section_main``.
    """
```

Aim for `common.py` ≤ 350 lines. If it crosses, split into
`common/{providers,distributions,workload,dispatch}.py`.

---

## 5. CLI surface

### 5.1 New surface

```bash
routewise simulator list
routewise simulator cost-layer       [--policy P] [--scenario S] [--seed N] [--p P] [--output-dir DIR]
routewise simulator latency-layer    [...same...]
routewise simulator hedging          [...same...]
```

`routewise simulator end-to-end` is **deliberately not registered** in
this refactor. The `end_to_end.py` module exists (file, scenario
factories, doc) but `routewise_cli/main.py` does not add the
`end-to-end` subparser. The follow-up commit that lands real RW3 / RW8
form also wires the CLI subcommand at the same time.

`--policy`, `--scenario`, `--seed` are repeatable; default is the
section's full set. `--p` is a repeatable float in [0, 1]; default is
`P_SWEEP` for sections that vary `p`, single value `0.75` for sections
that don't.

`routewise simulator list` prints:

```
Sections (paper-aligned):
  cost-layer        paper §3.2   — same latency / different cost
  latency-layer    paper §3.3    — same cost / different latency (no hedging)
  hedging          paper §3.3.4  — Hedging gain on heavy-tail / real-world

Scenarios per section:
  cost-layer:        cost_layer_uniform, cost_layer_normal, cost_layer_heavy_tail,
                     cost_layer_real_world, cost_layer_quota_q1..q4,
                     cost_layer_concurrency_c1..c4
  latency-layer:     latency_layer_uniform_no_overlap, latency_layer_uniform_half_overlap,
                     latency_layer_normal_no_overlap, latency_layer_normal_half_overlap,
                     latency_layer_heavy_tail_no_overlap, latency_layer_heavy_tail_half_overlap
  hedging:           hedging_heavy_tail
```

### 5.2 Removed surface

```bash
routewise suite                  # the whole subcommand goes away
routewise suite simulator_grid   # subsumed by the four section commands
routewise list --suites          # subsumed by `routewise simulator list`
routewise validate               # was a YAML scenario validator; no YAML scenarios
routewise describe               # was a YAML scenario describer; no YAML scenarios
routewise run                    # was generic --experiment X --scenario Y --policy Z; subsumed
```

### 5.3 `routewise_cli/main.py` shape

```python
parser = argparse.ArgumentParser(prog="routewise")
sub = parser.add_subparsers(dest="command", required=True)

simulator = sub.add_parser("simulator", help="Run paper-section simulator experiments.")
sim_sub = simulator.add_subparsers(dest="section", required=True)
sim_sub.add_parser("list")
for section in ("cost-layer", "latency-layer", "hedging"):
    p = sim_sub.add_parser(section)
    p.add_argument("--policy", action="append")
    p.add_argument("--scenario", action="append")
    p.add_argument("--seed", action="append", type=int)
    p.add_argument("--p", action="append", type=float, dest="p_values")
    p.add_argument("--output-dir", type=Path)

# end-to-end is intentionally not registered here. It will be added in
# the commit that also lands the real RW3 / RW8 scenarios.

# Future: routewise live, routewise offline, routewise estimator-ablation
```

The dispatch is one `import experiments.simulation.<section>` + call its
`main()` with the parsed args.

---

## 6. Test and golden migration

### 6.1 Goldens

Layout:

```text
tests/golden/
  cost_layer/                # appears in Phase 0/1
    scenarios.json           # smoke golden: all scenarios, all policies, 32 requests
  latency_layer/             # appears in Phase 2
    latency_layer_uniform_no_overlap.json
    latency_layer_uniform_half_overlap.json
    latency_layer_normal_no_overlap.json
    latency_layer_normal_half_overlap.json
    latency_layer_heavy_tail_no_overlap.json
    latency_layer_heavy_tail_half_overlap.json
  hedging/                   # appears in Phase 3
    hedging_heavy_tail.json
  end_to_end/                # appears in Phase 4
    end_to_end_rw3_with_hedging.json
    end_to_end_rw3_no_hedge.json
    end_to_end_rw8_with_hedging.json
    end_to_end_rw8_no_hedge.json
```

**Goldens are added phase-by-phase, not all at once.** Phase N adds only
the goldens for the section that becomes runnable in Phase N. Legacy
goldens (`tests/golden/simulation/`, `tests/golden/calibrated/`,
`tests/golden/stress/`, etc., if any still exist) are deleted only in
Phase 5 alongside the legacy code that produces them.

`tests/golden_capture.py` runs `make_scenarios()` for each section,
captures the canonical `Run` record digest per (scenario, policy, seed),
writes `tests/golden/<section>/<scenario>.json`. Compare mode diffs the
captured payload against the committed file.

`tests/golden/simulation/`, `tests/golden/calibrated/`,
`tests/golden/stress/` are removed by previous commits or by this
refactor's prep step — confirm none remain after the cleanup commit.

### 6.2 Architecture tests

Add to `tests/test_architecture_scaffold.py`:

```python
def test_simulator_section_modules_present(self) -> None:
    # CLI-registered sections must expose the full surface including main().
    for section in ("cost_layer", "latency_layer", "hedging"):
        module = importlib.import_module(f"experiments.simulation.{section}")
        for name in ("SECTION_NAME", "list_scenarios", "make_scenarios",
                     "policies_for_section", "main"):
            self.assertTrue(hasattr(module, name), f"{section}.{name} missing")

    # end_to_end must exist as a planning module but is intentionally
    # not CLI-registered. It exposes scenario factories but not main().
    end_to_end = importlib.import_module("experiments.simulation.end_to_end")
    for name in ("SECTION_NAME", "list_scenarios", "make_scenarios"):
        self.assertTrue(hasattr(end_to_end, name), f"end_to_end.{name} missing")

def test_end_to_end_is_not_cli_registered(self) -> None:
    # Until empirical-distribution wire and RW3/RW8 profiles land,
    # `routewise simulator end-to-end` must NOT be a runnable command.
    cli_source = (ROOT_DIR / "routewise_cli" / "main.py").read_text(encoding="utf-8")
    self.assertNotIn('"end-to-end"', cli_source)
    self.assertNotIn("'end-to-end'", cli_source)

def test_legacy_simulator_infrastructure_is_gone(self) -> None:
    deleted = (
        ROOT_DIR / "experiments" / "simulation" / "eval_grid.py",
        ROOT_DIR / "experiments" / "simulation" / "lp_budget_eval.py",
        ROOT_DIR / "experiments" / "simulation" / "suites",
        ROOT_DIR / "experiments" / "simulation" / "scenarios.py",
        ROOT_DIR / "experiments" / "simulation" / "experiment.py",
        ROOT_DIR / "experiments" / "simulation" / "materialize.py",
        ROOT_DIR / "experiments" / "simulation" / "runner.py",
        ROOT_DIR / "experiments" / "simulation" / "configs",
        ROOT_DIR / "experiments" / "suites.py",
        ROOT_DIR / "experiments" / "_configs.py",
    )
    for path in deleted:
        self.assertFalse(path.exists(), f"{path} should be deleted")

def test_no_legacy_simulator_imports(self) -> None:
    forbidden = (
        "from experiments.simulation.eval_grid",
        "from experiments.simulation.lp_budget_eval",
        "from experiments.simulation.materialize",
        "from experiments.suites",
        "import experiments.suites",
    )
    for dirname in ("rwsim", "experiments", "routewise_cli", "tests"):
        for path in (ROOT_DIR / dirname).rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, source, f"{path} still has {token}")
```

### 6.3 Unit tests

`tests/unit/simulation/test_<section>.py` minimum coverage:

1. `list_scenarios()` returns a non-empty tuple with no duplicates.
2. `make_scenarios()` keys equal `set(list_scenarios())`.
3. Each scenario has at least one provider and a strictly positive SLO.
4. `policies_for_section()` is a non-empty subset of all paper-name policies.
5. (Section-specific) provider invariants:
   - cost_layer: all S_A providers share the same TTFT distribution; input-token
     cost ratio is `(1, 2, 4) * 1e-6`; output-token cost ratio is
     `(5, 10, 20) * 1e-6`.
   - latency_layer: all providers have the same cost; P50 ladder is `(100, 300, 1000)` ms.
   - hedging: at least one provider's distribution is heavy-tail (LogNormal).
   - end_to_end: planning module importable; module does not define `main()` before CLI registration.

---

## 7. Locked decisions (Notion 2026-05-04 + Murphy 5/4)

These are the boundary calls that drove the numbers in §3.1. They are
locked; do not re-litigate during implementation. Recorded here so a
reviewer can trace each parameter back to a decision.

| Decision | Value | Source |
|---|---|---|
| cost_layer input cost ratio | `$1 / $2 / $4` per million input tokens | Notion §1.1 |
| cost_layer output cost ratio | `$5 / $10 / $20` per million output tokens (5× input) | Murphy |
| cost_layer common P50 | `300ms` | Notion §1.1 |
| cost_layer subscription-count sweep | `{1, 2, 3, 4}` for both quota and concurrency | Murphy + Juncheng 5/4 |
| cost_layer real-world distribution | `cost_layer_real_world` uses `rw8_pooled` empirical Qwen3/OpenRouter TTFT for all three S_A providers | Murphy |
| latency_layer P50 ladder | `100ms / 300ms / 1000ms` (10× geometric) | Notion §2.1 + 5/4 stage1 spec |
| latency_layer overlap regimes | `no_overlap` + `half_overlap` only (two regimes, not three) | Notion §2.1 |
| latency_layer p sweep | not the dominant axis here; default `p = 0.75`. Add sweep only if a paper figure asks | Murphy |
| hedging cells | heavy_tail (synthetic) + real_world (later) — two scenarios per Notion §2.2 | Notion §2.2 |
| hedging policies | `ablation_lp_only` vs `ablation_lp_hedging` (no `routewise` preset) | derived from explorer-off invariant |
| end_to_end no-hedge column | yes — keep as ablation column | Murphy + Juncheng 5/4 |
| end_to_end CLI surface | not registered until empirical-distribution wire + RW3 / RW8 profiles land | Murphy |
| simulator explorer mode | OFF — explorer is real-eval-only | Notion Evaluation page |
| canonical p sweep | `(0.0, 0.25, 0.5, 0.75, 1.0)` | Notion Evaluation page |
| canonical workload | `sharegpt_burstgpt` for paper figures; `freeinference` and `enterprise (rednote)` available as additional sweeps | Notion |

If implementation discovers a contradiction with these values, stop and
flag it; do not silently retune.

---

## 8. Phase plan

Implementation is **phased**, not a single big-bang commit. Each phase
is its own commit, runs end-to-end, ships golden updates only for what
that phase covers. The legacy infrastructure stays alive in parallel
until coverage is demonstrated, then is deleted in the final phase.

Invariants enforced across all phases:

- `pytest -m "not slow"` and golden-compare stay green after every phase.
- No `NotImplementedError`-raising public CLI surface at any phase
  boundary. Sections without `main()` are not registered.
- No "compatibility shims" that pretend to be the new path while
  delegating to the old. Either a section is real, or it is unregistered.

### Phase 0 — Skeleton + `routewise simulator list` + cost-layer module

Lands together:

- `experiments/simulation/common.py` — provider builders, workload
  loader, `P_SWEEP`, `COST_RATIO_PER_MILLION` constants, and
  `run_section()` dispatch.
- `experiments/simulation/cost_layer.py` — full module per §3.1, with
  `main()` defined.
- Planning modules (no `main()`, no CLI registration) for
  `experiments/simulation/{latency_layer,hedging,end_to_end}.py` —
  each exposes `SECTION_NAME`, `list_scenarios()`, and
  `make_scenarios()` so architecture tests can find them. End-to-end
  may list planned scenario names, but still has no public `main()`.
- `routewise_cli/main.py` adds the `simulator` subcommand and
  registers only `cost-layer` + `list`.
- Architecture tests in §6.2 (the `test_end_to_end_is_not_cli_registered`
  one is already meaningful here because end-to-end is unregistered).

Legacy stays alive: `eval_grid.py`, `lp_budget_eval.py`,
`run_simulator_grid.py`, `routewise suite simulator_grid` — all
unchanged. New `simulator` subcommand and old `suite` subcommand
co-exist for these phases.

Goldens this phase: `tests/golden/cost_layer/scenarios.json` only. It
is a smoke golden captured by `tests/golden_capture.py --families
cost_layer`: all cost-layer scenarios and policy names, including the
section-local `offline` baseline, 3 seeds, fixed 32-request workload
prefix. No deletion of legacy goldens yet.

### Phase 1 — Cost-layer fully runnable

Same code surface as Phase 0; this is the "verify in real use" phase.
Run `routewise simulator cost-layer` on a bounded trace slice, regenerate
the cost-layer smoke golden, and compare it with
`tests/golden_capture.py --mode compare --families cost_layer`. Full
30-day paper runs are not committed as normal goldens because the trace
is multi-GB and the full section sweep is intentionally heavy.

When preparing paper figures, run the full trace out-of-band and compare
high-level metrics against the smoke golden shape plus any equivalent
legacy `simulator_grid` output still available. Fix anything that drifts
for code reasons; commit with either zero new code or the minimal fix.

If §7 numbers (P50 = 300ms, cost = $1/$2/$4, etc.) produce surprising
results, stop and re-litigate before proceeding to Phase 2.

### Phase 2 — Latency-layer runnable

Lands:

- `experiments/simulation/latency_layer.py` full form, including
  `_construct_overlap_distributions()`.
- `routewise_cli/main.py` registers `latency-layer` subcommand.
- `tests/golden/latency_layer/*.json` for the 6 `family × overlap`
  cells.
- Unit test `tests/unit/simulation/test_latency_layer.py` for
  per-section invariants from §6.3.

Legacy still untouched.

### Phase 3 — Hedging runnable

Lands:

- `experiments/simulation/hedging.py` full form for the `heavy_tail`
  cell (synthetic). The `real_world` cell is left for a later phase
  alongside EmpiricalDistribution wire.
- `routewise_cli/main.py` registers `hedging` subcommand.
- `tests/golden/hedging/hedging_heavy_tail.json`.
- Unit test.

After Phase 3, three of the four sections are runnable. Legacy still
untouched. Coverage check: every paper figure that previously came
from `simulator_grid` has an equivalent CLI invocation in the new
shape (modulo end-to-end).

### Phase 4 — Empirical distribution + end-to-end registration

Lands:

- EmpiricalDistribution wired to provider config
  (`ttft_distribution: type=empirical, source=...`).
- `experiments/simulation/profiles/` populated with `qwen3_24h.parquet`
  + `pools.yaml`.
- `experiments/simulation/end_to_end.py` gets a `main()` definition
  (using the new empirical wire).
- `routewise_cli/main.py` registers `end-to-end` subcommand.
- `latency_layer.py` and `hedging.py` add their `real_world`-family
  scenarios.
- `tests/golden/end_to_end/*.json` for RW3 and RW8 cells.
- Update `test_end_to_end_is_not_cli_registered` test to its
  "after phase 4" form (allow `end-to-end` in main.py).

Legacy still untouched — but **all 4 sections are now CLI-registered
and produce paper-relevant figures**. This is the coverage gate before
Phase 5.

### Phase 5 — Delete legacy infrastructure

Only after Phase 4 closes the coverage gap. Lands:

- Delete the §2.2 list:
  `eval_grid.py`, `lp_budget_eval.py`, `experiments/simulation/suites/`,
  `scenarios.py`, `experiments/simulation/experiment.py`,
  `materialize.py`, `experiments/simulation/runner.py`, `configs/`,
  `experiments/suites.py`, `experiments/_configs.py`.
- Remove `routewise suite` / `validate` / `describe` / `run`
  subcommands from `routewise_cli/main.py`.
- Add the §6.2 `test_legacy_simulator_infrastructure_is_gone` and
  `test_no_legacy_simulator_imports` tests.
- Delete legacy `tests/golden/` subdirs that referenced
  `simulator_grid` / `eval_grid`.
- Update `README.md`, `docs/EXPERIMENT_LAYOUT.md`,
  `docs/RWSIM_REFACTOR_PLAN.md`, `docs/REPRODUCIBILITY.md`.

If Phase 5 surfaces a missing capability (e.g. some legacy suite
feature has no equivalent in the section files), pause and add the
capability to the right section before deleting. **Do not delete
legacy until parity is real.**

### Out-of-band follow-ups (any phase, independent)

- Optional `hedging.py` extension to add `real_world` cell once
  Phase 4 lands EmpiricalDistribution.
- Plot scaffolding under `plots/{cost_layer, latency_layer, hedging,
  end_to_end}/` — can land any time after the relevant section's
  Phase, consumes `Run` records produced by `routewise simulator <section>`.

---

## 9. Out of scope

| Item | Reason |
|---|---|
| `experiments/real_evaluation/` | separate domain, separate CLI namespace later (`routewise live ...`) |
| `experiments/offline_stage/` | separate domain (`routewise offline ...` later) |
| `experiments/estimator_ablation/` | separate domain (`routewise estimator-ablation ...` later) |
| `rwsim/` core | unchanged; this refactor is experiment-layer only |
| Plot code | `plots/<section>/*.py` consumes `Run` records produced by section runs; plot scaffolding is a separate concern |
| Real-world data migration to `results/cached/` | pre-existing open item, not blocked by this refactor |
| Prefix cache discount | deferred per Murphy |

---

## 10. Sign-off checklist

This document is the source of truth for the section refactor if all of
the following hold:

- §2 final shape matches the eventual file tree (4 section files; 3
  CLI-registered after Phase 3, all 4 after Phase 4)
- §3.1 numbers match Notion: P50 ladder `100/300/1000ms`, cost ratio
  input `$1/$2/$4` and output `$5/$10/$20`, overlap regimes `no_overlap` + `half_overlap` only,
  simulator runs explorer OFF
- §5 final CLI surface registers exactly `cost-layer`, `latency-layer`,
  `hedging`, `end-to-end`, `list` — and intermediate phases register
  the prefix that is genuinely runnable
- §6.2 architecture tests are added phase-by-phase as the relevant
  surface lands (e.g. `test_end_to_end_is_not_cli_registered` is added
  in Phase 0 and updated/removed only in Phase 4)
- §7 locked decisions are not re-litigated during implementation
- §8 phase plan is followed: each phase commits independently, runs
  tests green, leaves legacy alive until Phase 5
- No `NotImplementedError`-raising public CLI command surfaces at any
  phase boundary
- No return to a flat 405-cell universal grid runner

Disagreements: name the section + cite Notion / 5-4 meeting record;
propose the smallest alternative that preserves the per-section
invariant.
