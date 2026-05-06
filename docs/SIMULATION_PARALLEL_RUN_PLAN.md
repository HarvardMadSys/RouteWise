# Section Simulator Parallel Run Plan

## 1. TL;DR

Add cell-level parallelism to the section-based simulator.

The unit of parallel work is one cell:

```text
(scenario, policy, seed)
```

Examples:

```text
(cost_layer_uniform, greedy_cost, 42)
(cost_layer_real_world, ablation_lp_only_p50, 43)
```

The public contract:

```bash
routewise simulator cost-layer --jobs 1   # current behavior, deterministic default
routewise simulator cost-layer --jobs 8   # run independent cells in parallel
```

`--jobs 1` remains the default and should produce numerically identical
summaries relative to the current serial path. `--jobs N` is an
execution-mode change, not an algorithm change.

This plan is independent of any solver work. It does not change RouteWise LP
semantics, offline-baseline semantics, or any future offline-oracle work.

## 2. Why This Exists

The previous empirical-distribution work made each cell lighter:

- `EmpiricalDistribution` caches summary statistics and sorted samples.
- `RunAggregator` avoids retaining millions of per-request records.
- `run_section()` writes compact TTFT histograms instead of requiring raw
  records for paper plots.

That work reduces per-cell cost and memory. It does not change the fact that
`run_section()` still runs cells sequentially:

```python
for scenario in scenarios.values():
    for policy in policies:
        for seed in seeds:
            run one cell
```

Cost-layer §1.1 has many independent cells. Once one cell no longer needs to
retain raw records, the next natural speedup is to run several cells at the
same time.

## 3. Scope

### In Scope

- Add `jobs: int = 1` to `experiments/simulation/common.py::run_section`.
- Add `--jobs` to `routewise simulator cost-layer`.
- Use process-level parallelism for `jobs > 1`.
- Return `Run` objects with streaming aggregates populated and records empty
  unless `retain_records=True` is explicitly requested.
- Preserve existing output files:
  - `metadata.json`
  - `summary.json`
  - `summary.csv`
  - `ttft_histograms.json`
  - `ttft_histograms_by_seed.json`
- Add metadata fields:
  - `jobs`
  - `execution_mode`: `"serial"` or `"parallel"`
- Treat the metadata change as additive only; downstream readers should
  ignore these fields if they do not need execution provenance.

### Out Of Scope

- Solver/offline-oracle changes.
- GPU acceleration.
- Changing RouteWise LP semantics.
- Streaming workload from disk request-by-request.
- Parallelizing inside one request loop.
- Changing golden capture behavior.

## 4. Current Serial Shape

Current `run_section()` loads the workload once in the parent process, then
runs every cell serially.

```python
requests = load_workload(...)
for scenario in scenarios.values():
    for policy in policies:
        runs = [
            run_cell(scenario, requests, policy, seed)
            for seed in seeds
        ]
        summarize_runs(...)
```

This is simple and deterministic, but it underuses multi-core machines. Most
cost-layer cells are independent: one cell's policy state, provider state,
random seed, and aggregate output do not affect any other cell.

## 5. Target Architecture

### 5.1 Cell Descriptor

Introduce a small serializable descriptor:

```python
@dataclass(frozen=True)
class SectionCell:
    scenario_name: str
    policy: str
    seed: int
```

The parent process builds:

```python
cells = [
    SectionCell(scenario.name, policy, seed)
    for scenario in scenarios.values()
    for policy in policies
    for seed in seeds
]
```

### 5.2 Worker Input

Do not send large request lists or hydrated real-world scenarios to every
task. Each worker receives small immutable inputs:

- `section_name`
- `scenario_name`
- `policy`
- `seed`
- `presets`
- workload spec: `dataset`, `duration_sec`, `max_requests`
- `retain_records`

Use section-local worker functions, not a global section registry in the
first implementation. For cost-layer, the worker lives next to
`cost_layer.main()` and calls `cost_layer.make_scenarios()` directly:

```python
scenario = cost_layer.make_scenarios()[cell.scenario_name]
requests = load_workload(dataset, duration_sec, max_requests)
```

This avoids introducing a generic simulator-section registry while only one
section is parallel-enabled. When `latency-layer` and `hedging` are live, we
can promote the pattern into a registry if duplication becomes real.

The scenario must be rebuilt in the worker by name. This matters for
`cost_layer_real_world`: the hydrated `ScenarioConfig` contains empirical
sample arrays, and pickling that object once per submitted cell would move
many redundant megabytes through IPC.

Do not cache the workload list in the first implementation. `Request` is a
mutable dataclass and `metadata` is a mutable dict, so sharing one cached list
across multiple cells relies on a read-only convention that is not yet tested.
Loading the workload per cell is simpler and safer. If workload loading shows
up as a measured bottleneck, add a separate follow-up with an explicit
read-only request contract or a shared-memory/mmap-backed trace format.

### 5.3 Worker Result

The worker returns a compact result:

```python
@dataclass
class SectionCellResult:
    scenario_name: str
    policy: str
    seed: int
    run: Run
```

For paper section runs, `retain_records=False`, so `run.records == []` and
`run.aggregate` carries the data needed for summary and histogram artifacts.

Golden and debug code keep `retain_records=True` and should continue to use
the serial/default path unless explicitly opted into parallel mode.

### 5.4 Parent Merge

The parent sorts results back into deterministic order:

```text
scenario order from scenarios.values()
policy order from policies
seed order from seeds
```

Then it reuses the existing summary path:

```python
runs = results[(scenario.name, policy)]
aggregate = _merge_run_aggregates(runs)
summarize_runs(...)
```

Output JSON/CSV row ordering must be identical for `jobs=1` and `jobs>1`.
Floating-point fields should match within tight numerical tolerance rather
than relying on a byte-identical-file guarantee.

## 6. Public CLI

Add to `routewise simulator cost-layer`:

```bash
--jobs N
```

Rules:

- Default: `1`
- Minimum: `1`
- `1` means serial path.
- `N > 1` means process pool with at most `N` workers.

Example:

```bash
routewise simulator cost-layer \
  --scenario cost_layer_real_world \
  --workload burstgpt \
  --policy ablation_lp_only_p50 \
  --jobs 8
```

Do not add a top-level global `--jobs` yet. Keep the flag local to section
runners until at least `latency-layer` exists.

## 7. Multiprocessing Model

Use `concurrent.futures.ProcessPoolExecutor` with an explicit `spawn` context:

```python
mp_context = multiprocessing.get_context("spawn")
executor = ProcessPoolExecutor(
    max_workers=jobs,
    mp_context=mp_context,
    max_tasks_per_child=10,
)
```

Why processes, not threads:

- The simulator is Python-loop heavy.
- RouteWise calls SciPy/HiGHS from many request decisions.
- Threads would still fight the GIL around Python-level logic.

Why explicit `spawn`:

- macOS defaults to spawn, Linux commonly defaults to fork.
- Using spawn makes behavior consistent across dev laptops and servers.
- It avoids inherited solver/native-library state from forked parents.

Why `max_tasks_per_child=10`:

- Full section runs can last many cells.
- Reclaiming worker processes bounds memory growth from solver objects,
  numpy temporaries, and Python allocator fragmentation.

Implementation detail:

```python
if jobs == 1:
    return _run_section_serial(...)
return _run_section_parallel(..., jobs=jobs)
```

Keep the serial path explicit. It is easier to debug and is the source of
truth for parity tests.

## 8. Memory Expectations

Parallelism trades wall time for memory.

With `retain_records=False`, each worker should primarily hold:

- one loaded workload list,
- one scenario's providers,
- one policy instance,
- one streaming aggregate.

The workload list is still the largest object. That means `--jobs 16` may be
too aggressive on a laptop, even if it is fine on a 20-vCPU server.

Because the first implementation reloads workload data per cell, memory is
roughly proportional to active workers. This is intentional: it favors
correctness and predictable ownership over untested shared mutable requests.

Recommended initial usage:

```text
laptop:       --jobs 2 or --jobs 4
20-vCPU box:  --jobs 8 first, then try --jobs 16
```

The CLI should not auto-detect and max out CPU cores. Explicit `--jobs` keeps
resource use predictable.

## 9. Determinism Contract

Parallel execution must not change results.

Expected parity:

- Same cell `(scenario, policy, seed)` produces the same `RunAggregate`.
- `summary.json` and `summary.csv` rows appear in the same order.
- `ttft_histograms.json` merged rows are numerically identical.
- `ttft_histograms_by_seed.json` per-seed rows are numerically identical.

Allowed difference:

- `metadata.json` gains `jobs` and `execution_mode`.

Randomness is already seed-local. Each worker constructs its own simulator
with the cell seed. No global RNG should be used in worker code.

## 10. Tests

### 10.1 Unit Test: Serial/Parallel Parity

Add a small cost-layer smoke test:

```python
rows_serial = run_section(..., jobs=1, max_requests=100)
rows_parallel = run_section(..., jobs=2, max_requests=100)
assert_rows_close(rows_parallel, rows_serial, rel=1e-12)
```

Use:

- one synthetic scenario,
- two policies: `greedy_cost`, `random`,
- two seeds,
- `max_requests=100`.

This avoids expensive RouteWise LP during unit tests but still validates
multi-cell parallel scheduling and merge order.

### 10.2 Artifact Test

Assert both modes write:

- `ttft_histograms.json`
- `ttft_histograms_by_seed.json`

and that `ttft_histograms_by_seed.json` has:

```text
len(scenarios) * len(policies) * len(seeds)
```

rows.

### 10.3 CLI Integration Test

Smoke:

```bash
routewise simulator cost-layer \
  --scenario cost_layer_uniform \
  --workload burstgpt \
  --max-requests 100 \
  --policy greedy_cost \
  --policy random \
  --jobs 2
```

Mark this test `integration` or `slow`. Starting subprocesses and loading a
trace should not be part of the default tight unit-test loop.

## 11. Implementation Steps

### Commit 0: Scenario Lookup Refactor

- Add a section-local cost-layer helper that rebuilds scenarios by name:
  `make_scenarios()[scenario_name]`.
- Keep the helper top-level and import-safe so future process workers can call
  it under spawn.
- Do not add a generic section registry yet.
- Keep this as a pure refactor with no parallel execution yet.
- The goal is to make the future worker IPC payload KB-scale, not MB-scale.
- Acceptance:
  - `make_scenarios()["cost_layer_real_world"]` works on repeated calls.
  - The returned scenario has the same provider names on repeated calls.
  - No `ProcessPoolExecutor` is introduced in this commit.

### Commit 1: Section Runner Parallel Primitive

- Add `SectionCell` / `SectionCellResult` helpers in
  `experiments/simulation/common.py`.
- Split current `run_section()` body into:
  - `_run_section_serial(...)`
  - `_run_section_parallel(...)`
  - `_write_section_outputs(...)`
- Add `jobs: int = 1` to `run_section()`.
- Keep serial behavior unchanged.
- Worker payloads contain `scenario_name`, not hydrated `ScenarioConfig`.

### Commit 2: Cost-Layer CLI

- Add `--jobs` to `experiments/simulation/cost_layer.py`.
- Pass `jobs=args.jobs` into `run_section()`.
- Add `jobs` and `execution_mode` to `metadata.json`.

### Commit 3: Tests And Benchmark

- Add serial/parallel parity test.
- Add CLI smoke with `--jobs 2`.
- Add a lightweight benchmark script or documented command:

```bash
/usr/bin/time -l uv run routewise simulator cost-layer \
  --scenario cost_layer_real_world \
  --workload burstgpt \
  --max-requests 100000 \
  --policy ablation_lp_only_p50 \
  --jobs 1

/usr/bin/time -l uv run routewise simulator cost-layer \
  --scenario cost_layer_real_world \
  --workload burstgpt \
  --max-requests 100000 \
  --policy ablation_lp_only_p50 \
  --jobs 4
```

## 12. Expected Speedup

Conservative estimate for full §1.1 on a 20-vCPU machine:

```text
--jobs 1:   baseline
--jobs 4:   ~2.5-4x faster
--jobs 8:   ~4-8x faster
--jobs 16:  ~5-10x faster, with higher memory pressure
```

Why not 20x:

- Workload load/copy overhead.
- Uneven cell duration: RouteWise cells are much slower than baselines.
- Process startup overhead.
- Real-world and heavy-tail cells may have longer tails.
- SciPy/HiGHS may already use internal native code.

Also, wall time cannot beat the slowest single cell. If
`cost_layer_real_world × ablation_lp_only_p50 × seed=42` takes 20 minutes,
then `--jobs 16` still cannot finish the full run in less than roughly that
cell time.

The first target should be `--jobs 8`; only try `--jobs 16` after checking
RSS on the real machine.

## 13. Risks

### 13.1 Pickle Failures

`ProcessPoolExecutor` requires task inputs and outputs to be picklable.
`RunAggregate` should be a dataclass/simple Python object, but custom
distributions inside hydrated scenarios should not be sent as task inputs.

Mitigation:

- Keep worker function top-level.
- Pass scenario names and policy names, not live scenarios or policy instances.
- Rebuild the scenario inside the worker.
- Test with `cost_layer_real_world`, not only synthetic distributions.

### 13.2 Memory Blowup From Workload Duplication

Each process may load its own workload. This is acceptable for the first
parallel implementation but should be monitored.

Mitigation:

- User controls `--jobs`.
- Document recommended values.
- Later follow-up: workload mmap / Arrow / shared memory / process grouping.

### 13.3 Hidden State In Policies

Policies must be constructed inside each worker. Do not reuse policy instances
across cells.

### 13.4 Local Section Runners

`section_runners` such as `offline` must be top-level functions so they are
picklable. If future sections use closures, parallel mode should fail fast
with a clear error.

## 14. Sign-Off Checklist

- [ ] `--jobs 1` preserves current behavior.
- [ ] `--jobs 2` passes serial/parallel parity test.
- [ ] `cost_layer_uniform` CLI smoke passes with `--jobs 2`.
- [ ] `cost_layer_real_world` CLI smoke passes with `--jobs 2`.
- [ ] `metadata.json` records `jobs` and `execution_mode`.
- [ ] `summary.csv` row order is deterministic.
- [ ] `ttft_histograms.json` and `ttft_histograms_by_seed.json` are written.
- [ ] Full fast tests pass.
- [ ] Touched-file ruff passes.
