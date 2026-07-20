# Effective Cost Calibration Ablation

This ablation tests how the `L` and `U` envelope used by the quota shadow-price
curve affects RouteWise's quota-vs-API routing decision.

The curve-shape ablation answers: which scarcity curve shape should we use?
This calibration ablation answers the follow-up: once the shape is fixed to
`exp_lu`, which workload percentile envelope should set `L` and `U`?

The fixed shadow-price form is:

```text
psi(z) = L * (U / L)^z
```

where `z` is quota usage fraction.

## Claim Being Tested

The paper-facing question is:

```text
With the curve shape fixed to exp_lu, does the default envelope
cheapest_api + p10_p90 remain competitive against nearby percentile choices?
```

This ablation is intentionally narrow. It does not test API price tradeoffs,
latency routing, hedging, or the full end-to-end provider mix.

## Experimental Surface

The default run uses a clean two-provider surface:

```text
quota_clean__plan=chutes__n=16

providers:
  chutes_quota   tier S_Q, fixed subscription cost, zero marginal request cost
  api_cheap      tier S_A, $1/M input tokens and $5/M output tokens
```

This excludes `api_mid` and `api_expensive` on purpose. The curve-shape
ablation is meant to isolate quota scarcity pricing against one outside API
option. Keeping this ablation on the same clean surface prevents the LP budget
knob from mixing in unrelated API choice behavior.

Default fixed settings:

```text
workload:          burstgpt
quota plan:        chutes
subscription count q*: 16
latency family:    heavy_tail
curve shape:       exp_lu
LP alpha value:    0.5
seed:              42
```

Default sweep:

```text
api reference: cheapest_api
percentile envelopes:
  p05_p95
  p10_p90
  p25_p75
  min_max
```

The default command therefore runs 4 cells: one scenario, one seed, four
percentile envelopes.

## Requirements

Run all commands from the repository root:

```bash
cd /path/to/RouteWise
```

Required local inputs:

```text
data/burstgpt_30d.jsonl
experiments/subscription_plans.yaml
```

No GPU is required. The experiment is CPU-bound and can run on a laptop. Use
`--jobs` to parallelize cells across CPU cores.

## Quick Smoke Test

Use this before a full artifact run:

```bash
uv run python -m routewise_cli.main ablation effective-cost-calibration \
  --max-requests 20 \
  --seed 42 \
  --jobs 1 \
  --output-dir outputs/ablations/effective_cost_calibration_smoke
```

Expected high-level output:

```text
{"output_dir": ".../effective_cost_calibration_smoke", "rows": 4, "section": "effective-cost-calibration"}
```

Expected files:

```text
outputs/ablations/effective_cost_calibration_smoke/
  metadata.json
  summary.csv
  summary.json
  ttft_histograms.json
  ttft_histograms_by_seed.json
```

Smoke checks:

```bash
python - <<'PY'
import csv, json
from pathlib import Path

root = Path("outputs/ablations/effective_cost_calibration_smoke")
metadata = json.loads((root / "metadata.json").read_text())
rows = list(csv.DictReader((root / "summary.csv").open()))

assert metadata["section"] == "effective-cost-calibration"
assert metadata["scenarios"] == ["quota_clean__plan=chutes__n=16"]
assert len(rows) == 4
assert {row["api_reference"] for row in rows} == {"cheapest_api"}
assert {row["percentile_envelope"] for row in rows} == {
    "p05_p95", "p10_p90", "p25_p75", "min_max"
}
assert all(float(row["envelope_L"]) > 0.0 for row in rows)
assert all(float(row["envelope_U"]) > float(row["envelope_L"]) for row in rows)
print("smoke ok")
PY
```

## Full Artifact Run

Run the paper-facing default:

```bash
uv run python -m routewise_cli.main ablation effective-cost-calibration \
  --seed 42 \
  --jobs 8 \
  --output-dir outputs/ablations/effective_cost_calibration
```

The full run should still produce 4 rows. It uses the full BurstGPT trace rather
than the 20-request smoke cap. Runtime depends on CPU count; on an 8-core run it
is expected to finish in minutes, not hours.

## Output Schema

`summary.csv` and `summary.json` contain one row per scenario-policy aggregate.
The columns needed for this ablation are:

```text
scenario
policy
n_requests
api_cost_usd
subscription_fixed_cost_usd
total_cost_usd
provider_mix
tier_mix
api_reference
percentile_envelope
envelope_L
envelope_U
```

The key calibration fields are:

```text
api_reference          request value reference used to compute L/U
percentile_envelope    named percentile pair
envelope_L             numeric lower bound used by the policy
envelope_U             numeric upper bound used by the policy
```

`metadata.json` additionally contains:

```text
calibration_envelopes
```

This is a machine-readable list of the actual `(scenario, policy, L, U)` values
used in the run. Artifact reviewers should use this list rather than
recomputing L/U from policy names.

`ttft_histograms.json` and `ttft_histograms_by_seed.json` are emitted by the
shared simulator runner for distribution plots. They are not the main artifact
for this ablation, but they should be kept with the run output.

## Result Sanity Checks

After the full run, use:

```bash
python - <<'PY'
import csv, json
from pathlib import Path

root = Path("outputs/ablations/effective_cost_calibration")
metadata = json.loads((root / "metadata.json").read_text())
rows = list(csv.DictReader((root / "summary.csv").open()))

print("section:", metadata["section"])
print("scenarios:", metadata["scenarios"])
print("policies:", len(metadata["policies"]))
print("loaded_requests:", metadata["loaded_requests"])
print()

for row in rows:
    print(
        row["percentile_envelope"],
        "L=", row["envelope_L"],
        "U=", row["envelope_U"],
        "total_cost=", row["total_cost_usd"],
        "provider_mix=", row["provider_mix"],
    )
PY
```

Expected structural properties:

```text
section == effective-cost-calibration
scenarios == ["quota_clean__plan=chutes__n=16"]
len(policies) == 4
all rows have api_reference == cheapest_api
all rows have positive envelope_L and envelope_U > envelope_L
provider_mix contains only chutes_quota and api_cheap
```

Interpretation guide:

- `p10_p90` is the default candidate.
- `p05_p95` is wider and more tail-sensitive.
- `p25_p75` is narrower and compresses the shadow-price dynamic range.
- `min_max` is a sanity extreme and may be unstable under heavy-tailed request
  token counts.

The best calibration should be judged by total cost and quota/API mix under the
same clean scenario. Avoid interpreting this ablation as an API price-choice
study; the clean surface has exactly one API provider by design.

## Extended Diagnostics

The CLI supports additional sweeps, but they are not the default paper-facing
run.

Reference sweep:

```bash
uv run python -m routewise_cli.main ablation effective-cost-calibration \
  --sweep reference \
  --seed 42 \
  --jobs 8 \
  --output-dir outputs/ablations/effective_cost_calibration_reference
```

Full cross-product:

```bash
uv run python -m routewise_cli.main ablation effective-cost-calibration \
  --sweep cross-product \
  --seed 42 \
  --jobs 8 \
  --output-dir outputs/ablations/effective_cost_calibration_cross
```

On the default clean surface, `cheapest_api`, `median_api`, `mean_api`, and
`max_api` collapse to the same numeric reference because there is only one API
provider. These commands are useful mainly if a future diagnostic deliberately
changes the surface to multiple API providers.

## Code Map

```text
experiments/ablations/effective_cost_calibration/harness.py
  CLI entrypoint, clean scenario construction, policy construction, artifact
  enrichment.

experiments/ablations/effective_cost_calibration/envelope.py
  Workload-cost reference calculation and percentile L/U helpers.

experiments/ablations/effective_cost_calibration/README.md
  This artifact-facing reproduction guide.

tests/unit/ablations/test_effective_cost_calibration.py
  Locks the default clean surface, default 4-policy percentile sweep, L/U
  enrichment, and CLI registration.
```

## Regression Tests

Before submitting artifacts, run:

```bash
uv run ruff check experiments/ablations/effective_cost_calibration \
  tests/unit/ablations/test_effective_cost_calibration.py

uv run pytest -q tests/unit/ablations/test_effective_cost_calibration.py \
  tests/unit/ablations/test_effective_cost_harness.py \
  tests/unit/ablations/test_effective_cost_policy.py

uv run pytest -q -m "not slow"
```

The last command is the repository's fast test suite. It should pass before the
artifact output is regenerated.

## Common Failure Modes

If the output has 7 rows, the run was produced by the older non-clean default
that included the reference sweep and multiple API providers. Re-run with the
current harness default or explicitly pass:

```bash
--sweep percentile --api-reference cheapest_api
```

If `summary.csv` lacks `envelope_L` and `envelope_U`, the run was produced
before artifact enrichment was added. Re-run the experiment with the current
code.

If `provider_mix` contains `api_mid` or `api_expensive`, the run is not the
clean calibration surface and should not be used for the main paper-facing
claim.
