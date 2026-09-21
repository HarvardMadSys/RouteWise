# Figure 8 simulator revision

The Figure 8 experiment used the simulator at
[`99c5f3fd6504377fb57bd4edecef89e3865b7765`](https://github.com/HarvardMadSys/RouteWise/commit/99c5f3fd6504377fb57bd4edecef89e3865b7765).
The later `86a26e2` change set the concurrency shadow price to zero, and
subsequent revisions refactored the simulator. These changes affect
RouteWise routing decisions. The artifact retains the historical source
explicitly so the released trace can reproduce the paper's experiment.

## Algorithm-to-figure mapping

| Item | Available concurrency shadow price | Interpretation |
|---|---|---|
| Paper Figure 8; bundled source `99c5f3f` | `L` (`scarcity_price("constant_l", ...)`) | Earlier routing variant used by the archived experiment |
| Paper §3.2.3 and Equation (3) | `0` | Described algorithm; not the variant behind the archived Figure 8 values |
| Subsequent code change `86a26e2f66f723cf15a2c45048f21ce0c1daf1f0` | `0` | Changed concurrency pricing and hence some routing decisions |

Matching the archived numbers reproduces the earlier
experiment; it does not establish that the zero-cost rule gives those
numbers. This wrapper retains the original variant and reference values.
The [simulation guide](README.md#output-length-feedback) documents the
output-length feedback timing that also applies to this experiment.

## Source and command

`paper_figure8_source.tar.gz` contains the unmodified `rwsim/` and
`experiments/` trees from that commit, created with:

```bash
git archive --format=tar.gz \
    --output=experiments/simulation/paper_figure8_source.tar.gz \
    99c5f3fd6504377fb57bd4edecef89e3865b7765 rwsim experiments
```

SHA-256: `5e3299d7e56c1d8ae2421d0bb2dd61fff7d081fc6fdc1e7fea24af9372ce6dce`.
This is repository source code, distributed under the repository's MIT
license. It contains the historical provider profiles and configurations,
but no private input trace or precomputed Figure 8 simulation outputs.

From the artifact root, run:

```bash
uv run python scripts/reproduce_figure8.py --jobs 4
# Lower-memory alternative; independently checked against the same eight rows:
uv run python scripts/reproduce_figure8.py --jobs 1
```

The script verifies the source checksum, extracts it to a temporary
directory, supplies the released `data/freeinference.jsonl`, and executes
the historical simulator using the artifact's locked Python environment.
Before execution it builds the dataset cache inside that temporary directory.
This fixes the fresh-directory failure in the historical single-worker path;
building a cache in the outer checkout alone is insufficient. The archived
source, trace, policy settings, and numeric reference are unchanged.
It fixes seed 42, scenario `end_to_end_rw8`, SLO 3000 ms, the `bucket_mean`
predictor with the historical `q50` interface, one `chutes` quota plan,
one `featherless_premium` concurrency plan, model alias `qwen3-235b`, and
prefix-cache accounting enabled. No API keys, network access, Git history,
or GPU are needed after `uv sync --frozen`.

All eight policies process 21,678 requests. The script normalizes the
historical `ablation_lp_only_pN` policy labels to `ablation_lp_only_alphaN`
for the current plotter, without modifying the numeric results. It then
checks the independent archived reference in
`data/figure8_reference_summary.csv` and creates all four Figure 8 panels,
including the TTFT boxplot. A mismatch fails the command; the allowed
relative/absolute tolerance is `1e-9` for numeric aggregates, and request
and provider counts must match exactly. The generated `source_revision.json`
records the source commit and checksum alongside the simulation output.

The input contains 24,035 raw records. Filtering removes 2,238 failed
requests and 119 remaining zero-token requests, leaving 21,678 simulation
requests. The paper's approximately 24K-request workload description in
Table 2 refers to the raw population. Cache-read counters are reused for
candidate providers with cached-input rates, without reconstructing
provider-local cache residency after counterfactual routing.

The reference summary comes from the archived 2026-05-14 paper run, not
from this wrapper's output. Validation on 2026-09-10 reproduced the archived
costs, TTFT statistics, SLO rates, and provider counts for all eight policies
using the released trace. The current simulator remains available for new
experiments and the other simulation sections.
