# RouteWise — EuroSys'27 Artifact (Paper #96)

This branch is the evaluated artifact for:

> **RouteWise: Latency–Cost Optimization for Multi-Provider LLM Routing**
> Muxin Tian, Haoran Ni, Yiyan Zhai, Yangsun Park, Juncheng Yang.
> EuroSys 2027, paper #96.

Canonical artifact branch: `eurosys27-ae`. The legacy branch `eurosys2027` is
not part of the submitted artifact. The `main` branch hosts the separately
released `llm-routewise` library, which evolves independently — do **not**
`pip install llm-routewise` to evaluate this artifact; everything runs from
this checkout.

## Quick start (kick the tires)

```bash
uv sync --frozen
uv run python -m artifact smoke
```

`smoke` finishes in about a minute on a laptop: import checks, one simulator
section replayed on the committed 120-request synthetic fixture, verification
of the produced summary numbers, and the fast unit tests. It needs **no API
keys and no network access**.

Run every `python -m artifact` command from the repository root: like
`experiments/`, the `artifact/` package is repository-only and is not
installed into the environment. Outputs always land under the repository's
`outputs/` directory regardless of stray working-directory changes.

If you do not have uv (the Python package manager by Astral): install it from
https://docs.astral.sh/uv/getting-started/installation/ — it also provisions
the pinned Python interpreter, so no system Python setup is required.

## Artifact components

| Path | Role | Paper connection |
|---|---|---|
| `artifact/` | Reviewer interface: `manifest.yaml` (claims → entrypoints), runner, verify, smoke | claims table below |
| `llm_routewise/` | Routing core, LP mixture solver, simulator engine, metrics | §2–3 |
| `experiments/simulation/` | Section-driven simulator experiments | §3.2–3.5, Figs 8–9 |
| `experiments/offline_stage/` | Offline/stage configuration and loaders | offline-gap numbers |
| `experiments/real_evaluation/` | Live-provider runner (optional B2 path; costs money) | §4, 24h evaluation |
| `plots/` | Figure generation | Figs 1–9 |
| `data/` | Committed inputs: motivation data, drift sources, smoke fixture | Figs 2–3 |
| `docs/research/REPRODUCIBILITY.md` | Detailed operational guide | — |

The dependency direction is one-way: `artifact/` → `experiments/` →
`llm_routewise/`. The `artifact/` package contains no experiment logic.

## Environment

- **OS**: the canonical evaluated environment is Ubuntu 24.04 x86-64 (the
  `ae.yml` CI workflow runs `smoke` there on every push); macOS arm64 is also
  tested locally.
- **Python and dependencies**: Python is pinned by `.python-version` (3.14)
  and every package by `uv.lock`; `uv sync --frozen` installs the interpreter
  and the exact locked set. No system-level dependencies, no GPU, no
  commercial solver (the LP path uses open-source solvers).
- **Network**: only two touch points, both optional for smoke — the one-time
  public trace download below, and the optional live-provider path (B2).

## Claims and figures

```bash
uv run python -m artifact list
```

prints every claim/figure target with its class and current status:

- **A — full reproduction**: trace-driven simulation; reviewers rerun the
  experiment and regenerate the numbers/figures.
- **B1 — recorded-result reproduction**: recorded live-evaluation data →
  metrics → figures (the live experiment itself is not rerun). The 24-hour
  records land in `data/real_eval_records/` during the kick-the-tires window;
  the motivation-figure inputs are already committed.
- **B2 — live rerun (optional)**: full scripts to redo the live 24h
  evaluation with your own provider keys; costs real money and is
  non-deterministic — provider load, pricing, and rate limits have changed,
  so it produces a new timestamped result that is checked for schema,
  completeness, and qualitative sanity, never against the paper's recorded
  numbers. We are confirming with the AE chairs that the recorded-result
  path (B1) is the evaluated route for the live-evaluation claims; B2 is
  provided for completeness.

These classes are this artifact's own organization (the official EuroSys
badges are Available, Functional, and Reproduced); they exist so every
figure states plainly which kind of evidence backs it.
- **C — non-computational**: illustrative figures with no reproduction
  target (architecture diagram, timeline, author-reconstructed snapshots —
  each documented as such).

Every claim, figure number, and expected value is calibrated against one
pinned arXiv version of the paper (recorded with its version suffix and PDF
SHA-256 in `artifact/manifest.yaml`). Targets marked `pending-*` are being
pinned against that PDF during the kick-the-tires window (this is a
cooperative process); `reproduce` refuses them with an explanation rather
than running something uncalibrated.

## Data setup for full class-A runs

```bash
uv run python scripts/prepare_workload.py --days 30
```

downloads the two public source traces (BurstGPT v2.0 release CSV and the
ShareGPT V3 dump) from their original hosts with SHA256-pinned URLs and
composes the simulator workload `data/burstgpt_30d.jsonl`. This covers the
30-day simulation targets; the Figure 8 end-to-end target additionally needs
the FreeInference derived trace (arrival times and token counts only), which
is being added during the kick-the-tires window.

## Reproduce and verify

```bash
uv run python -m artifact reproduce <target> [...]   # or --group A
uv run python -m artifact verify                      # PASS/FAIL per check
```

`verify` reads `artifact/expected.yaml` (expected values with justified
tolerances) and compares the produced summaries; the runner itself never
sees the expected values.

## Resource expectations

| Path | Wall time | Memory | Network |
|---|---|---|---|
| `smoke` | ~1 min | < 1 GB | none |
| workload preparation | ~10 min (download-bound) | < 2 GB | ~1 GB download |
| class-A targets | being measured; filled in per target in `artifact/manifest.yaml` during kick-the-tires | | none |

## Cost warning — live path only

`experiments/real_evaluation/` (class B2) sends real requests to commercial
LLM providers and **spends real money**. It is optional, never run by
`smoke`/`reproduce` defaults, and requires explicit provider keys
(`cp .env.example .env`). Nothing else in this artifact contacts providers.

## License

Code is MIT-licensed (see `LICENSE`). Third-party trace licenses and the
license for our recorded evaluation data are documented alongside the data
directories.
