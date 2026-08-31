"""EuroSys'27 artifact-evaluation entrypoint for RouteWise (paper #96).

This package is the reviewer-facing interface of the artifact. It contains no
experiment logic of its own: `manifest.yaml` maps paper claims and figures to
entrypoints under `experiments/`, `runner` dispatches them, and `verify`
compares produced summaries against `expected.yaml`.

Dependency direction is one-way: `artifact` may import `experiments` and
`llm_routewise`; nothing outside this package may import `artifact`.
"""

from __future__ import annotations
