"""Public RouteWise package.

The top-level package stays lightweight and dependency-free at import time:

- ``routewise.core`` — the environment-agnostic routing algorithm
  (stdlib-only; enforced by the import-boundary test)
- ``routewise.capacity`` / ``routewise.schemas`` / ``routewise.const`` —
  dependency-light contracts shared by both worlds
- ``routewise.metrics`` — run result containers and aggregations shared by
  simulator and live-eval paths; importing it requires the scientific stack
  from the ``[sim]`` or ``[real-eval]`` extras
- ``routewise.sim`` — the simulated world (engine, world model, policies);
  requires the ``[sim]`` extra
- the live world lives in ``experiments.real_evaluation``

Importing ``routewise`` itself does not import the scientific stack eagerly;
optional subpackages are imported explicitly by consumers.
"""

from __future__ import annotations

__all__ = ["capacity", "const", "core", "metrics", "offline", "schemas", "sim"]
