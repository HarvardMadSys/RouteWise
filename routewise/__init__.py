"""Public RouteWise package.

The top-level package stays lightweight and dependency-free at import time:

- ``routewise.core`` — the environment-agnostic routing algorithm
  (stdlib-only; enforced by the import-boundary test)
- ``routewise.capacity`` / ``routewise.schemas`` / ``routewise.const`` /
  ``routewise.metrics`` — contracts shared by both worlds
- ``routewise.sim`` — the simulated world (engine, world model, policies);
  requires the ``[sim]`` extra
- the live world lives in ``experiments.real_evaluation``

Nothing here imports the scientific stack eagerly; subpackages are imported
explicitly by consumers.
"""

from __future__ import annotations

__all__ = ["capacity", "const", "core", "metrics", "offline", "schemas", "sim"]
