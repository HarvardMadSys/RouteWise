"""Compatibility wrapper for offline counterfactual plots."""

from __future__ import annotations

from experiments.offline_counterfactual.plots import *  # noqa: F401,F403
from experiments.offline_counterfactual.plots import main


if __name__ == "__main__":
    main()
