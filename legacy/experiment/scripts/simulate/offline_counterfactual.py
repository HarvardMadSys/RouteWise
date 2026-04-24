"""Compatibility wrapper for the offline counterfactual experiment."""

from __future__ import annotations

from experiments.offline_counterfactual.experiment import *  # noqa: F401,F403
from experiments.offline_counterfactual.experiment import main


if __name__ == "__main__":
    main()
