#!/usr/bin/env python3
"""Compatibility wrapper for Phase 4 smart hedging plots."""

from __future__ import annotations

from experiments.latency_phase4.plots import *  # noqa: F401,F403
from experiments.latency_phase4.plots import main


if __name__ == "__main__":
    main()
