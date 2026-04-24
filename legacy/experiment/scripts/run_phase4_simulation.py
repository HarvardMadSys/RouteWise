#!/usr/bin/env python3
"""Compatibility wrapper for the Phase 4 smart hedging experiment."""

from __future__ import annotations

from experiments.latency_phase4.experiment import *  # noqa: F401,F403
from experiments.latency_phase4.experiment import main


if __name__ == "__main__":
    main()
