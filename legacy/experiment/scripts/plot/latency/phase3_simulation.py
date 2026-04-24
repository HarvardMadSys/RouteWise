#!/usr/bin/env python3
"""Compatibility wrapper for Phase 3 latency routing plots."""

from __future__ import annotations

from experiments.latency_phase3.plots import *  # noqa: F401,F403
from experiments.latency_phase3.plots import main


if __name__ == "__main__":
    main()
