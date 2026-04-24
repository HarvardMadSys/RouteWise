#!/usr/bin/env python3
"""Compatibility wrapper for the Phase 3 latency routing experiment."""

from __future__ import annotations

from experiments.latency_phase3.experiment import *  # noqa: F401,F403
from experiments.latency_phase3.experiment import main


if __name__ == "__main__":
    main()
