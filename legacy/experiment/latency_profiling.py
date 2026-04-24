"""Compatibility wrapper for the offline/stage latency profiling probe."""

from experiments.offline_stage.latency_profiling import *  # noqa: F401,F403
from experiments.offline_stage.latency_profiling import main

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
