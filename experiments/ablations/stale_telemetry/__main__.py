"""Allow ``python -m experiments.ablations.stale_telemetry``."""

from experiments.ablations.stale_telemetry.harness import main

if __name__ == "__main__":
    raise SystemExit(main())
