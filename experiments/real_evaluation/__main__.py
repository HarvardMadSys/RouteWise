"""Package entry point for ``python -m experiments.real_evaluation``."""

from __future__ import annotations

import sys

from experiments.real_evaluation.runner import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
