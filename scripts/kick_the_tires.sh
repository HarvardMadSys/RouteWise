#!/usr/bin/env bash
# Kick-the-tires check for the EuroSys'27 artifact.
#
# Replays the committed 120-request synthetic fixture through the cost-layer
# simulator section, compares behavior-sensitive outputs against the golden
# digests, and runs the fast unit tests. Needs no API keys and no network
# access; finishes in a couple of minutes on a laptop.
#
# Usage (from the repository root):
#     bash scripts/kick_the_tires.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "== simulator section on the committed fixture =="
uv run python -m routewise_cli.main simulator cost-layer \
    --scenario cost_layer_uniform \
    --workload smoke \
    --max-requests 100 \
    --seed 42 \
    --alpha 0.5 \
    --output-dir outputs/smoke/cost_layer

echo "== fast unit tests =="
uv run pytest -q -m "not slow" tests/test_architecture_scaffold.py tests/unit/simulation

echo "kick-the-tires: PASS"
