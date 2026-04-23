#!/bin/bash
# Run Stage 2 experiments for all datasets
# Usage: ./scripts/run_stage2_all.sh

set -e  # Exit on error

SOLVER="gurobi"
DELTA=60
WORKERS=8  # Reduce workers to avoid memory issues with large datasets

echo "=========================================="
echo "Running Stage 2 Experiments (delta=${DELTA}s, workers=${WORKERS})"
echo "=========================================="

# Activate virtual environment
source .venv/bin/activate

# Ensure Gurobi license is set
if [ -z "$GRB_LICENSE_FILE" ]; then
    export GRB_LICENSE_FILE=/home/murphy/gurobi.lic
    echo "Set GRB_LICENSE_FILE=$GRB_LICENSE_FILE"
fi

echo ""
echo "=========================================="
echo "[1/2] Running RedNote dataset..."
echo "=========================================="
python -m experiment.scripts.run_stage2_experiments --data rednote --solver $SOLVER --delta $DELTA --workers $WORKERS

echo ""
echo "=========================================="
echo "[2/2] Running FreeInference dataset..."
echo "=========================================="
python -m experiment.scripts.run_stage2_experiments --data freeinference --solver $SOLVER --delta $DELTA --workers $WORKERS

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
