#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# quick_test.sh
#
# Smoke test for the full repository pipeline.
#
# This script runs a short version of the workflow to verify that:
#   1. inputs can be built
#   2. hidden weights can be built
#   3. training starts and completes with a tiny epoch count
#   4. analysis runs
#   5. figure scripts run
#
# Run from repository root:
#
#   bash scripts/quick_test.sh
#
# Note
# ----
# This is not intended to reproduce manuscript results.
# ============================================================================

echo "============================================================"
echo "Running quick smoke test..."
echo "============================================================"

# Build required inputs and initial weights.
bash scripts/01_build_inputs.sh
bash scripts/02_build_hidden_weights.sh

# Run very short training jobs.
EPOCHS=5 RUNS=1 SEED=42 bash scripts/03_train_random.sh
EPOCHS=5 RUNS=1 SEED=42 bash scripts/04_train_mexican_hat.sh

# Run analysis and figure generation.
bash scripts/05_analyze.sh
bash scripts/06_make_figures.sh

echo
echo "============================================================"
echo "Quick smoke test completed successfully."
echo "============================================================"