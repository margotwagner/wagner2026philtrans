#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 03_train_random.sh
#
# Train the dense Elman RNN using the random hidden-weight initialization.
#
# Run from the repository root:
#
#   bash scripts/03_train_random.sh
#
# Outputs
# -------
# Training checkpoints and logs are written to:
#
#   data/runs/random/
#
# Requirements
# ------------
# Must be run after:
#
#   bash scripts/01_build_inputs.sh
#   bash scripts/02_build_hidden_weights.sh
# ============================================================================

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

DATA="data/inputs/Ns100_SeqN100_asym1.pth.tar"
INIT_ROOT="data/hidden_weight_inits"
RUN_ROOT="data/runs/random"
MAIN_SCRIPT="Main.py"

mkdir -p "$RUN_ROOT"

# --------------------------------------------------------------------------
# Input and Hidden Weight Checker
# --------------------------------------------------------------------------
if [[ ! -f "$DATA" ]]; then
  echo "ERROR: Input file not found:"
  echo "  $DATA"
  echo
  echo "Run:"
  echo "  bash scripts/01_build_inputs.sh"
  exit 1
fi

if [[ ! -f "$INIT_ROOT/random/seed000/Whh.npy" ]]; then
  echo "ERROR: Random initialization not found."
  echo
  echo "Run:"
  echo "  bash scripts/02_build_hidden_weights.sh"
  exit 1
fi

# --------------------------------------------------------------------------
# Training configuration
# --------------------------------------------------------------------------

EPOCHS=100000
RUNS=3
SEED=42

# --------------------------------------------------------------------------
# Shared model/training arguments
# --------------------------------------------------------------------------

COMMON_ARGS=(
  --input "$DATA"
  --ae 1
  --pred 1
  --n 100
  --hidden-n 100
  --epochs "$EPOCHS"
  --rnn_act tanh
  --act_output sigmoid
  --fixi 5
  --fixo 5
  --amp off
  --num_runs "$RUNS"
  --seed "$SEED"
  --print-freq 1000
)

# --------------------------------------------------------------------------
# Random initialization experiment
# --------------------------------------------------------------------------

echo "============================================================"
echo "Running random initialization experiment..."
echo "============================================================"

LOG="$RUN_ROOT/train_$(date +%Y%m%d_%H%M%S).log"

python "$MAIN_SCRIPT" \
  "${COMMON_ARGS[@]}" \
  --whh_path "$INIT_ROOT/random/seed000/Whh.npy" \
  --savename "$RUN_ROOT" \
  2>&1 | tee "$LOG"

echo
echo "Finished random initialization experiment."
echo "Outputs saved to:"
echo "  $RUN_ROOT"