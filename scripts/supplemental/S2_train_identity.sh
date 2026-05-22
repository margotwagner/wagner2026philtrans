#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# S2_train_identity.sh
#
# Train the dense Elman RNN using the identity hidden-weight initialization.
#
# Run from the repository root:
#
#   bash scripts/supplemental/S2_train_identity.sh
#
# Outputs
# -------
# Training checkpoints and logs are written to:
#
#   data/runs/identity/
#
# Requirements
# ------------
# Must be run after:
#
#   bash scripts/01_build_inputs.sh
#   bash scripts/supplemental/S1_build_extra_hidden_weights.sh
# ============================================================================

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

DATA="data/inputs/Ns100_SeqN100_asym1.pth.tar"
INIT_ROOT="data/hidden_weight_inits"
RUN_ROOT="data/runs/identity"
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

if [[ ! -f "$INIT_ROOT/identity/Whh.npy" ]]; then
  echo "ERROR: Identity initialization not found."
  echo
  echo "Run:"
  echo "  bash scripts/supplemental/S2_build_extra_hidden_weights.sh"
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
# Identity initialization experiment
# --------------------------------------------------------------------------

echo "============================================================"
echo "Running identity initialization experiment..."
echo "============================================================"

LOG="$RUN_ROOT/train_$(date +%Y%m%d_%H%M%S).log"

python "$MAIN_SCRIPT" \
  "${COMMON_ARGS[@]}" \
  --whh_path "$INIT_ROOT/identity/Whh.npy" \
  --savename "$RUN_ROOT" \
  2>&1 | tee "$LOG"

echo
echo "Finished identity initialization experiment."
echo "Outputs saved to:"
echo "  $RUN_ROOT"
echo "Log saved to:"
echo "  $LOG"