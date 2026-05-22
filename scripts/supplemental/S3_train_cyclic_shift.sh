#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# S3_train_cyclic_shift.sh
#
# Train the dense Elman RNN using cyclic-shift hidden-weight initializations
# across alpha values.
#
# Run from the repository root:
#
#   bash scripts/supplemental/S3_train_cyclic_shift.sh
#
# Outputs
# -------
# Training checkpoints and logs are written to:
#
#   data/runs/cyclic_shift/
#
# Requirements
# ------------
# Must be run after:
#
#   bash scripts/01_build_inputs.sh
#   bash scripts/supplemental/S1_build_extra_hidden_weights.sh
# ============================================================================

DATA="data/inputs/Ns100_SeqN100_asym1.pth.tar"
INIT_ROOT="data/hidden_weight_inits"
RUN_ROOT="data/runs/cyclic_shift"
MAIN_SCRIPT="Main.py"

mkdir -p "$RUN_ROOT"

if [[ ! -f "$DATA" ]]; then
  echo "ERROR: Input file not found:"
  echo "  $DATA"
  echo
  echo "Run:"
  echo "  bash scripts/01_build_inputs.sh"
  exit 1
fi

EPOCHS=100000
RUNS=3
SEED=42

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

ALPHAS=(
  "0p00"
  "0p25"
  "0p50"
  "0p75"
  "1p00"
)

echo "============================================================"
echo "Running cyclic-shift initialization experiments..."
echo "============================================================"

for ALPHA in "${ALPHAS[@]}"; do
  WHH_PATH="$INIT_ROOT/cyclic_shift/alpha${ALPHA}/Whh.npy"
  SAVE_DIR="$RUN_ROOT/alpha${ALPHA}"
  LOG="$SAVE_DIR/train_$(date +%Y%m%d_%H%M%S).log"

  if [[ ! -f "$WHH_PATH" ]]; then
    echo "ERROR: Cyclic-shift initialization not found:"
    echo "  $WHH_PATH"
    echo
    echo "Run:"
    echo "  bash scripts/supplemental/S2_build_hidden_weights.sh"
    exit 1
  fi

  mkdir -p "$SAVE_DIR"

  echo
  echo "------------------------------------------------------------"
  echo "Running cyclic-shift alpha=${ALPHA}..."
  echo "Weights: $WHH_PATH"
  echo "Output:  $SAVE_DIR"
  echo "Log:     $LOG"
  echo "------------------------------------------------------------"

  python "$MAIN_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --whh_path "$WHH_PATH" \
    --savename "$SAVE_DIR" \
    2>&1 | tee "$LOG"

  echo "Finished cyclic-shift alpha=${ALPHA}"
done

echo
echo "Finished all cyclic-shift initialization experiments."
echo "Outputs saved to:"
echo "  $RUN_ROOT"