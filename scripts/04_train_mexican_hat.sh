#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 04_train_mexican_hat.sh
#
# Train the dense Elman RNN using Mexican-hat hidden-weight initializations.
#
# Run from the repository root:
#
#   bash scripts/04_train_mexican_hat.sh
#
# Outputs
# -------
# Training checkpoints and logs are written to:
#
#   data/runs/mexican_hat/
#
# Requirements
# ------------
# Must be run after:
#
#   bash scripts/01_build_inputs.sh
#   bash scripts/02_build_hidden_weights.sh
# ============================================================================

DATA="data/inputs/Ns100_SeqN100_asym1.pth.tar"
INIT_ROOT="data/hidden_weight_inits"
RUN_ROOT="data/runs/mexican_hat"
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

EPOCHS="${EPOCHS:-100000}"
RUNS="${RUNS:-3}"
SEED="${SEED:-42}"

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
  --skip_fro_norm_hh
)

run_condition () {
  local label="$1"
  local whh_path="$2"
  local save_dir="$3"

  if [[ ! -f "$whh_path" ]]; then
    echo "ERROR: Mexican-hat initialization not found:"
    echo "  $whh_path"
    echo
    echo "Run:"
    echo "  bash scripts/02_build_hidden_weights.sh"
    exit 1
  fi

  mkdir -p "$save_dir"

  local log
  log="$save_dir/train_$(date +%Y%m%d_%H%M%S).log"

  echo
  echo "------------------------------------------------------------"
  echo "Running Mexican-hat condition: $label"
  echo "Weights: $whh_path"
  echo "Output:  $save_dir"
  echo "Log:     $log"
  echo "------------------------------------------------------------"

  python "$MAIN_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --whh_path "$whh_path" \
    --savename "$save_dir" \
    2>&1 | tee "$log"

  echo "Finished Mexican-hat condition: $label"
}

echo "============================================================"
echo "Running Mexican-hat α₀=0.70 experiment..."
echo "============================================================"

run_condition \
  "k5 alpha=0p75" \
  "$INIT_ROOT/mexican_hat/k5/alpha0p70/Whh.npy" \
  "$RUN_ROOT/k5/alpha0p70"

echo
echo "Finished Mexican-hat α₀=0.70 experiment."
echo "Outputs saved to:"
echo "  $RUN_ROOT/k5/alpha0p70"
echo