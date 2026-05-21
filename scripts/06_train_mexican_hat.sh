#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 06_train_mexican_hat.sh
#
# Train the dense Elman RNN using Mexican-hat hidden-weight initializations.
#
# Run from the repository root:
#
#   bash scripts/06_train_mexican_hat.sh
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
echo "Running Mexican-hat initialization experiments..."
echo "============================================================"

# Centered Mexican hat initialization: k = 0
run_condition \
  "k0" \
  "$INIT_ROOT/mexican_hat/k0/Whh.npy" \
  "$RUN_ROOT/k0"

# Shifted Mexican hat initializations: k = 5, alpha sweep
ALPHAS=(
  "0p00"
  "0p25"
  "0p50"
  "0p75"
  "1p00"
)

 for ALPHA in "${ALPHAS[@]}"; do
  run_condition \
    "k5 alpha=${ALPHA}" \
    "$INIT_ROOT/mexican_hat/k5/alpha${ALPHA}/Whh.npy" \
    "$RUN_ROOT/k5/alpha${ALPHA}"
done

echo
echo "Finished all Mexican-hat initialization experiments."
echo "Outputs saved to:"
echo "  $RUN_ROOT"