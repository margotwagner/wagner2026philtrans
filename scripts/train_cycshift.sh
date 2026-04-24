#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# train_cycshift.sh
#
# Author: Margot Wagner
# Date: 2026-04-23
#
# Train the dense Elman RNN from the cyclic shift hidden-weight initialization.
#
# SEED:
# Base random seed. Main.py automatically offsets this by run index so each
# run receives a distinct seed.
#
# Run this script from the project root:
#   ./scripts/train_cycshift.sh
# ============================================================================

DATA="./data/inputs/Ns100_SeqN100_asym1.pth.tar"
INIT_ROOT="./data/hidden_weight_inits"
RUN_ROOT="./data/runs"

mkdir -p "$RUN_ROOT"

EPOCHS=100000
RUNS=1
SEED=42
CFG="cycshift"

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

# α = 0.0
echo "Running cyclic shift initialization experiment..."
python Main.py "${COMMON_ARGS[@]}" --whh_path "$INIT_ROOT/$CFG/alphasym0p00/Whh.npy" --savename "$RUN_ROOT/$CFG/alpha0p00"
echo "Saving alpha = 0.00 to: $RUN_ROOT/$CFG/alpha0p00"

# α = 0.25
python Main.py "${COMMON_ARGS[@]}" --whh_path "$INIT_ROOT/$CFG/alphasym0p25/Whh.npy" --savename "$RUN_ROOT/$CFG/alpha0p25"
echo "Saving alpha = 0.25 to: $RUN_ROOT/$CFG/alpha0p25"

# α = 0.5
python Main.py "${COMMON_ARGS[@]}" --whh_path "$INIT_ROOT/$CFG/alphasym0p50/Whh.npy" --savename "$RUN_ROOT/$CFG/alpha0p50"
echo "Saving alpha = 0.50 to: $RUN_ROOT/$CFG/alpha0p50"

# α = 0.75
python Main.py "${COMMON_ARGS[@]}" --whh_path "$INIT_ROOT/$CFG/alphasym0p75/Whh.npy" --savename "$RUN_ROOT/$CFG/alpha0p75"
echo "Saving alpha = 0.75 to: $RUN_ROOT/$CFG/alpha0p75"

# α = 1.0
python Main.py "${COMMON_ARGS[@]}" --whh_path "$INIT_ROOT/$CFG/alphasym1p00/Whh.npy" --savename "$RUN_ROOT/$CFG/alpha1p00"
echo "Saving alpha = 1.00 to: $RUN_ROOT/$CFG/alpha1p00"