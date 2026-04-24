#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# train_mexicanhat.sh
#
# Author: Margot Wagner
# Date: 2026-04-23
#
# Train the dense Elman RNN from the Mexican hat hidden-weight initialization.
#
# SEED:
# Base random seed. Main.py automatically offsets this by run index so each
# run receives a distinct seed.
#
# Run this script from the project root:
#   ./scripts/train_mexicanhat.sh
# ============================================================================

DATA=DATA="./data/inputs/Ns100_SeqN100_asym1.pth.tar"
INIT_ROOT="./data/hidden_weight_inits"
RUN_ROOT="./data/runs"

mkdir -p "$RUN_ROOT"

EPOCHS=100000
RUNS=1
SEED=42
CFG="mexicanhat"

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

# centered Mexican hat initialization (k = 0)
echo "Running Mexican hat initialization experiment..."
nohup python Main.py "${COMMON_ARGS[@]}" \
  --whh_path $INIT_ROOT/k0/Whh.npy \
  --savename $RUN_ROOT/k0/$CFG

# α = 0.0
nohup python Main.py "${COMMON_ARGS[@]}" --whh_path "$INIT_ROOT/$CFG/k5/alphasym0p00/Whh.npy" --savename "$RUN_ROOT/$CFG/k5/alpha0p00"

# α = 0.25
nohup python Main.py "${COMMON_ARGS[@]}" --whh_path "$INIT_ROOT/$CFG/k5/alphasym0p25/Whh.npy" --savename "$RUN_ROOT/$CFG/k5/alpha0p25"

# α = 0.50
nohup python Main.py "${COMMON_ARGS[@]}" --whh_path "$INIT_ROOT/$CFG/k5/alphasym0p50/Whh.npy" --savename "$RUN_ROOT/$CFG/k5/alpha0p50"

# α = 0.75
nohup python Main.py "${COMMON_ARGS[@]}" --whh_path "$INIT_ROOT/$CFG/k5/alphasym0p75/Whh.npy" --savename "$RUN_ROOT/$CFG/k5/alpha0p75"

# α = 1.0
nohup python Main.py "${COMMON_ARGS[@]}" --whh_path "$INIT_ROOT/$CFG/k5/alphasym1p00/Whh.npy" --savename "$RUN_ROOT/$CFG/k5/alpha1p00"