#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# train_identity.sh
#
# Author: Margot Wagner
# Date: 2026-04-23
#
# Train the dense Elman RNN from the identity hidden-weight initialization used
# as the baseline condition in the paper.
#
# SEED:
# Base random seed. Main.py automatically offsets this by run index so each
# run receives a distinct seed.
#
# Run this script from the project root:
#   ./scripts/train_identity.sh
# ============================================================================

DATA="./data/inputs/Ns100_SeqN100_asym1.pth.tar"
INIT_ROOT="./data/hidden_weight_inits"
RUN_ROOT="./data/runs"

mkdir -p "$RUN_ROOT"

EPOCHS=100000
RUNS=1
SEED=42
CFG="identity"

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

# Identity hidden weight initialization
python Main.py \
  "${COMMON_ARGS[@]}" \
  --whh_path "$INIT_ROOT/identity/Whh.npy" \
  --savename "$RUN_ROOT/$CFG"
echo "Running identity initialization experiment..."
echo "Saving to: $RUN_ROOT/$CFG"