#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# Train_Random.sh
#
# Author: Margot Wagner
# Date: 2026-04-23
#
# Train the dense Elman RNN from the random hidden-weight initialization used
# as the baseline condition in the paper.
#
# SEED: base seed; Main.py offsets by run_idx to ensure different seeds across # runs
#
# Run this script from the project root:
#   ./scripts/Train_Random.sh
# ============================================================================

DATA="./data/inputs/prediction/Ns100_SeqN100_asym1.pth.tar"
INIT_ROOT="./data/hidden_weight_inits"
RUN_ROOT="./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5"

mkdir -p "$RUN_ROOT"

EPOCHS=100000
RUNS=3
SEED=42
CFG="cfg_tanh_sigmoid_fi5_fo5"

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

####################################
########## DENSE ###################
####################################

# Vanilla PyTorch random
python Main.py \
  "${COMMON_ARGS[@]}" \
  --whh_path "$INIT_ROOT/random_pytorch/seed000/Whh.npy" \
  --savename "$RUN_ROOT/random_pytorch/$CFG" \