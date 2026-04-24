#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# train_cycshift.sh
#
# Author: Margot Wagner
# Date: 2026-04-23
#
# Train the dense Elman RNN from the cyclic shift hidden-weight initialization used
# as the baseline condition in the paper.
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
nohup python Main.py $COMMON \
  --whh_path $INIT_ROOT/cycshift/alphasym0p00/Whh.npy \
  --savename $RUN_ROOT/cycshift/alpha0p00/$CFG \
  > $LOG_ROOT/cycshift_alpha0p00.out 2>&1 &

# α = 0.25
nohup python Main.py $COMMON \
  --whh_path $INIT_ROOT/cycshift/alphasym0p25/Whh.npy \
  --savename $RUN_ROOT/cycshift/alpha0p25/$CFG \
  > $LOG_ROOT/cycshift_alpha0p25.out 2>&1 &

# α = 0.5
nohup python Main.py $COMMON \
  --whh_path $INIT_ROOT/cycshift/alphasym0p50/Whh.npy \
  --savename $RUN_ROOT/cycshift/alpha0p50/$CFG \
  > $LOG_ROOT/cycshift_alpha0p50.out 2>&1 &

# α = 0.75
nohup python Main.py $COMMON \
  --whh_path $INIT_ROOT/cycshift/alphasym0p75/Whh.npy \
  --savename $RUN_ROOT/cycshift/alpha0p75/$CFG \
  > $LOG_ROOT/cycshift_alpha0p75.out 2>&1 &

# α = 1.0
nohup python Main.py $COMMON \
  --whh_path $INIT_ROOT/cycshift/alphasym1p00/Whh.npy \
  --savename $RUN_ROOT/cycshift/alpha1p00/$CFG \
  > $LOG_ROOT/cycshift_alpha1p00.out 2>&1 &