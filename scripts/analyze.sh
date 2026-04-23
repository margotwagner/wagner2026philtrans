#!/usr/bin/env bash
set -euo pipefail

# Analyze a single condition:
#  1) evaluate.py        (per-run metrics -> CSV)
#  2) offline_metrics.py (per-run offline metrics)
#  3) aggregate_metrics  (condition-level tables)
#
# Usage:
#   ./Analyze.sh BASE_DIR [RUNS_RANGE] [MODE]
#
# Examples:
#   ./Analyze.sh ./runs/ElmanRNN/cfg_tanh_linear_fi0_fo0_11252025/dense/random_pytorch 0-2
#   ./Analyze.sh ./runs/ElmanRNN/cfg_tanh_linear_fi0_fo0_11252025/dense/mexican_hat/dog2/k5/alpha0p50 0-2 all

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 BASE_DIR [RUNS_RANGE] [MODE]"
  echo "  BASE_DIR    Directory containing run_XX subdirs (e.g."
  echo "              ./runs/ElmanRNN/cfg_tanh_linear_fi0_fo0_11252025/dense/random_pytorch)"
  echo "  RUNS_RANGE  String for --runs (default: 0-2)"
  echo "  MODE        evaluate.py --mode (default: all)"
  exit 1
fi

BASE_DIR="$1"
RUNS="${2:-0-2}"
MODE="${3:-all}"

# infer a tag for the CSV name from the last path component
TAG="$(basename "$BASE_DIR")"
CSV_PATH="${BASE_DIR}/${TAG}_eval.csv"

echo "========== Analyzing condition =========="
echo "BASE_DIR : $BASE_DIR"
echo "RUNS     : $RUNS"
echo "MODE     : $MODE"
echo "CSV      : $CSV_PATH"
echo "========================================="
echo

# 1) Evaluate all runs for this condition
python ../src/analyze/evaluate.py \
  --base-dir "$BASE_DIR" \
  --runs "$RUNS" \
  --mode "$MODE" \
  --csv "$CSV_PATH"

echo
echo "Finished evaluate.py"
echo

# 2) Run offline metrics generation
python ../src/analyze/offline_metrics.py --ckpt "$BASE_DIR"

echo
echo "Finished offline_metrics.py"
echo

# 3) Aggregate across runs (condition-level tables)
python ../src/analyze/aggregate_metrics.py --root "$BASE_DIR"

echo
echo "Finished aggregate_metrics.py"
echo "All analysis steps done for: $BASE_DIR"
