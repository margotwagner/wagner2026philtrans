#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 07_analyze.sh
#
# Run evaluation, offline metrics, and aggregation for all trained conditions.
#
# Run from the repository root:
#
#   bash scripts/07_analyze.sh
#
# Outputs
# -------
# For each condition directory, this script writes:
#
#   *_eval.csv
#   run_XX/*_train_summary.csv
#   run_XX/*_offline_metrics.csv
#   run_XX/*_spectra.npz
#   run_level.csv
#   condition_summary.csv
#
# Requirements
# ------------
# Must be run after training:
#
#   bash scripts/03_train_random.sh
#   bash scripts/04_train_mexican_hat.sh
#   bash scripts/05_train_cycshift.sh
#   bash scripts/06_train_identity.sh
# ============================================================================

RUNS_ROOT="data/runs"
MODE="all"
RUNS=""   # Empty means auto-discover all run_* folders
MAIN_EVAL="src/analyze/evaluate.py"
MAIN_OFFLINE="src/analyze/offline_metrics.py"
MAIN_AGG="src/analyze/aggregate_metrics.py"

CONDITION_ROOTS=(
  "$RUNS_ROOT/random"
  "$RUNS_ROOT/identity"

  "$RUNS_ROOT/cycshift/alpha0p00"
  "$RUNS_ROOT/cycshift/alpha0p25"
  "$RUNS_ROOT/cycshift/alpha0p50"
  "$RUNS_ROOT/cycshift/alpha0p75"
  "$RUNS_ROOT/cycshift/alpha1p00"

  "$RUNS_ROOT/mexicanhat/k0"
  "$RUNS_ROOT/mexicanhat/k5/alpha0p00"
  "$RUNS_ROOT/mexicanhat/k5/alpha0p25"
  "$RUNS_ROOT/mexicanhat/k5/alpha0p50"
  "$RUNS_ROOT/mexicanhat/k5/alpha0p75"
  "$RUNS_ROOT/mexicanhat/k5/alpha1p00"
)

echo "============================================================"
echo "Running analysis pipeline..."
echo "============================================================"

for BASE_DIR in "${CONDITION_ROOTS[@]}"; do
  if [[ ! -d "$BASE_DIR" ]]; then
    echo
    echo "WARNING: condition directory not found, skipping:"
    echo "  $BASE_DIR"
    continue
  fi

  TAG="$(basename "$BASE_DIR")"

  # Make CSV names less ambiguous for nested conditions.
  SAFE_TAG="$(echo "$BASE_DIR" | sed 's#^data/runs/##' | tr '/' '_')"
  CSV_PATH="$BASE_DIR/${SAFE_TAG}_eval.csv"

  LOG="$BASE_DIR/analyze_$(date +%Y%m%d_%H%M%S).log"

  echo
  echo "------------------------------------------------------------"
  echo "Analyzing condition:"
  echo "  $BASE_DIR"
  echo "CSV:"
  echo "  $CSV_PATH"
  echo "Log:"
  echo "  $LOG"
  echo "------------------------------------------------------------"

  {
    echo "========== evaluate.py =========="
    python "$MAIN_EVAL" \
      --base-dir "$BASE_DIR" \
      --runs "$RUNS" \
      --mode "$MODE" \
      --csv "$CSV_PATH"

    echo
    echo "========== offline_metrics.py =========="
    python "$MAIN_OFFLINE" \
      --ckpt "$BASE_DIR"

    echo
    echo "========== aggregate_metrics.py =========="
    python "$MAIN_AGG" \
      --root "$BASE_DIR"

    echo
    echo "Finished analysis for: $BASE_DIR"
  } 2>&1 | tee "$LOG"
done

echo
echo "Finished full analysis pipeline."