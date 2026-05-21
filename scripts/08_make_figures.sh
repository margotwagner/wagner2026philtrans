#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 08_make_figures.sh
#
# Generate manuscript figures.
#
# Run from repository root:
#
#   bash scripts/08_make_figures.sh
#
# Requirements
# ------------
# Must be run after:
#
#   bash scripts/07_analyze.sh
# ============================================================================

FIG_ROOT="data/figures"

mkdir -p "$FIG_ROOT/figure1"
mkdir -p "$FIG_ROOT/figure2"
mkdir -p "$FIG_ROOT/figure3"
mkdir -p "$FIG_ROOT/figure4"

echo "============================================================"
echo "Generating manuscript figures..."
echo "============================================================"

# --------------------------------------------------------------------------
# Figure 1: initial connectivity
# --------------------------------------------------------------------------
echo
echo "Generating Figure 1..."

if [[ ! -f "data/hidden_weight_inits/mexican_hat/k5/alpha0p75/Whh.npy" ]]; then
  echo "ERROR: Figure 1 input not found."
  exit 1
fi

python src/figures/figure1_initial_connectivity.py \
  data/hidden_weight_inits/mexican_hat/k5/alpha0p75/Whh.npy \
  --savepath "$FIG_ROOT/figure1/mexican_hat/k5/alpha0p75.png" \
  --alpha-label 0.75 \
  --trace-lw 4 \
  --no-show

# --------------------------------------------------------------------------
# Figure 2: replay and prediction polar trajectories
# --------------------------------------------------------------------------
echo
echo "Generating Figure 2..."

python src/figures/figure2_polar_trajectory.py \
  data/runs/mexican_hat/k5/alpha0p75/run_00 \
  --savepath "$FIG_ROOT/figure2/mexican_hat/k5/run_00_polar.png" \
  --title "Mexican hat initialization" \
  --fontsize 22 \
  --linewidth 4

echo
echo "Finished generating figures."
echo "Figures saved to:"
echo "  $FIG_ROOT"

# --------------------------------------------------------------------------
# Figure 3: replay and prediction hidden activity
# --------------------------------------------------------------------------
echo
echo "Generating Figure 3..."

python src/figures/figure3_hidden_activity.py \
  data/runs/mexican_hat/k5/alpha0p75/run_00 \
  --savepath "$FIG_ROOT/figure3/mexican_hat/k5/alpha0p75_run_00.png" \
  --fontsize 20

# --------------------------------------------------------------------------
# Figure 4: learned recurrent weights
# --------------------------------------------------------------------------
echo
echo "Generating Figure 4..."

python src/figures/figure4_learned_weights.py \
  data/runs/mexican_hat/k5/alpha0p75 \
  --savepath "$FIG_ROOT/figure4/mexican_hat/k5/alpha0p75.png" \
  --fontsize 14 \
  --no-show