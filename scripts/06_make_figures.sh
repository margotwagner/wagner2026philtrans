#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 06_make_figures.sh
#
# Generate manuscript figures.
#
# Run from repository root:
#
#   bash scripts/06_make_figures.sh
#
# Requirements
# ------------
# Must be run after:
#
#   bash scripts/05_analyze.sh
#
# Note: this script assumes the following have been run:
#
#   bash scripts/01_build_inputs.sh
#   bash scripts/02_build_hidden_weights.sh
#   bash scripts/03_train_random.sh
#   bash scripts/04_train_mexican_hat.sh
#   bash scripts/05_analyze.sh
#
# ============================================================================

FIG_ROOT="data/figures"

mkdir -p "$FIG_ROOT/figure1/mexican_hat/k5"
mkdir -p "$FIG_ROOT/figure2/mexican_hat/k5"
mkdir -p "$FIG_ROOT/figure3/mexican_hat/k5"
mkdir -p "$FIG_ROOT/figure4/mexican_hat/k5"
mkdir -p "$FIG_ROOT/figure5/mexican_hat/k5"
mkdir -p "$FIG_ROOT/figure6/mexican_hat/k5"
mkdir -p "$FIG_ROOT/figure7/mexican_hat/k5"

mkdir -p "$FIG_ROOT/figure1/random"
mkdir -p "$FIG_ROOT/figure2/random"
mkdir -p "$FIG_ROOT/figure3/random"
mkdir -p "$FIG_ROOT/figure4/random"
mkdir -p "$FIG_ROOT/figure5/random"
mkdir -p "$FIG_ROOT/figure6/random"
mkdir -p "$FIG_ROOT/figure7/random"


echo "============================================================"
echo "Generating manuscript figures..."
echo "============================================================"

# --------------------------------------------------------------------------
# Figure 1: initial connectivity
# --------------------------------------------------------------------------
echo
echo "Generating Figure 1..."

if [[ ! -f "data/hidden_weight_inits/mexican_hat/k5/alpha0p70/Whh.npy" ]]; then
  echo "ERROR: Figure 1 input not found [Mexican hat]."
  exit 1
fi


# Mexican hat initial connectivity with α₀=0.70
python src/figures/figure1_initial_connectivity.py \
  data/hidden_weight_inits/mexican_hat/k5/alpha0p70/Whh.npy \
  --savepath "$FIG_ROOT/figure1/mexican_hat/k5/alpha0p70.png" \
  --alpha-label 0.70 \
  --trace-lw 4 \
  --no-show

if [[ ! -f "data/hidden_weight_inits/random/seed000/Whh.npy" ]]; then
  echo "ERROR: Figure 1 input not found [Random]."
  exit 1
fi

# Random initialization baseline
python src/figures/figure1_initial_connectivity.py \
  data/hidden_weight_inits/random/seed000/Whh.npy \
  --savepath "$FIG_ROOT/figure1/random/random.png" \
  --trace-lw 4 \
  --no-show

# --------------------------------------------------------------------------
# Figure 2: replay and prediction polar trajectories
# --------------------------------------------------------------------------
echo
echo "Generating Figure 2..."

# Mexican hat initial connectivity with α₀=0.70
python src/figures/figure2_polar_trajectory.py \
  data/runs/mexican_hat/k5/alpha0p70/run_00 \
  --savepath "$FIG_ROOT/figure2/mexican_hat/k5/run_00_polar.png" \
  --title "Mexican hat initialization" \
  --fontsize 22 \
  --linewidth 4

echo "Saved figure to: $FIG_ROOT/figure2/mexican_hat/k5/run_00_polar.png"

# Random initialization baseline
python src/figures/figure2_polar_trajectory.py \
  data/runs/random/run_00 \
  --savepath "$FIG_ROOT/figure2/random/run_00_polar.png" \
  --title "Random initialization" \
  --fontsize 22 \
  --linewidth 4

echo "Saved figure to: $FIG_ROOT/figure2/random/run_00_polar.png"

# --------------------------------------------------------------------------
# Figure 3: replay and prediction hidden activity
# --------------------------------------------------------------------------
echo
echo "Generating Figure 3..."

# Mexican hat initial connectivity with α₀=0.70
python src/figures/figure3_hidden_activity.py \
  data/runs/mexican_hat/k5/alpha0p70/run_00 \
  --savepath "$FIG_ROOT/figure3/mexican_hat/k5/alpha0p70_run_00.png" \
  --fontsize 20

echo "Saved figure to: $FIG_ROOT/figure3/mexican_hat/k5/alpha0p70_run_00.png"

# Random initialization baseline
python src/figures/figure3_hidden_activity.py \
  data/runs/random/run_00 \
  --savepath "$FIG_ROOT/figure3/random/run_00.png" \
  --fontsize 20

echo "Saved figure to: $FIG_ROOT/figure3/random/run_00.png"

# --------------------------------------------------------------------------
# Figure 4: learned recurrent weights
# --------------------------------------------------------------------------
echo
echo "Generating Figure 4..."

# Mexican hat initial connectivity with α₀=0.70
python src/figures/figure4_learned_weights.py \
  data/runs/mexican_hat/k5/alpha0p70 \
  --savepath "$FIG_ROOT/figure4/mexican_hat/k5/alpha0p70.png" \
  --fontsize 14 \
  --no-show

# Random initialization baseline
python src/figures/figure4_learned_weights.py \
  data/runs/random \
  --savepath "$FIG_ROOT/figure4/random/random.png" \
  --fontsize 14 \
  --no-show

# --------------------------------------------------------------------------
# Figure 5: decomposed learned weights
# --------------------------------------------------------------------------
echo
echo "Generating Figure 5..."

# Mexican hat initial connectivity with α₀=0.70
python src/figures/figure5_decomposed_weights.py \
  data/runs/mexican_hat/k5/alpha0p70 \
  --savepath "$FIG_ROOT/figure5/mexican_hat/k5/alpha0p70" \
  --fontsize 14 \

# Random initialization baseline
python src/figures/figure5_decomposed_weights.py \
  data/runs/random \
  --savepath "$FIG_ROOT/figure5/random/random" \
  --fontsize 14 \

# --------------------------------------------------------------------------
# Figure 6: training dynamics
# --------------------------------------------------------------------------
echo
echo "Generating Figure 6..."

python src/figures/figure6_training_dynamics.py \
  data/runs/random \
  data/runs/mexican_hat/k5/alpha0p70 \
  --labels \
    "random" \
    "mexican hat α₀=0.70" \
  --savepath "$FIG_ROOT/figure6/random_vs_mexican_hat_alpha0p70_log.png" \
  --fontsize 16 \
  --logx \
  --logy \
  --no-slope \
  --mh-color "#2c7fb8" \
  --lw 4

python src/figures/figure6_training_dynamics.py \
  data/runs/random \
  data/runs/mexican_hat/k5/alpha0p70 \
  --labels \
    "random" \
    "mexican hat α₀=0.70" \
  --savepath "$FIG_ROOT/figure6/random_vs_mexican_hat_alpha0p70_raw.png" \
  --fontsize 16 \
  --no-slope \
  --mh-color "#2c7fb8" \
  --lw 4

# --------------------------------------------------------------------------
# Figure 7: alpha training dynamics
# --------------------------------------------------------------------------
echo
echo "Generating Figure 7..."

# Mexican hat initial connectivity with α₀=0.70
python src/figures/figure7_alpha_training.py \
  --condition-roots \
    data/runs/random \
    data/runs/mexican_hat/k5/alpha0p70 \
  --savepath "$FIG_ROOT/figure7/random_vs_mexican_hat_alpha0p70.png" \
  --fontsize 16 \
  --median-lw 4 \
  --mh-color "#2c7fb8"

echo
echo "Finished generating figures."
echo "Figures saved to:"
echo "  $FIG_ROOT"