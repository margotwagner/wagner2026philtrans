#!/usr/bin/env bash
# =============================================================================
# figure1.sh
#
# Generate Figure 1 (training dynamics) using make_figures.py.
#
# This script wraps the Figure 1 pipeline defined in make_figures.py:
#   fig1_training_dynamics(...)
#
# It produces a 2×2 panel figure with:
#   A: Loss vs epoch
#   B: Gradient L2 norm vs epoch
#   C: Spectral radius ρ(W) vs epoch
#   D: Frobenius norm ||W||_F vs epoch
#
# ---------------------------------------------------------------------------
# REQUIRED INPUT STRUCTURE (per condition root)
# ---------------------------------------------------------------------------
# Each path in the `conditions` array must point to a directory containing:
#
#   run_00/, run_01/, ...
#       *_loss_curve.csv     (columns: epoch, loss)
#       *_grad_curve.csv     (columns: epoch, grad_L2_post)
#       *_wstruct_curve.csv  (columns: epoch, spectral_radius, fro_W)
#
# These are discovered automatically via glob inside make_figures.py.
#
# ---------------------------------------------------------------------------
# HOW CONDITIONS ARE USED
# ---------------------------------------------------------------------------
# - Each entry in `conditions` = one curve in Figure 1
# - Curves are averaged across runs (mean ± SEM)
# - Labels MUST match the order of conditions
#
# ---------------------------------------------------------------------------
# LABELS
# ---------------------------------------------------------------------------
# - `labels` array must match `conditions` exactly (same length/order)
# - Labels are passed via --fig1_labels (comma-separated string)
# - These override automatic condition naming
#
# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------
# Figures are saved to:
#   $FIGDIR/
#
# Filenames are:
#   fig1_training_dynamics_<figtag>.png
#
# This script generates three variants:
#   1) log-y      (--fig1_logyA --fig1_logyB)
#   2) log-log    (--fig1_logxA --fig1_logxB --fig1_logyA --fig1_logyB)
#   3) raw        (no log scaling)
#
# ---------------------------------------------------------------------------
# KEY FLAGS (passed to make_figures.py)
# ---------------------------------------------------------------------------
# --just 1            → only Figure 1
# --conditions        → comma-separated list of condition roots
# --fig1_labels       → comma-separated labels (must match conditions)
# --figdir            → output directory
# --figtag            → suffix added to filename
# --fig1_logxA/B      → log-scale x-axis (Panels A/B)
# --fig1_logyA/B      → log-scale y-axis (Panels A/B)
#
# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
# Run directly:
#   bash figure1.sh
#
# Or modify:
#   - condition paths (data selection)
#   - labels (figure legend)
#   - FIGDIR (output location)
#
# ---------------------------------------------------------------------------
# NOTES
# ---------------------------------------------------------------------------
# - Uses Matplotlib backend (no display required)
# - Safe for cluster / headless environments
# - Assumes make_figures.py is located at:
#     ./src/analyze/make_figures.py
#
# - If curves are missing:
#     → check that *_loss_curve.csv etc. exist in run_* directories
#
# =============================================================================
# Figure 1: Training Dynamics

conditions=(
  "./data/runs/random"
  "./data/runs/identity"
  "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_tuned/k5/alpha0p00/cfg_tanh_sigmoid_fi5_fo5_12092025"
  "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_tuned/k5/alpha0p25/cfg_tanh_sigmoid_fi5_fo5_12092025"
  "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_tuned/k5/alpha0p50/cfg_tanh_sigmoid_fi5_fo5_12092025"
  "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_tuned/k5/alpha0p70/cfg_tanh_sigmoid_fi5_fo5_12092025"
  "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_tuned/k5/alpha0p80/cfg_tanh_sigmoid_fi5_fo5_12092025"
  "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_tuned/k5/alpha0p90/cfg_tanh_sigmoid_fi5_fo5_12092025"
  "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_tuned/k5/alpha1p00/cfg_tanh_sigmoid_fi5_fo5_12092025"
)

# Custom labels for Fig 1 (must match number/order of conditions above)
# NO COMMAS
labels=(
  "Random"
  "Mexican Hat k=0"
  "Mexican Hat k=5 α₀=0.00"
  "Mexican Hat k=5 α₀=0.25"
  "Mexican Hat k=5 α₀=0.50"
  "Mexican Hat k=5 α₀=0.70"
  "Mexican Hat k=5 α₀=0.80"
  "Mexican Hat k=5 α₀=0.90"
  "Mexican Hat k=5 α₀=1.00"
)

# Join arrays with commas into single CLI arguments
IFS=, conds_csv="${conditions[*]}"; unset IFS
IFS=, labels_csv="${labels[*]}";   unset IFS

FIGDIR=./data/figures/fig1
mkdir -p "$FIGDIR"

# Log-y version
python ./src/analyze/make_figures.py --just 1 \
  --conditions "$conds_csv" \
  --fig1_labels "$labels_csv" \
  --figdir "$FIGDIR" \
  --figtag logy_mh_tuned \
  --fig1_logyA --fig1_logyB

# Log-log version
python ./src/analyze/make_figures.py --just 1 \
  --conditions "$conds_csv" \
  --fig1_labels "$labels_csv" \
  --figdir "$FIGDIR" \
  --figtag loglog_mh_tuned \
  --fig1_logyA --fig1_logyB --fig1_logxA --fig1_logxB

# Raw-y version
python ./src/analyze/make_figures.py --just 1 \
  --conditions "$conds_csv" \
  --fig1_labels "$labels_csv" \
  --figdir "$FIGDIR" \
  --figtag raw_mh_tuned
