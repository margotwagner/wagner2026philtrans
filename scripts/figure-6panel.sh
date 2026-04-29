#!/usr/bin/env bash
# =============================================================================
# figure2.sh
#
# Generate Figure 2 (performance vs α₀) using make_figures.py.
#
# This script wraps:
#   fig2_performance_sixpanel(...)
#
# It produces a 2×3 panel figure showing performance metrics as a function of
# the initial symmetry/antisymmetry mixing ratio (α₀).
#
# ---------------------------------------------------------------------------
# FIGURE DESCRIPTION (from make_figures.py)
# ---------------------------------------------------------------------------
# Panels:
#   A: Best training loss (↓)            vs α₀   [best_loss]
#   B: Best open-loop MSE (↓)           vs α₀   [mse_open]
#   C: Best closed-loop MSE (↓)         vs α₀   [mse_free_closed]
#   D: Prediction MSE (↓)               vs α₀   [mse_prediction]
#   E: Replay MSE (↓)                   vs α₀   [mse_replay]
#   F: Replay ring R² (↑)               vs α₀   [ring_decode_R2_replay]
#
# Each curve corresponds to a model family (e.g., random, identity, Mexican hat).
# Points are aggregated across runs (mean ± std if available).
#
# ---------------------------------------------------------------------------
# REQUIRED INPUT STRUCTURE (per condition root)
# ---------------------------------------------------------------------------
# Each condition directory must contain:
#
#   run_level.csv
#   condition_summary.csv
#
# Figure 2 uses ONLY condition_summary.csv, which must include:
#   - condition_id
#   - metric
#   - mean
#   - (optional) std
#
# The script automatically:
#   - extracts α₀ from condition_id (e.g., "alpha0p25" → 0.25)
#   - groups conditions by "family" (parsed from directory structure)
#   - sorts points by α₀ before plotting
#
# ---------------------------------------------------------------------------
# HOW CONDITIONS ARE SPECIFIED
# ---------------------------------------------------------------------------
# Uses:
#   --cond_glob "<pattern>"
#
# Key behavior:
#   - Each matched directory = one condition
#   - Wildcards (e.g., alpha*) are used to sweep α₀
#   - Multiple globs can be provided to compare different families
#
# Examples:
#   alpha sweep:
#     mexican_hat/.../k5/alpha*/cfg_...
#
#   fixed conditions:
#     random_pytorch/
#     identity/
#
# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------
# Figures are saved to:
#   --figdir
#
# Filename:
#   fig2_performance_vs_alpha_6panel_<figtag>.png
#
# ---------------------------------------------------------------------------
# KEY FLAGS (passed to make_figures.py)
# ---------------------------------------------------------------------------
# --just 2        → render only Figure 2
# --cond_glob     → one or more condition glob patterns
# --figdir        → output directory
# --figtag        → suffix for output filename
#
# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
# Run directly:
#   bash figure2.sh
#
# Common patterns:
#
# 1) Compare fixed initializations:
#   random vs identity vs MH (k=0)
#
# 2) Sweep α₀ for a single family:
#   alpha*/...
#
# 3) Compare multiple families across α₀:
#   multiple calls with different --figtag
#
# ---------------------------------------------------------------------------
# NOTES / GOTCHAS
# ---------------------------------------------------------------------------
# - α₀ is parsed from condition_id strings (must contain "alphaXpYY")
# - Missing metrics → panel will show "No data"
# - Families are inferred from directory structure (not explicitly passed)
# - condition_summary.csv must be present (run_level.csv is ignored here)
#
# - Globs are expanded by the script (NOT the shell), so always quote them:
#     --cond_glob "./path/alpha*/cfg_*"
#
# =============================================================================

# Multiple Conditions
python make_figures.py --just 2 \
  --cond_glob "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/random_pytorch/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/identity/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_balanced/dog1/k0/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_balanced/dog2/k0/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_balanced/dog3/k0/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig2/cfg_tanh_sigmoid_fi5_fo5_12092025/dense --figtag rand_identity_k0

###############################################################
python ./src/analyze/make_figures.py --just 2 \
  --cond_glob "./data/runs/prediction/cfg_relu_sigmoid_fi6_fo6_12092025/dense/mexican_hat/dog_balanced/dog1/k5/alpha*/cfg_relu_sigmoid_fi6_fo6_12092025" \
  --figdir ./data/figures/prediction/fig2/cfg_relu_sigmoid_fi6_fo6_12092025/dense --figtag dog1

python ./src/analyze/make_figures.py --just 2 \
  --cond_glob "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_tuned/k5/alpha*/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig2/cfg_tanh_sigmoid_fi5_fo5_12092025/dense --figtag mh_tuned

###############################################################
python ./src/analyze/make_figures.py --just 2 \
  --cond_glob "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_balanced/dog3/k5/alpha*/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig2/cfg_tanh_sigmoid_fi5_fo5_12092025/dense --figtag dog3

python ./src/analyze/make_figures.py --just 2 \
  --cond_glob "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift/cyclic/alpha*/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig2/cfg_tanh_sigmoid_fi5_fo5_12092025/dense --figtag shift