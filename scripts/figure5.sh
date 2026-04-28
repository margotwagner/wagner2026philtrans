#!/usr/bin/env bash
# =============================================================================
# figure5.sh
#
# Generate Figure 5 (recurrent weight structure and eigenspectrum) using
# make_figures.py.
#
# This script wraps:
#   fig5_weights_and_spectrum_from_checkpoints(...)
#
# Figure 5 is generated separately for each selected condition. Each call loads
# checkpoint weight histories, sorts recurrent weights using hidden-state peak
# timing when available, and saves weight-structure panels across training.
#
# ---------------------------------------------------------------------------
# FIGURE DESCRIPTION
# ---------------------------------------------------------------------------
# Each Figure 5 output contains three panels:
#
#   1) Sorted recurrent weight heatmap
#      - mean W_hh across runs
#      - units are sorted by replay/prediction peak time when possible
#
#   2) Diagonal trace summary
#      - mean ± SD of diagonal-offset traces across runs
#      - shows whether weights are centered, shifted, or asymmetric
#
#   3) Eigenspectrum
#      - eigenvalues of each run shown as a faint cloud
#      - eigenvalues of mean(W) overlaid
#      - unit circle shown as a stability reference
#
# ---------------------------------------------------------------------------
# REQUIRED INPUT STRUCTURE
# ---------------------------------------------------------------------------
# Each --conditions path must point to one condition root containing:
#
#   run_00/, run_01/, ...
#       *.pth.tar
#       optional: *_replay_hidden.npy
#       optional: *_prediction_hidden.npy
#
# Checkpoints must contain:
#   weights["W_hh_history"]
#   snapshot_epochs
#
# Hidden traces are optional but useful. If present, they are used to compute a
# peak-time permutation so W_hh can be displayed in functional order.
#
# ---------------------------------------------------------------------------
# TIMEPOINTS
# ---------------------------------------------------------------------------
# This script uses:
#   --fig5_time all
#
# In make_figures.py, "all" means:
#   first, middle, last
#
# Therefore, each condition produces three PNGs:
#   *_first.png
#   *_middle.png
#   *_last.png
#
# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------
# Figures are saved to:
#   --figdir
#
# Filename pattern:
#   fig5_weights_and_spectrum_<figtag>_<condition_name>_<timepoint>.png
#
# Example tags:
#   dense_random_pytorch
#   dense_identity
#   dense_shift_alpha0p25
#   mh_k5_alpha0p75
#   circ_mh_tuned_k0
#
# ---------------------------------------------------------------------------
# KEY FLAGS
# ---------------------------------------------------------------------------
# --just 5       → render only Figure 5
# --conditions   → one condition root directory
# --fig5_time    → timepoint(s): first, middle, last, all, or epoch values
# --figdir       → output directory
# --fontsize     → base font size
# --figtag       → suffix identifying the model condition
#
# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
# Run directly:
#   bash figure5.sh
#
# Common uses:
#   1) Compare learned W_hh structure across initializations
#   2) Inspect how recurrent structure changes from first to middle to last
#   3) Compare dense vs circulant models
#   4) Compare random, identity, shift, and Mexican-hat initializations
#
# ---------------------------------------------------------------------------
# NOTES / GOTCHAS
# ---------------------------------------------------------------------------
# - Figure 5 uses checkpoints as the source of truth, not run_level.csv
# - If W_hh_history or snapshot_epochs are missing, the condition is skipped
# - If hidden traces are missing, W_hh is plotted unsorted
# - A cached sorted weight matrix is saved under:
#     run_XX/analysis/weights/
# - The script assumes make_figures.py is located at:
#     ./src/analyze/make_figures.py
#
# =============================================================================

python ./src/analyze/make_figures.py \
  --just 5 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/random_pytorch/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --fig5_time all \
  --figdir ./data/figures/prediction/fig5/cfg_tanh_sigmoid_fi5_fo5_12092025/dense --fontsize 12 \
  --figtag dense_random_pytorch

python ./src/analyze/make_figures.py \
  --just 5 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/identity/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --fig5_time all \
  --figdir ./data/figures/prediction/fig5/cfg_tanh_sigmoid_fi5_fo5_12092025/dense --fontsize 12 \
  --figtag dense_identity

python ./src/analyze/make_figures.py \
  --just 5 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift/cyclic/alpha0p60/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --fig5_time all \
  --figdir ./data/figures/prediction/fig5/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift --fontsize 12 \
  --figtag dense_shift_alpha0p60

python ./src/analyze/make_figures.py \
  --just 5 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift/cyclic/alpha0p25/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --fig5_time all \
  --figdir ./data/figures/prediction/fig5/cfg_tanh_sigmoid_fi5_fo5_12092025/dense --fontsize 12 \
  --figtag dense_shift_alpha0p25

python ./src/analyze/make_figures.py \
  --just 5 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift/cyclic/alpha1p00/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --fig5_time all \
  --figdir ./data/figures/prediction/fig5/cfg_tanh_sigmoid_fi5_fo5_12092025/dense --fontsize 12 \
  --figtag dense_shift_alpha1p00

#######################################################################
python ./src/analyze/make_figures.py \
  --just 5 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_tuned/k5/alpha0p75/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --fig5_time all \
  --figdir ./data/figures/prediction/fig5/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/dog_tuned --fontsize 12 \
  --figtag mh_k5_alpha0p75

#######################################################################
python ./src/analyze/make_figures.py \
  --just 5 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_balanced/dog2/k5/alpha1p00/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --fig5_time all \
  --figdir ./data/figures/prediction/fig5/cfg_tanh_sigmoid_fi5_fo5_12092025/dense --fontsize 12 \
  --figtag dense_mh_balanced_k5_alpha1p00

python ./src/analyze/make_figures.py \
  --just 5 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_balanced/dog2/k0/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --fig5_time all \
  --figdir ./data/figures/prediction/fig5/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/dog2 --fontsize 12 \
  --figtag dense_mh_balanced_k0

python ./src/analyze/make_figures.py \
  --just 5 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/circulant/mexican_hat/dog_tuned/k0/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --fig5_time all \
  --figdir ./data/figures/prediction/fig5/cfg_tanh_sigmoid_fi5_fo5_12092025/circulant/dog_tuned --fontsize 12 \
  --figtag circ_mh_tuned_k0