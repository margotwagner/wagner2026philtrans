#!/usr/bin/env bash
# =============================================================================
# figure4.sh
#
# Generate Figure 4 (replay/prediction dynamics for selected conditions)
# using make_figures.py.
#
# This script wraps:
#   fig4_traveling_waves_and_replay(...)
#
# Figure 4 is generated separately for each selected condition. Each call points
# to one condition directory and saves one PNG with replay/prediction dynamics.
#
# ---------------------------------------------------------------------------
# FIGURE DESCRIPTION
# ---------------------------------------------------------------------------
# Each Figure 4 output contains:
#
#   A: Replay hidden-state heatmap
#      - hidden activity is z-scored
#      - units are sorted by time of peak activation
#
#   B: Prediction hidden-state heatmap
#      - same sorting/format as replay
#
#   C: Polar angle trajectories
#      - decoded replay and/or prediction angle over time
#      - true angle is overlaid when available
#
#   D: Replay/prediction metrics table
#      - pulled from condition_summary.csv when matching metrics exist
#
# ---------------------------------------------------------------------------
# REQUIRED INPUT STRUCTURE
# ---------------------------------------------------------------------------
# Each --conditions path must point to one condition root containing:
#
#   run_level.csv
#   condition_summary.csv
#   run_00/, run_01/, ...
#       *.pth.tar
#       *_replay_hidden.npy
#       *_prediction_hidden.npy
#       *_replay_angles.csv
#       *_prediction_angles.csv
#
# Figure 4 selects one "best" run automatically:
#   1) highest ring_decode_R2_replay, if available
#   2) otherwise lowest mse_open
#   3) otherwise lowest last_loss
#   4) otherwise the first row in run_level.csv
#
# ---------------------------------------------------------------------------
# HOW CONDITIONS ARE SPECIFIED
# ---------------------------------------------------------------------------
# Uses:
#   --conditions "<condition_root>"
#
# This script intentionally calls make_figures.py multiple times, once per
# condition, rather than passing many conditions at once. This makes each output
# file easy to name and organize by initialization family / alpha value.
#
# ---------------------------------------------------------------------------
# COLOR SCALE
# ---------------------------------------------------------------------------
# Uses:
#   --vmin -2 --vmax 2
#
# This fixes the heatmap color scale across conditions, making replay and
# prediction activity easier to compare visually across model families.
#
# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------
# Figures are saved to:
#   --figdir
#
# Filename pattern:
#   fig4_<condition_name>_<figtag>.png
#
# where <figtag> identifies the model family / condition, e.g.:
#   dense_random_pytorch
#   dense_identity
#   dense_shift_alpha0p90
#   mh_k5_alpha0p70_1
#   circ_shift_alpha0p50
#
# ---------------------------------------------------------------------------
# KEY FLAGS
# ---------------------------------------------------------------------------
# --just 4        → render only Figure 4
# --conditions    → one condition root directory
# --figdir        → output directory
# --figtag        → suffix added to output filename
# --vmin/--vmax   → fixed heatmap color limits
#
# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
# Run directly:
#   bash figure4.sh
#
# Common uses:
#   1) Inspect replay/prediction trajectories for a single trained condition
#   2) Compare hidden-state traveling-wave structure across initializations
#   3) Generate condition-specific panels for random, identity, shift,
#      Mexican-hat, dense, and circulant models
#
# ---------------------------------------------------------------------------
# NOTES / GOTCHAS
# ---------------------------------------------------------------------------
# - run_level.csv is required because it is used to choose the best run
# - hidden-state .npy files are required for Panels A/B
# - angle .csv files are required for Panel C
# - condition_summary.csv is only needed for the metrics table in Panel D
# - missing files do not crash the script; make_figures.py will skip or mark
#   missing panels where possible
# - the script assumes make_figures.py is located at:
#     ./src/analyze/make_figures.py
#
# =============================================================================

python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/random_pytorch/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/dense --figtag dense_random_pytorch --vmin -2 --vmax 2

python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/identity/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/dense --figtag dense_identity --vmin -2 --vmax 2

python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift/cyclic/alpha0p00/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift --figtag dense_shift_alpha0p00 --vmin -2 --vmax 2

python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift/cyclic/alpha0p75/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift --figtag dense_shift_alpha0p75 --vmin -2 --vmax 2

python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift/cyclic/alpha0p90/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift --figtag dense_shift_alpha0p90 --vmin -2 --vmax 2

python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift/cyclic/alpha1p00/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift --figtag dense_shift_alpha1p00 --vmin -2 --vmax 2

python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_balanced/dog2/k0/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/dog2 --figtag dense_mh_dog2_k0 --vmin -2 --vmax 2

python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_balanced/dog2/k5/alpha1p00/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/dog2 --figtag dense_mh_dog2_k5_alpha1p00 --vmin -2 --vmax 2

#######################################################################
python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_tuned/k5/alpha0p70/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mh_tuned --figtag mh_k5_alpha0p70_1 --vmin -2 --vmax 2

python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_tuned/k0/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mh_tuned --figtag mh_k0 --vmin -2 --vmax 2
#######################################################################
python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/circulant/shift/alpha0p50/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/circulant --figtag circ_shift_alpha0p50 --vmin -2 --vmax 2

python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/circulant/identity/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/circulant --figtag circ_identity --vmin -2 --vmax 2

python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/circulant/mexican_hat/dog_tuned/k0/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/circulant --figtag mh_tuned_k0 --vmin -2 --vmax 2
#######################################################################
python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift/cyclic/alpha0p90/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift --figtag dense_shift_alpha0p90 --vmin -2 --vmax 2
