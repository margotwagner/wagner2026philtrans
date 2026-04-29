#!/usr/bin/env bash
# =============================================================================
# figure3.sh
#
# Generate Figure 3 (alpha over training) using make_figures.py.
#
# This script wraps:
#   fig3_alpha_time_series(...)
#
# Figure 3 plots how the learned symmetry balance changes during training:
#
#   alpha(t) = ||S||_F / (||S||_F + ||A||_F)
#
# where:
#   S = symmetric component of W_hh
#   A = antisymmetric component of W_hh
#
# ---------------------------------------------------------------------------
# FIGURE DESCRIPTION
# ---------------------------------------------------------------------------
# The output figure shows:
#   - thin traces for individual runs
#   - thicker per-condition median trajectory
#   - shaded interquartile range (IQR)
#
# The y-axis is alpha(t), where:
#   alpha(t) close to 1 → mostly symmetric recurrence
#   alpha(t) close to 0 → mostly antisymmetric recurrence
#
# ---------------------------------------------------------------------------
# REQUIRED INPUT STRUCTURE
# ---------------------------------------------------------------------------
# Each condition root passed through --cond_glob must contain:
#
#   run_level.csv
#
# Figure 3 requires run_level.csv to include:
#   - fro_S_offline
#   - fro_A_offline
#   - epoch or snapshot_idx
#   - run_id, if multiple runs are present
#
# make_figures.py computes alpha(t) directly from fro_S_offline and
# fro_A_offline.
#
# ---------------------------------------------------------------------------
# HOW CONDITIONS ARE SPECIFIED
# ---------------------------------------------------------------------------
# Uses:
#   --cond_glob "<pattern1>" "<pattern2>" ...
#
# Each matched directory is treated as one condition.
#
# Wildcards such as:
#   alpha*/
#   sym*/
#
# are used to collect sweeps over initialization symmetry.
#
# ---------------------------------------------------------------------------
# COLORING / LEGENDS
# ---------------------------------------------------------------------------
# By default, curves are colored by model family inferred from the path.
#
# Special case:
#   If exactly one --cond_glob pattern contains "*" and expands to multiple
#   conditions, make_figures.py colors/labels curves by alpha folder instead
#   of family.
#
# Line style is used to distinguish alpha0 values when coloring by family.
#
# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------
# Figures are saved to:
#   --figdir
#
# Filename:
#   fig3_alpha_time_series_<figtag>.png
#
# ---------------------------------------------------------------------------
# KEY FLAGS
# ---------------------------------------------------------------------------
# --just 3      → render only Figure 3
# --cond_glob   → one or more condition root globs
# --figdir      → output directory
# --figtag      → suffix added to output filename
#
# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
# Run directly:
#   bash figure3.sh
#
# Common uses:
#   1) Compare random / identity / Mexican-hat initializations
#   2) Plot alpha(t) across a full alpha*/ sweep
#   3) Compare shift vs Mexican-hat families across symmetry conditions
#
# ---------------------------------------------------------------------------
# NOTES / GOTCHAS
# ---------------------------------------------------------------------------
# - run_level.csv is required; condition_summary.csv is not used for Figure 3
# - missing fro_S_offline or fro_A_offline will cause Figure 3 to be skipped
# - alpha0 is parsed from path/condition strings such as alpha0p25 or sym0p25
# - always quote glob patterns so Python receives the wildcard:
#     --cond_glob "./path/alpha*/cfg_*"
#
# =============================================================================

# Multiple Conditions
python make_figures.py --just 3 \
  --cond_glob "./runs/ElmanRNN/shift-variants/shift-cyc/frobenius/sym*/shiftcyc_n100_fro_sym*" \
  "./runs/ElmanRNN/shift-variants/shift/frobenius/sym*/shift_n100_fro_sym*" \
  "./runs/ElmanRNN/mh-variants/shifted-cyc/frobenius/sym*/shiftcycmh_n100_fro_sym*" \
  "./runs/ElmanRNN/mh-variants/shifted/frobenius/sym*/shiftmh_n100_fro_sym*" \
  "./runs/ElmanRNN/random-init/random_n100" \
  "./runs/ElmanRNN/shift-variants/identity/frobenius/identity_n100_fro" \
  --figdir ./figs/fig3 --figtag all

# Unconstrained
python make_figures.py --just 3 \
  --cond_glob "./runs/ElmanRNN/identityih/random_baseline" \
  "./runs/ElmanRNN/identityih/shift-variants/identity" \
  "./runs/ElmanRNN/identityih/shift-variants/shift/sym*/shift_sym*" \
  "./runs/ElmanRNN/identityih/shift-variants/cyc-shift/sym*/cycshift_sym*" \
  "./runs/ElmanRNN/identityih/mh-variants/shifted/sym*/shiftmh_sym*" \
  --figdir ./figs/fig3 --figtag idih

# Constrained
python make_figures.py --just 3 \
  --cond_glob "./runs/ElmanRNN/circulant/identity" \
   "./runs/ElmanRNN/circulant/centeredmh" \
  "./runs/ElmanRNN/circulant/shift/sym*/shift_circ_sym*" \
  "./runs/ElmanRNN/circulant/shiftedmh/sym*/shiftedmh_circ_sym*" \
  --figdir ./figs/fig3 --figtag circ

###############################################################
python ./src/analyze/make_figures.py --just 3 \
  --cond_glob "./data/runs/prediction/cfg_relu_sigmoid_fi2_fo2_12092025/dense/random_pytorch/cfg_relu_sigmoid_fi2_fo2_12092025" \
  "./data/runs/prediction/cfg_relu_sigmoid_fi2_fo2_12092025/dense/identity/cfg_relu_sigmoid_fi2_fo2_12092025" \
  "./data/runs/prediction/cfg_relu_sigmoid_fi2_fo2_12092025/dense/shift/cyclic/alpha*/cfg_relu_sigmoid_fi2_fo2_12092025" \
  "./data/runs/prediction/cfg_relu_sigmoid_fi2_fo2_12092025/dense/mexican_hat/dog2/k5/alpha0p00/cfg_relu_sigmoid_fi2_fo2_12092025" \
  --figdir ./data/figures/fig3/cfg_relu_sigmoid_fi2_fo2_12092025/dense

###############################################################
python ./src/analyze/make_figures.py --just 3 \
  --cond_glob "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/random_pytorch/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/identity/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_balanced/dog2/k0/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig3/cfg_tanh_sigmoid_fi5_fo5_12092025/dense --figtag rand_identity_k0

###############################################################
python ./src/analyze/make_figures.py --just 3 \
  --cond_glob "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_balanced/dog2/k5/alpha*/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig3/cfg_tanh_sigmoid_fi5_fo5_12092025/dense \
  --figtag dog2

python ./src/analyze/make_figures.py --just 3 \
  --cond_glob "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_tuned/k5/alpha*/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig3/cfg_tanh_sigmoid_fi5_fo5_12092025/dense \
  --figtag mh_tuned

python ./src/analyze/make_figures.py --just 3 \
  --cond_glob "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift/cyclic/alpha*/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig3/cfg_tanh_sigmoid_fi5_fo5_12092025/dense --figtag shift