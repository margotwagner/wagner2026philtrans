#!/usr/bin/env bash
set -euo pipefail

# Build hidden recurrent weight initializations needed for the main figures.
# Outputs:
#   data/hidden_weight_inits/random/seed000/Whh.npy
#   data/hidden_weight_inits/mexican_hat/k5/alpha0p75/Whh.npy
#
# Note: the current builder also creates mexican_hat/k0/Whh.npy when building
# mexican_hat. This is harmless unless you want to add a --offsets option later.

python src/setup/build_hidden_weights.py \
  --output-dir data/hidden_weight_inits \
  --families random mexican_hat \
  --mix-ratios 0.70 \
  --overwrite