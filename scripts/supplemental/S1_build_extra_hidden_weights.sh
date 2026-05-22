#!/usr/bin/env bash
set -euo pipefail

# Build supplemental hidden recurrent weight initializations.
# Outputs include:
#   identity
#   cyclic_shift alpha sweep
#   mexican_hat alpha sweep

python src/setup/build_hidden_weights.py \
  --output-dir data/hidden_weight_inits \
  --families identity cyclic_shift mexican_hat \
  --mix-ratios 0.00,0.25,0.50,0.75,1.00 \
  --overwrite