#!/usr/bin/env bash
set -euo pipefail

# Build and save hidden recurrent weight initializations.
# Outputs are written to:
#   data/hidden_weight_inits/

python src/setup/build_hidden_weights.py \
    --output-dir data/hidden_weight_inits \
    --overwrite