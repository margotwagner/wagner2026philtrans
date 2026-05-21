#!/usr/bin/env bash
set -euo pipefail

# Build and save the Gaussian-bump input sequence used for training.
# Outputs:
#   data/inputs/Ns100_SeqN100_asym1.pth.tar
#   data/inputs/Ns100_SeqN100_asym1.png

python src/setup/build_inputs.py \
    --output-dir data/inputs \
    --save-outputs