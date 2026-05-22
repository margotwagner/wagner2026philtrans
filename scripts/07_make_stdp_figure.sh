#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 07_make_stdp_figure.sh
#
# Generate the STDP symmetry-breaking figure used in the manuscript.
#
# Run from repository root:
#
#   bash scripts/07_make_stdp_figure.sh
#
# Outputs
# -------
#   data/figures/stdp/STDP_eps1p00.png
# ============================================================================

FIG_ROOT="data/figures/stdp"

mkdir -p "$FIG_ROOT"

python src/stdp/figure_stdp.py \
  --epsilon 1.0 \
  --savepath "$FIG_ROOT/STDP_eps1p00.png"

python src/stdp/figure_stdp.py \
  --epsilon 0.0 \
  --savepath "$FIG_ROOT/STDP_eps0p00.png"

python src/stdp/figure_stdp.py \
  --epsilon 0.85 \
  --savepath "$FIG_ROOT/STDP_eps0p85.png"

echo
echo "Finished generating STDP figure."
echo "Figures saved to:"
echo "  $FIG_ROOT"