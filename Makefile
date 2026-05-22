# ============================================================================
# Makefile
#
# Main manuscript workflow:
#   make all
#
# Supplemental workflow:
#   make supplemental
# ============================================================================

.PHONY: \
	inputs \
	weights \
	train_random \
	train_mexican_hat \
	train \
	analyze \
	figures \
	all \
	supplemental_weights \
	supplemental_train \
	supplemental_analyze \
	supplemental \
	check \
	clean \
	quick_test

# --------------------------------------------------------------------------
# Main manuscript pipeline
# --------------------------------------------------------------------------

inputs:
	bash scripts/01_build_inputs.sh

weights:
	bash scripts/02_build_hidden_weights.sh

train_random: inputs weights
	bash scripts/03_train_random.sh

train_mexican_hat: inputs weights
	bash scripts/04_train_mexican_hat.sh

train: train_random train_mexican_hat

analyze: train
	bash scripts/05_analyze.sh

figures: analyze
	bash scripts/06_make_figures.sh

all: figures

# --------------------------------------------------------------------------
# Supplemental pipeline
# --------------------------------------------------------------------------

supplemental_weights:
	bash scripts/supplemental/S1_build_extra_hidden_weights.sh

supplemental_train: supplemental_weights
	bash scripts/supplemental/S2_train_identity.sh
	bash scripts/supplemental/S3_train_cyclic_shift.sh
	bash scripts/supplemental/S4_train_extra_mexican_hats.sh

supplemental_analyze: supplemental_train
	bash scripts/supplemental/S5_analyze.sh

supplemental: supplemental_analyze

# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

check:
	bash -n scripts/*.sh
	bash -n scripts/supplemental/*.sh

clean:
	rm -rf data/figures

quick_test:
	bash scripts/quick_test.sh