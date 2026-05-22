# wagner2026philtrans

Code for training, analyzing, and plotting Elman-style recurrent neural network
experiments for sequential prediction. The main workflow builds a synthetic
Gaussian-bump sequence, initializes recurrent hidden weights, trains dense RNNs
under random and Mexican-hat initializations, computes offline metrics, and
generates manuscript figures.

## Repository Layout

```text
.
|-- Main.py                         # Main RNN training entry point
|-- Makefile                        # Reproducible pipeline targets
|-- environment.yml                 # Conda environment specification
|-- requirements.txt                # Pinned Python dependencies
|-- notebooks/
|   `-- STDP_interactive.py         # Interactive STDP exploration
|-- scripts/
|   |-- 01_build_inputs.sh          # Build synthetic input tensors
|   |-- 02_build_hidden_weights.sh  # Build main hidden-weight initializations
|   |-- 03_train_random.sh          # Train random initialization condition
|   |-- 04_train_mexican_hat.sh     # Train Mexican-hat initialization condition
|   |-- 05_analyze.sh               # Evaluate runs and aggregate metrics
|   |-- 06_make_figures.sh          # Generate main manuscript figures
|   |-- 07_make_stdp_figure.sh      # Generate STDP summary figures
|   |-- quick_test.sh               # Short smoke test of the full workflow
|   `-- supplemental/               # Supplemental initialization sweeps
`-- src/
    |-- analyze/                    # Evaluation, offline metrics, aggregation
    |-- figures/                    # Figure-specific plotting scripts
    |-- setup/                      # Input and weight-initialization builders
    |-- stdp/                       # STDP analysis and plotting
    `-- train/                      # Elman RNN model definition
```

Generated inputs, checkpoints, metrics, logs, and figures are written under
`data/`. This directory is ignored by git.

## Setup

The project was developed with Python 3.10 and PyTorch. The easiest way to
recreate the environment is with conda:

```bash
conda env create -f environment.yml
conda activate hcprediction-fast
```

You can also install from the pinned requirements file in an existing Python
environment:

```bash
python -m pip install -r requirements.txt
```

Several scripts can use CUDA-enabled PyTorch if available, but the analysis and
figure scripts are designed to run headlessly on servers and clusters.

## Quick Smoke Test

To verify that the repository is wired up correctly without reproducing the full
manuscript training runs:

```bash
make quick_test
```

This builds inputs and hidden weights, runs one very short training run per main
condition, then runs the analysis and figure scripts. It is intended only as a
sanity check, not as a reproduction of the reported results.

## Main Workflow

Run commands from the repository root.

```bash
make inputs
make weights
make train
make analyze
make figures
```

Or run the full main workflow:

```bash
make all
```

The main workflow uses:

- `data/inputs/Ns100_SeqN100_asym1.pth.tar` as the input/target tensor file
- `data/hidden_weight_inits/random/seed000/Whh.npy` for the random baseline
- `data/hidden_weight_inits/mexican_hat/k5/alpha0p70/Whh.npy` for the Mexican-hat condition
- `data/runs/random/` and `data/runs/mexican_hat/k5/alpha0p70/` for training outputs
- `data/figures/` for generated figures

By default, the training scripts run 3 independent runs for 100,000 epochs. To
override this from the shell:

```bash
EPOCHS=1000 RUNS=1 SEED=42 make train
```

## Supplemental Workflow

Supplemental scripts build and train additional initialization families:
identity, cyclic-shift alpha sweeps, centered Mexican hat, and shifted
Mexican-hat alpha sweeps.

```bash
make supplemental_weights
make supplemental_train
make supplemental_analyze
```

Or run all supplemental steps:

```bash
make supplemental
```

Supplemental outputs are also written under `data/runs/` and `data/hidden_weight_inits/`.

## Outputs

Training creates one directory per run, for example:

```text
data/runs/random/run_00/
data/runs/mexican_hat/k5/alpha0p70/run_00/
```

Each run directory contains model checkpoints and training diagnostics saved by
`Main.py`. Analysis scripts add per-run summaries, offline metrics, spectra, and
condition-level CSV files such as:

```text
*_eval.csv
run_XX/*_train_summary.csv
run_XX/*_offline_metrics.csv
run_XX/*_spectra.npz
run_level.csv
condition_summary.csv
```

Figure scripts write PNGs into `data/figures/`, including main manuscript
figures and STDP figures.

## Useful Commands

Check shell scripts for syntax errors:

```bash
make check
```

Remove generated figures:

```bash
make clean
```

Build only the default input sequence:

```bash
bash scripts/01_build_inputs.sh
```

Train only the random or Mexican-hat main condition:

```bash
bash scripts/03_train_random.sh
bash scripts/04_train_mexican_hat.sh
```

Generate only the STDP figures:

```bash
bash scripts/07_make_stdp_figure.sh
```

## Direct Training Entry Point

`Main.py` can also be called directly for custom experiments. It expects an
input checkpoint containing `X_mini` and `Target_mini`, and can optionally load a
hidden-hidden initialization matrix from `--whh_path`.

Example:

```bash
python Main.py \
  --input data/inputs/Ns100_SeqN100_asym1.pth.tar \
  --whh_path data/hidden_weight_inits/random/seed000/Whh.npy \
  --savename data/runs/custom_random \
  --ae 1 \
  --pred 1 \
  --n 100 \
  --hidden-n 100 \
  --epochs 1000 \
  --num_runs 1 \
  --rnn_act tanh \
  --act_output sigmoid \
  --fixi 5 \
  --fixo 5 \
  --amp off
```

Run `python Main.py --help` for the full list of training options.

## Reproducibility Notes

- Run commands from the repository root so relative paths resolve correctly.
- `data/` is ignored by git and may become large because it stores checkpoints,
  logs, metrics, and figures.
- The main scripts use `SEED`, `RUNS`, and `EPOCHS` environment variables for
  quick overrides.
- Figure generation assumes the relevant training and analysis outputs already
  exist.
- The full training workflow can be computationally expensive. Use
  `make quick_test` first when setting up a new machine or environment.
