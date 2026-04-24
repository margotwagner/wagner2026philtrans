# Figure 1 
conditions=(
  "./data/runs/random"
  "./data/runs/identity"
  "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_tuned/k5/alpha0p00/cfg_tanh_sigmoid_fi5_fo5_12092025"
  "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_tuned/k5/alpha0p25/cfg_tanh_sigmoid_fi5_fo5_12092025"
  "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_tuned/k5/alpha0p50/cfg_tanh_sigmoid_fi5_fo5_12092025"
  "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_tuned/k5/alpha0p70/cfg_tanh_sigmoid_fi5_fo5_12092025"
  "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_tuned/k5/alpha0p80/cfg_tanh_sigmoid_fi5_fo5_12092025"
  "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_tuned/k5/alpha0p90/cfg_tanh_sigmoid_fi5_fo5_12092025"
  "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_tuned/k5/alpha1p00/cfg_tanh_sigmoid_fi5_fo5_12092025"
)

# Custom labels for Fig 1 (must match number/order of conditions above)
# NO COMMAS
labels=(
  "Random"
  "Mexican Hat k=0"
  "Mexican Hat k=5 α₀=0.00"
  "Mexican Hat k=5 α₀=0.25"
  "Mexican Hat k=5 α₀=0.50"
  "Mexican Hat k=5 α₀=0.70"
  "Mexican Hat k=5 α₀=0.80"
  "Mexican Hat k=5 α₀=0.90"
  "Mexican Hat k=5 α₀=1.00"
)

# Join arrays with commas into single CLI arguments
IFS=, conds_csv="${conditions[*]}"; unset IFS
IFS=, labels_csv="${labels[*]}";   unset IFS

FIGDIR=./data/figures/fig1
mkdir -p "$FIGDIR"

# Log-y version
python ./src/analyze/make_figures.py --just 1 \
  --conditions "$conds_csv" \
  --fig1_labels "$labels_csv" \
  --figdir "$FIGDIR" \
  --figtag logy_mh_tuned \
  --fig1_logyA --fig1_logyB

# Log-log version
python ./src/analyze/make_figures.py --just 1 \
  --conditions "$conds_csv" \
  --fig1_labels "$labels_csv" \
  --figdir "$FIGDIR" \
  --figtag loglog_mh_tuned \
  --fig1_logyA --fig1_logyB --fig1_logxA --fig1_logxB

# Raw-y version
python ./src/analyze/make_figures.py --just 1 \
  --conditions "$conds_csv" \
  --fig1_labels "$labels_csv" \
  --figdir "$FIGDIR" \
  --figtag raw_mh_tuned
