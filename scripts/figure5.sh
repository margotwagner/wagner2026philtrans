python ./src/analyze/make_figures.py \
  --just 5 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/random_pytorch/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --fig5_time all \
  --figdir ./data/figures/prediction/fig5/cfg_tanh_sigmoid_fi5_fo5_12092025/dense --fontsize 12 \
  --figtag dense_random_pytorch

python ./src/analyze/make_figures.py \
  --just 5 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/identity/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --fig5_time all \
  --figdir ./data/figures/prediction/fig5/cfg_tanh_sigmoid_fi5_fo5_12092025/dense --fontsize 12 \
  --figtag dense_identity

python ./src/analyze/make_figures.py \
  --just 5 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift/cyclic/alpha0p60/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --fig5_time all \
  --figdir ./data/figures/prediction/fig5/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift --fontsize 12 \
  --figtag dense_shift_alpha0p60

python ./src/analyze/make_figures.py \
  --just 5 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift/cyclic/alpha0p25/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --fig5_time all \
  --figdir ./data/figures/prediction/fig5/cfg_tanh_sigmoid_fi5_fo5_12092025/dense --fontsize 12 \
  --figtag dense_shift_alpha0p25

python ./src/analyze/make_figures.py \
  --just 5 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift/cyclic/alpha1p00/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --fig5_time all \
  --figdir ./data/figures/prediction/fig5/cfg_tanh_sigmoid_fi5_fo5_12092025/dense --fontsize 12 \
  --figtag dense_shift_alpha1p00

#######################################################################
python ./src/analyze/make_figures.py \
  --just 5 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_tuned/k5/alpha0p75/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --fig5_time all \
  --figdir ./data/figures/prediction/fig5/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/dog_tuned --fontsize 12 \
  --figtag mh_k5_alpha0p75

#######################################################################
python ./src/analyze/make_figures.py \
  --just 5 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_balanced/dog2/k5/alpha1p00/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --fig5_time all \
  --figdir ./data/figures/prediction/fig5/cfg_tanh_sigmoid_fi5_fo5_12092025/dense --fontsize 12 \
  --figtag dense_mh_balanced_k5_alpha1p00

python ./src/analyze/make_figures.py \
  --just 5 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_balanced/dog2/k0/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --fig5_time all \
  --figdir ./data/figures/prediction/fig5/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/dog2 --fontsize 12 \
  --figtag dense_mh_balanced_k0

python ./src/analyze/make_figures.py \
  --just 5 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/circulant/mexican_hat/dog_tuned/k0/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --fig5_time all \
  --figdir ./data/figures/prediction/fig5/cfg_tanh_sigmoid_fi5_fo5_12092025/circulant/dog_tuned --fontsize 12 \
  --figtag circ_mh_tuned_k0