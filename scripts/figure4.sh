python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/random_pytorch/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/dense --figtag dense_random_pytorch --vmin -2 --vmax 2

python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/identity/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/dense --figtag dense_identity --vmin -2 --vmax 2

python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift/cyclic/alpha0p00/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift --figtag dense_shift_alpha0p00 --vmin -2 --vmax 2

python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift/cyclic/alpha0p75/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift --figtag dense_shift_alpha0p75 --vmin -2 --vmax 2

python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift/cyclic/alpha0p90/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift --figtag dense_shift_alpha0p90 --vmin -2 --vmax 2

python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift/cyclic/alpha1p00/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift --figtag dense_shift_alpha1p00 --vmin -2 --vmax 2

python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_balanced/dog2/k0/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/dog2 --figtag dense_mh_dog2_k0 --vmin -2 --vmax 2

python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_balanced/dog2/k5/alpha1p00/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/dog2 --figtag dense_mh_dog2_k5_alpha1p00 --vmin -2 --vmax 2

#######################################################################
python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_tuned/k5/alpha0p70/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mh_tuned --figtag mh_k5_alpha0p70_1 --vmin -2 --vmax 2

python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_tuned/k0/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mh_tuned --figtag mh_k0 --vmin -2 --vmax 2
#######################################################################
python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/circulant/shift/alpha0p50/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/circulant --figtag circ_shift_alpha0p50 --vmin -2 --vmax 2

python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/circulant/identity/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/circulant --figtag circ_identity --vmin -2 --vmax 2

python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/circulant/mexican_hat/dog_tuned/k0/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/circulant --figtag mh_tuned_k0 --vmin -2 --vmax 2
#######################################################################
python ./src/analyze/make_figures.py --just 4 \
  --conditions "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift/cyclic/alpha0p90/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig4/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift --figtag dense_shift_alpha0p90 --vmin -2 --vmax 2
