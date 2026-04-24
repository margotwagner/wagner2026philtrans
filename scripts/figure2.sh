# Multiple Conditions
python make_figures.py --just 2 \
  --cond_glob "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/random_pytorch/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/identity/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_balanced/dog1/k0/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_balanced/dog2/k0/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_balanced/dog3/k0/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig2/cfg_tanh_sigmoid_fi5_fo5_12092025/dense --figtag rand_identity_k0

###############################################################
python ./src/analyze/make_figures.py --just 2 \
  --cond_glob "./data/runs/prediction/cfg_relu_sigmoid_fi6_fo6_12092025/dense/mexican_hat/dog_balanced/dog1/k5/alpha*/cfg_relu_sigmoid_fi6_fo6_12092025" \
  --figdir ./data/figures/prediction/fig2/cfg_relu_sigmoid_fi6_fo6_12092025/dense --figtag dog1

python ./src/analyze/make_figures.py --just 2 \
  --cond_glob "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_tuned/k5/alpha*/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig2/cfg_tanh_sigmoid_fi5_fo5_12092025/dense --figtag mh_tuned

###############################################################
python ./src/analyze/make_figures.py --just 2 \
  --cond_glob "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_balanced/dog3/k5/alpha*/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig2/cfg_tanh_sigmoid_fi5_fo5_12092025/dense --figtag dog3

python ./src/analyze/make_figures.py --just 2 \
  --cond_glob "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift/cyclic/alpha*/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig2/cfg_tanh_sigmoid_fi5_fo5_12092025/dense --figtag shift