# Multiple Conditions
python make_figures.py --just 3 \
  --cond_glob "./runs/ElmanRNN/shift-variants/shift-cyc/frobenius/sym*/shiftcyc_n100_fro_sym*" \
  "./runs/ElmanRNN/shift-variants/shift/frobenius/sym*/shift_n100_fro_sym*" \
  "./runs/ElmanRNN/mh-variants/shifted-cyc/frobenius/sym*/shiftcycmh_n100_fro_sym*" \
  "./runs/ElmanRNN/mh-variants/shifted/frobenius/sym*/shiftmh_n100_fro_sym*" \
  "./runs/ElmanRNN/random-init/random_n100" \
  "./runs/ElmanRNN/shift-variants/identity/frobenius/identity_n100_fro" \
  --figdir ./figs/fig3 --figtag all

# Unconstrained
python make_figures.py --just 3 \
  --cond_glob "./runs/ElmanRNN/identityih/random_baseline" \
  "./runs/ElmanRNN/identityih/shift-variants/identity" \
  "./runs/ElmanRNN/identityih/shift-variants/shift/sym*/shift_sym*" \
  "./runs/ElmanRNN/identityih/shift-variants/cyc-shift/sym*/cycshift_sym*" \
  "./runs/ElmanRNN/identityih/mh-variants/shifted/sym*/shiftmh_sym*" \
  --figdir ./figs/fig3 --figtag idih

# Constrained
python make_figures.py --just 3 \
  --cond_glob "./runs/ElmanRNN/circulant/identity" \
   "./runs/ElmanRNN/circulant/centeredmh" \
  "./runs/ElmanRNN/circulant/shift/sym*/shift_circ_sym*" \
  "./runs/ElmanRNN/circulant/shiftedmh/sym*/shiftedmh_circ_sym*" \
  --figdir ./figs/fig3 --figtag circ

###############################################################
python ./src/analyze/make_figures.py --just 3 \
  --cond_glob "./data/runs/prediction/cfg_relu_sigmoid_fi2_fo2_12092025/dense/random_pytorch/cfg_relu_sigmoid_fi2_fo2_12092025" \
  "./data/runs/prediction/cfg_relu_sigmoid_fi2_fo2_12092025/dense/identity/cfg_relu_sigmoid_fi2_fo2_12092025" \
  "./data/runs/prediction/cfg_relu_sigmoid_fi2_fo2_12092025/dense/shift/cyclic/alpha*/cfg_relu_sigmoid_fi2_fo2_12092025" \
  "./data/runs/prediction/cfg_relu_sigmoid_fi2_fo2_12092025/dense/mexican_hat/dog2/k5/alpha0p00/cfg_relu_sigmoid_fi2_fo2_12092025" \
  --figdir ./data/figures/fig3/cfg_relu_sigmoid_fi2_fo2_12092025/dense

###############################################################
python ./src/analyze/make_figures.py --just 3 \
  --cond_glob "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/random_pytorch/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/identity/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_balanced/dog2/k0/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig3/cfg_tanh_sigmoid_fi5_fo5_12092025/dense --figtag rand_identity_k0

###############################################################
python ./src/analyze/make_figures.py --just 3 \
  --cond_glob "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_balanced/dog2/k5/alpha*/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig3/cfg_tanh_sigmoid_fi5_fo5_12092025/dense \
  --figtag dog2

python ./src/analyze/make_figures.py --just 3 \
  --cond_glob "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_tuned/k5/alpha*/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig3/cfg_tanh_sigmoid_fi5_fo5_12092025/dense \
  --figtag mh_tuned

python ./src/analyze/make_figures.py --just 3 \
  --cond_glob "./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/shift/cyclic/alpha*/cfg_tanh_sigmoid_fi5_fo5_12092025" \
  --figdir ./data/figures/prediction/fig3/cfg_tanh_sigmoid_fi5_fo5_12092025/dense --figtag shift