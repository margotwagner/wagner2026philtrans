# TODO: update
# From project root (PTPredict)
DATA=./data/inputs/Ns100_SeqN100_asym1.pth.tar
INIT_ROOT=./data/hidden_weight_inits/mexican_hat/dog_2ned
RUN_ROOT=./data/runs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_2ned
LOG_ROOT=./data/logs/prediction/cfg_tanh_sigmoid_fi5_fo5_12092025/dense/mexican_hat/dog_2ned

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

EPOCHS=100000      # or whatever you want
RUNS=3             # num_runs per condition
SEED=42            # base seed; Main.py offsets by run_idx
CFG=cfg_tanh_sigmoid_fi5_fo5_12092025         # tag for Phase 0 config
#DOG=dog2

COMMON="--input $DATA \
  --ae 1 --pred 1 \
  --n 100 --hidden-n 100 \
  --epochs $EPOCHS \
  --rnn_act tanh \
  --act_output sigmoid \
  --fixi 5 --fixo 5 \
  --amp off \
  --num_runs $RUNS \
  --seed $SEED \
  --print-freq 1000 \
  --skip_fro_norm_hh"

nohup python Main.py $COMMON \
  --whh_path $INIT_ROOT/k0/Whh.npy \
  --savename $RUN_ROOT/k0/$CFG \
  > $LOG_ROOT/k0.out 2>&1 &

# α = 0.0
nohup python Main.py $COMMON \
  --whh_path $INIT_ROOT/k5/alphasym0p00/Whh.npy \
  --savename $RUN_ROOT/k5/alpha0p00/$CFG \
  > $LOG_ROOT/k5_alpha0p00.out 2>&1 &

# α = 0.25
nohup python Main.py $COMMON \
  --whh_path $INIT_ROOT/k5/alphasym0p25/Whh.npy \
  --savename $RUN_ROOT/k5/alpha0p25/$CFG \
  > $LOG_ROOT/k5_alpha0p25.out 2>&1 &

# α = 0.50
nohup python Main.py $COMMON \
  --whh_path $INIT_ROOT/k5/alphasym0p50/Whh.npy \
  --savename $RUN_ROOT/k5/alpha0p50/$CFG \
  > $LOG_ROOT/k5_alpha0p50.out 2>&1 &

# α = 0.75
nohup python Main.py $COMMON \
  --whh_path $INIT_ROOT/k5/alphasym0p75/Whh.npy \
  --savename $RUN_ROOT/k5/alpha0p75/$CFG \
  > $LOG_ROOT/k5_alpha0p75.out 2>&1 &

# α = 1.0
nohup python Main.py $COMMON \
  --whh_path $INIT_ROOT/k5/alphasym1p00/Whh.npy \
  --savename $RUN_ROOT/k5/alpha1p00/$CFG \
  > $LOG_ROOT/k5_alpha1p00.out 2>&1 &