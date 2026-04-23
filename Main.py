"""Main.py

Author: Margot Wagner
Date: 2026-04-23

Train an Elman-style recurrent neural network on a sequential prediction or
reconstruction task.

This script is the main training entry point used in the project. It loads a
saved input/target tensor pair, constructs a dense Elman RNN, applies any
requested initialization or parameter-freezing scheme, and optimizes the
network with backpropagation through time (BPTT).

Main capabilities
-----------------
- next-step prediction or autoencoding objectives
- dense recurrent connectivity
- optional fixed input/output parameterizations
- optional recurrent-weight initialization from disk
- gradient clipping, mixed precision, and checkpoint resume
- periodic logging of hidden-state, gradient, weight, and error diagnostics
- multi-run execution with separate output folders for reproducibility

Expected input file
-------------------
The file passed via ``--input`` should contain a dictionary with keys
``'X_mini'`` and ``'Target_mini'``.

Outputs
-------
For each run, the script writes a log file, a training-loss figure, and a
checkpoint containing model weights, hidden/output snapshots, diagnostic
summaries, and run metadata.
"""

import argparse
import math
import os
import random
import re
import sys
import time
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append("./src")
matplotlib.use("Agg")

from train.ElmanRNN import ElmanRNN


parser = argparse.ArgumentParser(description="PyTorch Elman BPTT Training")
parser.add_argument(
    "--epochs",
    default=50000,
    type=int,
    metavar="N",
    help="Number of total epochs to run (default: 50k)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.01,
    type=float,
    metavar="LR",
    help="Initial learning rate (SGD only, default: 0.01)",
)
parser.add_argument(
    "--lr_step",
    default="",
    type=str,
    help="Comma-separated epochs at which LR is halved (e.g. 1000,2000)",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=1000,
    type=int,
    metavar="N",
    help="How often to log and store hidden/output snapshots",
)
parser.add_argument(
    "-g", "--gpu", default=1, type=int, help="Enable GPU computing if nonzero"
)
parser.add_argument(
    "-n",
    "--n",
    default=200,
    type=int,
    help="Input/output dimensionality (feature size)",
)
parser.add_argument("--hidden-n", default=200, type=int, help="Hidden dimension size")
parser.add_argument(
    "--savename", default="", type=str, help="Base path (no extension) for outputs"
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="Path to a checkpoint to resume from (loads state_dict only)",
)
parser.add_argument(
    "--ae",
    default=0,
    type=int,
    help="If 1, use Autoencoder objective (Target == Input).",
)
parser.add_argument(
    "--partial",
    default=0,
    type=float,
    help="(DEPRECATED / NONFUNCTIONAL). Sparsity level (0-1). Proportion of RNN params that are frozen by mask.",
)
parser.add_argument(
    "--input",
    default="",
    type=str,
    help="Path to a .pt/.pth.tar containing {'X_mini','Target_mini'}",
)
parser.add_argument(
    "--fixi",
    default=0,
    type=int,
    help="Fix input matrix with modes: 0=off, 1=positive constant, 2=freeze init, 3=abs-fold nonnegative, 4=identity (frozen, zero bias)",
)
parser.add_argument(
    "--fixw",
    default=0,
    type=int,
    help="(DEPRECATED / NONFUNCTIONAL) Fix recurrent weight (rnn.weight_hh_l0) and its bias to constants.",
)
parser.add_argument(
    "--constraini",
    default=0,
    type=int,
    help="(DEPRECATED / NONFUNCTIONAL) If nonzero, clamp input matrix (weight_ih) to be nonnegative after each step.",
)
parser.add_argument(
    "--constraino",
    default=0,
    type=int,
    help="(DEPRECATED / NONFUNCTIONAL) If nonzero, clamp output matrix (linear.weight) to be nonnegative after each step.",
)
parser.add_argument(
    "--fixo",
    default=0,
    type=int,
    help=(
        "Fix the output matrix: "
        "0=off, 1=positive constant (uniform, frozen), "
        "2=freeze init, 3=abs-fold nonnegative (frozen), "
        "4=identity (frozen, zero bias)"
    ),
)
parser.add_argument(
    "--clamp_norm",
    default=0,
    type=float,
    help="If >0, clip total gradient norm to this value each step.",
)
parser.add_argument(
    "--nobias_hh",
    default=0,
    type=int,
    help="whether to remove bias term from hidden-hidden in RNN module",
)
parser.add_argument(
    "--rnn_act",
    type=str,
    default="tanh",
    choices=["linear", "tanh", "relu"],
    help="Hidden activation: linear | tanh | relu",
)
parser.add_argument(
    "--act_output",
    default="",
    type=str,
    help=(
        "Output activation override: linear|tanh|relu|sigmoid "
        "(default is softmax along dim=2 as defined in the model class)"
    ),
)
parser.add_argument(
    "--pred",
    default=0,
    type=int,
    help="If nonzero, use one-step prediction loss (shift X/Target by 1)",
)
parser.add_argument(
    "--noisy_train",
    default=0,
    type=float,
    help="(DEPRECATED / NONFUNCTIONAL) If > 0, add multiplicative Gaussian noise ~N(0, (X*noisy_train)^2) to X and Target each step",
)
parser.add_argument("--seed", type=int, default=42, help="Global RNG seed")
parser.add_argument(
    "--num_runs",
    type=int,
    default=1,
    help="How many sequential runs to perform; each run saves under run_XX/",
)
parser.add_argument(
    "--run_offset",
    type=int,
    default=-1,
    help="If >=0, start run numbering at this index; otherwise auto-pick the next free index.",
)
parser.add_argument(
    "--whh_path",
    type=str,
    default="",
    help=("Path to a hidden-weight .npy of shape (H, H)."),
)
parser.add_argument(
    "--compile",
    action="store_true",
    help="Wrap the model with torch.compile (PyTorch >= 2.0).",
)
parser.add_argument(
    "--amp",
    type=str,
    default="auto",
    choices=["off", "fp16", "bf16", "auto"],
    help="Use mixed precision (auto picks bf16 if available, else fp16).",
)
parser.add_argument(
    "--skip_fro_norm_hh",
    action="store_true",
    help="If set, do NOT Frobenius-normalize the hidden->hidden weights at init.",
)


def main():
    """Entry point: parse args, build data/model/optimizer, train, and save artifacts"""
    global args, f

    args = parser.parse_args()
    if getattr(args, "partial", 0):
        raise ValueError(
            "--partial is deprecated and currently nonfunctional. "
            "Please omit it or set --partial 0."
        )
    if getattr(args, "constraini", 0):
        raise ValueError(
            "--constraini is deprecated and currently nonfunctional. "
            "Please omit it or set --constraini 0."
        )
    if getattr(args, "constraino", 0):
        raise ValueError(
            "--constraino is deprecated and currently nonfunctional. "
            "Please omit it or set --constraino 0."
        )
    if getattr(args, "fixw", 0):
        raise ValueError(
            "--fixw is deprecated and currently nonfunctional. "
            "Please omit it or set --fixw 0."
        )
    if getattr(args, "noisy_train", 0):
        raise ValueError(
            "--noisy_train is deprecated and currently nonfunctional. "
            "Please omit it or set --noisy_train 0."
        )
    # Require whh_path ONLY for fresh runs (not when resuming)
    if not args.resume and not args.whh_path:
        raise ValueError(
            "--whh_path is required unless --resume is provided. "
            "For fresh runs, it must point to a .npy file containing "
            "the hidden weights (HxH for dense)."
        )
    if not args.savename:
        raise ValueError(
            "--savename is required and must be a base path (no extension) "
            "for outputs."
        )

    lr = args.lr
    n_epochs = args.epochs
    N = args.n
    hidden_N = args.hidden_n
    set_seed(args.seed)
    f = None  # global log handle (opened per run later)

    # -----------------
    # Load training data
    # -----------------
    # The saved tensor file is expected to contain:
    #   - X_mini: input sequence
    #   - Target_mini: target sequence
    loaded = torch.load(args.input)
    X_mini = loaded["X_mini"]
    Target_mini = loaded["Target_mini"]

    # Autoencoder mode: use the input sequence itself as the target
    if args.ae:
        log("Autoencoder scenario: Target = Input")
        Target_mini = loaded["X_mini"]

    # One-step prediction: shift inputs and targets for next-step forecasting
    if args.pred:
        X_mini = X_mini[:, :-1, :]
        Target_mini = Target_mini[:, 1:, :]
        log("Predicting one-step ahead")

    # -----------------------
    # Initial hidden state h0
    # -----------------------
    # Shape: (num_layers=1, batch_size, hidden_N)
    h0 = torch.zeros(1, X_mini.shape[0], hidden_N)  # n_layers * BatchN * NHidden

    # -------------------------------------------------
    # Multi-run wrapper: create run_XX dirs and loop
    # -------------------------------------------------
    # Decide where to start numbering runs
    if args.run_offset >= 0:
        start_idx = args.run_offset
    else:
        start_idx = _next_free_run_idx(args.savename)

    for k in range(args.num_runs):
        # 1) Pick a unique run directory (never overwrite)
        run_dir, run_idx = _ensure_unique_run_dir(args.savename, start_idx + k)
        base_stub = os.path.basename(args.savename)
        savename_run = os.path.join(run_dir, base_stub)  # use this for ALL outputs

        # 2) Per-run seeding so each run is reproducible & distinct
        set_seed(args.seed + run_idx)

        # 3) Open a per-run log file for this specific run
        f = open(savename_run + ".log", "w")
        log("Settings:")
        log(str(sys.argv))
        log(f"[run] index={run_idx}, seed={args.seed + run_idx}")

        # ---------------------
        # Define the network
        # ---------------------
        net = ElmanRNN(N, hidden_N, N, rnn_act=args.rnn_act)

        # Optional: initialize recurrent weight from disk
        if args.whh_path:
            try:
                log(f"[whh] loading hidden weight from: {args.whh_path}")
                W = np.load(args.whh_path)
                if W.shape != (hidden_N, hidden_N):
                    raise ValueError(
                        f"Hidden weight shape mismatch: expected ({hidden_N}, {hidden_N}), got {W.shape}"
                    )
                _load_hidden_into_elman(
                    net, args.whh_path, device=net.rnn.weight_hh_l0.device
                )
            except Exception as e:
                log(f"[whh] WARNING: failed to load init from whh_path: {e}")

        # ------------------------------
        # Frobenius-normalize W_hh at init
        # ------------------------------
        if args.skip_fro_norm_hh:
            log("[fro] skipping Frobenius normalization of W_hh (user flag)")
        else:
            _frobenius_normalize_hidden(net, hidden_N)

        # Cache initial recurrent matrix (after any override)
        W0 = _export_dense_Whh(net).detach().cpu().numpy()

        # Optionally override output activation
        if args.act_output == "tanh":
            net.act_output = nn.Tanh()
            log("Change output activation function to tanh")
        elif args.act_output == "relu":
            net.act_output = nn.ReLU()
            log("Change output activation function to relu")
        elif args.act_output == "sigmoid":
            net.act_output = nn.Sigmoid()
            log("Change output activation function to sigmoid")
        elif args.act_output in ("linear", "identity"):
            # no nonlinearity – raw linear readout
            net.act_output = nn.Identity()
            log("Change output activation function to identity (linear)")

        # ------------------------------
        # Optional parameter constraints
        # ------------------------------
        if args.nobias_hh:
            for name, p in net.named_parameters():
                if name == "rnn.bias_hh_l0":
                    p.requires_grad = False
                    p.data.fill_(0)
                    log("Fixing RNN hidden-hidden bias to 0")

        if args.fixi:
            for name, p in net.named_parameters():
                # Handle BOTH architectures:
                # - dense Elman: rnn.weight_ih_l0 / rnn.bias_ih_l0
                if name == "rnn.weight_ih_l0":
                    if args.fixi == 1:
                        # Positive constant matrix (uniform average)
                        with torch.no_grad():
                            p.copy_(torch.ones_like(p) / (p.shape[0] * p.shape[1]))
                        log(f"[fixi] set {name} to positive constant (uniform)")

                    elif args.fixi == 2:
                        # Preserve initialization but freeze it
                        p.requires_grad_(False)
                        log(f"[fixi] froze {name} at its initialization")

                    elif args.fixi == 3:
                        # Make current init nonnegative by folding absolute values
                        with torch.no_grad():
                            p.copy_(p + torch.abs(p))
                        log(f"[fixi] made {name} nonnegative (abs-fold)")

                    elif args.fixi == 4:
                        # Identity (rectangular eye) and freeze
                        H, N_in = p.shape  # (hidden, input)
                        eye = torch.eye(H, N_in, device=p.device, dtype=p.dtype)
                        with torch.no_grad():
                            p.copy_(eye)
                        p.requires_grad_(False)
                        log(f"[fixi] set {name} to identity and froze it")

                        # Also zero & freeze the matching input bias
                        if name == "rnn.weight_ih_l0":
                            if (
                                hasattr(net.rnn, "bias_ih_l0")
                                and net.rnn.bias_ih_l0 is not None
                            ):
                                with torch.no_grad():
                                    net.rnn.bias_ih_l0.zero_()
                                net.rnn.bias_ih_l0.requires_grad_(False)
                                log("[fixi] zeroed & froze rnn.bias_ih_l0")

                    elif args.fixi == 5:
                        # Identity (rectangular eye) without bias frozen
                        H, N_in = p.shape  # (hidden, input)
                        eye = torch.eye(H, N_in, device=p.device, dtype=p.dtype)
                        with torch.no_grad():
                            p.copy_(eye)
                        p.requires_grad_(False)
                        log(f"[fixi] set {name} to identity and froze it")

                    elif args.fixi == 6:
                        # Freeze to random initialization. Zero and freeze bias.
                        p.requires_grad_(False)
                        log(f"[fixi] froze {name} at its initialization")
                        # Also zero & freeze the matching input bias
                        if name == "rnn.weight_ih_l0":
                            if (
                                hasattr(net.rnn, "bias_ih_l0")
                                and net.rnn.bias_ih_l0 is not None
                            ):
                                with torch.no_grad():
                                    net.rnn.bias_ih_l0.zero_()
                                net.rnn.bias_ih_l0.requires_grad_(False)
                                log("[fixi] zeroed & froze rnn.bias_ih_l0")

        if args.fixo:
            for name, p in net.named_parameters():
                if name == "linear.weight":
                    if args.fixo == 1:
                        # Positive constant matrix (uniform average)
                        with torch.no_grad():
                            p.copy_(torch.ones_like(p) / (p.shape[0] * p.shape[1]))
                        log(f"[fixo] set {name} to positive constant and froze it")
                    elif args.fixo == 2:
                        # Preserve initialization but freeze it
                        log(f"[fixo] freezing {name} at initialization")
                    elif args.fixo == 3:
                        # Make current init nonnegative by folding absolute values
                        with torch.no_grad():
                            p.copy_(p + torch.abs(p))
                        log(f"[fixo] made {name} nonnegative and froze it")
                    elif args.fixo == 4:
                        # Identity decoder (rectangular eye) and freeze
                        out_dim, hid_dim = p.shape  # (output_dim, hidden_dim)
                        eye = torch.eye(
                            out_dim, hid_dim, device=p.device, dtype=p.dtype
                        )
                        with torch.no_grad():
                            p.copy_(eye)
                        log(f"[fixo] set {name} to identity and froze it")

                        # Also zero & freeze the matching output bias
                        with torch.no_grad():
                            net.linear.bias.zero_()
                        net.linear.bias.requires_grad_(False)
                        log("[fixo] zeroed & froze linear.bias")

                    elif args.fixo == 5:
                        # Identity decoder (rectangular eye) without bias frozen
                        out_dim, hid_dim = p.shape  # (output_dim, hidden_dim)
                        eye = torch.eye(
                            out_dim, hid_dim, device=p.device, dtype=p.dtype
                        )
                        with torch.no_grad():
                            p.copy_(eye)
                        log(f"[fixo] set {name} to identity and froze it")

                    elif args.fixo == 6:
                        # Random frozen initialization. Zero and freeze bias.
                        log(f"[fixo] set {name} to identity and froze it")
                        # Also zero & freeze the matching output bias
                        with torch.no_grad():
                            net.linear.bias.zero_()
                        net.linear.bias.requires_grad_(False)
                        log("[fixo] zeroed & froze linear.bias")

                    p.requires_grad_(False)

        if args.fixw:
            for name, p in net.named_parameters():
                # Dense Elman recurrent weight / bias
                if name == "rnn.weight_hh_l0":
                    with torch.no_grad():
                        # random in [-1/sqrt(N), 1/sqrt(N)]
                        p.copy_(
                            torch.rand_like(p) * 2.0 * (1.0 / np.sqrt(N))
                            - (1.0 / np.sqrt(N))
                        )
                    p.requires_grad_(False)
                    log("[fixw] fixed rnn.weight_hh_l0 to random matrix and froze it")

                elif name == "rnn.bias_hh_l0":
                    with torch.no_grad():
                        p.fill_(0.0)
                    p.requires_grad_(False)
                    log("[fixw] fixed rnn.bias_hh_l0 to 0 and froze it")

        # ------------------
        # Loss & resume
        # ------------------
        criterion = nn.MSELoss(reduction="mean")

        if args.resume:
            if os.path.isfile(args.resume):
                log("=> loading previous network '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                net.load_state_dict(checkpoint["state_dict"])
                log("=> loaded previous network '{}' ".format(args.resume))
            else:
                log("=> no checkpoint found at '{}'".format(args.resume))

        # -----------------------
        # Move to device
        # -----------------------
        if args.gpu:
            log("Cuda device availability: {}".format(torch.cuda.is_available()))
            criterion = criterion.cuda()
            net = net.cuda()
            X_mini = X_mini.cuda()
            Target_mini = Target_mini.cuda()
            h0 = h0.cuda()

        # ---- precision knobs (AMP + matmul) ----
        if torch.cuda.is_available():
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        amp_enabled = (
            (args.amp != "off") and torch.cuda.is_available() and _has_native_amp()
        )
        if args.amp == "bf16" or (args.amp == "auto" and _has_bf16_cuda()):
            autocast_dtype = torch.bfloat16
        elif args.amp in ("fp16", "auto"):
            autocast_dtype = torch.float16
        else:
            autocast_dtype = None

        # Optional: torch.compile (after moving to device)
        if getattr(args, "compile", False) and hasattr(torch, "compile"):
            try:
                net = torch.compile(net, mode="max-autotune", fullgraph=False)
                log("[compile] torch.compile enabled")
            except Exception as e:
                log(f"[compile] WARNING: fell back (reason: {e})")

        # ---------------
        # Optimizer (SGD)
        # ---------------
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)

        if amp_enabled and autocast_dtype is not None and _has_native_amp():
            try:
                # Preferred in PyTorch 2.4+: device-agnostic AMP
                from torch.amp import GradScaler  # type: ignore[attr-defined]
            except Exception:
                # Fallback for older 2.x
                from torch.cuda.amp import GradScaler
            scaler = GradScaler(enabled=(autocast_dtype is not torch.bfloat16))
        else:

            class _NoopScaler:
                def scale(self, loss):
                    return loss

                def step(self, opt):
                    opt.step()

                def update(self):
                    pass

                def unscale_(self, opt):
                    pass

            scaler = _NoopScaler()
            if args.amp != "off":
                log(
                    "[amp] Native AMP not available on this build; running without AMP."
                )

        # ------------------------------------------------
        # Build parameter masks for partial training (if any)
        # ------------------------------------------------
        if args.partial:
            log("Training sparsity:{}".format(args.partial))
            Mask_W = np.random.uniform(0, 1, (hidden_N, hidden_N))
            Mask_B = np.random.uniform(0, 1, (hidden_N))
            Mask_W = Mask_W > args.partial
            Mask_B = Mask_B > args.partial

            if hasattr(args, "nonoverlap") and args.nonoverlap:
                Mask_W = ~(~(Mask_W) & checkpoint["Mask_W"])
                Mask_B = ~(~(Mask_B) & checkpoint["Mask_B"])

            Mask = []
            for name, p in net.named_parameters():
                if name == "rnn.weight_hh_l0" or name == "hidden_linear.weight":
                    Mask.append(Mask_W)
                    log("Partially train RNN weight")
                elif name == "rnn.bias_hh_l0" or name == "hidden_linear.bias":
                    Mask.append(Mask_B)
                    log("Partially train RNN bias")
                else:
                    Mask.append(np.zeros(p.shape))
        else:
            Mask = []
            for name, p in net.named_parameters():
                Mask.append(np.zeros(p.shape))

        # ----------------------
        # Train and time it
        # ----------------------
        start = time.time()
        (
            net,
            loss_list,
            grad_list,
            hidden_rep,
            output_rep,
            snapshot_epochs,
            grad_metrics_history,
            hidden_metrics_history,
            weight_structure_history,
            W_hh_history,
            error_metrics_history,
            epochs_trained,
        ) = train_partial(
            X_mini,
            Target_mini,
            h0,
            n_epochs,
            net,
            criterion,
            optimizer,
            scaler,
            Mask,
            W0,
        )
        end = time.time()
        deltat = end - start
        log("Total training time: {0:.1f} minutes".format(deltat / 60))
        log(
            f"[train] epochs_requested={int(n_epochs)}, "
            f"epochs_trained={int(epochs_trained)}"
        )

        # -----------------
        # Plot loss curve
        # -----------------
        plt.figure()
        plt.plot(loss_list)
        plt.title("Loss iteration")
        plt.savefig(savename_run + ".png")  # <--- use savename_run here

        # -----------------------------------
        # Save checkpoint + artifacts (PER RUN)
        # -----------------------------------
        save_dict = {
            "state_dict": net.state_dict(),
            "y_hat": np.array(output_rep),
            "hidden": np.array(hidden_rep),
            "X_mini": X_mini.detach().cpu(),
            "Target_mini": Target_mini.detach().cpu(),
            "loss": loss_list,
            "mean_squared_grads": grad_list,
            "grad_metrics": {
                "history": grad_metrics_history,
                "global_L2_pre": [
                    h["pre"]["global"]["L2"] for h in grad_metrics_history
                ],
                "global_L2_post": [
                    h["post"]["global"]["L2"] for h in grad_metrics_history
                ],
                "global_RMS_pre": [
                    h["pre"]["global"]["RMS"] for h in grad_metrics_history
                ],
                "global_RMS_post": [
                    h["post"]["global"]["RMS"] for h in grad_metrics_history
                ],
                "param_names": list(grad_metrics_history[0]["post"]["groups"].keys())
                if len(grad_metrics_history) > 0
                else [],
                "param_shapes": (
                    {
                        k: v["shape"]
                        for k, v in grad_metrics_history[0]["post"]["groups"].items()
                    }
                    if len(grad_metrics_history) > 0
                    else {}
                ),
            },
            "hidden_metrics": {"history": hidden_metrics_history},
            "weight_structure": {"history": weight_structure_history},
            "weights": {
                "W_hh_init": W0.astype(np.float16),
                "W_hh_history": np.array(W_hh_history, dtype=np.float16),
            },
            "error_metrics": {
                "history": error_metrics_history,
                "angle_error_R": [e["angle_error_R"] for e in error_metrics_history],
                "residual_lag1_autocorr": [
                    e["residual_lag1_autocorr"] for e in error_metrics_history
                ],
                "residual_L2_mean": [
                    e["residual_L2_mean"] for e in error_metrics_history
                ],
            },
            "n_epochs": int(n_epochs),
            "training": {
                "epochs_requested": int(n_epochs),
                "epochs_trained": int(epochs_trained),
            },
            "args": {
                **vars(args),
                "run_idx": run_idx,
            },  # store run_idx for traceability
            "env": env_report(),
            "rng": rng_snapshot(),
            "snapshot_epochs": snapshot_epochs,
        }
        if args.partial:
            save_dict["Mask_W"] = Mask_W
            save_dict["Mask_B"] = Mask_B

        torch.save(save_dict, savename_run + ".pth.tar")
        f.close()
        f = None  # close log file


# -------------------------------------------------
# Training loop with optional partial-parameter masks
# -------------------------------------------------
def train_partial(
    X_mini, Target_mini, h0, n_epochs, net, criterion, optimizer, scaler, Mask, W0=None
):
    """
    Train with optional untrainable masks on selected parameters.


    Args:
    X_mini (torch.Tensor): [batch, seq, feat]
    Target_mini (torch.Tensor): [batch, seq, feat]
    h0 (torch.Tensor): initial hidden state [1, batch, hidden]
    n_epochs (int): number of epochs
    net (nn.Module): RNN model
    criterion: loss function (MSE)
    optimizer: torch optimizer
    Mask (list[np.ndarray]): aligned with net.parameters(); True == freeze (zero grad)


    Returns:
    net: trained model
    loss_list (np.ndarray): per-epoch losses
    grad_list (list[list[float]]): mean(grad^2) per parameter group each epoch
    hidden_rep (list[np.ndarray]): periodic hidden snapshots
    output_rep (list[np.ndarray]): periodic output snapshots
    """
    # Count trainable parameters and echo names
    count = 0
    for name, p in net.named_parameters():
        if p.requires_grad:
            print(name)
            count += 1
    print("{} parameters to optimize".format(count))

    loss_list = []
    batch_size, SeqN, N = X_mini.shape
    _, _, hidden_N = h0.shape
    start = time.time()

    hidden_rep = []  # periodic hidden state snapshots
    output_rep = []  # periodic output snapshots
    grad_list = []  # mean(grad^2) per param group each snapshot
    grad_metrics_history = []  # detailed grad metrics per snapshot
    snapshot_epochs = []  # epochs at which snapshots were taken
    hidden_metrics_history = []  # per-snapshot hidden stats/dynamics/geometry/function
    weight_structure_history = []  # per-snapshot W_hh symmetry/asymmetry metrics
    W_hh_history = []  # raw W_hh per snapshot (float16)
    error_metrics_history = []  # per-snapshot error metrics (angle + residuals)
    stop = False  # flag for early stopping

    with tqdm(total=n_epochs, desc="Progress", unit="epoch") as pbar:
        for epoch in range(n_epochs):
            if stop:
                break
            # Optional LR schedule: halve LR at specified epochs
            if args.lr_step:
                lr_step = list(map(int, args.lr_step.split(",")))
                if epoch in lr_step:
                    log("Decrease lr to 50per at epoch {}".format(epoch))
                    for param_group in optimizer.param_groups:
                        param_group["lr"] *= 0.5

            # Optional noise injection (proportional to input magnitude)
            if args.noisy_train:
                # std scales with magnitude of X_mini; avoid any CPU hops
                std = X_mini.abs() * args.noisy_train
                random_part = torch.randn_like(X_mini) * std
                X = X_mini + random_part
                Target = Target_mini + random_part
            else:
                X = X_mini
                Target = Target_mini

            # Forward + backward with optional AMP; grad metrics only on snapshot epochs
            optimizer.zero_grad()

            # Decide AMP dtype on the fly from args (safe here even if repeated each epoch)
            use_amp = (
                (args.amp != "off") and torch.cuda.is_available() and _has_native_amp()
            )
            if use_amp and (
                args.amp == "bf16" or (args.amp == "auto" and _has_bf16_cuda())
            ):
                _amp_dtype = torch.bfloat16
            elif use_amp and (args.amp in ("fp16", "auto")):
                _amp_dtype = torch.float16
            else:
                _amp_dtype = None
                use_amp = False  # no autocast

            if use_amp:
                # -----------------------
                # AMP branch
                # -----------------------
                try:
                    # Preferred (PyTorch 2.4+): device-agnostic AMP
                    from torch import amp as _amp  # device-agnostic AMP

                    ctx = _amp.autocast("cuda", dtype=_amp_dtype)
                except Exception:
                    # Fallback for older 2.x
                    ctx = torch.cuda.amp.autocast(dtype=_amp_dtype)

                with ctx:
                    output, h_seq = net(X, h0)
                    loss = criterion(output, Target)

                # backward (scaled), then unscale for clipping/metrics
                scaler.scale(loss).backward()

                # Snapshot-only PRE metrics & simple grad stats
                if epoch % args.print_freq == 0:
                    grad_metrics_pre = _compute_grad_metrics(net)
                    tmp = []
                    for name, p in net.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            g = p.grad.detach()
                            tmp.append(float((g * g).mean().item()))
                        elif p.requires_grad:
                            tmp.append(0.0)
                    grad_list.append(tmp)

                # Optional global norm clip (on unscaled grads)
                scaler.unscale_(optimizer)
                if args.clamp_norm:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.clamp_norm)

                # Snapshot-only POST metrics
                if epoch % args.print_freq == 0:
                    grad_metrics_post = _compute_grad_metrics(net)
                    grad_metrics_history.append(
                        {
                            "epoch": int(epoch),
                            "pre": grad_metrics_pre,
                            "post": grad_metrics_post,
                            "clipped": bool(
                                args.clamp_norm
                                and (
                                    grad_metrics_post["global"]["L2"]
                                    < grad_metrics_pre["global"]["L2"]
                                )
                            ),
                        }
                    )

                # step
                scaler.step(optimizer)
                scaler.update()

            else:
                # -----------------------
                # Non-AMP branch
                # -----------------------
                output, h_seq = net(X, h0)
                loss = criterion(output, Target)
                loss.backward()

                # Snapshot-only PRE metrics & simple grad stats
                if epoch % args.print_freq == 0:
                    grad_metrics_pre = _compute_grad_metrics(net)
                    tmp = []
                    for name, p in net.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            g = p.grad.detach()
                            tmp.append(float((g * g).mean().item()))
                        elif p.requires_grad:
                            tmp.append(0.0)
                    grad_list.append(tmp)

                # Optional global norm clip
                if args.clamp_norm:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.clamp_norm)

                # Snapshot-only POST metrics
                if epoch % args.print_freq == 0:
                    grad_metrics_post = _compute_grad_metrics(net)
                    grad_metrics_history.append(
                        {
                            "epoch": int(epoch),
                            "pre": grad_metrics_pre,
                            "post": grad_metrics_post,
                            "clipped": bool(
                                args.clamp_norm
                                and (
                                    grad_metrics_post["global"]["L2"]
                                    < grad_metrics_pre["global"]["L2"]
                                )
                            ),
                        }
                    )

                optimizer.step()

            # Optional nonegativity constraints after the step
            if args.constraini:
                for name, p in net.named_parameters():
                    if name == "rnn.weight_ih_l0":
                        p.data.clamp_(0)
            if args.constraino:
                for name, p in net.named_parameters():
                    if name == "linear.weight":
                        p.data.clamp_(0)

            # Early-stopping if plateau is reached
            # Implementation same as Y.C. HCPrediction Main_clean.py
            if epoch > 1000 and len(loss_list) > 5:
                diffs = [
                    loss_list[i + 1] - loss_list[i] for i in range(len(loss_list) - 1)
                ]
                mean_diff = np.mean(np.abs(np.array(diffs[-5:-1])))
                init_loss = loss_list[0]
                if mean_diff < loss.item() * 0.00001 and loss.item() < init_loss * 0.01:
                    log(
                        f"[early-stop] plateau detected at epoch {epoch}, stopping training"
                    )
                    stop = True

            # Bookkeeping
            loss_list = np.append(loss_list, loss.item())

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})
            pbar.update(1)

            # Periodic logging and snapshots of hidden/output states
            if epoch % args.print_freq == 0:
                end = time.time()
                deltat = end - start
                start = time.time()
                # Cast to float32 so NumPy always supports the dtype (handles bf16/fp16 safely)
                hidden_rep.append(h_seq.detach().to(torch.float32).cpu().numpy())
                output_rep.append(output.detach().to(torch.float32).cpu().numpy())
                snapshot_epochs.append(epoch)
                # --- Hidden-state metrics (Activation, Stability, Dynamics, Geometry, Function) ---
                act_stats = _hidden_activation_stats(h_seq, act=args.rnn_act)
                dyn_metrics = _temporal_metrics(h_seq)
                geom_metrics = _geometry_metrics(h_seq, max_components=10)
                # Function: decode ring variable from Target (use same time slice as h_seq)
                tgt_slice = (
                    Target[:, : h_seq.shape[1], :]
                    if "Target" in locals()
                    else Target_mini[:, : h_seq.shape[1], :]
                )
                # --- Error-centric metrics (angular concentration + residual autocorr) ---
                # Use same time window as hidden/output snapshot
                out_slice = output[
                    :, : h_seq.shape[1], :
                ]  # output already matches, this is explicit
                err_ang = _angle_error_metrics(out_slice, tgt_slice)
                err_res = _residual_autocorr(out_slice, tgt_slice)
                error_metrics_history.append(
                    {"epoch": int(epoch), **err_ang, **err_res}
                )
                func_metrics = _decode_ring_linear(h_seq, tgt_slice)

                # Merge into one record
                hidden_metrics_history.append(
                    {
                        "epoch": int(epoch),
                        "activation": act_stats,
                        "dynamics": dyn_metrics,
                        "geometry": geom_metrics,
                        "function": func_metrics,
                    }
                )

                # --- Weight-structure metrics (S/A mix, non-normality, drift) ---
                wstruct = _weight_structure_metrics(net, W0=W0)
                weight_structure_history.append({"epoch": int(epoch), **wstruct})
                # --- Save raw W_hh for offline analysis (compact float16) ---
                W_snap = (
                    _export_dense_Whh(net).detach().cpu().numpy().astype(np.float16)
                )
                W_hh_history.append(W_snap)
                print("Epoch: {}/{}.............".format(epoch, n_epochs), end=" ")
                print("Loss: {:.4f}".format(loss.item()))
                print("Time Elapsed since last display: {0:.1f} seconds".format(deltat))
                print(
                    "Estimated remaining time: {0:.1f} minutes".format(
                        deltat * (n_epochs - epoch) / args.print_freq / 60
                    )
                )

    # --- Summarize how many epochs actually ran ---
    epochs_trained = int(len(loss_list))
    try:
        log(f"[train] epochs_trained={epochs_trained}/{int(n_epochs)}")
    except Exception:
        pass  # logging is best-effort

    return (
        net,
        loss_list,
        grad_list,
        hidden_rep,
        output_rep,
        snapshot_epochs,
        grad_metrics_history,
        hidden_metrics_history,
        weight_structure_history,
        W_hh_history,
        error_metrics_history,
        epochs_trained,
    )


# ------------------
# Utility functions
# ------------------
def set_seed(seed: int = 1337):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Keep cuDNN in fast, non-deterministic mode for training throughput.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def env_report():
    return {
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def rng_snapshot():
    return {
        "py_random": None,  # fill only if you seeded Python's random
        "np_random": np.random.get_state()[1].tolist(),  # the uint32 keys
        "torch_cpu": torch.get_rng_state().tolist(),
        "torch_cuda": [s.tolist() for s in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None,
    }


def _load_hidden_into_elman(
    net: nn.Module, npy_path: str, device: Optional[torch.device] = None
):
    """Copy an (H,H) numpy array into ElmanRNN's recurrent weight."""
    W = np.load(npy_path)
    thW = torch.as_tensor(W, dtype=torch.float32)
    if device is None:
        device = next(net.parameters()).device
    thW = thW.to(device)
    with torch.no_grad():
        # ElmanRNN uses nn.RNN, so the recurrent weight is weight_hh_l0.
        net.rnn.weight_hh_l0.copy_(thW)
        # Leave recurrent biases unchanged.


def _compute_grad_metrics(net: nn.Module):
    """
    Returns a dict with global and per-parameter gradient metrics.
    Assumes .backward() has been called. Reads current .grad tensors.
    """
    total_sq = 0.0
    total_n = 0
    groups = {}  # name -> {'L2':..., 'RMS':..., 'n':..., 'shape':...}

    for name, p in net.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        g = p.grad.detach()
        sq = float(torch.sum(g * g).item())
        n = g.numel()
        # L2 = sqrt(sum g^2); RMS = sqrt(mean g^2)
        L2 = float(torch.sqrt(torch.sum(g * g)).item())
        RMS = float((sq / max(n, 1)) ** 0.5)

        groups[name] = {
            "L2": L2,
            "RMS": RMS,
            "n": int(n),
            "shape": tuple(p.shape),
        }
        total_sq += sq
        total_n += n

    global_L2 = float((total_sq**0.5))
    global_RMS = float(((total_sq / max(total_n, 1)) ** 0.5))

    return {
        "global": {"L2": global_L2, "RMS": global_RMS, "n": int(total_n)},
        "groups": groups,
    }


def _hidden_activation_stats(h_seq, act="tanh", sat_eps=0.95):
    """
    h_seq: torch.Tensor [B, T, H]
    Returns stats over B×T×H:
      - mean, std
      - sat_ratio: |h| >= sat_eps (only for tanh; else None)
      - zero_frac: fraction of exact zeros (useful for ReLU; else None)
      - energy_L2_mean: mean per-step L2 norm averaged over time
    """
    # be robust to mixed precision
    h = h_seq.detach().to(torch.float32)

    # basic stats
    mean = float(h.mean().item())
    std = float(h.std(unbiased=False).item())

    # activation-specific extras
    if act == "tanh":
        sat_ratio = float((h.abs() >= sat_eps).float().mean().item())
        zero_frac = None
    elif act == "relu":
        # dead ReLU proxy
        zero_frac = float((h == 0).float().mean().item())
        sat_ratio = None
    else:  # "linear" (identity) or anything else
        sat_ratio = None
        zero_frac = None

    # energy: average L2 norm per step
    H = h.shape[-1]
    h2 = h.reshape(-1, H)
    # if you already have _vector_norm_compat, keep it; else torch.norm is fine:
    energy = float(torch.norm(h2, dim=-1).mean().item())

    return {
        "mean": mean,
        "std": std,
        "sat_ratio": sat_ratio,
        "zero_frac": zero_frac,
        "energy_L2_mean": energy,
    }


def _temporal_metrics(h_seq):
    """
    Quantify simple temporal metrics from hidden state sequence.
    Returns:
      - lag1_autocorr: average across units of corr(h_t, h_{t+1})
      - dominant_freq_idx: argmax non-DC FFT bin averaged over units (index)
    """
    h = h_seq.detach()  # [B,T,H]
    B, T, H = h.shape
    if T < 3:
        return {"lag1_autocorr": None, "dominant_freq_idx": None}

    # center over time per (B, unit)
    x = h - h.mean(dim=1, keepdim=True)

    # lag-1 autocorr
    num = (x[:, :-1, :] * x[:, 1:, :]).sum(dim=1)
    den = (
        x[:, :-1, :].pow(2).sum(dim=1) * x[:, 1:, :].pow(2).sum(dim=1)
    ).sqrt() + 1e-12
    r1 = _nanmean_compat(num / den, dim=0)  # [H]
    lag1 = float(_nanmean_compat(r1).item())  # scalar

    # dominant frequency via 1D RFFT power (drop DC=0)
    # power: [B, F, H] along time dimension
    power = _rfft_power_time_compat(x, time_dim=1)  # helper handles old/new torch
    mag = power.mean(dim=0).mean(dim=1)  # [F], avg over batch & units
    if mag.numel() > 1:
        dom_idx = int(torch.argmax(mag[1:]).item() + 1)
    else:
        dom_idx = None

    return {"lag1_autocorr": lag1, "dominant_freq_idx": dom_idx}


def _geometry_metrics(h_seq, max_components=50):
    """
    Measure low-dimensional structure (e.g. ring manifold) via PCA.
    Low-d geometry via PCA on [B*T, H].
    Returns: participation_ratio, evr_top (list of first few EV ratios)
    """
    h = h_seq.detach()
    BT, H = h.shape[0] * h.shape[1], h.shape[2]
    X = h.reshape(BT, H) - h.reshape(BT, H).mean(dim=0, keepdim=True)
    C = (X.T @ X) / max(BT - 1, 1)

    try:
        evals = _eigvalsh_sym_compat(C.to(dtype=torch.float64)).clamp(min=0)
        s = evals.sum()
        if s <= 0:
            # Degenerate case: all evals ~ 0
            return {"participation_ratio": None, "evr_top": []}
        pr = float((s**2 / (evals.pow(2).sum() + 1e-12)).item())
        ev_sorted = torch.sort(evals, descending=True).values
        evr = (ev_sorted / (s + 1e-12)).tolist()
        return {
            "participation_ratio": pr,
            "evr_top": evr[: min(max_components, len(evr))],
        }
    except Exception as e:
        # Never crash training because of a diagnostic
        try:
            log(
                f"[geom] WARNING: geometry eigendecomposition failed, skipping metrics: {e}"
            )
        except Exception:
            pass
        return {"participation_ratio": None, "evr_top": []}


def _targets_to_angles(Target_stepwise):
    """
    Convert per-step target distribution over N positions to angle θ ∈ [-π, π).
    Target_stepwise: torch.Tensor [B, T, N] (not one-hot required; any nonneg weights)
    """
    B, T, N = Target_stepwise.shape
    idx = torch.arange(N, device=Target_stepwise.device)
    theta = 2 * math.pi * idx / N  # [N]
    cos_th = torch.cos(theta)
    sin_th = torch.sin(theta)
    # vector mean on circle (per B,T)
    w = Target_stepwise / (Target_stepwise.sum(dim=2, keepdim=True) + 1e-12)
    C = (w * cos_th).sum(dim=2)
    S = (w * sin_th).sum(dim=2)
    ang = torch.atan2(S, C)  # [-pi, pi]
    return ang  # [B, T]


def _decode_ring_linear(h_seq, Target_stepwise):
    """
    Test how well hidden states linearly encode ring position.
    Simple linear decode: h -> [cos θ, sin θ], returns R^2 mean of both heads.

    h_seq: [B, T, H]
    Target_stepwise: [B, T, N]
    """
    # --- work in float32 for numerical stability, regardless of AMP dtype ---
    h = h_seq.detach().to(torch.float32)
    tgt = Target_stepwise.detach().to(torch.float32)

    ang = _targets_to_angles(tgt)  # [B, T]
    y = torch.stack([torch.cos(ang), torch.sin(ang)], dim=2)  # [B, T, 2]

    B, T, H = h.shape
    X = h.reshape(-1, H)  # [B*T, H]
    Y = y.reshape(-1, 2)  # [B*T, 2]

    # center
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)

    # closed-form linear reg: W = (X^T X)^-1 X^T Y  (with small ridge)
    lam = 1e-4  # slightly larger ridge, still tiny
    I = torch.eye(H, device=X.device, dtype=X.dtype)
    XtX = Xc.T @ Xc + lam * I
    XtY = Xc.T @ Yc

    W = _solve_spd_compat(XtX, XtY, ridge=1e-4)  # [H, 2]

    Yhat = Xc @ W + Y.mean(dim=0, keepdim=True)

    # R^2 per head
    ss_res = ((Y - Yhat) ** 2).sum(dim=0)
    ss_tot = ((Y - Y.mean(dim=0, keepdim=True)) ** 2).sum(dim=0) + 1e-12
    r2 = (1.0 - ss_res / ss_tot).mean().item()
    return {"ring_decode_R2": float(r2)}


def _weight_structure_metrics(net, W0=None):
    """
    Summarize symmetry/asymmetry and non-normality of W_hh.
    If W0 is provided (numpy array), also report Frobenius drift and relative drift.
    """
    W = _export_dense_Whh(net).detach().float().cpu().numpy()
    S = 0.5 * (W + W.T)
    A = 0.5 * (W - W.T)

    def fro(x):
        return float(np.linalg.norm(x, "fro"))

    nW, nS, nA = fro(W), fro(S), fro(A)
    sym_ratio = nS / (nW + 1e-12)
    asym_ratio = nA / (nW + 1e-12)
    mix = nA / (nS + 1e-12)
    # non-normality: || W W^T - W^T W ||_F
    comm = W @ W.T - W.T @ W
    nnorm = float(np.linalg.norm(comm, "fro"))

    out = {
        "fro_W": nW,
        "fro_S": nS,
        "fro_A": nA,
        "sym_ratio": sym_ratio,
        "asym_ratio": asym_ratio,
        "mix_A_over_S": mix,
        "non_normality_commutator": nnorm,
    }
    if W0 is not None:
        d = W - W0
        drift = float(np.linalg.norm(d, "fro"))
        out["fro_drift_W_minus_W0"] = drift
        out["rel_drift_W_minus_W0"] = drift / (float(np.linalg.norm(W0, "fro")) + 1e-12)
    return out


def _angle_error_metrics(output, Target_stepwise):
    """
    Angular error concentration for circular targets.
    Uses vector-mean angle of distributions to get θ̂ and θ, then Δθ = wrap(θ̂-θ).
    Returns: resultant length R (higher = more concentrated), circular variance = 1-R.
    """
    # Ensure proper distributions (nonneg, sum=1 over features)
    pred = output.detach()
    pred = pred.clamp_min(0)
    pred = pred / (pred.sum(dim=2, keepdim=True) + 1e-12)

    # True angles from targets (already a distribution)
    theta_true = _targets_to_angles(Target_stepwise)  # [B, T]

    # Predicted angles from model outputs
    B, T, N = pred.shape
    idx = torch.arange(N, device=pred.device)
    theta = 2 * math.pi * idx / N
    cos_th, sin_th = torch.cos(theta), torch.sin(theta)

    C_pred = (pred * cos_th).sum(dim=2)
    S_pred = (pred * sin_th).sum(dim=2)
    theta_pred = torch.atan2(S_pred, C_pred)  # [B, T]

    # Δθ wrapped to [-π, π)
    dtheta = torch.atan2(
        torch.sin(theta_pred - theta_true), torch.cos(theta_pred - theta_true)
    )  # [B, T]

    # Resultant length R of the phase errors
    R = torch.sqrt((torch.cos(dtheta).mean()) ** 2 + (torch.sin(dtheta).mean()) ** 2)
    R = float(R.item())
    return {"angle_error_R": R, "angle_error_circ_var": float(1.0 - R)}


def _residual_autocorr(output, Target_stepwise):
    """
    Lag-1 autocorrelation of residual magnitudes.
    Residual per step is L2 norm of (Target - Output) over features; then compute lag-1 autocorr.
    Also returns mean residual L2 for scale context.
    """
    pred = output.detach()
    res = Target_stepwise - pred  # [B, T, N]
    rnorm = _vector_norm_compat(res, dim=2)  # [B, T]  <-- compat (no torch.linalg)

    if rnorm.shape[1] < 3:
        return {
            "residual_lag1_autocorr": None,
            "residual_L2_mean": float(rnorm.mean().item()),
        }

    # center over time per sequence
    x = rnorm - rnorm.mean(dim=1, keepdim=True)  # [B, T]

    # lag-1 autocorr per sequence, then average across batch
    num = (x[:, :-1] * x[:, 1:]).sum(dim=1)  # [B]
    den = ((x[:, :-1] ** 2).sum(dim=1) * (x[:, 1:] ** 2).sum(dim=1)).sqrt() + 1e-12
    r1 = num / den  # [B]

    # if you have _nanmean_compat already, this is slightly safer:
    # lag1 = float(_nanmean_compat(r1).item())
    lag1 = float(r1.mean().item())

    return {
        "residual_lag1_autocorr": lag1,
        "residual_L2_mean": float(rnorm.mean().item()),
    }


def _vector_norm_compat(x: torch.Tensor, dim=None, keepdim=False, eps=0.0):
    # L2 norm without torch.linalg
    if dim is None:
        return torch.sqrt(torch.clamp((x * x).sum(), min=eps))
    return torch.sqrt(torch.clamp((x * x).sum(dim=dim, keepdim=keepdim), min=eps))


def _nanmean_compat(x: torch.Tensor, dim=None, keepdim=False):
    """Mean that ignores NaN/Inf; supports dim=None or an int/tuple."""
    mask = torch.isfinite(x)
    x_safe = torch.where(mask, x, torch.zeros((), dtype=x.dtype, device=x.device))
    if dim is None:
        summed = x_safe.sum()
        count = mask.sum()
    else:
        summed = x_safe.sum(dim=dim, keepdim=keepdim)
        count = mask.sum(dim=dim, keepdim=keepdim)
    return summed / count.clamp(min=1)


def _rfft_power_time_compat(x: torch.Tensor, time_dim: int = 1) -> torch.Tensor:
    """
    Returns the power spectrum |RFFT|^2 along the time dimension.
    - Accepts x shaped [..., T, ...] (e.g., [B, T, H] with time_dim=1).
    - Returns a real tensor with frequencies on the same axis position 'time_dim'.
    """
    # Move time_dim to last
    x_moved = x.transpose(time_dim, -1)
    dev = x_moved.device
    orig_dtype = x_moved.dtype

    # --- critical fix: do FFT in float32 on CUDA if we're in half/bfloat16 ---
    if x_moved.is_cuda and orig_dtype in (torch.float16, torch.bfloat16):
        x_moved = x_moved.to(torch.float32)

    # Path 1: modern torch.fft (preferred)
    if hasattr(torch, "fft") and hasattr(torch.fft, "rfft"):
        X = torch.fft.rfft(x_moved, dim=-1)  # complex
        power = X.real * X.real + X.imag * X.imag
        power = power.transpose(-1, time_dim)
        return power

    # Path 2: old torch.rfft
    if hasattr(torch, "rfft"):
        Xri = torch.rfft(x_moved, 1, normalized=False, onesided=True)
        power = Xri[..., 0] ** 2 + Xri[..., 1] ** 2
        power = power.transpose(-1, time_dim)
        return power

    # Path 3: NumPy fallback
    x_np = x_moved.detach().cpu().numpy()
    X_np = np.fft.rfft(x_np, axis=-1)
    power_np = (X_np.real**2 + X_np.imag**2).astype(np.float32)
    power = torch.from_numpy(power_np).to(dev)
    power = power.transpose(-1, time_dim)
    return power


def _eigvalsh_sym_compat(C: torch.Tensor) -> torch.Tensor:
    """
    Return eigenvalues of a real symmetric matrix C (float64 recommended).
    Robust to ill-conditioned matrices via jitter + fallbacks.

    - Enforces symmetry numerically
    - Cleans NaN/Inf
    - Tries torch.linalg.eigh with small ridge
    - Falls back to torch.symeig or NumPy if needed
    """
    # Enforce symmetry
    Cs = 0.5 * (C + C.transpose(-1, -2))

    # Clean NaN/Inf just in case
    if not torch.isfinite(Cs).all():
        Cs = torch.nan_to_num(Cs, nan=0.0, posinf=1e6, neginf=-1e6)

    # Use float64 internally for stability
    Cs = Cs.to(dtype=torch.float64)
    H = Cs.shape[-1]
    I = torch.eye(H, dtype=Cs.dtype, device=Cs.device)

    # Try a few jitters with torch.linalg.eigh
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "eigh"):
        for eps in (0.0, 1e-8, 1e-6, 1e-4):
            try:
                evals = torch.linalg.eigh(Cs + eps * I, UPLO="U").eigenvalues
                return evals
            except Exception as e:
                # only log intermittently to avoid spam
                try:
                    log(f"[eig] torch.linalg.eigh failed with eps={eps}: {e}")
                except Exception:
                    pass

    # Fallback: older torch.symeig
    if hasattr(torch, "symeig"):
        try:
            evals, _ = torch.symeig(Cs, eigenvectors=False, upper=True)
            return evals
        except Exception as e:
            try:
                log(f"[eig] torch.symeig also failed: {e}")
            except Exception:
                pass

    # Final fallback: NumPy
    try:
        vals = np.linalg.eigvalsh(Cs.detach().cpu().numpy()).astype(np.float64)
        evals = torch.from_numpy(vals).to(device=C.device, dtype=torch.float64)
        return evals
    except Exception as e:
        # Absolute last resort: don't crash training; return zeros.
        try:
            log(f"[eig] NumPy eigvalsh failed as well, returning zeros: {e}")
        except Exception:
            pass
        return torch.zeros(H, dtype=torch.float64, device=C.device)


def _solve_spd_compat(A: torch.Tensor, B: torch.Tensor, ridge: float = 1e-6):
    """
    Solve (A + ridge*I) X = B for X.

    - Adds small Tikhonov regularization for stability.
    - Prefers torch.linalg.solve, but if the system is singular or ill-conditioned,
      falls back to a least-squares / pinv-based solution instead of raising.
    """
    I = torch.eye(A.size(-1), dtype=A.dtype, device=A.device)
    A_reg = A + ridge * I

    # Preferred path: direct solve
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "solve"):
        try:
            return torch.linalg.solve(A_reg, B)
        except Exception as e:
            # Fallbacks if A_reg is effectively singular
            try:
                # least-squares solution (handles rank-deficient A_reg)
                lstsq = torch.linalg.lstsq(A_reg, B)
                return lstsq.solution
            except Exception:
                # final fallback: pseudo-inverse
                return torch.linalg.pinv(A_reg) @ B

    # Older Torch path: try inverse, then pinverse
    try:
        return torch.inverse(A_reg) @ B
    except Exception:
        return torch.pinverse(A_reg) @ B


def _run_dir_for(base_savename: str, run_idx: int) -> str:
    """
    base_savename: the computed savename (no extension),
                   e.g. runs/.../shiftmh_n100_fro
    returns: runs/.../shiftmh_n100_fro/run_00
    """
    base_dir = os.path.dirname(base_savename)
    base_stub = os.path.basename(base_savename)
    return os.path.join(base_dir, base_stub, f"run_{run_idx:02d}")


def _next_free_run_idx(base_savename: str) -> int:
    """
    Scan base folder for existing run_XX directories and return the next free index.
    Robust to gaps (will return max+1).
    """
    parent = os.path.join(
        os.path.dirname(base_savename), os.path.basename(base_savename)
    )
    if not os.path.isdir(parent):
        return 0
    pat = re.compile(r"^run_(\d+)$")
    seen = []
    for name in os.listdir(parent):
        m = pat.match(name)
        if m:
            try:
                seen.append(int(m.group(1)))
            except ValueError:
                pass
    return (max(seen) + 1) if seen else 0


def _ensure_unique_run_dir(base_savename: str, desired_idx: int) -> (str, int):
    """
    Try desired_idx; if it exists, bump until a free run_XX is found.
    Returns (run_dir, final_idx).
    """
    idx = max(0, int(desired_idx))
    while True:
        run_dir = _run_dir_for(base_savename, idx)
        try:
            os.makedirs(run_dir, exist_ok=False)  # fail if exists
            return run_dir, idx
        except FileExistsError:
            idx += 1


def log(*args, sep=" ", end="\n"):
    """Print to run log if open; otherwise to stdout."""
    global f
    msg = sep.join(str(a) for a in args) + end
    if f is not None:
        f.write(msg)
        f.flush()
    else:
        # fallback to stdout if no run log open yet
        print(msg, end="")


def _export_dense_Whh(net: nn.Module) -> torch.Tensor:
    """
    Return the hidden-to-hidden weight matrix of a dense Elman RNN.
    """
    if hasattr(net, "rnn") and hasattr(net.rnn, "weight_hh_l0"):
        return net.rnn.weight_hh_l0.detach().clone()

    raise AttributeError("No hidden-to-hidden weight found on this model.")


def _has_bf16_cuda() -> bool:
    """
    True if this build/GPU practically supports bfloat16.
    Uses torch.cuda.is_bf16_supported() when present; else checks SM version >= 8.0.
    """
    try:
        fn = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(fn):
            return bool(fn())
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            return major >= 8  # Ampere+
    except Exception:
        pass
    return False


def _has_native_amp() -> bool:
    try:
        # Prefer the new, device-agnostic API (PyTorch 2.4+)
        from torch import amp as _amp  # noqa: F401

        return torch.cuda.is_available()
    except Exception:
        pass
    try:
        # Fallback for slightly older 2.x that still has cuda.amp
        import torch.cuda.amp as _amp  # noqa: F401

        return torch.cuda.is_available()
    except Exception:
        return False


def _target_fro(hidden_N: int) -> float:
    """
    Return the desired Frobenius norm for W_hh.
    Uses THEO_VAR = 1/(3H) => ||W||_F = sqrt(H^2 * THEO_VAR) = sqrt(H/3).
    """
    H = float(hidden_N)
    theo_var = 1.0 / (3.0 * H)
    return float(((H * H) * theo_var) ** 0.5)


@torch.no_grad()
def _frobenius_normalize_hidden(net: nn.Module, hidden_N: int):
    """
    Rescale the recurrent weight matrix so that its Frobenius norm matches
    the target value returned by `_target_fro(hidden_N)`.
    """
    target_fro = _target_fro(hidden_N)

    # Export implied dense W_hh (CPU or current device)
    W = _export_dense_Whh(net).detach()
    cur_fro = float(torch.norm(W, p="fro").item())

    if cur_fro <= 0.0:
        log(
            f"[fro] skipped Frobenius normalization; current Fro={cur_fro:.6g} "
            f"(target={target_fro:.6g})"
        )
        return target_fro, cur_fro

    scale = target_fro / cur_fro

    # Scale the actual parameters that implement W_hh
    for name, p in net.named_parameters():
        if name == "rnn.weight_hh_l0":
            p.mul_(scale)

    log(
        f"[fro] normalized hidden weight: "
        f"fro_before={cur_fro:.6g}, fro_after={target_fro:.6g}, scale={scale:.6g}"
    )
    return target_fro, cur_fro


if __name__ == "__main__":
    main()
