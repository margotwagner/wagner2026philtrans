# evaluate.py
# ------------------------------------------------------------
# Minimal evaluation for ElmanRNN checkpoints (CSV only).
# Modes:
#   - open:       teacher-forced evaluation on (saved or held-out) data
#   - replay:     drive the trained model with noise input (no teacher forcing)
#   - prediction: teacher-forced prefix, then continue with constructed inputs
#   - closed:     teacher-forced prefix, then fully autonomous (output -> next input)
#
# Output:
#   Appends one row of metrics per run to a CSV you specify (or a default).
#
# Assumptions:
#   - Checkpoint contains: state_dict, args, X_mini, Target_mini
#   - Tensors are [batch, time, features]
#   - Input and output dims both N (ring task)
#   - RNN_Class.ElmanRNN returns (output_seq, hidden_seq)
#
# Usage examples:
# 1) Evaluate a specfici set of run folders
# python evaluate.py \
#  --base-dir ./runs/ElmanRNN/random-init/random_n100 \
#  --runs 1-3,7,10-12 \
#  --mode all \
#  --csv ./runs_eval/random_n100_eval.csv
#
# 2) Evaluate all runs under a base directory
# python evaluate.py \
#   --base-dir ./runs/ElmanRNN/random-init/random_n100 \
#   --mode closed \
#   --free-steps 200 \
#   --csv ./runs_eval/random_n100_closed.csv
#
# 3) Evaluate all checkpoints matching a glob
# python evaluate.py --glob "./runs/**/run_*/**/*.pth.tar" --mode all
# ------------------------------------------------------------

import argparse
from pathlib import Path
import math
import csv
import numpy as np
import torch
import torch.nn as nn
from typing import List
import re
import numpy as np, pandas as pd, os
import sys

sys.path.append("../../src")

from train.ElmanRNN import ElmanRNN


# ------------------------- Rebuild helpers -------------------------


def _set_output_activation(net: nn.Module, act_output: str):
    """Match training-time output activation if it was overridden."""
    if act_output == "tanh":
        net.act_output = nn.Tanh()
    elif act_output == "relu":
        net.act_output = nn.ReLU()
    elif act_output == "sigmoid":
        net.act_output = nn.Sigmoid()
    # else: keep whatever default (Softmax) the model has


def _unwrap_compiled_state_dict(state_dict: dict) -> dict:
    """
    Handle checkpoints saved from a torch.compile-wrapped model.

    If keys are prefixed with '_orig_mod.', strip that prefix so that they
    match the uncompiled module's parameter names.
    """
    if not any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return state_dict

    cleaned = {}
    prefix = "_orig_mod."
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_k = k[len(prefix) :]
        else:
            new_k = k
        cleaned[new_k] = v
    return cleaned


def _load_ckpt(ckpt_path: Path, map_location="cpu"):
    """
    Load a training checkpoint saved by Main_clean.py.

    For torch>=2.6, explicitly set weights_only=False so we can load
    older checkpoints that contain pickled objects (NumPy, etc.).
    For older torch versions that don't support weights_only, fall
    back to the original signature.
    """
    ckpt_path = str(ckpt_path)
    try:
        # PyTorch ≥ 2.6: override the new default weights_only=True
        return torch.load(ckpt_path, map_location=map_location, weights_only=False)
    except TypeError:
        # PyTorch < 2.6: no weights_only arg
        return torch.load(ckpt_path, map_location=map_location)


def _rebuild_model_from_args(saved_args: dict, device: str, state_dict: dict = None):
    """
    Reconstruct the network architecture from the training-time args
    stored in the checkpoint.

    For circulant models, we also adapt the conv kernel size to match
    the checkpoint (since init_from_row0 may have compressed K < H).
    """
    # Input / hidden sizes
    N = int(saved_args.get("n", saved_args.get("N", 100)))
    H = int(saved_args.get("hidden_n", saved_args.get("hidden_size", 100)))

    # Hidden nonlinearity used at training
    rnn_act = saved_args.get("rnn_act", "tanh")
    enforce_circ = bool(saved_args.get("enforce_circulant", False))

    if enforce_circ:
        # Build circulant model with default kernel size (= H)
        net = ElmanRNN_circulant(
            input_dim=N,
            hidden_dim=H,
            output_dim=N,
            rnn_act=rnn_act,
        )

        # If we have a checkpoint, adapt the kernel size to what was
        # actually used during training (from hh_circ.conv.weight).
        if state_dict is not None and "hh_circ.conv.weight" in state_dict:
            desired_K = int(state_dict["hh_circ.conv.weight"].shape[-1])
            current_K = int(net.hh_circ.conv.weight.shape[-1])
            if desired_K != current_K:
                # Rebuild conv to have the right kernel size; weights
                # will be loaded from the state_dict afterwards.
                net.hh_circ.conv = torch.nn.Conv1d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=desired_K,
                    padding=0,
                    padding_mode="circular",
                    bias=False,
                )
    else:
        # Dense Elman RNN; only "tanh" and "relu" are actually supported
        net = ElmanRNN(
            input_dim=N,
            hidden_dim=H,
            output_dim=N,
            rnn_act=("relu" if rnn_act == "relu" else "tanh"),
        )

    net = net.to(device)

    # Match training-time output activation. Training uses --act_output;
    # some older code had --ac_output, so we support both keys.
    act_output = saved_args.get("act_output", saved_args.get("ac_output", ""))
    if act_output:
        _set_output_activation(net, act_output)

    return net, N, H


def _forward_sequence(
    state_dict,
    X_in: torch.Tensor,
    saved_args: dict,
    device: str,
    return_hidden: bool = False,
):
    """
    Runs the saved model on X_in (teacher-forced).

    Args
    ----
    state_dict: checkpoint['state_dict']
    X_in:       [B, T, N] input sequence
    saved_args: checkpoint['args'] dict from training
    device:     "cpu" or "cuda:0"
    return_hidden: if True, also return hidden sequence

    Returns
    -------
    Y_out (cpu tensor) or (Y_out, h_seq) both on cpu.
    """
    # Handle torch.compile checkpoints: strip "_orig_mod." if present
    state_dict = _unwrap_compiled_state_dict(state_dict)

    # Rebuild net architecture (this will also see the cleaned state_dict
    # so circulant kernel size can be adapted correctly).
    net, N, H = _rebuild_model_from_args(saved_args, device, state_dict)
    net.load_state_dict(state_dict)
    net.eval()

    with torch.no_grad():
        # h0: [1, B, H] for both dense and circulant Elman variants
        h0 = torch.zeros(1, X_in.shape[0], H, device=device)
        Y_out, h_seq = net(X_in.to(device), h0)

    Y_out = Y_out.cpu()
    h_seq = h_seq.cpu()
    return (Y_out, h_seq) if return_hidden else Y_out


# ---------- Training-summary writer (reads keys saved by Main_clean.py) ----------


def _safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _last_history_entry(hist_list):
    if isinstance(hist_list, list) and len(hist_list) > 0:
        return hist_list[-1]
    return {}


def _write_train_summary(ckpt_path: Path):
    """
    Create <stub>_train_summary.csv next to <stub>.pth.tar.
    Pulls training-time diagnostics saved by Main_clean.py and summarizes them in one row.
    Safe for Python 3.6.7 / torch 1.4.0.
    """
    import csv
    import os
    import numpy as np
    import torch

    ckpt_path = Path(ckpt_path)
    # Build <stub>_train_summary.csv next to the checkpoint, where <stub> drops ".pth.tar"
    name = ckpt_path.name
    if name.endswith(".pth.tar"):
        stub = name[:-8]  # drop 8 chars: len(".pth.tar") == 8
    else:
        # fallback for unusual filenames; removes only the last suffix
        stub = ckpt_path.stem
    out_csv = ckpt_path.with_name(stub + "_train_summary.csv")

    # Load checkpoint
    ckpt = _load_ckpt(ckpt_path, map_location="cpu")

    # ---------- Basic run info ----------
    args_dict = ckpt.get("args", {}) or {}
    whh_type = args_dict.get("whh_type", "none")
    whh_norm = args_dict.get("whh_norm", "raw")
    alpha = args_dict.get("alpha", None)
    hidden_n = args_dict.get("hidden_n", None)
    seed = args_dict.get("seed", None)
    pred = args_dict.get("pred", None)
    run_tag = args_dict.get("run_tag", "")

    # ---------- Loss curve ----------
    loss_arr = ckpt.get("loss", None)
    if loss_arr is None:
        loss_arr = np.array([], dtype=np.float64)
    else:
        loss_arr = np.asarray(loss_arr, dtype=np.float64).ravel()

    n_epochs_logged = int(loss_arr.size)
    last_epoch = int(n_epochs_logged - 1) if n_epochs_logged > 0 else None
    last_loss = float(loss_arr[-1]) if n_epochs_logged > 0 else None

    if n_epochs_logged > 0:
        best_epoch = int(np.nanargmin(loss_arr))
        best_loss = float(loss_arr[best_epoch])
        # crude convergence speed: epoch to reach within 1.10 * best
        thresh = 1.10 * best_loss
        # first index where loss <= thresh
        within = np.where(loss_arr <= thresh)[0]
        conv_epoch_110 = int(within[0]) if within.size > 0 else None
        loss_auc = float(np.trapz(loss_arr, dx=1.0))
    else:
        best_epoch = None
        best_loss = None
        conv_epoch_110 = None
        loss_auc = None

    # ---------- Gradient metrics ----------
    gm_hist = ckpt.get("grad_metrics", {}).get("history", []) or []

    # helper to pull last/pre/post global L2/RMS if present
    def _take_grad_stat(which, key):
        if not gm_hist:
            return None
        # use last snapshot in history
        snap = gm_hist[-1]
        if which not in snap:
            return None
        g = snap[which].get("global", {})
        return float(g.get(key)) if key in g else None

    global_L2_pre = _take_grad_stat("pre", "L2")
    global_L2_post = _take_grad_stat("post", "L2")
    global_RMS_pre = _take_grad_stat("pre", "RMS")
    global_RMS_post = _take_grad_stat("post", "RMS")

    # ---------- Hidden metrics (last snapshot) ----------
    hm_hist = ckpt.get("hidden_metrics", {}).get("history", []) or []
    if hm_hist:
        hm_last = hm_hist[-1]
        act = hm_last.get("activation", {}) or {}
        dyn = hm_last.get("dynamics", {}) or {}
        geo = hm_last.get("geometry", {}) or {}
        func = hm_last.get("function", {}) or {}
        h_mean = _safe_float(act.get("mean"))
        h_std = _safe_float(act.get("std"))
        h_sat = _safe_float(act.get("sat_ratio"))
        h_energy = _safe_float(act.get("energy_L2_mean"))
        lag1 = _safe_float(dyn.get("lag1_autocorr"))
        domfreq = _safe_int(dyn.get("dominant_freq_idx"))
        pr = _safe_float(geo.get("participation_ratio"))
        r2_ring = _safe_float(func.get("ring_decode_R2"))
    else:
        h_mean = h_std = h_sat = h_energy = lag1 = pr = r2_ring = None
        domfreq = None

    # ---------- Weight structure (last snapshot) ----------
    ws_hist = ckpt.get("weight_structure", {}).get("history", []) or []
    if ws_hist:
        ws_last = ws_hist[-1]
        fro_W = _safe_float(ws_last.get("fro_W"))
        fro_S = _safe_float(ws_last.get("fro_S"))
        fro_A = _safe_float(ws_last.get("fro_A"))
        sym_ratio = _safe_float(ws_last.get("sym_ratio"))
        asym_ratio = _safe_float(ws_last.get("asym_ratio"))
        mix_A_over_S = _safe_float(ws_last.get("mix_A_over_S"))
        non_normality = _safe_float(ws_last.get("non_normality_commutator"))
        fro_drift = _safe_float(ws_last.get("fro_drift_W_minus_W0"))
        rel_drift = _safe_float(ws_last.get("rel_drift_W_minus_W0"))
    else:
        fro_W = fro_S = fro_A = sym_ratio = asym_ratio = mix_A_over_S = None
        non_normality = fro_drift = rel_drift = None

    # ---------- Error-centric metrics (last snapshot) ----------
    err_hist = ckpt.get("error_metrics", {}).get("history", []) or []
    if err_hist:
        err_last = err_hist[-1]
        angle_R = _safe_float(err_last.get("angle_error_R"))
        angle_cv = _safe_float(err_last.get("angle_error_circ_var"))
        res_lag1 = _safe_float(err_last.get("residual_lag1_autocorr"))
        res_L2 = _safe_float(err_last.get("residual_L2_mean"))
    else:
        angle_R = angle_cv = res_lag1 = res_L2 = None

    # ---------- Compose row ----------
    row = {
        "ckpt": str(ckpt_path),
        "whh_type": whh_type,
        "whh_norm": whh_norm,
        "alpha": alpha,
        "hidden_n": hidden_n,
        "seed": seed,
        "pred": pred,
        "run_tag": run_tag,
        "epochs_logged": n_epochs_logged,
        "last_epoch": last_epoch,
        "last_loss": last_loss,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "conv_epoch_110pct": conv_epoch_110,
        "loss_auc": loss_auc,
        "grad_global_L2_pre": global_L2_pre,
        "grad_global_L2_post": global_L2_post,
        "grad_global_RMS_pre": global_RMS_pre,
        "grad_global_RMS_post": global_RMS_post,
        "h_mean": h_mean,
        "h_std": h_std,
        "h_sat_ratio": h_sat,
        "h_energy_L2_mean": h_energy,
        "h_lag1_autocorr": lag1,
        "h_dom_freq_idx": domfreq,
        "h_participation_ratio": pr,
        "h_ring_decode_R2": r2_ring,
        "fro_W": fro_W,
        "fro_S": fro_S,
        "fro_A": fro_A,
        "sym_ratio": sym_ratio,
        "asym_ratio": asym_ratio,
        "mix_A_over_S": mix_A_over_S,
        "non_normality_commutator": non_normality,
        "fro_drift_W_minus_W0": fro_drift,
        "rel_drift_W_minus_W0": rel_drift,
        "angle_error_R": angle_R,
        "angle_error_circ_var": angle_cv,
        "residual_lag1_autocorr": res_lag1,
        "residual_L2_mean": res_L2,
    }

    # ---------- Write (append or create) ----------
    # Always write a single-row CSV per checkpoint (overwrites any existing one)
    fieldnames = list(row.keys())
    with open(str(out_csv), "w", newline="") as fw:
        w = csv.DictWriter(fw, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(row)
    print("[train_summary] wrote:", out_csv)

    # ---- per-epoch series exports (for Fig 1) ----
    run_dir = ckpt_path.parent
    stub = ckpt_path.stem.replace(".pth", "")

    # 1) Loss curve
    loss = ckpt.get("loss", [])
    if len(loss):
        df = pd.DataFrame({"epoch": np.arange(len(loss), dtype=int), "loss": loss})
        df.to_csv(run_dir / f"{stub}_loss_curve.csv", index=False)
        print("[train_summary] wrote:", run_dir / f"{stub}_loss_curve.csv")

    # 2) Gradient global L2/LMS pre/post per snapshot epoch (if available)
    g_hist = ckpt.get("grad_metrics", {}).get("history", [])
    if g_hist:
        rows = []
        for rec in g_hist:
            e = int(rec["epoch"])
            pre = rec["pre"]["global"]
            post = rec["post"]["global"]
            rows.append(
                {
                    "epoch": e,
                    "grad_L2_pre": float(pre["L2"]),
                    "grad_RMS_pre": float(pre["RMS"]),
                    "grad_L2_post": float(post["L2"]),
                    "grad_RMS_post": float(post["RMS"]),
                }
            )
        pd.DataFrame(rows).sort_values("epoch").to_csv(
            run_dir / f"{stub}_grad_curve.csv", index=False
        )
        print("[train_summary] wrote:", run_dir / f"{stub}_grad_curve.csv")

    # 3) Weight structure trajectory + spectral radius per snapshot
    w_hist = ckpt.get("weight_structure", {}).get("history", [])
    W_snaps = ckpt.get("weights", {}).get("W_hh_history", None)  # [K, H, H] float16
    rows = []
    for i, rec in enumerate(w_hist):
        e = int(rec.get("epoch", i))
        row = {
            "epoch": e,
            "fro_W": rec.get("fro_W", np.nan),
            "fro_S": rec.get("fro_S", np.nan),
            "fro_A": rec.get("fro_A", np.nan),
            "non_normality_commutator": rec.get("non_normality_commutator", np.nan),
        }
        # spectral radius from corresponding snapshot if available
        sp = np.nan
        if W_snaps is not None and i < len(W_snaps):
            W = np.array(W_snaps[i], dtype=np.float32)
            try:
                sp = float(np.max(np.abs(np.linalg.eigvals(W))))
            except Exception:
                pass
        row["spectral_radius"] = sp
        rows.append(row)
    if rows:
        pd.DataFrame(rows).sort_values("epoch").to_csv(
            run_dir / f"{stub}_wstruct_curve.csv", index=False
        )
        print("[train_summary] wrote:", run_dir / f"{stub}_wstruct_curve.csv")


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


def _ckpt_stub(ckpt_path):
    """Return filename minus compound checkpoint suffixes ('.pth.tar', '.pt', '.ckpt')."""
    name = Path(ckpt_path).name
    for suf in (".pth.tar", ".pt", ".ckpt"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return Path(ckpt_path).stem  # fallback


# ------------------------- Angle / residual utils -------------------------
def _cossin_from_dist(P):  # P: [B,T,N] probs (nonneg, sum=1)
    B, T, N = P.shape
    idx = torch.arange(N, device=P.device)
    theta = 2 * math.pi * idx / float(N)
    cos_th, sin_th = torch.cos(theta), torch.sin(theta)
    C = (P * cos_th).sum(dim=2)  # [B,T]
    S = (P * sin_th).sum(dim=2)  # [B,T]
    return C, S


def _r2_twohead(Ytrue, Ypred):  # Y*: [B,T,2]
    # flatten over batch/time
    Yt = Ytrue.reshape(-1, 2)
    Yp = Ypred.reshape(-1, 2)
    mu = Yt.mean(dim=0, keepdim=True)
    ss_res = ((Yt - Yp) ** 2).sum(dim=0)  # per head
    ss_tot = ((Yt - mu) ** 2).sum(dim=0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot  # per head
    return float(r2.mean().item())  # average two heads


def _normalize_distribution(x: torch.Tensor, dim=-1, eps=1e-12):
    """Clamp>=0 and L1-normalize along dim to get a valid probability vector."""
    x = x.clamp_min(0)
    s = x.sum(dim=dim, keepdim=True)
    return x / (s + eps)


def _targets_to_angles(Target: torch.Tensor):
    """Convert ring distributions to angles θ ∈ (-pi, pi]."""
    B, T, N = Target.shape
    idx = torch.arange(N, device=Target.device)
    theta = 2 * math.pi * idx / N
    cos_th, sin_th = torch.cos(theta), torch.sin(theta)
    C = (Target * cos_th).sum(dim=2)
    S = (Target * sin_th).sum(dim=2)
    return torch.atan2(S, C)


def _angles_from_distribution(P: torch.Tensor):
    """Same as above, but first normalize arbitrary outputs to a distribution."""
    Pn = _normalize_distribution(P, dim=2)
    return _targets_to_angles(Pn)


def _wrap_circular(delta: torch.Tensor):
    """Wrap angular differences to (-pi, pi]."""
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def _angle_error_concentration(output: torch.Tensor, Target: torch.Tensor):
    """Circular concentration R of Δθ = θ_pred - θ_true (higher=better)."""
    theta_true = _targets_to_angles(_normalize_distribution(Target, dim=2))
    theta_pred = _angles_from_distribution(output)
    dtheta = _wrap_circular(theta_pred - theta_true)  # [B,T]
    c = torch.cos(dtheta).mean()
    s = torch.sin(dtheta).mean()
    R = torch.sqrt(c * c + s * s).item()
    return {"angle_error_R": float(R), "angle_error_circ_var": float(1.0 - R)}


def _residual_stats(output: torch.Tensor, Target: torch.Tensor):
    """Residual magnitude mean (L2) and lag-1 autocorrelation over time."""
    res = Target - output
    rnorm = _vector_norm_compat(res, dim=2)  # [B,T]
    mean_L2 = float(rnorm.mean().item())
    if rnorm.shape[1] < 3:
        return {"residual_lag1_autocorr": None, "residual_L2_mean": mean_L2}
    x = rnorm - rnorm.mean(dim=1, keepdim=True)
    num = (x[:, :-1] * x[:, 1:]).sum(dim=1)
    den = torch.sqrt((x[:, :-1] ** 2).sum(dim=1) * (x[:, 1:] ** 2).sum(dim=1)) + 1e-12
    lag1 = float((num / den).mean().item())
    return {"residual_lag1_autocorr": lag1, "residual_L2_mean": mean_L2}


def _ring_decode_r2_from_outputs(
    output: torch.Tensor, Target: torch.Tensor, ridge=1e-4
):
    """Ridge decode of [cosθ, sinθ] from outputs; report mean R^2."""
    B, T, N = output.shape
    out = output.reshape(B * T, N)
    X = _normalize_distribution(out, dim=1)
    theta = _targets_to_angles(_normalize_distribution(Target, dim=2)).reshape(B * T)
    Y = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)  # [BT,2]
    XtX = X.T @ X
    I = torch.eye(N, device=X.device, dtype=X.dtype)
    W = W = torch.inverse(XtX + ridge * I) @ (X.T @ Y)
    Yhat = X @ W
    ss_tot = ((Y - Y.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    ss_res = ((Y - Yhat) ** 2).sum(dim=0)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
    return {"ring_decode_R2": float(r2.mean().item())}


def _circular_error_series(output: torch.Tensor, Target: torch.Tensor):
    """|Δθ| time-series in radians (batch handled by later mean)."""
    theta_true = _targets_to_angles(_normalize_distribution(Target, dim=2))
    theta_pred = _angles_from_distribution(output)
    return _wrap_circular(theta_pred - theta_true).abs()  # [B,T]


def _time_to_divergence(output, Target, thresh_rad=math.pi / 6, consec=10):
    """First t where mean(|Δθ|) ≥ threshold for 'consec' consecutive steps."""
    dtheta = _circular_error_series(output, Target).mean(dim=0).numpy()
    if dtheta.shape[0] < consec:
        return None
    mask = (dtheta >= thresh_rad).astype(np.float64)
    win = np.convolve(mask, np.ones(consec, dtype=float), mode="valid")
    idx = np.where(win >= consec)[0]
    return int(idx[0]) if len(idx) > 0 else None


def _phase_drift_per_step(output, Target):
    """Slope (rad/step) of unwrapped signed Δθ over time (mean across batch)."""
    dtheta = _wrap_circular(
        _angles_from_distribution(output)
        - _targets_to_angles(_normalize_distribution(Target, dim=2))
    )
    dmean = dtheta.mean(dim=0).numpy()
    unwrapped = np.unwrap(dmean)
    t = np.arange(len(unwrapped))
    if len(t) < 2:
        return None
    b = np.polyfit(t, unwrapped, 1)[0]
    return float(b)


def _decode_angles_from_outputs(Y):  # Y: [B,T,N] probs or raw
    B, T, N = Y.shape
    # Normalize to prob-simplex for angle decoding
    pred = Y.clamp_min(0)
    pred = pred / (pred.sum(dim=2, keepdim=True) + 1e-12)
    idx = torch.arange(N, device=Y.device)
    theta = 2 * math.pi * idx / N
    cos_th, sin_th = torch.cos(theta), torch.sin(theta)
    C = (pred * cos_th).sum(dim=2)
    S = (pred * sin_th).sum(dim=2)
    theta_pred = torch.atan2(S, C)  # [B,T]
    return theta_pred


def _targets_to_angles_simple(Target):
    # identical to your existing _targets_to_angles; keep a thin wrapper if needed
    return _targets_to_angles(Target)


def _write_angles_csv(path, t, theta_pred, theta_true=None):
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if theta_true is None:
            w.writerow(["t", "theta_pred"])
            for i in range(len(t)):
                w.writerow([int(t[i]), float(theta_pred[i])])
        else:
            w.writerow(["t", "theta_true", "theta_pred"])
            for i in range(len(t)):
                w.writerow([int(t[i]), float(theta_true[i]), float(theta_pred[i])])


# ------------------------- Basic summaries -------------------------


def _metric_mse(Y_true, Y_pred):
    return float(torch.mean((Y_true - Y_pred) ** 2).item())


def _metric_corr_per_feat(Y_true, Y_pred):
    """Mean Pearson r across features, over (batch * time)."""
    yt = Y_true.reshape(-1, Y_true.shape[-1]).numpy()
    yp = Y_pred.reshape(-1, Y_pred.shape[-1]).numpy()
    yt = yt - yt.mean(axis=0, keepdims=True)
    yp = yp - yp.mean(axis=0, keepdims=True)
    num = (yt * yp).sum(axis=0)
    den = np.linalg.norm(yt, axis=0) * np.linalg.norm(yp, axis=0) + 1e-12
    r = num / den
    return float(np.nanmean(r))


# ------------------------- Closed-loop core -------------------------


def _forward_closed_loop(
    state_dict,
    saved_args: dict,
    X_true: torch.Tensor,
    prefix_T: int,
    free_steps: int,
    device: str,
    feedback_norm: str = "prob",
):
    """
    Closed-loop rollout with teacher-forced warm-up.
    Returns:
      Y_all: outputs for warm-up+free  [B, prefix_T+free_steps, N]
      X_all: inputs actually fed       [B, prefix_T+free_steps, N]
      prefix_T_used: int
    """
    # Handle torch.compile checkpoints: strip "_orig_mod." if present
    state_dict = _unwrap_compiled_state_dict(state_dict)

    net, N, H = _rebuild_model_from_args(saved_args, device, state_dict)
    net.load_state_dict(state_dict)
    net.eval()

    B, T_avail, N_in = X_true.shape
    assert N_in == N, "Input/Output dimension mismatch."

    total_T = max(0, prefix_T) + max(0, free_steps)
    X_all = torch.zeros(B, total_T, N, device=device)
    Y_all = torch.zeros(B, total_T, N, device=device)

    with torch.no_grad():
        # warm-up
        prefix_T_used = max(0, min(prefix_T, T_avail))
        if prefix_T_used > 0:
            X_warm = X_true[:, :prefix_T_used, :].to(device)
            h0 = torch.zeros(1, B, H, device=device)
            Y_warm, H_warm_seq = net(X_warm, h0)
            X_all[:, :prefix_T_used, :] = X_warm
            Y_all[:, :prefix_T_used, :] = Y_warm
            h_prev = H_warm_seq[:, -1:, :]  # [B,1,H]
            y_prev = Y_warm[:, -1:, :]  # [B,1,N]
        else:
            h_prev = torch.zeros(B, 1, H, device=device)
            y_prev = torch.zeros(B, 1, N, device=device)

        # free run (step-by-step)
        for k in range(max(0, free_steps)):
            if feedback_norm == "prob":
                x_next = _normalize_distribution(y_prev, dim=2)
            elif feedback_norm == "none":
                x_next = y_prev
            else:
                raise ValueError(f"Unknown feedback_norm {feedback_norm}")

            y_next, h_seq = net(x_next, h_prev.transpose(0, 1))  # expects [1,B,H]
            h_prev = h_seq
            y_prev = y_next

            t = prefix_T_used + k
            X_all[:, t : t + 1, :] = x_next
            Y_all[:, t : t + 1, :] = y_next

    return Y_all.cpu(), X_all.cpu(), int(prefix_T_used)


# ------------------------- Evaluators -------------------------


def evaluate_open(ckpt_path, device="cpu", data_path=None):
    """Teacher-forced evaluation: true inputs X drive the model."""
    ckpt = _load_ckpt(ckpt_path, map_location=device)
    args = ckpt["args"]

    if data_path is None:
        X = ckpt["X_mini"].clone()
        Y_true = ckpt["Target_mini"].clone()
    else:
        external = torch.load(str(data_path), map_location=device)
        X, Y_true = external["X_mini"], external["Target_mini"]

    Y_out = _forward_sequence(ckpt["state_dict"], X, args, device)

    out = {
        "mode": "open",
        "ckpt": str(ckpt_path),
        "data": "" if data_path is None else str(data_path),
        "noise_scale": "",
        "prefix_T": "",
        "div_thresh_deg": "",
        "div_consec": "",
        "feedback_norm": "",
        "free_steps": "",
        "mse": _metric_mse(Y_true, Y_out),
        "mean_corr": _metric_corr_per_feat(Y_true, Y_out),
    }
    out.update(_angle_error_concentration(Y_out, Y_true))
    out.update(_residual_stats(Y_out, Y_true))
    out.update(_ring_decode_r2_from_outputs(Y_out, Y_true))
    out["time_to_divergence"] = ""
    out["phase_drift_per_step"] = ""
    out["mse_free"] = out["mean_corr_free"] = ""
    out["angle_error_R_free"] = out["angle_error_circ_var_free"] = ""
    out["residual_lag1_autocorr_free"] = out["residual_L2_mean_free"] = ""
    out["ring_decode_R2_free"] = ""
    out["replay_r2"] = ""
    return out


def evaluate_replay(ckpt_path, device="cpu", noise_scale=0.01, save_traces=True):
    ckpt = _load_ckpt(ckpt_path, map_location=device)
    args = ckpt["args"]
    X = ckpt["X_mini"].clone()
    Y_true = ckpt["Target_mini"].clone()

    X_noise = torch.normal(mean=0.0, std=noise_scale, size=X.shape)
    Y_out, h_seq = _forward_sequence(
        ckpt["state_dict"], X_noise, args, device, return_hidden=True
    )

    # --- write replay traces (first batch row) ---
    if save_traces:
        stub = _ckpt_stub(ckpt_path)
        run_dir = Path(ckpt_path).parent

        out_hidden = run_dir / f"{stub}_replay_hidden.npy"
        np.save(str(out_hidden), h_seq.detach().cpu().numpy())

        theta_pred = _decode_angles_from_outputs(Y_out)[0].detach().cpu().numpy()
        t = np.arange(theta_pred.shape[0])

        # Replay has no ground-truth angles (noise drive), so we store predicted angle only.
        out_angles = run_dir / f"{stub}_replay_angles.csv"
        _write_angles_csv(str(out_angles), t, theta_pred)

        print(f"[save] {out_hidden}")
        print(f"[save] {out_angles}")

    out = {
        "mode": "replay",
        "ckpt": str(ckpt_path),
        "data": "",
        "noise_scale": float(noise_scale),
        "prefix_T": "",
        "div_thresh_deg": "",
        "div_consec": "",
        "feedback_norm": "",
        "free_steps": "",
        "mse": _metric_mse(Y_true, Y_out),
        "mean_corr": _metric_corr_per_feat(Y_true, Y_out),
    }
    out.update(_angle_error_concentration(Y_out, Y_true))
    out.update(_residual_stats(Y_out, Y_true))
    out.update(
        _ring_decode_r2_from_outputs(Y_out, Y_true)
    )  # gives 'ring_decode_R2' (your replay R²)
    out["time_to_divergence"] = ""
    out["phase_drift_per_step"] = ""
    out["mse_free"] = out["mean_corr_free"] = ""
    out["angle_error_R_free"] = out["angle_error_circ_var_free"] = ""
    out["residual_lag1_autocorr_free"] = out["residual_L2_mean_free"] = ""
    out["ring_decode_R2_free"] = ""
    return out


def evaluate_prediction(
    ckpt_path, device="cpu", noise_scale=0.01, prefix_T=10, save_traces=True
):
    ckpt = _load_ckpt(ckpt_path, map_location=device)
    args = ckpt["args"]
    X_in = ckpt["X_mini"].clone()
    Y_true = ckpt["Target_mini"].clone()

    T = X_in.shape[1]
    prefix_T = max(0, min(prefix_T, T))

    # build pred input: real prefix + noise tail
    X_noise = torch.normal(mean=0.0, std=noise_scale, size=X_in.shape)
    X_pred = X_noise.clone()
    if prefix_T > 0:
        X_pred[:, :prefix_T, :] = X_in[:, :prefix_T, :]

    Y_out, h_seq = _forward_sequence(
        ckpt["state_dict"], X_pred, args, device, return_hidden=True
    )

    # --- write prediction traces (first batch row) ---
    if save_traces:
        stub = _ckpt_stub(ckpt_path)  # <-- uses the helper
        run_dir = Path(ckpt_path).parent  # save next to the ckpt

        out_hidden = run_dir / f"{stub}_prediction_hidden.npy"
        np.save(str(out_hidden), h_seq.detach().cpu().numpy())

        theta_pred = _decode_angles_from_outputs(Y_out)[0].detach().cpu().numpy()
        theta_true = _targets_to_angles_simple(Y_true)[0].detach().cpu().numpy()
        t = np.arange(theta_pred.shape[0])

        out_angles = run_dir / f"{stub}_prediction_angles.csv"
        _write_angles_csv(str(out_angles), t, theta_pred, theta_true)

        print(f"[save] {out_hidden}")
        print(f"[save] {out_angles}")

    # ... keep your existing metrics (mse, drift slope, etc.)
    out = {
        "mode": "prediction",
        "ckpt": str(ckpt_path),
        "data": "",
        "noise_scale": float(noise_scale),
        "prefix_T": int(prefix_T),
        "div_thresh_deg": "",
        "div_consec": "",
        "feedback_norm": "",
        "free_steps": "",
        "mse": _metric_mse(Y_true, Y_out),
        "mean_corr": _metric_corr_per_feat(Y_true, Y_out),
    }
    out.update(_angle_error_concentration(Y_out, Y_true))
    out.update(_residual_stats(Y_out, Y_true))
    out.update(_ring_decode_r2_from_outputs(Y_out, Y_true))
    # ensure you compute/keep 'phase_drift_per_step' somewhere in prediction path
    out["time_to_divergence"] = ""
    # out["phase_drift_per_step"] = <your existing computation>
    out["mse_free"] = out["mean_corr_free"] = ""
    out["angle_error_R_free"] = out["angle_error_circ_var_free"] = ""
    out["residual_lag1_autocorr_free"] = out["residual_L2_mean_free"] = ""
    out["ring_decode_R2_free"] = ""
    return out


def evaluate_closed(
    ckpt_path,
    device="cpu",
    prefix_T=10,
    free_steps=100,
    feedback_norm="prob",
    div_thresh_deg=30.0,
    div_consec=10,
):
    """Closed-loop: teacher-forced warm-up (prefix_T), then fully autonomous."""
    ckpt = _load_ckpt(ckpt_path, map_location=device)
    args = ckpt["args"]
    X_true = ckpt["X_mini"].clone()
    Y_true = ckpt["Target_mini"].clone()

    T_avail = X_true.shape[1]
    prefix_T = max(0, min(prefix_T, T_avail))
    free_steps = max(0, min(free_steps, T_avail - prefix_T))

    Y_all, X_all, prefix_T_used = _forward_closed_loop(
        ckpt["state_dict"],
        args,
        X_true,
        prefix_T,
        free_steps,
        device,
        feedback_norm=feedback_norm,
    )
    Y_true_eval = Y_true[:, : prefix_T_used + free_steps, :]

    out = {
        "mode": "closed",
        "ckpt": str(ckpt_path),
        "data": "",
        "noise_scale": "",
        "prefix_T": int(prefix_T_used),
        "div_thresh_deg": float(div_thresh_deg),
        "div_consec": int(div_consec),
        "feedback_norm": feedback_norm,
        "free_steps": int(free_steps),
        "mse": _metric_mse(Y_true_eval, Y_all),
        "mean_corr": _metric_corr_per_feat(Y_true_eval, Y_all),
    }
    out.update(_angle_error_concentration(Y_all, Y_true_eval))
    out.update(_residual_stats(Y_all, Y_true_eval))
    out.update(_ring_decode_r2_from_outputs(Y_all, Y_true_eval))

    if free_steps > 1:
        sl = slice(prefix_T_used, prefix_T_used + free_steps)
        Y_free = Y_all[:, sl, :]
        T_free = Y_true[:, sl, :]
        thresh_rad = math.radians(div_thresh_deg)
        out["time_to_divergence"] = _time_to_divergence(
            Y_free, T_free, thresh_rad, div_consec
        )
        out["phase_drift_per_step"] = _phase_drift_per_step(Y_free, T_free)

        out["mse_free"] = _metric_mse(T_free, Y_free)
        out["mean_corr_free"] = _metric_corr_per_feat(T_free, Y_free)
        ang_free = _angle_error_concentration(Y_free, T_free)
        res_free = _residual_stats(Y_free, T_free)
        r2_free = _ring_decode_r2_from_outputs(Y_free, T_free)
        out["angle_error_R_free"] = ang_free["angle_error_R"]
        out["angle_error_circ_var_free"] = ang_free["angle_error_circ_var"]
        out["residual_lag1_autocorr_free"] = res_free["residual_lag1_autocorr"]
        out["residual_L2_mean_free"] = res_free["residual_L2_mean"]
        out["ring_decode_R2_free"] = r2_free["ring_decode_R2"]
    else:
        out["time_to_divergence"] = out["phase_drift_per_step"] = ""
        out["mse_free"] = out["mean_corr_free"] = ""
        out["angle_error_R_free"] = out["angle_error_circ_var_free"] = ""
        out["residual_lag1_autocorr_free"] = out["residual_L2_mean_free"] = ""
        out["ring_decode_R2_free"] = ""
    out["replay_r2"] = ""

    return out


# ------------------------- CSV I/O -------------------------


def _write_csv(row_dict: dict, csv_path: Path):
    """Append one row to CSV, creating header if needed."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        # run description
        "mode",
        "ckpt",
        "data",
        "noise_scale",
        "prefix_T",
        "div_thresh_deg",
        "div_consec",
        "feedback_norm",
        "free_steps",
        # core metrics
        "mse",
        "mean_corr",
        "angle_error_R",
        "angle_error_circ_var",
        "residual_lag1_autocorr",
        "residual_L2_mean",
        "ring_decode_R2",
        "replay_r2",
        # dynamics
        "time_to_divergence",
        "phase_drift_per_step",
        # free-only (closed)
        "mse_free",
        "mean_corr_free",
        "angle_error_R_free",
        "angle_error_circ_var_free",
        "residual_lag1_autocorr_free",
        "residual_L2_mean_free",
        "ring_decode_R2_free",
    ]
    row = {k: ("" if row_dict.get(k) is None else row_dict.get(k)) for k in header}
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ------------------------- Sweep helpers -------------------------


def _zero_pad(n: int, pad: int) -> str:
    return f"{n:0{pad}d}"


def _parse_run_spec(spec: str) -> List[int]:
    """
    Parse run ranges like '1-3,7,10-12' -> [1,2,3,7,10,11,12].
    Empty/None returns [].
    """
    if not spec:
        return []
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            if a <= b:
                out.extend(range(a, b + 1))
            else:
                out.extend(range(a, b - 1, -1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _find_ckpt_in_run_dir(run_dir: Path):
    """
    Return a single .pth.tar inside run_dir, or None if not found/unambiguous.
    If multiple exist, prefer the newest modification time.
    """
    cands = list(run_dir.glob("*.pth.tar"))
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    # choose the latest modified if multiple
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _iter_run_ckpts_from_base(base_dir: Path, runs_spec: str, pad: int = 2):
    """
    Yield checkpoints for runs contained in base_dir/run_XX/.
    If runs_spec is empty, iterate all run_* directories, sorted by XX.
    """
    if runs_spec:
        run_ids = _parse_run_spec(runs_spec)
        for r in run_ids:
            run_name = f"run_{_zero_pad(r, pad)}"
            run_dir = base_dir / run_name
            if run_dir.is_dir():
                ckpt = _find_ckpt_in_run_dir(run_dir)
                if ckpt is not None:
                    yield ckpt
                else:
                    print(f"[WARN] No .pth.tar found in {run_dir}")
            else:
                print(f"[WARN] Missing run dir: {run_dir}")
    else:
        # auto-discover all run_* dirs
        pat = re.compile(r"^run_(\d+)$")
        candidates = []
        for d in base_dir.iterdir():
            if d.is_dir():
                m = pat.match(d.name)
                if m:
                    candidates.append((int(m.group(1)), d))
        candidates.sort(key=lambda x: x[0])  # by run number
        if not candidates:
            print(f"[WARN] No run_* subdirs found under {base_dir}")
            return
        for _, run_dir in candidates:
            ckpt = _find_ckpt_in_run_dir(run_dir)
            if ckpt is not None:
                yield ckpt
            else:
                print(f"[WARN] No .pth.tar found in {run_dir}")


def _iter_ckpts(glob_pattern: str):
    """
    Yield checkpoint Paths matching a glob (supports **).
    Example: './runs/**/*.pth.tar'
    """
    base = Path(".")
    # Path.glob supports '**' recursion directly
    for p in base.glob(glob_pattern):
        if p.is_file() and (
            p.suffixes[-2:] == [".pth", ".tar"]
            or p.suffix == ".pt"
            or p.suffix == ".pth"
        ):
            yield p


def _evaluate_one_ckpt(ckpt: Path, args, csv_path: Path):
    """Run the selected mode(s) for a single checkpoint and append to CSV."""
    if args.mode in ("open", "all"):
        r = evaluate_open(ckpt, device=args.device, data_path=args.data)
        _write_csv(r, csv_path)
        print(f"[Open]       {ckpt} -> {csv_path}")

    if args.mode in ("replay", "all"):
        r = evaluate_replay(ckpt, device=args.device, noise_scale=args.noise_scale)
        _write_csv(r, csv_path)
        print(f"[Replay]     {ckpt} -> {csv_path}")

    if args.mode in ("prediction", "all"):
        r = evaluate_prediction(
            ckpt,
            device=args.device,
            noise_scale=args.noise_scale,
            prefix_T=args.prefix_T,
        )
        _write_csv(r, csv_path)
        print(f"[Prediction] {ckpt} -> {csv_path}")

    if args.mode in ("closed", "all"):
        r = evaluate_closed(
            ckpt,
            device=args.device,
            prefix_T=args.prefix_T,
            free_steps=args.free_steps,
            feedback_norm=args.feedback_norm,
            div_thresh_deg=args.div_thresh_deg,
            div_consec=args.div_consec,
        )
        _write_csv(r, csv_path)
        print(f"[Closed]     {ckpt} -> {csv_path}")


# ------------------------- CLI -------------------------


def main():
    p = argparse.ArgumentParser(
        description="Evaluate ElmanRNN: open / replay / prediction / closed (CSV only)"
    )

    # Choose one checkpoint OR a glob
    p.add_argument(
        "--ckpt", type=Path, default=None, help="Path to a single .pth.tar checkpoint"
    )
    p.add_argument(
        "--glob",
        type=str,
        default=None,
        help="Glob for multiple checkpoints, e.g. './runs/**/*.pth.tar'",
    )
    # Evaluate multiple runs under a base directory containing run_XX subfolders
    p.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Parent directory containing run_XX subfolders (e.g., .../random_n100).",
    )
    p.add_argument(
        "--runs",
        type=str,
        default="",
        help="Run ids/ranges, e.g. '1-3,7,10-12'. If omitted, evaluate all run_* found.",
    )
    p.add_argument(
        "--pad",
        type=int,
        default=2,
        help="Zero padding for run dirs (default 2 ⇒ run_01).",
    )

    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--mode",
        type=str,
        choices=["open", "replay", "prediction", "closed", "all"],
        default="all",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Output CSV. Default: single ckpt -> ./runs_eval/<stem>_<mode>.csv; "
        "glob -> ./runs_eval/aggregate_<mode>.csv",
    )

    # open-loop options
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional external .pt/.pth.tar with {'X_mini','Target_mini'} for open-loop",
    )

    # replay / prediction options
    p.add_argument("--noise-scale", type=float, default=0.01)
    p.add_argument("--prefix-T", type=int, default=10)  # used by prediction and closed

    # divergence/drift thresholds (prediction & closed)
    p.add_argument("--div-thresh-deg", type=float, default=30.0)
    p.add_argument("--div-consec", type=int, default=10)

    # closed-loop options
    p.add_argument(
        "--free-steps",
        type=int,
        default=100,
        help="Closed-loop autonomous rollout length after prefix",
    )
    p.add_argument(
        "--feedback-norm",
        type=str,
        default="prob",
        choices=["prob", "none"],
        help="Map output->input during closed loop: prob=normalize, none=raw",
    )

    args = p.parse_args()

    # CASE 0: base-dir + optional runs (explicit multi-run convenience)
    if args.base_dir is not None:
        mode_tag = args.mode
        csv_path = (
            Path("./runs_eval") / f"aggregate_{mode_tag}.csv"
            if args.csv is None
            else args.csv
        )
        matched = list(
            _iter_run_ckpts_from_base(args.base_dir, args.runs, pad=args.pad)
        )
        if not matched:
            print(
                f"[WARN] No checkpoints found using base-dir={args.base_dir} runs='{args.runs}'"
            )
            return
        print(
            f"[INFO] Found {len(matched)} checkpoints in {args.base_dir} (runs='{args.runs or 'ALL'}')."
        )
        for ckpt in matched:
            _evaluate_one_ckpt(ckpt, args, csv_path)
            _write_train_summary(ckpt)
        return

    # Decide CSV path
    if args.glob:
        mode_tag = args.mode
        csv_path = (
            Path("./runs_eval") / f"aggregate_{mode_tag}.csv"
            if args.csv is None
            else args.csv
        )
        matched = list(_iter_ckpts(args.glob))
        if not matched:
            print(f"[WARN] No checkpoints match glob: {args.glob}")
            return
        print(f"[INFO] Found {len(matched)} checkpoints via glob.")
        for ckpt in matched:
            _evaluate_one_ckpt(ckpt, args, csv_path)
            _write_train_summary(ckpt)
    else:
        if args.ckpt is None:
            raise SystemExit("Provide --ckpt or --glob")
        ckpt_stem = args.ckpt.stem
        default_name = f"{ckpt_stem}_{args.mode}.csv"
        csv_path = Path("./runs_eval") / default_name if args.csv is None else args.csv
        _write_train_summary(ckpt)
        _evaluate_one_ckpt(args.ckpt, args, csv_path)


def _vector_norm_compat(x: torch.Tensor, dim=None, keepdim=False, eps=0.0):
    # L2 norm without torch.linalg
    if dim is None:
        return torch.sqrt(torch.clamp((x * x).sum(), min=eps))
    return torch.sqrt(torch.clamp((x * x).sum(dim=dim, keepdim=keepdim), min=eps))


if __name__ == "__main__":
    main()
