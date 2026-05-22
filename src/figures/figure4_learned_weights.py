#!/usr/bin/env python3
"""
Plot recurrent connectivity summaries averaged across trained RNN runs.

This script loads checkpoints from run_* directories under a single condition/config folder, extracts the final recurrent weight matrix for each run, optionally sorts neurons by their peak hidden-state activation time on the stored training sequence, and plots three summary panels:

1. Mean recurrent connectivity matrix
2. Mean ± SD diagonal trace across runs
3. Eigenspectrum of each run and of the run-averaged matrix

Example
-------
python ./src/figures/figure4_learned_recurrent.py \
    ./data/runs/random \
    --savepath ./data/figures/figure4 \
    --device cpu \
    --outlier-real-thr 2.0 \
    --fontsize 14

Optional save example
---------------------
python ./src/figures/figure4_learned_recurrent.py <CONFIG_PATH> \
    --savepath ./data/figures/recurrent_connectivity_summary.png
    
Example
python ./src/figures/figure4_learned_weights.py ./data/runs/random --savepath ./data/figures/figure4/random.png --fontsize 14
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_recurrent_connectivity_summary(
    config_path,
    ckpt_glob="*.pth.tar",
    device="cpu",
    fontsize=12,
    figsize=(16, 4.0),
    cache_perm=True,
    outlier_real_thr=2.0,
    annotate_outliers=False,
    annotate_k=3,
    src_path="../src",
    savepath=None,
    dpi=300,
    show=True,
):
    """
    Plot final recurrent connectivity summaries averaged across run_* directories.

    - Computes neuron permutation from training sequence hidden states
      (teacher-forced) using X_mini if present in the checkpoint.
    - If checkpoint has W_hh_history + snapshot_epochs, uses the last snapshot.
    - Otherwise falls back to extracting final W_hh from state_dict (dense case).

    Parameters
    ----------
    config_path : str or pathlib.Path
        Directory containing run_* subdirectories.
    ckpt_glob : str, optional
        Glob pattern used inside each run_* directory to find checkpoints.
    device : str, optional
        Torch device used for hidden-state extraction.
    fontsize : int, optional
        Base matplotlib font size.
    figsize : tuple[float, float], optional
        Figure size in inches.
    cache_perm : bool, optional
        If True, save/load cached peak-time permutations from run analysis folders.
    outlier_real_thr : float, optional
        Threshold for reporting eigenvalues of mean(W) with Re(lambda) above this value.
    annotate_outliers : bool, optional
        If True, annotate outlier eigenvalues on the eigenspectrum plot.
    annotate_k : int, optional
        Maximum number of outliers to annotate.
    src_path : str or pathlib.Path, optional
        Path appended to sys.path so train.ElmanRNN can be imported.
    savepath : str or pathlib.Path, optional
        If provided, save the figure to this path.
    dpi : int, optional
        DPI used when saving the figure.
    show : bool, optional
        If True, display the figure with plt.show().

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """

    src_path = Path(src_path).expanduser()
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))

    import torch
    import torch.nn as nn
    from train.ElmanRNN import ElmanRNN

    def _set_output_activation(net: nn.Module, act_output: str):
        if act_output == "tanh":
            net.act_output = nn.Tanh()
        elif act_output == "relu":
            net.act_output = nn.ReLU()
        elif act_output == "sigmoid":
            net.act_output = nn.Sigmoid()

    def _unwrap_compiled_state_dict(sd: dict) -> dict:
        if not any(k.startswith("_orig_mod.") for k in sd.keys()):
            return sd
        out = {}
        pref = "_orig_mod."
        for k, v in sd.items():
            out[k[len(pref) :] if k.startswith(pref) else k] = v
        return out

    def _rebuild_model_from_ckpt(ckpt, state_dict):
        args = ckpt.get("args", {}) or {}
        N = int(args.get("n", args.get("N", 100)))
        H = int(args.get("hidden_n", args.get("hidden_size", 100)))
        rnn_act = args.get("rnn_act", "tanh")

        net = ElmanRNN(
            input_dim=N,
            hidden_dim=H,
            output_dim=N,
            rnn_act=("relu" if rnn_act == "relu" else "tanh"),
        )

        act_output = args.get("act_output", args.get("ac_output", ""))
        if act_output:
            _set_output_activation(net, act_output)

        return net, N, H

    def _peak_time_perm(H_TN):
        Z = (H_TN - H_TN.mean(axis=0, keepdims=True)) / (
            H_TN.std(axis=0, keepdims=True) + 1e-8
        )
        peak_t = np.argmax(Z, axis=0)
        return np.argsort(peak_t).astype(int)

    def _apply_perm(W, perm):
        return W if perm is None else W[np.ix_(perm, perm)]

    def _load_last_W_from_history(ckpt):
        weights = ckpt.get("weights", {}) or {}
        hist = weights.get("W_hh_history", None)
        epochs = ckpt.get("snapshot_epochs", None)
        if hist is None or epochs is None:
            return None
        W = np.asarray(hist[-1])
        if W.dtype == np.float16:
            W = W.astype(np.float32)
        return W

    def _extract_dense_W_from_state_dict(sd, H_expected=None):
        """
        Heuristic: pick the (H,H) matrix that most likely corresponds to recurrent weights.
        """
        candidates = []
        for k, v in sd.items():
            try:
                arr = (
                    v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
                )
            except Exception:
                continue
            if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                if H_expected is None or arr.shape[0] == H_expected:
                    score = 0
                    lk = k.lower()
                    # Common patterns.
                    if "whh" in lk or "w_hh" in lk:
                        score += 5
                    if "weight_hh" in lk:
                        score += 5
                    if "hh" in lk:
                        score += 2
                    if "recurrent" in lk:
                        score += 2
                    if "rnn" in lk:
                        score += 1
                    candidates.append((score, k, arr))
        if not candidates:
            return None, None
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        _, best_k, best = candidates[0]
        return best.astype(np.float32, copy=False), best_k

    def _hidden_from_training_sequence(ckpt, ckpt_path):
        # Prefer checkpoint-stored X_mini if available.
        X = ckpt.get("X_mini", None)
        if X is None:
            return None

        sd = _unwrap_compiled_state_dict(ckpt["state_dict"])
        net, N, H = _rebuild_model_from_ckpt(ckpt, sd)
        net.load_state_dict(sd)
        net.to(device)
        net.eval()

        with torch.no_grad():
            # X: [B,T,N]
            h0 = torch.zeros(1, X.shape[0], H, device=device)
            _, h_seq = net(X.to(device), h0)  # [B,T,H]
        return h_seq[0].detach().cpu().numpy()  # [T,H]

    config_path = Path(config_path).expanduser().resolve()
    run_dirs = sorted([d for d in config_path.glob("run_*") if d.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run_* dirs under {config_path}")

    Ws_sorted = []
    used = []

    for rd in run_dirs:
        ckpts = sorted(rd.glob(ckpt_glob))
        if not ckpts:
            continue
        ckpt_path = ckpts[0]
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

        # 1) Get W_last: history if present, else state_dict.
        W_last = _load_last_W_from_history(ckpt)
        sd = _unwrap_compiled_state_dict(ckpt.get("state_dict", {}))

        # Need H for the extractor heuristic.
        args = ckpt.get("args", {}) or {}
        H_expected = int(args.get("hidden_n", args.get("hidden_size", 100)))

        if W_last is None:
            W_last, which_key = _extract_dense_W_from_state_dict(
                sd, H_expected=H_expected
            )
            if W_last is None:
                # Could not find a square matrix; skip.
                continue

        # 2) Permutation from training sequence hidden states.
        perm = None
        perm_path = rd / "analysis" / "perm_peak_train.npy"
        if cache_perm and perm_path.exists():
            try:
                perm = np.load(perm_path).astype(int)
            except Exception:
                perm = None
        if perm is None:
            H_TN = _hidden_from_training_sequence(ckpt, ckpt_path)
            if H_TN is None:
                # No X_mini in checkpoint => cannot do training-ordering; fall back to unsorted.
                perm = None
            else:
                perm = _peak_time_perm(H_TN)
                if cache_perm:
                    perm_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(str(perm_path), perm)

        Ws_sorted.append(_apply_perm(W_last, perm))
        used.append(rd.name)

    if not Ws_sorted:
        raise FileNotFoundError(
            f"No usable runs found under {config_path}.\n"
            f"Likely: checkpoints missing X_mini (for training ordering) and/or "
            f"state_dict lacks a square (H,H) recurrent matrix."
        )

    Ws_sorted = np.stack(Ws_sorted, axis=0)  # [R,H,H]
    Wm = Ws_sorted.mean(axis=0)

    # ---- Plot summary panels ----
    plt.rcParams.update({"font.size": fontsize})
    fig, (axW, axT, axE) = plt.subplots(1, 3, figsize=figsize)

    vmax = float(np.max(np.abs(Wm))) if Wm.size else 1.0
    vmax = 1.0 if (not np.isfinite(vmax) or vmax <= 0) else vmax
    im = axW.imshow(
        Wm,
        cmap="RdBu_r",
        vmin=-1.05 * vmax,
        vmax=+1.05 * vmax,
        interpolation="nearest",
        aspect="equal",
    )
    cb = plt.colorbar(im, ax=axW, fraction=0.046, pad=0.04)
    cb.set_label("weight", rotation=270, labelpad=12)
    axW.set_title(f"Recurrent Connectivity", fontsize=fontsize + 4)
    axW.set_xlabel("Presynaptic")
    axW.set_ylabel("Postsynaptic")

    H = Wm.shape[0]
    offs = np.arange(-(H - 1), H, dtype=int)
    traces = np.stack(
        [[np.trace(Ws_sorted[r], k) for k in offs] for r in range(Ws_sorted.shape[0])],
        axis=0,
    )
    tr_mean = traces.mean(axis=0)
    tr_sd = (
        traces.std(axis=0, ddof=1) if traces.shape[0] > 1 else np.zeros_like(tr_mean)
    )
    axT.plot(offs, tr_mean, lw=2, label="mean")
    if traces.shape[0] > 1:
        axT.fill_between(offs, tr_mean - tr_sd, tr_mean + tr_sd, alpha=0.2, label="std")
    axT.axhline(0, ls="--", lw=1, alpha=0.6, color="k")
    axT.axvline(0, ls=":", lw=1, alpha=0.6, color="k")
    axT.set_title("Trace (mean±sd)", fontsize=fontsize + 4)
    axT.set_xlabel("Diagonal offset")
    axT.set_ylabel("Weight avg.")
    axT.legend(frameon=False, fontsize=max(8, fontsize - 2))

    eig_mean = np.linalg.eigvals(Wm.astype(np.float64, copy=False))
    eig_runs = np.concatenate(
        [
            np.linalg.eigvals(Ws_sorted[r].astype(np.float64, copy=False))
            for r in range(Ws_sorted.shape[0])
        ],
        axis=0,
    )
    # ---- Outlier reporting: Re(lambda) > threshold on mean(W) ----
    thr = float(outlier_real_thr)
    out_idx = np.where(eig_mean.real > thr)[0]

    if out_idx.size > 0:
        # Sort by real part descending.
        out_idx = out_idx[np.argsort(eig_mean.real[out_idx])[::-1]]
        print(f"[Recurrent summary] mean(W): {out_idx.size} eigenvalue(s) with Re(λ) > {thr}:")
        for j in out_idx:
            lam = eig_mean[j]
            print(
                f"  idx={j:3d}  Re={lam.real:+.6f}  Im={lam.imag:+.6f}  |λ|={abs(lam):.6f}"
            )

        # Print the complex mean of these outliers.
        lam_out = eig_mean[out_idx]
        lam_mean = lam_out.mean()
        print(
            f"  mean(outliers): Re={lam_mean.real:+.6f}  Im={lam_mean.imag:+.6f}  |mean|={abs(lam_mean):.6f}"
        )
    else:
        # Print the maximum real part for debugging.
        j = int(np.argmax(eig_mean.real))
        lam = eig_mean[j]
        print(
            f"[Recurrent summary] mean(W): no Re(λ) > {thr}. "
            f"Max Re(λ) at idx={j}: Re={lam.real:+.6f} Im={lam.imag:+.6f} |λ|={abs(lam):.6f}"
        )

    # ---- Optional plot annotation ----
    if annotate_outliers and out_idx.size > 0:
        for j in out_idx[: int(annotate_k)]:
            lam = eig_mean[j]
            axE.annotate(
                f"Re={lam.real:.2f}\nIm={lam.imag:.2f}",
                xy=(lam.real, lam.imag),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=max(8, fontsize - 3),
                ha="left",
                va="bottom",
            )

    axE.scatter(eig_runs.real, eig_runs.imag, s=6, alpha=0.12, label="runs")
    axE.scatter(eig_mean.real, eig_mean.imag, s=10, alpha=0.9, label="mean(W)")
    th = np.linspace(0, 2 * np.pi, 512)
    axE.plot(np.cos(th), np.sin(th), ls="--", lw=1, alpha=0.4, color="k", label="|λ|=1")
    axE.axhline(0, color="k", lw=0.5)
    axE.axvline(0, color="k", lw=0.5)
    axE.set_aspect("equal", adjustable="box")
    rho = float(np.max(np.abs(eig_mean))) if eig_mean.size else float("nan")
    axE.set_title(f"Eigenspectrum (ρ≈{rho:.3f})", fontsize=fontsize + 4)
    axE.set_xlabel("Re(λ)")
    axE.set_ylabel("Im(λ)")
    axE.legend(frameon=False, fontsize=max(8, fontsize - 2))

    fig.suptitle(
        "Learned recurrent connectivity and eigenspectrum",
        fontsize=fontsize + 6,
    )
    plt.tight_layout()

    if savepath is not None:
        savepath = Path(savepath).expanduser()
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to: {savepath}")

    if show:
        plt.show()

    return fig

def _parse_figsize(value: str) -> Tuple[float, float]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("figsize must look like '16,4' or '16x4'")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("figsize values must be numeric") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot final recurrent connectivity, diagonal trace, and eigenspectrum "
            "averaged across run_* checkpoints in a config directory."
        )
    )
    parser.add_argument(
        "config_path",
        help="Path to a condition/config directory containing run_* subdirectories.",
    )
    parser.add_argument(
        "--ckpt-glob",
        default="*.pth.tar",
        help="Checkpoint glob searched within each run_* directory. Default: *.pth.tar",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for hidden-state extraction. Default: cpu",
    )
    parser.add_argument(
        "--fontsize",
        type=int,
        default=12,
        help="Base font size. Default: 12",
    )
    parser.add_argument(
        "--figsize",
        type=_parse_figsize,
        default=(16, 4.0),
        help="Figure size as WIDTH,HEIGHT or WIDTHxHEIGHT. Default: 16,4",
    )
    parser.add_argument(
        "--no-cache-perm",
        action="store_true",
        help="Disable saving/loading cached neuron peak-time permutations.",
    )
    parser.add_argument(
        "--outlier-real-thr",
        type=float,
        default=2.0,
        help="Report mean(W) eigenvalues with Re(lambda) above this threshold. Default: 2.0",
    )
    parser.add_argument(
        "--annotate-outliers",
        action="store_true",
        help="Annotate outlier eigenvalues on the eigenspectrum plot.",
    )
    parser.add_argument(
        "--annotate-k",
        type=int,
        default=3,
        help="Maximum number of outlier eigenvalues to annotate. Default: 3",
    )
    parser.add_argument(
        "--src-path",
        default="./src",
        help="Path appended to sys.path so train.ElmanRNN can be imported. Default: ./src",
    )
    parser.add_argument(
        "--savepath",
        default=None,
        help="Optional output path for saving the figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figure. Default: 300",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not call plt.show(); useful on headless/HPC runs when using --savepath.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    plot_recurrent_connectivity_summary(
        config_path=args.config_path,
        ckpt_glob=args.ckpt_glob,
        device=args.device,
        fontsize=args.fontsize,
        figsize=args.figsize,
        cache_perm=not args.no_cache_perm,
        outlier_real_thr=args.outlier_real_thr,
        annotate_outliers=args.annotate_outliers,
        annotate_k=args.annotate_k,
        src_path=args.src_path,
        savepath=args.savepath,
        dpi=args.dpi,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
