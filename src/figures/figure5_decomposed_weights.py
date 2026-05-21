#!/usr/bin/env python3
"""
Create decomposed Fig. 4 recurrent-weight summary plots.

This script loads all run_* checkpoint directories under a condition/configuration folder, extracts the final recurrent weight matrix for each run, optionally sorts neurons by peak hidden-state time on the saved training sequence, averages weights across runs, decomposes the mean matrix into symmetric and antisymmetric parts, and plots each component as a separate three-panel figure:

    1. recurrent connectivity heatmap
    2. diagonal trace summary across runs
    3. eigenspectrum of per-run and mean matrices

The script also prints Frobenius-norm decomposition statistics:

    ||S||_F, ||A||_F, ||A||/(||S||+||A||), and ||A||/||S||

Example
-------
Run from the repository root:

    python ./src/figures/figure5_decomposed_weights.py \
        ./data/runs/random \
        --savepath ./data/figures/figure5/random \
        --fontsize 14

Outputs
-------
If --savepath is provided, the script writes:

    <savepath>_symmetric.png
    <savepath>_antisymmetric.png

If --show is provided, the figures are also displayed interactively.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _add_src_to_path(src_dir: Optional[str] = None) -> None:
    """Add the repository src directory to sys.path for train.ElmanRNN imports."""
    if src_dir is not None:
        src_path = Path(src_dir).expanduser().resolve()
    else:
        # Works when this file lives in ./src/figures/ or when run from repo root.
        here = Path(__file__).resolve()
        candidates = [
            here.parents[1] if len(here.parents) > 1 else None,  # ./src from ./src/figures
            Path.cwd() / "src",
            Path.cwd(),
        ]
        src_path = next(
            (p for p in candidates if p is not None and (p / "train" / "ElmanRNN.py").exists()),
            Path.cwd() / "src",
        )

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def plot_last_decomposed(
    config_path: str | Path,
    ckpt_glob: str = "*.pth.tar",
    device: str = "cpu",
    fontsize: int = 12,
    figsize: Tuple[float, float] = (16, 4.0),
    cache_perm: bool = True,
    outlier_real_thr: float = 2.0,
    annotate_outliers: bool = False,
    annotate_k: int = 3,
    show: bool = False,
    eig_xlim: Tuple[float, float] = (-1.0, 2.5),
):
    """
    Plot symmetric and antisymmetric decompositions of final recurrent weights.

    Parameters
    ----------
    config_path:
        Directory containing run_* subdirectories with checkpoints.
    ckpt_glob:
        Glob pattern used inside each run_* directory to find checkpoints.
    device:
        Torch device used only when hidden states must be recomputed for sorting.
    fontsize:
        Base matplotlib font size.
    figsize:
        Size of each three-panel figure.
    cache_perm:
        If True, save/load peak-time permutation at run_*/analysis/perm_peak_train.npy.
    outlier_real_thr:
        Print eigenvalues of the mean matrix with real part greater than this threshold.
    annotate_outliers:
        If True, annotate outlier eigenvalues in the eigenspectrum panel.
    annotate_k:
        Maximum number of outlier eigenvalues to annotate.
    show:
        If True, display figures interactively.
    eig_xlim:
        x-axis limits for eigenspectrum plots.

    Returns
    -------
    fig_sym, fig_anti:
        Matplotlib figures for symmetric and antisymmetric components.
    """
    import torch
    import torch.nn as nn
    from train.ElmanRNN import ElmanRNN

    def _set_output_activation(net: nn.Module, act_output: str) -> None:
        if act_output == "tanh":
            net.act_output = nn.Tanh()
        elif act_output == "relu":
            net.act_output = nn.ReLU()
        elif act_output == "sigmoid":
            net.act_output = nn.Sigmoid()

    def _unwrap_compiled_state_dict(sd: dict) -> dict:
        if not any(k.startswith("_orig_mod.") for k in sd.keys()):
            return sd
        pref = "_orig_mod."
        return {k[len(pref) :] if k.startswith(pref) else k: v for k, v in sd.items()}

    def _rebuild_model_from_ckpt(ckpt: dict, state_dict: dict):
        args = ckpt.get("args", {}) or {}
        n_input = int(args.get("n", args.get("N", 100)))
        hidden_n = int(args.get("hidden_n", args.get("hidden_size", 100)))
        rnn_act = args.get("rnn_act", "tanh")

        net = ElmanRNN(
            input_dim=n_input,
            hidden_dim=hidden_n,
            output_dim=n_input,
            rnn_act=("relu" if rnn_act == "relu" else "tanh"),
        )

        act_output = args.get("act_output", args.get("ac_output", ""))
        if act_output:
            _set_output_activation(net, act_output)

        return net, n_input, hidden_n

    def _peak_time_perm(hidden_time_by_neuron: np.ndarray) -> np.ndarray:
        z = (hidden_time_by_neuron - hidden_time_by_neuron.mean(axis=0, keepdims=True)) / (
            hidden_time_by_neuron.std(axis=0, keepdims=True) + 1e-8
        )
        return np.argsort(np.argmax(z, axis=0)).astype(int)

    def _apply_perm(weight: np.ndarray, perm: Optional[np.ndarray]) -> np.ndarray:
        return weight if perm is None else weight[np.ix_(perm, perm)]

    def _load_last_w_from_history(ckpt: dict) -> Optional[np.ndarray]:
        weights = ckpt.get("weights", {}) or {}
        hist = weights.get("W_hh_history", None)
        epochs = ckpt.get("snapshot_epochs", None)
        if hist is None or epochs is None:
            return None
        weight = np.asarray(hist[-1])
        if weight.dtype == np.float16:
            weight = weight.astype(np.float32)
        return weight

    def _extract_dense_w_from_state_dict(sd: dict, hidden_expected: Optional[int] = None):
        candidates = []
        for key, value in sd.items():
            try:
                arr = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
            except Exception:
                continue
            if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                if hidden_expected is None or arr.shape[0] == hidden_expected:
                    score = 0
                    lower_key = key.lower()
                    if "whh" in lower_key or "w_hh" in lower_key:
                        score += 5
                    if "weight_hh" in lower_key:
                        score += 5
                    if "hh" in lower_key:
                        score += 2
                    if "recurrent" in lower_key:
                        score += 2
                    if "rnn" in lower_key:
                        score += 1
                    candidates.append((score, key, arr))
        if not candidates:
            return None, None
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        _, best_key, best = candidates[0]
        return best.astype(np.float32, copy=False), best_key

    def _hidden_from_training_sequence(ckpt: dict) -> Optional[np.ndarray]:
        x_mini = ckpt.get("X_mini", None)
        if x_mini is None:
            return None

        sd = _unwrap_compiled_state_dict(ckpt["state_dict"])
        net, _, hidden_n = _rebuild_model_from_ckpt(ckpt, sd)
        net.load_state_dict(sd)
        net.to(device)
        net.eval()

        with torch.no_grad():
            h0 = torch.zeros(1, x_mini.shape[0], hidden_n, device=device)
            _, h_seq = net(x_mini.to(device), h0)  # [B,T,H]
        return h_seq[0].detach().cpu().numpy()  # [T,H]

    def _report_outliers_real(eig_mean: np.ndarray, threshold: float, tag: str) -> np.ndarray:
        out_idx = np.where(eig_mean.real > threshold)[0]
        if out_idx.size > 0:
            out_idx = out_idx[np.argsort(eig_mean.real[out_idx])[::-1]]
            print(f"[{tag}] mean: {out_idx.size} eigenvalue(s) with Re(λ) > {threshold}:")
            for j in out_idx:
                lam = eig_mean[j]
                print(f"  idx={j:3d}  Re={lam.real:+.6f}  Im={lam.imag:+.6f}  |λ|={abs(lam):.6f}")
        else:
            j = int(np.argmax(eig_mean.real))
            lam = eig_mean[j]
            print(
                f"[{tag}] mean: no Re(λ) > {threshold}. "
                f"Max Re(λ): Re={lam.real:+.6f} Im={lam.imag:+.6f} |λ|={abs(lam):.6f}"
            )
        return out_idx

    def _plot_three_panels(
        w_mean: np.ndarray,
        w_runs: np.ndarray,
        suptitle: str,
        tag_for_print: str,
        eig_xlim_: Tuple[float, float],
    ):
        plt.rcParams.update({"font.size": fontsize})
        fig, (ax_w, ax_t, ax_e) = plt.subplots(1, 3, figsize=figsize)

        vmax = float(np.max(np.abs(w_mean))) if w_mean.size else 1.0
        vmax = 1.0 if (not np.isfinite(vmax) or vmax <= 0) else vmax
        im = ax_w.imshow(
            w_mean,
            cmap="RdBu_r",
            vmin=-1.05 * vmax,
            vmax=+1.05 * vmax,
            interpolation="nearest",
            aspect="equal",
        )
        cb = plt.colorbar(im, ax=ax_w, fraction=0.046, pad=0.04)
        cb.set_label("weight", rotation=270, labelpad=12)
        ax_w.set_title("Recurrent connectivity", fontsize=fontsize + 4)
        ax_w.set_xlabel("Presynaptic")
        ax_w.set_ylabel("Postsynaptic")

        hidden_n = w_mean.shape[0]
        offsets = np.arange(-(hidden_n - 1), hidden_n, dtype=int)
        traces = np.stack(
            [[np.trace(w_runs[r], k) for k in offsets] for r in range(w_runs.shape[0])],
            axis=0,
        )
        trace_mean = traces.mean(axis=0)
        trace_sd = traces.std(axis=0, ddof=1) if traces.shape[0] > 1 else np.zeros_like(trace_mean)

        ax_t.plot(offsets, trace_mean, lw=2, label="mean")
        if traces.shape[0] > 1:
            ax_t.fill_between(offsets, trace_mean - trace_sd, trace_mean + trace_sd, alpha=0.2, label="std")
        ax_t.axhline(0, ls="--", lw=1, alpha=0.6, color="k")
        ax_t.axvline(0, ls=":", lw=1, alpha=0.6, color="k")
        ax_t.set_title("Trace (mean±sd)", fontsize=fontsize + 4)
        ax_t.set_xlabel("Diagonal offset")
        ax_t.set_ylabel("Weight avg.")
        ax_t.legend(frameon=False, fontsize=max(8, fontsize - 2))

        eig_mean = np.linalg.eigvals(w_mean.astype(np.float64, copy=False))
        eig_runs = np.concatenate(
            [np.linalg.eigvals(w_runs[r].astype(np.float64, copy=False)) for r in range(w_runs.shape[0])],
            axis=0,
        )

        out_idx = _report_outliers_real(eig_mean, float(outlier_real_thr), tag_for_print)

        ax_e.scatter(eig_runs.real, eig_runs.imag, s=6, alpha=0.12, label="runs")
        ax_e.scatter(eig_mean.real, eig_mean.imag, s=10, alpha=0.9, label="mean")

        if annotate_outliers and out_idx.size > 0:
            out_idx = out_idx[np.argsort(eig_mean.real[out_idx])[::-1]]
            for j in out_idx[: int(annotate_k)]:
                lam = eig_mean[j]
                ax_e.annotate(
                    f"Re={lam.real:.2f}\nIm={lam.imag:.2f}",
                    xy=(lam.real, lam.imag),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=max(8, fontsize - 3),
                    ha="left",
                    va="bottom",
                )

        theta = np.linspace(0, 2 * np.pi, 512)
        ax_e.plot(np.cos(theta), np.sin(theta), ls="--", lw=1, alpha=0.4, color="k", label="|λ|=1")
        ax_e.axhline(0, color="k", lw=0.5)
        ax_e.axvline(0, color="k", lw=0.5)
        ax_e.set_aspect("equal", adjustable="box")
        rho = float(np.max(np.abs(eig_mean))) if eig_mean.size else float("nan")
        ax_e.set_title(f"Eigenspectrum (ρ≈{rho:.3f})", fontsize=fontsize + 4)
        ax_e.set_xlabel("Re(λ)")
        ax_e.set_ylabel("Im(λ)")
        ax_e.set_xlim(eig_xlim_)
        ax_e.legend(frameon=False, fontsize=max(8, fontsize - 2))

        fig.suptitle(suptitle, fontsize=fontsize + 6)
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    config_path = Path(config_path).expanduser().resolve()
    run_dirs = sorted([d for d in config_path.glob("run_*") if d.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run_* dirs under {config_path}")

    weights_sorted = []
    used_runs = []

    for run_dir in run_dirs:
        ckpts = sorted(run_dir.glob(ckpt_glob))
        if not ckpts:
            continue
        ckpt_path = ckpts[0]
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

        w_last = _load_last_w_from_history(ckpt)
        sd = _unwrap_compiled_state_dict(ckpt.get("state_dict", {}))

        args = ckpt.get("args", {}) or {}
        hidden_expected = int(args.get("hidden_n", args.get("hidden_size", 100)))

        if w_last is None:
            w_last, _ = _extract_dense_w_from_state_dict(sd, hidden_expected=hidden_expected)
            if w_last is None:
                continue

        perm = None
        perm_path = run_dir / "analysis" / "perm_peak_train.npy"

        if cache_perm and perm_path.exists():
            try:
                perm = np.load(perm_path).astype(int)
            except Exception:
                perm = None

        if perm is None:
            hidden_time_by_neuron = _hidden_from_training_sequence(ckpt)
            if hidden_time_by_neuron is not None:
                perm = _peak_time_perm(hidden_time_by_neuron)
                if cache_perm:
                    perm_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(str(perm_path), perm)

        weights_sorted.append(_apply_perm(w_last, perm))
        used_runs.append(run_dir.name)

    if not weights_sorted:
        raise FileNotFoundError(
            f"No usable runs found under {config_path}.\n"
            "Likely: checkpoints are missing X_mini for training ordering and/or "
            "state_dict lacks a square (H,H) recurrent matrix."
        )

    weights_sorted = np.stack(weights_sorted, axis=0)  # [R,H,H]
    w_mean = weights_sorted.mean(axis=0)

    w_sym = 0.5 * (w_mean + w_mean.T)
    w_anti = 0.5 * (w_mean - w_mean.T)

    weights_sym = 0.5 * (weights_sorted + weights_sorted.transpose(0, 2, 1))
    weights_anti = 0.5 * (weights_sorted - weights_sorted.transpose(0, 2, 1))

    norm_s = np.linalg.norm(w_sym, ord="fro")
    norm_a = np.linalg.norm(w_anti, ord="fro")
    norm_tot = norm_s + norm_a
    frac_a = norm_a / norm_tot if norm_tot > 0 else np.nan
    ratio_as = norm_a / norm_s if norm_s > 0 else np.inf

    print("\n[Fig4] Recurrent weight decomposition (mean over runs)")
    print("-----------------------------------------------------")
    print(f"  ||W_sym||_F   = {norm_s:.6f}")
    print(f"  ||W_anti||_F  = {norm_a:.6f}")
    print(f"  ||A|| / (||S||+||A||) = {frac_a:.4f}")
    print(f"  ||A|| / ||S||          = {ratio_as:.4f}")
    print(f"  runs used: {len(weights_sorted)}  ({', '.join(used_runs[:8])}{'...' if len(used_runs) > 8 else ''})\n")

    fig_sym = _plot_three_panels(
        w_sym,
        weights_sym,
        suptitle="Symmetric recurrent component (final)",
        tag_for_print="Fig4-SYM",
        eig_xlim_=eig_xlim,
    )

    fig_anti = _plot_three_panels(
        w_anti,
        weights_anti,
        suptitle="Antisymmetric recurrent component (final)",
        tag_for_print="Fig4-ANTI",
        eig_xlim_=eig_xlim,
    )

    return fig_sym, fig_anti


def _parse_figsize(values: Sequence[float]) -> Tuple[float, float]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError("figsize requires exactly two values: WIDTH HEIGHT")
    return float(values[0]), float(values[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot symmetric and antisymmetric decompositions of final recurrent weights."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Condition/config directory containing run_* checkpoint subdirectories.",
    )
    parser.add_argument(
        "--ckpt-glob",
        default="*.pth.tar",
        help="Checkpoint glob inside each run_* directory. Default: *.pth.tar",
    )
    parser.add_argument("--device", default="cpu", help="Torch device for hidden-state ordering. Default: cpu")
    parser.add_argument("--fontsize", type=int, default=12, help="Base font size. Default: 12")
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(16, 4.0),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size for each output figure. Default: 16 4.0",
    )
    parser.add_argument(
        "--savepath",
        type=str,
        default=None,
        help="Output path prefix. Writes <savepath>_symmetric.png and <savepath>_antisymmetric.png.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figures. Default: 300")
    parser.add_argument(
        "--no-cache-perm",
        action="store_true",
        help="Do not read/write run_*/analysis/perm_peak_train.npy.",
    )
    parser.add_argument(
        "--outlier-real-thr",
        type=float,
        default=2.0,
        help="Report eigenvalues with Re(lambda) above this value. Default: 2.0",
    )
    parser.add_argument(
        "--annotate-outliers",
        action="store_true",
        help="Annotate eigenvalue outliers in eigenspectrum panels.",
    )
    parser.add_argument(
        "--annotate-k",
        type=int,
        default=3,
        help="Maximum number of outliers to annotate. Default: 3",
    )
    parser.add_argument(
        "--eig-xlim",
        nargs=2,
        type=float,
        default=(-1.0, 2.5),
        metavar=("XMIN", "XMAX"),
        help="Eigenspectrum x-axis limits. Default: -1.0 2.5",
    )
    parser.add_argument(
        "--src-dir",
        type=str,
        default=None,
        help="Path to repository src directory. Usually unnecessary when run from repo root.",
    )
    parser.add_argument("--show", action="store_true", help="Display figures interactively.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _add_src_to_path(args.src_dir)

    fig_sym, fig_anti = plot_last_decomposed(
        config_path=args.config_path,
        ckpt_glob=args.ckpt_glob,
        device=args.device,
        fontsize=args.fontsize,
        figsize=(float(args.figsize[0]), float(args.figsize[1])),
        cache_perm=not args.no_cache_perm,
        outlier_real_thr=args.outlier_real_thr,
        annotate_outliers=args.annotate_outliers,
        annotate_k=args.annotate_k,
        show=args.show,
        eig_xlim=(float(args.eig_xlim[0]), float(args.eig_xlim[1])),
    )

    if args.savepath is not None:
        save_prefix = Path(args.savepath).expanduser()
        save_prefix.parent.mkdir(parents=True, exist_ok=True)
        sym_path = save_prefix.with_name(save_prefix.name + "_symmetric.png")
        anti_path = save_prefix.with_name(save_prefix.name + "_antisymmetric.png")
        fig_sym.savefig(sym_path, dpi=args.dpi, bbox_inches="tight")
        fig_anti.savefig(anti_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {sym_path}")
        print(f"Saved: {anti_path}")

    if not args.show:
        plt.close(fig_sym)
        plt.close(fig_anti)


if __name__ == "__main__":
    main()
