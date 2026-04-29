#!/usr/bin/env python3
"""
Plot replay and prediction hidden-state heatmaps for a single run.

Usage
-----
Run on a directory:

    python ./src/figures/plot_replay_prediction_heatmaps.py \
        ./data/runs/prediction/.../run_09

Optional arguments:

    --savepath ../data/figures/run_09_heatmap.png
    --max-units 100
    --max-time 100
    --fontsize 20

Expected files (produced by evaluate.py):
    <stub>_replay_hidden.npy
    <stub>_prediction_hidden.npy
    
Example:
    python ./src/figures/figure2_hidden_activity.py ./data/runs/random/run_00/ --savepath ./data/figures/figure2/random_run_00.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def find_checkpoint(run_dir, ckpt_glob):
    ckpts = sorted(run_dir.glob(ckpt_glob))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint matching {ckpt_glob} in {run_dir}")
    return ckpts[0]


def load_TN(path):
    """Load array and return shape [T, H]."""
    arr = np.load(path)

    if arr.ndim == 3:  # [B, T, H]
        arr = arr[0]

    if arr.ndim != 2:
        raise ValueError(f"Unexpected shape {arr.shape} in {path}")

    return arr  # [T, H]


def zscore_peak_sort(H_TN):
    """
    Z-score each unit and sort by peak activation time.
    Returns [units, time]
    """
    Z = (H_TN - H_TN.mean(axis=0, keepdims=True)) / (
        H_TN.std(axis=0, keepdims=True) + 1e-8
    )

    order = np.argsort(np.argmax(Z, axis=0))
    return Z[:, order].T


def plot_replay_and_pred(
    run_path,
    ckpt_glob="*.pth.tar",
    max_units=100,
    max_time=100,
    fontsize=12,
    figsize=(10, 4),
    cmap="viridis",
    vmin=None,
    vmax=None,
    savepath=None,
):
    """
    Plot replay and prediction hidden-state heatmaps for a run.
    """
    run_dir = Path(run_path).expanduser().resolve()

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    ckpt = find_checkpoint(run_dir, ckpt_glob)

    # Build stub (drop '.pth.tar')
    stub = ckpt.name[:-8] if ckpt.name.endswith(".pth.tar") else ckpt.stem

    replay_path = run_dir / f"{stub}_replay_hidden.npy"
    pred_path = run_dir / f"{stub}_prediction_hidden.npy"

    if not replay_path.exists():
        raise FileNotFoundError(f"Missing {replay_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing {pred_path}")

    rp = zscore_peak_sort(load_TN(replay_path))[:max_units, :max_time]
    pr = zscore_peak_sort(load_TN(pred_path))[:max_units, :max_time]

    # Shared color scale
    vmin_eff = np.min([rp.min(), pr.min()]) if vmin is None else vmin
    vmax_eff = np.max([rp.max(), pr.max()]) if vmax is None else vmax

    plt.rcParams.update({"font.size": fontsize})

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    im0 = axes[0].imshow(rp, aspect="auto", cmap=cmap, vmin=vmin_eff, vmax=vmax_eff)
    axes[0].set_title("Replay Output")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Neuron")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pr, aspect="auto", cmap=cmap, vmin=vmin_eff, vmax=vmax_eff)
    axes[1].set_title("Prediction Output")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Neuron")

    if savepath:
        savepath = Path(savepath).expanduser().resolve()
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot replay and prediction hidden-state heatmaps."
    )

    parser.add_argument("run_path", help="Path to run directory")

    parser.add_argument("--ckpt-glob", default="*.pth.tar")

    parser.add_argument("--max-units", type=int, default=100)
    parser.add_argument("--max-time", type=int, default=100)

    parser.add_argument("--fontsize", type=int, default=12)
    parser.add_argument("--figsize", type=float, nargs=2, default=(10, 4))

    parser.add_argument("--cmap", default="viridis")

    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)

    parser.add_argument("--savepath", default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    plot_replay_and_pred(
        run_path=args.run_path,
        ckpt_glob=args.ckpt_glob,
        max_units=args.max_units,
        max_time=args.max_time,
        fontsize=args.fontsize,
        figsize=tuple(args.figsize),
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        savepath=args.savepath,
    )


if __name__ == "__main__":
    main()