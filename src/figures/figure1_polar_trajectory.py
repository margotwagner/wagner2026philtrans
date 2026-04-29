#!/usr/bin/env python3
"""
Plot replay and prediction polar trajectories for a single trained run.

Usage
-----
Pass either a run directory:

    python src/figures/plot_run_polar_trajectories.py \
        ./data/runs/prediction/.../run_09

or a specific checkpoint:

    python src/figures/plot_run_polar_trajectories.py \
        ./data/runs/prediction/.../run_09/model.pth.tar

Optionally save the figure:

    python src/figures/plot_run_polar_trajectories.py \
        ./data/runs/prediction/.../run_09 \
        --savepath ../data/figures/run_09_polar.png \
        --fontsize 22 \
        --linewidth 4

Example:
    python ./src/figures/figure1_polar_trajectory.py ./data/runs/random/run_00 --fontsize 22 --linewidth 4 --savepath ./data/figures/figure1/random_run_00.png
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_angles_csv(path):
    """Robust CSV loader matching make_figures.py Fig. 4 behavior."""
    if not os.path.exists(path):
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        with open(path, "r") as f:
            rows = list(csv.reader(f))

        if not rows:
            return None

        header = rows[0]
        values = rows[1:]

        try:
            arr = np.array(values, dtype=float)
            return pd.DataFrame(arr, columns=header[: arr.shape[1]])
        except Exception:
            return None


def first_angle_col(df):
    """Pick an angle column, allowing for slight naming differences."""
    if df is None or df.empty:
        return None

    preferred = ["theta_pred", "theta_true", "theta", "angle", "phi", "ang", "phase"]
    lower = {c.lower(): c for c in df.columns}

    for key in preferred:
        if key in lower:
            return lower[key]

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            return col

    return None


def plot_polar_angles(ax, df, label, style="-", linewidth=1.8, true_scale=0.85):
    """Plot angle versus time on a polar axis."""
    if df is None or df.empty:
        return None

    t = df["t"].values if "t" in df.columns else np.arange(len(df))

    angle_col = "theta_pred" if "theta_pred" in df.columns else first_angle_col(df)
    if angle_col is None:
        return None

    (line,) = ax.plot(
        df[angle_col].values,
        t,
        linestyle=style,
        linewidth=linewidth,
        label=label,
    )

    if "theta_true" in df.columns:
        ax.plot(
            df["theta_true"].values,
            t,
            linestyle="--",
            linewidth=linewidth * true_scale,
            label=f"{label} true",
            color=line.get_color(),
        )

    return line


def find_checkpoint(run_path):
    """
    Accept either a checkpoint path or a run directory.

    Returns
    -------
    Path
        Path to the checkpoint.
    """
    path = Path(run_path).expanduser().resolve()

    if path.is_file() and path.name.endswith(".pth.tar"):
        return path

    checkpoints = sorted(path.glob("*.pth.tar"))

    if not checkpoints:
        raise FileNotFoundError(f"No .pth.tar checkpoint found in {path}")

    return checkpoints[0]


def plot_run_polar_trajectories(
    run_path,
    savepath=None,
    title="Polar Trajectories",
    fontsize=12,
    linewidth=2.5,
):
    """
    Plot replay and prediction trajectories for one trained run.

    Expected files are located from the checkpoint stub:

        <checkpoint_stub>_replay_angles.csv
        <checkpoint_stub>_prediction_angles.csv
    """
    checkpoint_path = find_checkpoint(run_path)

    # Drop ".pth.tar" exactly.
    stub = str(checkpoint_path)[:-8]

    replay_angles = read_angles_csv(stub + "_replay_angles.csv")
    prediction_angles = read_angles_csv(stub + "_prediction_angles.csv")

    plt.rcParams.update({"font.size": fontsize})

    fig = plt.figure(figsize=(7.5, 5.5))
    ax = fig.add_subplot(111, projection="polar")

    any_line = False

    if plot_polar_angles(
        ax,
        replay_angles,
        "Replay θ̂(t)",
        "-",
        linewidth=linewidth,
    ) is not None:
        any_line = True

    if plot_polar_angles(
        ax,
        prediction_angles,
        "Prediction θ̂(t)",
        "-",
        linewidth=linewidth,
    ) is not None:
        any_line = True

    ax.set_title(title, fontsize=fontsize)

    if any_line:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.15, 0.5),
            frameon=False,
            fontsize=max(8, fontsize - 2),
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No angle CSVs found",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    plt.tight_layout()

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
        description="Plot replay and prediction polar trajectories for one run."
    )

    parser.add_argument(
        "run_path",
        help="Path to a run directory or a .pth.tar checkpoint.",
    )

    parser.add_argument(
        "--savepath",
        default=None,
        help="Optional path to save the figure. If omitted, the figure is shown.",
    )

    parser.add_argument(
        "--title",
        default="Polar Trajectories",
        help="Figure title.",
    )

    parser.add_argument(
        "--fontsize",
        type=int,
        default=12,
        help="Base font size.",
    )

    parser.add_argument(
        "--linewidth",
        type=float,
        default=2.5,
        help="Trajectory line width.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    plot_run_polar_trajectories(
        run_path=args.run_path,
        savepath=args.savepath,
        title=args.title,
        fontsize=args.fontsize,
        linewidth=args.linewidth,
    )


if __name__ == "__main__":
    main()