#!/usr/bin/env python3
"""
Plot Fig. 5 loss curves: loss vs. epoch.

This script plots mean ± SEM loss curves across run_* folders for one or more
training conditions. It expects each condition directory to contain files like:

    <condition_root>/run_*/<pattern>

where each CSV has columns:

    epoch, loss

Example:
    python ./src/figures/figure5_training_dynamics.py \
        ./data/runs/random \
        ./data/runs/identity \
        ./data/runs/cycshift/alpha0p00 \
        --labels random identity "cyclic shift α₀=0.00" \
        --savepath ./data/figures/figure5/random_identity_cyc.png \
        --fontsize 16 \
        --logx \
        --logy \
        --no-slope \
        --mh-color "#2c7fb8" \
        --lw 4

    python ./src/figures/figure5_training_dynamics.py \
            ./data/runs/random \
            ./data/runs/identity \
            ./data/runs/cycshift/alpha0p00 \
            ./data/runs/cycshift/alpha0p25 \
            ./data/runs/cycshift/alpha0p50 \
            ./data/runs/cycshift/alpha0p75 \
            ./data/runs/cycshift/alpha1p00 \
            --labels random identity "cyclic shift α₀=0.00" "cyclic shift α₀=0.25" "cyclic shift α₀=0.50" "cyclic shift α₀=0.75" "cyclic shift α₀=1.00" \
            --savepath ./data/figures/figure5/random_identity_cyc.png \
            --fontsize 16 \
            --logx \
            --logy \
            --no-slope \
            --mh-color "#2c7fb8" \
            --lw 4
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def fig1_loss_vs_epoch(
    condition_roots,
    pattern="*_loss_curve.csv",
    savepath=None,
    labels=None,
    fontsize=12,
    figsize=(6.5, 4.5),
    logx=False,
    logy=False,
    add_slope_in_legend=True,
    title="Loss vs epoch",
    xlabel="epoch",
    ylabel="loss",
    sem_alpha=0.2,
    lw=2.0,
    use_constrained_layout=True,
    mh_color="tab:blue",
):
    """Plot loss vs. epoch as mean ± SEM across run_* folders."""

    def _short_label_from_root(root):
        return os.path.normpath(str(root)).split(os.sep)[-1]

    def _read_csv_safe(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            print("[WARN] Failed to read CSV:", path, "error:", e)
            return None

    def _gather_loss_series(roots, pat):
        out = {}
        for root in roots:
            root = str(root)
            cond_id = os.path.normpath(root)
            paths = glob.glob(os.path.join(root, "run_*", pat))

            runs = []
            for p in sorted(paths):
                df = _read_csv_safe(p)
                if df is None or "epoch" not in df.columns or "loss" not in df.columns:
                    continue
                df = df.sort_values("epoch").set_index("epoch")
                runs.append(df)

            if runs:
                out[cond_id] = runs

        return out

    def _mean_sem_align(dfs, col):
        if not dfs:
            return None, None, None

        aligned = pd.concat([d[col] for d in dfs if col in d.columns], axis=1)
        aligned = aligned.sort_index()

        vals = aligned.values
        mean = np.nanmean(vals, axis=1)
        sem = np.nanstd(vals, axis=1, ddof=1) / np.sqrt(
            np.sum(np.isfinite(vals), axis=1).clip(min=1)
        )

        return aligned.index.values, mean, sem

    def _fit_slope(x, y, logx=False, logy=False):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        if logx:
            mask &= x > 0
        if logy:
            mask &= y > 0

        x, y = x[mask], y[mask]
        if x.size < 2:
            return float("nan")

        xt = np.log10(x) if logx else x
        yt = np.log10(y) if logy else y

        m, _b = np.polyfit(xt, yt, 1)
        return float(m)

    if isinstance(condition_roots, (str, os.PathLike)):
        condition_roots = [condition_roots]

    condition_roots = [str(r) for r in condition_roots]

    loss_series = _gather_loss_series(condition_roots, pattern)
    if not loss_series:
        raise FileNotFoundError(
            "No loss curves found. Looked for:\n"
            f"  <condition_root>/run_*/{pattern}\n"
            "and required columns ['epoch', 'loss']."
        )

    cond_ids = list(loss_series.keys())
    label_map = None

    if labels is not None:
        if len(labels) != len(cond_ids):
            print("[WARN] labels length != number of plotted conditions; ignoring labels.")
        else:
            label_map = {cid: lab for cid, lab in zip(cond_ids, labels)}

    plt.rcParams.update({"font.size": fontsize})
    fig, ax = plt.subplots(
        1,
        1,
        figsize=figsize,
        constrained_layout=use_constrained_layout,
    )

    mh_ids_in_order = [
        os.path.normpath(r)
        for r in condition_roots
        if "mexican_hat" in str(r)
    ]
    mh_ids_in_order = [cid for cid in mh_ids_in_order if cid in loss_series]

    mh_alpha_map = {}
    if mh_ids_in_order:
        a_min, a_max = 0.25, 1.0
        if len(mh_ids_in_order) == 1:
            mh_alpha_map[mh_ids_in_order[0]] = a_max
        else:
            for i, cid in enumerate(mh_ids_in_order):
                t = i / (len(mh_ids_in_order) - 1)
                mh_alpha_map[cid] = a_min + (a_max - a_min) * t

    for cond_id, runs in loss_series.items():
        x, mean, sem = _mean_sem_align(runs, "loss")
        if x is None:
            continue

        x_plot = x + 1 if logx else x

        base_label = (
            label_map.get(cond_id, _short_label_from_root(cond_id))
            if label_map
            else _short_label_from_root(cond_id)
        )

        if add_slope_in_legend:
            slope = _fit_slope(x_plot, mean, logx=logx, logy=logy)
            label = f"{base_label} (m={slope:.3g})"
        else:
            label = base_label

        if "mexican_hat" in cond_id:
            label = None
            color = mh_color
            line_alpha = mh_alpha_map.get(cond_id, 0.6)
        elif "random_pytorch" in cond_id:
            color = "k"
            line_alpha = 1.0
        else:
            color = None
            line_alpha = 1.0

        linestyle = "--" if "alpha0p70" in cond_id else "-"

        ax.plot(
            x_plot,
            mean,
            lw=lw,
            label=label,
            color=color,
            alpha=line_alpha,
            linestyle=linestyle,
        )

        ax.fill_between(
            x_plot,
            mean - sem,
            mean + sem,
            alpha=line_alpha * sem_alpha,
            color=color,
        )

    ax.set_title(title, fontsize=fontsize + 2)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    handles, legend_labels = ax.get_legend_handles_labels()

    if any("mexican_hat" in cid for cid in loss_series.keys()):
        handles.append(Line2D([0], [0], color=mh_color, lw=lw, alpha=1.0))
        legend_labels.append("mexican hat")

    ax.legend(handles, legend_labels, fontsize=max(8, fontsize - 2), frameon=False)

    if not use_constrained_layout:
        fig.tight_layout()

    if savepath:
        os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("[SAVE]", savepath)
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Fig. 1 loss vs. epoch curves from run_* loss CSVs."
    )

    parser.add_argument(
        "condition_roots",
        nargs="+",
        help="Condition root directories containing run_*/<pattern> CSVs.",
    )

    parser.add_argument(
        "--pattern",
        default="*_loss_curve.csv",
        help="Loss CSV filename pattern inside each run_* directory.",
    )

    parser.add_argument("--savepath", default=None, help="Output figure path.")
    parser.add_argument("--labels", nargs="*", default=None, help="Legend labels.")
    parser.add_argument("--fontsize", type=int, default=12)
    parser.add_argument("--figsize", type=float, nargs=2, default=(6.5, 4.5))
    parser.add_argument("--logx", action="store_true")
    parser.add_argument("--logy", action="store_true")
    parser.add_argument("--no-slope", action="store_true")
    parser.add_argument("--title", default="Loss vs epoch")
    parser.add_argument("--xlabel", default="epoch")
    parser.add_argument("--ylabel", default="loss")
    parser.add_argument("--sem-alpha", type=float, default=0.2)
    parser.add_argument("--lw", type=float, default=2.0)
    parser.add_argument("--mh-color", default="tab:blue")
    parser.add_argument("--tight-layout", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    fig1_loss_vs_epoch(
        condition_roots=args.condition_roots,
        pattern=args.pattern,
        savepath=args.savepath,
        labels=args.labels,
        fontsize=args.fontsize,
        figsize=tuple(args.figsize),
        logx=args.logx,
        logy=args.logy,
        add_slope_in_legend=not args.no_slope,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        sem_alpha=args.sem_alpha,
        lw=args.lw,
        use_constrained_layout=not args.tight_layout,
        mh_color=args.mh_color,
    )


if __name__ == "__main__":
    main()