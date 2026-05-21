#!/usr/bin/env python3
"""
Plot Figure 6 alpha(t) training dynamics from run_level.csv files.

This script plots

    alpha(t) = ||S||_F / (||S||_F + ||A||_F)

where S and A are the symmetric and antisymmetric components of the recurrent
weight matrix, using precomputed columns in run_level.csv.

Expected columns in each run_level.csv:
    - fro_S_offline
    - fro_A_offline
    - epoch, snapshot_idx, or neither; if neither exists, row index is used
    - run_id is optional

Example:
    python ./src/figures/figure7_alpha_training.py \
        --condition-roots \
        ./data/runs/random \
        ./data/runs/identity \
        ./data/runs/cycshift/alpha0p00 \
        --savepath ./data/figures/figure7/random_identity_cyc.png \
        --fontsize 16 \
        --median-lw 4 \
        --mh-color '#2c7fb8'

You can also use one or more glob patterns:
    python ./src/figures/figure7_alpha_training.py \
        --condition-globs './data/runs/random' \
        --condition-roots './data/runs/cycshift/alpha*/' \
        --savepath ./data/figures/figure7/alpha_time_series.png
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def _read_csv_safe(path: str | os.PathLike) -> pd.DataFrame | None:
    """Read a CSV if it exists; otherwise return None with a warning."""
    path = str(path)
    if os.path.isfile(path):
        try:
            return pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to read CSV: {path} error: {exc}")
    else:
        print(f"[WARN] Missing CSV: {path}")
    return None


def _pick_time_col(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Return the preferred time column, adding row_idx if needed."""
    for col in ("epoch", "snapshot_idx"):
        if col in df.columns:
            return df, col
    df = df.copy()
    df["row_idx"] = np.arange(len(df))
    return df, "row_idx"


def _alpha_from_df(df: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute alpha(t) from fro_S_offline and fro_A_offline columns."""
    if df is None or df.empty:
        return None, None
    if not {"fro_S_offline", "fro_A_offline"}.issubset(df.columns):
        return None, None

    df, time_col = _pick_time_col(df)
    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    s_norm = pd.to_numeric(df["fro_S_offline"], errors="coerce").to_numpy(dtype=float)
    a_norm = pd.to_numeric(df["fro_A_offline"], errors="coerce").to_numpy(dtype=float)
    denom = s_norm + a_norm

    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where(denom > 0, s_norm / denom, np.nan)

    return t, alpha


def _load_run_level_files(
    condition_roots: Sequence[str | os.PathLike],
    run_level_filename: str,
) -> pd.DataFrame:
    """Load and concatenate run_level.csv files from condition directories."""
    dfs: list[pd.DataFrame] = []

    for root in condition_roots:
        root = os.path.normpath(str(root))
        csv_path = os.path.join(root, run_level_filename)
        df = _read_csv_safe(csv_path)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["condition_root"] = root
        df["condition_id"] = root
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No usable {run_level_filename} files were found under the provided condition roots."
        )

    return pd.concat(dfs, ignore_index=True, sort=False)


def fig3_alpha_time_series_mh_style(
    run_level_df: pd.DataFrame | None = None,
    condition_roots: Sequence[str | os.PathLike] | None = None,
    run_level_filename: str = "run_level.csv",
    savepath: str | os.PathLike | None = None,
    fontsize: int = 12,
    figsize: tuple[float, float] = (7.5, 4.8),
    thin_lw: float = 1.0,
    thin_alpha: float = 0.30,
    show_median_iqr: bool = True,
    median_lw: float = 2.2,
    iqr_alpha: float = 0.12,
    legend_outside: bool = False,
    use_constrained_layout: bool = True,
    mh_color: str = "tab:blue",
    mh_alpha_min: float = 0.25,
    mh_alpha_max: float = 1.0,
    collapse_mh_legend: bool = True,
    dashed_if_contains: tuple[str, ...] = ("alpha0p70",),
    show_runs: bool = False,
):
    """Plot alpha(t) over training with random in black and Mexican-hat curves in one color."""
    if run_level_df is None:
        if condition_roots is None:
            raise ValueError("Provide either run_level_df or condition_roots.")
        condition_roots = [os.path.normpath(str(root)) for root in condition_roots]
        df_rl = _load_run_level_files(condition_roots, run_level_filename)
    else:
        df_rl = run_level_df.copy()
        if "condition_id" not in df_rl.columns:
            raise ValueError("run_level_df must have a 'condition_id' column.")

    if df_rl.empty:
        raise ValueError("run_level_df is empty.")
    if not {"fro_S_offline", "fro_A_offline"}.issubset(df_rl.columns):
        raise ValueError("Need columns: fro_S_offline, fro_A_offline.")

    if condition_roots is not None:
        mh_ids_in_order = [cid for cid in condition_roots if "mexican_hat" in str(cid)]
    else:
        mh_ids_in_order = sorted(
            [cid for cid in df_rl["condition_id"].unique() if "mexican_hat" in str(cid)]
        )

    valid_condition_ids = set(df_rl["condition_id"].unique())
    mh_ids_in_order = [cid for cid in mh_ids_in_order if cid in valid_condition_ids]

    mh_alpha_map: dict[str, float] = {}
    if len(mh_ids_in_order) == 1:
        mh_alpha_map[str(mh_ids_in_order[0])] = mh_alpha_max
    elif len(mh_ids_in_order) > 1:
        for i, cid in enumerate(mh_ids_in_order):
            ramp = i / (len(mh_ids_in_order) - 1)
            mh_alpha_map[str(cid)] = mh_alpha_min + (mh_alpha_max - mh_alpha_min) * ramp

    plt.rcParams.update({"font.size": fontsize})
    fig, ax = plt.subplots(
        1,
        1,
        figsize=figsize,
        constrained_layout=use_constrained_layout,
    )

    group_cols = ["condition_id"]
    if "run_id" in df_rl.columns:
        group_cols.append("run_id")

    if show_runs:
        for _, group in df_rl.groupby(group_cols):
            cond_id = str(group["condition_id"].iloc[0])
            t, alpha = _alpha_from_df(group)
            if t is None or alpha is None:
                continue
            order = np.argsort(t)

            if "random" in cond_id:
                color = "k"
                line_alpha = 1.0
            elif "mexican_hat" in cond_id:
                color = mh_color
                line_alpha = mh_alpha_map.get(cond_id, 0.6)
            else:
                color = None
                line_alpha = 0.8

            linestyle = "--" if any(s in cond_id for s in dashed_if_contains) else "-"
            ax.plot(
                t[order],
                alpha[order],
                lw=thin_lw,
                alpha=thin_alpha * line_alpha,
                color=color,
                linestyle=linestyle,
            )

    if show_median_iqr:
        df_rl, time_col = _pick_time_col(df_rl)

        for cond_id_raw, cdf in df_rl.groupby("condition_id"):
            cond_id = str(cond_id_raw)
            if "random" in cond_id:
                color = "k"
                line_alpha = 1.0
            elif "mexican_hat" in cond_id:
                color = mh_color
                line_alpha = mh_alpha_map.get(cond_id, 0.6)
            else:
                color = None
                line_alpha = 0.9

            linestyle = "--" if any(s in cond_id for s in dashed_if_contains) else "-"
            xs = np.sort(pd.to_numeric(cdf[time_col], errors="coerce").dropna().unique()).astype(float)
            med, q1, q3 = [], [], []

            numeric_time = pd.to_numeric(cdf[time_col], errors="coerce")
            for x_val in xs:
                sub = cdf[numeric_time == x_val]
                _, alpha_vals = _alpha_from_df(sub)
                if alpha_vals is None:
                    alpha_vals = np.array([])
                alpha_vals = alpha_vals[np.isfinite(alpha_vals)]

                if alpha_vals.size == 0:
                    med.append(np.nan)
                    q1.append(np.nan)
                    q3.append(np.nan)
                else:
                    med.append(np.nanmedian(alpha_vals))
                    q1.append(np.nanpercentile(alpha_vals, 25))
                    q3.append(np.nanpercentile(alpha_vals, 75))

            ax.plot(
                xs,
                med,
                lw=median_lw,
                color=color,
                alpha=line_alpha,
                linestyle=linestyle,
            )
            ax.fill_between(xs, q1, q3, color=color, alpha=iqr_alpha * line_alpha)

    ax.set_title("α(t) over training", fontsize=fontsize + 2)
    ax.set_xlabel("epoch", fontsize=fontsize)
    ax.set_ylabel(r"$\alpha(t)=\|S\|_F / (\|S\|_F+\|A\|_F)$", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)

    out_handles: list[Line2D] = []
    out_labels: list[str] = []

    if any("random" in str(cid) for cid in df_rl["condition_id"].unique()):
        out_handles.append(Line2D([0], [0], color="k", lw=median_lw, alpha=1.0))
        out_labels.append("random")

    if collapse_mh_legend and any(
        "mexican_hat" in str(cid) for cid in df_rl["condition_id"].unique()
    ):
        out_handles.append(Line2D([0], [0], color=mh_color, lw=median_lw, alpha=1.0))
        out_labels.append("mexican hat")

    if out_handles:
        ax.legend(
            out_handles,
            out_labels,
            frameon=False,
            fontsize=max(8, fontsize - 2),
            loc="best" if not legend_outside else "center left",
            bbox_to_anchor=(1.02, 0.5) if legend_outside else None,
        )

    if not use_constrained_layout:
        fig.tight_layout()

    if savepath:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVE] {savepath}")
        return None, None

    plt.show()
    return fig, ax


def _expand_condition_inputs(
    condition_roots: Iterable[str] | None,
    condition_globs: Iterable[str] | None,
) -> list[str]:
    """Combine explicit condition roots and glob-expanded condition roots."""
    roots: list[str] = []
    if condition_roots:
        roots.extend(str(root) for root in condition_roots)
    if condition_globs:
        for pattern in condition_globs:
            matches = sorted(glob.glob(pattern))
            if not matches:
                print(f"[WARN] No matches for glob: {pattern}")
            roots.extend(matches)

    # Preserve order while removing duplicates.
    seen: set[str] = set()
    unique_roots: list[str] = []
    for root in roots:
        norm = os.path.normpath(root)
        if norm not in seen:
            seen.add(norm)
            unique_roots.append(norm)

    return unique_roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot alpha(t) training dynamics from one or more run_level.csv files."
    )
    parser.add_argument(
        "--condition-roots",
        nargs="+",
        default=None,
        help="Condition directories containing run_level.csv files.",
    )
    parser.add_argument(
        "--condition-globs",
        nargs="+",
        default=None,
        help="Glob patterns that expand to condition directories containing run_level.csv files.",
    )
    parser.add_argument(
        "--run-level-filename",
        default="run_level.csv",
        help="Name of the run-level CSV file inside each condition root. Default: run_level.csv.",
    )
    parser.add_argument("--savepath", default=None, help="Output image path. If omitted, show interactively.")
    parser.add_argument("--fontsize", type=int, default=12, help="Base font size.")
    parser.add_argument("--figsize", type=float, nargs=2, default=(7.5, 4.8), help="Figure size: width height.")
    parser.add_argument("--mh-color", default="tab:blue", help="Color for mexican_hat conditions.")
    parser.add_argument("--mh-alpha-min", type=float, default=0.25, help="Minimum alpha for mexican_hat ramp.")
    parser.add_argument("--mh-alpha-max", type=float, default=1.0, help="Maximum alpha for mexican_hat ramp.")
    parser.add_argument("--median-lw", type=float, default=2.2, help="Median line width.")
    parser.add_argument("--iqr-alpha", type=float, default=0.12, help="IQR band alpha.")
    parser.add_argument("--thin-lw", type=float, default=1.0, help="Per-run trace line width.")
    parser.add_argument("--thin-alpha", type=float, default=0.30, help="Per-run trace alpha multiplier.")
    parser.add_argument("--show-runs", action="store_true", help="Also show thin per-run traces.")
    parser.add_argument("--no-median-iqr", action="store_true", help="Do not show median ± IQR curves.")
    parser.add_argument("--legend-outside", action="store_true", help="Place legend outside the axes.")
    parser.add_argument("--no-constrained-layout", action="store_true", help="Disable constrained layout.")
    parser.add_argument(
        "--dashed-if-contains",
        nargs="*",
        default=["alpha0p70"],
        help="Condition-id substrings that should be plotted with dashed lines.",
    )
    parser.add_argument(
        "--no-collapse-mh-legend",
        action="store_true",
        help="Do not collapse mexican_hat conditions into one legend entry.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    condition_roots = _expand_condition_inputs(args.condition_roots, args.condition_globs)

    if not condition_roots:
        raise ValueError("Provide at least one condition directory with --condition-roots or --condition-globs.")

    fig3_alpha_time_series_mh_style(
        condition_roots=condition_roots,
        run_level_filename=args.run_level_filename,
        savepath=args.savepath,
        fontsize=args.fontsize,
        figsize=tuple(args.figsize),
        thin_lw=args.thin_lw,
        thin_alpha=args.thin_alpha,
        show_median_iqr=not args.no_median_iqr,
        median_lw=args.median_lw,
        iqr_alpha=args.iqr_alpha,
        legend_outside=args.legend_outside,
        use_constrained_layout=not args.no_constrained_layout,
        mh_color=args.mh_color,
        mh_alpha_min=args.mh_alpha_min,
        mh_alpha_max=args.mh_alpha_max,
        collapse_mh_legend=not args.no_collapse_mh_legend,
        dashed_if_contains=tuple(args.dashed_if_contains),
        show_runs=args.show_runs,
    )


if __name__ == "__main__":
    main()
