#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py
- Matplotlib-only figure suite for your Elman RNN project.
- Works both as a CLI script and as an importable module for notebooks.
- Python 3.6.7-compatible

Inputs (flexible):
- Aggregated CSVs:
    run_level.csv           (per-run rows)
    condition_summary.csv   (per-condition rows)
- Optional per-run offline & eval CSVs discovered by glob if not pre-merged.

Outputs:
- PNGs saved to --figdir (default: ./figs)

Notebook usage:
    TODO
    ...
"""

from __future__ import print_function
import os
import glob
import csv
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
from pathlib import Path
from typing import Optional
from matplotlib.lines import Line2D

matplotlib.use("Agg")  # safe for headless
import matplotlib.pyplot as plt

# ---------------------------
# Data loading & safe helpers
# ---------------------------


def _ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _read_csv_safe(path):
    if path and os.path.isfile(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            print("[WARN] Failed to read CSV:", path, "error:", e)
            return None
    return None


def _first_existing(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def _col(df, name, default=None):
    """Get column by name if exists, else default."""
    if df is None or name not in df.columns:
        return None if default is None else default
    return df[name]


def _has_cols(df, cols):
    return df is not None and all(c in df.columns for c in cols)


def _filter_df(df, condition_regex=None, run_regex=None):
    if df is None:
        return df
    out = df.copy()
    if condition_regex and "condition_id" in out.columns:
        out = out[
            out["condition_id"].astype(str).str.contains(condition_regex, regex=True)
        ]
    if run_regex and "run_id" in out.columns:
        out = out[out["run_id"].astype(str).str.contains(run_regex, regex=True)]
    return out


def _infer_condition_id_from_root(root):
    # Try to make a readable condition_id from the folder path
    # e.g., ".../shifted-cyc/frobenius/sym0p90/shiftmh_n100_fro" -> "shifted-cyc/frobenius/sym0p90/shiftmh_n100_fro"
    parts = []
    cur = os.path.normpath(root).split(os.sep)
    # take last 3-4 components; adjust to taste
    take = min(4, len(cur))
    parts = cur[-take:]
    return "/".join(parts)


def load_many_conditions(condition_roots):
    """
    Read aggregated CSVs for each condition root and concatenate:
      - expects each root to contain run_level.csv and condition_summary.csv
      - also globs per-run offline CSVs like *_offline_metrics.csv under each root
    Returns: (run_level_df, condition_df, offline_df, eval_df_or_None)
    """
    rl_list, cs_list, off_list = [], [], []

    for root in condition_roots:
        # per-condition run/summary tables
        rl = _read_csv_safe(os.path.join(root, "run_level.csv"))
        cs = _read_csv_safe(os.path.join(root, "condition_summary.csv"))
        # offline per-run CSV(s)
        cand = glob.glob(os.path.join(root, "**", "*offline*.csv"), recursive=True)

        cond_id = _infer_condition_id_from_root(root)

        if rl is not None:
            if "condition_id" not in rl.columns:
                rl["condition_id"] = cond_id
            if "condition_root" not in rl.columns:
                rl["condition_root"] = root
            rl_list.append(rl)

        if cs is not None:
            if "condition_id" not in cs.columns:
                cs["condition_id"] = cond_id
            if "condition_root" not in cs.columns:
                cs["condition_root"] = root
            cs_list.append(cs)

        for c in cand:
            df = _read_csv_safe(c)
            if df is not None:
                if "condition_id" not in df.columns:
                    df["condition_id"] = cond_id
                off_list.append(df)

    rl = pd.concat(rl_list, sort=False, ignore_index=True) if rl_list else None
    cs = pd.concat(cs_list, sort=False, ignore_index=True) if cs_list else None
    off = pd.concat(off_list, sort=False, ignore_index=True) if off_list else None
    return rl, cs, off, None


def _cs_to_wide(cs, value_col="mean"):
    """
    Convert tall condition_summary (columns: ['metric','mean','std',...,'condition_id'])
    into wide format with one row per condition and one column per metric.
    """
    if cs is None or not {"metric", value_col}.issubset(set(cs.columns)):
        return None
    if "condition_id" not in cs.columns:
        cs = cs.copy()
        cs["condition_id"] = "cond"  # fallback if not provided
    wide = cs.pivot_table(
        index="condition_id", columns="metric", values=value_col, aggfunc="first"
    )
    wide = wide.reset_index()
    # Ensure column names are plain strings (matplotlib friendly)
    wide.columns = [str(c) for c in wide.columns]
    return wide


def _alpha_from_condition_id(cid):
    """
    Extract alpha from condition_id strings that contain either:
      - 'symXpYY'   (e.g., 'sym0p25')
      - 'alphaXpYY' (e.g., 'alpha0p25')
      - 'alpha=0.25' or 'alpha0.25'
    Returns float in [0,1] or None if not found.
    """
    if cid is None:
        return None
    s = str(cid)

    # 1) sym0p25 or alpha0p25 (most common in your runs)
    m = re.search(r"(?:sym|alpha)(\d+)p(\d{2})", s)
    if m:
        major = int(m.group(1))
        minor = int(m.group(2))
        return major + minor / 100.0

    # 2) alpha=0.25 or alpha0.25
    m = re.search(r"alpha(?:=)?(\d+(?:\.\d+)?)", s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None

    return None


def _short_label_from_root(root: str) -> str:
    # last path component only
    return os.path.normpath(str(root)).split(os.sep)[-1]


def _fit_slope(x, y, logx=False, logy=False):
    """
    Return slope m of a best-fit line after transforming axes
    according to logx/logy (base-10). We fit y_t = m*x_t + b where
    x_t = log10(x) if logx else x, and y_t = log10(y) if logy else y.
    """

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
    m, b = np.polyfit(xt, yt, 1)
    return float(m)


def _pick_time_col_runlevel(df: pd.DataFrame) -> str:
    """Preferred time key for run_level: epoch > snapshot_idx > synthetic row index."""
    for c in ("epoch", "snapshot_idx"):
        if c in df.columns:
            return c
    df = df.copy()
    df["row_idx"] = np.arange(len(df))
    return "row_idx"


def _alpha_series_from_runlevel(df: pd.DataFrame):
    """
    Compute alpha = fro_S_offline / (fro_S_offline + fro_A_offline) for every row.
    Returns (time_col_name, time_values_np, alpha_values_np), or (None,None,None) if unavailable.
    """
    if df is None or df.empty:
        return None, None, None
    if not {"fro_S_offline", "fro_A_offline"}.issubset(df.columns):
        return None, None, None
    tcol = _pick_time_col_runlevel(df)
    t = pd.to_numeric(df[tcol], errors="coerce").values
    S = pd.to_numeric(df["fro_S_offline"], errors="coerce").astype(float).values
    A = pd.to_numeric(df["fro_A_offline"], errors="coerce").astype(float).values
    denom = S + A
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where(denom > 0, S / denom, np.nan)
    return tcol, t, alpha


def _linestyle_map_for_alphas(alpha_vals):
    """Stable map: sorted unique α0 -> a distinct linestyle.
    Always map 'no alpha' (None) to solid line.
    """
    LINESTYLES_ORDER = ["-", "--", "-.", ":", (0, (1, 1))]
    levels = sorted(set(float(a) for a in alpha_vals if pd.notna(a)))
    ls = {a: LINESTYLES_ORDER[i % len(LINESTYLES_ORDER)] for i, a in enumerate(levels)}
    ls[None] = "-"  # ensure single-variant / no-α₀ conditions are solid
    return ls, levels  # 'levels' excludes None, so the style legend only shows real α₀


# ---------------------------
# Matplotlib style utilities
# ---------------------------


def _style(ax, fontsize=12, title=None, xlabel=None, ylabel=None, legend=False):
    ax.tick_params(axis="both", labelsize=fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    if title:
        ax.set_title(title, fontsize=fontsize + 2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if legend:
        leg = ax.legend(fontsize=max(8, fontsize - 2), frameon=False)
        if leg is not None:
            for l in leg.get_lines():
                l.set_linewidth(2.0)


def _savefig(fig, path, constrain=True):
    _ensure_dir(os.path.dirname(path))
    # Use constrained layout if available
    if constrain:
        try:
            fig.set_constrained_layout(True)
        except Exception:
            fig.tight_layout()
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[SAVE]", path)


def _build_metrics_table_from_condition_summary(
    cs: pd.DataFrame, prefer: list
) -> Optional[pd.DataFrame]:
    """
    From condition_summary (long form: ['condition_id','metric','mean','std']),
    build a compact table (rows = metrics present & ordered by 'prefer';
    columns = condition_id; cells = 'mean±std' rounded).

    Returns a DataFrame of strings, or None if nothing usable.
    """
    if cs is None or cs.empty:
        return None
    need = {"condition_id", "metric", "mean", "std"}
    if not need.issubset(cs.columns):
        return None

    present = [m for m in prefer if m in cs["metric"].unique()]
    if not present:
        return None

    def _fmt(m, s):
        if pd.isna(m):
            return ""
        if pd.isna(s) or s == 0:
            return f"{float(m):.3g}"
        return f"{float(m):.3g}±{float(s):.2g}"

    rows = []
    conds = sorted(cs["condition_id"].unique())
    for m in present:
        row = {"metric": m}
        sub = cs[cs["metric"] == m]
        for c in conds:
            chunk = sub[sub["condition_id"] == c]
            if chunk.empty:
                row[c] = ""
            else:
                row[c] = _fmt(
                    chunk["mean"].iloc[0],
                    chunk["std"].iloc[0] if "std" in chunk.columns else float("nan"),
                )
        rows.append(row)
    table = pd.DataFrame(rows)
    return table[["metric"] + conds]


# ---------------------------
# Figure 1 – Training speed
# ---------------------------


def _gather_series(condition_roots, pattern):
    """Collect per-run timeseries matching a filename pattern under each condition root.
    Returns dict: {cond_id: [DataFrame_per_run_with_epoch_index, ...]}."""

    out = {}
    for root in condition_roots:
        cond_id = _infer_condition_id_from_root(root)
        paths = glob.glob(os.path.join(root, "run_*", pattern))
        runs = []
        for p in sorted(paths):
            df = _read_csv_safe(p)
            if df is None or "epoch" not in df.columns:
                continue
            df = df.sort_values("epoch").set_index("epoch")
            runs.append(df)
        if runs:
            out[cond_id] = runs
    return out


def _mean_sem_align(dfs, col):
    """Align on union of epochs; return (epochs, mean, sem)."""

    if not dfs:
        return None, None, None
    # outer-join on epoch
    aligned = pd.concat([d[col] for d in dfs if col in d.columns], axis=1)
    aligned = aligned.sort_index()
    vals = aligned.values  # [T, R]
    mean = np.nanmean(vals, axis=1)
    sem = np.nanstd(vals, axis=1, ddof=1) / np.sqrt(
        np.sum(np.isfinite(vals), axis=1).clip(min=1)
    )
    return aligned.index.values, mean, sem


def fig1_training_dynamics(
    condition_roots,
    savepath=None,
    fontsize=12,
    logxA=False,
    logyA=False,
    logxB=False,
    logyB=False,
    labels=None,
):
    """
    Fig 1 (2x2):
      A: loss vs epoch
      B: grad L2 (post) vs epoch
      C: spectral radius vs epoch
      D: Frobenius norm vs epoch
    """
    L = _gather_series(condition_roots, "*_loss_curve.csv")
    G = _gather_series(condition_roots, "*_grad_curve.csv")
    W = _gather_series(condition_roots, "*_wstruct_curve.csv")
    if not (L or G or W):
        print("[SKIP] fig1_training_dynamics: no per-run series found")
        return

    # --- Optional custom labels for Figure 1 ---
    user_labels = None
    label_map = None
    if labels:
        # Split on commas, ignore empty chunks
        user_labels = [s.strip() for s in labels.split(",") if s.strip()]

        # Only care about conditions that actually have loss curves (L)
        cond_ids = list(L.keys())
        if len(user_labels) != len(cond_ids):
            print(
                "[WARN] --fig1_labels count does not match number of plotted conditions; ignoring."
            )
            user_labels = None
        else:
            # Map cond_id -> label
            label_map = {cid: lab for cid, lab in zip(cond_ids, user_labels)}

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    (axA, axB), (axC, axD) = axs

    # Panel A: loss
    for cond, runs in L.items():
        x, m, s = _mean_sem_align(runs, "loss")
        if x is None:
            continue
        x_plot = (x + 1) if logxA else x
        # slope on the transformed axes requested by flags
        m_slope = _fit_slope(x_plot, m, logx=logxA, logy=logyA)
        base_label = (
            label_map.get(cond, _short_label_from_root(cond))
            if label_map
            else _short_label_from_root(cond)
        )
        label = f"{base_label} (m={m_slope:.3g})"
        axA.plot(x_plot, m, lw=2, label=label)
        axA.fill_between(x_plot, m - s, m + s, alpha=0.2)
    _style(
        axA, fontsize, title="Loss vs epoch", xlabel="epoch", ylabel="loss", legend=True
    )
    if logxA:
        axA.set_xscale("log")
    if logyA:
        axA.set_yscale("log")

    # Panel B: grad L2 (post)
    for cond, runs in G.items():
        x, m, s = _mean_sem_align(runs, "grad_L2_post")
        if x is None:
            continue
        x_plot = (x + 1) if logxB else x
        m_slope = _fit_slope(x_plot, m, logx=logxB, logy=logyB)
        base_label = (
            label_map.get(cond, _short_label_from_root(cond))
            if label_map
            else _short_label_from_root(cond)
        )
        label = f"{base_label} (m={m_slope:.3g})"
        axB.plot(x_plot, m, lw=2, label=label)
        axB.fill_between(x_plot, m - s, m + s, alpha=0.2)
    _style(
        axB,
        fontsize,
        title="Grad L2 (post) vs epoch",
        xlabel="epoch",
        ylabel="‖∇‖₂",
        legend=True,
    )
    if logxB:
        axB.set_xscale("log")
    if logyB:
        axB.set_yscale("log")

    # Panel C: spectral radius
    for cond, runs in W.items():
        x, m, s = _mean_sem_align(runs, "spectral_radius")
        if x is None:
            continue
        base_label = (
            label_map.get(cond, _short_label_from_root(cond))
            if label_map
            else _short_label_from_root(cond)
        )
        label = f"{base_label} (m={m_slope:.3g})"
        axC.plot(x, m, lw=2, label=label)
        axC.fill_between(x, m - s, m + s, alpha=0.15)
    _style(
        axC,
        fontsize,
        title="Spectral radius ρ(W)",
        xlabel="epoch",
        ylabel="ρ(W)",
        legend=True,
    )

    # Panel D: Frobenius norm
    for cond, runs in W.items():
        x, m, s = _mean_sem_align(runs, "fro_W")
        if x is None:
            continue
        base_label = (
            label_map.get(cond, _short_label_from_root(cond))
            if label_map
            else _short_label_from_root(cond)
        )
        label = f"{base_label} (m={m_slope:.3g})"
        axD.plot(x, m, lw=2, label=label)
        axD.fill_between(x, m - s, m + s, alpha=0.15)
    _style(
        axD,
        fontsize,
        title="Frobenius ‖W‖_F",
        xlabel="epoch",
        ylabel="‖W‖_F",
        legend=True,
    )

    if savepath:
        _savefig(fig, savepath, constrain=False)
    else:
        plt.show()


def fig2_performance_sixpanel(condition_df, savepath=None, fontsize=12):
    """
    Figure 2 (2x3):
      A: Best training loss (↓) vs α0           ['best_loss']
      B: Best open-loop MSE (↓) vs α0           ['mse_open']
      C: Best closed-loop MSE (↓) vs α0         ['mse_free_closed']  # label as "mse_free"
      D: Prediction MSE (↓) vs α0               ['mse_prediction']
      E: Replay MSE (↓) vs α0                   ['mse_replay']
      F: Replay ring R^2 (↑) vs α0              ['ring_decode_R2_replay']
    """
    import matplotlib.pyplot as plt

    if condition_df is None or not {"condition_id", "metric", "mean"}.issubset(
        condition_df.columns
    ):
        print("[SKIP] fig2_performance_sixpanel: condition_summary missing columns.")
        return

    df = condition_df.copy()
    df["shortname"] = df["condition_id"].apply(
        lambda cid: os.path.basename(str(cid)).split("_")[0]
    )

    def _wide_for_family(fam_df):
        csw_mean = _cs_to_wide(fam_df, value_col="mean")
        csw_std = _cs_to_wide(fam_df, value_col="std")
        if csw_mean is None:
            return None, None
        csw_mean["alpha"] = csw_mean["condition_id"].apply(_alpha_from_condition_id)
        csw_mean = csw_mean.dropna(subset=["alpha"]).sort_values("alpha")
        if csw_std is not None and "condition_id" in csw_std.columns:
            csw_std["alpha"] = csw_std["condition_id"].apply(_alpha_from_condition_id)
            csw_std = (
                csw_std.set_index("condition_id")
                .reindex(csw_mean["condition_id"])
                .reset_index()
            )
        return csw_mean, csw_std

    def _plot_metric(ax, fam_df, metric, ylabel, title):
        families = sorted(fam_df["shortname"].unique().tolist())
        plotted_any = False
        for fam in families:
            sub = fam_df[fam_df["shortname"] == fam]
            mW, sW = _wide_for_family(sub)
            if mW is None or metric not in mW.columns:
                continue
            x = mW["alpha"].values
            y = mW[metric].values
            if sW is not None and metric in sW.columns:
                yerr = (
                    sW.set_index("condition_id")[metric]
                    .reindex(mW["condition_id"])
                    .values
                )
                ax.errorbar(
                    x, y, yerr=yerr, fmt="-o", capsize=3, linewidth=1.6, label=fam
                )
            else:
                ax.plot(x, y, "-o", linewidth=1.6, label=fam)
            plotted_any = True
        _style(ax, fontsize, title=title, xlabel=r"$\alpha_0$", ylabel=ylabel)
        if plotted_any:
            ax.legend(frameon=False, fontsize=max(8, fontsize - 2), loc="best")
        else:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=fontsize - 1,
            )

    # Layout
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    (axA, axB, axC), (axD, axE, axF) = axs
    plt.rcParams.update({"font.size": fontsize})

    # Panels
    _plot_metric(axA, df, "best_loss", "best_loss", "A — Best training loss (↓) vs α₀")
    _plot_metric(axB, df, "mse_open", "mse_open", "B — Best open-loop MSE (↓) vs α₀")
    _plot_metric(
        axC, df, "mse_free_closed", "mse_free", "C — Best closed-loop MSE (↓) vs α₀"
    )
    _plot_metric(
        axD, df, "mse_prediction", "mse_prediction", "D — Prediction MSE (↓) vs α₀"
    )
    _plot_metric(axE, df, "mse_replay", "mse_replay", "E — Replay MSE (↓) vs α₀")
    _plot_metric(
        axF,
        df,
        "ring_decode_R2_replay",
        "ring_decode_R2_replay",
        "F — Replay ring $R^2$ (↑) vs α₀",
    )

    fig.suptitle(
        "Performance vs Initial Mixing Ratio (α₀) — 6 metrics", fontsize=fontsize + 2
    )

    if savepath:
        _savefig(fig, savepath, constrain=False)
    else:
        plt.show()


def _family_from_path(p):
    parts = os.path.normpath(str(p)).split(os.sep)

    # Prefer parsing relative to "dense"/"circulant" if present
    start = 0
    for i, seg in enumerate(parts):
        if seg in ("dense", "circulant"):
            start = i + 1
            break

    fam_parts = []
    for seg in parts[start:]:
        # stop when we hit alpha folder or cfg folder/name
        if seg.startswith("alpha") or seg.startswith("cfg"):
            break
        fam_parts.append(seg)

    # fallback: parent folder of cfg if parsing fails
    if not fam_parts and len(parts) >= 2:
        fam_parts = [parts[-2]]

    return "/".join(fam_parts) if fam_parts else "unknown"


def fig3_alpha_time_series(run_level_df, savepath=None, fontsize=12, color_by="family"):
    """
    Figure 3:
      α(t) over training (per-run thin lines; per-condition median±IQR)
    Color/linestyle are initialized here (no dependency on Figure 2’s colors).
    Legends are placed outside (to the right).
    """
    import matplotlib.pyplot as plt

    if run_level_df is None or run_level_df.empty:
        print("[SKIP] fig3_alpha_time_series: no run_level.csv available")
        return

    df_rl = run_level_df.copy()
    if "condition_id" not in df_rl.columns:
        if "condition_root" in df_rl.columns:
            df_rl["condition_id"] = df_rl["condition_root"].apply(
                _infer_condition_id_from_root
            )
        else:
            df_rl["condition_id"] = "condition"

    have_cols = {"fro_S_offline", "fro_A_offline"} <= set(df_rl.columns)
    tcol = _pick_time_col_runlevel(df_rl) if have_cols else None
    if not have_cols or tcol is None:
        print(
            "[SKIP] fig3_alpha_time_series: missing fro_S_offline/fro_A_offline or time col"
        )
        return

    # Own parsing (independent of Fig 2)
    # Use condition_root if present; otherwise fall back to condition_id
    if "condition_root" in df_rl.columns:
        df_rl["family"] = df_rl["condition_root"].apply(_family_from_path)
    else:
        df_rl["family"] = df_rl["condition_id"].apply(_family_from_path)
    df_rl["alpha0"] = df_rl["condition_id"].apply(_alpha_from_condition_id)

    # Distinct line styles by α0 present
    ls_map, alpha_levels = _linestyle_map_for_alphas(df_rl["alpha0"].unique())

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.8))
    plt.rcParams.update({"font.size": fontsize})

    def _alpha_label_from_any(row):
        # Prefer condition_root, but fall back to condition_id
        root = row.get("condition_root", None)
        if isinstance(root, str):
            parts = [p for p in root.replace("\\", "/").split("/") if p]
            alpha_parts = [p for p in parts if p.startswith("alpha")]
            if alpha_parts:
                return alpha_parts[-1]

        cid = row.get("condition_id", None)
        if cid is not None:
            m = re.search(r"(alpha\d+p\d{2})", str(cid))
            if m:
                return m.group(1)

        return None

    if color_by == "alpha":
        df_rl["color_label"] = df_rl.apply(_alpha_label_from_any, axis=1)
        # if still missing, fall back to family
        df_rl["color_label"] = df_rl["color_label"].fillna(df_rl["family"])
        legend_title = "alpha"
    else:
        df_rl["color_label"] = df_rl["family"]
        legend_title = "family"

    # Use Matplotlib's default color cycle, assign per label (first-seen order)
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    labels_in_order = list(dict.fromkeys(df_rl["color_label"].tolist()))
    label_color = {
        lab: default_colors[i % len(default_colors)]
        for i, lab in enumerate(labels_in_order)
    }

    # Per-run thin traces
    for _, g in df_rl.groupby(
        ["condition_id"] + (["run_id"] if "run_id" in df_rl.columns else [])
    ):
        tcol_g, t, a = _alpha_series_from_runlevel(g)
        if tcol_g is None:
            continue
        order = np.argsort(t)
        lab = g["color_label"].iloc[0]
        a0 = float(g["alpha0"].iloc[0]) if pd.notna(g["alpha0"].iloc[0]) else None
        linestyle = "-" if color_by == "alpha" else ls_map.get(a0, "-")
        ax.plot(
            t[order],
            a[order],
            lw=1.0,
            alpha=0.30,  # optional: make them readable
            linestyle=linestyle,
            color=label_color.get(lab, None),
            label=lab,
        )

    # Collapse labels to unique families (avoid duplicates)
    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    # Replace legend with unique family colors (outside)
    leg_colors = ax.legend(
        uniq.values(),
        uniq.keys(),
        title=legend_title,
        frameon=False,
        fontsize=max(8, fontsize - 3),
        title_fontsize=max(9, fontsize - 2),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    # Make legend swatches opaque (not the light per-run alpha)
    for lh in leg_colors.get_lines():
        lh.set_alpha(1.0)

    for cond, cdf in df_rl.groupby("condition_id"):
        lab = cdf["color_label"].iloc[0]
        a0 = float(cdf["alpha0"].iloc[0]) if pd.notna(cdf["alpha0"].iloc[0]) else None
        color = label_color.get(lab, None)
        lstyle = "-" if color_by == "alpha" else ls_map.get(a0, "-")

        xs = np.sort(
            pd.to_numeric(cdf[tcol], errors="coerce").dropna().unique()
        ).astype(float)
        med, q1, q3 = [], [], []
        for x in xs:
            sub = cdf[pd.to_numeric(cdf[tcol], errors="coerce") == x]
            _, _, aa = _alpha_series_from_runlevel(sub)
            aa = aa[np.isfinite(aa)]
            if aa.size == 0:
                med.append(np.nan)
                q1.append(np.nan)
                q3.append(np.nan)
            else:
                med.append(np.nanmedian(aa))
                q1.append(np.nanpercentile(aa, 25))
                q3.append(np.nanpercentile(aa, 75))
        ax.plot(xs, med, lw=2.2, color=color, linestyle=lstyle)
        ax.fill_between(xs, q1, q3, alpha=0.12, color=color)

    _style(
        ax,
        fontsize,
        title="Figure 3 — α(t) from run_level",
        xlabel="epoch",
        ylabel=r"alpha(t) = $\|S\|_F / (\|S\|_F + \|A\|_F)$",
    )

    # Add α0-line-style key as a second legend, also outside on the right
    from matplotlib.lines import Line2D

    style_handles = [
        Line2D([0], [0], color="k", lw=2.6, linestyle=ls_map[a], label=f"α₀={a:.2f}")
        for a in alpha_levels
    ]
    ax.add_artist(leg_colors)
    if color_by != "alpha":
        ax.legend(
            handles=style_handles,
            title="α₀ line style",
            frameon=False,
            fontsize=max(8, fontsize - 3),
            title_fontsize=max(9, fontsize - 2),
            loc="center left",
            bbox_to_anchor=(1.02, 0.10),
        )

    if savepath:
        _savefig(fig, savepath, constrain=False)
    else:
        plt.show()


# -----------------------------------------------------------
# Figure 4 – Emergent traveling-waves and replay dynamics
# -----------------------------------------------------------


def fig4_traveling_waves_and_replay(
    condition_dir: str,
    run_level_df: Optional[pd.DataFrame],
    condition_summary_df: Optional[pd.DataFrame],
    savepath: Optional[str] = None,
    fontsize: int = 12,
    max_units: int = 100,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """
    Top-Left  (A): Replay hidden heatmap (z-scored, units sorted by peak time)
    Top-Right (B): Prediction hidden heatmap (same)
    Bottom-Left (C): Polar angle trajectories (decoded vs true if present)
    Bottom-Right (D): Metrics table from condition_summary (mean±std, only metrics that exist)

    Self-contained: selects the "best" run and loads required files inline:
      *_replay_hidden.npy, *_prediction_hidden.npy,
      *_replay_angles.csv,  *_prediction_angles.csv
    """

    cond = Path(str(condition_dir).strip()).expanduser()

    # --- Locate run_level.csv inside the condition (robust to ./runs vs runs/)
    candidates = [
        cond / "run_level.csv",
        Path("." + str(cond)) / "run_level.csv",
    ]
    candidates += list(cond.glob("*run_level*.csv"))
    rpath = next((p for p in candidates if p.exists()), None)

    print("[fig4] condition_dir:", cond)
    if rpath is None:
        try:
            print("[fig4] ls:", os.listdir(str(cond)))
        except Exception as e:
            print("[fig4] cannot list dir:", e)
        print(f"[SKIP] fig4: no run_level.csv in {cond}")
        return

    # Anchor to the actual CSV parent to avoid path mismatches
    cond = rpath.parent.resolve()

    # --- Read run_level.csv and pick a run (old inline logic, with robust fallbacks)
    df = pd.read_csv(str(rpath))
    if df.empty:
        print(f"[SKIP] fig4: run_level.csv is empty in {cond}")
        return

    def _pick_best_row(frame: pd.DataFrame):
        # Prefer highest replay metric if present; else lowest mse_open; else lowest final_loss
        if (
            "ring_decode_R2_replay" in frame.columns
            and not frame["ring_decode_R2_replay"].isna().all()
        ):
            return frame.sort_values("ring_decode_R2_replay", ascending=False).head(1)
        key = (
            "mse_open"
            if "mse_open" in frame.columns
            else ("last_loss" if "last_loss" in frame.columns else None)
        )
        if key is None or frame[key].isna().all():
            return frame.head(1)
        return frame.sort_values(key, ascending=True).head(1)

    pick = _pick_best_row(df)
    if pick.empty or "run_id" not in pick.columns:
        print(f"[SKIP] fig4: cannot identify run_id from run_level.csv")
        return
    try:
        run_id = int(pick.iloc[0]["run_id"])
    except Exception:
        print(f"[SKIP] fig4: invalid run_id value: {pick.iloc[0]['run_id']!r}")
        return
    run_id = int(pick.iloc[0]["run_id"])

    # --- Resolve checkpoint stub to find trace/angle files
    run_dir = cond / f"run_{run_id:02d}"
    ckpts = list(run_dir.glob("*.pth.tar"))
    if not ckpts:
        print(f"[SKIP] fig4: no checkpoint in {run_dir}")
        return
    ckpt_path = str(ckpts[0])
    stub = (
        ckpt_path[:-8]
        if ckpt_path.endswith(".pth.tar")
        else str(Path(ckpt_path).with_suffix(""))
    )

    # --- Load hidden traces (if present); accept [B,T,N] or [T,N]; cap N
    def _to_TN(arr):
        if arr is None:
            return None
        A = np.asarray(arr)
        if A.ndim == 3:  # [B,T,N]
            A = A[0]
        return A if A.ndim == 2 else None

    def _load_hidden_if(path):
        return _to_TN(np.load(path)) if os.path.exists(path) else None

    rp_h = _load_hidden_if(stub + "_replay_hidden.npy")
    pr_h = _load_hidden_if(stub + "_prediction_hidden.npy")

    if rp_h is None and pr_h is None:
        print(f"[SKIP] fig4: no hidden traces in {run_dir}")
        return
    if rp_h is not None and rp_h.shape[1] > max_units:
        rp_h = rp_h[:, :max_units]
    if pr_h is not None and pr_h.shape[1] > max_units:
        pr_h = pr_h[:, :max_units]

    # --- Z-score per unit and sort by time-of-peak (wave reveal)
    def _zsort(H):
        if H is None:
            return None, None
        Z = (H - H.mean(axis=0, keepdims=True)) / (H.std(axis=0, keepdims=True) + 1e-8)
        peak_t = np.argmax(Z, axis=0)
        order = np.argsort(peak_t)
        return Z[:, order].T, order  # [units,time], permutation

    rp_z, _ = _zsort(rp_h)
    pr_z, _ = _zsort(pr_h)

    # --- Angle CSV reading (robust to simple csv)
    def _read_angles_csv(path):
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path)
            return df
        except Exception:
            # fallback: bare csv reader
            with open(path, "r") as f:
                rows = list(csv.reader(f))
            if not rows:
                return None
            hdr = rows[0]
            vals = rows[1:]
            try:
                arr = np.array(vals, dtype=float)
                return pd.DataFrame(arr, columns=hdr[: arr.shape[1]])
            except Exception:
                return None

    rp_ang = _read_angles_csv(stub + "_replay_angles.csv")
    pr_ang = _read_angles_csv(stub + "_prediction_angles.csv")

    def _first_angle_col(df):
        if df is None or df.empty:
            return None
        prefer = ["theta_pred", "theta_true", "theta", "angle", "phi", "ang", "phase"]
        lower = {c.lower(): c for c in df.columns}
        for k in prefer:
            if k in lower:
                return lower[k]
        # fallback: first numeric column
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
        return None

    # --- Build figure (2x2)
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])
    axA = fig.add_subplot(gs[0, 0])  # replay heatmap
    axB = fig.add_subplot(gs[0, 1])  # prediction heatmap
    axC = fig.add_subplot(gs[1, 0], projection="polar")  # polar
    axD = fig.add_subplot(gs[1, 1])  # table

    plt.rcParams.update({"font.size": fontsize})

    # --- Panel A: replay heatmap
    def _heatmap(ax, Z, title):
        if Z is None:
            ax.axis("off")
            ax.set_title(f"{title} (missing)", fontsize=fontsize)
            return
        vmin_eff = np.min(Z) if vmin is None else vmin
        vmax_eff = np.max(Z) if vmax is None else vmax
        im = ax.imshow(
            Z,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=vmin_eff,
            vmax=vmax_eff,
        )
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel("time")
        ax.set_ylabel("units (sorted)")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.set_ylabel("activation (z)", rotation=270, labelpad=15)
        cb.ax.tick_params(labelsize=max(8, fontsize - 3))

    _heatmap(axA, rp_z, "A — Replay hidden (z; peak-sorted)")
    _heatmap(axB, pr_z, "B — Prediction hidden (z; peak-sorted)")

    # --- Panel C: polar trajectories (decoded vs true if available)
    def _plot_polar_angles(ax, df, label, style="-"):
        if df is None or df.empty:
            return None
        t = df["t"].values if "t" in df.columns else np.arange(len(df))
        # prefer explicit decoded/true; else pick first numeric column as angle
        a_pred_col = (
            "theta_pred" if "theta_pred" in df.columns else _first_angle_col(df)
        )
        if a_pred_col is None:
            return None
        (ln,) = ax.plot(
            df[a_pred_col].values, t, linestyle=style, linewidth=1.8, label=label
        )
        # overlay true in same hue if present
        if "theta_true" in df.columns:
            ax.plot(
                df["theta_true"].values,
                t,
                linestyle="--",
                linewidth=1.6,
                label=f"{label} (true)",
                color=ln.get_color(),
            )
        return ln

    any_line = False
    if _plot_polar_angles(axC, rp_ang, "Replay θ̂(t)", "-") is not None:
        any_line = True
    if _plot_polar_angles(axC, pr_ang, "Prediction θ̂(t)", "-") is not None:
        any_line = True

    if any_line:
        axC.set_title("C — Polar angle trajectories", fontsize=fontsize)
        axC.legend(
            loc="center left",
            bbox_to_anchor=(1.12, 0.5),
            frameon=False,
            fontsize=max(8, fontsize - 2),
        )
    else:
        axC.set_title("C — Polar angle trajectories (missing)", fontsize=fontsize)

    # --- Panel D: compact metrics table (reuse your existing builder)
    axD.axis("off")
    prefer_metrics = [
        # REPLAY
        "ring_decode_R2_replay",
        "angle_error_R_replay",
        "residual_lag1_autocorr_replay",
        # PREDICTION
        "mse_prediction",
        "ring_decode_R2_prediction",
        "angle_error_R_prediction",
        "mean_corr_prediction",
    ]
    # Filter to this condition where possible
    cs_sub = condition_summary_df
    if cs_sub is not None and "condition_id" in cs_sub.columns:
        cid = _infer_condition_id_from_root(str(cond))
        cs_sub = cs_sub[cs_sub["condition_id"] == cid]
    table_df = _build_metrics_table_from_condition_summary(cs_sub, prefer_metrics)

    if table_df is None or table_df.empty:
        axD.set_title(
            "D — Metrics table (no matching metrics found)", fontsize=fontsize
        )
    else:
        axD.set_title("D — Replay & Prediction Metrics (mean±std)", fontsize=fontsize)
        tbl = axD.table(
            cellText=table_df.values, colLabels=["metric", cond.name], loc="center"
        )

        tbl.auto_set_font_size(False)
        tbl.set_fontsize(max(7, fontsize - 5))
        tbl.scale(1.0, 1.2)
        # bold header
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_text_props(weight="bold")

    fig.suptitle(
        f"Fig 4 — Waves & Polar (cond: {cond.name}, run: {run_id})",
        fontsize=fontsize + 2,
    )
    if savepath:
        _savefig(fig, savepath, constrain=False)
    else:
        plt.show()
    return fig


# ==============================
# Figure 5 helpers (checkpoints + cache + heatmap permutation)
# ==============================


def _list_checkpoints(run_dir: str):
    pats = ["*.pth.tar", "*.pt", "*.ckpt"]
    found = []
    for p in pats:
        found.extend(glob.glob(os.path.join(run_dir, p)))

    def _key(path):
        m = re.search(r"epoch[_\-]?(\d+)", os.path.basename(path))
        return (0, int(m.group(1))) if m else (1, 10**12)

    return sorted(found, key=_key)


def _load_W_history_from_ckpt(ckpt_path: str):
    try:
        import torch

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[SKIP] fig5: failed to load ckpt {ckpt_path}: {e}")
        return None, None
    weights = ckpt.get("weights", {})
    hist = weights.get("W_hh_history", None)
    epochs = ckpt.get("snapshot_epochs", None)

    if hist is None or epochs is None:
        return None, None
    Ws = []
    for w in hist:
        if hasattr(w, "detach"):
            w = w.detach().cpu().numpy()
        else:
            w = np.asarray(w)
        if w.dtype == np.float16:
            w = w.astype(np.float32, copy=False)
        if w.ndim == 2 and w.shape[0] == w.shape[1]:
            Ws.append(w)
    if not Ws:
        return None, None
    return list(epochs), np.stack(Ws, axis=0)  # [T,H,H]


def _select_time_indices_generic(times_list, spec: str):
    if not times_list:
        return []
    want = [s.strip().lower() for s in (spec or "last").split(",")]
    idxs, n = set(), len(times_list)
    for w in want:
        if w == "all":
            return list(range(n))
        if w == "first":
            idxs.add(0)
        elif w == "middle":
            idxs.add(n // 2)
        elif w == "last":
            idxs.add(n - 1)
        else:
            try:
                tnum = int(w)
                if tnum in times_list:
                    idxs.add(times_list.index(tnum))
                else:
                    arr = np.asarray(times_list, dtype=float)
                    idxs.add(int(np.argmin(np.abs(arr - tnum))))
            except Exception:
                pass
    return sorted(idxs)


def _load_hidden_for_perm(run_dir: str):
    # Prefer replay hidden, then prediction hidden
    cands = []
    cands.extend(glob.glob(os.path.join(run_dir, "*replay*hidden*.npy")))
    cands.extend(glob.glob(os.path.join(run_dir, "*prediction*hidden*.npy")))
    for f in sorted(cands):
        try:
            arr = np.load(f)
            A = np.asarray(arr)
            if A.ndim == 3:  # [B,T,N]
                A = A[0]
            if A.ndim == 2:
                return A  # [T,N]
        except Exception:
            continue
    return None


def _peak_time_sort_perm(H_TN: np.ndarray):
    H = (H_TN - H_TN.mean(axis=0, keepdims=True)) / (
        H_TN.std(axis=0, keepdims=True) + 1e-8
    )
    peak_t = np.argmax(H, axis=0)  # [N]
    return np.argsort(peak_t).astype(int)


def _get_or_make_perm(run_dir: str):
    cache_dir = os.path.join(run_dir, "analysis")
    os.makedirs(cache_dir, exist_ok=True)
    fperm = os.path.join(cache_dir, "perm_peak.npy")
    if os.path.exists(fperm):
        try:
            p = np.load(fperm)
            if p.ndim == 1:
                return p.astype(int)
        except Exception:
            pass
    H = _load_hidden_for_perm(run_dir)
    if H is None:
        print(
            f"[WARN] fig5: no hidden traces found to compute permutation in {run_dir}; leaving W unsorted."
        )
        return None
    perm = _peak_time_sort_perm(H)
    np.save(fperm, perm)
    return perm


def _apply_perm(W: np.ndarray, perm: Optional[np.ndarray]):
    return W if perm is None else W[np.ix_(perm, perm)]


def _ensure_weight_cache(run_dir: str):
    p = os.path.join(run_dir, "analysis", "weights")
    os.makedirs(p, exist_ok=True)
    return p


def _find_or_make_sorted_W_snapshot(run_dir: str, epoch_val: int):
    """
    Returns path to cached sorted W for this run+epoch, creating it from checkpoint if needed.
    Cache file: <run_dir>/analysis/weights/Wsorted_epoch{epoch:06d}.npy
    """
    cache_dir = _ensure_weight_cache(run_dir)
    fout = os.path.join(cache_dir, f"Wsorted_epoch{int(epoch_val):06d}.npy")
    if os.path.exists(fout):
        return fout

    ckpts = _list_checkpoints(run_dir)
    if not ckpts:
        print(f"[SKIP] fig5: no checkpoints in {run_dir}")
        return None
    epochs, Ws = None, None
    for c in reversed(ckpts):  # latest first
        epochs, Ws = _load_W_history_from_ckpt(c)
        if epochs is not None:
            break
    if epochs is None:
        print(f"[SKIP] fig5: no W_hh_history in checkpoints for {run_dir}")
        return None

    arr = np.asarray(epochs, dtype=float)
    i = int(np.argmin(np.abs(arr - float(epoch_val))))
    W = Ws[i]
    perm = _get_or_make_perm(run_dir)
    W_sorted = _apply_perm(W, perm)

    try:
        np.save(fout, W_sorted)
    except Exception as e:
        print(f"[WARN] fig5: failed to cache {fout}: {e}")
    return fout


# -----------------------------------------------------------
# Figure 5 – Mean weight trace & eigenspectrum (per condition)
# -----------------------------------------------------------


def fig5_weights_and_spectrum_from_checkpoints(
    condition_root: str,
    time_spec: str = "last",
    savepath: Optional[
        str
    ] = None,  # treated as a *base* path; we append suffixes per timepoint
    fontsize: int = 12,
):
    """
    One PNG per requested timepoint.
    Each PNG has 3 columns: [W_sorted heatmap, Trace mean±sd, Eigenspectrum].
    Uses checkpoints as source of truth, peak-time permutation, and cached Wsorted.
    """
    run_dirs = [
        d
        for d in sorted(glob.glob(os.path.join(condition_root, "run_*")))
        if os.path.isdir(d)
    ]
    if not run_dirs:
        print("[SKIP] fig5: no run_* under", condition_root)
        return None

    # Discover epochs per run
    per_run_epochs = {}
    for rd in run_dirs:
        ckpts = _list_checkpoints(rd)
        epochs, Ws = (None, None)
        for c in reversed(ckpts):
            epochs, Ws = _load_W_history_from_ckpt(c)
            if epochs is not None:
                break
        if not epochs:  # handles None or empty
            print(f"[SKIP] fig5: no W history in {rd}")
            continue
        per_run_epochs[rd] = list(epochs)
    if not per_run_epochs:
        print("[SKIP] fig5: no runs with W history in", condition_root)
        return None

    # Reference run for labeling
    ref_run = next(iter(per_run_epochs))
    ref_times = per_run_epochs[ref_run]

    # Interpret time_spec: "all" -> first,middle,last  (per your request)
    spec_norm = (time_spec or "last").strip().lower()
    if spec_norm == "all":
        want_labels = ["first", "middle", "last"]
    else:
        want_labels = [s.strip() for s in spec_norm.split(",") if s.strip()]

    # Resolve each requested label or numeric into an index *and* a label suffix
    def _index_and_suffix(label: str):
        if label in ("first", "middle", "last"):
            if label == "first":
                return 0, "first"
            if label == "middle":
                return len(ref_times) // 2, "middle"
            if label == "last":
                return len(ref_times) - 1, "last"
        # numeric epoch or index-like token
        try:
            tnum = int(label)
            # snap to nearest epoch value of the reference run
            arr = np.asarray(ref_times, dtype=float)
            idx = int(np.argmin(np.abs(arr - float(tnum))))
            return idx, f"t{tnum}"
        except Exception:
            # fallback: try to parse like "idx12"
            m = re.match(r"idx(\d+)$", label)
            if m:
                idx = min(int(m.group(1)), max(0, len(ref_times) - 1))
                return idx, f"idx{idx}"
        return None, None

    # Utility: build per-timepoint filename by inserting suffix before extension
    def _with_suffix(path: str, suffix: str) -> str:
        if not path:
            # default name if no savepath given
            base = os.path.join(
                "./figs",
                f"fig5_weights_and_spectrum_{_short_label_from_root(condition_root)}.png",
            )
        else:
            base = path
        root, ext = os.path.splitext(base)
        return f"{root}_{suffix}{ext or '.png'}"

    made_any = False
    for req in want_labels:
        idx, suffix = _index_and_suffix(req)
        if idx is None:
            print(f"[SKIP] fig5: could not interpret time token {req!r}")
            continue
        if not (0 <= idx < len(ref_times)):
            print(f"[SKIP] fig5: index {idx} out of range for {condition_root}")
            continue

        epoch_label = ref_times[idx]  # for titles

        # Collect cached sorted W for each run at this row’s per-run index
        Ws_sorted = []
        for rd, epochs in per_run_epochs.items():
            if not epochs:
                continue
            use_idx = idx if idx < len(epochs) else (len(epochs) - 1)
            e = epochs[use_idx]
            fW = _find_or_make_sorted_W_snapshot(rd, e)
            if fW is None or not os.path.exists(fW):
                continue
            try:
                W = np.load(fW)
                if W.ndim == 2 and W.shape[0] == W.shape[1]:
                    Ws_sorted.append(W)
            except Exception:
                continue

        if not Ws_sorted:
            print(f"[SKIP] fig5 {suffix}: no sorted Ws for epoch ~{epoch_label}")
            continue

        Ws_sorted = np.stack(Ws_sorted, axis=0)  # [R,H,H]
        if Ws_sorted.dtype == np.float16:
            Ws_sorted = Ws_sorted.astype(np.float32, copy=False)
        Wm = Ws_sorted.mean(axis=0)

        # Build a 1-row, 3-column figure for this timepoint
        fig, (axW, axL, axR) = plt.subplots(1, 3, figsize=(16, 4.0))
        plt.rcParams.update({"font.size": fontsize})

        # --- Heatmap (Wm)
        vmax = float(np.max(np.abs(Wm))) if Wm.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
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
        cb.ax.tick_params(labelsize=max(8, fontsize - 3))
        _style(
            axW,
            fontsize,
            title=f"W (sorted) — epoch {epoch_label}",
            xlabel="Presynaptic (sorted)",
            ylabel="Postsynaptic (sorted)",
        )

        # --- Diagonal-offset trace mean ± sd
        offs = np.arange(-(Wm.shape[0] - 1), Wm.shape[0], dtype=int)
        traces = np.stack(
            [
                np.array([np.trace(Ws_sorted[r], k) for k in offs], dtype=float)
                for r in range(Ws_sorted.shape[0])
            ],
            axis=0,
        )  # [R,K]
        tr_mean = traces.mean(axis=0)
        tr_sd = (
            traces.std(axis=0, ddof=1)
            if traces.shape[0] > 1
            else np.zeros_like(tr_mean)
        )

        axL.plot(offs, tr_mean, lw=2, label="mean")
        if traces.shape[0] > 1:
            axL.fill_between(
                offs, tr_mean - tr_sd, tr_mean + tr_sd, alpha=0.2, label="±1 sd"
            )
        axL.axhline(0, ls="--", lw=1, alpha=0.6, color="k")
        axL.axvline(0, ls=":", lw=1, alpha=0.6, color="k")
        _style(
            axL,
            fontsize,
            title="Trace (mean±sd)",
            xlabel="Diagonal offset k",
            ylabel="∑ diag_k W",
        )
        axL.legend(frameon=False, fontsize=max(8, fontsize - 2))

        # --- Eigenspectrum (mean + per-run cloud)
        eig_mean = np.linalg.eigvals(Wm.astype(np.float64, copy=False))
        evs = np.concatenate(
            [
                np.linalg.eigvals(Ws_sorted[r].astype(np.float64, copy=False))
                for r in range(Ws_sorted.shape[0])
            ]
        )
        axR.scatter(evs.real, evs.imag, s=6, alpha=0.12, label="runs")
        axR.scatter(eig_mean.real, eig_mean.imag, s=10, alpha=0.9, label="mean(W)")
        th = np.linspace(0, 2 * np.pi, 512)
        axR.plot(
            np.cos(th), np.sin(th), ls="--", lw=1, alpha=0.4, color="k", label="|λ|=1"
        )
        axR.axhline(0, color="k", lw=0.5)
        axR.axvline(0, color="k", lw=0.5)
        axR.set_aspect("equal", adjustable="box")
        rho = float(np.max(np.abs(eig_mean))) if eig_mean.size else float("nan")
        _style(
            axR,
            fontsize,
            title=f"Eigenspectrum (ρ≈{rho:.3f})",
            xlabel="Re(λ)",
            ylabel="Im(λ)",
            legend=True,
        )

        # Save (or show) this timepoint’s figure
        if savepath:
            out_path = _with_suffix(savepath, suffix)
            _savefig(fig, out_path, constrain=False)
        else:
            plt.show()
        made_any = True

    if not made_any:
        print("[SKIP] fig5: no figures produced for", condition_root)
    return None


# ---------------------------
# CLI
# ---------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate figures from aggregated metrics")
    p.add_argument("--figdir", type=str, default="./figs", help="Where to save PNGs")
    p.add_argument(
        "--fontsize", type=int, default=12, help="Base font size for all plots"
    )
    p.add_argument(
        "--condition_regex",
        type=str,
        default=None,
        help="Regex to filter conditions (applied to condition_id)",
    )
    p.add_argument(
        "--run_regex",
        type=str,
        default=None,
        help="Regex to filter runs (applied to run_id)",
    )
    p.add_argument(
        "--just",
        type=str,
        default="",
        help="Comma list of figures to render (1..7). Empty=all.",
    )
    p.add_argument(
        "--conditions",
        type=str,
        default="",
        help="Comma-separated list of condition root folders (each has run_level.csv etc.)",
    )
    p.add_argument(
        "--cond_glob",
        nargs="+",  # accept 1..N globs
        default=[],
        help="One or more globs for condition roots (space- or comma-separated). Example: --cond_glob './runs/.../sym*/shiftmh_*' './runs/.../sym*/shiftcycmh_*'",
    )
    p.add_argument("--fig1_logxA", action="store_true", help="Fig1A: log x-axis")
    p.add_argument("--fig1_logyA", action="store_true", help="Fig1A: log y-axis")
    p.add_argument("--fig1_logxB", action="store_true", help="Fig1B: log x-axis")
    p.add_argument("--fig1_logyB", action="store_true", help="Fig1B: log y-axis")
    p.add_argument(
        "--fig1_labels",
        type=str,
        default="",
        help="Comma-separated custom labels for Figure 1 curves (overrides auto labels). "
        "Count must match number of condition roots.",
    )

    p.add_argument(
        "--figtag",
        type=str,
        default="",
        help="Append this tag to output figure filenames",
    )
    p.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Minimum value for color scale in Figure 4 heatmaps",
    )
    p.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Maximum value for color scale in Figure 4 heatmaps",
    )
    p.add_argument(
        "--fig5_time",
        type=str,
        default="last",
        help="Figure 5 timepoint(s): first|middle|last|all or comma-list (e.g., 'first,last' or '100,500').",
    )

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    suffix = f"_{args.figtag}" if args.figtag else ""

    # Build list of condition roots if provided
    cond_roots = []
    if args.conditions:
        cond_roots.extend([s.strip() for s in args.conditions.split(",") if s.strip()])
    patterns = []
    if args.cond_glob:
        # args.cond_glob is a list when provided; support comma-separated inside each
        raw_entries = (
            args.cond_glob
            if isinstance(args.cond_glob, (list, tuple))
            else [args.cond_glob]
        )
        for entry in raw_entries:
            # split on commas but keep simple whitespace-only entries too
            for part in [s.strip() for s in re.split(r",", entry) if s.strip()]:
                patterns.append(part)
        for pat in patterns:
            cond_roots.extend(sorted(glob.glob(pat)))

    # --- Fig3 special-case: if exactly 1 cond_glob pattern includes '*' and expands to >1 condition,
    #     color/legend by alpha folder (alpha0p00, alpha0p10, ...)
    fig3_color_by = "family"
    if len(patterns) == 1 and "*" in patterns[0] and len(cond_roots) > 1:
        fig3_color_by = "alpha"

    if not cond_roots:
        print("[ERROR] Please provide condition roots via --conditions or --cond_glob.")
        return
    rl, cs, off, ev = load_many_conditions(cond_roots)

    rl = _filter_df(rl, args.condition_regex, args.run_regex)
    cs = _filter_df(cs, args.condition_regex, args.run_regex)
    off = _filter_df(off, args.condition_regex, args.run_regex)

    _ensure_dir(args.figdir)
    requested = (
        set([s.strip() for s in args.just.split(",") if s.strip()])
        if args.just
        else set()
    )

    def want(k):
        return (not requested) or (str(k) in requested)

    if want(1):
        fig1_path = os.path.join(args.figdir, f"fig1_training_dynamics{suffix}.png")
        fig1_training_dynamics(
            cond_roots,
            fig1_path,
            fontsize=args.fontsize,
            logxA=args.fig1_logxA,
            logyA=args.fig1_logyA,
            logxB=args.fig1_logxB,
            logyB=args.fig1_logyB,
            labels=args.fig1_labels,
        )

    if want(2):
        fig2_path = os.path.join(
            args.figdir, f"fig2_performance_vs_alpha_6panel{suffix}.png"
        )
        fig2_performance_sixpanel(
            condition_df=cs,
            savepath=fig2_path,
            fontsize=args.fontsize,
        )

    if want(3):
        fig3_path = os.path.join(args.figdir, f"fig3_alpha_time_series{suffix}.png")
        fig3_alpha_time_series(
            run_level_df=rl,
            savepath=fig3_path,
            fontsize=args.fontsize,
            color_by=fig3_color_by,
        )

    if want(4):
        for root in cond_roots:
            fig4_path = os.path.join(
                args.figdir, f"fig4_{_short_label_from_root(root)}{suffix}.png"
            )
            fig4_traveling_waves_and_replay(
                condition_dir=root,
                run_level_df=rl,
                condition_summary_df=cs,
                savepath=fig4_path,
                fontsize=args.fontsize,
                max_units=100,
                vmin=args.vmin,
                vmax=args.vmax,
            )
    if want(5):
        for root in cond_roots:
            short = _short_label_from_root(root)  # e.g., identity_n100_fro
            per_suffix = f"_{args.figtag}_{short}" if args.figtag else f"_{short}"
            fig5_path = os.path.join(
                args.figdir, f"fig5_weights_and_spectrum{per_suffix}.png"
            )
            fig5_weights_and_spectrum_from_checkpoints(
                condition_root=root,
                time_spec=args.fig5_time,
                savepath=fig5_path,
                fontsize=args.fontsize,
            )


if __name__ == "__main__":
    main()
