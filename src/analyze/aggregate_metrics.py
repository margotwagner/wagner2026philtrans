#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate per-run CSVs into condition-level tables.

Inputs (per condition root folder):
- <condition_root>/<stub>_eval.csv                 (written by evaluate.py; one or more modes/rows)
- <condition_root>/run_XX/<stub>_train_summary.csv (written by evaluate.py for each run)
- <condition_root>/run_XX/<stub>_offline_metrics.csv (written by offline_metrics.py for each run)

Outputs (written back into each condition root):
- <condition_root>/run_level.csv          (one row per run with merged [train,eval,offline] metrics)
- <condition_root>/condition_summary.csv  (mean & std across runs for numeric columns)

Usage:
    python aggregate_metrics.py --root /runs/ElmanRNN/random-init/random_n100/
    python aggregate_metrics.py --root /runs/ElmanRNN/mh-variants/shifted-cyc/frobenius/sym0p90/
    # Multiple conditions:
    python aggregate_metrics.py --root /runs/ElmanRNN/random-init/random_n100/ \
                                --root /runs/ElmanRNN/mh-variants/shifted-cyc/frobenius/sym0p90/
"""

from __future__ import print_function
import argparse
import os
import glob
import json
import csv
import re

import numpy as np
import pandas as pd


def _parse_run_id(run_dir):
    """Extract integer run id from a folder name like 'run_03'. Return None if not found."""
    if not isinstance(run_dir, str):
        return None
    m = re.search(r"run_(\d+)", run_dir)
    return int(m.group(1)) if m else None


def _is_dir(p):
    return os.path.isdir(p)


def _load_csv_safe(path, **read_kwargs):
    if not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path, **read_kwargs)
    except Exception as e:
        print("[WARN] Failed to read CSV:", path, "error:", repr(e))
        return None


def _find_runs(condition_root):
    # run folders named run_XX
    return sorted(
        [d for d in glob.glob(os.path.join(condition_root, "run_*")) if _is_dir(d)]
    )


def _guess_stub_from_any_csv(csv_path):
    # e.g., /.../random_n100_eval.csv  -> stub = random_n100
    base = os.path.basename(csv_path)
    if base.endswith("_eval.csv"):
        return base[: -len("_eval.csv")]
    if base.endswith("_train_summary.csv"):
        return base[: -len("_train_summary.csv")]
    if base.endswith("_offline_metrics.csv"):
        return base[: -len("_offline_metrics.csv")]
    # fallback: remove extension
    return os.path.splitext(base)[0]


def _collect_eval_rows(condition_root):
    """
    Returns DataFrame of eval rows, with a 'run_dir' column parsed from 'ckpt' path if present.
    If multiple eval CSVs exist, we concat them. We also make one wide row per run by pivoting modes.
    """
    eval_csvs = sorted(glob.glob(os.path.join(condition_root, "*_eval.csv")))
    if not eval_csvs:
        return None, None

    dfs = []
    stub = None
    for p in eval_csvs:
        df = _load_csv_safe(p)
        if df is None or df.empty:
            continue
        if stub is None:
            stub = _guess_stub_from_any_csv(p)
        # Derive run_dir from ckpt path if present
        if "ckpt" in df.columns:
            # ckpt should contain ".../run_XX/<stub>.pth.tar"
            run_dirs = []
            for s in df["ckpt"].astype(str).tolist():
                parts = s.split("/")
                r = None
                for i in range(len(parts)):
                    if parts[i].startswith("run_"):
                        r = parts[i]
                        break
                run_dirs.append(r)
            df["run_dir"] = run_dirs
            df["run_id"] = df["run_dir"].apply(_parse_run_id)

        else:
            # If no ckpt column, try to infer run_dir as NaN (will be merged sparsely)
            df["run_dir"] = np.nan
            df["run_id"] = np.nan

        # If there's a 'mode' column (open_loop, replay, prediction, closed_loop), we keep it for pivot
        dfs.append(df)

    if not dfs:
        return None, None

    all_eval = pd.concat(dfs, ignore_index=True)

    # If there's a "mode" column, pivot to get per-run wide metrics
    # We'll average duplicates per (run_dir, mode) if any.
    if "mode" in all_eval.columns and "run_dir" in all_eval.columns:
        # numeric cols only to aggregate; keep first for non-numerics later if needed
        num_cols = all_eval.select_dtypes(include=[np.number]).columns.tolist()
        # Aggregate repeated rows per (run_dir, mode)
        grouped = all_eval.groupby(["run_dir", "mode"], as_index=False)[num_cols].mean()

        # Pivot: columns like mse_open_loop, angle_R_prediction, etc
        wide = grouped.pivot(index="run_dir", columns="mode", values=num_cols)
        # Flatten MultiIndex columns: (metric, mode) -> f"{metric}_{mode}"
        wide.columns = ["%s_%s" % (c[0], c[1]) for c in wide.columns.to_flat_index()]
        wide = wide.reset_index()
        # Add numeric run_id extracted from run_dir
        if "run_dir" in wide.columns:
            wide["run_id"] = wide["run_dir"].apply(_parse_run_id)
        return wide, stub

    # No mode column; we just keep as-is (but ensure run_dir present).
    return all_eval, stub


def _collect_train_summary(condition_root):
    runs = _find_runs(condition_root)
    rows = []
    stub = None
    for r in runs:
        csvs = glob.glob(os.path.join(r, "*_train_summary.csv"))
        if not csvs:
            continue
        p = sorted(csvs)[-1]  # expect one per run; take last if multiple
        if stub is None:
            stub = _guess_stub_from_any_csv(p)
        df = _load_csv_safe(p)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["run_dir"] = os.path.basename(r)
        df["run_id"] = df["run_dir"].apply(_parse_run_id)
        rows.append(df)

    if not rows:
        return None, None

    train_df = pd.concat(rows, ignore_index=True)
    return train_df, stub


def _collect_offline_metrics(condition_root):
    runs = _find_runs(condition_root)
    rows = []
    stub = None
    for r in runs:
        csvs = glob.glob(os.path.join(r, "*_offline_metrics.csv"))
        if not csvs:
            continue
        p = sorted(csvs)[-1]
        if stub is None:
            stub = _guess_stub_from_any_csv(p)
        df = _load_csv_safe(p)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["run_dir"] = os.path.basename(r)
        df["run_id"] = df["run_dir"].apply(_parse_run_id)
        rows.append(df)

    if not rows:
        return None, None

    off_df = pd.concat(rows, ignore_index=True)
    return off_df, stub


def _merge_per_run(train_df, eval_df, offline_df):
    """
    Left-join train_df with eval_df and offline_df on 'run_dir'.
    """
    df = None
    if train_df is not None:
        df = train_df.copy()
    if df is None and eval_df is not None:
        df = eval_df.copy()
    if df is None and offline_df is not None:
        df = offline_df.copy()
    if df is None:
        return None

    if eval_df is not None:
        df = pd.merge(df, eval_df, how="left", on="run_dir", suffixes=("", "_eval"))
    if offline_df is not None:
        # avoid column collisions by suffixing offline
        off_cols = [c for c in offline_df.columns if c not in ("run_dir",)]
        rename_map = {
            c: (c if c not in df.columns else c + "_offline") for c in off_cols
        }
        off_df_renamed = offline_df.rename(columns=rename_map)
        df = pd.merge(df, off_df_renamed, how="left", on="run_dir")

    return df


def _write_condition_outputs(condition_root, run_level_df):
    # Write run_level
    run_level_csv = os.path.join(condition_root, "run_level.csv")
    # Reorder: run_dir first if present
    cols = run_level_df.columns.tolist()
    front = []
    for key in ["run_id", "run_dir"]:
        if key in cols:
            cols.remove(key)
            front.append(key)
    cols = front + cols
    run_level_df[cols].to_csv(run_level_csv, index=False)

    # Compute means and stds across runs for numeric cols
    num_cols = run_level_df.select_dtypes(include=[np.number]).columns.tolist()
    # Do not average over run_id
    if "run_id" in num_cols:
        num_cols.remove("run_id")
    if not num_cols:
        # Nothing numeric to summarize; write empty condition_summary with headers
        cond_csv = os.path.join(condition_root, "condition_summary.csv")
        pd.DataFrame({"metric": [], "mean": [], "std": []}).to_csv(
            cond_csv, index=False
        )
        return

    means = run_level_df[num_cols].mean(axis=0)
    stds = run_level_df[num_cols].std(axis=0, ddof=0)
    cond_df = pd.DataFrame(
        {"metric": num_cols, "mean": means.values, "std": stds.values}
    )
    cond_csv = os.path.join(condition_root, "condition_summary.csv")
    cond_df.to_csv(cond_csv, index=False)


def process_condition(condition_root):
    print("[INFO] Condition:", condition_root)

    # Collect the three ingredient tables
    train_df, stub_t = _collect_train_summary(condition_root)
    eval_df, stub_e = _collect_eval_rows(condition_root)
    off_df, stub_o = _collect_offline_metrics(condition_root)

    if train_df is None and eval_df is None and off_df is None:
        print("[WARN] Nothing to aggregate in:", condition_root)
        return

    # Merge per run
    run_level = _merge_per_run(train_df, eval_df, off_df)
    if run_level is None or run_level.empty:
        print("[WARN] Merged run-level table is empty for:", condition_root)
        return

    # Write outputs
    _write_condition_outputs(condition_root, run_level)
    print("[OK] Wrote:", os.path.join(condition_root, "run_level.csv"))
    print("[OK] Wrote:", os.path.join(condition_root, "condition_summary.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        action="append",
        required=True,
        help="Condition root folder (repeatable). Each gets run_level.csv and condition_summary.csv.",
    )
    args = ap.parse_args()

    roots = []
    for r in args.root:
        # allow globbing in the CLI if user passes patterns
        matches = sorted(glob.glob(r))
        roots.extend(matches if matches else [r])

    # De-duplicate & validate
    seen = set()
    final_roots = []
    for r in roots:
        r = os.path.abspath(r)
        if r in seen:
            continue
        seen.add(r)
        if not _is_dir(r):
            print("[WARN] --root is not a directory, skipping:", r)
            continue
        final_roots.append(r)

    if not final_roots:
        print("[ERR] No valid condition roots.")
        return

    for cr in final_roots:
        process_condition(cr)


if __name__ == "__main__":
    main()
