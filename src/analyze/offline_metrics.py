#!/usr/bin/env python3
"""
offline_metrics.py
Compute heavier, offline metrics for recurrent weights saved by Main_clean.py.

Inputs (flexible):
  - A single checkpoint:          --ckpt path/to/run_00/<stub>.pth.tar
  - A directory of runs:          --ckpt path/to/<stub>/     (will recurse)
  - A glob pattern:               --ckpt "runs/**/run_*/**.pth.tar"

Outputs per checkpoint:
  - <stub>_offline_metrics.csv    (one row per W_hh snapshot + init)
  - <stub>_spectra.npz            (eigs [num_snaps, H], svals [num_snaps, H])

Optional:
  --save_SA [first,middle,last or indices]  Save raw S/A matrices (npz)
  --grid_ps  (disabled by default)          Crude pseudospectrum summary on a coarse grid

Designed to be simple & robust for personal use.

Usage examples:
1) Single run
python offline_metrics.py --ckpt runs/ElmanRNN/random-init/random_n100/run_00/random_n100.pth.tar

2) All runs for a condition (directory)
python offline_metrics.py --ckpt runs/ElmanRNN/random-init/random_n100/

3) Glob across conditions
python offline_metrics.py --ckpt "runs/**/run_*/**.pth.tar"

4) Also dump S/A for init, middle, and last; include coarse pseudospectrum
python offline_metrics.py --ckpt runs/ElmanRNN/shift-variants/shiftcyc_n100_fro/ --save_SA first,middle,last --grid_ps
"""

import argparse
import os
import sys
import glob
import math
from typing import List, Tuple, Optional

import numpy as np
import torch
import csv

# ----------------------------
# Helpers (pure NumPy / Torch)
# ----------------------------


def _load_ckpt(path: str):
    """
    Load a training checkpoint saved by Main_clean.py.

    For torch>=2.6, explicitly set weights_only=False so we can load
    older checkpoints that contain pickled objects (NumPy, etc.).
    For older torch versions that don't support weights_only, fall
    back to the original signature.
    """
    path = str(path)
    try:
        # PyTorch ≥ 2.6: override new default weights_only=True
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # PyTorch < 2.6: no weights_only arg
        return torch.load(path, map_location="cpu")


def _fro(x: np.ndarray) -> float:
    return float(np.linalg.norm(x, "fro"))


def _spectral_radius(W: np.ndarray) -> float:
    # max |lambda|
    try:
        ev = np.linalg.eigvals(W)
        return float(np.max(np.abs(ev)))
    except Exception:
        # fallback: power iteration approx for speed/robustness
        # (not exact spectral radius of non-normal matrix, but a rough proxy)
        v = np.random.randn(W.shape[0])
        v = v / (np.linalg.norm(v) + 1e-12)
        for _ in range(50):
            v = W @ v
            n = np.linalg.norm(v) + 1e-12
            v /= n
        # Rayleigh quotient magnitude
        rq = np.linalg.norm(W @ v) / (np.linalg.norm(v) + 1e-12)
        return float(rq)


def _operator_norm_2(W: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Compute 2-operator norm via SVD.

    If SVD fails (e.g., ill-conditioned matrix), fall back to a power
    iteration estimate for the largest singular value and return NaN
    for the smallest / full spectrum.
    """
    try:
        s = np.linalg.svd(W, compute_uv=False)
    except Exception as e:
        # Log once per call; this is offline so stderr noise is okay
        print(
            f"[WARN] _operator_norm_2: SVD failed, using power-iteration fallback: {e}",
            file=sys.stderr,
        )

        # Power iteration for largest singular value
        H = W.shape[0]
        v = np.random.randn(H)
        v /= np.linalg.norm(v) + 1e-12
        for _ in range(50):
            v = W @ v
            n = np.linalg.norm(v) + 1e-12
            v /= n

        smax = float(np.linalg.norm(W @ v) / (np.linalg.norm(v) + 1e-12))
        # We don't have the full spectrum; just return smax and NaN for smin
        return smax, float("nan"), np.array([smax], dtype=np.float32)

    if s.size == 0:
        return 0.0, 0.0, s
    smax = float(s.max())
    smin = float(s.min())
    return smax, smin, s


def _non_normality_commutator(W: np.ndarray) -> float:
    C = W @ W.T - W.T @ W
    return _fro(C)


def _sym_asym(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    S = 0.5 * (W + W.T)
    A = 0.5 * (W - W.T)
    nS, nA = _fro(S), _fro(A)
    return S, A, nS, nA


def _condition_number(svals: np.ndarray, eps: float = 1e-12) -> float:
    if svals.size == 0:
        return float("nan")
    smax = float(svals.max())
    smin = float(svals.min())
    return smax / max(smin, eps)


def _maybe_save_SA(path_npz: str, S: np.ndarray, A: np.ndarray):
    np.savez_compressed(path_npz, S=S.astype(np.float32), A=A.astype(np.float32))


def _coarse_pseudospectrum_summary(
    W: np.ndarray,
    grid: int = 25,
    radius_pad: float = 0.25,
    eps_levels: Tuple[float, ...] = (1e-1, 5e-2, 1e-2),
) -> dict:
    """
    VERY coarse/optional: evaluate min singular value of (zI - W) on a square grid
    around the eigen cloud extent. Returns the fraction of grid points below each eps.
    This is expensive (SVD per grid point). Keep grid small.
    """
    try:
        ev = np.linalg.eigvals(W)
    except Exception:
        return {
            "ps_frac_eps1e-1": None,
            "ps_frac_eps5e-2": None,
            "ps_frac_eps1e-2": None,
        }

    xr = np.max(np.abs(ev.real)) + radius_pad
    yi = np.max(np.abs(ev.imag)) + radius_pad
    xs = np.linspace(-xr, xr, grid)
    ys = np.linspace(-yi, yi, grid)
    H = W.shape[0]
    I = np.eye(H)

    counts = {e: 0 for e in eps_levels}
    total = grid * grid

    for a in xs:
        for b in ys:
            zI_minus_W = (a + 1j * b) * I - W
            s = np.linalg.svd(zI_minus_W, compute_uv=False)
            smin = float(s.min()) if s.size > 0 else float("inf")
            for e in eps_levels:
                if smin < e:
                    counts[e] += 1

    return {
        "ps_frac_eps1e-1": counts[1e-1] / total,
        "ps_frac_eps5e-2": counts[5e-2] / total,
        "ps_frac_eps1e-2": counts[1e-2] / total,
    }


def _nice_stub(ckpt_path: str) -> str:
    # e.g., /.../run_03/<stub>.pth.tar  -> /.../run_03/<stub>
    if ckpt_path.endswith(".pth.tar"):
        return ckpt_path[:-8]
    return ckpt_path


def _pick_indices_to_save(num_snaps: int, spec: str) -> List[int]:
    """
    spec accepts:
      - 'none'                 -> []
      - 'first,middle,last'    -> [0, mid, last] (unique)
      - 'all'                  -> list(range(num_snaps))
      - comma-separated ints   -> explicit indices
    """
    spec = spec.strip().lower()
    if spec == "none":
        return []
    if spec == "all":
        return list(range(num_snaps))
    if spec == "first,middle,last":
        if num_snaps == 0:
            return []
        mid = num_snaps // 2
        cand = [0, mid, num_snaps - 1]
        return sorted(set([i for i in cand if 0 <= i < num_snaps]))
    # explicit list
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok.isdigit():
            i = int(tok)
            if 0 <= i < num_snaps:
                out.append(i)
    return sorted(set(out))


# ----------------------------
# Core runner for one ckpt
# ----------------------------
def process_checkpoint(ckpt_path: str, save_SA: str, do_pseudo: bool):
    ckpt = _load_ckpt(ckpt_path)

    # Pull weights and epochs saved by Main_clean.py
    W0 = ckpt["weights"]["W_hh_init"].astype(np.float32)
    Whist = ckpt["weights"]["W_hh_history"].astype(np.float32)  # [num_snaps, H, H]
    epochs = ckpt.get("snapshot_epochs", list(range(len(Whist))))

    # Stack init in front to have a consistent index (snapshot_idx=0 == init)
    W_all = np.concatenate([W0[None, ...], Whist], axis=0)
    ep_all = [0] + [int(e) for e in epochs]  # epoch==0 for init (convention)

    # Prepare outputs
    stub = _nice_stub(ckpt_path)
    csv_path = stub + "_offline_metrics.csv"
    npz_path = stub + "_spectra.npz"

    # spectra accumulators
    all_eigs = []
    all_svals = []

    # Which S/A snapshots (raw matrices) to save?
    save_idx = _pick_indices_to_save(len(W_all), save_SA)
    save_SA_dir = None
    if save_idx:
        save_SA_dir = stub + "_SA"
        os.makedirs(save_SA_dir, exist_ok=True)

    # CSV header & writer
    header = [
        "snapshot_idx",
        "epoch",
        "fro_W",
        "fro_S",
        "fro_A",
        "mix_A_over_S",
        "sym_ratio",  # ||S||/||W||
        "asym_ratio",  # ||A||/||W||
        "non_normality_comm",  # ||WW^T - W^T W||_F
        "spectral_radius_W",
        "op_norm_2",
        "cond_number",
        "spectral_radius_S",  # of S
        "spectral_radius_A",  # of A (max |imag|)
    ]
    if do_pseudo:
        header += ["ps_frac_eps1e-1", "ps_frac_eps5e-2", "ps_frac_eps1e-2"]

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        for i, (W, ep) in enumerate(zip(W_all, ep_all)):
            # If W has NaN or Inf, we cannot trust any of the metrics.
            # Instead of crashing, write a NaN row and continue.
            if not np.isfinite(W).all():
                print(
                    f"[WARN] offline_metrics: W contains NaN/Inf "
                    f"(snapshot {i}, epoch {ep}) in {ckpt_path}; "
                    f"marking metrics as NaN.",
                    file=sys.stderr,
                )
                nan_row = [i, ep] + [float("nan")] * (len(header) - 2)
                w.writerow(nan_row)
                # still append placeholders so shapes line up for spectra
                all_eigs.append(np.array([], dtype=np.complex64))
                all_svals.append(np.array([], dtype=np.float32))
                # optional: still save S/A as NaN if this snapshot is in save_idx
                if i in save_idx and save_SA_dir is not None:
                    out_npz = os.path.join(save_SA_dir, f"SA_snapshot_{i:03d}.npz")
                    _maybe_save_SA(
                        out_npz, np.full_like(W, np.nan), np.full_like(W, np.nan)
                    )
                continue

            # --- existing logic below stays the same ---

            # Sym/Asym + norms
            S, A, nS, nA = _sym_asym(W)
            nW = _fro(W)
            sym_ratio = nS / (nW + 1e-12)
            asym_ratio = nA / (nW + 1e-12)
            mix = nA / (nS + 1e-12)
            nnorm = _non_normality_commutator(W)

            # Spectra
            try:
                eigs = np.linalg.eigvals(W).astype(np.complex64)
            except Exception:
                eigs = np.array([], dtype=np.complex64)
            all_eigs.append(eigs)

            smax, smin, svals = _operator_norm_2(W)
            all_svals.append(svals.astype(np.float32))
            cond = _condition_number(svals)

            # spectral radii of S and A
            try:
                eigS = np.linalg.eigvals(S)
                srS = float(np.max(np.abs(eigS)))
            except Exception:
                srS = float("nan")

            try:
                eigA = np.linalg.eigvals(A)
                srA = float(np.max(np.abs(eigA)))
            except Exception:
                srA = float("nan")

            row = [
                i,
                ep,
                nW,
                nS,
                nA,
                mix,
                sym_ratio,
                asym_ratio,
                nnorm,
                _spectral_radius(W),
                smax,
                cond,
                srS,
                srA,
            ]

            if do_pseudo:
                ps = _coarse_pseudospectrum_summary(W)
                row += [
                    ps["ps_frac_eps1e-1"],
                    ps["ps_frac_eps5e-2"],
                    ps["ps_frac_eps1e-2"],
                ]

            w.writerow(row)

            # Optional: save raw S/A for selected snapshots
            if i in save_idx and save_SA_dir is not None:
                out_npz = os.path.join(save_SA_dir, f"SA_snapshot_{i:03d}.npz")
                _maybe_save_SA(out_npz, S, A)

    # Save spectra bundle (aligned by snapshot_idx)
    # Shape them as [num_snaps, H] (pad ragged eigs if needed)
    maxH = max((len(x) for x in all_eigs), default=0)
    num = len(all_eigs)
    eigs_mat = np.full((num, maxH), np.nan + 1j * np.nan, dtype=np.complex64)
    for i, ev in enumerate(all_eigs):
        eigs_mat[i, : len(ev)] = ev

    # svals are all length H; we can stack directly (if H consistent)
    maxS = max((len(x) for x in all_svals), default=0)
    svals_mat = np.full((num, maxS), np.nan, dtype=np.float32)
    for i, sv in enumerate(all_svals):
        svals_mat[i, : len(sv)] = sv

    np.savez_compressed(
        npz_path,
        eigs=eigs_mat,
        svals=svals_mat,
        snapshot_idx=np.arange(num, dtype=np.int32),
        epoch=np.array(ep_all, dtype=np.int32),
    )

    print(f"[ok] {ckpt_path}")
    print(f"     CSV : {csv_path}")
    print(f"     NPZ : {npz_path}")
    if save_idx:
        print(f"     S/A : saved {len(save_idx)} snapshots under {save_SA_dir}")


# ----------------------------
# CLI
# ----------------------------
def find_checkpoints(arg_ckpt: str) -> List[str]:
    # If it's a file, return it
    if os.path.isfile(arg_ckpt):
        return [arg_ckpt]
    # If it's a directory, find all *.pth.tar under it
    if os.path.isdir(arg_ckpt):
        return sorted(
            glob.glob(os.path.join(arg_ckpt, "**", "*.pth.tar"), recursive=True)
        )
    # Else treat as glob
    return sorted(glob.glob(arg_ckpt, recursive=True))


def main():
    ap = argparse.ArgumentParser(
        description="Compute offline recurrent-weight metrics."
    )
    ap.add_argument(
        "--ckpt",
        required=True,
        help="Path to a .pth.tar, a directory, or a glob (e.g. 'runs/**/run_*/**.pth.tar').",
    )
    ap.add_argument(
        "--save_SA",
        default="none",
        help="Which S/A snapshots to save: 'none' (default), 'first,middle,last', 'all', or comma-separated indices (e.g. '0,5,10').",
    )
    ap.add_argument(
        "--grid_ps",
        action="store_true",
        help="Also compute a very coarse pseudospectrum summary (slow).",
    )
    args = ap.parse_args()

    ckpts = find_checkpoints(args.ckpt)
    if not ckpts:
        print(f"[warn] No checkpoints found for {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    for path in ckpts:
        try:
            process_checkpoint(path, save_SA=args.save_SA, do_pseudo=args.grid_ps)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[err] Failed on {path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
