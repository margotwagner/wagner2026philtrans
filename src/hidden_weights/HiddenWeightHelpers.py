"""
HiddenWeightHelpers.py

Author: Margot Wagner
Date: 2026-04-23

Helper functions for inspecting, normalizing, and saving recurrent weight
matrices used by `hidden-weight-builder.ipynb`.

This module is intentionally limited to the utilities needed for building and
exporting the hidden-weight initializations used in the paper. The functions
fall into four groups:

1. plotting weight matrices and eigenspectra
2. computing summary statistics for saved metadata
3. normalizing matrices by Frobenius norm
4. extracting and saving the first row of circulant matrices

All public functions operate on NumPy arrays and save data in a simple
`.npy` + `.json` format for reproducibility.
"""

import json
import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _svd_sigma_max(W: np.ndarray) -> float:
    """Return the largest singular value of a matrix."""
    return float(np.linalg.svd(W, compute_uv=False)[0])


def _decompose_sym_skew(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a square matrix into symmetric and skew-symmetric parts.

    Parameters
    ----------
    W : np.ndarray
        Square matrix.

    Returns
    -------
    W_sym : np.ndarray
        Symmetric component, 0.5 * (W + W.T).
    W_skew : np.ndarray
        Skew-symmetric component, 0.5 * (W - W.T).
    """
    W_sym = 0.5 * (W + W.T)
    W_skew = 0.5 * (W - W.T)
    return W_sym.astype(np.float32), W_skew.astype(np.float32)


def plot_weight_all(
    W: np.ndarray,
    title: str = "Weights",
    show_unit_circle: bool = True,
    unit_radius: float = 1.0,
    trace_band_summary: str = "trace",
) -> None:
    """
    Plot a weight matrix, its diagonal-offset summary, and its eigenspectrum.

    Parameters
    ----------
    W : np.ndarray
        Square weight matrix of shape `(n, n)`.
    title : str, default="Weights"
        Base title used across panels.
    show_unit_circle : bool, default=True
        If True, draw a reference circle of radius `unit_radius` in the
        eigenspectrum panel.
    unit_radius : float, default=1.0
        Radius of the reference circle.
    trace_band_summary : {"trace", "bandmean"}, default="trace"
        Method used for the middle panel:
        - `"trace"` plots the sum along each diagonal offset
        - `"bandmean"` plots the mean value along each diagonal offset
    """
    W = np.asarray(W)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be a square 2D array, got shape {W.shape}.")

    eigvals = np.linalg.eigvals(W)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.0))
    ax_matrix, ax_diag, ax_eigs = axes

    # ------------------------------------------------------------------
    # Panel 1: matrix heatmap
    # ------------------------------------------------------------------
    vmax = float(np.max(np.abs(W))) if W.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    image = ax_matrix.imshow(
        W,
        cmap="RdBu_r",
        vmin=-1.05 * vmax,
        vmax=1.05 * vmax,
        interpolation="nearest",
        aspect="equal",
        origin="upper",
    )
    colorbar = fig.colorbar(image, ax=ax_matrix, fraction=0.046, pad=0.04)
    colorbar.set_label("Weight", rotation=270, labelpad=12)

    ax_matrix.set_title(f"{title}: matrix")
    ax_matrix.set_xlabel("Presynaptic unit")
    ax_matrix.set_ylabel("Postsynaptic unit")

    # ------------------------------------------------------------------
    # Panel 2: diagonal-offset summary
    # ------------------------------------------------------------------
    n = W.shape[0]
    offsets = np.arange(-(n - 1), n, dtype=int)
    diagonal_sums = np.array([np.trace(W, int(k)) for k in offsets], dtype=float)

    if trace_band_summary == "bandmean":
        counts = (n - np.abs(offsets)).astype(float)
        counts[counts <= 0] = np.nan
        y = diagonal_sums / counts
        ylabel = "Mean diagonal value"
        diag_title = "Diagonal band mean"
    else:
        y = diagonal_sums
        ylabel = "Sum along diagonal"
        diag_title = "Diagonal trace"

    ax_diag.plot(offsets, y, lw=2)
    ax_diag.axhline(0, ls="--", lw=1, alpha=0.6, color="k")
    ax_diag.axvline(0, ls=":", lw=1, alpha=0.6, color="k")
    ax_diag.set_title(diag_title)
    ax_diag.set_xlabel("Diagonal offset")
    ax_diag.set_ylabel(ylabel)

    # ------------------------------------------------------------------
    # Panel 3: eigenspectrum
    # ------------------------------------------------------------------
    spectral_radius = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
    ax_eigs.scatter(eigvals.real, eigvals.imag, s=10)

    spectral_circle = plt.Circle((0, 0), spectral_radius, fill=False, linestyle="--")
    ax_eigs.add_artist(spectral_circle)

    if show_unit_circle and unit_radius is not None and unit_radius > 0:
        unit_circle = plt.Circle(
            (0, 0),
            unit_radius,
            fill=False,
            linestyle=":",
            linewidth=1.5,
            alpha=0.9,
            color="tab:red",
        )
        ax_eigs.add_artist(unit_circle)

    ax_eigs.axhline(0, lw=0.5, color="k")
    ax_eigs.axvline(0, lw=0.5, color="k")
    ax_eigs.set_aspect("equal", "box")

    lim = max(
        spectral_radius,
        unit_radius if show_unit_circle else 0.0,
        np.max(np.abs(eigvals.real)) if eigvals.size else 1.0,
        np.max(np.abs(eigvals.imag)) if eigvals.size else 1.0,
    )
    lim = 1.05 * (lim if lim > 0 else 1.0)
    ax_eigs.set_xlim(-lim, lim)
    ax_eigs.set_ylim(-lim, lim)
    ax_eigs.set_title(f"{title}: eigenspectrum (ρ ≈ {spectral_radius:.3f})")

    plt.tight_layout()
    plt.show()


def with_stats_meta(W: np.ndarray, extra: Optional[Dict] = None) -> Dict:
    """
    Compute summary statistics for a weight matrix.

    Parameters
    ----------
    W : np.ndarray
        Weight matrix.
    extra : dict, optional
        Additional metadata to merge into the returned dictionary.

    Returns
    -------
    dict
        Dictionary containing matrix shape, mean, variance, Frobenius norm,
        largest singular value, spectral radius, and an asymmetry ratio.
    """
    W = np.asarray(W, dtype=np.float32)
    eigvals = np.linalg.eigvals(W)

    meta = {
        "shape": list(W.shape),
        "mean": float(W.mean()),
        "var": float(((W - W.mean()) ** 2).mean()),
        "fro_norm": float(np.linalg.norm(W, ord="fro")),
        "sigma_max": _svd_sigma_max(W),
        "spectral_radius_abs_eigs": float(np.max(np.abs(eigvals)))
        if eigvals.size
        else 0.0,
        "asymmetry_ratio": float(np.linalg.norm(W - W.T) / (np.linalg.norm(W) + 1e-12)),
    }

    if extra:
        meta.update(extra)

    return meta


def save_matrix(
    W: np.ndarray,
    save_dir: str,
    name: str,
    meta: Optional[Dict] = None,
) -> None:
    """
    Save a weight matrix as `.npy` plus a `.json` metadata file.

    Parameters
    ----------
    W : np.ndarray
        Matrix to save.
    save_dir : str
        Output directory.
    name : str
        Base filename without extension.
    meta : dict, optional
        Additional metadata to merge with the automatically computed summary.
    """
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.join(save_dir, name)

    W = np.asarray(W, dtype=np.float32)
    np.save(base + ".npy", W)

    meta_all = with_stats_meta(W, extra=(meta or {}))
    with open(base + ".json", "w") as f:
        json.dump(meta_all, f, indent=2)

    print(f"Saved: {base}.npy and {base}.json")


def normalize_by_fro(W: np.ndarray, target_fro: float) -> Tuple[np.ndarray, Dict]:
    """
    Rescale a matrix so that its Frobenius norm matches `target_fro`.

    Parameters
    ----------
    W : np.ndarray
        Input matrix.
    target_fro : float
        Desired Frobenius norm after rescaling.

    Returns
    -------
    W_scaled : np.ndarray
        Rescaled matrix.
    info : dict
        Dictionary containing the scale factor and norms before and after
        normalization.
    """
    W = np.asarray(W, dtype=np.float32).copy()
    fro_before = float(np.linalg.norm(W, ord="fro"))

    if fro_before <= 0:
        return W, {
            "scale": 1.0,
            "fro_before": fro_before,
            "fro_after": fro_before,
            "status": "degenerate",
        }

    scale = target_fro / fro_before
    W_scaled = scale * W
    fro_after = float(np.linalg.norm(W_scaled, ord="fro"))

    return W_scaled, {
        "scale": scale,
        "fro_before": fro_before,
        "fro_after": fro_after,
        "status": "ok",
    }


def plot_sym_asym(W: np.ndarray, base_title: str = "W") -> None:
    """
    Plot the symmetric and skew-symmetric components of a weight matrix.

    Parameters
    ----------
    W : np.ndarray
        Square weight matrix.
    base_title : str, default="W"
        Base title used for the derived plots.
    """
    W_sym, W_skew = _decompose_sym_skew(W)
    plot_weight_all(W_sym, title=f"{base_title} (symmetric)")
    plot_weight_all(W_skew, title=f"{base_title} (skew-symmetric)")


def is_circulant(W: np.ndarray, tol: float = 1e-6, verbose: bool = False) -> bool:
    """
    Check whether a square matrix is circulant.

    A matrix is circulant if each wrap-around diagonal contains a constant value.

    Parameters
    ----------
    W : np.ndarray
        Square matrix to test.
    tol : float, default=1e-6
        Maximum allowed deviation within each wrap-around diagonal.
    verbose : bool, default=False
        If True, print the maximum deviation for each diagonal.

    Returns
    -------
    bool
        True if the matrix is circulant up to the requested tolerance.
    """
    W = np.asarray(W)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square 2D array.")

    n = W.shape[0]
    idx = np.arange(n)

    for k in range(n):
        vals = W[idx, (idx + k) % n]
        deviation = np.max(np.abs(vals - vals[0]))

        if verbose:
            print(f"k={k:2d}  max_dev={float(deviation):.3e}")

        if float(deviation) > tol:
            return False

    return True


def extract_first_row(W: np.ndarray) -> np.ndarray:
    """
    Extract the first row of a square weight matrix.

    This is useful for circulant matrices, which are fully determined by their
    first row.

    Parameters
    ----------
    W : np.ndarray
        Square matrix of shape `(H, H)`.

    Returns
    -------
    np.ndarray
        First row of shape `(H,)`.
    """
    W = np.asarray(W)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix of shape (H, H).")

    return W[0, :].astype(np.float32, copy=True)


def save_first_row(
    row0: np.ndarray,
    save_dir: str,
    name: str,
    meta: Optional[Dict] = None,
) -> None:
    """
    Save a circulant first row as `.npy` plus a small `.json` metadata file.

    Parameters
    ----------
    row0 : np.ndarray
        First row of a circulant matrix.
    save_dir : str
        Output directory.
    name : str
        Base filename without extension.
    meta : dict, optional
        Additional metadata to include in the JSON file.
    """
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.join(save_dir, name)

    row0 = np.asarray(row0, dtype=np.float32)
    np.save(base + ".npy", row0)

    row_meta = {
        "length": int(row0.shape[0]),
        "mean": float(row0.mean()),
        "var": float(((row0 - float(row0.mean())) ** 2).mean()),
        "l2_norm": float(np.linalg.norm(row0)),
        "type": "first_row",
    }
    if meta:
        row_meta.update(meta)

    with open(base + ".json", "w") as f:
        json.dump(row_meta, f, indent=2)

    print(f"Saved: {base}.npy and {base}.json")


def extract_and_optionally_save_first_row(
    W: np.ndarray,
    save_dir: Optional[str] = None,
    name: Optional[str] = None,
    meta: Optional[Dict] = None,
) -> np.ndarray:
    """
    Extract the first row of a matrix and optionally save it.

    Parameters
    ----------
    W : np.ndarray
        Square matrix.
    save_dir : str, optional
        Directory in which to save the row.
    name : str, optional
        Base filename without extension.
    meta : dict, optional
        Additional metadata to store if saving.

    Returns
    -------
    np.ndarray
        Extracted first row.
    """
    row0 = extract_first_row(W)

    if save_dir is not None and name is not None:
        save_first_row(row0, save_dir, name, meta=meta)

    return row0
