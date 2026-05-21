"""
BuildHiddenWeights.py

Author: Margot Wagner
Date: 2026-04-23

Construct recurrent weight matrices used by `hidden-weight-builder.ipynb`.

This module contains only the hidden-weight builders needed for the published
workflow:
- shifted-band matrices
- Difference-of-Gaussians (DoG) Mexican-hat matrices
- DC-balanced, spectral-radius-targeted DoG Mexican-hat matrices

All builders return NumPy arrays with dtype `float32`. The matrices can be
saved directly or passed to helper functions for normalization, inspection,
and export.
"""

from typing import Optional, Tuple, Dict, Union
import numpy as np


def _ring_distances(n: int) -> np.ndarray:
    """
    Return absolute distances on a ring of length `n`.

    Parameters
    ----------
    n : int
        Number of units on the ring.

    Returns
    -------
    np.ndarray
        Array of shape `(n,)` containing wrap-around distances from zero.
    """
    offsets = np.arange(n)
    offsets_signed = ((offsets + n // 2) % n) - n // 2
    return np.abs(offsets_signed).astype(np.float64)


def build_shift(
    n: int,
    value: float = 1.0,
    offset: int = 1,
    cyclic: bool = False,
) -> np.ndarray:
    """
    Build a single shifted band matrix.

    Parameters
    ----------
    n : int
        Matrix size.
    value : float, default=1.0
        Value written along the shifted band.
    offset : int, default=1
        Diagonal offset. For example, `offset=0` with `value=1` produces the
        identity matrix.
    cyclic : bool, default=False
        If True, wrap indices around the matrix boundaries. If False, entries
        that fall outside the matrix are dropped.

    Returns
    -------
    np.ndarray
        Matrix of shape `(n, n)` with dtype `float32`.
    """
    W = np.zeros((n, n), dtype=np.float32)
    idx = np.arange(n)

    if cyclic:
        j = (idx + offset) % n
        W[idx, j] = value
    else:
        j = idx + offset
        mask = (j >= 0) & (j < n)
        W[idx[mask], j[mask]] = value

    return W


def build_mexican_hat_dog_balanced(
    n: int,
    sigma_e: float,
    sigma_i: float,
    a: float = 1.0,
    diag_offset: int = 0,
    rho_target: float = 0.9,
    eps: float = 1e-12,
    return_row0: bool = False,
) -> Union[Tuple[np.ndarray, Dict], Tuple[np.ndarray, Dict, np.ndarray]]:
    """
    Build a cyclic DoG Mexican-hat matrix with DC balancing and spectral scaling.

    This variant is intended for circulant/ring settings where it is useful to:
    1. choose the inhibitory amplitude so the kernel sum is approximately zero
    2. scale the kernel so its maximum Fourier magnitude matches `rho_target`

    Parameters
    ----------
    n : int
        Number of hidden units.
    sigma_e : float
        Width of the excitatory Gaussian.
    sigma_i : float
        Width of the inhibitory Gaussian.
    a : float, default=1.0
        Excitatory amplitude.
    diag_offset : int, default=0
        Integer shift applied to the kernel relative to the main diagonal.
    rho_target : float, default=0.9
        Target spectral radius for the circulant kernel before embedding.
    eps : float, default=1e-12
        Small constant used to avoid division by zero.
    return_row0 : bool, default=False
        If True, also return the kernel used as the first row of the circulant
        matrix.

    Returns
    -------
    W : np.ndarray
        Circulant weight matrix of shape `(n, n)` with dtype `float32`.
    meta : dict
        Metadata describing the kernel parameters and applied scaling.
    row0 : np.ndarray, optional
        Returned only if `return_row0=True`.
    """
    d = _ring_distances(n)

    ke = np.exp(-(d**2) / (2.0 * sigma_e**2))
    ki = np.exp(-(d**2) / (2.0 * sigma_i**2))

    # Choose inhibitory amplitude so the kernel has approximately zero DC component.
    b = a * (ke.sum() / (ki.sum() + eps))

    row0 = a * ke - b * ki

    eigvals = np.fft.fft(row0)
    rho_before = float(np.max(np.abs(eigvals)))
    gain = float(rho_target / (rho_before + eps))
    row0 = (gain * row0).astype(np.float32)

    W = np.zeros((n, n), dtype=np.float32)
    i = np.arange(n)
    for j in range(n):
        k_idx = (i - j - diag_offset) % n
        W[i, j] = row0[k_idx]

    meta = {
        "builder": "build_mexican_hat_dog_balanced",
        "structure": "DoG Mexican Hat (cyclic/circulant, DC-balanced, rho-targeted)",
        "n": int(n),
        "sigma_e": float(sigma_e),
        "sigma_i": float(sigma_i),
        "a": float(a),
        "b": float(b),
        "diag_offset": int(diag_offset),
        "rho_target": float(rho_target),
        "rho_before_gain": float(rho_before),
        "gain": float(gain),
        "kernel_sum_after_balance": float(np.sum(a * ke - b * ki)),
    }

    if return_row0:
        return W, meta, row0
    return W, meta
