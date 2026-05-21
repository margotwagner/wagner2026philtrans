#!/usr/bin/env python3
"""
Build recurrent hidden-weight initialization matrices.

By default, it builds the same families used in the publication:

1. Vanilla PyTorch random recurrent weights
2. Identity recurrent weights
3. Cyclic-shift recurrent weights with an alpha sweep
4. Difference-of-Gaussians Mexican-hat recurrent weights, including:
   - centered k0 matrix
   - shifted k5 alpha sweep with condition-specific rho targets

This script builds and saves matrices plus JSON metadata.

Default output directory
------------------------
``data/hidden_weight_inits``

Examples
--------
Build all default hidden-weight initializations:

    python src/setup/build_hidden_weights.py

Build only cyclic-shift and Mexican-hat initializations:

    python src/setup/build_hidden_weights.py --families cyclic_shift mexican_hat

Use a different output directory:

    python src/setup/build_hidden_weights.py --output-dir data/hidden_weight_inits

Overwrite existing files without prompting:

    python src/setup/build_hidden_weights.py --overwrite
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch


# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------


def mix_ratio_tag(alpha: float) -> str:
    """
    Convert an alpha value into the directory tag used by the original notebook.

    Examples
    --------
    0.00 -> ``sym0p00``
    0.25 -> ``sym0p25``
    1.00 -> ``sym1p00``
    """
    pct = int(round(float(alpha) * 100))
    major = pct // 100
    minor = pct % 100
    return f"sym{major}p{minor:02d}"


# -----------------------------------------------------------------------------
# Matrix builders
# -----------------------------------------------------------------------------


def _ring_distances(n: int) -> np.ndarray:
    """
    Return absolute distances on a ring of length ``n``.

    Parameters
    ----------
    n : int
        Number of units on the ring.

    Returns
    -------
    np.ndarray
        Array of shape ``(n,)`` containing wrap-around distances from zero.
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
    Build a single shifted-band matrix.

    Parameters
    ----------
    n : int
        Matrix size.
    value : float, default=1.0
        Value written along the shifted band.
    offset : int, default=1
        Diagonal offset. ``offset=0`` with ``value=1`` produces identity.
    cyclic : bool, default=False
        If True, wrap indices around matrix boundaries.

    Returns
    -------
    np.ndarray
        Matrix of shape ``(n, n)`` with dtype ``float32``.
    """
    if n <= 0:
        raise ValueError("n must be positive.")

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

    This variant:
    1. chooses the inhibitory amplitude so the kernel sum is approximately zero
    2. scales the kernel so its maximum Fourier magnitude matches ``rho_target``

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
        If True, also return the kernel used as the first row.

    Returns
    -------
    W : np.ndarray
        Circulant weight matrix of shape ``(n, n)`` with dtype ``float32``.
    meta : dict
        Metadata describing the kernel parameters and applied scaling.
    row0 : np.ndarray, optional
        Returned only if ``return_row0=True``.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if sigma_e <= 0 or sigma_i <= 0:
        raise ValueError("sigma_e and sigma_i must be positive.")
    if rho_target <= 0:
        raise ValueError("rho_target must be positive.")

    d = _ring_distances(n)

    ke = np.exp(-(d**2) / (2.0 * sigma_e**2))
    ki = np.exp(-(d**2) / (2.0 * sigma_i**2))

    # Choose inhibitory amplitude so the kernel has approximately zero DC.
    b = a * (ke.sum() / (ki.sum() + eps))

    row0_unscaled = a * ke - b * ki

    eigvals = np.fft.fft(row0_unscaled)
    rho_before = float(np.max(np.abs(eigvals)))
    gain = float(rho_target / (rho_before + eps))
    row0 = (gain * row0_unscaled).astype(np.float32)

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
        "kernel_sum_after_balance": float(np.sum(row0_unscaled)),
    }

    if return_row0:
        return W, meta, row0
    return W, meta


# -----------------------------------------------------------------------------
# Non-plotting helper functions
# -----------------------------------------------------------------------------


def _svd_sigma_max(W: np.ndarray) -> float:
    """Return the largest singular value of a matrix."""
    return float(np.linalg.svd(W, compute_uv=False)[0])


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
        Metadata dictionary with matrix statistics and any extra fields.
    """
    W = np.asarray(W, dtype=np.float32)
    eigvals = np.linalg.eigvals(W)

    meta = {
        "shape": list(W.shape),
        "mean": float(W.mean()),
        "var": float(((W - W.mean()) ** 2).mean()),
        "fro_norm": float(np.linalg.norm(W, ord="fro")),
        "sigma_max": _svd_sigma_max(W),
        "spectral_radius_abs_eigs": float(np.max(np.abs(eigvals))) if eigvals.size else 0.0,
        "asymmetry_ratio": float(np.linalg.norm(W - W.T) / (np.linalg.norm(W) + 1e-12)),
    }

    if extra:
        meta.update(extra)

    return meta


def save_matrix(
    W: np.ndarray,
    save_dir: Path,
    name: str,
    meta: Optional[Dict] = None,
    overwrite: bool = False,
) -> None:
    """
    Save a matrix as ``.npy`` plus a JSON metadata file.

    Parameters
    ----------
    W : np.ndarray
        Matrix to save.
    save_dir : Path
        Output directory.
    name : str
        Base filename without extension.
    meta : dict, optional
        Additional metadata to merge with automatic summary statistics.
    overwrite : bool, default=False
        If False, raise an error when either output file already exists.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    base = save_dir / name
    matrix_path = base.with_suffix(".npy")
    meta_path = base.with_suffix(".json")

    if not overwrite and (matrix_path.exists() or meta_path.exists()):
        raise FileExistsError(
            f"Refusing to overwrite existing output: {matrix_path} or {meta_path}. "
            "Pass --overwrite to replace existing files."
        )

    W = np.asarray(W, dtype=np.float32)
    np.save(matrix_path, W)

    meta_all = with_stats_meta(W, extra=(meta or {}))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_all, f, indent=2)

    print(f"Saved: {matrix_path} and {meta_path}")


def is_circulant(W: np.ndarray, tol: float = 1e-6, verbose: bool = False) -> bool:
    """
    Check whether a square matrix is circulant.

    A matrix is circulant if each wrap-around diagonal contains a constant value.
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


# -----------------------------------------------------------------------------
# Family-specific build routines
# -----------------------------------------------------------------------------


def build_random_pytorch_family(
    output_dir: Path,
    hidden_n: int,
    seeds: Sequence[int],
    overwrite: bool = False,
) -> None:
    """
    Build vanilla PyTorch random recurrent-weight initializations.

    This intentionally uses ``torch.nn.RNN`` so the saved matrix corresponds to
    PyTorch's default ``weight_hh_l0`` initialization for an Elman-style RNN.
    """
    dense_root = output_dir / "random_pytorch"

    for seed in seeds:
        print(f"\n=== Vanilla PyTorch random, seed={seed} ===")
        torch.manual_seed(int(seed))

        model = torch.nn.RNN(
            input_size=hidden_n,
            hidden_size=hidden_n,
            nonlinearity="tanh",
            batch_first=True,
        )

        W_dense = model.weight_hh_l0.detach().cpu().numpy().astype(np.float32)
        emp_var = float(W_dense.var())
        print(f"[vanilla] empirical var(Whh) ≈ {emp_var:.6f}")

        ok = is_circulant(W_dense, tol=1e-7)
        print(f"[vanilla] circulant? {ok} (tol=1e-7)")

        save_dir_dense = dense_root / f"seed{seed:03d}"
        meta_dense = {
            "backend": "dense",
            "family": "vanilla_pytorch",
            "hidden_n": int(hidden_n),
            "seed": int(seed),
            "norm": "raw",
            "source": "torch.nn.RNN.weight_hh_l0_default_init",
        }
        save_matrix(W_dense, save_dir_dense, "Whh", meta=meta_dense, overwrite=overwrite)


def build_identity_family(
    output_dir: Path,
    hidden_n: int,
    overwrite: bool = False,
) -> None:
    """Build the cyclic identity recurrent-weight initialization."""
    print(f"\n=== Identity, H={hidden_n} ===")

    W = build_shift(n=hidden_n, value=1.0, offset=0, cyclic=True)

    ok = is_circulant(W, tol=1e-7)
    print(f"[identity] circulant? {ok} (tol=1e-7)")
    if not ok:
        raise RuntimeError("Expected identity matrix to be circulant.")

    save_matrix(
        W,
        output_dir / "identity",
        "Whh",
        meta={
            "backend": "dense",
            "family": "identity",
            "hidden_n": int(hidden_n),
            "norm": "raw",
        },
        overwrite=overwrite,
    )


def build_cyclic_shift_family(
    output_dir: Path,
    hidden_n: int,
    mix_ratios: Sequence[float],
    overwrite: bool = False,
) -> None:
    """Build cyclic-shift recurrent-weight initializations across alpha values."""
    print(f"\n=== Cyclic shift, H={hidden_n} ===")

    W = build_shift(n=hidden_n, value=1.0, offset=1, cyclic=True)

    ok = is_circulant(W, tol=1e-7)
    print(f"[cyclic_shift raw] circulant? {ok} (tol=1e-7)")
    if not ok:
        raise RuntimeError("Expected cyclic-shift matrix to be circulant.")

    S = 0.5 * (W + W.T)
    A = 0.5 * (W - W.T)

    dense_root = output_dir / "cyclic_shift"

    for alpha in mix_ratios:
        alpha = float(alpha)
        W_mix = (alpha * S + (1.0 - alpha) * A).astype(np.float32)

        ok_mix = is_circulant(W_mix, tol=1e-7)
        print(f"[cyclic_shift alpha={alpha:.2f}] circulant? {ok_mix} (tol=1e-7)")

        sub = mix_ratio_tag(alpha)
        save_dir_dense = dense_root / f"alpha{sub}"

        meta_dense = {
            "backend": "dense",
            "family": "shift",
            "geometry": "cyclic",
            "hidden_n": int(hidden_n),
            "alpha": alpha,
            "decomposition": "W=αS+(1-α)A",
            "norm": "raw",
        }

        save_matrix(W_mix, save_dir_dense, "Whh", meta=meta_dense, overwrite=overwrite)


def build_mexican_hat_family(
    output_dir: Path,
    hidden_n: int,
    mix_ratios: Sequence[float],
    overwrite: bool = False,
) -> None:
    """
    Build DoG Mexican-hat recurrent-weight initializations.

    This follows the notebook defaults:
    - dog config: sigma_e=20, sigma_i=30, a=1
    - k0 uses rho_target=2.2 and no alpha sweep
    - k5 uses rho_target values that depend on alpha
    """
    print(f"\n=== Mexican-hat DoG, H={hidden_n} ===")

    dog_configs = [
        ("dog", dict(sigma_e=20.0, sigma_i=30.0, a=1.0)),
    ]

    offset_configs = [
        ("k0", 0),
        ("k5", -5),
    ]

    rho_targets_k5 = {
        0.00: 7.0,
        0.25: 6.5,
        0.50: 4.3,
        0.75: 3.0,
        1.00: 2.3,
    }

    rho_target_k0 = 2.2
    dense_root = output_dir / "mexican_hat"

    for dog_name, params in dog_configs:
        for offset_tag, diag_offset in offset_configs:
            if offset_tag == "k0":
                rho_target = rho_target_k0

                print(
                    f"\n=== Building {dog_name}_{offset_tag} "
                    f"(diag_offset={diag_offset}, rho_target={rho_target}) ==="
                )

                W, meta_build = build_mexican_hat_dog_balanced(
                    n=hidden_n,
                    diag_offset=diag_offset,
                    sigma_e=params["sigma_e"],
                    sigma_i=params["sigma_i"],
                    a=params["a"],
                    rho_target=rho_target,
                    return_row0=False,
                )

                ok = is_circulant(W, tol=1e-7)
                print(f"[mexican_hat {offset_tag}] circulant? {ok} (tol=1e-7)")

                base_meta = {
                    "structure": "DoG Mexican Hat (cyclic, DC-balanced, rho-targeted)",
                    "family": "mexican_hat_dog_balanced",
                    "hidden_n": int(hidden_n),
                    "sigma_e": float(params["sigma_e"]),
                    "sigma_i": float(params["sigma_i"]),
                    "a": float(params["a"]),
                    "diag_offset": int(diag_offset),
                    "rho_target": float(rho_target),
                    "norm": "raw",
                    **meta_build,
                }

                save_matrix(
                    W,
                    dense_root / offset_tag,
                    "Whh",
                    meta=base_meta,
                    overwrite=overwrite,
                )

            elif offset_tag == "k5":
                for alpha in mix_ratios:
                    alpha = float(alpha)
                    rounded_alpha = round(alpha, 2)

                    if rounded_alpha not in rho_targets_k5:
                        raise ValueError(
                            f"Mexican-hat k5 rho target is only defined for "
                            f"alpha values {sorted(rho_targets_k5)}. Got {alpha}."
                        )

                    rho_target = rho_targets_k5[rounded_alpha]
                    sub = mix_ratio_tag(alpha)

                    print(
                        f"\n=== Building {dog_name}_{offset_tag} "
                        f"(diag_offset={diag_offset}, alpha={alpha:.2f}, "
                        f"rho_target={rho_target}) ==="
                    )

                    W, meta_build = build_mexican_hat_dog_balanced(
                        n=hidden_n,
                        diag_offset=diag_offset,
                        sigma_e=params["sigma_e"],
                        sigma_i=params["sigma_i"],
                        a=params["a"],
                        rho_target=rho_target,
                        return_row0=False,
                    )

                    S = 0.5 * (W + W.T)
                    A = 0.5 * (W - W.T)
                    W_mix = (alpha * S + (1.0 - alpha) * A).astype(np.float32)

                    ok_mix = is_circulant(W_mix, tol=1e-7)
                    print(f"[mexican_hat {offset_tag} alpha={alpha:.2f}] circulant? {ok_mix} (tol=1e-7)")

                    meta = {
                        "structure": "DoG Mexican Hat (cyclic, DC-balanced, rho-targeted)",
                        "family": "mexican_hat_dog_balanced",
                        "hidden_n": int(hidden_n),
                        "sigma_e": float(params["sigma_e"]),
                        "sigma_i": float(params["sigma_i"]),
                        "a": float(params["a"]),
                        "diag_offset": int(diag_offset),
                        "rho_target": float(rho_target),
                        "alpha": alpha,
                        "decomposition": "W=αS+(1-α)A",
                        "norm": "raw",
                        **meta_build,
                    }

                    save_matrix(
                        W_mix,
                        dense_root / offset_tag / f"alpha{sub}",
                        "Whh",
                        meta=meta,
                        overwrite=overwrite,
                    )

            else:
                raise ValueError(f"Unexpected offset_tag={offset_tag!r}")


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------


def parse_int_list(values: str) -> Tuple[int, ...]:
    """Parse comma-separated integer values."""
    if not values:
        return tuple()
    return tuple(int(v.strip()) for v in values.split(",") if v.strip())


def parse_float_list(values: str) -> Tuple[float, ...]:
    """Parse comma-separated float values."""
    if not values:
        return tuple()
    return tuple(float(v.strip()) for v in values.split(",") if v.strip())


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build hidden recurrent weight initializations."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/hidden_weight_inits"),
        help="Directory where hidden-weight initializations will be saved.",
    )
    parser.add_argument(
        "--hidden-n",
        type=int,
        default=100,
        help="Number of hidden units / recurrent matrix size.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=["random", "identity", "cyclic_shift", "mexican_hat"],
        choices=["random", "identity", "cyclic_shift", "mexican_hat"],
        help="Which hidden-weight families to build.",
    )
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=(0,),
        help="Comma-separated random seeds for the vanilla PyTorch baseline.",
    )
    parser.add_argument(
        "--mix-ratios",
        type=parse_float_list,
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
        help="Comma-separated alpha values for cyclic-shift and Mexican-hat sweeps.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy and .json outputs.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Build requested hidden-weight initialization families."""
    args = parse_args(argv)

    if args.hidden_n <= 0:
        raise ValueError("--hidden-n must be positive.")

    print(f"Output directory: {args.output_dir}")
    print(f"Hidden units: {args.hidden_n}")
    print(f"Families: {', '.join(args.families)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if "random" in args.families:
        build_random_pytorch_family(
            output_dir=args.output_dir,
            hidden_n=args.hidden_n,
            seeds=args.seeds,
            overwrite=args.overwrite,
        )

    if "identity" in args.families:
        build_identity_family(
            output_dir=args.output_dir,
            hidden_n=args.hidden_n,
            overwrite=args.overwrite,
        )

    if "cyclic_shift" in args.families:
        build_cyclic_shift_family(
            output_dir=args.output_dir,
            hidden_n=args.hidden_n,
            mix_ratios=args.mix_ratios,
            overwrite=args.overwrite,
        )

    if "mexican_hat" in args.families:
        build_mexican_hat_family(
            output_dir=args.output_dir,
            hidden_n=args.hidden_n,
            mix_ratios=args.mix_ratios,
            overwrite=args.overwrite,
        )

    print("\nDone building hidden-weight initializations.")


if __name__ == "__main__":
    main()
