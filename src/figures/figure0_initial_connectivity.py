#!/usr/bin/env python3
"""
Plot initial recurrent connectivity.

Creates 3-panel figures showing:
1. Recurrent weight heatmap
2. Diagonal-offset trace or band mean
3. Eigenspectrum with optional unit circle

By default, this script now saves/plots the full initial recurrent matrix and
its symmetric and antisymmetric components:

    W_sym  = 0.5 * (W + W.T)
    W_asym = 0.5 * (W - W.T)

This mirrors the inspection workflow used in ``hidden-weight-builder.ipynb``,
but keeps the plotting in ``src/figures`` rather than in the setup/build stage.

Example:
    python ./src/figures/figure0_initial_connectivity.py \
        ./data/hidden_weight_inits/mexicanhat/k5/alphasym0p75/Whh.npy \
        --savepath ./data/figures/figure0/mexicanhat_alpha0p75.png \
        --alpha-label 0.75 \
        --trace-lw 4 \
        --no-show

This writes:
    ./data/figures/figure0/mexicanhat_alpha0p75.png
    ./data/figures/figure0/mexicanhat_alpha0p75_symmetric.png
    ./data/figures/figure0/mexicanhat_alpha0p75_antisymmetric.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def decompose_sym_asym(W):
    """
    Decompose a square matrix into symmetric and antisymmetric components.

    Parameters
    ----------
    W : array-like, shape (N, N)
        Square recurrent weight matrix.

    Returns
    -------
    W_sym : np.ndarray
        Symmetric component, 0.5 * (W + W.T).
    W_asym : np.ndarray
        Antisymmetric/skew-symmetric component, 0.5 * (W - W.T).
    """
    W = np.asarray(W)

    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square 2D, got shape={W.shape}")

    W_sym = 0.5 * (W + W.T)
    W_asym = 0.5 * (W - W.T)

    return W_sym.astype(np.float32), W_asym.astype(np.float32)


def component_savepath(savepath, suffix):
    """
    Add a suffix before the file extension.

    Examples
    --------
    figure.png + "symmetric" -> figure_symmetric.png
    """
    if savepath is None:
        return None

    savepath = Path(savepath)
    return savepath.with_name(f"{savepath.stem}_{suffix}{savepath.suffix}")


def plot_init(
    W,
    title="Weights",
    suptitle=None,
    show_unit_circle=True,
    unit_radius=1.0,
    trace_band_summary="trace",
    trace_lw=2.0,
    savepath=None,
    show=True,
):
    """
    Plot heatmap, diagonal-offset trace, and eigenspectrum.

    Parameters
    ----------
    W : array-like, shape (N, N)
        Recurrent weight matrix.
    title : str
        Title label for the weight matrix.
    suptitle : str or None
        Figure-level title.
    show_unit_circle : bool
        Whether to show the |lambda| = unit_radius circle.
    unit_radius : float
        Radius of the reference circle.
    trace_band_summary : {"trace", "bandmean"}
        "trace": sum of each diagonal offset.
        "bandmean": mean of each diagonal offset.
    trace_lw : float
        Line width for trace panel.
    savepath : str or Path or None
        Where to save the figure. If None, does not save.
    show : bool
        Whether to display the figure interactively.
    """
    W = np.asarray(W)

    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square 2D, got shape={W.shape}")

    eig = np.linalg.eigvals(W)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.0))
    axW, axT, axE = axes

    # -------------------------
    # Panel 1: Heatmap
    # -------------------------
    vmax = float(np.max(np.abs(W))) if W.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    im = axW.imshow(
        W,
        cmap="RdBu_r",
        vmin=-1.05 * vmax,
        vmax=+1.05 * vmax,
        interpolation="nearest",
        aspect="equal",
        origin="upper",
    )

    cb = fig.colorbar(im, ax=axW, fraction=0.046, pad=0.04)
    cb.set_label("weight", rotation=270, labelpad=12)

    axW.set_title("Recurrent Connectivity")
    axW.set_xlabel("Presynaptic")
    axW.set_ylabel("Postsynaptic")

    # -------------------------
    # Panel 2: Diagonal-offset trace
    # -------------------------
    n = W.shape[0]
    offs = np.arange(-(n - 1), n, dtype=int)

    tr = np.array([np.trace(W, int(k)) for k in offs], dtype=float)

    if trace_band_summary == "bandmean":
        counts = (n - np.abs(offs)).astype(float)
        counts[counts <= 0] = np.nan
        y = tr / counts
        ylabel = "mean diag_k W"
        ttl = "Diagonal band mean"
    elif trace_band_summary == "trace":
        y = tr
        ylabel = "Weight avg."
        ttl = "Trace"
    else:
        raise ValueError(
            "trace_band_summary must be either 'trace' or 'bandmean', "
            f"got {trace_band_summary!r}"
        )

    axT.plot(offs, y, lw=trace_lw)
    axT.axhline(0, ls="--", lw=1, alpha=0.6, color="k")
    axT.axvline(0, ls=":", lw=1, alpha=0.6, color="k")
    axT.set_title(ttl)
    axT.set_xlabel("Diagonal offset k")
    axT.set_ylabel(ylabel)

    # -------------------------
    # Panel 3: Eigenspectrum
    # -------------------------
    radius = float(np.max(np.abs(eig))) if eig.size else 0.0

    axE.scatter(eig.real, eig.imag, s=10)

    if show_unit_circle and unit_radius is not None and unit_radius > 0:
        unit = plt.Circle(
            (0, 0),
            unit_radius,
            fill=False,
            linestyle="--",
            linewidth=1,
            alpha=0.4,
            color="k",
        )
        axE.add_artist(unit)

    eig_proxy = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=5,
        color="tab:blue",
        label="eigenvalues",
    )

    unit_proxy = Line2D(
        [0],
        [0],
        linestyle="--",
        linewidth=1,
        color="k",
        alpha=0.4,
        label=f"|λ|={unit_radius:g}",
    )

    axE.axhline(0, lw=0.5, color="k")
    axE.axvline(0, lw=0.5, color="k")
    axE.set_aspect("equal", "box")

    axE.set_xlim(-1.2, 2.8)
    axE.set_ylim(-1.1, 1.1)

    axE.set_title(f"Eigenspectrum (ρ≈{radius:.3f})")
    axE.set_xlabel("Re(λ)")
    axE.set_ylabel("Im(λ)")

    if show_unit_circle:
        axE.legend(handles=[eig_proxy, unit_proxy], frameon=False, fontsize=12)
    else:
        axE.legend(handles=[eig_proxy], frameon=False, fontsize=12)

    if suptitle is None:
        suptitle = title

    fig.suptitle(suptitle)
    fig.tight_layout()

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {savepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_init_with_components(
    W,
    title="Weights",
    suptitle=None,
    show_unit_circle=True,
    unit_radius=1.0,
    trace_band_summary="trace",
    trace_lw=2.0,
    savepath=None,
    show=True,
    include_components=True,
):
    """
    Plot the full initial matrix and, optionally, its symmetric and
    antisymmetric components.

    If ``savepath`` is provided and ``include_components=True``, component
    figures are saved by appending ``_symmetric`` and ``_antisymmetric`` before
    the extension.
    """
    plot_init(
        W,
        title=title,
        suptitle=suptitle,
        show_unit_circle=show_unit_circle,
        unit_radius=unit_radius,
        trace_band_summary=trace_band_summary,
        trace_lw=trace_lw,
        savepath=savepath,
        show=show,
    )

    if not include_components:
        return

    W_sym, W_asym = decompose_sym_asym(W)

    base = suptitle if suptitle is not None else title

    plot_init(
        W_sym,
        title=f"{title} symmetric component",
        suptitle=f"{base}: symmetric component",
        show_unit_circle=show_unit_circle,
        unit_radius=unit_radius,
        trace_band_summary=trace_band_summary,
        trace_lw=trace_lw,
        savepath=component_savepath(savepath, "symmetric"),
        show=show,
    )

    plot_init(
        W_asym,
        title=f"{title} antisymmetric component",
        suptitle=f"{base}: antisymmetric component",
        show_unit_circle=show_unit_circle,
        unit_radius=unit_radius,
        trace_band_summary=trace_band_summary,
        trace_lw=trace_lw,
        savepath=component_savepath(savepath, "antisymmetric"),
        show=show,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot initial recurrent connectivity from a Whh.npy file, including "
            "the full matrix and optionally its symmetric/antisymmetric components."
        )
    )

    parser.add_argument(
        "whh_path",
        type=str,
        help="Path to recurrent weight matrix .npy file.",
    )

    parser.add_argument(
        "--savepath",
        type=str,
        default=None,
        help=(
            "Optional path to save the full-matrix figure. If components are "
            "enabled, component figures are saved with _symmetric and "
            "_antisymmetric suffixes."
        ),
    )

    parser.add_argument(
        "--title",
        type=str,
        default="Weights",
        help="Panel/figure title label.",
    )

    parser.add_argument(
        "--suptitle",
        type=str,
        default=None,
        help="Main figure title. If omitted, uses --title.",
    )

    parser.add_argument(
        "--alpha-label",
        type=str,
        default=None,
        help="Optional alpha label, e.g. 0.70, used in title.",
    )

    parser.add_argument(
        "--trace-band-summary",
        choices=["trace", "bandmean"],
        default="trace",
        help="How to summarize diagonal offsets.",
    )

    parser.add_argument(
        "--trace-lw",
        type=float,
        default=2.0,
        help="Line width for diagonal trace panel.",
    )

    parser.add_argument(
        "--unit-radius",
        type=float,
        default=1.0,
        help="Radius for unit circle reference.",
    )

    parser.add_argument(
        "--no-unit-circle",
        action="store_true",
        help="Do not show the unit circle on eigenspectrum.",
    )

    parser.add_argument(
        "--no-components",
        action="store_true",
        help="Only plot the full matrix; do not plot symmetric/antisymmetric components.",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save only; do not display interactively.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    W = np.load(args.whh_path)

    title = args.title
    suptitle = args.suptitle

    if args.alpha_label is not None:
        title = f"Whh (α={args.alpha_label})"
        if suptitle is None:
            suptitle = f"Mexican hat initial connectivity (α={args.alpha_label})"

    plot_init_with_components(
        W,
        title=title,
        suptitle=suptitle,
        show_unit_circle=not args.no_unit_circle,
        unit_radius=args.unit_radius,
        trace_band_summary=args.trace_band_summary,
        trace_lw=args.trace_lw,
        savepath=args.savepath,
        show=not args.no_show,
        include_components=not args.no_components,
    )


if __name__ == "__main__":
    main()
