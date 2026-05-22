#!/usr/bin/env python3
"""
Generate the STDP analysis figure for the Phil. Trans. paper.

This script creates a deterministic four-panel summary figure from the reusable
analysis functions in ``src/analyze/stdp_analysis.py``:

1. modified STDP kernel
2. spatial recurrent weight profile
3. recurrent eigenspectrum
4. simulated ring-network dynamics

Example
-------
Run from the repository root::

    python src/figures/figure_stdp.py \
        --epsilon 1.0 \
        --savepath data/figures/stdp/STDP_eps1p00.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

# Allow running as ``python src/figures/figure_stdp.py`` from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.stdp.stdp_analysis import STDPParameters, run_stdp_analysis


def plot_stdp_summary(
    results: dict,
    savepath: Optional[Path] = None,
    svg_savepath: Optional[Path] = None,
    figsize: tuple[float, float] = (12.0, 3.0),
    fontsize: int = 10,
    warmup_steps: int = 5,
    show: bool = False,
) -> plt.Figure:
    """
    Plot the four-panel STDP summary figure.

    Parameters
    ----------
    results : dict
        Output from ``run_stdp_analysis``.
    savepath : Path, optional
        File path for the raster/vector figure.
    svg_savepath : Path, optional
        Optional additional SVG output path.
    figsize : tuple, default=(12, 3)
        Figure size in inches.
    fontsize : int, default=10
        Base font size for labels and titles.
    warmup_steps : int, default=5
        Initial time steps omitted from the dynamics panel.
    show : bool, default=False
        If True, display the figure interactively.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    params = results["params"]
    delays = results["delays"]
    stdp_kernel = results["stdp_kernel"]
    diag_offsets = results["diag_offsets"]
    weight_profile = results["weight_profile"]
    eigvals = results["eigvals_num"]
    history = results["history"]

    plt.rcParams.update({"font.size": fontsize})

    fig = plt.figure(figsize=figsize)

    # Panel 1: STDP kernel.
    ax = fig.add_subplot(1, 4, 1)
    ax.plot(delays, stdp_kernel, "k-", lw=2)
    ax.axhline(0, color="gray", linestyle=":", lw=1)
    ax.axvline(0, color="gray", linestyle=":", lw=1)
    ax.set_title(f"STDP Kernel ($\\epsilon={params.epsilon:g}$)")
    ax.set_xlabel("Delay $\\Delta t = t_{post} - t_{pre}$ (ms)")
    ax.set_ylabel("$\\Delta W_{ij}$")
    ax.grid(True, alpha=0.3)

    # Panel 2: spatial weight profile.
    ax = fig.add_subplot(1, 4, 2)
    ax.plot(diag_offsets, weight_profile, "k-", lw=2, label="Weight Profile")
    ax.axhline(0, color="gray", linestyle=":", lw=1)
    ax.axvline(0, color="gray", linestyle=":", lw=1)
    ax.set_title(f"Spatial Weight Profile ($\\epsilon={params.epsilon:g}$)")
    ax.set_xlabel("Diagonal Offset ($i-j$)")
    ax.set_ylabel("$W_{ij}$")
    ax.grid(True, alpha=0.3)
    ax.set_xlim((-25, 25))

    # Panel 3: eigenspectrum.
    ax = fig.add_subplot(1, 4, 3)
    unit_circle = plt.Circle((0, 0), 1, fill=False, color="k")
    ax.add_patch(unit_circle)
    ax.scatter(
        np.real(eigvals),
        np.imag(eigvals),
        c="blue",
        s=20,
        alpha=0.8,
        edgecolors="k",
        lw=0.5,
        label="Numerical",
    )
    ax.axhline(0, color="k", linestyle="-", lw=0.5)
    ax.axvline(0, color="k", linestyle="-", lw=0.5)
    ax.set_title(f"Eigenvalue Spectrum ($\\epsilon={params.epsilon:g}$)")
    ax.set_xlabel("Re$(\\lambda)$")
    ax.set_ylabel("Im$(\\lambda)$")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim((-1.5, 1.5))
    ax.set_ylim((-1.5, 1.5))

    # Panel 4: dynamics.
    ax = fig.add_subplot(1, 4, 4)
    im = ax.imshow(history[warmup_steps:].T, aspect="auto", cmap="viridis", origin="upper")
    fig.colorbar(im, ax=ax, label="Firing Rate")
    ax.set_title(f"Network Dynamics ($\\epsilon={params.epsilon:g}$)")
    ax.set_ylabel("Neuron Index")
    ax.set_xlabel("Time Step")

    fig.tight_layout()

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight")
        print(f"Saved figure: {savepath}")

    if svg_savepath is not None:
        svg_savepath = Path(svg_savepath)
        svg_savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(svg_savepath, bbox_inches="tight")
        print(f"Saved figure: {svg_savepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate the STDP summary figure.")
    parser.add_argument("--epsilon", type=float, default=1.0, help="STDP symmetry parameter.")
    parser.add_argument("--a", type=float, default=1.0, help="Input amplitude.")
    parser.add_argument("--b", type=float, default=1.0, help="Input precision.")
    parser.add_argument("--mu", type=float, default=10.0, help="Initial Gaussian bump center.")
    parser.add_argument("--A-plus", type=float, default=1.0, help="LTP amplitude.")
    parser.add_argument("--tau-plus", type=float, default=1.0, help="LTP time constant.")
    parser.add_argument("--A-minus", type=float, default=0.5, help="LTD amplitude.")
    parser.add_argument("--tau-minus", type=float, default=3.0, help="LTD time constant.")
    parser.add_argument("--n-neurons", type=int, default=200, help="Number of ring neurons.")
    parser.add_argument("--steps", type=int, default=200, help="Simulation steps.")
    parser.add_argument("--dt", type=float, default=1.0, help="Simulation time step.")
    parser.add_argument("--fontsize", type=int, default=10, help="Base plot font size.")
    parser.add_argument("--savepath", type=Path, default=Path("data/figures/stdp/STDP_eps1p00.png"), help="Output figure path.")
    parser.add_argument("--svg-savepath", type=Path, default=None, help="Optional SVG output path.")
    parser.add_argument("--no-show", action="store_true", help="Do not display figure interactively.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the analysis and generate the STDP figure."""
    args = parse_args(argv)

    params = STDPParameters(
        epsilon=args.epsilon,
        a=args.a,
        b=args.b,
        mu=args.mu,
        A_plus=args.A_plus,
        tau_plus=args.tau_plus,
        A_minus=args.A_minus,
        tau_minus=args.tau_minus,
        n_neurons=args.n_neurons,
        steps=args.steps,
        dt=args.dt,
    )
    results = run_stdp_analysis(params)
    plot_stdp_summary(
        results,
        savepath=args.savepath,
        svg_savepath=args.svg_savepath,
        fontsize=args.fontsize,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
