#!/usr/bin/env python3
"""
Build synthetic sequential Gaussian-bump input sequences.

This script is a reproducible command-line entry point for generating the input sequence used in predictive sequence-learning experiments.

Each time step contains a Gaussian-like activity bump across a population of
units. The bump is shifted across neurons to create a structured sequential
population code, then downsampled to the desired number of time steps.

The main output is ``X_mini``, a tensor of shape ``(1, T, N)``, where:
    - 1 is the batch dimension
    - T is the number of sampled time steps
    - N is the number of neurons / units

For these experiments, the target is identical to the input:
``Target_mini = X_mini``.

By default, this script builds the input sequence and prints summary statistics
without saving files. Use ``--save-outputs`` to save both the tensor checkpoint
and a PNG preview figure, matching the behavior of the original notebook when
``save_outputs = True``.

Examples
--------
Build the default sequence and print activity statistics::

    python src/setup/build_inputs.py

Save the generated tensors and diagnostic PNG to data/inputs::

    python src/setup/build_inputs.py --save-outputs

Save outputs to a custom directory::

    python src/setup/build_inputs.py --save-outputs --output-dir data/inputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

# Use a non-interactive backend so this script works cleanly on servers/HPC.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm


def activity_stats(X_mini: torch.Tensor | np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Summarize the fraction of active and inactive entries in an encoding matrix.

    Parameters
    ----------
    X_mini : torch.Tensor or np.ndarray
        Encoding array of shape ``(1, T, N)`` or ``(T, N)``.
    threshold : float, default=0.5
        Values greater than or equal to this threshold are counted as active.

    Returns
    -------
    dict
        Dictionary containing:
        - ``frac_active``: fraction of entries above threshold
        - ``frac_inactive``: fraction of entries below threshold
        - ``active_inactive_ratio``: active / inactive ratio
    """
    if isinstance(X_mini, torch.Tensor):
        arr = X_mini.detach().cpu().numpy()
    else:
        arr = np.asarray(X_mini)

    arr = arr.reshape(-1)

    total = arr.size
    active = np.sum(arr >= threshold)
    inactive = total - active

    return {
        "frac_active": float(active / total),
        "frac_inactive": float(inactive / total),
        "active_inactive_ratio": float(active / inactive) if inactive > 0 else float("inf"),
    }


def build_inputs(
    n_neurons: int = 100,
    total_steps: int = 2000,
    sampled_steps: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a downsampled sequential Gaussian-bump input tensor.

    Parameters
    ----------
    n_neurons : int, default=100
        Number of neurons / units in the population code.
    total_steps : int, default=2000
        Number of high-resolution time steps used before downsampling.
    sampled_steps : int, default=100
        Number of time steps retained in the final sequence.

    Returns
    -------
    X_mini : torch.Tensor
        Input tensor of shape ``(1, sampled_steps, n_neurons)``.
    Target_mini : torch.Tensor
        Target tensor. For these experiments, this is identical to ``X_mini``.
    """
    if n_neurons <= 0:
        raise ValueError("n_neurons must be positive.")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    if sampled_steps <= 0:
        raise ValueError("sampled_steps must be positive.")
    if total_steps % 2 != 0:
        raise ValueError("total_steps must be even because the template is half Gaussian and half zeros.")
    if total_steps % n_neurons != 0:
        raise ValueError("total_steps must be divisible by n_neurons.")
    if total_steps % sampled_steps != 0:
        raise ValueError("total_steps must be divisible by sampled_steps.")

    # The template consists of a Gaussian bump followed by zeros. Rolling this
    # template across neurons produces a sequential activation pattern.
    gaussian_support = np.linspace(norm.ppf(0.05), norm.ppf(0.95), total_steps // 2)
    gaussian_bump = norm.pdf(gaussian_support)
    template = np.concatenate([gaussian_bump, np.zeros(total_steps // 2)])

    # X_full has shape (n_neurons, total_steps): rows are neurons, columns are
    # time points.
    X_full = np.zeros((n_neurons, total_steps), dtype=np.float32)

    shift_per_neuron = total_steps // n_neurons
    for neuron_idx in range(n_neurons):
        X_full[neuron_idx, :] = np.roll(template, neuron_idx * shift_per_neuron)

    selected_steps = np.arange(0, total_steps, total_steps // sampled_steps, dtype=int)

    # Final shape: (1, sampled_steps, n_neurons), where the first dimension is
    # batch size.
    X_np = np.expand_dims(X_full[:, selected_steps].T, axis=0)
    X_np = X_np / X_np.max()

    X_mini = torch.tensor(X_np.astype(np.float32))
    Target_mini = X_mini

    return X_mini, Target_mini


def default_stem(n_neurons: int, sampled_steps: int) -> str:
    """
    Return the default filename stem.

    Example
    -------
    ``Ns100_SeqN100_asym1``
    """
    return f"Ns{n_neurons}_SeqN{sampled_steps}_asym1"


def save_inputs(savepath: Path, X_mini: torch.Tensor, Target_mini: torch.Tensor) -> None:
    """Save generated input and target tensors to disk."""
    savepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"X_mini": X_mini, "Target_mini": Target_mini}, savepath)


def save_input_plot(savepath: Path, X_mini: torch.Tensor) -> None:
    """
    Save a diagnostic heatmap of the generated sequential Gaussian-bump input.
    """
    savepath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(X_mini[0].T.detach().cpu().numpy(), aspect="auto")
    fig.colorbar(im, ax=ax, label="Activity")
    ax.set_title("Sequential Gaussian-bump input")
    ax.set_ylabel("Neuron")
    ax.set_xlabel("Time step")
    fig.tight_layout()
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build synthetic sequential Gaussian-bump input sequences."
    )
    parser.add_argument("--n-neurons", type=int, default=100, help="Number of neurons / units.")
    parser.add_argument(
        "--total-steps",
        type=int,
        default=2000,
        help="Number of high-resolution time steps before downsampling.",
    )
    parser.add_argument(
        "--sampled-steps",
        type=int,
        default=100,
        help="Number of sampled time steps in the final sequence.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Activity threshold used when printing summary statistics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/inputs"),
        help="Directory where outputs are saved when --save-outputs is used.",
    )
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help=(
            "Save both the generated tensor checkpoint and diagnostic PNG preview "
            "using the historical notebook naming convention."
        ),
    )
    parser.add_argument(
        "--savepath",
        type=Path,
        default=None,
        help=(
            "Optional explicit tensor checkpoint path. This is kept for backwards "
            "compatibility. If provided, tensors are saved even without --save-outputs."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """Build inputs, print summary statistics, and optionally save outputs."""
    args = parse_args(argv)

    X_mini, Target_mini = build_inputs(
        n_neurons=args.n_neurons,
        total_steps=args.total_steps,
        sampled_steps=args.sampled_steps,
    )

    stats = activity_stats(X_mini, threshold=args.threshold)

    print(f"X_mini shape: {tuple(X_mini.shape)}")
    print(f"Target_mini shape: {tuple(Target_mini.shape)}")
    print("Activity statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.6f}")

    stem = default_stem(args.n_neurons, args.sampled_steps)

    if args.save_outputs:
        tensor_path = args.output_dir / f"{stem}.pth.tar"
        figure_path = args.output_dir / f"{stem}.png"

        save_inputs(tensor_path, X_mini, Target_mini)
        save_input_plot(figure_path, X_mini)

        print(f"Saved tensors to: {tensor_path}")
        print(f"Saved diagnostic figure to: {figure_path}")

    if args.savepath is not None:
        save_inputs(args.savepath, X_mini, Target_mini)
        print(f"Saved tensors to: {args.savepath}")


if __name__ == "__main__":
    main()
