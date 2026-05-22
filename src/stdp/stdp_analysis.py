#!/usr/bin/env python3
"""
STDP analysis utilities for the Phil. Trans. paper.

This module contains the reusable, non-interactive calculations extracted from
``STDP_paper.py``. It computes modified STDP kernels, recurrent weight matrices,
eigenspectra, ring-network dynamics, and Fourier-mode summaries used by the STDP
analysis/figure scripts.

The functions here intentionally avoid plotting and notebook-specific code so
that they can be imported by both command-line figure scripts and interactive
notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional

import numpy as np
from scipy.special import erfc


SignDelta = Literal["forward", "backward"]


@dataclass(frozen=True)
class STDPParameters:
    """Parameter bundle for the STDP ring-network analysis."""

    epsilon: float = 1.0
    a: float = 1.0
    b: float = 1.0
    mu: float = 10.0
    A_plus: float = 1.0
    tau_plus: float = 1.0
    A_minus: float = 0.5
    tau_minus: float = 3.0
    n_neurons: int = 200
    steps: int = 200
    dt: float = 1.0
    tau_neuron: float = 1.0
    matrix_scale: float = 1.2
    analytic_eig_scale: float = 1.4
    delay_min: int = -60
    delay_max: int = 60
    delay_step: int = 1
    stdp_sigma: float = 5.0
    integration_points: int = 2000


def gaussian_overlap(diff_centers: np.ndarray, sigma: float) -> np.ndarray:
    """
    Return the overlap integral of two normalized Gaussians.

    Parameters
    ----------
    diff_centers : np.ndarray
        Difference between Gaussian centers.
    sigma : float
        Standard deviation of each Gaussian.

    Returns
    -------
    np.ndarray
        Gaussian overlap evaluated at ``diff_centers``.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    effective_sigma = np.sqrt(2.0) * sigma
    return (1.0 / (np.sqrt(2.0 * np.pi) * effective_sigma)) * np.exp(
        -(diff_centers**2) / (2.0 * effective_sigma**2)
    )


def compute_modified_stdp(
    delays: Iterable[float],
    epsilon: float = 0.0,
    sigma: float = 5.0,
    tau_pos: float = 20.0,
    tau_neg: float = 20.0,
    A_pos: float = 1.0,
    A_neg: float = 1.0,
    integration_points: int = 2000,
) -> np.ndarray:
    """
    Compute a modified STDP kernel using the double-integral rule.

    The calculation matches the original interactive notebook logic:

    ``W = ∫ K(s) Inner1(s) ds + epsilon ∫ K(s) Inner2(s) ds``

    Parameters
    ----------
    delays : iterable of float
        Time differences ``t_post - t_pre``.
    epsilon : float, default=0.0
        Scaling factor for the secondary, transposed interaction term.
    sigma : float, default=5.0
        Width of the Gaussian activity profile.
    tau_pos, tau_neg : float, default=20.0
        Positive and negative STDP time constants.
    A_pos, A_neg : float, default=1.0
        Positive and negative STDP amplitudes.
    integration_points : int, default=2000
        Number of numerical integration points over the lag variable.

    Returns
    -------
    np.ndarray
        Modified STDP kernel values, reversed to preserve the plotting
        convention used in the original notebook.
    """
    delays = np.asarray(list(delays), dtype=float)
    if delays.ndim != 1:
        raise ValueError("delays must be one-dimensional.")
    if tau_pos <= 0 or tau_neg <= 0:
        raise ValueError("tau_pos and tau_neg must be positive.")
    if integration_points < 2:
        raise ValueError("integration_points must be at least 2.")

    limit = max(tau_pos, tau_neg, sigma) * 6.0
    s_values = np.linspace(-limit, limit, integration_points)

    K_s = np.where(
        s_values > 0,
        A_pos * np.exp(-s_values / tau_pos),
        -A_neg * np.exp(s_values / tau_neg),
    )

    weight_changes = []
    for delay in delays:
        inner_1 = gaussian_overlap(delay - s_values, sigma)
        term1_integral = np.trapezoid(K_s * inner_1, s_values)

        inner_2 = gaussian_overlap(-delay - s_values, sigma)
        term2_integral = np.trapezoid(K_s * inner_2, s_values)

        weight_changes.append(term1_integral + epsilon * term2_integral)

    return np.asarray(weight_changes)[::-1]


def calculate_stdp_term(
    delta: np.ndarray,
    a: float,
    b: float,
    A: float,
    tau: float,
    sign_delta: SignDelta,
) -> np.ndarray:
    """
    Compute one analytic STDP contribution for a spatial offset matrix.

    Parameters
    ----------
    delta : np.ndarray
        Matrix of signed ring offsets.
    a : float
        Input amplitude.
    b : float
        Input precision.
    A : float
        STDP amplitude.
    tau : float
        STDP time constant.
    sign_delta : {'forward', 'backward'}
        Direction convention for the delay term.
    """
    if b <= 0:
        raise ValueError("b must be positive.")
    if tau <= 0:
        raise ValueError("tau must be positive.")
    if sign_delta not in {"forward", "backward"}:
        raise ValueError("sign_delta must be 'forward' or 'backward'.")

    sign = -1.0 if sign_delta == "forward" else 1.0
    arg_exp = (1.0 / (b * tau**2)) + (sign * delta / tau)
    arg_erfc = (1.0 + (sign * b * tau * delta)) / (tau * np.sqrt(2.0 * b))
    return A * np.exp(arg_exp) * erfc(arg_erfc)


def ring_delta_matrix(n_neurons: int) -> np.ndarray:
    """Return a signed periodic distance matrix for a ring of neurons."""
    if n_neurons <= 0:
        raise ValueError("n_neurons must be positive.")

    i, j = np.meshgrid(np.arange(n_neurons), np.arange(n_neurons), indexing="ij")
    diff = i - j
    delta = diff.copy()
    delta[diff > n_neurons // 2] -= n_neurons
    delta[diff < -n_neurons // 2] += n_neurons
    return delta


def build_stdp_matrix(
    epsilon: float = 1.0,
    a: float = 1.0,
    b: float = 1.0,
    A_plus: float = 1.0,
    tau_plus: float = 1.0,
    A_minus: float = 0.5,
    tau_minus: float = 3.0,
    n_neurons: int = 200,
    matrix_scale: float = 1.2,
) -> Dict[str, np.ndarray | float]:
    """
    Build the recurrent matrix induced by the modified STDP rule.

    Returns the total matrix along with the asymmetric and transposed
    components used to construct it.
    """
    delta = ring_delta_matrix(n_neurons)
    factor = (np.pi * (a**2)) / (2.0 * b)

    term_plus = calculate_stdp_term(delta, a, b, A_plus, tau_plus, "forward")
    term_minus = calculate_stdp_term(delta, a, b, A_minus, tau_minus, "backward")
    W_stdp = factor * (term_plus - term_minus)

    term_plus_T = calculate_stdp_term(-delta, a, b, A_plus, tau_plus, "forward")
    term_minus_T = calculate_stdp_term(-delta, a, b, A_minus, tau_minus, "backward")
    W_transposed = factor * (term_plus_T - term_minus_T)

    W_total = W_stdp + epsilon * W_transposed
    norm_factor = float(np.linalg.norm(W_total, ord=2))
    if norm_factor == 0:
        raise ValueError("STDP matrix has zero spectral norm and cannot be scaled.")

    W_total = W_total * matrix_scale / norm_factor

    return {
        "W_total": W_total,
        "W_stdp": W_stdp,
        "W_transposed": W_transposed,
        "delta_matrix": delta,
        "norm_factor": norm_factor,
    }


def compute_weight_profile(W: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute the mean weight as a function of diagonal offset ``i-j``."""
    W = np.asarray(W)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix.")

    n = W.shape[0]
    diag_offsets = np.arange(-(n - 1), n, 1)
    offset_values = {int(offset): [] for offset in diag_offsets}

    for i in range(n):
        for j in range(n):
            offset_values[i - j].append(W[i, j])

    weight_profile = np.asarray(
        [np.mean(offset_values[int(offset)]) for offset in diag_offsets]
    )
    return {"diag_offsets": diag_offsets, "weight_profile": weight_profile}


def compute_eigenspectra(
    W: np.ndarray,
    epsilon: float,
    a: float,
    b: float,
    A_plus: float,
    tau_plus: float,
    A_minus: float,
    tau_minus: float,
    analytic_eig_scale: float = 1.4,
) -> Dict[str, np.ndarray]:
    """Compute numerical and analytic eigenspectra for the STDP matrix."""
    W = np.asarray(W)
    n = W.shape[0]

    eigvals_num = np.linalg.eigvals(W)
    omega = 2.0 * np.pi * np.fft.fftfreq(n)

    eigvals_analytic = (np.pi / b) * (a**2) * np.exp(-(omega**2) / (2.0 * b)) + 0j
    eigvals_analytic *= (
        (1.0 + epsilon)
        * (
            A_plus * tau_plus / (1.0 + (omega**2) * (tau_plus**2))
            - A_minus * tau_minus / (1.0 + (omega**2) * (tau_minus**2))
        )
        - 1j
        * omega
        * (1.0 - epsilon)
        * (
            A_plus * (tau_plus**2) / (1.0 + (omega**2) * (tau_plus**2))
            + A_minus * (tau_minus**2) / (1.0 + (omega**2) * (tau_minus**2))
        )
    )

    eig_norm = float(np.max(np.abs(eigvals_analytic)))
    if eig_norm > 0:
        eigvals_analytic = eigvals_analytic * analytic_eig_scale / eig_norm

    eigvals_discrete = np.fft.fft(W[0, :])

    return {
        "eigvals_num": eigvals_num,
        "eigvals_analytic": eigvals_analytic,
        "eigvals_discrete": eigvals_discrete,
        "omega": omega,
    }


def simulate_ring_dynamics(
    W: np.ndarray,
    a: float = 1.0,
    b: float = 1.0,
    mu: float = 10.0,
    steps: int = 200,
    dt: float = 1.0,
    tau_neuron: float = 1.0,
    use_discrete_update: bool = True,
) -> np.ndarray:
    """
    Simulate recurrent ring dynamics from a Gaussian initial condition.

    By default, this preserves the final update convention from the notebook,
    where the Euler update is computed but the state is then set to the recurrent
    input. Set ``use_discrete_update=False`` to use the continuous Euler update.
    """
    W = np.asarray(W)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if tau_neuron <= 0:
        raise ValueError("tau_neuron must be positive.")

    n = W.shape[0]
    norm = float(np.linalg.norm(W, ord=2))
    if norm == 0:
        raise ValueError("W has zero spectral norm and cannot be normalized.")

    u = a * np.exp(-b * (np.arange(n) - mu) ** 2)
    history = np.zeros((steps, n), dtype=float)

    for t in range(steps):
        history[t] = u.copy()
        recurrent_input = np.dot(W / norm, u)
        du = (-u + np.tanh(recurrent_input)) / tau_neuron
        euler_update = u + du * dt
        u = recurrent_input if use_discrete_update else euler_update

    return history


def analyze_fourier_modes(history: np.ndarray, dt: float = 1.0) -> Dict[str, np.ndarray | float]:
    """Compute Fourier-mode power and phase drift from population activity."""
    history = np.asarray(history)
    if history.ndim != 2:
        raise ValueError("history must have shape (steps, n_neurons).")

    steps, n = history.shape
    time_axis = np.arange(steps) * dt
    u_fft = np.fft.fft(history, axis=1)
    power_spectrum = (np.abs(u_fft) / n) ** 2
    phase_k1 = np.angle(u_fft[:, 1])
    phase_unwrapped = np.unwrap(phase_k1)

    drift_velocity = np.nan
    fit = np.asarray([np.nan, np.nan])
    fit_line = np.full_like(time_axis, np.nan, dtype=float)

    if steps > 10:
        fit = np.polyfit(time_axis[steps // 2 :], phase_unwrapped[steps // 2 :], 1)
        slope = float(fit[0])
        drift_velocity = (slope / (2.0 * np.pi)) * n * (1000.0 if dt < 1.0 else 1.0)
        fit_line = np.polyval(fit, time_axis)

    return {
        "time_axis": time_axis,
        "power_spectrum": power_spectrum,
        "phase_k1": phase_k1,
        "phase_unwrapped": phase_unwrapped,
        "fit_coefficients": fit,
        "fit_line": fit_line,
        "drift_velocity_neurons_per_time": drift_velocity,
    }


def run_stdp_analysis(params: Optional[STDPParameters] = None, **overrides) -> Dict[str, object]:
    """
    Run the full STDP analysis pipeline and return all computed arrays.

    Parameters can be provided either as an ``STDPParameters`` object or as
    keyword overrides, for example ``run_stdp_analysis(epsilon=0.5)``.
    """
    if params is None:
        params = STDPParameters()
    if overrides:
        params = STDPParameters(**{**params.__dict__, **overrides})

    delays = np.arange(params.delay_min, params.delay_max, params.delay_step)
    stdp_kernel = -compute_modified_stdp(
        delays,
        epsilon=params.epsilon,
        sigma=params.stdp_sigma,
        tau_pos=params.tau_plus,
        tau_neg=params.tau_minus,
        A_pos=params.A_plus,
        A_neg=params.A_minus,
        integration_points=params.integration_points,
    )

    matrix_results = build_stdp_matrix(
        epsilon=params.epsilon,
        a=params.a,
        b=params.b,
        A_plus=params.A_plus,
        tau_plus=params.tau_plus,
        A_minus=params.A_minus,
        tau_minus=params.tau_minus,
        n_neurons=params.n_neurons,
        matrix_scale=params.matrix_scale,
    )
    W_total = matrix_results["W_total"]

    profile_results = compute_weight_profile(W_total)
    eig_results = compute_eigenspectra(
        W_total,
        epsilon=params.epsilon,
        a=params.a,
        b=params.b,
        A_plus=params.A_plus,
        tau_plus=params.tau_plus,
        A_minus=params.A_minus,
        tau_minus=params.tau_minus,
        analytic_eig_scale=params.analytic_eig_scale,
    )
    history = simulate_ring_dynamics(
        W_total,
        a=params.a,
        b=params.b,
        mu=params.mu,
        steps=params.steps,
        dt=params.dt,
        tau_neuron=params.tau_neuron,
    )
    fourier = analyze_fourier_modes(history, dt=params.dt)

    return {
        "params": params,
        "delays": delays,
        "stdp_kernel": stdp_kernel,
        **matrix_results,
        **profile_results,
        **eig_results,
        "history": history,
        "fourier": fourier,
    }
