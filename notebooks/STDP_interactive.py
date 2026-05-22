"""
Interactive STDP exploration tool for the Phil. Trans. paper.

This marimo application was used to explore how different spike-timing-
dependent plasticity (STDP) kernels shape recurrent connectivity,
eigenspectra, and traveling-wave dynamics in ring networks.

The application allows interactive manipulation of:
    - STDP symmetry (epsilon)
    - LTP/LTD amplitudes and time constants
    - input bump parameters

and visualizes:
    - the resulting STDP kernel
    - emergent spatial weight profiles
    - eigenspectra of the recurrent connectivity matrix
    - simulated network dynamics
    - Fourier-mode evolution of activity patterns

This notebook is intended for interactive exploration and supplemental
analysis. Reproducible manuscript figures are generated separately by the
scripts in ``src/figures/``.

Run with:
    marimo run notebooks/STDP_interactive.py
"""

import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    # STDP Plots for Philtrans paper
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import erfc
    # import seaborn as sns
    # sns.set_theme()

    # import scienceplots

    # plt.style.use('science')
    return erfc, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    # --- UI Controls ---

    # 1. Symmetry Control
    eps_slider = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=1.0, label="Symmetry (epsilon)")

    # 2. Input Parameters
    a_slider = mo.ui.slider(start=0.0, stop=2.0, step=0.1, value=1.0, label="Input Amplitude (a)")
    b_slider = mo.ui.slider(start=0.0, stop=10.0, step=0.01, value=1.0, label="Input Precision (b)")
    mu_slider = mo.ui.slider(start=0.0, stop=100.0, step=1.0, value=10.0, label="Initial Bump (mu)")

    # 3. LTP Parameters
    Ap_slider = mo.ui.slider(start=0.0, stop=2.0, step=0.1, value=1.0, label="LTP Amplitude (A+)")
    taup_slider = mo.ui.slider(start=0.0, stop=5.0, step=0.1, value=1.0, label="LTP Time Constant (tau+)")

    # 4. LTD Parameters
    Am_slider = mo.ui.slider(start=0.0, stop=2.0, step=0.1, value=0.5, label="LTD Amplitude (A-)")
    taum_slider = mo.ui.slider(start=0.0, stop=5.0, step=0.1, value=3.0, label="LTD Time Constant (tau-)")

    # Layout
    controls = mo.vstack([
        mo.md("### 1. Symmetry Control"),
        eps_slider,
        mo.md("### 2. Input Settings"),
        mo.hstack([a_slider, b_slider, mu_slider]),
        mo.md("### 3. LTP Settings"),
        mo.hstack([Ap_slider, taup_slider]),
        mo.md("### 4. LTD Settings"),
        mo.hstack([Am_slider, taum_slider])
    ])

    controls
    return (
        Am_slider,
        Ap_slider,
        a_slider,
        b_slider,
        eps_slider,
        mu_slider,
        taum_slider,
        taup_slider,
    )


@app.cell(hide_code=True)
def _(np):
    def compute_modified_stdp(delays, epsilon=0.0, sigma=5.0, 
                              tau_pos=20.0, tau_neg=20.0, 
                              A_pos=1.0, A_neg=1.0):
        """
        Computes the modified STDP learning kernel based on the double-integral rule:
        W = Int[ K(s) * Inner1(s) ] + epsilon * Int[ K(s) * Inner2(s) ]

        Args:
            delays (np.array): Array of time differences (t_post - t_pre) in ms.
            epsilon (float): Scaling factor for the secondary interaction term.
            sigma (float): Standard deviation (width) of the Gaussian activity.
            tau_pos, tau_neg (float): Time constants for the STDP window K(s).
            A_pos, A_neg (float): Amplitudes for the STDP window K(s).

        Returns:
            np.array: The computed weight change W_ij for each delay.
        """

        # 1. Define the integration range for 's' (the lag variable)
        # We need a range wide enough to cover the decay of K(s) and the spread of Gaussians
        limit = max(tau_pos, tau_neg, sigma) * 6
        s_values = np.linspace(-limit, limit, 2000)
        ds = s_values[1] - s_values[0]

        # 2. Pre-compute K(s) for the s_range
        # K(s) is the standard STDP exponential window
        K_s = np.where(s_values > 0, 
                       A_pos * np.exp(-s_values / tau_pos), 
                       -A_neg * np.exp(s_values / tau_neg))

        weight_changes = []

        # Helper: Analytical solution for Int[-inf, inf] u_a(t) * u_b(t-s) dt
        # This is the overlap of two Gaussians. 
        # If Gaussians have variance sigma^2, the integral is a Gaussian of the 
        # difference in their centers with variance 2*sigma^2.
        def gaussian_overlap(diff_centers, sigma):
            # The integral of the product of two normalized Gaussians
            effective_sigma = np.sqrt(2) * sigma
            return (1.0 / (np.sqrt(2 * np.pi) * effective_sigma)) * np.exp(-(diff_centers)**2 / (2 * effective_sigma**2))

        for D in delays:
            # D = t_post - t_pre

            # --- TERM 1: Int K(s) * [ Int u_i(t) u_j(t-s) dt ] ---
            # Center of u_i(t) is D (relative to u_j at 0)
            # Center of u_j(t-s) is s
            # Difference = D - s
            inner_1 = gaussian_overlap(D - s_values, sigma)

            # Integrate Term 1 over s
            term1_integral = np.trapz(K_s * inner_1, s_values)

            # --- TERM 2: epsilon * Int K(s) * [ Int u_j(t) u_i(t-s) dt ] ---
            # Center of u_j(t) is 0
            # Center of u_i(t-s) is D + s
            # Difference = 0 - (D + s) = -D - s
            inner_2 = gaussian_overlap(-D - s_values, sigma)

            # Integrate Term 2 over s
            term2_integral = np.trapz(K_s * inner_2, s_values)

            # Total Weight Change
            W_ij = term1_integral + (epsilon * term2_integral)
            weight_changes.append(W_ij)

        return np.array(weight_changes)[::-1]
    return (compute_modified_stdp,)


@app.cell(hide_code=True)
def _(
    Am_slider,
    Ap_slider,
    a_slider,
    b_slider,
    compute_modified_stdp,
    eps_slider,
    erfc,
    mo,
    mu_slider,
    np,
    plt,
    taum_slider,
    taup_slider,
):
    # --- Calculation Engine ---
    N = 200  # Number of neurons for numerical check

    # 1. Parameter Retrieval
    epsilon = eps_slider.value
    a = a_slider.value
    b = b_slider.value
    A_plus = Ap_slider.value
    tau_plus = taup_slider.value
    A_minus = Am_slider.value
    tau_minus = taum_slider.value
    mu = mu_slider.value

    # --- Part A: Spatial Weight Profile (Numerical Matrix) ---

    def calculate_stdp_term(delta, a, b, A, tau, sign_delta):
        """Helper for analytic weights"""
        # sign_delta controls Forward (-delta) vs Backward (+delta) logic
        # For standard STDP: LTP uses -delta, LTD uses +delta
        sign = -1.0 if sign_delta == 'forward' else 1.0

        arg_exp = (1.0 / (b * tau**2)) + (sign * delta / tau)
        arg_erfc = (1.0 + (sign * b * tau * delta)) / (tau * np.sqrt(2*b))

        return A * np.exp(arg_exp) * erfc(arg_erfc)

    def get_cross_diagonal(W):
        """
        Extracts the anti-diagonal (cross diagonal) from matrix W.
        Corresponds to elements W[i, N-1-i].
        """
        # np.fliplr flips the matrix left-to-right
        # Then .diagonal() takes the main diagonal of the flipped matrix
        return np.diagonal(np.fliplr(W))

    # Global Prefactor
    factor = (np.pi * (a**2)) / (2 * b)

    # Create Matrix Grid
    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    diff = I - J
    # Periodic boundary distances
    delta_matrix = diff.copy()
    delta_matrix[diff > N//2] -= N
    delta_matrix[diff < -N//2] += N

    # Compute Asymmetric Matrix (Standard STDP)
    term_plus = calculate_stdp_term(delta_matrix, a, b, A_plus, tau_plus, 'forward')
    term_minus = calculate_stdp_term(delta_matrix, a, b, A_minus, tau_minus, 'backward')
    W_stdp_mat = factor * (term_plus - term_minus)

    # Compute Symmetric Matrix (Transposed STDP)
    # Transpose implies delta -> -delta
    term_plus_T = calculate_stdp_term(-delta_matrix, a, b, A_plus, tau_plus, 'forward')
    term_minus_T = calculate_stdp_term(-delta_matrix, a, b, A_minus, tau_minus, 'backward')
    W_trans_mat = factor * (term_plus_T - term_minus_T)

    # Total Matrix
    W_total = W_stdp_mat + (epsilon * W_trans_mat)
    W_norm_factor = np.linalg.norm(W_total, ord=2)
    print(W_norm_factor)
    W_total = W_total * 1.2 / W_norm_factor

    # --- Part B: Eigenvalue Analysis ---

    # 1. Numerical Eigenvalues
    eig_vals_num = np.linalg.eigvals(W_total)

    # 2. Analytic Eigenvalues
    # Define frequency range k = 2*pi*n / N
    # k = (2 * np.pi * np.arange(N) / N)
    omega = 2 * np.pi * np.fft.fftfreq(N)

    # plt.plot(np.fft.fftfreq(N))
    # plt.show()

    eigvals_analytic = (np.pi/b) * (a**2) * np.exp(-(omega**2)/(2*b)) + 1j*0
    eigvals_analytic *= (1+epsilon) * (A_plus*tau_plus/(1 + (omega**2) * (tau_plus**2)) - A_minus*tau_minus/(1 + (omega**2) * (tau_minus**2))) - 1j*omega*(1-epsilon) * (A_plus*(tau_plus**2)/(1 + (omega**2) * (tau_plus**2)) + A_minus*(tau_minus**2)/(1 + (omega**2) * (tau_minus**2)))
    eigvals_norm_factor = np.max(np.abs(eigvals_analytic))
    eigvals_analytic = eigvals_analytic * 1.4 / eigvals_norm_factor

    eigvals_discrete = np.fft.fft(W_total[0, :])

    # vrange = np.max(np.abs(W_total))
    # plt.imshow(W_total, vmin=-vrange, vmax=vrange, cmap="coolwarm")
    # plt.show()

    diag_offsets = np.arange(-N, N+1, 1)

    weight_profile = { int(i): [] for i in diag_offsets }
    for i in range(N):
        for j in range(N):
            weight_profile[i-j].append(W_total[i, j])

    W_profile = [ np.mean(weight_profile[d]) for d in diag_offsets ]


    # --- 2. Simulation Parameters ---
    dt = 1.0              # Time step
    tau_neuron = 1.0     # Neural time constant
    steps = 200           # Simulation duration
    warmup_steps = 5

    # # Initialize Activity (Random Noise)
    # u = np.random.rand(N) * 0.1
    # # Optional: Seed a bump in the middle
    # # u[45:55] += 1.0

    # Initialize Gaussian
    u = a * np.exp(-b*(np.arange(N)-mu)**2)

    # Storage
    history = np.zeros((steps, N))

    # --- 4. Run Dynamics (Euler Integration) ---
    print("Running simulation...")
    for t in mo.status.progress_bar(range(steps)):
        history[t] = u.copy()

        # Compute Recurrent Input
        I_rec = np.dot(W_total/np.linalg.norm(W_total, ord=2), u)

        ## continuous
        # Rate Equation: du/dt = (-u + f(I))/tau
        # ReLU Activation: np.maximum(0, x)
        # du = (-u + np.maximum(0, I_rec)) / tau_neuron
        du = (-u + np.tanh(I_rec)) / tau_neuron
        # Update
        u = u + du * dt

        # discrete 
        u = I_rec



    # --- Plotting ---
    fig = plt.figure(figsize=(12, 3))

    ax = fig.add_subplot(1, 4, 1)

    delay = np.arange(-60, 60, 1)
    ax.plot(delay, -compute_modified_stdp(delay, epsilon=epsilon, 
                              tau_pos=tau_plus, tau_neg=tau_minus, 
                              A_pos=A_plus, A_neg=A_minus), 'k-', lw=2)

    ax.axhline(0, color='gray', linestyle=':', lw=1)
    ax.axvline(0, color='gray', linestyle=':', lw=1)
    ax.set_title(f"STDP Kernel ($\epsilon={epsilon}$)")
    ax.set_xlabel("Delay $\\Delta t = t_{post} - t_{pre}$ (ms)")
    ax.set_ylabel("$\\Delta W_{ij}$")
    ax.grid(True, alpha=0.3)
    # ax.set_xlim((-25, 25))

    # Plot 1: Spatial Profile
    ax1 = fig.add_subplot(1, 4, 2)

    ax1.plot(diag_offsets, W_profile, 'k-', lw=2, label='Weight Profile')
    ax1.axhline(0, color='gray', linestyle=':', lw=1)
    ax1.axvline(0, color='gray', linestyle=':', lw=1)
    ax1.set_title(f"Spatial Weight Profile ($\epsilon={epsilon}$)")
    ax1.set_xlabel("Diagonal Offset ($i-j$)")
    ax1.set_ylabel("$W_{ij}$")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim((-25, 25))

    # vrange = np.max(np.abs(W_total))
    # plt.imshow(W_total, cmap="coolwarm", vmin=-vrange, vmax=vrange)

    # Plot 2: Eigenvalue Spectrum (Complex Plane)
    ax2 = fig.add_subplot(1, 4, 3)

    # Plot Analytic Curve
    # Separate strictly real and imaginary parts for the line plot
    # ax2.plot(np.real(eig_vals_ana), np.imag(eig_vals_ana), 'r-', lw=2, alpha=0.6, label='Analytic Theory')

    # Plot Numerical Dots
    circ = plt.Circle((0, 0), 1, fill=False, color="k")
    ax2.add_patch(circ)
    ax2.scatter(np.real(eig_vals_num), np.imag(eig_vals_num), c='blue', s=20, alpha=0.8, edgecolors='k', lw=0.5, label='Numerical')
    # ax2.plot(eigvals_analytic.real, eigvals_analytic.imag, c="r", 
    #          marker="x", label="continous FT")
    # ax2.plot(eigvals_discrete.real, eigvals_discrete.imag, c="g", alpha=0.5,
    #          marker="x", label="discrete FT")

    # Formatting
    ax2.axhline(0, color='k', linestyle='-', lw=0.5)
    ax2.axvline(0, color='k', linestyle='-', lw=0.5)
    ax2.set_title(f"Eigenvalue Spectrum ($\epsilon={epsilon}$)")
    ax2.set_xlabel("Re$(\\lambda)$")
    ax2.set_ylabel("Im$(\\lambda)$")
    ax2.legend()
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim((-1.5, 1.5))
    ax2.set_ylim((-1.5, 1.5))

    # Plot 3: Dynamics
    ax2 = fig.add_subplot(1, 4, 4)

    plt.imshow(history[warmup_steps:].T, aspect='auto', cmap='viridis', origin='upper')
    plt.colorbar(label='Firing Rate')
    plt.title(f'Network Dynamics ($\epsilon={epsilon}$)')
    plt.ylabel('Neuron Index')
    plt.xlabel('Time Step')

    plt.tight_layout()
    # fig
    plt.savefig(f"STDP_eps={epsilon}.png")
    plt.savefig(f"STDP_eps={epsilon}.svg")
    fig
    return W_total, history


@app.cell(hide_code=True)
def _(W_total, np, plt):
    plt.clf()
    plt.hist(np.abs(W_total).flatten())
    plt.show()
    return


@app.cell
def _(W_total, np, plt):
    plt.clf()
    plt.hist(np.log(np.abs(W_total)).flatten())
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fourier Mode Analysis
    """)
    return


@app.cell(hide_code=True)
def _(history, np, plt):
    def analyze_fourier_modes(u_history, dt=1.0):
        """
        Computes the evolution of spatial Fourier modes over time.

        Parameters:
        u_history (ndarray): Shape (steps, N). Neural activity over time.
        dt (float): Simulation time step (for plotting axis).
        """
        steps, N = u_history.shape

        # 1. Compute Spatial FFT at each time step (along axis 1)
        # Result is complex: contains Magnitude (Amplitude) and Phase (Position)
        u_fft = np.fft.fft(u_history, axis=1)

        # 2. Compute Power (Magnitude Squared) normalized by N
        # We look at the first few modes: k=0 (DC), k=1 (Fundamental), k=2
        power_spectrum = (np.abs(u_fft) / N)**2

        # 3. Compute Phase of the Fundamental Mode (k=1)
        # This tracks the center of mass of the bump on the ring [0, 2pi]
        phase_k1 = np.angle(u_fft[:, 1])

        # "Unwrap" the phase to track continuous movement past 360 degrees
        # This turns sawtooth jumps into a continuous line
        phase_unwrapped = np.unwrap(phase_k1)

        # --- Visualization ---
        time_axis = np.arange(steps) * dt

        plt.figure(figsize=(14, 5))

        # Plot A: Mode Strengths (Stability)
        plt.subplot(1, 2, 1)
        modes = [0, 1, 2, 3]
        for k in modes:
            plt.plot(time_axis, power_spectrum[:, k], linewidth=2, label=f'Mode k={k}')

        plt.title("Fourier Mode Power (Bump Stability)")
        plt.xlabel("Time (ms)")
        plt.ylabel(r"Power $|\tilde{u}_k|^2$")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot B: Phase Drift (Velocity)
        plt.subplot(1, 2, 2)
        plt.plot(time_axis, phase_unwrapped, 'r-', linewidth=2)

        # Calculate Velocity (Slope of the phase)
        # Linear regression on the second half of the simulation to ignore transients
        if steps > 10:
            fit_coeffs = np.polyfit(time_axis[steps//2:], phase_unwrapped[steps//2:], 1)
            slope = fit_coeffs[0] # rad/ms

            # Convert rad/ms to neurons/sec
            # Velocity = (Slope / 2pi) * N * 1000
            vel_neurons = (slope / (2 * np.pi)) * N * (1000 if dt < 1 else 1)

            plt.plot(time_axis, np.polyval(fit_coeffs, time_axis), 'k--', alpha=0.5, label='Linear Fit')
            plt.text(0.05, 0.9, f"Drift Velocity: {vel_neurons:.2f} neurons/sec", 
                     transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

        plt.title("Phase of Mode k=1 (Bump Position)")
        plt.xlabel("Time (ms)")
        plt.ylabel("Unwrapped Phase (radians)")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Example Usage:
    # assuming 'history' is your (steps, 100) array
    analyze_fourier_modes(history, dt=1.0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # For Paper
    """)
    return


if __name__ == "__main__":
    app.run()
