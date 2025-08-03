"""
Visualization module for the Deep Well problem

This module provides graphical comparison between variational wavefunctions
and the exact analytical solutions.

Author: Ricard Santiago Raigada García
Date: 2025-08-03
"""

import numpy as np
import matplotlib.pyplot as plt

def basis_function(n, x):
    """Polynomial basis function: ψₙ(x) = xⁿ (x - 1)(x + 1)"""
    return x**n * (x - 1) * (x + 1)

def analytical_solution(n, x):
    """Analytical eigenfunctions for the infinite well"""
    k_n = n * np.pi / 2
    if n % 2 == 0:  # even → sin
        return np.sin(k_n * x)
    else:           # odd → cos
        return np.cos(k_n * x)

def plot_all_wavefunctions_grid(
    eigenvectors: dict[int, np.ndarray],
    max_states: int = 5
) -> None:
    """
    Display a grid of comparisons between variational and analytical wavefunctions.

    Parameters:
    - eigenvectors: dict with eigenvectors for each N
    - max_states: number of lowest energy states to plot
    """
    N_values = sorted(eigenvectors.keys())
    x = np.linspace(-1, 1, 500)
    num_rows = max_states
    num_cols = len(N_values)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 2.5 * num_rows), sharex=True, sharey=True)

    for col, N in enumerate(N_values):
        for row in range(max_states):
            coeffs = eigenvectors[N][:, row]
            psi_variational = sum(coeffs[n] * basis_function(n, x) for n in range(len(coeffs)))
            psi_variational /= np.sqrt(np.trapz(np.abs(psi_variational)**2, x))

            n_analytical = row + 1
            psi_exact = analytical_solution(n_analytical, x)
            psi_exact /= np.max(np.abs(psi_exact))

            # Align phase (sign)
            if np.dot(psi_variational, psi_exact) < 0:
                psi_variational *= -1

            # Plot
            ax = axes[row, col] if num_rows > 1 else axes[col]
            ax.plot(x, psi_variational, label="Variational")
            ax.plot(x, psi_exact, '--', label="Analytical", alpha=0.8)
            ax.plot(x, psi_variational - psi_exact, ':', color='gray', alpha=0.6, label="Δψ (error)")

            ecm = np.sqrt(np.trapz((psi_variational - psi_exact)**2, x))
            if row == 0:
                ax.set_title(f"N = {N}")
            if col == 0:
                ax.set_ylabel(f"ψ(x), n = {n_analytical}")
            if row == num_rows - 1:
                ax.set_xlabel("x")
            ax.grid(True)
            
            ax.text(
                0.95, 0.90,
                f"ECM = {ecm:.2e}",
                transform=ax.transAxes,
                fontsize=9,
                ha='right',
                va='top',
                color='black'
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3)
    plt.suptitle("Comparison of variational vs analytical wavefunctions", fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
