"""
# visualization.py
This module provides functions to visualize the results of the quantum scattering simulation.
It includes functions to plot the total cross section and the wavefunction for a given energy value.

Author: Ricard Santiago Raigada García
Date: 30/07/2025
"""
import matplotlib.pyplot as plt
import numpy as np
from .potential import lennard_jones_potential as V
from .config import IMG_DIR

plt.style.use('ggplot')

def plot_cross_section(E_values: np.ndarray, sigma: np.ndarray, show=False) -> None:
    """
    Plot the total cross section as a function of energy.
    
    Parameters:
        E_values (np.ndarray): Array of energy values.
        sigma (np.ndarray): Array of total cross section values.
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(E_values, sigma, linewidth=2)
    ax.set_xlabel("Energy (meV)", fontsize=12)
    ax.set_ylabel(r"Total Cross Section ($\rho^2$)", fontsize=12)
    ax.set_title(
        "Total Cross Section vs Energy for Lennard-Jones Potential",
        fontsize=16,
        loc='left',
        fontweight='bold'
        )
    
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    fig.savefig(f"{IMG_DIR}/total_cross_section.png", dpi=300)
    if show:
        plt.show()

def plot_wavefunction(r: np.ndarray, u: np.ndarray, E: float, show=False) -> None:
    """
    Plot the wave function and potential as a function of radial distance.
    Parameters:
        r (np.ndarray): Radial distance array.
        u (np.ndarray): Wave function array.
        E (float): Energy value.
    Returns:
        None
    """
    V_vals = V(r[1:])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r, u, label="u(r)", linewidth=2)
    ax.plot(r[1:], V_vals, '--', label="V(r)", linewidth=2)

    ax.set_xlim([0, 10])
    ax.set_ylim([min(V_vals), max(u) + 1])
    ax.set_xlabel("r", fontsize=12)
    ax.set_ylabel("W(r)", fontsize=12)
    ax.set_title(
        f"Wave Function for Energy = {E:.3f} meV",
        fontsize=16,
        loc='left',
        fontweight='bold'
        )

    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    fig.savefig(f"{IMG_DIR}/wave_function_E_{E:.3f}.png", dpi=300)
    if show:
        plt.show()
