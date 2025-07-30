"""
# main.py
This script serves as the main entry point for the quantum scattering simulation.
It computes the total cross section for a range of energy values and visualizes the results.
It also computes and visualizes the wavefunction for the maximum energy value.

Author: Ricard Santiago Raigada García
Date: 30/07/2025
"""
import numpy as np
from .config import rmin, rmax, dr, IMG_SHOW
from .cross_section import compute_total_cross_section
from .numerov import build_grid, numerov_integrate
from .visualization import plot_cross_section, plot_wavefunction

def main() -> None:
    """
    Main function to execute the quantum scattering simulation.
    It computes the total cross section for a range of energy values and visualizes the results.
    It also computes and visualizes the wavefunction for the maximum energy value.
    """
    de = 0.01
    E_values = np.arange(0.1, 3.5, de)
    sigma = compute_total_cross_section(E_values)
    plot_cross_section(E_values, sigma, show=IMG_SHOW)

    # Wavefunction for maximum energy
    E_max = E_values[-1]
    r = build_grid(dr, rmin, rmax, E_max)
    u, r, u1, u2, r1, r2 = numerov_integrate(r, dr, E_max, l=0)
    plot_wavefunction(r, u, E_max, show=IMG_SHOW)

if __name__ == "__main__":
    main()
