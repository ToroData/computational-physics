"""
# cross_section.py
This module computes the total cross section for quantum scattering
with a spherically symmetric potential using the Numerov method for solving
the radial Schrödinger equation and phase shifts for different angular momentum quantum numbers.

Author: Ricard Santiago Raigada García
Date: 30/07/2025
"""
import numpy as np
from .config import rmin, rmax, dr, rho
from .numerov import build_grid, numerov_integrate
from .phase_shift import compute_phase_shift

def compute_total_cross_section(E_values: np.ndarray, l_max: int = 10) -> np.ndarray:
    """
    Compute the total cross section for quantum scattering.
    Parameters:
        E_values (np.ndarray): Array of energy values.
        l_max (int): Maximum angular momentum quantum number to consider.
    Returns:
        np.ndarray: Total cross section for each energy value.
    """
    sigma = np.zeros_like(E_values)
    for e, E in enumerate(E_values):
        for l in range(l_max):
            r = build_grid(dr, rmin, rmax, E)
            u, r, u1, u2, r1, r2 = numerov_integrate(r, dr, E, l)
            delta = compute_phase_shift(l, E, r1, r2, u1, u2)
            sigma[e] += (2 * l + 1) * np.sin(delta)**2
        sigma[e] *= 4 * np.pi / (E * rho**2)
    return sigma
