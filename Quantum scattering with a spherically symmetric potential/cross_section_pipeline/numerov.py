"""
# numerov.py
This module implements the Numerov method for solving the radial Schrödinger equation
for quantum scattering with a spherically symmetric potential.
It provides functions to build the grid, initialize conditions, and perform the integration.

Author: Ricard Santiago Raigada García
Date: 30/07/2025
"""
import numpy as np
from .potential import effective_potential
from .config import rmin, rmax

def build_grid(h: float, rmin: float, rmax: float, E: float) -> np.ndarray:
    """
    Build a grid of radial points for the Numerov integration.
    Parameters:
        h (float): Step size in the radial direction.
        rmin (float): Minimum radial coordinate.
        rmax (float): Maximum radial coordinate.
        E (float): Energy value.
    Returns:
        np.ndarray: Array of radial points.
    """
    half_wave = np.pi / np.sqrt(E)
    return np.arange(rmin, rmax + half_wave, h)

def initialize_conditions(h: float, l: int) -> tuple[float, float]:
    """
    Initialize the boundary conditions for the Numerov integration.
    Parameters:
        h (float): Step size in the radial direction.
        l (int): Orbital angular momentum quantum number.
    Returns:
        tuple[float, float]: Initial conditions (u0, u1).
    """
    u0 = 0.0
    u1 = h**(l + 1)
    return u0, u1

def numerov_integrate(r: np.ndarray, h: float, E: float, l: int) -> tuple[np.ndarray, float, float, float, float]:
    """
    Perform the Numerov integration to solve the radial Schrödinger equation.
    Parameters:
        r (np.ndarray): Array of radial points.
        h (float): Step size in the radial direction.
        E (float): Energy value.
        l (int): Orbital angular momentum quantum number.
    Returns:
        tuple[np.ndarray, float, float, float, float]: (u, r, u1, u2, r1, r2)
    """
    u = np.zeros_like(r)
    u0, u1 = initialize_conditions(h, l)
    u[0], u[1] = u0, u1
    norm = u[1]**2 * h

    for n in range(1, len(r) - 1):
        f = effective_potential(r[n], l, E)
        un = u[n] / (1 - h**2 * f / 12)
        u[n + 1] = 2 * u[n] - u[n - 1] + h**2 * f * un
        norm += u[n + 1]**2 * h

    u /= np.sqrt(norm)
    r1_index = int((rmax - rmin) / h) + 1
    r1, r2 = r[r1_index], r[-1]
    u1, u2 = u[r1_index], u[-1]
    return u, r, u1, u2, r1, r2
