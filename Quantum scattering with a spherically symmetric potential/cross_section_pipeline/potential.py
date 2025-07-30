"""
# potential.py
This module defines the potential functions used in the quantum scattering simulation.
It includes the Lennard-Jones potential and the effective potential for the radial Schrödinger equation.

Author: Ricard Santiago Raigada García
Date: 30/07/2025
"""
import numpy as np
from .config import rho, eps

def lennard_jones_potential(r: np.ndarray) -> np.ndarray:
    """
    Compute the Lennard-Jones potential for a given radial distance.
    Parameters:
        r (np.ndarray): Radial distance array.
    Returns:
        np.ndarray: Lennard-Jones potential array.
    """
    pr = rho / r
    pr6 = pr**6
    pr12 = pr6**2
    return eps * (pr12 - 2 * pr6)

def effective_potential(r: np.ndarray, l: int, E: float) -> np.ndarray:
    """
    Compute the effective potential for the radial Schrödinger equation.
    Parameters:
        r (np.ndarray): Radial distance array.
        l (int): Orbital angular momentum quantum number.
        E (float): Energy value.
    Returns:
        np.ndarray: Effective potential array.
    """
    with np.errstate(divide='ignore'):
        return (l * (l + 1)) / r**2 - E + lennard_jones_potential(r)
