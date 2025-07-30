"""
# phase_shift.py
This module computes the phase shift for quantum scattering
with a spherically symmetric potential using the Bessel functions.
It provides a function to compute the phase shift based on the radial wavefunction.

Author: Ricard Santiago Raigada García
Date: 30/07/2025
"""
import numpy as np
from .bessel_functions import regular_bessel_upwards, irregular_bessel_upwards

def compute_phase_shift(l: int, E: float, r1: float, r2: float, u1: float, u2: float) -> float:
    """    Compute the phase shift for quantum scattering.
    Parameters:
        l (int): Orbital angular momentum quantum number.
        E (float): Energy value.
        r1 (float): First radial point.
        r2 (float): Second radial point.
        u1 (float): Wavefunction at r1.
        u2 (float): Wavefunction at r2.
    """
    k = np.sqrt(E)
    kr1 = k * r1
    kr2 = k * r2
    K = (r1 * u2) / (r2 * u1)
    numerator = K * regular_bessel_upwards(l, kr1) - regular_bessel_upwards(l, kr2)
    denominator = K * irregular_bessel_upwards(l, kr1) - irregular_bessel_upwards(l, kr2)
    return np.arctan(numerator / denominator)
