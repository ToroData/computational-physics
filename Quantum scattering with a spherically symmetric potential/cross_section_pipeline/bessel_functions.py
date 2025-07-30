"""
# bessel_functions.py
This module provides functions to compute regular and irregular Bessel functions
for given angular momentum quantum number `l` and argument `x`.

Author: Ricard Santiago Raigada García
Date: 30/07/2025
"""
import numpy as np

def regular_bessel_upwards(l: int, x: float) -> float:
    """
    Compute the regular Bessel function of the first kind for given `l` and `x`.
    Uses an upward recursion relation for efficiency.
    
    Parameters:
        l (int): Angular momentum quantum number.
        x (float): Argument for the Bessel function.
    Returns:
        float: Value of the regular Bessel function of the first kind.
    """
    j_prev = np.sin(x) / x
    if l == 0:
        return j_prev
    j_curr = (np.sin(x) / x**2) - (np.cos(x) / x)
    if l == 1:
        return j_curr
    for n in range(2, l + 1):
        j_next = ((2 * n - 1) / x) * j_curr - j_prev
        j_prev, j_curr = j_curr, j_next
    return j_curr

def irregular_bessel_upwards(l: int, x: float) -> float:
    """
    Compute the irregular Bessel function of the second kind for given `l` and `x`.
    Uses an upward recursion relation for efficiency.
    Parameters:
        l (int): Angular momentum quantum number.
        x (float): Argument for the Bessel function.
    Returns:
        float: Value of the irregular Bessel function of the second kind.
    """
    n_prev = -np.cos(x) / x
    if l == 0:
        return n_prev
    n_curr = (-np.cos(x) / x**2) - (np.sin(x) / x)
    if l == 1:
        return n_curr
    for n in range(2, l + 1):
        n_next = ((2 * n - 1) / x) * n_curr - n_prev
        n_prev, n_curr = n_curr, n_next
    return n_curr
