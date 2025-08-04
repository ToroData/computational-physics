"""
hydrogen_solver.py
==================

This module implements the variational solution to the Schrödinger equation 
for the hydrogen atom using a Gaussian basis set of the form:

    χₚ(r) = exp(-αₚ r²)

The method follows the variational procedure outlined in the book 
"Computational Physics" by Jos Thijssen, Section 3.2.2.

The system solved is the hydrogen atom in atomic units:

    [-1/2 ∇² - 1/r] ψ(r) = E ψ(r)

For a given set of Gaussian exponents αₚ, the module computes the:
    - Overlap matrix S
    - Kinetic energy matrix T
    - Coulomb potential matrix A

Then it builds the Hamiltonian H = T + A and solves the generalized 
eigenvalue problem:

    H C = E S C

Functions:
----------
- compute_overlap_matrix(alpha): 
    Computes the overlap matrix Sₚq = ⟨χₚ|χ_q⟩

- compute_kinetic_matrix(alpha): 
    Computes the kinetic matrix Tₚq = ⟨χₚ| -½∇² |χ_q⟩

- compute_coulomb_matrix(alpha): 
    Computes the Coulomb matrix Aₚq = ⟨χₚ| -1/r |χ_q⟩

- build_hamiltonian(T, A): 
    Combines T and A into the total Hamiltonian matrix H

- solve_generalized_eigenproblem(H, S): 
    Solves the generalized eigenvalue problem H C = E S C using scipy.linalg.eigh

Typical Usage:
--------------
>>> from hydrogen_solver import *
>>> S = compute_overlap_matrix(alpha_list)
>>> T = compute_kinetic_matrix(alpha_list)
>>> A = compute_coulomb_matrix(alpha_list)
>>> H = build_hamiltonian(T, A)
>>> E, C = solve_generalized_eigenproblem(H, S)

Returns:
--------
- E: Eigenvalues (energy levels)
- C: Eigenvectors (coefficients for basis expansion of ψ)

Author:
-------
Ricard Santiago Raigada García
"""

import numpy as np
from scipy.linalg import eigh

def compute_overlap_matrix(alpha: np.ndarray) -> np.ndarray:
    """
    Compute the overlap matrix S for the given Gaussian exponents.
    S[p, q] = (π / (alpha_p + alpha_q))**1.5

    Parameters:
    -----------
    alpha : np.ndarray
        1D array of Gaussian exponents.

    Returns:
    --------
    S : np.ndarray
        Overlap matrix.
    """
    N = len(alpha)
    S = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            S[p, q] = (np.pi / (alpha[p]+ alpha[q]))**1.5
    return S


def compute_kinetic_matrix(alpha: np.ndarray) -> np.ndarray:
    """Compute the kinetic energy matrix T for the given Gaussian exponents.
    T[p, q] = 3 * alpha_p * alpha_q * π^(1.5) / (alpha_p + alpha_q)^(2.5)

    Parameters:
    -----------
    alpha : np.ndarray
        1D array of Gaussian exponents.
    Returns:
    --------
    T : np.ndarray
        Kinetic energy matrix.
    """
    N = len(alpha)
    T = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            numerator = 3 * alpha[p] * alpha[q] * np.pi**(1.5)
            denominator = (alpha[p] + alpha[q]) ** 2.5
            T[p, q] = numerator / denominator
    return T


def compute_coulomb_matrix(alpha: np.ndarray) -> np.ndarray:
    """
    Compute the Coulomb matrix A for the given Gaussian exponents.
    A[p, q] = -2 * np.pi / (alpha[p] + alpha[q])

    Parameters:
    -----------
    alpha : np.ndarray
        1D array of Gaussian exponents.

    Returns:
    --------
    A : np.ndarray
        Coulomb matrix.
    """
    N = len(alpha)
    A = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            A[p, q] = - 2 * np.pi / (alpha[p] + alpha[q])
    return A


def build_hamiltonian(T: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Build the Hamiltonian matrix H from the kinetic and Coulomb matrices.
    H = T + A

    Parameters:
    -----------
    T : np.ndarray
        Kinetic energy matrix.
    A : np.ndarray
        Coulomb matrix.

    Returns:
    --------
    H : np.ndarray
        Hamiltonian matrix.
    """
    return T + A


def solve_generalized_eigenproblem(
    H: np.ndarray,
    S: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the generalized eigenvalue problem Hc = ES.
    This uses the scipy.linalg.eigh function to find the eigenvalues and eigenvectors.

    Parameters:
    -----------
    H : np.ndarray
        Hamiltonian matrix.
    S : np.ndarray
        Overlap matrix.
    
    Returns:
    --------
    E : np.ndarray
        Eigenvalues of the generalized eigenproblem.
    """
    E, C = eigh(H, S)
    return E, C
