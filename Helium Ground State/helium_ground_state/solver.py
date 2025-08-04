"""
Helium Atom Ground State Solver using Hartree–Fock Approximation
=================================================================

This module implements the self-consistent field (SCF) procedure for computing the 
ground state energy of the helium atom using the Hartree–Fock method in a minimal basis 
of four uncorrelated Gaussian functions.

It reproduces the calculation outlined in Section 4.3.2 of the book:
"Computational Physics" by Jos Thijssen.

The wavefunction is approximated as a linear combination of real s-type Gaussians:
    φ(r) = Σ_p C_p χ_p(r),     with     χ_p(r) = exp(-α_p r²)

Main steps:
-----------
1. Construct the overlap matrix S[p,q]
2. Construct the one-electron Hamiltonian h[p,q] including kinetic and nuclear terms
3. Construct the two-electron Coulomb interaction tensor Q[p,r,q,s]
4. Initialize coefficients C and build the Fock matrix F[p,q]
5. Solve the generalized eigenvalue problem:
       F C = E S C
6. Iterate until convergence and evaluate the total energy using:
       E_G = 2 ⟨φ|h|φ⟩ + ⟨φφ|1/r₁₂|φφ⟩

Modules:
--------
- `compute_overlap_matrix(alpha, N)`:
      Computes the overlap matrix S[p,q] = (π / (α_p + α_q))^1.5

- `compute_kinetic_matrix(alpha, N)`:
      Computes the kinetic energy contribution to the Hamiltonian.

- `compute_nuclear_potential(alpha, N)`:
      Computes the nuclear attraction term assuming nuclear charge Z = 2.

- `compute_h_matrix(T, V)`:
      Builds the full one-electron Hamiltonian h = T + V.

- `compute_Q_tensor(alpha, N)`:
      Builds the two-electron Coulomb tensor using Gaussian integrals.

- `build_F_matrix(h, Q, C, N)`:
      Constructs the Fock matrix for the SCF procedure.

- `normalize_C(C, S)`:
      Normalizes the coefficient vector using the overlap matrix.

- `solve_generalized_eigen(F, S)`:
      Solves the generalized eigenvalue problem F C = E S C.

- `compute_ground_state_energy(h, Q, C, S, N)`:
      Evaluates total energy E_G = 2⟨C|h|C⟩ + ⟨CC|Q|CC⟩.

- `helium_ground_state()`:
      Main routine to perform the self-consistent iteration and return energy and coefficients.

Returns:
--------
- Ground state energy (float) in Hartree units.
- Normalized eigenvector coefficients C (np.ndarray).

Reference:
----------
Check 1: Correct ground state energy should be ≈ -2.85516038 a.u. (atomic units)

Author: Ricard Santiago Raigada García
Date: August 2025
"""

import numpy as np
from scipy.linalg import eigh
from .config import ALPHAS

def compute_overlap_matrix(alpha: np.ndarray, N: int) -> np.ndarray:
    """Compute the overlap matrix S for the given Gaussian exponents.
    S[p, q] = (π / (alpha_p + alpha_q))^1.5
    Parameters:
    -----------
    alpha : np.ndarray
        1D array of Gaussian exponents.
    N : int
        Number of Gaussian functions.
    Returns:
    --------
    S : np.ndarray
        Overlap matrix.
    """
    S = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            S[p, q] = (np.pi / (alpha[p] + alpha[q])) ** 1.5
    return S


def compute_kinetic_matrix(alpha: np.ndarray, N: int) -> np.ndarray:
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
    T = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            numerator = 3 * alpha[p] * alpha[q] * np.pi**(1.5)
            denominator = (alpha[p] + alpha[q]) ** 2.5
            T[p, q] = numerator / denominator
    return T


def compute_nuclear_potential(alpha: np.ndarray, N: int) -> np.ndarray:
    """
    Compute the nuclear potential matrix V for the given Gaussian exponents.
    V[p, q] = - (4 * π) / (alpha[p] + alpha[q])

    Parameters:
    -----------
    alpha : np.ndarray
        1D array of Gaussian exponents.
    N : int
        Number of Gaussian functions.

    Returns:
    --------
    V : np.ndarray
        Nuclear potential matrix.
    """
    V = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            V[p, q] = - (4 * np.pi) /(alpha[p] +alpha[q])
    return V


def compute_h_matrix(T: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Build the full one-electron Hamiltonian h = T + V.
    
    Parameters:
    -----------
    T : np.ndarray
        Kinetic energy matrix.
    V : np.ndarray
        Nuclear potential matrix.

    Returns:
    --------
    h : np.ndarray
        One-electron Hamiltonian matrix.
    """
    return  T + V


def compute_Q_tensor(alpha: np.ndarray, N: int) -> np.ndarray:
    """
    Compute the two-electron Coulomb interaction tensor Q[p, r, q, s]
    for the given Gaussian exponents.

    Parameters:
    -----------
    alpha : np.ndarray
        1D array of Gaussian exponents.
    N : int
        Number of Gaussian functions.

    Returns:
    --------
    Q : np.ndarray
        Two-electron Coulomb interaction tensor.
    """
    Q = np.zeros((N, N, N, N))
    for p in range(N):
        for r in range(N):
            for q in range(N):
                for s in range(N):
                    numerator = 2 * np.pi ** 2.5
                    A = alpha[p] + alpha[q]
                    B = alpha[r] + alpha[s]
                    AB = A * B
                    C = np.sqrt(A + B)
                    Q[p, r, q, s] = numerator / (AB * C)
    return Q


def build_F_matrix(h: np.ndarray, Q: np.ndarray, C: np.ndarray, N: int) -> np.ndarray:
    """
    Build the Fock matrix F[p, q] for the SCF procedure.
    F[p, q] = h[p, q] + Σ_r Σ_s Q[p, r, q, s] C[r] C[s]
    
    Parameters:
    -----------
    h : np.ndarray
        One-electron Hamiltonian matrix.
    Q : np.ndarray
        Two-electron Coulomb interaction tensor.
    C : np.ndarray
        Coefficient vector.
    N : int
        Number of Gaussian functions.

    Returns:
    --------
    F : np.ndarray
        Fock matrix.
    """
    F = h.copy()
    for p in range(N):
        for q in range(N):
            for r in range(N):
                for s in range(N):
                    F[p, q] += Q[p, r, q, s] * C[r] * C[s]
    return F


def normalize_C(C: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    Normalize the coefficient vector C using the overlap matrix S.

    Parameters:
    -----------
    C : np.ndarray
        Coefficient vector.
    S : np.ndarray
        Overlap matrix.

    Returns:
    --------
    C : np.ndarray
        Normalized coefficient vector.
    """
    norm = C.T @ S @ C
    return C / np.sqrt(norm)


def solve_generalized_eigen(F: np.ndarray, S: np.ndarray) -> tuple:
    """
    Solve the generalized eigenvalue problem F C = E S C.

    Parameters:
    -----------
    F : np.ndarray
        Fock matrix.
    S : np.ndarray
        Overlap matrix.

    Returns:
    --------
    E_vals : np.ndarray
        Eigenvalues.
    C_vecs : np.ndarray
        Eigenvectors.
    """
    E_vals, C_vecs = eigh(F, S)
    return E_vals, C_vecs


def compute_ground_state_energy(
    h: np.ndarray,
    Q: np.ndarray,
    C: np.ndarray,
    S: np.ndarray,
    N: int
    ) -> float:
    """
    Compute the ground state energy E_G using the coefficients C and matrices h and Q.
    
    Parameters:
    -----------
    h : np.ndarray
        One-electron Hamiltonian matrix.
    Q : np.ndarray
        Two-electron Coulomb interaction tensor.
    C : np.ndarray
        Coefficient vector.
    S : np.ndarray
        Overlap matrix.
    N : int
        Number of Gaussian functions.

    Returns:
    --------
    E_G : float
        Ground state energy.
    """
    C = normalize_C(C, S)
    E1 = C.T @ h @ C
    E2 = 0.0
    for p in range(N):
        for q in range(N):
            for r in range(N):
                for s in range(N):
                    E2 += Q[p, r, q, s] * C[p] * C[q] * C[r] * C[s]
    return  2 * E1 +  E2


def helium_ground_state(max_iter=50, tol=1e-8) -> tuple:
    """
    Main routine to perform the self-consistent field iteration for the helium atom ground state.

    Parameters:
    -----------
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance.

    Returns:
    --------
    E_G : float
        Ground state energy.
    C : np.ndarray
        Coefficient vector.
    """
    N = len(ALPHAS)
    S = compute_overlap_matrix(ALPHAS, N)
    T = compute_kinetic_matrix(ALPHAS, N)
    V = compute_nuclear_potential(ALPHAS, N)
    h = compute_h_matrix(T, V)
    Q = compute_Q_tensor(ALPHAS, N)

    C = np.ones(N)
    C = normalize_C(C, S)

    for _ in range(max_iter):
        F = build_F_matrix(h, Q, C, N)
        E_vals, C_vecs = solve_generalized_eigen(F, S)
        C_new = C_vecs[:, 0]
        C_new = normalize_C(C_new, S)

        if np.linalg.norm(C_new - C) < tol:
            break
        C = C_new

    E_G = compute_ground_state_energy(h, Q, C, S, N)
    return E_G, C
