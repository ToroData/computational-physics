"""
solver.py

Module for solving the energy spectrum of a quantum particle 
in an infinitely deep potential well using variational methods.

Implements the construction of overlap and Hamiltonian matrices, 
solves the generalized eigenvalue problem, and compares the results 
with the analytical solution.

Author: Ricard Santiago Raigada García
Date: 2025-08-03
"""
import numpy as np
from scipy.linalg import eigh

def compute_overlap_matrix(N: int) -> np.ndarray:
    S: np.ndarray = np.zeros((N, N))
    for n in range(N):
        for m in range(N):
            if (n + m) % 2 == 0:
                S[n, m] = 2 / (n + m + 5) - 4 / (n + m + 3) + 2 / (n + m + 1)
    return S

def compute_hamiltonian_matrix(N: int) -> np.ndarray:
    H: np.ndarray = np.zeros((N, N))
    for n in range(N):
        for m in range(N):
            if (n + m) % 2 == 0:
                denom = (n + m + 3) * (n + m + 1) * (n + m - 1)
                H[n, m] = -8 * (1 - m - n - 2 * m * n) / denom
    return H

def solve_generalized_eigenproblem(H: np.ndarray, S: np.ndarray) -> np.ndarray:
    E, C = eigh(H, S)
    return E, C

def analytical_energies(num_levels: int) -> list[float]:
    return [(n**2 * np.pi**2) / 4 for n in range(1, num_levels+1)]

def generate_energy_table(
    N_values: list[int],
    num_levels: int = 5
    ) -> tuple[dict[int, np.ndarray], list[float]]:
    results = {}
    eigenvectors = {}
    for N in N_values:
        S = compute_overlap_matrix(N)
        H = compute_hamiltonian_matrix(N)
        E, C = solve_generalized_eigenproblem(H, S)
        results[N] = E[:num_levels]
        eigenvectors[N] = C[:, :num_levels]
    exact = analytical_energies(num_levels)
    return results, exact, eigenvectors

def print_table(results: dict[int, np.ndarray], exact: list[float]) -> None:
    Ns: list[int] = sorted(results.keys())
    print("Table 3.1. Energy levels of the infinitely deep potential well.")
    header = "n  | " + " | ".join([f"N = {N:<2}" for N in Ns]) + " | Exact"
    print(header)
    print("-" * len(header))
    for i in range(len(exact)):
        row = f"{i+1:<2} | " + " | ".join(f"{results[N][i]:<7.4f}" for N in Ns) + f" | {exact[i]:<7.4f}"
        print(row)