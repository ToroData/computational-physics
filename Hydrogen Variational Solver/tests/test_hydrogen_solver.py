"""
test_hydrogen_solver.py
=======================

Unit test module for the hydrogen_solver.py implementation.

This test suite validates the correctness of the variational solution for the hydrogen atom.

Tests included:
---------------
- test_ground_state_energy:
    Verifies that the computed ground state energy matches the known reference 
    value (-0.499278 hartree) within a tolerance of 1e-6.

Usage:
------
Run from the project root using:

    pytest

This will automatically discover and execute all test_*.py files under the /tests directory.

Dependencies:
-------------
- pytest
- numpy
- scipy
- hydrogen_solver.py and config.py

Author:
-------
Ricard Santiago Raigada García
"""
from hydrogen_variational_solver.solver import (
    compute_overlap_matrix,
    compute_kinetic_matrix,
    compute_coulomb_matrix,
    build_hamiltonian,
    solve_generalized_eigenproblem
)
from hydrogen_variational_solver.config import ALPHAS
import numpy as np

def test_ground_state_energy():
    S = compute_overlap_matrix(ALPHAS)
    T = compute_kinetic_matrix(ALPHAS)
    A = compute_coulomb_matrix(ALPHAS)
    H = build_hamiltonian(T, A)
    E_vals, C = solve_generalized_eigenproblem(H, S)
    assert np.isclose(E_vals[0], -0.499278, atol=1e-6)
