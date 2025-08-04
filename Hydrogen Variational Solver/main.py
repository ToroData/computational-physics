"""
main.py
=======

Entry point for executing the variational calculation of the hydrogen atom 
using a Gaussian basis set.

This script performs the following steps:

1. Loads the Gaussian exponents alpha from the configuration module.
2. Computes the required matrices:
   - Overlap matrix (S)
   - Kinetic energy matrix (T)
   - Coulomb potential matrix (A)
3. Constructs the full Hamiltonian matrix: H = T + A.
4. Solves the generalized eigenvalue problem: H C = E S C.
5. Prints the ground state energy.

Intended to be run from the command line:

    python main.py

Dependencies:
-------------
- numpy
- scipy
- hydrogen_solver.py (internal module)

Author:
-------
Ricard Santiago Raigada García
"""

from hydrogen_variational_solver import solver as hs
from hydrogen_variational_solver.config import ALPHAS

def main() -> None:
    S = hs.compute_overlap_matrix(ALPHAS)
    T = hs.compute_kinetic_matrix(ALPHAS)
    A = hs.compute_coulomb_matrix(ALPHAS)
    H = hs.build_hamiltonian(T, A)
    E_vals, C = hs.solve_generalized_eigenproblem(H, S)
    
    print(f"Ground state energy: {E_vals[0]}")

if __name__ == "__main__":
    main()