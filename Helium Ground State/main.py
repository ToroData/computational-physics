"""
Main Execution Script for Helium Ground State Calculation
=========================================================

This script imports the `helium_ground_state` function from the solver module
and executes it to compute the ground state energy of the helium atom using
a minimal Gaussian basis and the Hartree–Fock method.

Execution:
----------
Run this script as the main module to perform the self-consistent field
calculation and print the resulting ground state energy in Hartree units.

Functions:
----------
- `helium_ground_state()`:
      Performs the SCF loop and returns the ground state energy and coefficients.

Example:
--------
$ python main.py
Ground state energy: -2.85516038 Hartree

Author: Ricard Santiago Raigada García
Date: August 2025
"""
from helium_ground_state.solver import helium_ground_state

if __name__ == "__main__":
    E_G, _ = helium_ground_state()
    print(f"Ground state energy: {E_G:.8f} Hartree")
