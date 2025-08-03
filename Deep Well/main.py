"""
main.py

Main script to run the Deep Well problem.
This script imports the necessary functions from the solver and config modules,
generates the energy table, prints it in a formatted manner, and visualizes the results.

Author: Ricard Santiago Raigada García
Date: 2025-08-03
"""
from deepwell.solver import generate_energy_table, print_table
from deepwell.visualization import plot_all_wavefunctions_grid
from deepwell.config import N_values

if __name__ == "__main__":
    results, exact, eigenvectors = generate_energy_table(N_values)
    print_table(results, exact)
    plot_all_wavefunctions_grid(eigenvectors, max_states=5)
