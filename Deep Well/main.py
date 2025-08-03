"""
main.py

Main script to run the Deep Well problem.
This script imports the necessary functions from the solver and config modules,
generates the energy table, and prints it in a formatted manner.

Author: Ricard Santiago Raigada García
Date: 2025-08-03
"""
from deepwell.solver import generate_energy_table, print_table
from deepwell.config import N_values

if __name__ == "__main__":
    results, exact = generate_energy_table(N_values)
    print_table(results, exact)