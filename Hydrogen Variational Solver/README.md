# Hydrogen Variational Solver

This project implements a variational method for solving the Schrödinger equation of the hydrogen atom using a Gaussian basis set.

## Problem Overview

We aim to approximate the ground state energy of the hydrogen atom using a set of Gaussian functions:

\[
\chi_p(r) = e^{-\alpha_p r^2}
\]

The Schrödinger equation in atomic units is:

\[
\left( -\frac{1}{2} \nabla^2 - \frac{1}{r} \right) \psi(r) = E \psi(r)
\]

Using a finite basis, the problem becomes solving a generalized eigenvalue problem:

\[
H \mathbf{c} = E S \mathbf{c}
\]

Where:
- \( S \): Overlap matrix
- \( T \): Kinetic energy matrix
- \( A \): Coulomb interaction matrix
- \( H = T + A \): Total Hamiltonian

## Method

The basis is constructed using 4 predefined Gaussian exponents \( \alpha_p \), stored in `config.py`. The solver performs the following steps:

1. Compute matrices `S`, `T`, `A`
2. Build the total Hamiltonian `H = T + A`
3. Solve the generalized eigenvalue problem `Hc = ESc`
4. Extract the ground state energy and coefficients

## Project Structure

```

Hydrogen Variational Solver/
├── hydrogen\_variational\_solver/
│   ├── __init__.py
│   ├── hydrogen\_solver.py         # Solver logic and matrix construction
│   ├── config.py                   # Contains Gaussian exponent values
├── tests/
│   ├── test\_hydrogen\_solver.py   # Unit test using pytest
├── main.py                         # Script to run the calculation
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation

```

## Example Output

```bash
$ python main.py
Ground state energy: -0.499278
```

Expected theoretical value: `-0.499278 hartree`

## Running Tests

From the project root:

```bash
pytest
```

This will run all unit tests inside the `/tests` directory.
