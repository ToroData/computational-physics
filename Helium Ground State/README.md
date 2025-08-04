# Helium Ground State

This project implements a self-consistent field (SCF) method to compute the ground state energy of the helium atom using a minimal Gaussian basis set and the Hartree–Fock approximation. The basis consists of four fixed, real s-type Gaussian functions.

## Overview

The total energy of the helium atom is computed using the variational principle with a wavefunction of the form:

\[
\phi(r) = \sum_{p=1}^{4} C_p \chi_p(r), \quad \chi_p(r) = e^{-\alpha_p r^2}
\]

where \(\chi_p(r)\) are Gaussian functions and \(\alpha_p\) are fixed exponents.

The energy functional being minimized is:

\[
E_G = 2 \sum_{p,q} C_p C_q h_{pq} + \sum_{p,q,r,s} Q_{prqs} C_p C_q C_r C_s
\]

subject to normalization:

\[
\sum_{p,q} C_p S_{pq} C_q = 1
\]

## Matrix Elements

- **Overlap Matrix \(S_{pq}\):**

\[
S_{pq} = \left( \frac{\pi}{\alpha_p + \alpha_q} \right)^{3/2}
\]

- **Kinetic Energy Matrix \(T_{pq}\):**

\[
T_{pq} = \frac{3 \alpha_p \alpha_q \pi^{3/2}}{(\alpha_p + \alpha_q)^{5/2}}
\]

- **Nuclear Attraction Matrix \(V_{pq}\):**

\[
V_{pq} = - \frac{4\pi}{\alpha_p + \alpha_q}
\]

- **One-Electron Hamiltonian \(h_{pq}\):**

\[
h_{pq} = T_{pq} + V_{pq}
\]

- **Two-Electron Integral Tensor \(Q_{prqs}\):**

\[
Q_{prqs} = \frac{2\pi^{5/2}}{(\alpha_p + \alpha_q)(\alpha_r + \alpha_s)\sqrt{\alpha_p + \alpha_q + \alpha_r + \alpha_s}}
\]

## Self-Consistent Field Procedure

1. Initialize coefficients \(C_p\) uniformly and normalize with respect to \(S\).
2. Construct Fock matrix:

\[
F_{pq} = h_{pq} + \sum_{r,s} Q_{prqs} C_r C_s
\]

3. Solve the generalized eigenvalue problem:

\[
F C = E S C
\]

4. Normalize the resulting eigenvector and check for convergence.
5. Repeat steps 2–4 until convergence is achieved.
6. Compute the total energy \(E_G\).

## Usage

```bash
$ python main.py
Ground state energy: -2.85516038 Hartree
```