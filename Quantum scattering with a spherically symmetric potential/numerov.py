"""
This module implements the Numerov method for solving the radial Schrödinger equation
for a spherically symmetric potential. The potential is defined as V(r) = r^2, which is a common
model in quantum mechanics. The Numerov method is a numerical technique used to solve second-order differential equations.
It is particularly useful for problems in quantum mechanics where the wavefunction must be computed over a radial grid.
"""
import numpy as np

def build_grid(h: float, r_max: float, E: float) -> tuple[np.ndarray, int]:
    """
    Constructs the radial grid and computes total number of points.
    The grid is built such that it can accommodate the wavelength
    corresponding to the energy E, ensuring sufficient resolution.

    Parameters:
        h : float
        r_max : float
        E : float
    Returns:
        r : numpy array of radial points
        N : int, total number of points in the grid
    """
    λ = 2 * np.pi / np.sqrt(E)
    N = int((r_max + λ) / h) + 5
    r = np.linspace(0, h * N, N + 1)
    return r, N


def effective_potential(r: np.ndarray, l: int, E: float) -> np.ndarray:
    """
    Computes the effective potential F(r) = V(r) + l(l+1)/r^2 - E for V(r) = r^2.
    This is used in the Numerov method to solve the radial Schrödinger equation.
    Parameters:
        r : numpy array of radial points
        l : angular momentum quantum number
        E : energy eigenvalue
    Returns:
        F : numpy array of effective potential values at each radial point
    """
    F = np.zeros_like(r)
    with np.errstate(divide='ignore'):
        F[1:] = r[1:]**2 + l * (l + 1) / r[1:]**2 - E
    F[0] = F[1]
    return F


def initialize_conditions(h: float, l: int) -> tuple[float, float]:
    """
    Returns initial values u[0], u[1] and corresponding w[0], w[1].
    These values are used to start the Numerov integration process.
    Parameters:
        h : float, step size in the radial grid
        l : angular momentum quantum number
    """
    u0 = 0.0
    u1 = h**(l + 1)
    return u0, u1

def numerov_integrate(r: np.ndarray, h: float, F: np.ndarray, u0: float, u1: float) -> np.ndarray:
    """Applies Numerov method to compute the wavefunction u(r)."""
    N = len(r) - 1
    u = np.zeros(N + 1)
    w = np.zeros(N + 1)

    u[0], u[1] = u0, u1
    w[0] = u0 * (1 - h**2 * F[0] / 12)
    w[1] = u1 * (1 - h**2 * F[1] / 12)

    for n in range(1, N):
        w[n + 1] = 2 * w[n] - w[n - 1] + h**2 * F[n] * u[n]
        u[n + 1] = w[n + 1] / (1 - h**2 * F[n + 1] / 12)

    return u

def numerov_algorithm(h: float, l: int, E: float, r_max: float) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """
    Main function to execute the Numerov algorithm for solving the radial Schrödinger equation.
    It builds the radial grid, computes the effective potential, initializes conditions, and performs the integration.
    Parameters:
        h : float, step size in the radial grid
        l : int, angular momentum quantum number
        E : float, energy eigenvalue
        r_max : float, maximum radial distance
    Returns:
        r1 : float, radial point corresponding to the first boundary condition
        r2 : float, radial point corresponding to the second boundary condition
        u1 : float, wavefunction value at r1
        u2 : float, wavefunction value at r2
        r : numpy array of radial points
        u : numpy array of wavefunction values
    """
    r, N = build_grid(h, r_max, E)
    F = effective_potential(r, l, E)
    u0, u1 = initialize_conditions(h, l)
    u = numerov_integrate(r, h, F, u0, u1)

    k = np.sqrt(E)
    λ = 2 * np.pi / k
    r1_index = int(r_max / h)
    r2_index = min(len(r) - 1, int((r_max + λ / 2) / h))

    r1, r2 = r[r1_index], r[r2_index]
    u1, u2 = u[r1_index], u[r2_index]

    return r1, r2, u1, u2, r, u


if __name__ == "__main__":
    h = 0.01
    l = 0
    E = 3.0
    r_max = np.sqrt(E)
    r1, r2, u1, u2, r, u = numerov_algorithm(h, l, E, r_max)
    print(f"r1: {r1}, u1: {u1}")
    print(f"r2: {r2}, u2: {u2}")
