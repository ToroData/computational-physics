"""
bessel_functions.py

This module implements upward recursion methods for computing spherical Bessel functions:
- Regular (j_l)
- Irregular (n_l)

The method starts from the base cases l=0 and l=1, and uses recurrence relations to obtain higher orders.

Usage:
    Run as a script to see usage examples:
    $ python bessel_functions.py
"""
import math

def regular_bessel_upwards(l: int, x: float) -> float:
    """
    Calculates the regular spherical Bessel function j_l(x) using upward recursion.
    
    Parameters:
        l (int): Order of the Bessel function (l ≥ 0).
        x (float): Real-world argument of the function (x ≠ 0).
    
    Returns:
        float: Approximate value of j_l(x).
    
    Reference:
        Recurrence relation: j_{l+1}(x) = (2l+1)/x * j_l(x) - j_{l-1}(x)
    """
    j_prev: float = math.sin(x)/x
    j_curr: float = (math.sin(x) / x ** 2) - (math.cos(x) / x)

    if l == 0:
        return j_prev
    elif l == 1:
        return j_curr

    for i in range(2, l + 1):
        j_next: float = (2 * i - 1)/x * j_curr - j_prev
        j_prev, j_curr = j_curr, j_next
    return j_curr

def irregular_bessel_upwards(l: int, x: float) -> float:
    """
    Calculates the irregular spherical Bessel function n_l(x) using upward recurrence.
    
    Parameters:
        l (int): Order of the Bessel function (l ≥ 0).
        x (float): Real-world argument of the function (x ≠ 0).
    
    Returns:
        float: Approximate value of n_l(x).
    
    Reference:
        Recurrence relation: n_{l+1}(x) = (2l+1)/x * n_l(x) - n_{l-1}(x)
    """
    n_prev: float = - math.cos(x)/x
    n_curr: float = (- math.cos(x) / x ** 2) - (math.sin(x) / x)

    if l == 0:
        return n_prev
    elif l == 1:
        return n_curr

    for i in range(2, l + 1):
        n_next: float = (2 * i - 1)/x * n_curr - n_prev
        n_prev, n_curr = n_curr, n_next
    return n_curr

if __name__ == "__main__":
    print(regular_bessel_upwards(5, 1.5))
    print(irregular_bessel_upwards(5, 1.5))