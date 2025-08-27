"""
Python program for golden section search.  This implementation
does not reuse function evaluations and assumes the minimum is c
or d (not on the edges at a or b)
"""

import math

invphi = (math.sqrt(5) - 1) / 2  # 1 / phi


def gss(f, a, b, tolerance=1e-5):
    """
    Golden-section search
    to find the minimum of f on [a,b]

    * f: a strictly unimodal function on [a,b]

    Example:
    >>> def f(x): return (x - 2) ** 2
    >>> x = gss(f, 1, 5)
    >>> print(f"{x:.5f}")
    2.00000

    """
    while b - a > tolerance:
        c = b - (b - a) * invphi
        d = a + (b - a) * invphi
        if f(c) < f(d):
            b = d
        else:  # f(c) > f(d) to find the maximum
            a = c

    return (b + a) / 2