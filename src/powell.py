import numpy as np
from .line_search import gss
from typing import Callable, Tuple

Objective = Callable[[np.ndarray], float]


def _as_matrix_directions(D: np.ndarray, n: int) -> np.ndarray:
    """
    Ensure directions are an (n, n) matrix where columns are directions.
    Accepts either (n, n) or a list-like of n vectors of length n.
    """
    D = np.asarray(D, dtype=float)
    if D.ndim == 2 and D.shape == (n, n):
        return D.copy()
    if D.ndim == 2 and D.shape == (n,):
        # unlikely case, fall through to error
        pass
    if D.ndim == 1 and D.size == n * n:
        return D.reshape(n, n).copy()
    raise ValueError("direction_set must be an (n, n) array with column directions")


def powell(
    f: Objective,
    x0: np.ndarray,
    direction_set: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> np.ndarray:
    """
    Powell's direction-set derivative-free method (1964).

    Parameters
    ----------
    f : callable
        Objective function f(x) with x shape (n,).
    x0 : array_like
        Initial point (n,).
    direction_set : array_like
        Initial directions as matrix (n, n); columns are search directions.
    tol : float
        Tolerance for both step size and function reduction stopping tests.
    max_iter : int
        Maximum number of outer iterations.

    Returns
    -------
    x : ndarray
        Final iterate.
    """
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    D = _as_matrix_directions(np.asarray(direction_set, dtype=float), n)

    k = 0
    while k < max_iter:
        x_start = x.copy()
        f_start = f(x_start)

        # 1) Successive line minimizations along current directions
        temp = x.copy()
        for i in range(n):
            d = D[:, i]
            # guard against zero direction
            if np.linalg.norm(d) == 0.0:
                continue
            phi = lambda alpha, _temp=temp, _d=d: f(_temp + alpha * _d)
            # Use golden-section (gss) to get the best alpha on a bracket.
            # NOTE: gss should return the scalar alpha that minimizes phi on [a, b].
            alpha = gss(phi, 0.0, 1.0)
            temp = temp + alpha * d

        # 2) Net displacement direction
        d_new = temp - x
        if np.linalg.norm(d_new) <= tol * (1.0 + np.linalg.norm(x)):
            # no meaningful movement in this cycle
            x = temp
            break

        # 3) Shift direction matrix and append new direction as last column
        D = np.hstack([D[:, 1:], d_new.reshape(-1, 1)])

        # 4) Extra line search along the new direction (Powell's acceleration step)
        phi_new = lambda alpha, _temp=temp, _d=d_new: f(_temp + alpha * _d)
        alpha_new = gss(phi_new, 0.0, 1.0)
        x = temp + alpha_new * d_new

        # 5) Stopping tests: step size and function improvement
        if np.linalg.norm(x - x_start) <= tol * (1.0 + np.linalg.norm(x_start)):
            f_now = f(x)
            if abs(f_start - f_now) <= tol * (1.0 + abs(f_start)):
                break

        k += 1

    return x