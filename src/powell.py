import numpy as np
from typing import Callable, Tuple

Objective = Callable[[np.ndarray], float]

# ---------- 1D line search: bracket + Brent (derivative-free) ----------

def _bracket_minimum(phi, x0=0.0, step=1.0, grow=2.0, max_iter=50):
    """Expand interval [a,c] until phi(b) is a local min with a<b<c and phi(b)<=phi(a),phi(c)."""
    a = x0
    b = x0 + step
    fa = phi(a)
    fb = phi(b)
    if fb > fa:
        # swap so that fb < fa
        a, b = b, a
        fa, fb = fb, fa
        step = -step
    c = b + step
    fc = phi(c)
    it = 0
    while fc < fb and it < max_iter:
        step *= grow
        a, fa = b, fb
        b, fb = c, fc
        c = b + step
        fc = phi(c)
        it += 1
    return (a, b, c), (fa, fb, fc)


def _brent(phi, a, b, c, fa, fb, fc, tol=1e-8, max_iter=100):
    """Brent's method on a bracketing triplet a<b<c with fb the best."""
    # Ensure a < b < c and fb is min
    if not (a < b < c):
        xs = sorted([(a,fa), (b,fb), (c,fc)], key=lambda t: t[0])
        (a,fa), (b,fb), (c,fc) = xs
    x = w = v = b
    fx = fw = fv = fb
    d = e = 0.0
    for _ in range(max_iter):
        xm = 0.5*(a + c)
        tol1 = tol*abs(x) + 1e-12
        tol2 = 2.0*tol1

        # Convergence in x
        if abs(x - xm) <= (tol2 - 0.5*(c - a)):
            return x, fx

        # Parabolic fit
        p = q = r = 0.0
        if x != w and x != v and w != v:
            r = (x - w)*(fx - fv)
            q = (x - v)*(fx - fw)
            p = (x - v)*q - (x - w)*r
            q = 2.0*(q - r)
            if q > 0:
                p = -p
            q = abs(q)
        use_parabolic = (abs(p) < abs(0.5*q*e)) and (p > q*(a - x)) and (p < q*(c - x))
        if use_parabolic:
            d = p / q
            u = x + d
            # u must be within (a,c); if too close to boundary, move slightly
            if (u - a) < tol2 or (c - u) < tol2:
                d = np.sign(xm - x) * tol1
        else:
            e = (c - x) if x < xm else (a - x)
            d = 0.3819660112501051 * e  # golden step (1-1/phi)

        u = x + (d if d != 0 else np.sign(xm - x)*tol1)
        fu = phi(u)

        if fu <= fx:
            if u < x:
                c = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                c = u
            if (fu <= fw) or (w == x):
                v, w = w, u
                fv, fw = fw, fu
            elif (fu <= fv) or (v == x) or (v == w):
                v, fv = u, fu

    return x, fx  # best we have


def _line_minimize(f, x, p, tol=1e-8):
    """Minimize phi(α)=f(x+α p) over α∈R. Returns α*, f(x+α*p), and x_new."""
    def phi(alpha):
        return f(x + alpha * p)

    (a, b, c), (fa, fb, fc) = _bracket_minimum(phi, x0=0.0)
    alpha, f_new = _brent(phi, a, b, c, fa, fb, fc, tol=max(tol, 1e-12))
    x_new = x + alpha * p
    return alpha, f_new, x_new


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
    callback=None,
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
    callback : callable or None
        Optional callback with signature callback(x, fx, stage), called at key steps.

    Returns
    -------
    x : ndarray
        Final iterate.
    """
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    D = _as_matrix_directions(np.asarray(direction_set, dtype=float), n)

    fx = f(x)

    if callback is not None:
        try:
            callback(x.copy(), fx, "start")
        except Exception:
            pass

    for _ in range(max_iter):
        x_start = x.copy()
        f_start = fx

        best_drop = 0.0
        best_idx = -1
        biggest_step_norm = 0.0

        # 1) Successive line minimizations along current directions
        for j in range(n):
            p = D[:, j].copy()
            if not np.any(np.isfinite(p)) or np.allclose(p, 0.0):
                continue
            alpha, f_new, x_new = _line_minimize(f, x, p, tol=tol*0.1)
            step_norm = abs(alpha) * np.linalg.norm(p)
            if step_norm > biggest_step_norm:
                biggest_step_norm = step_norm
            drop = fx - f_new
            x, fx = x_new, f_new
            if callback is not None:
                try:
                    callback(x.copy(), fx, f"dir_{j}")
                except Exception:
                    pass
            if drop > best_drop:
                best_drop = drop
                best_idx = j

        # 2) Stopping: small step AND small function improvement
        if biggest_step_norm < tol and abs(f_start - fx) < tol:
            return x

        # 3) Net displacement direction
        d = x - x_start
        if np.allclose(d, 0.0):
            # No movement across the sweep; continue to next iteration
            if callback is not None:
                try:
                    callback(x.copy(), fx, "end_iter")
                except Exception:
                    pass
            continue

        # Prefer to replace a zero/near-zero column if no direction achieved a drop
        zero_cols = [j for j in range(n) if np.linalg.norm(D[:, j]) < 1e-15]
        rep_idx = best_idx if best_idx >= 0 else (zero_cols[0] if zero_cols else -1)

        # 4) Acceleration line search from x_start along d
        _, f_trial, x_trial = _line_minimize(f, x_start, d, tol=tol*0.1)

        # 5) Accept acceleration if it improves over either start or post-sweep
        norm_d = np.linalg.norm(d)
        if (f_trial < f_start or f_trial < fx) and rep_idx >= 0:
            x, fx = x_trial, f_trial
            if callback is not None:
                try:
                    callback(x.copy(), fx, "accel")
                except Exception:
                    pass
            if norm_d > 0:
                D[:, rep_idx] = d / norm_d
        else:
            # Still refresh a direction (prefer zero column) to avoid rank loss
            if rep_idx >= 0 and norm_d > 0:
                D[:, rep_idx] = d / norm_d

        if callback is not None:
            try:
                callback(x.copy(), fx, "end_iter")
            except Exception:
                pass

    return x