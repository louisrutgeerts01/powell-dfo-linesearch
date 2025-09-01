import numpy as np
import pytest

scipy = pytest.importorskip("scipy", reason="SciPy not installed")
from scipy.optimize import minimize

from src.powell import powell
from src.functions import rosenbrock, quad2d

@pytest.mark.parametrize(
    "f, x0, tol, max_iter, atol_f, atol_x",
    [
        (rosenbrock, np.array([-1.2, 1.0]), 1e-6, 1000, 1e-5, 5e-3),
        (quad2d,     np.array([1.5, -0.7]), 1e-8,  400, 1e-12, 1e-6),
    ],
)
def test_compare_with_scipy_powell(f, x0, tol, max_iter, atol_f, atol_x):
    n = x0.size
    D0 = np.eye(n)

    # Your implementation
    x_star = powell(f, x0, D0, tol=tol, max_iter=max_iter)
    f_star = f(x_star)

    # SciPy Powell — pass same initial directions for a fairer comparison
    res = minimize(
        f, x0, method="Powell",
        options={"xtol": tol, "ftol": tol, "maxiter": max_iter, "direc": D0.copy()}
    )
    x_sci = res.x
    f_sci = res.fun

    # Compare values tightly; locations moderately
    assert f_star <= f_sci + atol_f, f"Our f*={f_star} vs SciPy {f_sci}"
    assert np.linalg.norm(x_star - x_sci) <= max(atol_x, 10 * tol), \
        f"x* far from SciPy: ours={x_star}, scipy={x_sci}"

def test_compare_quad2d_exact_zero():
    x0 = np.array([2.0, -3.0])
    D0 = np.eye(2)

    x_star = powell(quad2d, x0, D0, tol=1e-10, max_iter=200)
    f_star = quad2d(x_star)

    res = minimize(
        quad2d, x0, method="Powell",
        options={"xtol": 1e-10, "ftol": 1e-10, "maxiter": 200, "direc": D0.copy()}
    )
    x_sci = res.x
    f_sci = res.fun

    assert np.linalg.norm(x_star) < 1e-6 and f_star < 1e-12
    assert np.linalg.norm(x_sci) < 1e-6 and f_sci < 1e-12
    assert np.allclose(x_star, x_sci, atol=1e-6)