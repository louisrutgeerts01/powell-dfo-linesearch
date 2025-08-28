import numpy as np
import pytest

from src.powell import powell
from src.functions import rosenbrock, quad2d, sphere


@pytest.mark.parametrize(
    "x0, tol, max_iter, atol_x, atol_f",
    [
        (np.array([-1.2, 1.0]), 1e-6, 500, 1e-3, 1e-6),  # classic Rosenbrock start
        (np.array([1.2, 1.2]),   1e-6, 500, 5e-3, 1e-6),  # other basin but should reach (1,1)
    ],
)
def test_rosenbrock_converges(x0, tol, max_iter, atol_x, atol_f):
    n = x0.size
    D0 = np.eye(n)
    x_star = powell(rosenbrock, x0, D0, tol=tol, max_iter=max_iter)
    f_star = rosenbrock(x_star)

    # Global minimum at (1,1) with f=0
    assert f_star <= atol_f, f"Rosenbrock f(x*) too large: {f_star}"
    assert np.allclose(x_star, np.ones_like(x0), atol=atol_x), f"x* not close to (1,...,1): {x_star}"


def test_quad2d_convex_quadratic_to_zero():
    x0 = np.array([1.5, -0.7])
    n = x0.size
    D0 = np.eye(n)
    x_star = powell(quad2d, x0, D0, tol=1e-8, max_iter=300)
    f_star = quad2d(x_star)

    assert np.linalg.norm(x_star) < 1e-6, f"x* should be near 0; got {x_star}"
    assert f_star < 1e-12, f"f(x*) should be near 0; got {f_star}"


def test_sphere_nd():
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=5)
    D0 = np.eye(x0.size)
    x_star = powell(sphere, x0, D0, tol=1e-8, max_iter=400)
    f_star = sphere(x_star)

    assert np.linalg.norm(x_star) < 1e-6
    assert f_star < 1e-12


def test_zero_direction_column_is_ignored():
    # Construct a direction set with a zero column; algorithm should not crash and still converge.
    x0 = np.array([-1.2, 1.0])
    D0 = np.eye(2)
    D0[:, 1] = 0.0  # make the second direction zero
    x_star = powell(rosenbrock, x0, D0, tol=1e-6, max_iter=500)
    f_star = rosenbrock(x_star)

    # We allow a looser tolerance here because losing a direction can slow convergence.
    assert f_star < 1e-4


# tests/test_compare_scipy.py
import numpy as np
import pytest

scipy = pytest.importorskip("scipy", reason="SciPy not installed")
from scipy.optimize import minimize

from src.powell import powell
from src.functions import rosenbrock, quad2d


@pytest.mark.parametrize(
    "f, x0, tol, max_iter, atol_f, atol_x",
    [
        (rosenbrock, np.array([-1.2, 1.0]), 1e-6, 500, 1e-5, 5e-3),
        (quad2d,     np.array([1.5, -0.7]), 1e-8, 300, 1e-12, 1e-6),
    ],
)
def test_compare_with_scipy_powell(f, x0, tol, max_iter, atol_f, atol_x):
    n = x0.size
    D0 = np.eye(n)

    # Your implementation
    x_star = powell(f, x0, D0, tol=tol, max_iter=max_iter)
    f_star = f(x_star)

    # SciPy Powell
    res = minimize(f, x0, method="Powell", options={"xtol": tol, "maxiter": max_iter, "ftol": tol})
    x_sci = res.x
    f_sci = res.fun

    # We compare function values tightly and locations moderately (due to line-search differences).
    assert f_star <= f_sci + atol_f, f"Our f*={f_star} is not comparable to SciPy's {f_sci}"
    assert np.linalg.norm(x_star - x_sci) <= max(atol_x, 10 * tol), f"x* too far from SciPy's: ours={x_star}, scipy={x_sci}"


def test_compare_quad2d_exact_zero():
    x0 = np.array([2.0, -3.0])
    n = x0.size
    D0 = np.eye(n)

    x_star = powell(quad2d, x0, D0, tol=1e-10, max_iter=200)
    f_star = quad2d(x_star)

    res = minimize(quad2d, x0, method="Powell", options={"xtol": 1e-10, "ftol": 1e-10, "maxiter": 200})
    x_sci = res.x
    f_sci = res.fun

    # Both should be essentially at zero.
    assert np.linalg.norm(x_star) < 1e-6 and f_star < 1e-12
    assert np.linalg.norm(x_sci) < 1e-6 and f_sci < 1e-12

    # Locations should be close (convex quadratic).
    assert np.allclose(x_star, x_sci, atol=1e-6)