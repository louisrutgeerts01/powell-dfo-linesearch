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





