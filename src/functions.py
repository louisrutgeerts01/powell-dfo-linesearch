import numpy as np


def rosenbrock(x: np.ndarray) -> float:
    """
    Rosenbrock function.

    f(x) = sum_{i=1}^{n-1} [100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2]
    
    Global minimum: f(1,...,1) = 0
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    total = 0.0
    for i in range(n - 1):
        total += 100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return total


def quad2d(x: np.ndarray) -> float:
    """
    Anisotropic quadratic bowl in 2D: 0.5 * x^T Q x with Q = diag(10, 1).
    Global minimum at (0, 0) with f = 0.
    """
    x = np.asarray(x, dtype=float)
    assert x.size == 2, "quad2d expects a 2D input"
    Q = np.array([[10.0, 0.0], [0.0, 1.0]])
    return 0.5 * float(x @ Q @ x)


def sphere(x: np.ndarray) -> float:
    """Sphere function: f(x) = sum(x_i^2). Global min at 0 with f=0."""
    x = np.asarray(x, dtype=float)
    return float(np.dot(x, x))


def sum_of_squares(x: np.ndarray) -> float:
    """Sum-of-squares with index weights: f(x) = sum_{i=1}^n i * x_i^2."""
    x = np.asarray(x, dtype=float)
    idx = np.arange(1, x.size + 1, dtype=float)
    return float(np.sum(idx * x**2))


def rastrigin(x: np.ndarray) -> float:
    """
    Rastrigin function (separable, multimodal):
    f(x) = 10 n + sum[x_i^2 - 10 cos(2π x_i)]. Global min at 0 with f=0.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    return float(10.0 * n + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


def ackley(x: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2.0 * np.pi) -> float:
    """
    Ackley function (multimodal):
    f(x) = -a * exp(-b * sqrt( (1/n) sum x_i^2 )) - exp( (1/n) sum cos(c x_i) ) + a + e
    Global min at 0 with f=0.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    s1 = np.sqrt(np.sum(x**2) / n)
    s2 = np.sum(np.cos(c * x)) / n
    return float(-a * np.exp(-b * s1) - np.exp(s2) + a + np.e)


def styblinski_tang(x: np.ndarray) -> float:
    """
    Styblinski–Tang function:
    f(x) = 0.5 * sum(x_i^4 - 16 x_i^2 + 5 x_i). Global min near x_i ≈ -2.903534.
    """
    x = np.asarray(x, dtype=float)
    return float(0.5 * np.sum(x**4 - 16.0 * x**2 + 5.0 * x))


def himmelblau(x: np.ndarray) -> float:
    """
    Himmelblau's function (2D, multiple minima).
    f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2.
    """
    x = np.asarray(x, dtype=float)
    assert x.size == 2, "himmelblau expects a 2D input"
    X, Y = x[0], x[1]
    return float((X**2 + Y - 11.0)**2 + (X + Y**2 - 7.0)**2)


def beale(x: np.ndarray) -> float:
    """
    Beale function (2D):
    f(x, y) = (1.5 - x + x y)^2 + (2.25 - x + x y^2)^2 + (2.625 - x + x y^3)^2.
    """
    x = np.asarray(x, dtype=float)
    assert x.size == 2, "beale expects a 2D input"
    X, Y = x[0], x[1]
    return float((1.5 - X + X * Y)**2 + (2.25 - X + X * Y**2)**2 + (2.625 - X + X * Y**3)**2)


def booth(x: np.ndarray) -> float:
    """Booth function (2D): f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2."""
    x = np.asarray(x, dtype=float)
    assert x.size == 2, "booth expects a 2D input"
    X, Y = x[0], x[1]
    return float((X + 2.0*Y - 7.0)**2 + (2.0*X + Y - 5.0)**2)


def matyas(x: np.ndarray) -> float:
    """Matyas function (2D): f(x,y) = 0.26(x^2 + y^2) - 0.48 x y."""
    x = np.asarray(x, dtype=float)
    assert x.size == 2, "matyas expects a 2D input"
    X, Y = x[0], x[1]
    return float(0.26*(X**2 + Y**2) - 0.48*X*Y)


def powell_badly_scaled(x: np.ndarray) -> float:
    """
    Powell's badly scaled function (2D):
    f(x, y) = (10^4 x y - 1)^2 + (e^{-x} + e^{-y} - 1.0001)^2.
    Global minimum near (x, y) ≈ (1.098e-5, 9.106)
    """
    x = np.asarray(x, dtype=float)
    assert x.size == 2, "powell_badly_scaled expects a 2D input"
    X, Y = x[0], x[1]
    return float((1.0e4 * X * Y - 1.0)**2 + (np.exp(-X) + np.exp(-Y) - 1.0001)**2)



FUNCTIONS = {
    'rosenbrock': rosenbrock,
    'quad2d': quad2d,
    'sphere': sphere,
    'sum_of_squares': sum_of_squares,
    'rastrigin': rastrigin,
    'ackley': ackley,
    'styblinski_tang': styblinski_tang,
    'himmelblau': himmelblau,
    'beale': beale,
    'booth': booth,
    'matyas': matyas,
    'powell-bad': powell_badly_scaled,
}
