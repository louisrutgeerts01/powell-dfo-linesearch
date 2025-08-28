# app.py
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
import io
import csv
import time

# Local imports (run from project root)
from src.powell import powell
from src.functions import FUNCTIONS as FUNC_MAP  # dict[name -> callable]

# ---------------- Sidebar controls ----------------
st.set_page_config(page_title="Powell DFO Playground", layout="wide")

st.sidebar.title("Powell’s Method — Playground")
func_name = st.sidebar.selectbox("Objective function", sorted(FUNC_MAP.keys()), index=sorted(FUNC_MAP.keys()).index("rosenbrock") if "rosenbrock" in FUNC_MAP else 0)

# Sensible defaults per function
DEFAULT_X0 = {
    "rosenbrock": [-1.2, 1.0],
    "quad2d": [1.5, -0.7],
    "powell-bad": [0.0, 0.0],
    "himmelblau": [-2.0, 2.0],
    "rastrigin": [0.2, -0.3],
    "ackley": [0.5, -0.5],
    "beale": [1.0, 1.0],
    "booth": [1.0, 3.0],
    "matyas": [0.3, 0.4],
}

default_x0 = DEFAULT_X0.get(func_name, [0.0, 0.0])
dim = st.sidebar.number_input("Dimension (n)", min_value=1, max_value=20, value=len(default_x0), step=1)

# Build x0 inputs dynamically
x0_cols = st.sidebar.columns(min(4, max(1, dim)))
x0_vals = []
for i in range(dim):
    col = x0_cols[i % len(x0_cols)]
    default_val = default_x0[i] if i < len(default_x0) else 0.0
    x0_vals.append(col.number_input(f"x0[{i}]", value=float(default_val), format="%.6f"))

tol = float(st.sidebar.text_input("Tolerance", value="1e-6"))
max_iter = st.sidebar.number_input("Max iterations", min_value=1, max_value=10_000, value=300, step=50)
plot_levels = st.sidebar.slider("Contour levels (2D only)", min_value=5, max_value=60, value=20, step=1)
plot_enabled = st.sidebar.checkbox("Show contour plot (2D only)", value=True)

run_btn = st.sidebar.button("Run Powell")

# ---------------- Main panel ----------------
st.title("Powell’s Direction-Set Method (Derivative-Free)")

st.markdown(
    """
This app runs **Powell’s method** on classic test functions from your repo.
- Choose a function and initial point \(x_0\) in the sidebar.
- Click **Run Powell** to compute the minimizer.
- For 2D functions, a contour plot with start/end points is shown.
    """
)

f = FUNC_MAP[func_name]
x0 = np.array(x0_vals, dtype=float)
n = x0.size
D0 = np.eye(n)

# Cache heavy contour grids so the UI is snappy
@st.cache_data(show_spinner=False)
def contour_grid(func_name: str, fcallable, xr, yr, resolution=400):
    X = np.linspace(*xr, resolution)
    Y = np.linspace(*yr, resolution)
    XX, YY = np.meshgrid(X, Y)
    ZZ = np.zeros_like(XX)
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            ZZ[i, j] = fcallable(np.array([XX[i, j], YY[i, j]]))
    return X, Y, XX, YY, ZZ

# Heuristic plotting ranges for 2D
RANGES = {
    'rosenbrock': ((-2, 2), (-1, 3)),
    'powell-bad': ((-1, 1), (-1, 1)),
    'himmelblau': ((-5, 5), (-5, 5)),
    'beale':      ((-4.5, 4.5), (-4.5, 4.5)),
    'booth':      ((-10, 10), (-10, 10)),
    'matyas':     ((-10, 10), (-10, 10)),
    'quad2d':     ((-3, 3), (-3, 3)),
    'rastrigin':  ((-5.12, 5.12), (-5.12, 5.12)),
    'ackley':     ((-5, 5), (-5, 5)),
}

# Run
if run_btn:
    start = time.perf_counter()
    x_star = powell(f, x0, D0, tol=tol, max_iter=max_iter)
    elapsed = time.perf_counter() - start
    f_star = float(f(x_star))

    st.subheader("Result")
    c1, c2, c3 = st.columns([2, 2, 1.2])
    with c1:
        st.write("**Function**:", func_name)
        st.write("**x0**:", np.array2string(x0, precision=6))
        st.write("**x***:", np.array2string(x_star, precision=9))
    with c2:
        st.write("**f(x*)**:", f"{f_star:.12g}")
        st.write("**tol**:", tol)
        st.write("**max_iter**:", max_iter)
        st.write("**elapsed s**:", f"{elapsed:.6f}")

    # Download CSV with a single result row
    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=["func","n","x0","x_star","f_star","tol","max_iter","elapsed_s"])
    writer.writeheader()
    writer.writerow({
        "func": func_name,
        "n": n,
        "x0": ",".join(f"{v:.6g}" for v in x0),
        "x_star": ",".join(f"{v:.9g}" for v in x_star),
        "f_star": f"{f_star:.12g}",
        "tol": tol,
        "max_iter": max_iter,
        "elapsed_s": f"{elapsed:.6f}",
    })
    st.download_button("Download result CSV", data=csv_buf.getvalue(), file_name=f"{func_name}_result.csv", mime="text/csv")

    # Plot contours for 2D problems
    if plot_enabled and n == 2:
        xr, yr = RANGES.get(func_name, ((-3, 3), (-3, 3)))
        X, Y, XX, YY, ZZ = contour_grid(func_name, f, xr, yr, resolution=400)

        fig, ax = plt.subplots(figsize=(6, 5))
        cs = ax.contour(XX, YY, ZZ, levels=plot_levels)
        ax.clabel(cs, inline=1, fontsize=8)
        ax.plot([x0[0]], [x0[1]], 'o', label='start')
        ax.plot([x_star[0]], [x_star[1]], '*', markersize=12, label='final')
        ax.set_title(f"Powell on {func_name}")
        ax.set_xlabel('x1'); ax.set_ylabel('x2')
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)