#!/usr/bin/env python3
"""Run Powell's direction-set method with simple CLI.

Usage examples:
  # Space-separated x0 (preferred)
  python -m scripts.run_experiment --func rosenbrock  --x0 -1.2  1.0  --tol 1e-6 --max-iter 300
  python -m scripts.run_experiment --func quad2d      --x0  1.5 -0.7  --tol 1e-8 --max-iter 200
  python -m scripts.run_experiment --func powell-bad  --x0  0.0  0.0  --tol 1e-6 --max-iter 400
  python -m scripts.run_experiment --func rastrigin   --x0  0.2 -0.3  --tol 1e-6 --max-iter 300
  python -m scripts.run_experiment --func himmelblau  --x0 -2.0  2.0  --tol 1e-6 --max-iter 300

Notes:
- Run from the project root: `python -m scripts.run_experiment ...`
- Uses src.powell.powell which returns the final iterate `x*`. We compute f(x*), save a CSV row,
  and, if 2D, generate a contour plot marking start and end points.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import sys
import time
from typing import Callable, Dict

import numpy as np
import matplotlib.pyplot as plt

# Local imports (must run from repo root)
try:
    from src.powell import powell
    # Import the whole registry so new functions automatically show up in the CLI
    from src.functions import FUNCTIONS as FUNC_MAP  # dict[str, Callable[[np.ndarray], float]]
except Exception:
    print("[ERROR] Failed to import local modules. Run from project root and use '-m'.")
    print("Example: python -m scripts.run_experiment --func rosenbrock --x0 -1.2 1.0")
    raise

# --- Helpers -----------------------------------------------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save_csv_row(path: Path, row: dict) -> None:
    header = list(row.keys())
    exists = path.exists()
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)

def make_contour_plot(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    x_star: np.ndarray,
    out_png: Path,
    func_name: str,
    levels: int = 20,
) -> None:
    n = x0.size
    if n != 2:
        return  # only plot 2D

    # Heuristic plot ranges per function
    ranges = {
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
    (xr, yr) = ranges.get(func_name, ((-3, 3), (-3, 3)))

    X = np.linspace(*xr, 400)
    Y = np.linspace(*yr, 400)
    XX, YY = np.meshgrid(X, Y)
    ZZ = np.zeros_like(XX)
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            ZZ[i, j] = f(np.array([XX[i, j], YY[i, j]]))

    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contour(XX, YY, ZZ, levels=levels)
    ax.clabel(cs, inline=1, fontsize=8)
    ax.plot([x0[0]], [x0[1]], 'o', label='start')
    ax.plot([x_star[0]], [x_star[1]], '*', markersize=12, label='final')
    ax.set_title(f"Powell on {func_name}")
    ax.set_xlabel('x1'); ax.set_ylabel('x2')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

# --- CLI ---------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run Powell's method on a chosen test function.")
    ap.add_argument(
        '--func',
        choices=sorted(FUNC_MAP.keys()),
        default='rosenbrock',
        help='Objective function to minimize.',
    )
    ap.add_argument(
        '--x0',
        type=float,
        nargs='+',
        default=[-1.2, 1.0],
        help="Initial point as space-separated floats, e.g. --x0 -1.2 1.0",
    )
    ap.add_argument('--tol', type=float, default=1e-6, help='Stopping tolerance.')
    ap.add_argument('--max-iter', type=int, default=300, help='Maximum outer iterations.')
    ap.add_argument('--outdir', type=str, default='plots', help='Directory to save outputs.')
    ap.add_argument('--prefix', type=str, default=None, help='Filename prefix for outputs.')
    ap.add_argument('--levels', type=int, default=20, help='Contour levels for 2D plots.')
    args = ap.parse_args(argv)

    f = FUNC_MAP[args.func]
    x0 = np.array(args.x0, dtype=float)
    n = x0.size
    D0 = np.eye(n)  # columns are initial directions

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    t0 = time.perf_counter()
    x_star = powell(f, x0, D0, tol=args.tol, max_iter=args.max_iter)
    t1 = time.perf_counter()
    f_star = float(f(x_star))

    # Console summary
    print("Function  :", args.func)
    print("x0        :", x0)
    print("x*        :", x_star)
    print("f(x*)     :", f_star)
    print("tol       :", args.tol)
    print("max_iter  :", args.max_iter)
    print("elapsed s :", round(t1 - t0, 6))

    # Save CSV
    prefix = args.prefix or f"{args.func}_n{n}"
    csv_path = outdir / f"{prefix}_results.csv"
    save_csv_row(csv_path, {
        'func': args.func,
        'n': n,
        'x0': ','.join(f"{v:.6g}" for v in x0),
        'tol': args.tol,
        'max_iter': args.max_iter,
        'x_star': ','.join(f"{v:.9g}" for v in x_star),
        'f_star': f"{f_star:.12g}",
        'elapsed_s': f"{t1 - t0:.6f}",
    })
    print(f"Saved results to {csv_path}")

    # Plot if 2D
    if n == 2:
        png_path = outdir / f"{prefix}_contour.png"
        # Fix accidental curly brace in filename if present
        png_path = outdir / f"{prefix}_contour.png"
        make_contour_plot(f, x0, x_star, png_path, args.func, levels=args.levels)
        print(f"Saved plot to {png_path}")

    return 0

if __name__ == '__main__':
    sys.exit(main())