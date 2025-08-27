# Powell’s Derivative-Free Method — Line Search Variants

This repository studies how different **line search strategies** affect the behavior of **Powell’s Conjugate Direction Method** (1964), one of the earliest and most influential **derivative-free optimization (DFO)** algorithms.

We compare two derivative-free line search approaches inside Powell’s method:
- **Brent’s method** (bracketing + quadratic interpolation with golden-section fallback)  
- **Pure quadratic interpolation** (with golden fallback for robustness)

---

## Why this repo?

Powell’s method is a direction-set algorithm: it performs a sequence of line searches along chosen directions, then updates the set with a new direction formed by the net displacement.  
Because it relies **entirely on line searches**, the choice of 1D minimization algorithm can have a major impact on:
- number of function evaluations  
- robustness in narrow valleys (e.g. Rosenbrock)  
- convergence speed  

This repo isolates the **line search component** so we can study these effects directly.


---

## Installation

```bash
# clone repo
git clone https://github.com/YOUR_USERNAME/powell-dfo-linesearch.git
cd powell-dfo-linesearch

# create environment
python -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
