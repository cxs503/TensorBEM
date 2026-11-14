"""CASE-BEM-CRACK-SIF: Centre-crack Mode-I SIF benchmark.

Reference
---------
Tada, Paris & Irwin, *The Stress Analysis of Cracks Handbook*, 3rd ed.,
ASME Press, 2000, Eq. 2.1a (p. 2.3):
    K_I = sigma * sqrt(pi * a)
for an infinite plate with a through-thickness centre crack of half-length ``a``
under remote uniform tension ``sigma``.

This benchmark verifies the TensorBEM displacement-discontinuity (DD) fracture
mechanics capability against the closed-form solution.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

# Ensure the package is importable
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from tensorbem.crack_sif import (
    CrackSIFResult,
    analytical_cod,
    analytical_sif,
    convergence_study,
    solve_crack_sif,
)

# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------
SIGMA = 100.0e6       # 100 MPa remote tension
A = 0.01              # 10 mm half crack length
YOUNG = 2.0e11        # 200 GPa (steel)
POISSON = 0.3
N_ELEMENTS = 200      # for constant-DD convergence study

# ---------------------------------------------------------------------------
# Run the benchmark
# ---------------------------------------------------------------------------

def main() -> CrackSIFResult:
    """Run the CASE-BEM-CRACK-SIF benchmark and return the result."""
    result = solve_crack_sif(
        sigma=SIGMA,
        a=A,
        young=YOUNG,
        poisson=POISSON,
        n_elements=N_ELEMENTS,
        method="chebyshev",
    )
    return result


def run_convergence() -> list[CrackSIFResult]:
    """Run a convergence study with the constant-DD method."""
    n_list = [10, 20, 40, 80, 160, 320, 640]
    return convergence_study(SIGMA, A, YOUNG, POISSON, n_list, method="constant")


if __name__ == "__main__":
    result = main()
    print(f"CASE-BEM-CRACK-SIF benchmark")
    print(f"  sigma = {SIGMA/1e6:.1f} MPa")
    print(f"  a     = {A*1e3:.1f} mm")
    print(f"  E     = {YOUNG/1e9:.1f} GPa")
    print(f"  nu    = {POISSON}")
    print(f"  K_ref = {result.sif_ref:.6f} MPa·sqrt(m)")
    print(f"  K_got = {result.sif_got:.6f} MPa·sqrt(m)")
    print(f"  error = {result.error_pct:.4e}%")
    print(f"  PASS" if result.error_pct < 5.0 else f"  FAIL")
