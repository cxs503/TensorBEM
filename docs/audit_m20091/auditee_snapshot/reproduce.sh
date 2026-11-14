#!/usr/bin/env python3
"""One-line reproduction of CASE-BEM-CRACK-SIF."""
import sys; sys.path.insert(0, "src")
from tensorbem.crack_sif import solve_crack_sif
r = solve_crack_sif(sigma=100000000.0, a=0.01, young=200000000000.0, poisson=0.3, method="chebyshev")
print(f"K_ref = {r.sif_ref:.6f} Pa·√m")
print(f"K_got = {r.sif_got:.6f} Pa·√m")
print(f"Error = {r.error_pct:.4e}%")
assert r.error_pct < 5.0, f"FAIL: error {r.error_pct:.2f}% >= 5%"
print("PASS")
