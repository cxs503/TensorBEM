#!/usr/bin/env python3
"""Independent re-derivation of the CASE-BEM-NEUMANN-KELVIN reference, mission 17305.

Auditor-side cross-check (board-audit).  Deliberately imports *nothing* from
TensorBEM: Michell's thin-ship integral is re-implemented from the textbook form

    Rw = (4 rho g^2 / (pi U^2)) * int_0^{pi/2} sec^3(th) |A(th)|^2 dth
    A(th) = int_{-T}^{0} int_{-L/2}^{L/2} (db/dx) exp(kappa sec^2(th) z)
            exp(i kappa sec(th) x) dx dz ,     kappa = g/U^2

with the Wigley half-breadth b(x,z) = (B/2)(1-(2x/L)^2)(1-(z/T)^2) and 2-D
Gauss-Legendre quadrature on the centreplane, so that the mission's published
"ref" (tensorbem.wigley.michell_wave_resistance_canonical) is judged against a
route that shares no code with it.

Output: JSON to runs/m17305/independent_michell.json (relative to worktree root).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

L, B, T = 10.0, 1.0, 0.625
RHO, G = 1025.0, 9.80665
FROUDES = (0.20, 0.25, 0.30, 0.35, 0.40)
NZ, NX, NTH = 160, 480, 2001


def _grid():
    zg, wz = np.polynomial.legendre.leggauss(NZ)
    xg, wx = np.polynomial.legendre.leggauss(NX)
    z = 0.5 * T * (zg - 1.0)          # [-T, 0]
    wz = 0.5 * T * wz
    x = 0.5 * L * xg                  # [-L/2, L/2]
    wx = 0.5 * L * wx
    Z, X = np.meshgrid(z, x, indexing="ij")
    WZ, WX = np.meshgrid(wz, wx, indexing="ij")
    db_dx = 0.5 * B * (1.0 - (Z / T) ** 2) * (-8.0 * X / L**2)
    return Z, X, db_dx * WZ * WX


def rw_michell(fr: float, Z, X, W) -> float:
    u = fr * math.sqrt(G * L)
    kappa = G / u**2
    th = np.linspace(1e-9, 0.5 * math.pi - 1e-9, NTH)
    sec = 1.0 / np.cos(th)
    base = np.exp(kappa * sec[:, None, None] ** 2 * Z[None]) * W[None]
    amp = np.einsum("nij,nij->n", base, np.exp(1j * kappa * sec[:, None, None] * X[None]))
    integ = float(np.trapezoid(np.abs(amp) ** 2 * sec**3, th))
    return 4.0 * RHO * G**2 / (math.pi * u**2) * integ


def main() -> int:
    Z, X, W = _grid()
    out = {"case": "CASE-BEM-NEUMANN-KELVIN", "audit_mission": 17305,
           "method": "independent Michell thin-ship quadrature (no TensorBEM import)",
           "quadrature": {"n_z": NZ, "n_x": NX, "n_theta": NTH,
                          "rule": "Gauss-Legendre in (x,z), trapezoid in theta"},
           "geometry": {"L": L, "B": B, "T": T, "B_over_L": B / L, "T_over_L": T / L,
                        "Cb_closed_form": 4.0 / 9.0},
           "rows": []}
    for fr in FROUDES:
        u = fr * math.sqrt(G * L)
        rw = rw_michell(fr, Z, X, W)
        out["rows"].append({"Fr": fr, "U": u, "rw_independent": rw,
                            "cw_independent_LT": rw / (0.5 * RHO * u**2 * L * T)})
    p = Path("runs/m17305/independent_michell.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    for r in out["rows"]:
        print(f"Fr={r['Fr']:.2f}  Rw={r['rw_independent']:.6f} N  Cw={r['cw_independent_LT']:.6e}")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())