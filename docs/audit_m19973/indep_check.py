#!/usr/bin/env python3
"""Independent audit checks for m19973 (CASE-BEM-HELMHOLTZ-SPHERE).

Route A: Mie series re-implemented from scratch with mpmath (Riccati-Bessel
         built from half-integer Bessel functions, no scipy).
Route B: scattering cross-section by direct angular integration of the Mie
         far-field amplitude (no Parseval shortcut).
Route C: optical theorem sigma = (4 pi / k) Im f(0).
Route D: small-ka Rayleigh asymptote sigma -> (7 pi / 9) (ka)^4 a^2.
Route E: sensitivity of the *audited* cross-section quadrature (N_theta=64,
         N_phi=64) by refining it, i.e. is the reported BEM error signal or
         quadrature noise.

Writes runs/audit_m19973/indep_check.json.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
sys.path.insert(0, str(WT / "src"))

import mpmath as mp  # noqa: E402

mp.mp.dps = 30

KA = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
A_RADIUS = 1.0


# --------------------------------------------------------------------------
# Route A: independent Mie series (mpmath)
# --------------------------------------------------------------------------
def riccati(n, x):
    """Return (psi_n, psi_n', xi_n, xi_n') as mpf/mpc."""
    j = lambda z, o: mp.sqrt(mp.pi / (2 * z)) * mp.besselj(o + mp.mpf(1) / 2, z)
    y = lambda z, o: mp.sqrt(mp.pi / (2 * z)) * mp.bessely(o + mp.mpf(1) / 2, z)
    h = lambda z, o: j(z, o) + 1j * y(z, o)
    # d/dx of psi via derivative of j_n
    def dj(z, o):
        return j(z, o - 1) - (o + 1) * j(z, o) / z if o > 0 else -j(z, 1)
    def dh(z, o):
        return h(z, o - 1) - (o + 1) * h(z, o) / z if o > 0 else -h(z, 1)
    return j(x, n), dj(x, n), h(x, n), dh(x, n)


def mie_sigma_mpmath(ka, n_max):
    x = mp.mpf(str(ka))
    coeffs = []
    for n in range(n_max + 1):
        _, dpsi, _, dxi = riccati(n, x)
        coeffs.append(-dpsi / dxi)
    s = mp.fsum([(2 * n + 1) * abs(a) ** 2 for n, a in enumerate(coeffs)])
    return 4 * mp.pi / x**2 * s, coeffs


def route_B_angular(ka, coeffs, n_theta=240):
    """sigma = int |f|^2 dOmega with f(theta) = -i/k sum (2n+1) A_n P_n(cos th)."""
    k = mp.mpf(str(ka))
    xs, ws = mp.quadgl(lambda t: (t, 1), [-1, 1], maxdegree=n_theta) if False else (None, None)
    # Gauss-Legendre on cos(theta) via mpmath nodes/weights
    nodes, weights = np.polynomial.legendre.leggauss(n_theta)
    total = 0.0 + 0j
    for mu, w in zip(nodes, weights):
        f = 0.0 + 0j
        for n, a in enumerate(coeffs):
            f += (2 * n + 1) * complex(a) * float(mp.legendre(n, float(mu)))
        f = -1j / complex(k) * f
        total += float(w) * abs(f) ** 2
    return float((2 * math.pi * total).real)  # azimuthal symmetry


def route_C_optical(ka, coeffs):
    k = complex(ka)
    f0 = -1j / k * sum((2 * n + 1) * complex(a) for n, a in enumerate(coeffs))
    return (4 * math.pi / ka) * f0.imag


def main():
    import tensorbem.acoustic3d as ac

    out = {"ka": KA, "routes": [], "quadrature_sensitivity": []}

    print(f"{'ka':>5} {'repo_mie':>12} {'routeA':>12} {'routeB':>12} "
          f"{'routeC':>12} {'A-B rel':>10} {'A-C rel':>10} {'repo-A rel':>11}")
    for ka in KA:
        n_max = int(math.ceil(ka + 4 * ka ** (1 / 3) + 10))
        sigmaA, coeffs = mie_sigma_mpmath(ka, n_max)
        sigmaA = float(sigmaA)
        sigmaB = route_B_angular(ka, coeffs)
        sigmaC = route_C_optical(ka, coeffs)
        repo = float(ac.rigid_sphere_scattering_cross_section(ka / A_RADIUS, A_RADIUS))
        rec = {
            "ka": ka,
            "repo_mie": repo,
            "routeA_mpmath_series": sigmaA,
            "routeB_direct_angular": sigmaB,
            "routeC_optical_theorem": sigmaC,
            "rel_diff_A_repo": abs(sigmaA - repo) / repo,
            "rel_diff_B_repo": abs(sigmaB - repo) / repo,
            "rel_diff_C_repo": abs(sigmaC - repo) / repo,
            "sigma_over_pi_a2": repo / math.pi,
        }
        out["routes"].append(rec)
        print(f"{ka:5.2f} {repo:12.6f} {sigmaA:12.6f} {sigmaB:12.6f} {sigmaC:12.6f} "
              f"{abs(sigmaA-sigmaB)/sigmaA:10.2e} {abs(sigmaA-sigmaC)/sigmaA:10.2e} "
              f"{abs(sigmaA-repo)/repo:11.2e}")

    # Route D: Rayleigh asymptote (7 pi/9)(ka)^4 a^2
    print("\nRoute D: small-ka Rayleigh asymptote")
    out["rayleigh"] = []
    for ka in (0.02, 0.05, 0.1, 0.25):
        repo = float(ac.rigid_sphere_scattering_cross_section(ka, A_RADIUS))
        rayleigh = 7 * math.pi / 9 * ka**4
        out["rayleigh"].append({"ka": ka, "mie": repo, "rayleigh_7pi_over_9": rayleigh,
                                "rel_diff": abs(rayleigh - repo) / repo})
        print(f"  ka={ka:5.3f} mie={repo:.6e} rayleigh={rayleigh:.6e} "
              f"rel={abs(rayleigh-repo)/repo:.2e}")

    # Route E: quadrature sensitivity of the audited sigma
    print("\nRoute E: far-field quadrature sensitivity (audited N_theta=N_phi=64)")
    from tensorbem.geometry3d import icosphere
    inc = np.array([1.0, 0.0, 0.0])
    for ka in (0.25, 2.0, 5.0):
        for sub in (2, 3):
            if ka == 5.0 and sub == 3:
                continue  # keep wall time bounded; covered by sub=2 trend
            mesh = icosphere(sub, A_RADIUS)
            k = ka / A_RADIUS
            t0 = time.time()
            sol = ac.solve_rigid_scattering(mesh, k, inc)
            sigs = {}
            for nt, npz in ((64, 64), (96, 96), (128, 128), (192, 192)):
                sigs[f"{nt}x{npz}"] = float(ac.scattering_cross_section(
                    mesh, k, sol.phi, sol.normal_velocity, n_theta=nt, n_phi=npz))
            base = sigs["64x64"]
            spread = max(abs(v - base) / base for v in sigs.values())
            mie = float(ac.rigid_sphere_scattering_cross_section(k, A_RADIUS))
            out["quadrature_sensitivity"].append({
                "ka": ka, "subdivisions": sub, "panels": mesh.panel_count,
                "sigma": sigs, "err_vs_mie_64x64": abs(base - mie) / mie,
                "max_rel_change_under_refinement": spread,
                "solve_s": round(time.time() - t0, 2),
            })
            print(f"  ka={ka} sub={sub} ({mesh.panel_count}p): "
                  + " ".join(f"{q}={v:.6f}" for q, v in sigs.items())
                  + f" | max drift {spread:.2e}, err(64x64) {abs(base-mie)/mie*100:.3f}%")

    (HERE / "indep_check.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\nwrote", HERE / "indep_check.json")


if __name__ == "__main__":
    main()