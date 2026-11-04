"""Independent verification of the Wigley forward-speed wave resistance.

Why this module exists
----------------------
:mod:`tensorbem.wigley` computes the steady wave resistance of the
mathematical Wigley hull by Michell's thin-ship theory.  The Kochin
function is evaluated in *closed form*:

    H(k) = 2 * (-4 B / L^2) * Ix(k) * Iz(k)

with

    Ix(k) = integral_{-L/2}^{L/2} x e^{ikx} dx          (closed form)
    Iz(k) = integral_{-T}^{0} (1 - (z/T)^2) e^{kz} dz    (closed form)

and the resistance is

    Rw = (4 rho / U^2) * integral_{1}^{lam_max}
              |H(k0 lam)|^2 / (lam^2 sqrt(lam^2 - 1)) dlam,

with the endpoint-removing substitution lam = 1 + s^2.

The merged feature was guarded only by *self-consistency* tests (Kochin
symmetry/decay, positivity, zero-at-zero-speed, mesh convergence, and the
Kochin-vs-pressure two-path agreement).  The closed-form integrals and the
overall prefactor had **no independent reference**: an algebraic slip in
either closed form, or in the factor 2 (port/starboard doubling), would pass
every existing test because both the far-field and the pressure path share
the same ``michell_kochin_function``.

This module closes that gap with references that share **no** closed-form
code with the feature under test:

* ``kochin_direct_2d`` evaluates the *defining* double integral

      H(k) = 2 * integral_{-L/2}^{L/2} integral_{-T}^{0}
               (dy/dx)(x, z) e^{ikx} e^{kz} dz dx

  by a tensor-product Gauss-Legendre rule on the (x, z) parameter domain.
  The integrand is a smooth polynomial times an exponential, so the
  quadrature error is exponentially small and the result is an unbiased
  target for the closed form.

* ``substitution_identity_error`` verifies the change of variables
  lam = 1 + s^2 pointwise: the substituted integrand weight
  ``2 / (lam^2 sqrt(2 + s^2))`` must equal the original weight
  ``2 s / (lam^2 sqrt(lam^2 - 1))`` (with lam = 1 + s^2) up to roundoff.

* ``resistance_from_direct_kochin`` evaluates the far-field integral with
  the *direct* Kochin (same substitution), isolating the closed-form
  contribution to the resistance error.

* ``run_verification`` runs a Froude-number sweep, reporting the relative
  deviation of the closed-form Kochin function and of the wave resistance
  against the direct-quadrature reference, plus the Kochin-vs-pressure
  two-path agreement.

All numbers are emitted programmatically (``--json`` / ``--figure``);
nothing is hand-copied.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from typing import List

import numpy as np
from numpy.polynomial.legendre import leggauss

from tensorbem.wigley import (
    michell_kochin_function,
    michell_wave_resistance,
    wave_resistance_pressure,
    wigley_hull_mesh,
)

_DEFAULT_DENSITY = 1025.0
_DEFAULT_GRAVITY = 9.80665


def _wigley_dydx(x: np.ndarray, z: np.ndarray, B: float, L: float, T: float) -> np.ndarray:
    """dy/dx of the Wigley half-breadth, as a function of (x, z).

    y(x, z) = (B/2) (1 - (2x/L)^2) (1 - (z/T)^2)
    dy/dx   = (B/2) (-8x/L^2) (1 - (z/T)^2) = (-4 B x / L^2) (1 - (z/T)^2)
    """
    return (-4.0 * B * x / L**2) * (1.0 - (z / T) ** 2)


def kochin_direct_2d(
    L: float, B: float, T: float, k: float,
    nx: int = 64, nz: int = 64,
) -> complex:
    """Kochin function by direct 2-D Gauss-Legendre quadrature.

    Independent of the closed-form factorisation in :mod:`tensorbem.wigley`:
    the integrand is assembled pointwise and integrated numerically.
    """
    if k <= 0:
        return 0.0j
    xn, wx = leggauss(nx)
    zn, wz = leggauss(nz)
    x = 0.5 * L * xn            # map [-1,1] -> [-L/2, L/2]
    z = -0.5 * T * (zn + 1.0)   # map [-1,1] -> [-T, 0]
    wxm = 0.5 * L * wx
    wzm = 0.5 * T * wz
    dydx = _wigley_dydx(x[:, None], z[None, :], B, L, T)
    phase = np.exp(1j * k * x[:, None]) * np.exp(k * z[None, :])
    inner = np.sum(dydx * phase * wzm[None, :], axis=1)   # z-integral per x
    return complex(2.0 * np.sum(inner * wxm))


def substitution_identity_error(n_check: int = 2000, seed: int = 0) -> float:
    """Max relative error of the lam = 1 + s^2 change of variables.

    Original integrand weight per dlam:  1 / (lam^2 sqrt(lam^2 - 1))
    Substituted weight per ds:           2 / (lam^2 sqrt(2 + s^2))
    with dlam = 2 s ds and sqrt(lam^2 - 1) = s sqrt(2 + s^2).
    """
    rng = np.random.default_rng(seed)
    s = rng.uniform(1e-6, 10.0, size=n_check)
    lam = 1.0 + s * s
    lhs = 2.0 / (lam**2 * np.sqrt(2.0 + s * s))
    rhs = 2.0 * s / (lam**2 * np.sqrt(lam * lam - 1.0))
    return float(np.max(np.abs(lhs - rhs) / np.maximum(lhs, 1e-300)))


def resistance_from_direct_kochin(
    L: float, B: float, T: float, U: float,
    rho: float = _DEFAULT_DENSITY, g: float = _DEFAULT_GRAVITY,
    n_quad: int = 256, nx: int = 48, nz: int = 48,
) -> float:
    """Far-field resistance using the direct 2-D Kochin (same substitution)."""
    if U <= 0:
        return 0.0
    k0 = g / U**2
    lam_max = max(5.0, 10.0 / (k0 * T))
    nodes, weights = leggauss(n_quad)
    s_max = math.sqrt(lam_max - 1.0)
    s = 0.5 * s_max * (nodes + 1.0)
    w = 0.5 * s_max * weights
    lam = 1.0 + s * s
    acc = 0.0
    for i in range(n_quad):
        Hk = kochin_direct_2d(L, B, T, lam[i] * k0, nx=nx, nz=nz)
        acc += w[i] * abs(Hk) ** 2 * 2.0 / (lam[i] ** 2 * math.sqrt(2.0 + s[i] ** 2))
    return max((4.0 * rho / U**2) * float(acc), 0.0)


@dataclass
class WigleyWaveResistanceVerification:
    L: float
    B: float
    T: float
    froude_numbers: List[float]
    density: float
    gravity: float
    substitution_rel_err: float = 0.0
    rows: List[dict] = field(default_factory=list)

    @property
    def max_kochin_rel_err(self) -> float:
        return max(r["kochin_rel_err"] for r in self.rows)

    @property
    def max_resistance_rel_err(self) -> float:
        return max(r["resistance_rel_err"] for r in self.rows)

    @property
    def max_two_path_rel_err(self) -> float:
        return max(r["two_path_rel_err"] for r in self.rows)


def run_verification(
    L: float = 6.0, B: float = 1.0, T: float = 0.375,
    froude_numbers=(0.15, 0.20, 0.25, 0.30, 0.35, 0.40),
    density: float = _DEFAULT_DENSITY, gravity: float = _DEFAULT_GRAVITY,
    nx: int = 64, nz: int = 64,
) -> WigleyWaveResistanceVerification:
    """Run the Froude sweep and compare closed form vs direct quadrature."""
    study = WigleyWaveResistanceVerification(
        L=L, B=B, T=T, froude_numbers=list(froude_numbers),
        density=density, gravity=gravity,
        substitution_rel_err=substitution_identity_error(),
    )
    for Fr in froude_numbers:
        U = Fr * math.sqrt(gravity * L)
        k0 = gravity / U**2
        lam_max = max(5.0, 10.0 / (k0 * T))
        k_probes = [k0 * lam for lam in (1.0, 1.5, 2.0, 3.0, 5.0, lam_max)]
        kochin_errs = []
        for kp in k_probes:
            h_code = michell_kochin_function(L, B, T, kp)
            h_ref = kochin_direct_2d(L, B, T, kp, nx=nx, nz=nz)
            kochin_errs.append(abs(h_code - h_ref) / max(abs(h_ref), 1e-300))
        rw_code = michell_wave_resistance(L, B, T, U, density, gravity)
        rw_ref = resistance_from_direct_kochin(L, B, T, U, density, gravity)
        mesh = wigley_hull_mesh(L, B, T, nx=32, nz=16)
        rw_press = wave_resistance_pressure(mesh, L, B, T, U, density, gravity)
        rw_scale = max(abs(rw_ref), 1e-300)
        study.rows.append({
            "froude": Fr,
            "speed": U,
            "kochin_rel_err": max(kochin_errs),
            "kochin_rel_err_by_probe": kochin_errs,
            "resistance_closed_form": rw_code,
            "resistance_direct": rw_ref,
            "resistance_rel_err": abs(rw_code - rw_ref) / rw_scale,
            "resistance_pressure": rw_press,
            "two_path_rel_err": abs(rw_code - rw_press) / rw_scale,
            "cw": rw_code / (density * gravity * (4.0 * L * B * T / 9.0)),
        })
    return study


def to_dict(study: WigleyWaveResistanceVerification) -> dict:
    return {
        "hull": {"L": study.L, "B": study.B, "T": study.T,
                 "L_over_B": study.L / study.B, "B_over_T": study.B / study.T,
                 "cubical_coefficient": 4.0 / 9.0},
        "density": study.density,
        "gravity": study.gravity,
        "substitution_rel_err": study.substitution_rel_err,
        "max_kochin_rel_err": study.max_kochin_rel_err,
        "max_resistance_rel_err": study.max_resistance_rel_err,
        "max_two_path_rel_err": study.max_two_path_rel_err,
        "rows": study.rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Wigley wave-resistance verification")
    ap.add_argument("--json", default=None)
    ap.add_argument("--figure", default=None)
    ap.add_argument("--froude", type=float, nargs="+",
                    default=[0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    args = ap.parse_args()
    study = run_verification(froude_numbers=args.froude)
    payload = to_dict(study)
    text = json.dumps(payload, indent=2)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text + "\n")
        print(f"wrote {args.json}")
    else:
        print(text)
    if args.figure:
        _make_figure(study, args.figure)
        print(f"wrote {args.figure}")


def _make_figure(study: WigleyWaveResistanceVerification, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fr = [r["froude"] for r in study.rows]
    cw = [r["cw"] for r in study.rows]
    kerr = [r["kochin_rel_err"] for r in study.rows]
    rerr = [r["resistance_rel_err"] for r in study.rows]
    tp = [r["two_path_rel_err"] for r in study.rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.plot(fr, cw, "o-", color="tab:blue")
    ax.set_xlabel("Froude number Fr = U/sqrt(gL)")
    ax.set_ylabel("Cw = Rw / (rho g V)")
    ax.set_title("Wigley wave-resistance coefficient")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.semilogy(fr, kerr, "s-", color="tab:green", label="Kochin (closed vs 2-D quad)")
    ax.semilogy(fr, rerr, "^-", color="tab:red", label="Rw (closed vs direct Kochin)")
    ax.axhline(0.03, color="k", ls=":", lw=1, label="3% gate")
    ax.set_xlabel("Froude number")
    ax.set_ylabel("relative error vs independent reference")
    ax.set_title("Closed form vs independent quadrature")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.semilogy(fr, tp, "d-", color="tab:purple")
    ax.axhline(0.03, color="k", ls=":", lw=1, label="3% gate")
    ax.set_xlabel("Froude number")
    ax.set_ylabel("relative difference")
    ax.set_title("Two-path agreement (Kochin vs pressure)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
