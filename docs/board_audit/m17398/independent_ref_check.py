#!/usr/bin/env python3
"""Independent re-derivation of the CASE-BEM-MARUO-DRIFT reference (v2).

Nothing from the TensorBEM repo is imported.  Rebuilt from first principles:

  phi_I = A (g/omega) exp(-k z) exp(i k r cos th)
        = A (g/omega) exp(-k z) sum_{m=-inf}^{inf} i^m J_m(kr) e^{i m th}
  phi_s = A (g/omega) exp(-k z) sum_{m=-inf}^{inf} alpha_m H_m(kr) e^{i m th}
  body BC at r = a:   i^m J_m'(ka) + alpha_m H_m'(ka) = 0
                      -> alpha_m = -i^m J_m'(ka) / H_m'(ka)
  far field (outgoing): H_m(kr) ~ sqrt(2/(pi k r)) e^{i(k r - m pi/2 - pi/4)}
                      -> S(th) = sum_m c_m e^{i m th},  c_m = alpha_m e^{-i m pi/2}

  F_x = rho g A^2 / (2 pi k) * Int_0^{2pi} |S|^2 (1 - cos th) d th

The angular integral is obtained two ways (all-orders series, no folding
shortcut): brute-force trapezoid quadrature of S(th), and the modal identity
2 pi [sum_m |c_m|^2 - Re sum_m c_m conj(c_{m+1})].  Both are compared against
the numbers the audited case ships (docs/case_bem_maruo_drift/study.json), and
the (ka -> 0) limit is compared against 5 pi^2 / 16.

Usage: python3 independent_ref_check.py [out.json]
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
from scipy.special import hankel1, jv, jvp

RHO, G, RAD, AMP = 1025.0, 9.81, 1.0, 1.0
MMAX = 60


def coeffs(ka: float, mmax: int = MMAX) -> dict[int, complex]:
    x = ka
    c: dict[int, complex] = {}
    for m in range(-mmax, mmax + 1):
        am = abs(m)
        dJ = jvp(am, x, 1) * ((-1) ** am if m < 0 else 1.0)
        dH = (hankel1(am - 1, x) - hankel1(am + 1, x)) / 2.0
        dH *= (-1) ** am if m < 0 else 1.0
        alpha = -(1j**m) * dJ / dH
        c[m] = alpha * np.exp(-1j * m * math.pi / 2.0)
    return c


def scattering(theta: np.ndarray, c: dict[int, complex]) -> np.ndarray:
    s = np.zeros(theta.shape, dtype=complex)
    for m, cm in c.items():
        s += cm * np.exp(1j * m * theta)
    return s


def integral_modal(c: dict[int, complex]) -> float:
    tot = sum(abs(cm) ** 2 for cm in c.values())
    cross = sum((c[m] * np.conj(c[m + 1])).real for m in c if m + 1 in c)
    return 2.0 * math.pi * (tot - cross)


def main() -> int:
    th = np.linspace(0.0, 2.0 * math.pi, 20000, endpoint=False)
    delivered = {0.2: 240.5907, 0.5: 2875.6570, 1.0: 6685.9935, 2.0: 6306.3981}
    rows = []
    for ka in (0.2, 0.5, 1.0, 2.0):
        c = coeffs(ka)
        S = scattering(th, c)
        iq = float(np.trapezoid((np.abs(S) ** 2) * (1.0 - np.cos(th)), th))
        im = integral_modal(c)
        scale = RHO * G * AMP**2 / (2.0 * math.pi * ka)
        f_modal = scale * im
        rows.append({
            "ka": ka,
            "S_meansquare": float(np.mean(np.abs(S) ** 2)),
            "integral_modal": im,
            "integral_quadrature": iq,
            "force_modal": f_modal,
            "force_quadrature": scale * iq,
            "delivered_ref_force": delivered[ka],
            "rel_diff_vs_delivered": abs(f_modal - delivered[ka]) / delivered[ka],
            "modal_vs_quadrature_rel": abs(f_modal - scale * iq) / abs(f_modal),
        })
    law = 5.0 * math.pi**2 / 16.0
    long_rows = []
    for ka in (0.01, 0.02, 0.05):
        c = coeffs(ka, mmax=50)
        f = RHO * G / (2.0 * math.pi * ka) * integral_modal(c)
        coeff = f / (RHO * G * RAD * ka**3)
        long_rows.append({"ka": ka, "coefficient": coeff, "law_5pi2_16": law,
                          "rel_err": abs(coeff - law) / law})
    out = {
        "method": "first-principles MacCamy-Fuchs series, all integer orders, no repo import",
        "rows": rows,
        "long_wave": long_rows,
        "max_rel_diff_vs_delivered": max(r["rel_diff_vs_delivered"] for r in rows),
        "max_modal_vs_quadrature_rel": max(r["modal_vs_quadrature_rel"] for r in rows),
        "max_long_wave_rel_err": max(r["rel_err"] for r in long_rows),
    }
    dest = sys.argv[1] if len(sys.argv) > 1 else "independent_ref.json"
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())