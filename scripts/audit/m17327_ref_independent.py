#!/usr/bin/env python3
"""Independent re-derivation of the *reference* used by CASE-BEM-MARUO-DRIFT.

The audited case (mission m17276) compares a BEM far-field route against a
closed form that the delivery itself implements.  An audit that merely re-runs
the delivery's ``closed_form_case`` would be circular, so this script rebuilds
the reference **from the textbook definitions only**, importing neither
``tensorbem`` nor the delivery:

    incident plane wave e^{ikx}, fixed bottom-mounted vertical cylinder radius a
    MacCamy & Fuchs (1954) exact scattering coefficients (real cosine form)
        A_m = J_m'(ka) / H_m'^(1)(ka),     eps_0 = 1, eps_m = 2 (m >= 1)
        psi_s ->  sqrt(2/(pi k r)) e^{i(kr - pi/4)} S(theta),
        S(theta) = - sum_{m>=0} eps_m A_m cos(m theta)
    Maruo (1957) far-field mean drift force, deep water, head seas
        F_x = rho g A^2 / (2 pi k) * Int_0^{2 pi} |S(theta)|^2 (1 - cos th) d th

Two *different* things are checked, and they are not interchangeable:

 (1) implementation check (internal): the azimuthal integral is evaluated by
     brute-force trapezoidal quadrature of the trigonometric polynomial
     ``|S(th)|^2 (1-cos th)`` (exact for the finite spectrum) and, separately,
     from the modal moments of the same spectrum.  Agreement only proves that
     the closed form used by the delivery reduces correctly; it is *not* an
     external anchor.
 (2) normalisation check (external): the (ka -> 0) limit of *this* implementation
     must reproduce the classical long-wave drift law of a bottom-mounted
     vertical cylinder, F_x = (5 pi^2/16) rho g A^2 a (ka)^3 (Maruo 1960;
     Newman 1977 Sec. 7.3).  The constant is not typed in: it is recomputed as
     ``5*pi**2/16`` and independently recovered from the numerical ka -> 0 trend.

A phase-convention trap worth recording: summing the *complex* modal expansion
over all integers m with ``(-i)^m`` (instead of keeping the cosine form) flips
the relative sign of the odd-m coefficients, and the long-wave limit then comes
out as pi^2/16 instead of 5 pi^2/16 -- a factor-5 error that only the external
anchor (2) exposes.  The first version of this script made exactly that mistake.

Usage
-----
    python3 scripts/audit/m17327_ref_independent.py \
        --study /root/tp-agent/work/m17276/artifacts/maruo_drift_far_field_case/study.json \
        --out artifacts/audit_m17327/independent_ref.json
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import numpy as np
from scipy.special import h1vp, jvp

DENSITY = 1025.0
GRAVITY = 9.81
RADIUS = 1.0
AMPLITUDE = 1.0


def scattering_spectrum_cos(ka: float, m_max: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """MacCamy-Fuchs cosine amplitudes ``eps_m A_m`` for ``m = 0 ... m_max``."""
    x = float(ka)
    m = np.arange(int(m_max) + 1)
    eps = np.where(m == 0, 1.0, 2.0)
    a_mf = jvp(m, x) / h1vp(m, x)
    return m, eps * a_mf


def maruo_integral(ka: float, *, m_max: int = 14, n_theta: int = 20001) -> dict[str, Any]:
    """``Int |S|^2 (1-cos th) d th`` by quadrature and by modal moments."""
    m, w = scattering_spectrum_cos(ka, m_max=m_max)
    th = np.linspace(0.0, 2.0 * math.pi, int(n_theta), endpoint=False)
    s = -(np.cos(np.outer(th, m)) @ w)
    quad = float(np.trapezoid(np.abs(s) ** 2 * (1.0 - np.cos(th)), th))

    # modal moments of the (complex-coefficient) cosine spectrum, from
    #   Int cos(m th) cos(n th) d th = pi delta_mn (twice at m=n=0)
    #   Int cos(m th) cos(n th) cos(th) d th = pi  if {m,n}={0,1};
    #                                         pi/2 if |m-n|=1, m,n>=1
    iso = math.pi * (2.0 * abs(w[0]) ** 2 + float(np.sum(np.abs(w[1:]) ** 2)))
    proj = 2.0 * math.pi * float(np.real(w[0] * np.conj(w[1]))) + math.pi * float(
        np.real(np.sum(w[1:-1] * np.conj(w[2:])))
    )
    modal = iso - proj
    return {
        "ka": float(ka),
        "m_max": int(m_max),
        "integral_quadrature": quad,
        "integral_modal": modal,
        "modal_vs_quadrature_rel": abs(modal - quad) / abs(quad),
    }


def drift_force(ka: float, *, m_max: int = 14, density: float = DENSITY,
                gravity: float = GRAVITY, radius: float = RADIUS,
                amplitude: float = AMPLITUDE) -> dict[str, Any]:
    red = maruo_integral(ka, m_max=m_max)
    k = float(ka) / float(radius)
    force = density * gravity * amplitude**2 * red["integral_quadrature"] / (2.0 * math.pi * k)
    out = dict(red)
    out.update({
        "force": force,
        "force_coefficient": force / (density * gravity * amplitude**2 * radius),
        "long_wave_coefficient": force
        / (density * gravity * amplitude**2 * radius * float(ka) ** 3),
    })
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, help="delivered study.json of m17276")
    ap.add_argument("--out", default="artifacts/audit_m17327/independent_ref.json")
    ap.add_argument("--m-max", type=int, default=14)
    args = ap.parse_args(argv)

    with open(args.study) as fh:
        study = json.load(fh)
    delivered = {float(row["ka"]): row for row in study["reference"]}
    kas = sorted(delivered)

    rows = []
    for ka in kas:
        mine = drift_force(ka, m_max=args.m_max)
        theirs = delivered[ka]
        ref_force = float(theirs["mean_drift_force_x"])
        rows.append({
            "ka": ka,
            "ref_force_delivered": ref_force,
            "ref_force_independent": mine["force"],
            "rel_diff": abs(mine["force"] - ref_force) / abs(ref_force),
            "integral_delivered": float(theirs["integral_weighted"]),
            "integral_independent_quadrature": mine["integral_quadrature"],
            "integral_independent_modal": mine["integral_modal"],
            "modal_vs_quadrature_rel": mine["modal_vs_quadrature_rel"],
            "coefficient_delivered": float(theirs["force_coefficient"]),
            "coefficient_independent": mine["force_coefficient"],
        })

    # external anchor: (ka -> 0) limit of *this* implementation vs 5 pi^2 / 16
    tiny = [0.002, 0.005, 0.01, 0.02, 0.05]
    anchor_rows = [{"ka": ka, **{k: drift_force(ka, m_max=args.m_max)[k]
                                 for k in ("force", "long_wave_coefficient")}}
                   for ka in tiny]
    law = 5.0 * math.pi**2 / 16.0
    xs = np.array([row["ka"] ** 2 for row in anchor_rows], dtype=float)
    ys = np.array([row["long_wave_coefficient"] for row in anchor_rows], dtype=float)
    slope, intercept = np.polyfit(xs, ys, 1)
    limit = float(intercept)
    trunc8 = drift_force(1.0, m_max=8)
    trunc14 = drift_force(1.0, m_max=args.m_max)

    # the same long-wave anchors as recorded by the delivery, re-derived here
    delivered_anchors = {float(row["ka"]): row for row in study.get("long_wave_anchors", [])}
    anchor_vs_delivered = []
    for ka, row in sorted(delivered_anchors.items()):
        mine = drift_force(ka, m_max=args.m_max)
        anchor_vs_delivered.append({
            "ka": ka,
            "closed_form_force_delivered": float(row["closed_form_force"]),
            "closed_form_force_independent": mine["force"],
            "rel_diff": abs(mine["force"] - float(row["closed_form_force"]))
            / abs(float(row["closed_form_force"])),
        })

    payload = {
        "what": "independent (tensorbem-free) re-derivation of the m17276 reference",
        "conventions": {
            "incident": "e^{ikx} plane wave, amplitude A",
            "A_m": "J_m'(ka)/H_m'^(1)(ka)  (MacCamy & Fuchs 1954), cosine form with eps_m",
            "S_theta": "-sum_m eps_m A_m cos(m theta)  (far-field scattering amplitude)",
            "F_x": "rho g A^2/(2 pi k) * Int |S|^2 (1-cos th) d th  (Maruo 1957)",
        },
        "m_max": args.m_max,
        "n_theta": 20001,
        "per_ka": rows,
        "long_wave_anchor": {
            "law": "F_x = (5 pi^2/16) rho g A^2 a (ka)^3",
            "law_constant_from_pi": law,
            "rows": anchor_rows,
            "extrapolated_limit_ka2_zero": limit,
            "rel_diff_limit_vs_law": abs(limit - law) / law,
            "fit_slope_vs_ka2": float(slope),
            "note": "the departure from the law is O(ka^2); intercept = ka->0 limit",
        },
        "long_wave_vs_delivered": anchor_vs_delivered,
        "modal_truncation": {
            "ka": 1.0,
            "integral_m_max_8": trunc8["integral_quadrature"],
            "integral_m_max_14": trunc14["integral_quadrature"],
            "rel_diff": abs(trunc8["integral_quadrature"] - trunc14["integral_quadrature"])
            / abs(trunc14["integral_quadrature"]),
        },
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")

    for row in rows:
        print(f"ka={row['ka']:.2f} ref_delivered={row['ref_force_delivered']:.12g} "
              f"ref_independent={row['ref_force_independent']:.12g} "
              f"rel={row['rel_diff']:.2e} "
              f"modal_vs_quad={row['modal_vs_quadrature_rel']:.2e}")
    print(f"long-wave limit (this implementation) = {limit:.12g} "
          f"vs 5pi^2/16 = {law:.12g} (rel {abs(limit - law) / law:.2e})")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())