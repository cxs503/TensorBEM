#!/usr/bin/env python3
"""Independent re-derivation of the CASE-BEM-ELLIPSOID-ADDED-MASS reference.

Board audit m17233 (subject: m17177).  The subject report takes its reference
values from ``tensorbem.analytic`` — i.e. from the same repository whose BEM
answer is being scored.  A reference that is produced by the code under test is
not an independent anchor, so this script re-derives the reference from the
classical definition and from a published closed form, using a different
implementation (mpmath adaptive quadrature, arbitrary precision) and never
importing the repository's shape-factor routine for the reference itself.

Anchor 1 (definition, Lamb 1932 §114):  for a tri-axial ellipsoid the shape
factors are

    alpha_i = a1 a2 a3 * int_0^inf dlambda / [ (a_i^2 + lambda) *
                                              sqrt(prod_j (a_j^2 + lambda)) ]

with the identity alpha_1 + alpha_2 + alpha_3 = 2, and the added mass along
principal axis i is  A_ii = rho V alpha_i / (2 - alpha_i),  V = 4/3 pi a1a2a3.

Anchor 2 (published closed form, prolate spheroid a > b = c):
    alpha_axial = 2 (1 - e^2) / e^3 * [ atanh(e) - e ],  e = sqrt(1 - b^2/a^2)
    alpha_trans = (2 - alpha_axial) / 2         (by the sum identity)

Anchor 3 (sphere limit):  alpha_i = 2/3  =>  A_ii = rho V / 2 exactly.

Usage:  PYTHONPATH=src python docs/board_audit/m17233/independent_ref_check.py \
            --output docs/board_audit/m17233/rerun/independent_reference.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mpmath as mp

mp.mp.dps = 40


def shape_factor_integral(axes: tuple[float, float, float], axis: int) -> mp.mpf:
    """alpha_axis from Lamb's definition, adaptive mpmath quadrature over [0, inf).

    Substitution lambda = t/(1-t), t in [0,1] maps the semi-infinite interval onto
    the unit interval and removes the need for a hand-picked upper cut-off.
    """
    a = [mp.mpf(x) for x in axes]

    def integrand(t: mp.mpf) -> mp.mpf:
        lam = t / (1 - t)
        delta = mp.sqrt((a[0] ** 2 + lam) * (a[1] ** 2 + lam) * (a[2] ** 2 + lam))
        return a[0] * a[1] * a[2] / ((a[axis] ** 2 + lam) * delta) / (1 - t) ** 2

    # Split at t=0.5 so the (1-t)^-2 Jacobian near t=1 is not sampled at a
    # singular endpoint; both halves are smooth on their open intervals.
    return mp.quad(integrand, [0, 0.5, 1 - mp.mpf("1e-12")])


def added_mass_from_alpha(alpha: mp.mpf, volume: mp.mpf, density: mp.mpf) -> mp.mpf:
    return density * volume * alpha / (2 - alpha)


def prolate_spheroid_closed_form(a: mp.mpf, b: mp.mpf) -> tuple[mp.mpf, mp.mpf]:
    """Lamb's prolate-spheroid shape factors (reference 2 above)."""
    e = mp.sqrt(1 - (b / a) ** 2)
    alpha_axial = 2 * (1 - e ** 2) / e ** 3 * (mp.atanh(e) - e)
    return alpha_axial, (2 - alpha_axial) / 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--density", type=float, default=1025.0)
    args = parser.parse_args()

    axes = (1.5, 1.0, 0.75)
    density = mp.mpf(args.density)
    volume = 4 * mp.pi / 3 * mp.mpf(axes[0]) * mp.mpf(axes[1]) * mp.mpf(axes[2])

    alpha = [shape_factor_integral(axes, i) for i in range(3)]
    alpha_sum = sum(alpha)
    ref_mass = [added_mass_from_alpha(a, volume, density) for a in alpha]

    # ---- comparison with the repository's own analytic module (cross-check) ----
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    try:
        from tensorbem.analytic import ellipsoid_added_mass, ellipsoid_shape_factors

        repo_alpha = [float(x) for x in ellipsoid_shape_factors(axes)]
        repo_mass = [float(x) for x in ellipsoid_added_mass(axes, float(density))]
    except Exception as exc:  # pragma: no cover - audit script, degrade honestly
        repo_alpha, repo_mass = None, None
        print(f"WARNING: repository analytic module unavailable ({exc})")

    checks: dict[str, object] = {
        "definition": (
            "Lamb (1932) Hydrodynamics 5th ed. §114: alpha_i = a1a2a3 * "
            "int_0^inf dlam / ((a_i^2+lam) sqrt(prod_j(a_j^2+lam))), "
            "A_ii = rho V alpha_i/(2-alpha_i)"
        ),
        "implementation": "mpmath adaptive quadrature, 40 digits, substitution lam=t/(1-t)",
        "semi_axes": list(axes),
        "density": float(density),
        "volume_exact": float(volume),
        "alpha_definition": [float(x) for x in alpha],
        "alpha_sum": float(alpha_sum),
        "alpha_sum_identity_residual": float(abs(alpha_sum - 2)),
        "ref_added_mass": [float(x) for x in ref_mass],
    }

    if repo_alpha is not None:
        checks["repo_alpha"] = repo_alpha
        checks["repo_ref_added_mass"] = repo_mass
        checks["alpha_abs_diff_vs_repo"] = [
            abs(float(alpha[i]) - repo_alpha[i]) for i in range(3)
        ]
        checks["alpha_max_rel_diff_vs_repo"] = max(
            abs(float(alpha[i]) - repo_alpha[i]) / repo_alpha[i] for i in range(3)
        )
        checks["ref_mass_max_rel_diff_vs_repo"] = max(
            abs(float(ref_mass[i]) - repo_mass[i]) / repo_mass[i] for i in range(3)
        )

    # ---- anchor 2: prolate spheroid (2,1,1) closed form ----
    a1, a2x = mp.mpf(2), mp.mpf(1)
    axial_cf, trans_cf = prolate_spheroid_closed_form(a1, a2x)
    axial_int = shape_factor_integral((2.0, 1.0, 1.0), 0)
    trans_int = shape_factor_integral((2.0, 1.0, 1.0), 1)
    checks["spheroid_2_1_1"] = {
        "alpha_axial_closed_form": float(axial_cf),
        "alpha_axial_integral": float(axial_int),
        "alpha_axial_abs_diff": float(abs(axial_cf - axial_int)),
        "alpha_transverse_closed_form": float(trans_cf),
        "alpha_transverse_integral": float(trans_int),
        "alpha_transverse_abs_diff": float(abs(trans_cf - trans_int)),
    }

    # ---- anchor 3: sphere limit ----
    sph = [shape_factor_integral((1.0, 1.0, 1.0), i) for i in range(3)]
    checks["sphere_limit"] = {
        "alpha": [float(x) for x in sph],
        "max_abs_diff_from_two_thirds": float(max(abs(x - mp.mpf(2) / 3) for x in sph)),
        "added_mass_over_rhoV": float(
            added_mass_from_alpha(sph[0], 4 * mp.pi / 3, mp.mpf(1)) / (4 * mp.pi / 3)
        ),
    }

    # ---- score the subject's BEM answer against this reference ----
    got = (1191.2955950090695, 2394.8177826981614, 4104.549811866565)  # sub=2, 320 panels
    errors = [abs(got[i] - float(ref_mass[i])) / float(ref_mass[i]) for i in range(3)]
    checks["got_added_mass_320_panels"] = list(got)
    checks["rel_error_320_panels"] = errors
    checks["max_rel_error_320_panels"] = max(errors)
    checks["passes_5pct"] = max(errors) < 0.05
    checks["passes_3pct"] = max(errors) < 0.03

    rendered = json.dumps(checks, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()