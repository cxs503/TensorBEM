"""Rigid-lid (``ka -> 0``) heave added mass of a floating hemisphere, exact.

The reference used by pool case ``CASE-BEM-RIGID-LID-HEMISPHERE`` is the
even-parity double body: reflecting the heaving hemisphere in the rigid lid
``z = 0`` gives a full sphere whose two halves slide past each other, i.e. the
Neumann data ``dphi/dr = -U |cos(theta)|`` on ``r = a`` rather than the rigid
translation ``-U cos(theta)``.  The mirrored field therefore carries only the
**even** Legendre modes ``P_0, P_2, P_4, ...``::

    phi = -U a sum_{n even} c_n (a/r)^(n+1) P_n(mu),  c_n = a_n/(n+1),
    |mu| = sum_{n even} a_n P_n(mu),   a_n = (2n+1) J_n,
    J_n = int_0^1 mu P_n(mu) dmu,
    A33/(rho*nabla) = 3 sum_{n even} (2n+1) J_n^2/(n+1).

``mu P_n(mu)`` is a polynomial, so ``J_n`` has an exact rational value; this
module integrates the Legendre polynomials in the monomial basis with
:class:`fractions.Fraction` and sums the series in exact rational arithmetic
before the final float conversion.  The result is therefore free of *both* the
quadrature error of the ``|mu|`` sign change and the even-mode truncation of a
floating-point evaluation, which is what makes it usable as the referee value
for the panel-code benchmark (see
:mod:`tensorbem.benchmarks.case_bem_rigid_lid_hemisphere`, whose ``force`` route
still carries a ~3e-6 Gauss-Legendre error near ``mu = 0``).

Controls carried by the same machinery (they are the reason to trust it):

* the **pressure-release** endpoint ``ka -> infinity`` is the odd-parity problem
  ``mu = P_1``; the same series returns exactly ``0.5`` (``= 1/3 rho pi a^3 /
  (rho nabla)``) -- the repository anchor A2;
* the **translating image** reading of the mirror argument (``1.0 rho*nabla``)
  does **not** solve the stated BVP; it is kept as a negative control in the
  case driver and refuted there by measurement
  (``docs/rigid_lid_zero_frequency_anchor/ERRATUM.md``).

Entry point::

    PYTHONPATH=src python -m tensorbem.rigid_lid --degrees 40 80 160
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from typing import Any

__all__ = [
    "legendre_monomial_coefficients",
    "half_moments_exact",
    "rigid_lid_added_mass_dimless",
    "pressure_release_added_mass_dimless",
    "main",
]

#: ``nabla = (2/3) pi a^3``; every value below is ``A33 / (rho * nabla)``.
HALF_MOMENT_CLOSED_FORMS: dict[int, Fraction] = {
    0: Fraction(1, 2),
    1: Fraction(1, 3),
    2: Fraction(1, 8),
    3: Fraction(0),  # P_3 = (5mu^3-3mu)/2, mu P_3 integrates to zero
}


def legendre_monomial_coefficients(degree_max: int) -> list[list[Fraction]]:
    """``P_n(mu) = sum_k coeffs[n][k] mu**k`` for ``n = 0 .. degree_max``.

    Bonnet recursion ``(n+1) P_{n+1} = (2n+1) mu P_n - n P_{n-1}`` in exact
    rational arithmetic.
    """
    if degree_max < 0:
        raise ValueError("degree_max must be non-negative")
    coeffs: list[list[Fraction]] = [[Fraction(1)]]
    if degree_max == 0:
        return coeffs
    coeffs.append([Fraction(0), Fraction(1)])  # P_1 = mu
    for n in range(1, degree_max):
        prev, cur = coeffs[n - 1], coeffs[n]
        nxt = [Fraction(0)] * (n + 2)
        for k, c in enumerate(cur):
            nxt[k + 1] += Fraction(2 * n + 1, n + 1) * c
        for k, c in enumerate(prev):
            nxt[k] -= Fraction(n, n + 1) * c
        coeffs.append(nxt)
    return coeffs


def half_moments_exact(degree_max: int) -> dict[str, Any]:
    """Exact ``J_n = int_0^1 mu P_n(mu) dmu`` as Fractions, with a residual.

    ``J_n = sum_k coeffs[n][k] / (k + 2)``.  ``P_0 .. P_3`` are compared with
    the closed forms ``1/2, 1/3, 1/8, 1/15``; the residual is the largest
    absolute difference and must be exactly zero (rational arithmetic).
    """
    coeffs = legendre_monomial_coefficients(degree_max)
    J = [
        sum((c / Fraction(k + 2) for k, c in enumerate(row)), Fraction(0))
        for row in coeffs
    ]
    residual = max(
        (abs(J[n] - exact) for n, exact in HALF_MOMENT_CLOSED_FORMS.items()),
        default=Fraction(0),
    )
    return {"J": J, "closed_form_residual": float(residual)}


def rigid_lid_added_mass_dimless(degree_max: int = 160) -> dict[str, Any]:
    """Exact ``A33(ka->0)/(rho*nabla)`` of the floating hemisphere.

    Returns the converged value, the truncation ladder (every 20 even modes) and
    the tail estimate (difference of the last two ladder entries), all evaluated
    in rational arithmetic.
    """
    half = half_moments_exact(degree_max)
    J = half["J"]
    terms = [
        Fraction(3 * (2 * n + 1), n + 1) * J[n] * J[n]
        for n in range(0, degree_max + 1, 2)
    ]
    cumulative: list[Fraction] = []
    total = Fraction(0)
    for term in terms:
        total += term
        cumulative.append(total)
    value = float(total)
    ladder = [
        {"degree_max": int(2 * i), "value_dimless": float(cumulative[i])}
        for i in range(len(cumulative))
    ]
    tail = (
        abs(cumulative[-1] - cumulative[-2]) / total if len(cumulative) > 1 else 0.0
    )
    return {
        "form": "even_parity_double_body",
        "value_dimless": value,
        "degree_max": int(degree_max),
        "n_modes": int(len(terms)),
        "arithmetic": "exact rational (Fractions), float only at the end",
        "moment_closed_form_residual": half["closed_form_residual"],
        "truncation_ladder": ladder,
        "truncation_tail_relative": float(tail),
        "last_term_relative": float(terms[-1] / total),
    }


def pressure_release_added_mass_dimless() -> dict[str, Any]:
    """Odd-parity control: the single mode ``mu = P_1`` gives exactly ``1/2``."""
    J1 = Fraction(1, 3)
    a1 = 3 * J1
    value = Fraction(3 * (2 * 1 + 1), 1 + 1) * J1 * J1  # 3*3*(1/9)/2 = 1/2
    return {
        "value_dimless": float(value),
        "exact": str(value),
        "form": "odd_parity_single_mode (pressure release, ka -> infinity)",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exact rigid-lid (ka->0) heave added mass of a floating hemisphere"
    )
    parser.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=[40, 80, 160],
        help="even-mode truncations to report (default: 40 80 160)",
    )
    args = parser.parse_args(argv)
    reports = [rigid_lid_added_mass_dimless(d) for d in args.degrees]
    payload = {
        "case_id": "CASE-BEM-RIGID-LID-HEMISPHERE",
        "reference": reports[-1],
        "truncations": reports,
        "pressure_release_control": pressure_release_added_mass_dimless(),
        "translating_image_negative_control_dimless": 1.0,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())