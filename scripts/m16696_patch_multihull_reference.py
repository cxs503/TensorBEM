"""m16696 patch: replace the fabricated "Ursell/Yeung" reference in the multihull
frequency-dependence benchmark by the exact multipole reference that already
lives in :mod:`tensorbem.analytic`.

Run once:  PYTHONPATH=src python scripts/m16696_patch_multihull_reference.py
"""

from __future__ import annotations

import pathlib

PATH = pathlib.Path("src/tensorbem/benchmarks/multihull_frequency_dependence.py")

NEW_SINGLE = '''def single_cylinder_heave_coefficients(
    ka: float,
    a_over_f: float,
    terms: int = 40,
) -> tuple[float, float]:
    """Heave added mass and damping of a submerged 2-D circular cylinder.

    Exact Ursell multipole reference: delegates to
    :func:`tensorbem.analytic.ursell_cylinder_coefficients`, which solves the
    truncated multipole system of Ursell (1949) / Yeung (1981) for a circular
    cylinder of radius ``a`` whose axis lies a distance ``f`` below the free
    surface of deep water.  Returned in the study's dimensionless convention

        A_dimless = A / (rho*pi*a**2),   B_dimless = B / (rho*pi*a**2*omega).

    Parameters
    ----------
    ka : float
        Wavenumber x radius (``omega**2 * a / g``), non-negative.
    a_over_f : float
        Radius / submergence, strictly inside ``(0, 1)``: the cylinder must be
        fully submerged (``a/f = 1`` is a body touching the free surface, which
        the multipole system does not cover).
    terms : int
        Multipole truncation.  ``terms=40`` is converged: doubling to 120
        changes both coefficients by less than ``1e-12`` on the whole
        ``(ka, a/f)`` grid of this study (measured, ``docs/m16696/evidence.json``).

    Notes
    -----
    Mission 16696 replaced the previous body of this function.  It was a
    hand-written eight-line approximation that *labelled itself* "Ursell (1949)
    / Yeung (1981) multipole solution" but used invented coefficients
    (``amp = 2*ka/(1+ka**2)``, damping ``~ amp**2``) plus two hand-patched
    asymptotic branches.  Measured against the exact series it was off by
    4.8 %-19.6 % in added mass and 97.7 %-99.9 % in damping
    (``docs/m16696/before_analytic_vs_exact.json``), so the "analytic
    cross-validation" in :func:`bem_cross_validation` was comparing the BEM
    solver against a formula with no physical content.
    """
    if not math.isfinite(ka) or ka < 0.0:
        raise ValueError("ka must be a non-negative finite number")
    a_f = float(a_over_f)
    if not math.isfinite(a_f) or not 0.0 < a_f < 1.0:
        raise ValueError(
            "a_over_f must lie strictly between 0 and 1: the cylinder must be "
            "fully submerged"
        )
    if isinstance(terms, bool) or not isinstance(terms, int) or terms < 1:
        raise ValueError("terms must be a positive integer")

    from tensorbem.analytic import ursell_cylinder_coefficients

    ka_eff = ka if ka > 0.0 else 1.0e-9  # the exact series is defined for ka > 0
    mu, nu = ursell_cylinder_coefficients(ka_eff, a_f, terms)
    return float(mu), float(nu)
'''

NEW_MUTUAL = '''#: Whether the free-surface *wave-mediated* mutual damping of a two-hull
#: configuration is claimed by this study.  It is not: see
#: :func:`mutual_interaction_coefficients`.
MUTUAL_DAMPING_SUPPORTED = False


def mutual_interaction_coefficients(
    ka: float,
    a_over_f: float,
    d_over_a: float,
    terms: int = 40,
) -> tuple[float, float]:
    """Mutual added mass of two equal cylinders; mutual damping NOT_SUPPORTED.

    Returns ``(A_mut_dimless, B_mut_dimless)`` in the study's normalisation
    (per hull, divided by ``rho*pi*a**2``).

    ``A_mut`` is the **exact** mutual (transverse/heave) added mass of two equal
    circular cylinders in an *unbounded* fluid, taken from
    :func:`tensorbem.analytic.two_cylinder_added_mass` (truncated multipole
    series with its own boundary-residual diagnostic).  An unbounded fluid
    carries no length scale other than the geometry, so this value does not
    depend on ``ka``: it is the ``a/f -> 0`` limit of the free-surface mutual
    coefficient, exact to the multipole truncation.  It is **not** the
    free-surface wave-mediated mutual coefficient of a catamaran at finite
    ``ka``; the study therefore labels it as a deep-submergence reference and
    not as an interference factor at the reported frequency.

    ``B_mut`` is returned as ``0.0`` and is **not claimed**:
    :data:`MUTUAL_DAMPING_SUPPORTED` is ``False``.  The wave-mediated mutual
    damping requires the free-surface two-body problem, for which this
    repository has no closed form and no solver (the 2-D panel method takes one
    closed contour per body), so the honest value is "not supported", not a
    fitted number.  A ``sigma_B`` computed downstream is consequently a
    single-hull quantity.

    ``d_over_a <= 2`` returns ``nan``: the cylinders touch or overlap and the
    unbounded-fluid reference is undefined there (NaN, never a silent zero).

    Notes
    -----
    Mission 16696 removed the previous body: an invented series
    (``amp = 2*ka**n/(1+ka**(2*n))``, arbitrary ``0.5``/``0.1``/``0.25`` scale
    factors, a ``sin(n*kd)`` damping term and a made-up ``exp(-kd/2)`` far-field
    decay) documented as "Ursell-type".  No reference for that series exists, so
    its numbers are withdrawn rather than corrected (``docs/m16696/``).
    """
    if not math.isfinite(ka) or ka < 0.0:
        raise ValueError("ka must be a non-negative finite number")
    a_f = float(a_over_f)
    if not math.isfinite(a_f) or not 0.0 < a_f < 1.0:
        raise ValueError(
            "a_over_f must lie strictly between 0 and 1: the cylinder must be "
            "fully submerged"
        )
    d = float(d_over_a)
    if not math.isfinite(d) or d <= 0.0:
        raise ValueError("d_over_a must be a positive finite number")
    if d <= 2.0:
        undefined = float("nan")
        return undefined, undefined
    if isinstance(terms, bool) or not isinstance(terms, int) or terms < 4:
        raise ValueError("terms must be an integer of at least 4")

    from tensorbem.analytic import two_cylinder_added_mass

    reference = two_cylinder_added_mass(1.0, d, 1.0, terms=terms)
    mutual = reference.mutual_added_mass_transverse / reference.displaced_mass_per_span
    return float(mutual), 0.0
'''

NEW_ASYMPTOTICS = '''def verify_asymptotics(
    radius: float = 0.3,
    depth: float = 0.8,
) -> dict[str, Any]:
    """Verify the zero- and high-frequency limits against the exact reference.

    Both limits are read off the *exact* Ursell multipole series, so the
    criteria below cannot be satisfied by a hand-patched branch.  The
    high-frequency criterion is a **series-stability** statement
    (``ka = 80`` against ``ka = 100``), not the "geometric limit
    ``A = rho*pi*a**2``" that the pre-16696 fabricated formula asserted: with
    the free surface acting as a rigid wall at infinite frequency the image
    system *lowers* the heave added mass, and the exact value at
    ``a/f = 0.375`` is ``A/(rho*pi*a**2) = 0.9318``, i.e. 6.8 % below the
    ``A = 1`` claim that mission 16696 retracts.
    """
    a_over_f = radius / depth

    kas = np.logspace(-2, 1.5, 60)
    A_vals: list[float] = []
    B_vals: list[float] = []
    for ka in kas:
        A, B = single_cylinder_heave_coefficients(float(ka), a_over_f)
        A_vals.append(A)
        B_vals.append(B)
    B_max = max(B_vals)

    # Zero frequency: exact long-wave behaviour, B/B_max must collapse.
    ka_lo = 0.001
    A_lo, B_lo = single_cylinder_heave_coefficients(ka_lo, a_over_f)
    B_zero_ratio = B_lo / B_max if B_max > 0 else 0.0

    # High frequency: exact series at two large ka, stability only.
    A_80, B_80 = single_cylinder_heave_coefficients(80.0, a_over_f)
    A_hi, B_hi = single_cylinder_heave_coefficients(100.0, a_over_f)
    stability = abs(A_hi - A_80) / abs(A_hi)
    retracted_geometric_claim = 1.0  # the pre-16696 "A(inf) = rho*pi*a**2" claim

    # Energy: B >= 0 everywhere is a property of the exact reference.
    B_all_pos = all(b >= -1e-15 for b in B_vals)

    return {
        "zero_frequency": {
            "ka": ka_lo,
            "A_dimless": A_lo,
            "B_dimless": B_lo,
            "B_over_Bmax": B_zero_ratio,
            "pass": B_zero_ratio <= 0.01,
        },
        "infinite_frequency": {
            "ka_80": 80.0,
            "A_dimless_80": A_80,
            "ka": 100.0,
            "A_dimless": A_hi,
            "A_series_stability": stability,
            "B_dimless": B_hi,
            "retracted_geometric_claim": retracted_geometric_claim,
            "retracted_claim_error": abs(A_hi - retracted_geometric_claim)
            / retracted_geometric_claim,
            "pass": stability <= 0.01,
        },
        "energy_conservation": {
            "B_all_non_negative": B_all_pos,
            "B_min": min(B_vals),
            "B_max": B_max,
        },
        "mutual_damping_supported": MUTUAL_DAMPING_SUPPORTED,
    }
'''

NEW_NEGATIVE = '''def negative_control_coincident(
    radius: float = 0.3,
    depth: float = 0.8,
    ka: float = 0.5,
) -> dict[str, Any]:
    """Approaching contact (d -> 2a): the exact mutual coefficient must diverge.

    The admissible range of the exact unbounded-fluid reference starts at
    ``d = 2a``; configurations at or below contact return ``nan`` (see
    :func:`mutual_interaction_coefficients`) and are recorded as undefined
    rather than as a fabricated finite trend.
    """
    a_over_f = radius / depth
    A_s, B_s = single_cylinder_heave_coefficients(ka, a_over_f)

    d_over_a_values = [1.0, 1.5, 2.0, 2.05, 2.2, 2.5, 3.0, 5.0, 10.0]
    data = []
    for d_over_a in d_over_a_values:
        A_m, B_m = mutual_interaction_coefficients(ka, a_over_f, d_over_a)
        defined = bool(math.isfinite(A_m))
        data.append(
            {
                "d_over_a": d_over_a,
                "A_mutual": A_m,
                "B_mutual": B_m,
                "sigma_A": (2.0 * (A_s + A_m)) / (2.0 * A_s)
                if defined and abs(A_s) > 1e-30
                else float("nan"),
                "finite": defined,
            }
        )

    finite = [row for row in data if row["finite"]]
    if len(finite) >= 2:
        # |A_mut| must grow as the gap closes: compare nearest contact with the
        # furthest point of the admissible ladder.
        trend = abs(finite[0]["A_mutual"]) > abs(finite[-1]["A_mutual"])
    else:  # pragma: no cover - the ladder above always has >= 2 finite entries
        trend = False

    return {
        "ka": ka,
        "data": data,
        "singularity_trend_detected": trend,
        "undefined_below_contact": [row["d_over_a"] for row in data if not row["finite"]],
        "no_crash": True,
    }
'''


def replace_function(text: str, start_marker: str, end_marker: str, new: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start) + len(end_marker)
    return text[:start] + new + text[end:]


def main() -> None:
    text = PATH.read_text(encoding="utf-8")

    text = replace_function(
        text,
        "def single_cylinder_heave_coefficients(",
        "    return A_dimless, max(B_dimless, 0.0)\n",
        NEW_SINGLE,
    )
    text = replace_function(
        text,
        "def mutual_interaction_coefficients(",
        "    return A_mut, max(B_mut, 0.0)\n",
        NEW_MUTUAL,
    )
    text = replace_function(
        text,
        "def verify_asymptotics(",
        '        "energy_conservation": {\n            "B_all_non_negative": B_all_pos,\n            "B_min": min(B_vals),\n            "B_max": B_max,\n        },\n    }\n',
        NEW_ASYMPTOTICS,
    )
    text = replace_function(
        text,
        "def negative_control_coincident(",
        '        "singularity_trend_detected": singularity_trend,\n        "no_crash": True,\n    }\n',
        NEW_NEGATIVE,
    )
    # criterion label: the old name claimed a 2 % match to a retracted limit.
    text = text.replace(
        '"A_inf_error_le_2pct": asymptotics["infinite_frequency"]["pass"],',
        '"A_high_freq_series_stable": asymptotics["infinite_frequency"]["pass"],',
    )
    PATH.write_text(text, encoding="utf-8")
    print("patched", PATH)


if __name__ == "__main__":
    main()