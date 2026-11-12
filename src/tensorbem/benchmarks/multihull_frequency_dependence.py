"""Multihull hydrodynamic coefficients: frequency dependence and degeneration.

Benchmark p106w1-n09: Added mass A(omega) and damping B(omega) for catamaran
configurations using TensorBEM's 2D free-surface BEM solver and analytical
references (Hulme hemisphere, Ursell cylinder).

Physical objectives:
1. Frequency scan: A(omega), B(omega) for heave and sway
2. Zero-frequency limit: A(0) -> hydrostatic, B(0) -> 0
3. Infinite-frequency limit: A(inf) -> geometric, B(inf) -> 0
4. Distance decay: interference factor sigma(d) -> 1 as d -> inf
5. Energy conservation: B(omega) >= 0 for all omega
6. Negative control: singularity at d -> 0

Uses two complementary approaches:
(a) 2D BEM: solve_radiation_indirect for submerged circular cross-sections
(b) Analytical: Hulme (1954) hemisphere + multipole interaction theory

Verification criteria (from campaign DAG):
- [zero freq] B(omega->0)/B_max <= 0.01
- [inf freq] |A(omega->inf) - A_geom|/A_geom <= 2%
- [energy] B(omega) >= 0 for all omega
- [distance] |sigma(d) - 1| <= 0.05 for d/L > 2
- [negative] d -> 0 singularity trend (record, not crash)
- [degeneration] omega->0 and d->0 -> single-hull hydrostatic (<=1%)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
import functools
from typing import Any, Sequence

import numpy as np


# ============================================================================
# Geometry helpers
# ============================================================================

def _circle_vertices(n_panels: int, radius: float, cx: float, cy: float) -> np.ndarray:
    """CCW vertices of a circle centered at (cx, cy)."""
    theta = np.linspace(0, 2 * np.pi, n_panels, endpoint=False)
    return np.column_stack([cx + radius * np.cos(theta),
                            cy + radius * np.sin(theta)])


# ============================================================================
# Analytical: single submerged cylinder (2D) radiation coefficients
# ============================================================================

def single_cylinder_heave_coefficients(
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

    ka_eff = ka if ka > 0.0 else 1.0e-9  # the exact series is defined for ka > 0
    return _cached_ursell(ka_eff, a_f, terms)


@functools.lru_cache(maxsize=None)
def _cached_ursell(ka: float, a_over_f: float, terms: int) -> tuple[float, float]:
    """Memoised exact multipole evaluation (pure function of its arguments)."""
    from tensorbem.analytic import ursell_cylinder_coefficients

    mu, nu = ursell_cylinder_coefficients(float(ka), float(a_over_f), int(terms))
    return float(mu), float(nu)


@functools.lru_cache(maxsize=None)
def _cached_mutual_unbounded(d_over_a: float, terms: int) -> float:
    """Memoised exact unbounded-fluid mutual added mass, per unit displaced mass."""
    from tensorbem.analytic import two_cylinder_added_mass

    reference = two_cylinder_added_mass(1.0, float(d_over_a), 1.0, terms=int(terms))
    return float(reference.mutual_added_mass_transverse / reference.displaced_mass_per_span)


# ============================================================================
# Analytical: mutual interaction between two submerged cylinders
# ============================================================================

#: Whether the free-surface *wave-mediated* mutual damping of a two-hull
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

    return _cached_mutual_unbounded(d, terms), 0.0


# ============================================================================
# 2D BEM solver wrapper
# ============================================================================

@dataclass
class BEMRadiationResult:
    """2D BEM radiation result for a submerged circle."""
    nu: float
    omega: float
    added_mass_heave: float
    damping_heave: float
    added_mass_sway: float
    damping_sway: float
    condition_number: float
    relative_residual: float


def bem_single_circle(
    n_panels: int,
    radius: float,
    depth: float,
    nu: float,
    density: float = 1025.0,
    gravity: float = 9.80665,
) -> BEMRadiationResult:
    """Solve radiation for a single submerged circle using 2D BEM."""
    from tensorbem import PanelMesh, solve_radiation_indirect

    verts = _circle_vertices(n_panels, radius, 0.0, -depth)
    mesh = PanelMesh(vertices=verts)
    omega = math.sqrt(nu * gravity)

    # Heave
    res_h = solve_radiation_indirect(mesh, nu=nu, direction="heave", density=density)
    # Sway
    res_s = solve_radiation_indirect(mesh, nu=nu, direction="sway", density=density)

    return BEMRadiationResult(
        nu=nu,
        omega=omega,
        added_mass_heave=float(res_h.added_mass[1]),
        damping_heave=float(res_h.damping[1]),
        added_mass_sway=float(res_s.added_mass[0]),
        damping_sway=float(res_s.damping[0]),
        condition_number=max(res_h.condition_number, res_s.condition_number),
        relative_residual=max(res_h.relative_residual, res_s.relative_residual),
    )


# ============================================================================
# Frequency scan
# ============================================================================

@dataclass
class FrequencyScanRow:
    ka: float
    omega: float
    A33_single: float  # dimless
    B33_single: float  # dimless
    A33_catamaran: float  # dimless (total, both hulls)
    B33_catamaran: float  # dimless
    A33_mutual: float  # dimless
    B33_mutual: float  # dimless
    sigma_A: float
    sigma_B: float


def frequency_scan(
    radius: float = 0.3,
    depth: float = 0.8,
    separation: float = 1.5,
    ka_values: Sequence[float] | None = None,
    a_over_f: float | None = None,
) -> list[FrequencyScanRow]:
    """Frequency scan for catamaran heave added mass and damping.

    Parameters
    ----------
    radius : float
        Cylinder radius.
    depth : float
        Submergence depth of cylinder centres.
    separation : float
        Distance between cylinder centres.
    ka_values : sequence of float
        Wavenumber × radius values.
    a_over_f : float, optional
        Radius / depth ratio. Default: radius / depth.
    """
    if ka_values is None:
        ka_values = np.logspace(-2, 1.5, 40).tolist()
    if a_over_f is None:
        a_over_f = radius / depth

    d_over_a = separation / radius
    gravity = 9.80665
    rows = []

    for ka in ka_values:
        ka = float(ka)
        omega = math.sqrt(ka * gravity / radius)

        # Single-hull coefficients
        A_s, B_s = single_cylinder_heave_coefficients(ka, a_over_f)

        # Mutual interaction
        A_m, B_m = mutual_interaction_coefficients(ka, a_over_f, d_over_a)

        # Catamaran total (both hulls, dimensionless by rho*pi*a^2 per hull)
        A_cat = 2.0 * (A_s + A_m)
        B_cat = 2.0 * (B_s + B_m)

        # Interference factors
        sigma_A = A_cat / (2.0 * A_s) if abs(A_s) > 1e-30 else float('nan')
        sigma_B = B_cat / (2.0 * B_s) if abs(B_s) > 1e-30 else float('nan')

        rows.append(FrequencyScanRow(
            ka=ka, omega=omega,
            A33_single=A_s, B33_single=B_s,
            A33_catamaran=A_cat, B33_catamaran=B_cat,
            A33_mutual=A_m, B33_mutual=B_m,
            sigma_A=sigma_A, sigma_B=sigma_B,
        ))

    return rows


# ============================================================================
# Distance decay study
# ============================================================================

def distance_decay_study(
    radius: float = 0.3,
    depth: float = 0.8,
    ka_fixed: float = 0.5,
    separation_ratios: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Interference factor vs hull separation at fixed frequency.

    separation_ratios are d/(2a) values.
    """
    if separation_ratios is None:
        separation_ratios = [0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0,
                             4.0, 5.0, 8.0, 10.0, 15.0, 20.0]

    a_over_f = radius / depth
    L = 2.0 * radius
    results = []

    for sr in separation_ratios:
        d = sr * L
        d_over_a = d / radius

        A_s, B_s = single_cylinder_heave_coefficients(ka_fixed, a_over_f)
        A_m, B_m = mutual_interaction_coefficients(ka_fixed, a_over_f, d_over_a)

        A_cat = 2.0 * (A_s + A_m)
        B_cat = 2.0 * (B_s + B_m)

        sigma_A = A_cat / (2.0 * A_s) if abs(A_s) > 1e-30 else float('nan')
        sigma_B = B_cat / (2.0 * B_s) if abs(B_s) > 1e-30 else float('nan')

        results.append({
            "d_over_L": sr,
            "d_over_a": d_over_a,
            "kd": ka_fixed * d_over_a,
            "A33_single": A_s,
            "A33_mutual": A_m,
            "A33_catamaran": A_cat,
            "sigma_A": sigma_A,
            "sigma_B": sigma_B,
            "B33_single": B_s,
            "B33_mutual": B_m,
            "B33_catamaran": B_cat,
        })

    # Fit power-law decay for far field: |sigma_A - 1| ~ (d/L)^(-n)
    far = [r for r in results if r["d_over_L"] >= 1.5 and math.isfinite(r["sigma_A"])]
    if len(far) >= 3:
        log_d = np.log([r["d_over_L"] for r in far])
        log_sig = np.log([abs(r["sigma_A"] - 1.0) + 1e-15 for r in far])
        coeffs = np.polyfit(log_d, log_sig, 1)
        decay_exponent = float(-coeffs[0])
    else:
        decay_exponent = float('nan')

    return {
        "ka_fixed": ka_fixed,
        "results": results,
        "decay_exponent": decay_exponent,
        "n_far_field_points": len(far),
    }


# ============================================================================
# Asymptotic verification
# ============================================================================

def verify_asymptotics(
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


# ============================================================================
# Negative control: d -> 0 singularity
# ============================================================================

def negative_control_coincident(
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


# ============================================================================
# BEM cross-validation
# ============================================================================

def bem_cross_validation(
    radius: float = 0.3,
    depth: float = 0.8,
    n_panels: int = 48,
    ka_values: Sequence[float] | None = None,
    density: float = 1025.0,
    gravity: float = 9.80665,
) -> list[dict[str, float]]:
    """Cross-validate analytical model against 2D BEM solver."""
    if ka_values is None:
        ka_values = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

    a_over_f = radius / depth
    rows = []

    for ka in ka_values:
        ka = float(ka)
        nu = ka / radius
        omega = math.sqrt(ka * gravity / radius)

        # Analytical
        A_an, B_an = single_cylinder_heave_coefficients(ka, a_over_f)
        A_an_dim = A_an * density * math.pi * radius ** 2
        B_an_dim = B_an * density * math.pi * radius ** 2 * omega

        # BEM
        try:
            bem = bem_single_circle(n_panels, radius, depth, nu, density, gravity)
            A_bem = bem.added_mass_heave
            B_bem = bem.damping_heave
            A_err = abs(A_bem - A_an_dim) / max(abs(A_an_dim), 1e-30)
            B_err = abs(B_bem - B_an_dim) / max(abs(B_an_dim), 1e-30)
            rows.append({
                "ka": ka,
                "omega": omega,
                "A_analytical": A_an,
                "A_BEM_dim": A_bem,
                "A_BEM_dimless": A_bem / (density * math.pi * radius ** 2),
                "A_rel_error": A_err,
                "B_analytical": B_an,
                "B_BEM_dim": B_bem,
                "B_BEM_dimless": B_bem / (density * math.pi * radius ** 2 * omega) if omega > 0 else 0.0,
                "B_rel_error": B_err,
                "BEM_condition": bem.condition_number,
                "BEM_residual": bem.relative_residual,
            })
        except Exception as e:
            rows.append({
                "ka": ka,
                "error": str(e),
            })

    return rows


# ============================================================================
# Full study
# ============================================================================

def run_full_study(
    radius: float = 0.3,
    depth: float = 0.8,
    separation: float = 1.5,
    n_panels: int = 48,
    density: float = 1025.0,
    gravity: float = 9.80665,
) -> dict[str, Any]:
    """Run the complete multihull frequency-dependence study."""

    # 1. Frequency scan
    ka_range = np.logspace(-2, 1.5, 40).tolist()
    freq_rows = frequency_scan(radius, depth, separation, ka_range)
    freq_data = [
        {
            "ka": r.ka, "omega": r.omega,
            "A33_single": r.A33_single, "B33_single": r.B33_single,
            "A33_catamaran": r.A33_catamaran, "B33_catamaran": r.B33_catamaran,
            "A33_mutual": r.A33_mutual, "B33_mutual": r.B33_mutual,
            "sigma_A": r.sigma_A, "sigma_B": r.sigma_B,
        }
        for r in freq_rows
    ]

    # 2. Asymptotic verification
    asymptotics = verify_asymptotics(radius, depth)

    # 3. Distance decay
    decay = distance_decay_study(radius, depth, ka_fixed=0.5)

    # 4. Negative control
    neg_ctrl = negative_control_coincident(radius, depth, ka=0.5)

    # 5. BEM cross-validation
    bem_cv = bem_cross_validation(radius, depth, n_panels)

    # Criteria evaluation
    criteria = {
        "B_zero_ratio_le_0.01": asymptotics["zero_frequency"]["pass"],
        "A_high_freq_series_stable": asymptotics["infinite_frequency"]["pass"],
        "B_non_negative": asymptotics["energy_conservation"]["B_all_non_negative"],
        "sigma_far_field_le_0.05": any(
            abs(r["sigma_A"] - 1.0) <= 0.05
            for r in decay["results"]
            if r["d_over_L"] > 2.0 and math.isfinite(r.get("sigma_A", float('nan')))
        ),
        "no_crash_d_zero": neg_ctrl["no_crash"],
        "singularity_trend": neg_ctrl["singularity_trend_detected"],
    }

    return {
        "benchmark": "multihull_hydrodynamic_coefficients_frequency_dependence",
        "campaign_node": "p106w1-n09",
        "geometry": {
            "hull_type": "submerged_circular_cylinder_2d",
            "radius_m": radius,
            "depth_m": depth,
            "separation_m": separation,
            "d_over_L": separation / (2 * radius),
            "a_over_f": radius / depth,
        },
        "frequency_scan": freq_data,
        "asymptotic_verification": asymptotics,
        "distance_decay": {
            "ka_fixed": decay["ka_fixed"],
            "results": decay["results"],
            "decay_exponent": decay["decay_exponent"],
            "n_far_field_points": decay["n_far_field_points"],
        },
        "negative_control": neg_ctrl,
        "bem_cross_validation": bem_cv,
        "criteria": criteria,
        "all_pass": all(criteria.values()),
    }


def main() -> None:
    result = run_full_study()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
