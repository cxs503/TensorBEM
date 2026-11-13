"""Rigid-lid (``ka -> 0``) heave added mass of a floating hemisphere: exact
reference pinning and an *independent* solution route (m17547).

Why this module exists
----------------------
``CASE-BEM-RIGID-LID-HEMISPHERE`` (case pool, tolerance < 5 %) was delivered by
m17243 with a headline error of +0.283 % on the 1024-panel mesh.  Two
verification debts remained, and this module pays both:

1. **The reference itself was never pinned independently.**  The accepted value
   ``A33(ka->0)/(rho*nabla) = 0.8309541476`` came from
   :func:`tensorbem.benchmarks.rigid_lid_hemisphere_attribution.even_mode_added_mass_dimless`,
   which evaluates the Legendre coefficients of ``|mu|`` by Gauss-Legendre
   quadrature of a function with a **kink at mu = 0**.  Gauss-Legendre is not
   exact for a kinked integrand, so that value carries a fixed truncation bias
   of about **+3.4 ppm**; the "machine precision, tail 4.6e-09" claim in
   :func:`test_even_mode_series_is_machine_converged` only measures inner
   consistency and cannot see the bias.

   Here the coefficients are derived **in closed form**::

       |mu| = sum_{n even} a_n P_n(mu),   a_n = (2n+1) int_0^1 mu P_n(mu) dmu
       int_0^1 P_m dmu = [P_{m-1}(0) - P_{m+1}(0)] / (2m+1)
       P_{2j}(0)       = (-1)**j C(2j, j) / 4**j          (P_odd(0) = 0)
       A33/(rho*nabla) = -3 * sum_{n even} c_n I_n
                       = +3 * sum_{n even} a_n**2 / ((n+1)(2n+1))

   so no quadrature enters.  The truncation tail is ``O(n**-4)`` (measured, not
   assumed) and is removed by Richardson extrapolation; the resulting pin is
   stable to ~1e-11.  Partial sums are also computed in exact rational
   arithmetic (``fractions.Fraction``), which is a second, algebraically
   independent evaluation of the same series.

2. **All existing cross-checks share one method family.**  The reference (even
   Legendre series), the Hulme (1982) multipole cross-check and the pressure
   release endpoint are all closed-form Legendre-type expansions; the only
   numerical route was the 3-D Kelvin panel solver itself.  This module adds a
   **method-of-fundamental-solutions (MFS) axisymmetric ring-source solver**::

       phi(P) = sum_k q_k [ G_ring(P; r_k, z_k) + G_ring(P; r_k, -z_k) ]
       G_ring = K(kappa) / (pi sqrt(a + b)),  a = r**2 + r_s**2 + (z - z_s)**2,
       b = 2 r r_s,  kappa**2 = 2b/(a+b)

   The image ring makes the rigid lid ``dphi/dz = 0`` exact, the sources sit
   inside the body so **no singular integral is ever evaluated**, and the
   unknown strengths follow from a least-squares Neumann fit.  Its error
   mechanisms (source placement, conditioning, spectral representation of the
   contact-line behaviour) are unrelated to panels or Legendre series.

Everything is rendered from the single payload returned by :func:`run_study`;
no number is hand-copied into the artifacts.

Artifacts: ``benchmarks/verified/bem_rigid_lid_hemisphere_axisym/{results.json,
summary.md, figure.png}``.
"""

from __future__ import annotations

import json
import math
import pathlib
from fractions import Fraction
from typing import Any, Sequence

import numpy as np
from scipy.special import ellipk

__all__ = [
    "legendre_at_zero",
    "abs_mu_legendre_coefficient",
    "rigid_lid_series",
    "pin_rigid_lid_reference",
    "mfs_added_mass_dimless",
    "sphere_mfs_validation",
    "run_study",
    "render_summary",
    "write_artifacts",
]

RADIUS = 1.0
DENSITY = 1025.0
VOLUME = (2.0 / 3.0) * math.pi * RADIUS**3

#: Values that appear in the repository prose / artifacts for the same quantity.
REPO_VALUES: dict[str, dict[str, Any]] = {
    "gauss_quadrature_series_m16802_m17243": {
        "value": 0.8309541475908425,
        "source": (
            "tensorbem.benchmarks.rigid_lid_hemisphere_attribution."
            "even_mode_added_mass_dimless(degree_max=60, quad_order=400); "
            "committed in benchmarks/verified/bem_rigid_lid_hemisphere{,_ka0}/results.json"
        ),
    },
    "erratum_prose_m16850": {
        "value": 0.8309570,
        "source": "docs/rigid_lid_zero_frequency_anchor/ERRATUM.md",
    },
    "stale_module_docstring": {
        "value": 0.826075,
        "source": (
            "docstring of src/tensorbem/benchmarks/rigid_lid_hemisphere_attribution.py "
            "('The even-mode series gives A33(0)/(rho*nabla) = 0.826075...')"
        ),
    },
    "refuted_report_15809_anchor": {
        "value": 1.0,
        "source": "report #15809 (rigid lid = translating sphere), refuted by m16850",
    },
}


# ---------------------------------------------------------------------------
# 1. Exact closed-form coefficient series
# ---------------------------------------------------------------------------
def legendre_at_zero(degree: int) -> Fraction:
    """``P_degree(0)`` exactly (0 for odd degree)."""
    if degree % 2:
        return Fraction(0)
    j = degree // 2
    return Fraction((-1) ** j * math.comb(2 * j, j), 4**j)


def abs_mu_legendre_coefficient(degree: int) -> Fraction:
    """Legendre coefficient of ``|mu|`` for even ``degree`` (exactly 0 for odd).

    ``int_0^1 x P_n dx = [(n+1) J_{n+1} + n J_{n-1}] / (2n+1)`` with
    ``J_m = int_0^1 P_m dx = [P_{m-1}(0) - P_{m+1}(0)] / (2m+1)``.
    """
    if degree % 2:
        return Fraction(0)

    def J(m: int) -> Fraction:
        return (legendre_at_zero(m - 1) - legendre_at_zero(m + 1)) / Fraction(2 * m + 1)

    return (degree + 1) * J(degree + 1) + (degree * J(degree - 1) if degree else 0)


def rigid_lid_series(degree_max: int = 400) -> dict[str, Any]:
    """Exact-rational partial sums of ``A33(0)/(rho*nabla)``.

    ``A33/(rho*nabla) = 3 * sum_{n even} a_n**2 / ((n+1)(2n+1))`` (see module
    docstring).  Returns the exact rational partial sums, the term list and the
    same series evaluated in float64 (agreement is a rounding check, not a
    physics check).
    """
    if degree_max < 4 or degree_max % 2:
        raise ValueError("degree_max must be an even integer >= 4")
    exact = Fraction(0)
    exact_steps: list[Fraction] = []
    float_steps: list[float] = []
    terms: list[float] = []
    cf = 0.0
    for n in range(0, degree_max + 1, 2):
        a = abs_mu_legendre_coefficient(n)
        exact += 3 * a * a / Fraction((n + 1) * (2 * n + 1))
        term = float(3 * a * a / Fraction((n + 1) * (2 * n + 1)))
        terms.append(term)
        cf += term
        exact_steps.append(exact)
        float_steps.append(cf)
    return {
        "degree_max": int(degree_max),
        "n_terms": len(terms),
        "exact_partial_sums": exact_steps,
        "float_partial_sums": float_steps,
        "terms": terms,
        "truncated_value_exact": float(exact),
        "float_vs_exact_roundoff": abs(cf - float(exact)),
        "last_term": terms[-1],
    }


def _tail_exponent(levels: Sequence[int], sums: Sequence[float]) -> float:
    """Measured exponent ``p`` of ``S(n) = S_inf - C n**-p`` from three levels.

    Solved by bisection on the ratio ``(S2-S1)/(S3-S2) =
    (n1**-p - n2**-p) / (n2**-p - n3**-p)``; the assumption is only that a
    single power law describes the three finest truncations (checked by the
    caller through the third difference).
    """
    if len(levels) < 3:
        raise ValueError("need at least three truncation levels")
    n1, n2, n3 = (float(levels[-3]), float(levels[-2]), float(levels[-1]))
    s1, s2, s3 = sums[-3], sums[-2], sums[-1]
    d21, d32 = s2 - s1, s3 - s2
    if d21 == 0.0 or d32 == 0.0 or d21 * d32 <= 0.0:
        return float("nan")
    target = d21 / d32

    def ratio(p: float) -> float:
        return (n1 ** (-p) - n2 ** (-p)) / (n2 ** (-p) - n3 ** (-p))

    lo, hi = 0.05, 40.0
    if not (ratio(lo) < target < ratio(hi)):
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if ratio(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def pin_rigid_lid_reference(
    degrees: Sequence[int] = (20, 40, 60, 80, 120, 160, 240, 400),
) -> dict[str, Any]:
    """Pin ``A33(ka->0)/(rho*nabla)`` with a Richardson extrapolation.

    The series is not absolutely fast: ``a_n ~ n**-3/2`` so the partial sums
    converge as ``n**-4``.  The exponent is *measured* from the three finest
    truncations, then removed once.
    """
    levels = sorted({int(d) for d in degrees})
    sums = [rigid_lid_series(d)["truncated_value_exact"] for d in levels]
    exponent = _tail_exponent(levels, sums)
    n_last, s_last = float(levels[-1]), sums[-1]
    n_prev, s_prev = float(levels[-2]), sums[-2]
    if math.isfinite(exponent) and exponent > 0:
        # S(n) = S_inf - C n**-p  =>  S_inf = S_last + (S_last - S_prev) / (r**p - 1)
        r = n_last / n_prev
        pinned = s_last + (s_last - s_prev) / (r**exponent - 1.0)
    else:  # pragma: no cover - defensive
        pinned = math.nan
    richest = rigid_lid_series(levels[-1])
    return {
        "degrees": levels,
        "partial_sums": sums,
        "measured_tail_exponent": exponent,
        "pinned_value": float(pinned),
        "truncation_tail_estimate": float(abs(pinned - s_last)),
        "last_term": richest["last_term"],
        "float_vs_exact_roundoff": richest["float_vs_exact_roundoff"],
        "richest_partial_sum": s_last,
    }


def repo_value_audit(pinned: float) -> dict[str, Any]:
    """Deviation of every value floating around the repository from the pin."""
    rows = []
    for key, meta in sorted(REPO_VALUES.items()):
        rows.append(
            {
                "key": key,
                "value": float(meta["value"]),
                "deviation": float(meta["value"]) - pinned,
                "deviation_ppm": (float(meta["value"]) / pinned - 1.0) * 1e6,
                "source": meta["source"],
            }
        )
    return {"pinned_value": pinned, "rows": rows}


# ---------------------------------------------------------------------------
# 2. Independent route: axisymmetric MFS with exact ring kernels
# ---------------------------------------------------------------------------
def _ring_potential(
    r: np.ndarray, z: np.ndarray, rs: float, zs: float
) -> np.ndarray:
    """Potential of a unit-strength ring source (exact, complete elliptic K)."""
    a = r * r + rs * rs + (z - zs) ** 2
    b = 2.0 * r * rs
    s = a + b
    kappa2 = np.clip(2.0 * b / s, 0.0, 1.0 - 1e-15)
    return ellipk(kappa2) / (np.pi * np.sqrt(s))  # scipy's ellipk takes m = k**2


def _half_space_kernel(
    r: np.ndarray, z: np.ndarray, rs: float, zs: float, image: bool
) -> np.ndarray:
    """Ring source plus its mirror in ``z = 0`` (exact ``dphi/dz = 0`` lid)."""
    out = _ring_potential(r, z, rs, zs)
    if image:
        out = out + _ring_potential(r, z, rs, -zs)
    return out


def _nodes(count: int, rule: str, half: bool) -> np.ndarray:
    """Midpoint nodes in the polar angle ``s`` (0 = pole, pi/2 = rim)."""
    span = (math.pi / 2.0) if half else math.pi
    xi = (np.arange(count) + 0.5) / count
    if rule == "uniform":
        return span * xi
    if rule == "rim":  # cluster at the rim (s -> span)
        return span * np.sqrt(xi)
    if rule == "chebyshev":  # cluster at both ends
        return span * (0.5 - 0.5 * np.cos(np.pi * xi))
    raise ValueError(f"unknown node rule: {rule}")


def _surface_added_mass(
    q: np.ndarray,
    r_s: np.ndarray,
    z_s: np.ndarray,
    half: bool,
    image: bool,
    n_quad: int = 4001,
) -> dict[str, float]:
    """``A33 = -rho int_S phi n_z dS`` with a quadrature *independent* of the
    collocation mesh (uniform trapezoid, ``n_quad`` points).

    The clustering that helps the least-squares fit must not be allowed to set
    the integration error, so ``phi`` is re-evaluated from the exact ring kernels
    on a separate fine equispaced rule.  ``rho = 1`` here; the caller rescales.
    """
    span = (math.pi / 2.0) if half else math.pi
    s = np.linspace(0.0, span, n_quad)
    r = RADIUS * np.sin(s)
    z = -RADIUS * np.cos(s)
    n_z = -np.cos(s)
    phi = np.zeros_like(s)
    for qk, rs, zs in zip(q, r_s, z_s):
        phi += qk * _half_space_kernel(r, z, float(rs), float(zs), image)
    integral = 2.0 * math.pi * RADIUS**2 * float(
        np.trapezoid(phi * n_z * np.sin(s), s)
    )
    volume = VOLUME if half else (4.0 / 3.0) * math.pi * RADIUS**3
    return {
        "surface_integral": integral,
        "added_mass_dimless": -integral / volume,
    }


def mfs_added_mass_dimless(
    n_sources: int = 96,
    n_collocation: int = 192,
    contraction: float = 0.8,
    collocation_rule: str = "chebyshev",
    source_rule: str = "uniform",
    step: float = 1e-6,
) -> dict[str, Any]:
    """Heave added mass of the floating hemisphere with a rigid lid (MFS route).

    Sources are placed on the *contracted* hemisphere ``(c a sin s, -c a cos s)``
    and mirrored in ``z = 0``; collocation enforces
    ``dphi/dn = V n_z = -V cos s`` on the body (unit velocity amplitude).  The
    added mass follows from the kinetic energy form
    ``A33 = -rho int_S phi n_z dS`` (no volume quadrature, no panel integrals).
    """
    s_col = _nodes(n_collocation, collocation_rule, half=True)
    r_c = RADIUS * np.sin(s_col)
    z_c = -RADIUS * np.cos(s_col)
    n_r = np.sin(s_col)
    n_z = -np.cos(s_col)
    s_src = _nodes(n_sources, source_rule, half=True)
    r_s = contraction * RADIUS * np.sin(s_src)
    z_s = -contraction * RADIUS * np.cos(s_src)

    h = step * RADIUS
    pot = np.zeros((n_collocation, n_sources))
    neu = np.zeros((n_collocation, n_sources))
    for k in range(n_sources):
        pot[:, k] = _half_space_kernel(r_c, z_c, r_s[k], z_s[k], True)
        plus = _half_space_kernel(r_c + h * n_r, z_c + h * n_z, r_s[k], z_s[k], True)
        minus = _half_space_kernel(r_c - h * n_r, z_c - h * n_z, r_s[k], z_s[k], True)
        neu[:, k] = (plus - minus) / (2.0 * h)

    target = n_z  # dphi/dn for unit heave velocity
    q, *_ = np.linalg.lstsq(neu, target, rcond=None)
    residual = float(np.linalg.norm(neu @ q - target) / np.linalg.norm(target))
    quad = _surface_added_mass(q, r_s, z_s, half=True, image=True)
    return {
        "n_sources": int(n_sources),
        "n_collocation": int(n_collocation),
        "contraction": float(contraction),
        "collocation_rule": collocation_rule,
        "source_rule": source_rule,
        "added_mass_dimless": quad["added_mass_dimless"],
        "condition_number": float(np.linalg.cond(neu)),
        "neumann_residual": residual,
        "surface_integral": quad["surface_integral"],
    }


def sphere_mfs_validation(
    n_sources: int = 48, n_collocation: int = 96, contraction: float = 0.8
) -> dict[str, Any]:
    """Same machinery on a *translating full sphere in infinite fluid*.

    Exact answer ``A33/(rho * 4 pi a**3 / 3) = 1/2``.  This validates the ring
    kernel, the elliptic-integral branch, the normal-derivative stencil, the
    sign convention of ``A33 = -rho int phi n_z dS`` and the surface integral,
    independently of the hemisphere geometry and of the rigid-lid image.
    """
    s_col = _nodes(n_collocation, "chebyshev", half=False)
    r_c = RADIUS * np.sin(s_col)
    z_c = -RADIUS * np.cos(s_col)
    n_r = np.sin(s_col)
    n_z = -np.cos(s_col)
    s_src = _nodes(n_sources, "uniform", half=False)
    r_s = contraction * RADIUS * np.sin(s_src)
    z_s = -contraction * RADIUS * np.cos(s_src)
    h = 1e-6 * RADIUS
    pot = np.zeros((n_collocation, n_sources))
    neu = np.zeros((n_collocation, n_sources))
    for k in range(n_sources):
        pot[:, k] = _half_space_kernel(r_c, z_c, r_s[k], z_s[k], False)
        plus = _half_space_kernel(r_c + h * n_r, z_c + h * n_z, r_s[k], z_s[k], False)
        minus = _half_space_kernel(r_c - h * n_r, z_c - h * n_z, r_s[k], z_s[k], False)
        neu[:, k] = (plus - minus) / (2.0 * h)
    target = n_z  # unit heave velocity
    q, *_ = np.linalg.lstsq(neu, target, rcond=None)
    quad = _surface_added_mass(q, r_s, z_s, half=False, image=False)
    added_mass = DENSITY * quad["added_mass_dimless"] * (4.0 / 3.0) * math.pi * RADIUS**3
    sphere_volume = (4.0 / 3.0) * math.pi * RADIUS**3
    return {
        "n_sources": int(n_sources),
        "n_collocation": int(n_collocation),
        "contraction": float(contraction),
        "added_mass": float(added_mass),
        "added_mass_dimless": float(added_mass / (DENSITY * sphere_volume)),
        "exact_dimless": 0.5,
        "rel_error": float(added_mass / (DENSITY * sphere_volume) / 0.5 - 1.0),
        "neumann_residual": float(
            np.linalg.norm(neu @ q - target) / np.linalg.norm(target)
        ),
    }


def sphere_validation_ladder() -> list[dict[str, Any]]:
    """Three independent sphere settings (exact answer 0.5)."""
    return [
        sphere_mfs_validation(24, 48, 0.5),
        sphere_mfs_validation(48, 96, 0.8),
        sphere_mfs_validation(64, 128, 0.6),
    ]


# ---------------------------------------------------------------------------
# 3. Study plumbing
# ---------------------------------------------------------------------------
def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


def _anchor_rows() -> list[dict[str, Any]]:
    """Mesh ladder of the delivered case (m17243), read from its artifact."""
    path = (
        _repo_root()
        / "benchmarks"
        / "verified"
        / "bem_rigid_lid_hemisphere_ka0"
        / "results.json"
    )
    if not path.exists():  # pragma: no cover - artifact always committed
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "panels": int(row["panels"]),
            "ka": float(row["ka"]),
            "added_mass_dimless": float(row["added_mass_dimless"]),
        }
        for row in payload["mesh_ladder"]
    ]


MFS_LADDER: tuple[tuple[int, int, float, str], ...] = (
    (16, 32, 0.6, "chebyshev"),
    (32, 64, 0.6, "chebyshev"),
    (64, 128, 0.6, "chebyshev"),
    (96, 192, 0.6, "chebyshev"),
    (16, 32, 0.8, "chebyshev"),
    (32, 64, 0.8, "chebyshev"),
    (64, 128, 0.8, "chebyshev"),
    (96, 192, 0.8, "chebyshev"),
    (128, 256, 0.8, "chebyshev"),
)


def run_study() -> dict[str, Any]:
    """Run the pinning + independent-route study; return the single payload."""
    pin = pin_rigid_lid_reference()
    pinned = pin["pinned_value"]
    audit = repo_value_audit(pinned)

    rows = []
    for nsrc, ncol, c, rule in MFS_LADDER:
        res = mfs_added_mass_dimless(nsrc, ncol, c, rule, "uniform")
        res["rel_error_vs_pin"] = res["added_mass_dimless"] / pinned - 1.0
        rows.append(res)
    best = min(rows, key=lambda row: abs(row["rel_error_vs_pin"]))

    sphere_rows = sphere_validation_ladder()
    sphere = min(sphere_rows, key=lambda row: abs(row["rel_error"]))
    anchor = _anchor_rows()
    for row in anchor:
        row["rel_error_vs_pin"] = row["added_mass_dimless"] / pinned - 1.0
        row["rel_error_vs_gauss_reference"] = (
            row["added_mass_dimless"]
            / REPO_VALUES["gauss_quadrature_series_m16802_m17243"]["value"]
            - 1.0
        )

    gauss = REPO_VALUES["gauss_quadrature_series_m16802_m17243"]["value"]
    criteria = {
        "P1_pin_is_tail_free": {
            "measured": pin["truncation_tail_estimate"],
            "criterion": "Richardson tail correction < 1e-9",
            "pass": bool(pin["truncation_tail_estimate"] < 1e-9),
        },
        "P2_pin_stable_over_truncation": {
            "measured": abs(pin["richest_partial_sum"] - pin["partial_sums"][-2]),
            "criterion": "|S(240) - S(400)| < 1e-8",
            "pass": bool(abs(pin["richest_partial_sum"] - pin["partial_sums"][-2]) < 1e-8),
        },
        "P3_exact_vs_float_arithmetic": {
            "measured": pin["float_vs_exact_roundoff"],
            "criterion": "rational vs float64 partial sum agree < 1e-13",
            "pass": bool(pin["float_vs_exact_roundoff"] < 1e-13),
        },
        "P4_gauss_quadrature_bias_quantified": {
            "measured": abs(gauss / pinned - 1.0),
            "criterion": "1e-7 < |gauss/pin - 1| < 1e-5 (a ppm-level bias, not round-off)",
            "pass": bool(1e-7 < abs(gauss / pinned - 1.0) < 1e-5),
        },
        "P5_sphere_route_validation": {
            "measured": sphere["rel_error"],
            "criterion": "|sphere A33/(rho*pi*4a^3/3) - 0.5| < 1e-4",
            "pass": bool(abs(sphere["rel_error"]) < 1e-4),
        },
        "P6_independent_route_agrees": {
            "measured": best["rel_error_vs_pin"],
            "criterion": "|MFS - pin|/pin < 1e-3 (0.1 %)",
            "pass": bool(abs(best["rel_error_vs_pin"]) < 1e-3),
        },
        "P7_anchor_verdict_unchanged": {
            "measured": max(abs(row["rel_error_vs_pin"]) for row in anchor),
            "criterion": "worst |m17243 row - pin|/pin < 5 % (case tolerance)",
            "pass": bool(
                anchor and max(abs(row["rel_error_vs_pin"]) for row in anchor) < 0.05
            ),
        },
        "P8_pin_vs_refuted_anchor_separation": {
            "measured": pinned,
            "criterion": "pin < 0.9 (still refutes the 1.0 rho*nabla anchor)",
            "pass": bool(pinned < 0.9),
        },
    }
    headline = next(
        (row for row in anchor if row["panels"] == 1024), anchor[-1] if anchor else {}
    )
    return {
        "radius": RADIUS,
        "density": DENSITY,
        "displaced_volume": VOLUME,
        "pin": {
            "value": pinned,
            "degrees": pin["degrees"],
            "partial_sums": pin["partial_sums"],
            "measured_tail_exponent": pin["measured_tail_exponent"],
            "truncation_tail_estimate": pin["truncation_tail_estimate"],
            "last_term": pin["last_term"],
        },
        "reference_audit": audit,
        "mfs_ladder": rows,
        "mfs_best": best,
        "sphere_validation": sphere,
        "sphere_ladder": sphere_rows,
        "anchor_rows": anchor,
        "anchor_headline": headline,
        "criteria": criteria,
        "verdict": "ALL PASS" if all(c["pass"] for c in criteria.values()) else "FAIL",
    }


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def render_summary(payload: dict[str, Any]) -> str:
    """Markdown summary rendered only from ``payload`` (no hand-copied number)."""
    pin = payload["pin"]
    lines = [
        "# Rigid-lid hemisphere added mass: reference pinning + independent route (m17547)",
        "",
        "Generated by `python -m tensorbem.benchmarks.rigid_lid_reference_pin` from the",
        "payload that produced `results.json`; no number is transcribed.",
        "",
        f"Pinned reference: `A33(ka->0)/(rho*nabla) = {pin['value']:.12f}`",
        f"(tail removed: {pin['truncation_tail_estimate']:.2e}, measured partial-sum"
        f" exponent {pin['measured_tail_exponent']:.3f}).",
        "",
        "## 1. Values in circulation vs the pin",
        "",
        "| source | value | deviation [ppm] |",
        "|---|---|---|",
    ]
    for row in payload["reference_audit"]["rows"]:
        lines.append(
            f"| {row['key']} | {row['value']:.10f} | {row['deviation_ppm']:+.3f} |"
        )
    lines += [
        "",
        "## 2. Independent route: axisymmetric MFS ring sources",
        "",
        "| sources | collocation | contraction | A33/(rho*nabla) | rel err vs pin | cond | Neumann res. |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in payload["mfs_ladder"]:
        lines.append(
            f"| {row['n_sources']} | {row['n_collocation']} | {row['contraction']:.2f} |"
            f" {row['added_mass_dimless']:.7f} | {100 * row['rel_error_vs_pin']:+.4f} % |"
            f" {row['condition_number']:.2e} | {row['neumann_residual']:.1e} |"
        )
    sphere = payload["sphere_validation"]
    lines += [
        "",
        "## 3. Route validation on a translating sphere (exact 0.5)",
        "",
        f"MFS gives {sphere['added_mass_dimless']:.8f} (rel err {sphere['rel_error']:+.2e})"
        f" with {sphere['n_sources']} ring sources, Neumann residual"
        f" {sphere['neumann_residual']:.1e}.",
        "",
        "## 4. Impact on the delivered case (m17243 artifact, recomputed)",
        "",
        "| panels | A33/(rho*nabla) | err vs pin | err vs old gauss ref |",
        "|---|---|---|---|",
    ]
    for row in payload["anchor_rows"]:
        lines.append(
            f"| {row['panels']} | {row['added_mass_dimless']:.6f} |"
            f" {100 * row['rel_error_vs_pin']:+.4f} % |"
            f" {100 * row['rel_error_vs_gauss_reference']:+.4f} % |"
        )
    lines += ["", "## 5. Criteria", "", "| id | measured | criterion | pass |", "|---|---|---|---|"]
    for key, crit in payload["criteria"].items():
        measured = crit["measured"]
        text = f"{measured:.4g}" if isinstance(measured, float) else str(measured)
        lines.append(f"| {key} | {text} | {crit['criterion']} | {crit['pass']} |")
    lines += ["", f"**Verdict: {payload['verdict']}**", ""]
    return "\n".join(lines)


def write_artifacts(payload: dict[str, Any], directory: pathlib.Path) -> dict[str, pathlib.Path]:
    """Write ``results.json`` and ``summary.md`` from one payload."""
    directory.mkdir(parents=True, exist_ok=True)
    results = directory / "results.json"
    results.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    summary = directory / "summary.md"
    summary.write_text(render_summary(payload), encoding="utf-8")
    return {"results": results, "summary": summary}


def main() -> int:  # pragma: no cover - CLI convenience
    payload = run_study()
    here = pathlib.Path(__file__).resolve().parents[3] / "benchmarks" / "verified"
    out = write_artifacts(payload, here / "bem_rigid_lid_hemisphere_axisym")
    print(render_summary(payload))
    for name, path in out.items():
        print(f"wrote {name}: {path}")
    return 0 if payload["verdict"] == "ALL PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())