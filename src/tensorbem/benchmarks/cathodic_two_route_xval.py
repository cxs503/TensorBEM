"""Independent cross-validation of the two-route current recovery (CP sphere/spheroid).

Ledger anchor
-------------
``XVAL-BEM-TWO-ROUTE-GAUSS-LAW-CURRENT`` -- *TensorBEM 的 two-route-gauss-law-current 做法*,
acceptance ``< 15 %`` (owner accuracy standard).

What is being validated
-----------------------
:func:`tensorbem.cathodic_protection.verify_conducting_sphere_routes` recovers the
total current of a uniformly polarised conducting body along two routes:

1. **surface route** -- sum the BEM single-layer density over the panels,
   ``I_surf = sigma * sum_i q_i A_i`` (anode-panel integral of ``j_n``);
2. **far-field (Gauss-law) route** -- integrate the potential over enclosing
   spheres, ``I(R) = 4 pi sigma R * <phi>_{S_R}``, then extrapolate
   ``I(R) = I_inf + c / R`` as ``R -> infinity``.

The landed verification reported ``|I_surf - I_far| / I ~ 1e-8`` and read that as
two independent confirmations of the physics.  This module quantifies what the
agreement actually covers, and adds the missing independent datum:

* **Independence audit.**  The far-field route consumes the *same* boundary
  density ``q`` (the exterior representation formula is an operator acting on
  ``q``), so it is a consistency check of the representation formula and of the
  Gauss flux bookkeeping -- not an independent discretisation of the PDE.  This
  is measured here: the far-field error tracks the surface error to ~1e-9, and
  is *subtracted* rather than *cancelled* (``far_err/surf_err = 1``).
* **Closed-form reference for a non-spherical body.**  On a sphere the exact
  exterior field is a pure monopole, i.e. the easiest possible case for the
  Gauss-law route (all higher multipoles vanish, so the ``1/R`` extrapolation is
  never exercised).  A prolate spheroid has a closed-form solution that *does*
  contain multipoles::

      I = 4 pi sigma |phi_a| a e / atanh(e),      e = sqrt(1 - b^2/a^2),

  which is the spheroid spreading conductance (``R = atanh(e)/(4 pi sigma a e)``;
  ``e -> 0`` recovers the sphere ``I = 4 pi sigma a phi_a``).  The local current
  density on the surface is also closed-form::

      j_n(eta) = sigma |phi_a| sqrt(xi0^2 - 1)
                 / (atanh(e) (xi0^2 - 1) f sqrt(xi0^2 - eta^2)),

  with ``f = a e``, ``xi0 = 1/e``, ``eta = x / a`` (prolate spheroidal
  coordinates on the surface ``xi = xi0``).  Both routes are compared against it,
  in total and pointwise.
* **Third route: local flux.**  ``I_flux = -sigma * int_{S_R} dphi/dn dS`` -- the
  Gauss law evaluated with the *local* current density (finite difference of the
  exterior representation), not with the potential mean.  It tests that ``q``
  reproduces the surface flux, which is the quantity the physics is about.
* **Directional-quadrature audit.**  The far-field mean uses ``n_directions``
  Fibonacci directions.  The same exterior field is re-integrated with an exact
  spherical rule (Gauss-Legendre in ``cos theta`` x uniform ``phi``) so that the
  sampling error is isolated from the ``q`` error.

Run
---
``python -m tensorbem.benchmarks.cathodic_two_route_xval --output outputs/xval``
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from ..cathodic_protection import (
    SEAWATER_CONDUCTIVITY,
    evaluate_exterior_potential,
    far_field_current,
    fibonacci_sphere_directions,
    solve_cathodic_protection,
)
from ..geometry3d import TriangleMesh, icosphere

_trapezoid = getattr(np, "trapezoid", None) or np.trapz

FloatArray = NDArray[np.float64]
_FOUR_PI = 4.0 * math.pi

#: Aspect ratio of the spheroidal anchor (semi-axes ``a : b : b = 2 : 1 : 1``).
SPHEROID_ASPECT = 2.0


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def prolate_spheroid_mesh(
    a: float = 2.0, b: float = 1.0, subdivisions: int = 2
) -> TriangleMesh:
    """Geodesic mesh of the prolate spheroid ``x^2/a^2 + (y^2+z^2)/b^2 = 1``.

    The icosphere is mapped by the diagonal affine map ``diag(a, b, b)``; the
    map is exact on the vertices, so the panel topology and the ``20 * 4**n``
    panel count are inherited unchanged from :func:`tensorbem.geometry3d.icosphere`.
    Because the map is affine, the relative facet deviation from the true
    surface is the same as for the unit sphere -- which is why the total-current
    error of both bodies follows the same ``O(h^2)`` curve.
    """
    a = float(a)
    b = float(b)
    if not (math.isfinite(a) and math.isfinite(b)) or a <= 0.0 or b <= 0.0:
        raise ValueError("semi-axes a, b must be positive finite numbers")
    if a < b:
        raise ValueError("require a >= b (prolate orientation along x)")
    base = icosphere(int(subdivisions), radius=1.0)
    return TriangleMesh(
        vertices=base.vertices * np.array([a, b, b], dtype=float), faces=base.faces
    )


def _spheroid_constants(a: float, b: float) -> tuple[float, float, float]:
    """Return ``(e, f, xi0)`` for the prolate spheroid (eccentricity, focal half-length, surface xi)."""
    e = math.sqrt(max(0.0, 1.0 - (b * b) / (a * a)))
    if e <= 0.0:
        raise ValueError("degenerate spheroid (a == b); use the sphere anchor")
    f = a * e
    return e, f, a / f


def prolate_spheroid_current_exact(
    a: float = 2.0,
    b: float = 1.0,
    conductivity: float = SEAWATER_CONDUCTIVITY,
    potential: float = -1.0,
) -> float:
    """Closed-form total current of a conducting prolate spheroid.

    ``I = 4 pi sigma |phi_a| a e / atanh(e)`` with ``e = sqrt(1 - b^2/a^2)``
    (the ``e -> 0`` limit is the sphere ``4 pi sigma a |phi_a|``).
    """
    e, _, _ = _spheroid_constants(a, b)
    return _FOUR_PI * float(conductivity) * abs(float(potential)) * float(a) * e / math.atanh(e)


def prolate_spheroid_surface_current_density(
    a: float = 2.0,
    b: float = 1.0,
    conductivity: float = SEAWATER_CONDUCTIVITY,
    potential: float = -1.0,
    points: FloatArray | None = None,
) -> FloatArray:
    """Closed-form ``|j_n|`` (A/m^2) on the spheroid surface at ``points``.

    ``points`` are expected to lie on (or be projected onto) the surface; the
    only coordinate entering the formula is ``eta = x / a``.  Points are
    projected radially onto the spheroid first, so panel centroids of the
    faceted mesh can be passed directly.
    """
    a = float(a)
    b = float(b)
    if points is None:
        raise ValueError("points must be provided")
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (M, 3)")
    e, f, xi0 = _spheroid_constants(a, b)
    scale = np.sqrt((pts[:, 0] / a) ** 2 + (pts[:, 1] ** 2 + pts[:, 2] ** 2) / (b * b))
    eta = np.clip(pts[:, 0] / (a * scale), -1.0, 1.0)
    norm = math.atanh(e)  # = arccoth(xi0) = the xi-profile normalisation
    denom = norm * (xi0 * xi0 - 1.0) * f * np.sqrt(xi0 * xi0 - eta * eta)
    return float(conductivity) * abs(float(potential)) * math.sqrt(xi0 * xi0 - 1.0) / denom


def spheroid_surface_integral(
    a: float = 2.0,
    b: float = 1.0,
    conductivity: float = SEAWATER_CONDUCTIVITY,
    potential: float = -1.0,
    n_eta: int = 2001,
) -> float:
    """Quadrature of the closed-form ``j_n(eta)`` over the spheroid surface.

    Used as a self-consistency check of the two closed forms (the result must
    reproduce :func:`prolate_spheroid_current_exact`).  The area element on
    ``xi = xi0`` is ``dA = 2 pi f^2 sqrt((xi0^2-1)(xi0^2-eta^2)) deta``.
    """
    e, f, xi0 = _spheroid_constants(a, b)
    eta = (np.arange(int(n_eta), dtype=float) + 0.5) * (2.0 / int(n_eta)) - 1.0
    # surface ring at each eta: x = a eta, polar radius rho = b sqrt(1 - eta^2)
    rho = float(b) * np.sqrt(np.maximum(0.0, 1.0 - eta * eta))
    pts = np.column_stack((float(a) * eta, rho, np.zeros_like(eta)))
    j = prolate_spheroid_surface_current_density(
        a, b, conductivity, potential, points=pts
    )
    dA = _FOUR_PI / 2.0 * f * f * np.sqrt((xi0 * xi0 - 1.0) * (xi0 * xi0 - eta * eta))
    # trapezoid in eta; the integrand vanishes at eta = +-1 like sqrt(1-eta)
    return float(_trapezoid(j * dA, eta))


# ---------------------------------------------------------------------------
# Enclosing-surface quadratures
# ---------------------------------------------------------------------------
def sphere_gauss_quadrature(
    radius: float, n_theta: int = 16, n_phi: int = 32
) -> tuple[FloatArray, FloatArray]:
    """Exact spherical rule: Gauss-Legendre in ``cos(theta)`` x uniform ``phi``.

    Returns quadrature points ``(M, 3)`` and area weights ``(M,)`` summing to
    ``4 pi R^2``.  Unlike the Fibonacci mean this rule integrates *exactly* the
    constant and linear components of any field, which is what isolates the
    directional-sampling error of :func:`far_field_current`.
    """
    nodes, weights = leggauss(int(n_theta))
    phi = 2.0 * math.pi * (np.arange(int(n_phi)) + 0.5) / int(n_phi)
    dphi = 2.0 * math.pi / int(n_phi)
    sin_theta = np.sqrt(1.0 - nodes * nodes)
    cos_theta = nodes
    z = cos_theta[:, None] * np.ones_like(phi)[None, :]
    rho = sin_theta[:, None] * np.ones_like(phi)[None, :]
    x = rho * np.cos(phi)[None, :]
    y = rho * np.sin(phi)[None, :]
    pts = float(radius) * np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    area = (float(radius) ** 2) * np.outer(weights, np.full_like(phi, dphi)).ravel()
    return pts, area


def gauss_law_mean_current(
    mesh: TriangleMesh,
    result: Any,
    radius: float,
    conductivity: float = SEAWATER_CONDUCTIVITY,
    n_theta: int = 16,
    n_phi: int = 32,
    quadrature_order: int = 6,
) -> dict[str, float]:
    """Potential-mean (Gauss-law) current on one sphere, exact spherical rule.

    ``I = sigma * int_{S_R} (-dphi/dn) dS``; for an exterior harmonic field the
    mean of ``phi`` over a sphere is exactly ``I / (4 pi sigma R)``, so this is
    the same estimator as :func:`far_field_current` with the directional mean
    replaced by an exact spherical quadrature.
    """
    pts, area = sphere_gauss_quadrature(radius, n_theta=n_theta, n_phi=n_phi)
    phi = evaluate_exterior_potential(mesh, result, pts, quadrature_order=quadrature_order)
    mean = float(np.sum(phi * area) / np.sum(area))
    return {
        "radius_m": float(radius),
        "mean_potential_V": mean,
        "current_A": _FOUR_PI * float(conductivity) * float(radius) * mean,
        "directional_spread": float(np.max(np.abs(phi - mean)) / abs(mean)) if mean else float("nan"),
    }


def gauss_law_flux_current(
    mesh: TriangleMesh,
    result: Any,
    radius: float,
    conductivity: float = SEAWATER_CONDUCTIVITY,
    n_theta: int = 16,
    n_phi: int = 32,
    quadrature_order: int = 6,
    fd_step_rel: float = 1.0e-3,
) -> dict[str, float]:
    """Third route: the *local* flux ``-sigma * int dphi/dn dS`` on a sphere.

    The normal derivative is a central finite difference of the exterior
    representation formula along the radius; the integrand is ``j_n`` itself, so
    this route does not rely on the "mean of phi over a sphere" identity.  It
    therefore probes whether the recovered density reproduces the surface flux
    (the physical current), not merely its monopole moment.
    """
    pts, area = sphere_gauss_quadrature(radius, n_theta=n_theta, n_phi=n_phi)
    h = fd_step_rel * float(radius)
    r_hat = pts / float(radius)
    phi_out = evaluate_exterior_potential(
        mesh, result, pts + h * r_hat, quadrature_order=quadrature_order
    )
    phi_in = evaluate_exterior_potential(
        mesh, result, pts - h * r_hat, quadrature_order=quadrature_order
    )
    dphi_dn = (phi_out - phi_in) / (2.0 * h)
    # j_n = -sigma dphi/dn (outward normal of the enclosing sphere = +r_hat)
    total = -float(conductivity) * float(np.sum(dphi_dn * area))
    return {
        "radius_m": float(radius),
        "fd_step_m": float(h),
        "current_A": total,
        "mean_normal_current_density_A_m2": float(np.mean(dphi_dn * -float(conductivity))),
    }


# ---------------------------------------------------------------------------
# Study drivers
# ---------------------------------------------------------------------------
def _solve_uniform_dirichlet(
    mesh: TriangleMesh,
    potential: float,
    conductivity: float,
    quadrature_order: int,
) -> Any:
    """Solve with every panel Dirichlet at ``potential`` (a uniformly polarised body)."""
    n = mesh.panel_count
    empty = np.array([], dtype=np.int64)
    every = np.arange(n, dtype=np.int64)
    return solve_cathodic_protection(
        mesh,
        empty,
        empty,
        every,
        protection_potential=float(potential),
        anode_potential=float(potential),
        conductivity=float(conductivity),
        quadrature_order=int(quadrature_order),
    )


def _relerr(value: float, reference: float) -> float:
    """Relative error of ``value`` against ``reference`` on magnitudes.

    The sign of the total current depends on the sign of the prescribed
    potential (``j_n`` points into the metal for a cathodic potential), so the
    comparison is made on magnitudes throughout this module.
    """
    reference = abs(float(reference))
    if reference == 0.0:
        return float("nan")
    return abs(abs(float(value)) - reference) / reference


def surface_route_current(result: Any) -> float:
    """Total outflow current from the surface route (anode-panel integral of ``j_n``)."""
    return float(result.anode_current_total)


def pointwise_distribution_error(
    mesh: TriangleMesh,
    result: Any,
    reference: FloatArray,
) -> dict[str, float]:
    """Relative L2 / L-inf of the numeric ``|j_n|`` against a panel-wise reference."""
    numeric = np.abs(np.asarray(result.current_density, dtype=float))
    ref = np.asarray(reference, dtype=float)
    denom = math.sqrt(float(np.mean(ref * ref)))
    return {
        "l2_rel": math.sqrt(float(np.mean((numeric - ref) ** 2))) / denom,
        "linf_rel": float(np.max(np.abs(numeric - ref))) / float(np.max(ref)),
        "mean_rel_bias": float(np.sum(numeric - ref)) / float(np.sum(ref)),
    }


def body_route_study(
    label: str,
    mesh: TriangleMesh,
    exact_current: float,
    radii: Sequence[float],
    conductivity: float = SEAWATER_CONDUCTIVITY,
    potential: float = -1.0,
    quadrature_order: int = 6,
    n_directions: int = 24,
    n_theta: int = 16,
    n_phi: int = 32,
    reference_density: FloatArray | None = None,
) -> dict[str, Any]:
    """Run all four routes on one mesh and report errors against ``exact_current``."""
    result = _solve_uniform_dirichlet(mesh, potential, conductivity, quadrature_order)
    surface = surface_route_current(result)
    far = far_field_current(
        mesh,
        result,
        conductivity=conductivity,
        radii=tuple(radii),
        n_directions=n_directions,
        quadrature_order=quadrature_order,
    )
    fluxes = [
        gauss_law_flux_current(
            mesh,
            result,
            radius=float(r),
            conductivity=conductivity,
            n_theta=n_theta,
            n_phi=n_phi,
            quadrature_order=quadrature_order,
        )
        for r in radii
    ]
    exact_sphere_rule = [
        gauss_law_mean_current(
            mesh,
            result,
            radius=float(r),
            conductivity=conductivity,
            n_theta=n_theta,
            n_phi=n_phi,
            quadrature_order=quadrature_order,
        )
        for r in radii
    ]
    out: dict[str, Any] = {
        "label": label,
        "n_panels": int(mesh.panel_count),
        "exact_current_A": float(exact_current),
        "surface_route": {
            "current_A": surface,
            "rel_error": _relerr(surface, exact_current),
        },
        "far_field_route": {
            "extrapolated_current_A": far["extrapolated_current_A"],
            "rel_error": _relerr(far["extrapolated_current_A"], exact_current),
            "current_by_radius_A": far["current_by_radius_A"],
            "radii_m": far["radii_m"],
            "n_directions": int(n_directions),
            "extrapolation_max_rel_residual": far["extrapolation_max_rel_residual"],
            "directional_spread_by_radius": far["monopole_max_rel_error_by_radius"],
            "surface_vs_far_field_rel_error": far["far_field_vs_surface_rel_error"],
        },
        "gauss_law_exact_sphere_rule": {
            "radii_m": [float(r) for r in radii],
            "current_A": [float(v["current_A"]) for v in exact_sphere_rule],
            "directional_spread_by_radius": [
                float(v["directional_spread"]) for v in exact_sphere_rule
            ],
        },
        "flux_route": {
            "radii_m": [float(r) for r in radii],
            "current_A": [float(v["current_A"]) for v in fluxes],
            "max_rel_error": float(
                max(_relerr(v["current_A"], exact_current) for v in fluxes)
            ),
        },
    }
    # Sampling audit: Fibonacci directional mean vs exact spherical rule, same field.
    fib_dirs = fibonacci_sphere_directions(n_directions)
    audit = []
    for r, exact_rule in zip(radii, exact_sphere_rule):
        pts = float(r) * fib_dirs
        phi = evaluate_exterior_potential(mesh, result, pts, quadrature_order=quadrature_order)
        fib_current = _FOUR_PI * conductivity * float(r) * float(np.mean(phi))
        audit.append(
            {
                "radius_m": float(r),
                "fibonacci_current_A": fib_current,
                "exact_sphere_rule_current_A": float(exact_rule["current_A"]),
                "fibonacci_vs_exact_rule_rel": _relerr(fib_current, exact_current),
                "exact_rule_rel_error": _relerr(exact_rule["current_A"], exact_current),
                "sampling_rel_difference": abs(fib_current - exact_rule["current_A"])
                / abs(exact_current),
            }
        )
    out["directional_sampling_audit"] = audit
    if reference_density is not None:
        out["pointwise_distribution"] = pointwise_distribution_error(
            mesh, result, reference_density
        )
    out["condition_number"] = float(result.condition_number)
    out["relative_residual"] = float(result.relative_residual)
    return out


def independence_audit(case: dict[str, Any]) -> dict[str, Any]:
    """Quantify how much of the far-field error is inherited from the same density."""
    surf = case["surface_route"]["rel_error"]
    far = case["far_field_route"]["rel_error"]
    ratio = far / surf if surf > 0.0 else float("nan")
    return {
        "surface_rel_error": surf,
        "far_field_rel_error": far,
        "far_over_surface_error_ratio": float(ratio),
        "surface_vs_far_field_rel_error": case["far_field_route"][
            "surface_vs_far_field_rel_error"
        ],
        "verdict": (
            "far-field and surface routes share the same boundary density q; the "
            "sub-1e-8 agreement measures the exterior representation formula and the "
            "Gauss flux bookkeeping, not an independent discretisation. Accepted as an "
            "implementation consistency check only; physics accuracy is carried by the "
            "closed-form comparison."
            if abs(ratio - 1.0) < 1.0e-3
            else "far-field error differs from the surface error; the route adds information"
        ),
    }


def _sphere_case(
    sub: int,
    radii: Sequence[float],
    conductivity: float,
    potential: float,
    quadrature_order: int,
    n_directions: int,
) -> dict[str, Any]:
    radius = 1.0
    mesh = icosphere(int(sub), radius=radius)
    exact = _FOUR_PI * conductivity * radius * abs(potential)
    case = body_route_study(
        f"sphere_sub{int(sub)}",
        mesh,
        exact,
        radii,
        conductivity=conductivity,
        potential=potential,
        quadrature_order=quadrature_order,
        n_directions=n_directions,
    )
    case["independence_audit"] = independence_audit(case)
    return case


def _spheroid_case(
    sub: int,
    radii: Sequence[float],
    aspect: float,
    conductivity: float,
    potential: float,
    quadrature_order: int,
    n_directions: int,
) -> dict[str, Any]:
    a = float(aspect)
    b = 1.0
    mesh = prolate_spheroid_mesh(a, b, int(sub))
    exact = prolate_spheroid_current_exact(a, b, conductivity=conductivity, potential=potential)
    centroids = mesh.vertices[mesh.faces].mean(axis=1)
    ref_density = prolate_spheroid_surface_current_density(
        a, b, conductivity=conductivity, potential=potential, points=centroids
    )
    case = body_route_study(
        f"spheroid{aspect:g}x1_sub{int(sub)}",
        mesh,
        exact,
        radii,
        conductivity=conductivity,
        potential=potential,
        quadrature_order=quadrature_order,
        n_directions=n_directions,
        reference_density=ref_density,
    )
    case["spheroid_semi_axes_m"] = [a, b, b]
    case["exact_spreading_resistance_ohm"] = abs(potential) / exact
    case["independence_audit"] = independence_audit(case)
    case["closed_form_self_consistency"] = {
        "surface_integral_of_local_jn_A": spheroid_surface_integral(
            a, b, conductivity=conductivity, potential=potential
        ),
        "exact_total_A": exact,
    }
    return case


def run_xval(
    subdivisions: Sequence[int] = (1, 2, 3),
    bodies: Sequence[str] = ("sphere", "spheroid"),
    radii: Sequence[float] = (3.0, 5.0, 8.0, 12.0, 20.0),
    conductivity: float = SEAWATER_CONDUCTIVITY,
    potential: float = -1.0,
    quadrature_order: int = 6,
    n_directions: int = 24,
    aspect: float = SPHEROID_ASPECT,
) -> dict[str, Any]:
    """Full cross-validation study: sphere + prolate spheroid, all routes."""
    bodies = tuple(bodies)
    for body in bodies:
        if body not in ("sphere", "spheroid"):
            raise ValueError(f"unknown body {body!r}")
    sphere_cases: list[dict[str, Any]] = []
    spheroid_cases: list[dict[str, Any]] = []
    for sub in subdivisions:
        if "sphere" in bodies:
            sphere_cases.append(
                _sphere_case(sub, radii, conductivity, potential, quadrature_order, n_directions)
            )
        if "spheroid" in bodies:
            spheroid_cases.append(
                _spheroid_case(
                    sub, radii, aspect, conductivity, potential, quadrature_order, n_directions
                )
            )

    worst_surface = max(c["surface_route"]["rel_error"] for c in sphere_cases + spheroid_cases)
    worst_far = max(c["far_field_route"]["rel_error"] for c in sphere_cases + spheroid_cases)
    return {
        "study": "xval_two_route_gauss_law_current",
        "ledger_anchor": "XVAL-BEM-TWO-ROUTE-GAUSS-LAW-CURRENT",
        "conductivity_S_m": float(conductivity),
        "potential_V": float(potential),
        "quadrature_order": int(quadrature_order),
        "n_directions": int(n_directions),
        "radii_m": [float(r) for r in radii],
        "sphere_cases": sphere_cases,
        "spheroid_cases": spheroid_cases,
        "worst_surface_route_rel_error": float(worst_surface),
        "worst_far_field_route_rel_error": float(worst_far),
        "acceptance_band": 0.15,
        "pass": bool(worst_surface < 0.15 and worst_far < 0.15),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _figures(study: dict[str, Any], out_dir: Path) -> list[dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    figures: list[dict[str, str]] = []

    # (1) convergence of both routes vs exact, both bodies
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for cases, marker, label in (
        (study["sphere_cases"], "o", "sphere"),
        (study["spheroid_cases"], "s", f"prolate spheroid {SPHEROID_ASPECT:g}:1"),
    ):
        n = np.array([c["n_panels"] for c in cases], dtype=float)
        surf = np.array([c["surface_route"]["rel_error"] for c in cases])
        far = np.array([c["far_field_route"]["rel_error"] for c in cases])
        ax.loglog(n, surf, marker + "-", label=f"surface route, {label}")
        ax.loglog(n, far, marker + "--", label=f"far-field route, {label}")
    ax.axhline(0.15, color="crimson", ls=":", label="acceptance band 15%")
    ax.axhline(0.03, color="orange", ls=":", label="3% target")
    ax.set_xlabel("panels")
    ax.set_ylabel("relative error of the total current")
    ax.set_title("Two-route current vs closed form")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    path = out_dir / "route_convergence.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append({"path": str(path), "caption": "relative error of the surface and far-field current routes against the closed form (sphere and 2:1 prolate spheroid)"})

    # (2) far-field current versus 1/R and the extrapolation
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for cases, marker, label in (
        (study["sphere_cases"], "o", "sphere"),
        (study["spheroid_cases"], "s", "spheroid 2:1"),
    ):
        case = cases[-1]
        radii = np.array(case["far_field_route"]["radii_m"])
        cur = np.abs(np.array(case["far_field_route"]["current_by_radius_A"]))
        ax.plot(1.0 / radii, cur / abs(case["exact_current_A"]), marker + "-", label=f"{label}, R-sweep")
        ax.axhline(1.0, color="k", ls=":", lw=1.0)
    ax.set_xlabel("1 / R  (1/m)")
    ax.set_ylabel("I(R) / I_exact")
    ax.set_title("Far-field (Gauss-law) current: 1/R extrapolation")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    path = out_dir / "far_field_decay.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append({"path": str(path), "caption": "far-field Gauss-law current versus 1/R; the intercept is the extrapolated total current"})

    # (3) spheroid pointwise distribution vs closed form
    finest = study["spheroid_cases"][-1]
    if "pointwise_distribution" in finest:
        mesh = prolate_spheroid_mesh(SPHEROID_ASPECT, 1.0, int(finest["label"].split("sub")[-1]))
        result = _solve_uniform_dirichlet(
            mesh, study["potential_V"], study["conductivity_S_m"], study["quadrature_order"]
        )
        centroids = mesh.vertices[mesh.faces].mean(axis=1)
        ref = prolate_spheroid_surface_current_density(
            SPHEROID_ASPECT, 1.0, study["conductivity_S_m"], study["potential_V"], points=centroids
        )
        eta = np.sort(np.clip(centroids[:, 0] / SPHEROID_ASPECT, -1, 1))
        order = np.argsort(np.clip(centroids[:, 0] / SPHEROID_ASPECT, -1, 1))
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.plot(eta, ref[order], "k-", lw=1.2, label="closed form j_n(eta)")
        ax.plot(eta, np.abs(result.current_density)[order], "o", ms=2.5, alpha=0.6,
                label=f"BEM panels (N={mesh.panel_count})")
        ax.set_xlabel("eta = x / a  (0 = equator, +-1 = tips)")
        ax.set_ylabel("|j_n|  (A/m^2)")
        ax.set_title("Spheroid surface current distribution, 2:1")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        path = out_dir / "spheroid_distribution.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        figures.append({"path": str(path), "caption": "panel-wise current density on the 2:1 prolate spheroid versus the closed-form eta profile"})

    return figures


def merge_fragments(
    out_dir: Path,
    conductivity: float = SEAWATER_CONDUCTIVITY,
    potential: float = -1.0,
    quadrature_order: int = 6,
    n_directions: int = 24,
    radii: Sequence[float] = (3.0, 5.0, 8.0, 12.0, 20.0),
    figures: bool = True,
) -> dict[str, Any]:
    """Merge per-refinement fragments written by ``--fragment`` into one study."""
    spheres: list[dict[str, Any]] = []
    spheroids: list[dict[str, Any]] = []
    for path in sorted(Path(out_dir).glob("fragment_*.json")):
        with open(path) as fh:
            frag = json.load(fh)
        spheres.extend(frag.get("sphere_cases", []))
        spheroids.extend(frag.get("spheroid_cases", []))
    key = lambda c: c["n_panels"]
    spheres.sort(key=key)
    spheroids.sort(key=key)
    study = {
        "study": "xval_two_route_gauss_law_current",
        "ledger_anchor": "XVAL-BEM-TWO-ROUTE-GAUSS-LAW-CURRENT",
        "conductivity_S_m": float(conductivity),
        "potential_V": float(potential),
        "quadrature_order": int(quadrature_order),
        "n_directions": int(n_directions),
        "radii_m": [float(r) for r in radii],
        "sphere_cases": spheres,
        "spheroid_cases": spheroids,
        "worst_surface_route_rel_error": float(
            max(c["surface_route"]["rel_error"] for c in spheres + spheroids)
        ),
        "worst_far_field_route_rel_error": float(
            max(c["far_field_route"]["rel_error"] for c in spheres + spheroids)
        ),
        "acceptance_band": 0.15,
    }
    study["pass"] = bool(
        study["worst_surface_route_rel_error"] < 0.15
        and study["worst_far_field_route_rel_error"] < 0.15
    )
    if figures:
        study["figures"] = _figures(study, Path(out_dir))
    with open(Path(out_dir) / "results.json", "w") as fh:
        json.dump(study, fh, indent=2)
    return study


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/xval_two_route_gauss_law_current")
    parser.add_argument("--subdivisions", default="1,2,3")
    parser.add_argument("--bodies", default="sphere,spheroid")
    parser.add_argument("--fragment", default=None,
                        help="write this chunk to fragment_<TAG>.json instead of results.json")
    parser.add_argument("--merge-only", action="store_true",
                        help="merge the fragment_*.json files already in --output")
    parser.add_argument("--n-directions", type=int, default=24)
    parser.add_argument("--aspect", type=float, default=SPHEROID_ASPECT)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        study = merge_fragments(out_dir, n_directions=args.n_directions,
                                figures=not args.no_figures)
        print(json.dumps({k: study[k] for k in (
            "worst_surface_route_rel_error", "worst_far_field_route_rel_error", "pass")},
            indent=2))
        return study

    subdivisions = [int(v) for v in args.subdivisions.split(",") if v.strip()]
    bodies = [v.strip() for v in args.bodies.split(",") if v.strip()]
    study = run_xval(
        subdivisions=subdivisions, bodies=bodies,
        n_directions=args.n_directions, aspect=args.aspect
    )
    if args.fragment:
        payload = {k: study[k] for k in ("sphere_cases", "spheroid_cases")}
        with open(out_dir / f"fragment_{args.fragment}.json", "w") as fh:
            json.dump(payload, fh, indent=2)
        print(json.dumps({"fragment": args.fragment,
                          "sphere_cases": len(payload["sphere_cases"]),
                          "spheroid_cases": len(payload["spheroid_cases"])}))
        return study
    if not args.no_figures:
        study["figures"] = _figures(study, out_dir)
    with open(out_dir / "results.json", "w") as fh:
        json.dump(study, fh, indent=2)
    print(
        json.dumps(
            {
                "worst_surface_route_rel_error": study["worst_surface_route_rel_error"],
                "worst_far_field_route_rel_error": study["worst_far_field_route_rel_error"],
                "pass": study["pass"],
                "output": str(out_dir),
            },
            indent=2,
        )
    )
    return study


if __name__ == "__main__":  # pragma: no cover
    main()