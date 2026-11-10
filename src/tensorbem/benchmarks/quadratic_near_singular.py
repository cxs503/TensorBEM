"""Near-singular layer integrals of quadratic panels against a semi-analytic reference.

Why this benchmark exists
-------------------------
:class:`tensorbem.panel_quadratic.QuadraticTriangle` evaluates the singular
*self* integrals (collocation point on the element: a vertex or an edge
midpoint) with a Duffy rule that cancels the ``1/r`` singularity exactly.
Nearly-singular pairs, i.e. a collocation point at a distance ``d`` off a
panel that is small compared with the panel size ``H``, were not covered: the
plain Gauss rule of :meth:`QuadraticTriangle.integrate_single_layer` needs
``O(H/d)`` points to resolve the ``1/r`` peak, so at ``d/H = 1e-3`` it is off
by tens of percent.

The graded Duffy rule
(:func:`tensorbem.panel_quadratic.graded_apex_quadrature`) keeps the Duffy
map centred on the projection of the target and splits the radial direction
into geometrically growing intervals, so the near-singular peak is resolved
with ``O(log(H/d))`` radial intervals.

Reference
---------
For a **planar** element the layer integrals reduce to a one-dimensional
integral: polar coordinates about the foot of the perpendicular ``F`` of the
target give ``r = sqrt(rho**2 + d**2)`` and the radial integral of a
*quadratic* integrand (the shape functions are quadratic in the plane
coordinates) has a closed form.  The remaining ``theta`` integral runs over
sectors bounded by the rays to the three vertices, inside which the exit
distance ``rho_max(theta)`` is analytic (the ray leaves through a fixed edge),
so a 64-point Gauss rule per sector is machine-accurate.  The result is
therefore a semi-analytic reference at the ``1e-15`` level, independent of the
quadrature under test.

Studies
-------
``distance_ladder``
    Single- and double-layer relative ``L2`` error versus ``d/H`` for three
    footprints (element centroid, edge midpoint, vertex) and two quadratures:
    the plain Gauss rule (baseline) and the graded Duffy rule (new).  The
    baseline degrades as ``d/H -> 0``; the graded rule does not.
``order_ladder``
    Convergence of the graded rule in its Gauss order and in the radial
    growth factor, at the fixed near-singular distance ``d/H = 1e-3``.
``curved_element``
    Order self-convergence of the graded rule on a *curved* (icosphere)
    element, where no semi-analytic reference exists: the order ladder must be
    flat to ``1e-8`` while the plain Gauss rule stalls.

All numbers in the JSON are computed here (nothing copied by hand) and the
figure hash is recorded in ``provenance``.

Run::

    PYTHONPATH=src python -m tensorbem.benchmarks.quadratic_near_singular
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from tensorbem.panel_quadratic import (
    FOUR_PI,
    QuadraticSurfaceMesh,
    QuadraticTriangle,
    icosphere_quadratic,
)

FloatArray = NDArray[np.float64]

DEFAULT_JSON = "docs/quadratic_near_singular/study.json"
DEFAULT_FIGURE = "docs/quadratic_near_singular/fig_near_singular.png"

#: Reference quadrature used for the ``theta`` integration (per sector).
THETA_ORDER = 64

#: Footprints on the reference triangle used by the distance ladder.
FOOTPRINTS: dict[str, tuple[float, float]] = {
    "centroid": (1.0 / 3.0, 1.0 / 3.0),
    "edge_midpoint_01": (0.5, 0.5),
    "vertex_v0": (1.0, 0.0),
}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def flat_reference_triangle(size: float = 1.0) -> QuadraticTriangle:
    """Unit right triangle in the ``z = 0`` plane, midpoints on the edges.

    The element is flat by construction (the quadratic mapping degenerates to
    an affine one), which is what makes the semi-analytic reference available.
    """
    v0 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([size, 0.0, 0.0])
    v2 = np.array([0.0, size, 0.0])
    return QuadraticTriangle(
        np.vstack([v0, v1, v2, 0.5 * (v0 + v1), 0.5 * (v1 + v2), 0.5 * (v2 + v0)])
    )


def _plane_frame(
    tri: QuadraticTriangle, tol: float = 1e-12
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Orthonormal in-plane frame ``(origin, e1, e2)`` plus the unit normal.

    Raises ``ValueError`` if the element is not planar within *tol*.
    """
    v0, v1, v2 = tri.nodes[0], tri.nodes[1], tri.nodes[2]
    normal = np.cross(v1 - v0, v2 - v0)
    normal = normal / np.linalg.norm(normal)
    off_plane = np.abs((tri.nodes - v0) @ normal)
    if float(np.max(off_plane)) > tol:
        raise ValueError("element is not planar; semi-analytic reference unavailable")
    e1 = (v1 - v0) / np.linalg.norm(v1 - v0)
    e2 = np.cross(normal, e1)
    return v0, e1, e2, normal


def planar_shape_polynomials(tri: QuadraticTriangle) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Coefficients of the shape functions as quadratics in the plane frame.

    Returns
    -------
    origin, e1, e2 : (3,) plane frame.
    coeffs : (6, 6) array; row ``k`` holds ``[1, x, y, x**2, x*y, y**2]``
        coefficients of shape function ``k`` in the in-plane coordinates.
    """
    v0, e1, e2, _ = _plane_frame(tri)
    local = np.column_stack(
        [(tri.nodes - v0) @ e1, (tri.nodes - v0) @ e2]
    )  # (6, 2)
    vandermonde = np.column_stack(
        [
            np.ones(6),
            local[:, 0],
            local[:, 1],
            local[:, 0] ** 2,
            local[:, 0] * local[:, 1],
            local[:, 1] ** 2,
        ]
    )
    identity = np.eye(6)
    coeffs = np.linalg.solve(vandermonde, identity)  # (6 basis, 6 nodes)
    return v0, e1, e2, coeffs.T


def _ray_exit(origin: FloatArray, direction: FloatArray, corners: FloatArray) -> float:
    """Distance from *origin* along *direction* to the boundary of the triangle."""
    best = 0.0
    for k in range(3):
        p = corners[k]
        q = corners[(k + 1) % 3]
        edge = q - p
        mat = np.array([[direction[0], -edge[0]], [direction[1], -edge[1]]])
        det = float(np.linalg.det(mat))
        if abs(det) < 1e-15:
            continue
        t, s = np.linalg.solve(mat, p - origin)
        if t > 0.0 and -1e-12 <= s <= 1.0 + 1e-12:
            if best == 0.0 or t < best:
                best = float(t)
    return best


def _radial_single(coeffs: tuple[float, float, float], radius: float, d: float) -> float:
    """``int_0^R (A0 + A1 rho + A2 rho**2) rho drho / sqrt(rho**2 + d**2)``."""
    if radius <= 0.0:
        return 0.0
    a0, a1, a2 = coeffs
    root = math.hypot(radius, d)
    i0 = root - d
    i1 = 0.5 * (radius * root - d * d * math.asinh(radius / d))
    i2 = root ** 3 / 3.0 - d * d * root + 2.0 * d ** 3 / 3.0
    return a0 * i0 + a1 * i1 + a2 * i2


def _radial_double(coeffs: tuple[float, float, float], radius: float, d: float) -> float:
    """``int_0^R (A0 + A1 rho + A2 rho**2) rho drho / (rho**2 + d**2)**1.5``."""
    if radius <= 0.0:
        return 0.0
    a0, a1, a2 = coeffs
    root = math.hypot(radius, d)
    j0 = 1.0 / d - 1.0 / root
    j1 = math.asinh(radius / d) - radius / root
    j2 = root + d * d / root - 2.0 * d
    return a0 * j0 + a1 * j1 + a2 * j2


def planar_layer_reference(
    tri: QuadraticTriangle,
    target: FloatArray,
    theta_order: int = THETA_ORDER,
) -> tuple[FloatArray, FloatArray]:
    """Semi-analytic single- and double-layer integrals for a planar element.

    The target must lie off the element plane (``d > 0`` on the outward side);
    ``d`` may be arbitrarily small — the radial integrals are evaluated in
    closed form, so there is no near-singularity in this reference at all.

    Returns
    -------
    single, double : (6,) ``int_T N_k G dS`` and ``int_T N_k dG/dn_y dS``.
    """
    v0, e1, e2, normal = _plane_frame(tri)
    _, _, _, coeffs = planar_shape_polynomials(tri)
    target = np.asarray(target, dtype=float).reshape(3)
    d = float((target - v0) @ normal)
    if d <= 0.0:
        raise ValueError("target must be on the outward-normal side of the element")
    foot = np.array([(target - v0) @ e1, (target - v0) @ e2])
    corners = np.column_stack(
        [(tri.nodes[:3] - v0) @ e1, (tri.nodes[:3] - v0) @ e2]
    )

    single = np.zeros(6)
    double = np.zeros(6)
    nodes_1d, weights_1d = leggauss(theta_order)

    angles = []
    for k in range(3):
        delta = corners[k] - foot
        if np.linalg.norm(delta) > 1e-14:
            angles.append(math.atan2(delta[1], delta[0]))
    angles = sorted(angles)
    sectors = list(zip(angles[:-1], angles[1:])) + [(angles[-1], angles[0] + 2.0 * math.pi)]

    for lo, hi in sectors:
        theta = 0.5 * (hi - lo) * nodes_1d + 0.5 * (hi + lo)
        w_theta = 0.5 * (hi - lo) * weights_1d
        for t, w in zip(theta, w_theta):
            direction = np.array([math.cos(t), math.sin(t)])
            radius = _ray_exit(foot, direction, corners)
            if radius <= 0.0:
                continue
            # Shape functions along the ray: A0 + A1*rho + A2*rho**2.
            x0, y0 = foot
            ux, uy = direction
            a0 = (
                coeffs[:, 0]
                + coeffs[:, 1] * x0
                + coeffs[:, 2] * y0
                + coeffs[:, 3] * x0 * x0
                + coeffs[:, 4] * x0 * y0
                + coeffs[:, 5] * y0 * y0
            )
            a1 = (
                coeffs[:, 1] * ux
                + coeffs[:, 2] * uy
                + 2.0 * coeffs[:, 3] * x0 * ux
                + coeffs[:, 4] * (x0 * uy + y0 * ux)
                + 2.0 * coeffs[:, 5] * y0 * uy
            )
            a2 = (
                coeffs[:, 3] * ux * ux
                + coeffs[:, 4] * ux * uy
                + coeffs[:, 5] * uy * uy
            )
            for k in range(6):
                abc = (float(a0[k]), float(a1[k]), float(a2[k]))
                single[k] += w * _radial_single(abc, radius, d) / FOUR_PI
                double[k] += w * d * _radial_double(abc, radius, d) / FOUR_PI
    return single, double


# ---------------------------------------------------------------------------
# Study helpers
# ---------------------------------------------------------------------------
def relative_l2(values: FloatArray, reference: FloatArray) -> float:
    """Relative ``L2`` error against a reference vector."""
    denom = float(np.linalg.norm(reference))
    if denom == 0.0:
        return float(np.linalg.norm(values))
    return float(np.linalg.norm(values - reference) / denom)


def offset_target(
    tri: QuadraticTriangle, footprint: tuple[float, float], distance: float
) -> FloatArray:
    """Point on the outward normal through the given reference footprint."""
    _, _, _, normal = _plane_frame(tri)
    base = tri.map_to_physical(float(footprint[0]), float(footprint[1]))
    return np.asarray(base, dtype=float) + distance * normal


def run_distance_ladder(
    tri: QuadraticTriangle,
    distances: tuple[float, ...],
    baseline_orders: tuple[int, ...] = (8, 64),
    graded_orders: tuple[int, ...] = (12, 24, 32),
    growth: float = 8.0,
    theta_order: int = THETA_ORDER,
) -> dict:
    """Error of the plain Gauss rule and of the graded rule versus ``d/H``."""
    size = tri.characteristic_size
    rows = []
    for name, footprint in FOOTPRINTS.items():
        for ratio in distances:
            distance = ratio * size
            target = offset_target(tri, footprint, distance)
            single_ref, double_ref = planar_layer_reference(
                tri, target, theta_order=theta_order
            )
            record: dict = {
                "footprint": name,
                "d_over_H": ratio,
                "distance": distance,
                "projected_footprint_error": float(
                    np.linalg.norm(tri.project_to_reference(target)[0] - np.array(footprint))
                ),
            }
            for order in baseline_orders:
                record[f"gauss_order_{order}"] = {
                    "single_error": relative_l2(
                        tri.integrate_single_layer(target, order=order), single_ref
                    ),
                    "double_error": relative_l2(
                        tri.integrate_double_layer(target, order=order), double_ref
                    ),
                }
            graded: dict[str, dict] = {}
            for order in graded_orders:
                single_near, double_near = tri.near_singular_layer_integrals(
                    target, order=order, growth=growth
                )
                graded[str(order)] = {
                    "single_error": relative_l2(single_near, single_ref),
                    "double_error": relative_l2(double_near, double_ref),
                }
            record["graded"] = graded
            rows.append(record)
    return {
        "panel_size": size,
        "graded_orders": list(graded_orders),
        "baseline_orders": list(baseline_orders),
        "rows": rows,
    }


def run_order_ladder(
    tri: QuadraticTriangle,
    footprint: tuple[float, float],
    d_over_H: float,
    orders: tuple[int, ...] = (4, 6, 8, 12, 16),
    growths: tuple[float, ...] = (2.0, 4.0, 8.0),
    theta_order: int = THETA_ORDER,
) -> dict:
    """Convergence of the graded rule in its order and growth factor."""
    size = tri.characteristic_size
    target = offset_target(tri, footprint, d_over_H * size)
    single_ref, double_ref = planar_layer_reference(
        tri, target, theta_order=theta_order
    )
    by_order = []
    for order in orders:
        single, double = tri.near_singular_layer_integrals(
            target, order=order, growth=8.0
        )
        by_order.append(
            {
                "order": order,
                "single_error": relative_l2(single, single_ref),
                "double_error": relative_l2(double, double_ref),
            }
        )
    by_growth = []
    for growth in growths:
        single, double = tri.near_singular_layer_integrals(
            target, order=12, growth=growth
        )
        by_growth.append(
            {
                "growth": growth,
                "single_error": relative_l2(single, single_ref),
                "double_error": relative_l2(double, double_ref),
            }
        )
    return {
        "footprint": list(footprint),
        "d_over_H": d_over_H,
        "distance": d_over_H * size,
        "by_order": by_order,
        "by_growth": by_growth,
    }


def pick_curved_element(mesh: QuadraticSurfaceMesh) -> int:
    """Index of the most strongly curved element (largest bulge at the centre).

    For a flat element the isoparametric centre coincides with the centroid of
    the three vertices; on a curved element it is displaced by ``O(h**2)``
    along the normal, which is what makes the element a genuine quadratic one.
    """
    best = 0
    best_bend = -1.0
    for index in range(mesh.element_count):
        tri = mesh.element(index)
        mid = tri.map_to_physical(1.0 / 3.0, 1.0 / 3.0)
        flat_centroid = (tri.nodes[0] + tri.nodes[1] + tri.nodes[2]) / 3.0
        bend = float(np.linalg.norm(mid - flat_centroid))
        if bend > best_bend:
            best_bend = bend
            best = index
    return best


def run_curved_element(
    subdivisions: int = 1,
    d_over_H: float = 1e-3,
    orders: tuple[int, ...] = (8, 16, 24, 32),
    reference_order: int = 40,
) -> dict:
    """Order self-convergence of the graded rule on a curved element.

    No semi-analytic reference exists for a curved element, so the reference is
    the graded rule at a higher order (*reference_order*) and the ladder must
    converge towards it: the near-singular peak is resolved once the radial
    grading is in place, which the plain Gauss rule never achieves at
    ``d/H = 1e-3``.
    """
    mesh = icosphere_quadratic(subdivisions=subdivisions)
    index = pick_curved_element(mesh)
    tri = mesh.element(index)
    size = tri.characteristic_size
    mid = tri.map_to_physical(1.0 / 3.0, 1.0 / 3.0)
    normals = tri._geometry_at(np.array([[1.0 / 3.0, 1.0 / 3.0]]))[3][0]
    target = mid + d_over_H * size * normals

    reference = tri.near_singular_layer_integrals(target, order=reference_order)
    ladder = []
    for order in orders:
        single, double = tri.near_singular_layer_integrals(target, order=order)
        ladder.append(
            {
                "order": order,
                "single_error": relative_l2(single, reference[0]),
                "double_error": relative_l2(double, reference[1]),
                "sum_single": float(np.sum(single)),
            }
        )

    gauss = []
    for order in (8, 16, 32, 64):
        single = tri.integrate_single_layer(target, order=order)
        gauss.append(
            {
                "order": order,
                "single_error": relative_l2(single, reference[0]),
                "sum_single": float(np.sum(single)),
            }
        )

    return {
        "subdivisions": subdivisions,
        "element_index": index,
        "d_over_H": d_over_H,
        "panel_size": size,
        "target_offset": float(
            np.linalg.norm(
                target - tri.map_to_physical(1.0 / 3.0, 1.0 / 3.0)
            )
        ),
        "graded_ladder": ladder,
        "gauss_ladder": gauss,
        "graded_max_error": max(
            max(row["single_error"], row["double_error"]) for row in ladder
        ),
        "gauss_min_error": min(row["single_error"] for row in gauss),
        "reference_order": reference_order,
    }


# ---------------------------------------------------------------------------
# Aggregate study
# ---------------------------------------------------------------------------
def run_study(
    distances: tuple[float, ...] = (1.0, 0.3, 0.1, 0.01, 1e-3, 1e-4),
    theta_order: int = THETA_ORDER,
) -> dict:
    """Run all three studies and the acceptance gates."""
    tri = flat_reference_triangle()

    # Sanity check of the semi-analytic reference against a very high order
    # Gauss rule in a benign (well-resolved) regime: both must agree to 1e-12.
    benign_target = offset_target(tri, FOOTPRINTS["centroid"], 0.5)
    single_ref, double_ref = planar_layer_reference(
        tri, benign_target, theta_order=theta_order
    )
    oracle_check = {
        "d_over_H": 0.5 * tri.characteristic_size,
        "single_rel_error_vs_gauss64": relative_l2(
            tri.integrate_single_layer(benign_target, order=64), single_ref
        ),
        "double_rel_error_vs_gauss64": relative_l2(
            tri.integrate_double_layer(benign_target, order=64), double_ref
        ),
    }

    distance_ladder = run_distance_ladder(tri, distances, theta_order=theta_order)
    order_ladder = run_order_ladder(
        tri, FOOTPRINTS["centroid"], 1e-3, theta_order=theta_order
    )

    graded_by_order: dict[str, float] = {}
    for order in distance_ladder["graded_orders"]:
        graded_by_order[str(order)] = max(
            max(row["graded"][str(order)]["single_error"],
                row["graded"][str(order)]["double_error"])
            for row in distance_ladder["rows"]
        )
    baseline_at_1e_minus_3 = [
        row
        for row in distance_ladder["rows"]
        if row["d_over_H"] == 1e-3 and row["footprint"] == "centroid"
    ][0]
    curved = run_curved_element()
    gates = {
        "oracle_vs_gauss": 1e-10,
        "graded_worst_error_order_12": 1e-3,
        "graded_worst_error_order_24": 1e-6,
        "graded_worst_error_order_32": 1e-9,
        "baseline_error_at_d_over_H_1e-3": 1e-2,
        "curved_graded_error_order_32": 1e-9,
    }
    acceptance = {
        "thresholds": gates,
        "oracle_consistent_with_gauss": bool(
            max(
                oracle_check["single_rel_error_vs_gauss64"],
                oracle_check["double_rel_error_vs_gauss64"],
            )
            < gates["oracle_vs_gauss"]
        ),
        "graded_worst_error_by_order": graded_by_order,
        "graded_passed": bool(
            graded_by_order["12"] < gates["graded_worst_error_order_12"]
            and graded_by_order["24"] < gates["graded_worst_error_order_24"]
            and graded_by_order["32"] < gates["graded_worst_error_order_32"]
        ),
        "baseline_single_error_at_d_over_H_1e-3": baseline_at_1e_minus_3[
            "gauss_order_8"
        ]["single_error"],
        "baseline_is_actually_bad": bool(
            baseline_at_1e_minus_3["gauss_order_8"]["single_error"]
            > gates["baseline_error_at_d_over_H_1e-3"]
        ),
        "curved_graded_error_order_32": max(
            [
                max(row["single_error"], row["double_error"])
                for row in curved["graded_ladder"]
                if row["order"] == 32
            ]
        ),
        "curved_graded_passed": bool(
            max(
                max(row["single_error"], row["double_error"])
                for row in curved["graded_ladder"]
                if row["order"] == 32
            )
            < gates["curved_graded_error_order_32"]
        ),
    }
    acceptance["passed"] = bool(
        acceptance["oracle_consistent_with_gauss"]
        and acceptance["graded_passed"]
        and acceptance["baseline_is_actually_bad"]
        and acceptance["curved_graded_passed"]
    )

    return {
        "module": "tensorbem.benchmarks.quadratic_near_singular",
        "quadrature": {
            "theta_order": theta_order,
            "graded_default_order": 12,
            "graded_default_growth": 8.0,
        },
        "oracle_check": oracle_check,
        "distance_ladder": distance_ladder,
        "order_ladder": order_ladder,
        "curved_element": curved,
        "acceptance": acceptance,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def build_figure(study: dict, path: Path) -> None:
    """Error-versus-distance figure (single and double layer)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = study["distance_ladder"]["rows"]
    footprints = sorted({row["footprint"] for row in rows})
    fig, axes = plt.subplots(2, len(footprints), figsize=(4.0 * len(footprints), 7.0))
    axes = np.atleast_2d(axes)
    for col, name in enumerate(footprints):
        subset = [row for row in rows if row["footprint"] == name]
        ratio = [row["d_over_H"] for row in subset]
        for key, label, style in (
            ("gauss_order_8", "plain Gauss, order 8", "o--"),
            ("gauss_order_64", "plain Gauss, order 64", "s-."),
        ):
            axes[0, col].loglog(
                ratio, [row[key]["single_error"] for row in subset], style,
                label=label, markersize=4,
            )
            axes[1, col].loglog(
                ratio, [row[key]["double_error"] for row in subset], style,
                label=label, markersize=4,
            )
        axes[0, col].loglog(
            ratio, [row["graded"]["12"]["single_error"] for row in subset], "^-",
            label="graded Duffy, order 12 (new)", color="crimson", markersize=5,
        )
        axes[0, col].loglog(
            ratio, [row["graded"]["32"]["single_error"] for row in subset], "^--",
            label="graded Duffy, order 32 (new)", color="darkgreen", markersize=5,
        )
        axes[1, col].loglog(
            ratio, [row["graded"]["32"]["double_error"] for row in subset], "^--",
            label="graded Duffy, order 32 (new)", color="darkgreen", markersize=5,
        )
        for ax, title in zip(axes[:, col], ("single layer", "double layer")):
            ax.set_title(f"{title}\nfootprint = {name}", fontsize=9)
            ax.set_xlabel("d / H")
            ax.set_ylabel("relative L2 error vs semi-analytic")
            ax.grid(True, which="both", alpha=0.3)
            ax.axhline(1e-8, color="k", lw=0.8, ls=":")
            ax.legend(fontsize=7)
        axes[0, col].invert_xaxis()
        axes[1, col].invert_xaxis()
    fig.suptitle(
        "Quadratic panel near-singular layer integrals: plain Gauss vs graded Duffy",
        fontsize=10,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict:
    """Run the study, write the artifacts and gate on acceptance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=str, default=DEFAULT_JSON)
    parser.add_argument("--figure", type=str, default=DEFAULT_FIGURE)
    parser.add_argument("--theta-order", type=int, default=THETA_ORDER)
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="exit 0 even when an acceptance gate fails (report the flags only)",
    )
    args = parser.parse_args(argv)

    study = run_study(theta_order=args.theta_order)

    if args.figure:
        build_figure(study, Path(args.figure))

    provenance: dict[str, dict] = {}
    if args.figure and Path(args.figure).exists():
        provenance["figure_near_singular"] = {
            "path": args.figure,
            "sha256": hashlib.sha256(Path(args.figure).read_bytes()).hexdigest(),
        }
    study["provenance"] = provenance

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(study, indent=2) + "\n")

    print("footprint            d/H     gauss8(S)   gauss64(S)   graded(S)   graded(D)")
    for row in study["distance_ladder"]["rows"]:
        print(
            f"{row['footprint']:18s} {row['d_over_H']:8.0e}"
            f"  {row['gauss_order_8']['single_error']:.3e}"
            f"  {row['gauss_order_64']['single_error']:.3e}"
            f"  {row['graded']['12']['single_error']:.3e}"
            f"  {row['graded']['32']['double_error']:.3e}"
        )
    print(
        "oracle vs plain Gauss(64) at d/H=0.5: "
        f"{study['oracle_check']['single_rel_error_vs_gauss64']:.3e}"
    )
    print(
        "curved element: graded max error "
        f"{study['curved_element']['graded_max_error']:.3e}, "
        f"gauss min error {study['curved_element']['gauss_min_error']:.3e}"
    )
    print(f"acceptance passed: {study['acceptance']['passed']}")
    print(f"JSON: {json_path}")
    if args.figure:
        print(f"figure: {args.figure}")

    if not study["acceptance"]["passed"] and not args.allow_fail:
        raise SystemExit(
            "acceptance gate failed: "
            + json.dumps(
                {
                    k: v
                    for k, v in study["acceptance"].items()
                    if isinstance(v, bool)
                },
                indent=2,
            )
        )
    return study


if __name__ == "__main__":  # pragma: no cover
    main()