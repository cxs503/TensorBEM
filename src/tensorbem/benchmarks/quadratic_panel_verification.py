"""Quadratic (6-node) panel operator verification against closed-form references.

The quadratic panel module (:mod:`tensorbem.panel_quadratic`) claims

* ``O(h**3)`` geometric approximation (isoparametric mapping of curved
  surfaces), and
* a second-kind boundary integral formulation
  ``(1/2 I - D) u = -S q`` built from the single-layer operator ``S`` and the
  principal-value double-layer operator ``D``.

This benchmark measures both claims against *closed-form* references for the
exterior 3-D Laplace Dirichlet problem on the unit sphere with an interior
point source at ``x0 = (0, 0, 0.5)``::

    u(x)  = 1 / (4 pi |x - x0|)                     (exact potential)
    q(x)  = grad(u) . n = -(x - x0) . n / (4 pi |x - x0|^3)   (exact flux)

Studies
-------
1. ``flux`` — solve the *direct* second-kind BIE ``q = -S^{-1} (1/2 I - D) u``
   for Dirichlet data ``u`` sampled from the exact potential and compare the
   nodal flux with the exact flux (relative L2 norm at element centroids).
   This is the physical acceptance quantity; it exercises ``S``, ``D`` and the
   singular self-panel quadrature in one number.
2. ``single_layer_identity`` — ``S 1 = 1`` on the unit sphere (the
   single-layer potential of unit density is exactly ``R`` on and inside the
   sphere).
3. ``solid_angle_identity`` — ``D 1 = -1/2`` (principal value of the
   double-layer kernel over a closed smooth surface).
4. ``second_kind_identity`` — ``(1/2 I - D) 1 = 1``.
5. ``flat_panel`` — the double-layer self block of a flat element is exactly
   zero (``(x - y) . n == 0`` in the panel plane); a pure analytic check of the
   singular quadrature.

All numbers written to the JSON report are computed here (nothing is copied by
hand).  Convergence orders are least-squares slopes of ``log(error)`` versus
``log(h)`` with ``h = sqrt(mean element area)``.

Run::

    PYTHONPATH=src python -m tensorbem.benchmarks.quadratic_panel_verification
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..panel_quadratic import (
    QuadraticSurfaceMesh,
    QuadraticTriangle,
    assemble_quadratic_matrices,
    icosphere_quadratic,
    quadratic_shape_functions_array,
    solve_laplace_dirichlet_quadratic,
    solve_neumann_from_dirichlet_quadratic,
)

FloatArray = NDArray[np.float64]

FOUR_PI = 4.0 * math.pi

#: Interior point source of the exact solution.
SOURCE = np.array([0.0, 0.0, 0.5])

#: Barycentric centroid of the reference triangle.
CENTROID_L = np.array([1.0 / 3.0, 1.0 / 3.0])

#: Acceptance threshold on the relative flux error at the finest mesh level.
FLUX_ACCEPTANCE = 0.03

#: Hard failure threshold (errors above this are unusable for engineering work).
FLUX_HARD_LIMIT = 0.15

DEFAULT_JSON = "docs/quadratic_panel_verification/study.json"
DEFAULT_FIGURE = "docs/quadratic_panel_verification/fig_quadratic_operator.png"


# ---------------------------------------------------------------------------
# Closed-form references
# ---------------------------------------------------------------------------
def analytic_potential(points: FloatArray, source: FloatArray = SOURCE) -> FloatArray:
    """Exact potential ``1 / (4 pi |x - x0|)`` of an interior point source."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    r = np.linalg.norm(pts - source, axis=1)
    return 1.0 / (FOUR_PI * r)


def analytic_flux(
    points: FloatArray, normals: FloatArray, source: FloatArray = SOURCE
) -> FloatArray:
    """Exact outward flux ``grad(u) . n`` of the same point source."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    nrm = np.atleast_2d(np.asarray(normals, dtype=float))
    delta = pts - source
    r = np.linalg.norm(delta, axis=1)
    grad = -delta / (FOUR_PI * r[:, None] ** 3)
    return np.einsum("ij,ij->i", grad, nrm)


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------
def mesh_characteristic_size(mesh: QuadraticSurfaceMesh) -> float:
    """``h = sqrt(mean flat vertex area)`` — a monotone mesh-size measure."""
    areas = [mesh.element(e).vertex_area for e in range(mesh.element_count)]
    return float(math.sqrt(sum(areas) / len(areas)))


def element_centroid_data(mesh: QuadraticSurfaceMesh) -> tuple[FloatArray, FloatArray]:
    """Isoparametric element centroids and their outward unit normals."""
    centres = np.empty((mesh.element_count, 3))
    normals = np.empty((mesh.element_count, 3))
    for e in range(mesh.element_count):
        elem = mesh.element(e)
        _X, _N, _det, nrm = elem._geometry_at(CENTROID_L)
        centres[e] = elem.map_to_physical(*CENTROID_L)
        normals[e] = nrm[0]
    return centres, normals


def interpolate_to_centroids(mesh: QuadraticSurfaceMesh, nodal: FloatArray) -> FloatArray:
    """Interpolate nodal values to the element centroids with the shape functions."""
    shape = quadratic_shape_functions_array(CENTROID_L)[0]
    return np.array([shape @ nodal[mesh.faces[e]] for e in range(mesh.element_count)])


def relative_l2(values: FloatArray, reference: FloatArray) -> float:
    """Relative L2 norm of ``values - reference``."""
    denom = float(np.linalg.norm(reference))
    return float(np.linalg.norm(values - reference) / denom) if denom > 0.0 else float("nan")


def loglog_slope(h: FloatArray, err: FloatArray) -> float:
    """Least-squares slope of ``log(err)`` against ``log(h)``."""
    h = np.asarray(h, dtype=float)
    err = np.asarray(err, dtype=float)
    mask = np.isfinite(h) & np.isfinite(err) & (h > 0.0) & (err > 0.0)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    x = np.log(h[mask])
    y = np.log(err[mask])
    return float(np.polyfit(x, y, 1)[0])


# ---------------------------------------------------------------------------
# Studies
# ---------------------------------------------------------------------------
def run_operator_identity_study(
    mesh: QuadraticSurfaceMesh,
    quadrature_order: int = 8,
    self_order: int = 16,
) -> dict:
    """Closed-form operator identities satisfied by S and D on the unit sphere."""
    n = mesh.node_count
    ones = np.ones(n)
    S, D = assemble_quadratic_matrices(
        mesh, quadrature_order=quadrature_order, self_order=self_order
    )
    second_kind = 0.5 * np.eye(n) - D
    return {
        "nodes": n,
        "elements": mesh.element_count,
        "single_layer_identity_error": float(np.max(np.abs(S @ ones - 1.0))),
        "solid_angle_identity_error": float(np.max(np.abs(D @ ones + 0.5))),
        "second_kind_identity_error": float(np.max(np.abs(second_kind @ ones - 1.0))),
        "double_layer_norm": float(np.linalg.norm(D)),
        "condition_number": float(np.linalg.cond(S)),
    }


def run_flux_study(
    mesh: QuadraticSurfaceMesh,
    quadrature_order: int = 8,
    self_order: int = 16,
) -> dict:
    """Direct second-kind BIE flux recovery versus the closed-form flux."""
    u_exact = analytic_potential(mesh.vertices)
    result = solve_neumann_from_dirichlet_quadratic(
        mesh, u_exact, quadrature_order=quadrature_order, self_order=self_order
    )
    centres, normals = element_centroid_data(mesh)
    q_exact = analytic_flux(centres, normals)
    q_computed = interpolate_to_centroids(mesh, result.normal_derivative)
    error = relative_l2(q_computed, q_exact)

    # Sanity contrast: the *equivalent source density* of the indirect
    # single-layer formulation (S sigma = u) is NOT the normal derivative; the
    # error of comparing the two is reported so the distinction stays visible.
    indirect = solve_laplace_dirichlet_quadratic(
        mesh, u_exact, quadrature_order=quadrature_order, self_order=self_order
    )
    density_as_flux = interpolate_to_centroids(mesh, indirect.nodal_values)

    return {
        "nodes": mesh.node_count,
        "elements": mesh.element_count,
        "h": mesh_characteristic_size(mesh),
        "flux_rel_error": error,
        "flux_max_abs_error": float(np.max(np.abs(q_computed - q_exact))),
        "flux_reference_max": float(np.max(np.abs(q_exact))),
        "condition_number": result.condition_number,
        "relative_residual_l2": result.relative_residual_l2,
        "solid_angle_identity_error": result.operator_row_sum_error,
        "single_layer_residual_l2": indirect.relative_residual_l2,
        "equivalent_density_vs_exact_flux_rel_error": relative_l2(density_as_flux, q_exact),
    }


def run_flat_panel_study(
    quadrature_orders: tuple[int, ...] = (8, 12, 16, 24, 32),
) -> dict:
    """Flat-panel analytic checks for the singular quadrature.

    * The double-layer self block of a flat triangle must vanish identically
      (target in the panel plane).
    * The single-layer apex quadrature must converge to its high-order value;
      the difference between order 16 and order 64 is the production error.
    """
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],  # v0
            [1.0, 0.0, 0.0],  # v1
            [0.0, 1.0, 0.0],  # v2
            [0.5, 0.0, 0.0],  # mid01
            [0.5, 0.5, 0.0],  # mid12
            [0.0, 0.5, 0.0],  # mid20
        ]
    )
    elem = QuadraticTriangle(nodes)
    S_self, D_self = elem.node_self_matrices(order=16)
    flat_self_max = float(np.max(np.abs(D_self)))

    convergence = []
    for order in quadrature_orders:
        s_test = np.array(
            [elem.integrate_single_layer_at_node(i, order=order) for i in range(6)]
        )
        s_ref = np.array(
            [elem.integrate_single_layer_at_node(i, order=64) for i in range(6)]
        )
        convergence.append(
            {
                "order": order,
                "single_layer_rel_error_vs_order64": float(
                    np.max(np.abs(s_test - s_ref)) / np.max(np.abs(s_ref))
                ),
            }
        )
    return {
        "flat_double_layer_self_max_abs": flat_self_max,
        "single_layer_quadrature_convergence": convergence,
        "single_layer_production_rel_error": convergence[-1][
            "single_layer_rel_error_vs_order64"
        ]
        if convergence
        else float("nan"),
    }


def run_study(
    levels: tuple[int, ...] = (0, 1, 2, 3),
    quadrature_order: int = 8,
    self_order: int = 16,
) -> dict:
    """Run every study over the requested icosphere refinement levels."""
    per_level = []
    for sub in levels:
        mesh = icosphere_quadratic(subdivisions=sub)
        identity = run_operator_identity_study(
            mesh, quadrature_order=quadrature_order, self_order=self_order
        )
        flux = run_flux_study(
            mesh, quadrature_order=quadrature_order, self_order=self_order
        )
        record = {"subdivisions": sub}
        record.update(flux)
        record.update(
            {
                "single_layer_identity_error": identity["single_layer_identity_error"],
                "second_kind_identity_error": identity["second_kind_identity_error"],
                "operator_condition_number": identity["condition_number"],
            }
        )
        per_level.append(record)

    h = np.array([rec["h"] for rec in per_level])
    orders = {
        "flux_rel_error": loglog_slope(h, [r["flux_rel_error"] for r in per_level]),
        "single_layer_identity_error": loglog_slope(
            h, [r["single_layer_identity_error"] for r in per_level]
        ),
        "solid_angle_identity_error": loglog_slope(
            h, [r["solid_angle_identity_error"] for r in per_level]
        ),
        "second_kind_identity_error": loglog_slope(
            h, [r["second_kind_identity_error"] for r in per_level]
        ),
    }

    finest = per_level[-1]
    finest_mesh = icosphere_quadratic(subdivisions=levels[-1])
    study = {
        "schema": "tensorbem.quadratic_panel_verification/v2",
        "case": {
            "geometry": "unit sphere (icosphere, quadratic 6-node panels)",
            "problem": "exterior 3-D Laplace Dirichlet",
            "source": SOURCE.tolist(),
            "exact_potential": "1/(4*pi*|x-x0|)",
            "exact_flux": "-(x-x0).n/(4*pi*|x-x0|^3)",
            "quadrature_order": quadrature_order,
            "self_order": self_order,
            "levels": list(levels),
            "h_definition": "sqrt(mean flat vertex area)",
        },
        "levels": per_level,
        "observed_orders": orders,
        "flat_panel": run_flat_panel_study(),
        "acceptance": {
            "threshold": FLUX_ACCEPTANCE,
            "hard_limit": FLUX_HARD_LIMIT,
            "flux_rel_error_finest": finest["flux_rel_error"],
            "passed_3pct": bool(finest["flux_rel_error"] <= FLUX_ACCEPTANCE),
            "passed_hard_limit": bool(finest["flux_rel_error"] <= FLUX_HARD_LIMIT),
        },
        # Internal (not serialised): nodal field of the finest level for plotting.
        "_finest_field": nodal_flux_field(
            finest_mesh, quadrature_order=quadrature_order, self_order=self_order
        ),
    }
    return study


def nodal_flux_field(
    mesh: QuadraticSurfaceMesh,
    quadrature_order: int = 8,
    self_order: int = 16,
) -> dict:
    """Computed versus closed-form nodal flux on the sphere (for the field map).

    The exact outward normal at a *node* of the unit sphere is the radial
    direction ``x / |x|``, so no mesh normal averaging is involved.

    Returns
    -------
    dict with ``nodes``, ``computed`` and ``exact`` arrays.
    """
    u = analytic_potential(mesh.vertices)
    result = solve_neumann_from_dirichlet_quadratic(
        mesh, u, quadrature_order=quadrature_order, self_order=self_order
    )
    radius = np.linalg.norm(mesh.vertices, axis=1)
    exact = analytic_flux(mesh.vertices, mesh.vertices / radius[:, None])
    return {
        "nodes": mesh.vertices,
        "computed": result.normal_derivative,
        "exact": exact,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def build_figure(study: dict, path: Path, field: dict | None = None) -> None:
    """Three-panel figure: flux convergence, operator identities, flux field map."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = study["levels"]
    h = np.array([rec["h"] for rec in levels])
    flux = np.array([rec["flux_rel_error"] for rec in levels])
    single = np.array([rec["single_layer_identity_error"] for rec in levels])
    solid = np.array([rec["solid_angle_identity_error"] for rec in levels])
    h_ref = np.array([h.min(), h.max()])

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16.0, 4.6))

    # (a) exact-flux convergence of the direct second-kind equation
    ax = fig.add_subplot(1, 3, 1)
    ax.loglog(h, flux, "o-", color="#1f77b4", label="quadratic panels (2nd-kind BIE)")
    ax.axhline(FLUX_ACCEPTANCE, color="#2ca02c", ls="--", lw=1.0, label="3% acceptance")
    ax.axhline(FLUX_HARD_LIMIT, color="#d62728", ls=":", lw=1.0, label="15% hard limit")
    scale = flux[-1] / h[-1] ** 2
    ax.loglog(h_ref, scale * h_ref ** 2, "k-.", lw=1.0, label=r"$O(h^2)$ reference")
    ax.set_xlabel("mesh size h [-]")
    ax.set_ylabel(r"relative $L_2$ flux error [-]")
    ax.set_title("Exact-flux convergence (unit sphere)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # (b) closed-form operator identities
    ax = fig.add_subplot(1, 3, 2)
    ax.loglog(h, single, "s-", color="#ff7f0e", label=r"$|S\mathbf{1}-\mathbf{1}|_\infty$")
    ax.loglog(
        h, solid, "^-", color="#9467bd", label=r"$|D\mathbf{1}+\frac{1}{2}|_\infty$"
    )
    scale2 = single[-1] / h[-1] ** 2
    scale3 = solid[-1] / h[-1] ** 3
    ax.loglog(h_ref, scale2 * h_ref ** 2, "k-.", lw=1.0, label=r"$O(h^2)$ reference")
    ax.loglog(h_ref, scale3 * h_ref ** 3, "k:", lw=1.0, label=r"$O(h^3)$ reference")
    ax.set_xlabel("mesh size h [-]")
    ax.set_ylabel("identity residual [-]")
    ax.set_title("Operator identities (closed-form values)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # (c) nodal flux deviation field on the sphere (Mollweide projection)
    ax = fig.add_subplot(1, 3, 3, projection="mollweide")
    if field is not None:
        nodes = np.asarray(field["nodes"])
        radius = np.linalg.norm(nodes, axis=1)
        lon = np.arctan2(nodes[:, 1], nodes[:, 0])
        lat = np.arcsin(np.clip(nodes[:, 2] / radius, -1.0, 1.0))
        deviation = 100.0 * (np.asarray(field["computed"]) - np.asarray(field["exact"]))
        deviation /= float(np.max(np.abs(field["exact"])))
        vmax = float(np.max(np.abs(deviation))) or 1.0
        sc = ax.scatter(
            lon, lat, c=deviation, cmap="coolwarm", s=6, vmin=-vmax, vmax=vmax
        )
        cb = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.09, shrink=0.85)
        cb.set_label("nodal $q_h - q_{exact}$ [% of max $|q|$]", fontsize=8)
    ax.set_title("Nodal flux deviation (finest mesh)")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "TensorBEM quadratic panel operator verification "
        f"(finest mesh: {levels[-1]['nodes']} nodes, {levels[-1]['elements']} elements)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=str, default=DEFAULT_JSON)
    parser.add_argument("--figure", type=str, default=DEFAULT_FIGURE)
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--quadrature-order", type=int, default=8)
    parser.add_argument("--self-order", type=int, default=16)
    args = parser.parse_args(argv)

    study = run_study(
        levels=tuple(args.levels),
        quadrature_order=args.quadrature_order,
        self_order=args.self_order,
    )

    field = study.pop("_finest_field", None)

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(study, indent=2) + "\n")

    if args.figure:
        build_figure(study, Path(args.figure), field=field)

    print("subdiv  nodes  elems      h     flux_err  |S1-1|   |D1+1/2|  cond(S)")
    for rec in study["levels"]:
        print(
            f"{rec['subdivisions']:6d} {rec['nodes']:6d} {rec['elements']:6d}"
            f" {rec['h']:.4f}  {rec['flux_rel_error']:.3e}"
            f"  {rec['single_layer_identity_error']:.3e}"
            f"  {rec['solid_angle_identity_error']:.3e}"
            f"  {rec['operator_condition_number']:8.2f}"
        )
    orders = study["observed_orders"]
    print(
        "observed orders: "
        f"flux={orders['flux_rel_error']:.2f} "
        f"|S1-1|={orders['single_layer_identity_error']:.2f} "
        f"|D1+1/2|={orders['solid_angle_identity_error']:.2f}"
    )
    acceptance = study["acceptance"]
    print(
        f"finest flux error {acceptance['flux_rel_error_finest']:.3e} "
        f"-> <3%: {acceptance['passed_3pct']}, <15%: {acceptance['passed_hard_limit']}"
    )
    print(f"JSON: {json_path}")
    if args.figure:
        print(f"figure: {args.figure}")
    return study


if __name__ == "__main__":  # pragma: no cover
    main()