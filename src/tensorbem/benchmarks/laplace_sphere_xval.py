"""Cross-validation of the exterior-Laplace boundary-element kernel against OSS.

The cathodic-protection (CP) module (:mod:`tensorbem.cathodic_protection`) rests
on one numerical kernel: the exterior Laplace single-layer BIE assembled by
:func:`tensorbem.solver3d.assemble_surface_matrices`.  Its only closed-form
anchor so far is the *uniformly* polarised conducting sphere, for which the
surface density is constant -- a test that exercises the self-term and the
geometry but not the off-diagonal quadrature, because a constant density makes
the panel-to-panel coupling error average out.

This module adds the round-2 method-selection cross-validation (XVAL) for that
kernel with three independent references on **one** non-trivial analytic case:

Case (dimensionless, sphere of radius ``a`` in an unbounded electrolyte of
conductivity ``sigma``)::

    Dirichlet data   phi(a, mu) = phi_a * (1 + mu),      mu = cos(theta)
    Exact solution   phi(r, mu) = phi_a * (a/r) + phi_a * (a/r)**2 * mu
    Exact density    q(a, mu)   = phi_a * (1 + 2 mu) / a
    Exact current    I          = 4 pi sigma a phi_a     (monopole only)

so the total current keeps the classical spreading-resistance form
``R = 1/(4 pi sigma a)`` while the *field*, the *surface density* and the
panel-to-panel coupling are all non-trivial.  Both Legendre modes (l = 0, 1)
are present; the l = 1 mode is what a quadrature defect corrupts first.

References implemented (each optional, gracefully skipped when the dependency
is absent):

``tensorbem``
    :func:`tensorbem.solver3d.assemble_surface_matrices` -- piecewise-constant
    centroid collocation, exact polar self-term, exact solid-angle double layer,
    Duffy order-``n`` off-diagonal quadrature.
``capytaine``
    Mature OSS BEM (Delhommeau Green function, compiled Nemoh core).  With
    ``free_surface=inf`` and ``wavenumber=0`` its Green function degenerates to
    the Rankine kernel ``-1/(4 pi |x-y|)``, so its influence matrices can be
    compared *element by element* with the TensorBEM assembly on the identical
    triangle mesh, and can drive the same single-layer BIE independently.
``skfem``
    scikit-fem (OSS Galerkin finite elements).  Because the data depends only on
    ``mu``, the 3-D exterior problem reduces exactly to the axisymmetric
    problem in ``(r, mu)``::

        int [ r^2 d_r phi d_r v + (1 - mu^2) d_mu phi d_mu v ] dr dmu = 0

    on ``[a, R] x [-1, 1]`` with exact Dirichlet data on both ends.  This is a
    different discretisation *family* (volume Galerkin vs boundary integral), so
    agreement is evidence about the formulation and not only about quadrature.

The module never imports the optional libraries at module import time; every
route reports ``available: False`` with a reason instead of raising.
"""

from __future__ import annotations

import importlib.util
import math
from typing import Any

import numpy as np

from ..geometry3d import TriangleMesh, icosphere
from ..kernels3d import duffy_quadrature_batch
from ..solver3d import assemble_surface_matrices, solve_laplace_surface

__all__ = [
    "DEFAULT_PHI_A",
    "DEFAULT_RADIUS",
    "DEFAULT_SIGMA",
    "exact_current",
    "exact_density",
    "exact_potential",
    "panel_mu",
    "probe_points",
    "representation_potential",
    "run_xval",
    "route_capytaine",
    "route_skfem",
    "route_tensorbem",
]
# Seawater at 15 deg C (DNV-RP-B401 design value), same constant as the CP module.
DEFAULT_SIGMA = 3.5
DEFAULT_RADIUS = 1.0
DEFAULT_PHI_A = -1.0


# ---------------------------------------------------------------------------
# analytic reference
# ---------------------------------------------------------------------------
def exact_potential(points: np.ndarray, radius: float = DEFAULT_RADIUS,
                    phi_a: float = DEFAULT_PHI_A) -> np.ndarray:
    """Exact exterior potential for ``phi(a, mu) = phi_a (1 + mu)``."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    r = np.linalg.norm(pts, axis=1)
    mu = np.divide(pts[:, 2], r, out=np.zeros_like(r), where=r > 0.0)
    ar = radius / r
    return phi_a * ar + phi_a * ar**2 * mu


def exact_density(mu: np.ndarray, radius: float = DEFAULT_RADIUS,
                  phi_a: float = DEFAULT_PHI_A) -> np.ndarray:
    """Exact single-layer density ``q = phi_a (1 + 2 mu) / a`` (units V/m)."""
    return phi_a * (1.0 + 2.0 * np.asarray(mu, dtype=float)) / radius


def exact_current(radius: float = DEFAULT_RADIUS, phi_a: float = DEFAULT_PHI_A,
                  sigma: float = DEFAULT_SIGMA) -> float:
    """Exact total current ``I = 4 pi sigma a phi_a`` (A)."""
    return 4.0 * math.pi * sigma * radius * phi_a


def panel_mu(mesh: TriangleMesh, radius: float = DEFAULT_RADIUS) -> np.ndarray:
    """``cos(theta)`` of every panel centroid (panel data is constant per panel)."""
    return mesh.centroids[:, 2] / radius


def probe_points(radius: float = DEFAULT_RADIUS, radii_ratio=(1.5, 2.0, 4.0),
                 angles_deg=(20.0, 50.0, 90.0)) -> np.ndarray:
    """Axisymmetric probe cloud in the x-z half plane (deterministic)."""
    pts = []
    for ratio in radii_ratio:
        for ang in angles_deg:
            t = math.radians(ang)
            pts.append([radius * ratio * math.sin(t), 0.0, radius * ratio * math.cos(t)])
    return np.asarray(pts, dtype=float)


# ---------------------------------------------------------------------------
# shared post-processing
# ---------------------------------------------------------------------------
def representation_potential(
    mesh: TriangleMesh,
    density: np.ndarray,
    boundary_values: np.ndarray,
    points: np.ndarray,
    quadrature_order: int = 10,
) -> np.ndarray:
    """Evaluate the exterior potential from the representation formula.

    ``phi(x) = int_S G(x,y) q(y) dS + int_S phi(y) dG/dn_body(x,y) dS`` with
    ``G = 1/(4 pi |x-y|)`` and ``q = -dphi/dn_body``.  The double-layer sign is
    the one that reproduces a pure ``l = 1`` (dipole) mode, see
    :func:`tests.test_laplace_sphere_xval.test_representation_formula_dipole_sign`;
    it is *invisible* for uniform (``l = 0``) boundary data because the
    double-layer of a constant vanishes outside a closed surface.  Used
    identically for every route so that the comparison isolates the boundary
    solution, not the post-processing.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    v0, v1, v2, normals = mesh.v0, mesh.v1, mesh.v2, mesh.normals
    qp, qw = duffy_quadrature_batch(v0, v1, v2, order=quadrature_order)
    out = np.empty(len(pts), dtype=float)
    for m, x in enumerate(pts):
        delta = x[None, None, :] - qp                      # (F, Q, 3)
        dist = np.linalg.norm(delta, axis=2)               # (F, Q)
        green = np.sum(qw / (4.0 * math.pi * dist), axis=1)          # int G dS
        # int (x-y).n / (4 pi |x-y|^3) dS == int dG/dn_body dS with G = 1/(4 pi r)
        dgreen = np.sum(qw * np.einsum("fqi,fi->fq", delta, normals)
                        / (4.0 * math.pi * dist**3), axis=1)
        # exterior representation: phi = int G q dS + int phi dG/dn_body dS
        out[m] = np.dot(green, density) + np.dot(dgreen, boundary_values)
    return out


def _surface_current(mesh: TriangleMesh, density: np.ndarray, sigma: float) -> float:
    """``I = sigma * int_S q dS`` (Ohm's law applied at post-processing)."""
    return float(sigma * np.sum(density * mesh.areas))


def _rel(a: float, b: float) -> float:
    return float(abs(a - b) / abs(b)) if b != 0.0 else float(abs(a))


def _level_metrics(
    mesh: TriangleMesh,
    density: np.ndarray,
    current: float,
    probes: np.ndarray,
    *,
    radius: float,
    phi_a: float,
    sigma: float,
) -> dict[str, Any]:
    mu = panel_mu(mesh, radius)
    q_exact = exact_density(mu, radius, phi_a)
    phi_probe = representation_potential(
        mesh, density, phi_a * (1.0 + mu), probes
    )
    phi_exact = exact_potential(probes, radius, phi_a)
    i_exact = exact_current(radius, phi_a, sigma)
    scale = float(np.max(np.abs(q_exact)))
    return {
        "panels": int(mesh.panel_count),
        "h_over_a": float(math.sqrt(4.0 * math.pi * radius**2 / mesh.panel_count) / radius),
        "current_A": float(current),
        "exact_current_A": float(i_exact),
        "current_rel_error": _rel(current, i_exact),
        "dimensionless_current": float(current / i_exact),
        "density_linf_rel_error": float(np.max(np.abs(density - q_exact)) / scale),
        "density_l2_rel_error": float(np.linalg.norm(density - q_exact) / np.linalg.norm(q_exact)),
        "probe_potential_linf_rel_error": float(
            np.max(np.abs(phi_probe - phi_exact)) / np.max(np.abs(phi_exact))
        ),
        "probe_potential_l2_rel_error": float(
            np.linalg.norm(phi_probe - phi_exact) / np.linalg.norm(phi_exact)
        ),
        "probe_phi": phi_probe.tolist(),
        "probe_r_over_a": (np.linalg.norm(probes, axis=1) / radius).tolist(),
    }


def _observed_order(levels: list[dict[str, Any]], key: str) -> list[float]:
    """Observed order ``p = log(e_i / e_{i+1}) / log(N_{i+1} / N_i)`` per level pair."""
    orders = []
    for lower, upper in zip(levels[:-1], levels[1:]):
        e0, e1 = lower[key], upper[key]
        n0, n1 = lower["panels"], upper["panels"]
        if e0 <= 0.0 or e1 <= 0.0 or n1 <= n0:
            orders.append(float("nan"))
            continue
        orders.append(float(math.log(e0 / e1) / math.log(n1 / n0)))
    return orders


# ---------------------------------------------------------------------------
# route 1: TensorBEM
# ---------------------------------------------------------------------------
def route_tensorbem(
    subdivisions=(1, 2, 3, 4),
    radius: float = DEFAULT_RADIUS,
    phi_a: float = DEFAULT_PHI_A,
    sigma: float = DEFAULT_SIGMA,
    quadrature_order: int = 8,
) -> dict[str, Any]:
    """Exterior BIE with the TensorBEM assembly + piecewise-constant collocation."""
    probes = probe_points(radius)
    levels = []
    for sub in subdivisions:
        mesh = icosphere(sub, radius)
        mu = panel_mu(mesh, radius)
        values = phi_a * (1.0 + mu)
        res = solve_laplace_surface(
            mesh, values, domain="exterior", quadrature_order=quadrature_order
        )
        current = _surface_current(mesh, res.normal_derivative, sigma)
        level = _level_metrics(
            mesh, res.normal_derivative, current, probes,
            radius=radius, phi_a=phi_a, sigma=sigma,
        )
        level.update({
            "subdivisions": int(sub),
            "quadrature_order": int(quadrature_order),
            "condition_number": float(res.condition_number),
            "relative_residual_l2": float(res.relative_residual_l2),
            "double_layer_row_sum_error": float(res.double_layer_row_sum_error),
        })
        levels.append(level)
    return {
        "route": "tensorbem",
        "available": True,
        "description": "piecewise-constant centroid collocation, exact polar "
                       "self-term + solid-angle double layer + Duffy off-diagonal",
        "quadrature_order": int(quadrature_order),
        "levels": levels,
    }


# ---------------------------------------------------------------------------
# route 2: capytaine (OSS BEM, Rankine limit)
# ---------------------------------------------------------------------------
def _capytaine_module():
    if importlib.util.find_spec("capytaine") is None:
        return None
    import capytaine  # noqa: WPS433 (lazy optional dependency)
    return capytaine


def capytaine_matrices(mesh: TriangleMesh):
    """Capytaine influence matrices for the Rankine (no-free-surface) limit.

    ``free_surface=inf`` and ``wavenumber=0`` remove every free-surface term, so
    the Delhommeau Green function reduces to the Rankine kernel and the returned
    ``(S, K)`` matrices are directly comparable with
    :func:`tensorbem.solver3d.assemble_surface_matrices`.

    Returns ``(S, K, info)`` with capytaine's own sign convention
    (``G = -1/(4 pi r)``), i.e. ``A_tensorbem = -S`` and ``D_tensorbem = -K``
    once capytaine's double-layer diagonal (its ``1/2`` jump) is removed.
    """
    cpy = _capytaine_module()
    if cpy is None:
        raise ImportError("capytaine is not installed")
    tris = mesh.vertices[mesh.faces]
    faces = [list(map(list, tri)) for tri in tris]
    cmesh = cpy.Mesh.from_list_of_faces(faces)
    engine = cpy.DefaultMatrixEngine()
    single, double = engine.build_matrices(
        cmesh,
        cmesh,
        free_surface=np.inf,
        water_depth=np.inf,
        wavenumber=0.0,
        adjoint_double_layer=False,
    )
    single = np.asarray(single, dtype=float)
    double = np.asarray(double, dtype=float)
    info = {
        "capytaine_version": getattr(cpy, "__version__", "unknown"),
        "green_function": "Delhommeau (Rankine limit: free_surface=inf, wavenumber=0)",
        "engine": type(engine).__name__,
        "mesh_panels": int(cmesh.nb_faces),
    }
    return single, double, info


def route_capytaine(
    subdivisions=(1, 2, 3, 4),
    radius: float = DEFAULT_RADIUS,
    phi_a: float = DEFAULT_PHI_A,
    sigma: float = DEFAULT_SIGMA,
    quadrature_order: int = 8,
) -> dict[str, Any]:
    """Same single-layer BIE driven by capytaine's influence matrices."""
    if _capytaine_module() is None:
        return {"route": "capytaine", "available": False,
                "reason": "python package 'capytaine' not importable"}
    probes = probe_points(radius)
    levels = []
    matrix_comparison = []
    info = {}
    for sub in subdivisions:
        mesh = icosphere(sub, radius)
        mu = panel_mu(mesh, radius)
        values = phi_a * (1.0 + mu)
        s_capy, k_capy, info = capytaine_matrices(mesh)
        single = -s_capy
        double = -k_capy.copy()
        np.fill_diagonal(double, 0.0)
        rhs = 0.5 * values - double @ values
        density = np.linalg.solve(single, rhs)
        current = _surface_current(mesh, density, sigma)
        level = _level_metrics(
            mesh, density, current, probes,
            radius=radius, phi_a=phi_a, sigma=sigma,
        )
        level.update({"subdivisions": int(sub),
                      "condition_number": float(np.linalg.cond(single))})

        # element-wise assembly comparison on the identical triangle mesh
        a_ref, d_ref = assemble_surface_matrices(mesh, quadrature_order=quadrature_order)
        matrix_comparison.append({
            "subdivisions": int(sub),
            "panels": int(mesh.panel_count),
            "single_fro_rel": float(np.linalg.norm(single - a_ref) / np.linalg.norm(a_ref)),
            "single_max_abs": float(np.max(np.abs(single - a_ref))),
            "single_max_abs_rel_to_scale": float(
                np.max(np.abs(single - a_ref)) / np.max(np.abs(a_ref))
            ),
            "double_fro_rel": float(np.linalg.norm(double - d_ref) / np.linalg.norm(d_ref)),
            "double_max_abs_rel_to_scale": float(
                np.max(np.abs(double - d_ref)) / max(np.max(np.abs(d_ref)), 1e-30)
            ),
            "diagonal_single_max_rel": float(
                np.max(np.abs(np.diag(single) - np.diag(a_ref)) / np.abs(np.diag(a_ref)))
            ),
        })
        levels.append(level)
    return {
        "route": "capytaine",
        "available": True,
        "description": "Delhommeau influence matrices in the Rankine limit, "
                       "driving the identical single-layer BIE",
        "engine_info": info,
        "levels": levels,
        "matrix_comparison": matrix_comparison,
    }


# ---------------------------------------------------------------------------
# route 3: scikit-fem axisymmetric Galerkin
# ---------------------------------------------------------------------------
def _skfem_modules():
    if importlib.util.find_spec("skfem") is None:
        return None
    import skfem  # noqa: WPS433
    return skfem


def _axisymmetric_mesh(nr: int, nmu: int, radius: float, outer_radius: float):
    """Structured triangular mesh of ``[a, R] x [-1, 1]`` (axisymmetric domain)."""
    rs = np.linspace(radius, outer_radius, nr + 1)
    mus = np.linspace(-1.0, 1.0, nmu + 1)
    rr, mm = np.meshgrid(rs, mus, indexing="ij")
    nodes = np.vstack([rr.ravel(), mm.ravel()])
    cells = []
    for i in range(nr):
        for j in range(nmu):
            n0 = i * (nmu + 1) + j
            n1 = n0 + (nmu + 1)
            cells.append([n0, n1, n1 + 1])
            cells.append([n0, n1 + 1, n0 + 1])
    return nodes, np.asarray(cells, dtype=np.int64).T


def route_skfem(
    radius: float = DEFAULT_RADIUS,
    phi_a: float = DEFAULT_PHI_A,
    sigma: float = DEFAULT_SIGMA,
    outer_ratios=(2.0, 4.0, 8.0),
    resolutions=((16, 12), (32, 24), (64, 48)),
    order: int = 1,
    probe_ratios=(1.5, 2.0, 3.0),
    probe_angles_deg=(20.0, 50.0, 90.0),
) -> dict[str, Any]:
    """Axisymmetric Galerkin reference (scikit-fem), exact Dirichlet far field.

    The 3-D exterior problem with data depending only on ``mu`` reduces exactly
    to the axisymmetric weak form

        int [ r^2 d_r phi d_r v + (1 - mu^2) d_mu phi d_mu v ] dr dmu = 0

    on ``[a, R] x [-1, 1]`` (the ``2 pi`` factor cancels).  Both ends carry the
    exact Dirichlet data, so uniqueness makes the exact solution the unique
    solution of the truncated problem and the FEM error is a pure
    discretisation error.  ``R/a`` is a *modelling* knob, not a discretisation
    knob: it is swept to show that the reference is insensitive to it.
    """
    skfem = _skfem_modules()
    if skfem is None:
        return {"route": "skfem", "available": False,
                "reason": "python package 'skfem' not importable"}
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    import skfem as _sf
    from skfem import Basis, BilinearForm, ElementTriP1, ElementTriP2, FacetBasis, Functional, asm
    from skfem.helpers import dot, grad

    element = ElementTriP1() if order == 1 else ElementTriP2()

    @BilinearForm
    def laplace(u, v, w):
        r = w.x[0]
        mu = w.x[1]
        gu = grad(u)
        gv = grad(v)
        return r * r * gu[0] * gv[0] + (1.0 - mu * mu) * gu[1] * gv[1]

    @Functional
    def radial_gradient(w):
        # ``-grad(phi) . n`` is d_r(phi) because the domain-outward facet normal
        # is -e_r on r = a and +e_r on r = R.
        return -dot(grad(w["u"]), w.n)

    def _bd_data(r, mu):
        return phi_a * (radius / r) + phi_a * (radius / r) ** 2 * mu

    # axisymmetric probes: (r, mu) pairs, r kept strictly inside [a, R]
    probe_pts = []
    for ratio in probe_ratios:
        for ang in probe_angles_deg:
            probe_pts.append([radius * ratio, math.cos(math.radians(ang))])
    probe_pts = np.asarray(probe_pts).T
    i_exact = exact_current(radius, phi_a, sigma)

    levels = []
    for outer_ratio in outer_ratios:
        outer = outer_ratio * radius
        for nr, nmu in resolutions:
            nodes, cells = _axisymmetric_mesh(nr, nmu, radius, outer)
            mesh = _sf.MeshTri(nodes, cells)
            basis = Basis(mesh, element)
            A = asm(laplace, basis).tocsr()

            inner_dofs = basis.get_dofs(
                lambda p: np.isclose(p[0], radius, rtol=0.0, atol=1e-12)
            ).flatten()
            outer_dofs = basis.get_dofs(
                lambda p: np.isclose(p[0], outer, rtol=0.0, atol=1e-12)
            ).flatten()
            fixed = np.unique(np.concatenate([inner_dofs, outer_dofs]))
            u = np.zeros(basis.N)
            doflocs = basis.doflocs
            u[inner_dofs] = _bd_data(doflocs[0][inner_dofs], doflocs[1][inner_dofs])
            u[outer_dofs] = _bd_data(doflocs[0][outer_dofs], doflocs[1][outer_dofs])

            free = np.setdiff1d(np.arange(basis.N), fixed)
            rhs = -(A[free][:, fixed] @ u[fixed])
            u[free] = spla.spsolve(A[free][:, free].tocsc(), rhs)

            inner_facets = mesh.facets_satisfying(
                lambda p: np.isclose(p[0], radius, rtol=0.0, atol=1e-12)
            )
            fbi = FacetBasis(mesh, element, facets=inner_facets)
            dphi_dr_inner = radial_gradient.assemble(fbi, u=u)     # int d_r phi dmu
            current = -sigma * 2.0 * math.pi * radius**2 * dphi_dr_inner

            outer_facets = mesh.facets_satisfying(
                lambda p: np.isclose(p[0], outer, rtol=0.0, atol=1e-12)
            )
            fbo = FacetBasis(mesh, element, facets=outer_facets)
            dphi_dr_outer = radial_gradient.assemble(fbo, u=u)     # int -d_r phi dmu
            current_outer = sigma * 2.0 * math.pi * outer**2 * dphi_dr_outer

            # probes must lie strictly inside the truncated domain
            keep = probe_pts[0] < 0.95 * outer
            if not np.any(keep):
                continue
            probe_matrix = basis.probes(probe_pts[:, keep])
            phi_num = np.asarray(probe_matrix @ u).ravel()
            phi_ref = _bd_data(probe_pts[0, keep], probe_pts[1, keep])
            levels.append({
                "outer_ratio": float(outer_ratio),
                "nr": int(nr), "nmu": int(nmu), "order": int(order),
                "elements": int(mesh.t.shape[1]),
                "dofs": int(basis.N),
                "current_A": float(current),
                "current_outer_sphere_A": float(current_outer),
                "current_conservation_rel": _rel(current_outer, current),
                "exact_current_A": float(i_exact),
                "current_rel_error": _rel(current, i_exact),
                "dimensionless_current": float(current / i_exact),
                "probe_potential_linf_rel_error": float(
                    np.max(np.abs(phi_num - phi_ref)) / np.max(np.abs(phi_ref))
                ),
                "probe_potential_l2_rel_error": float(
                    np.linalg.norm(phi_num - phi_ref) / np.linalg.norm(phi_ref)
                ),
                "probe_phi": phi_num.tolist(),
            })
    richardson = []
    for outer_ratio in outer_ratios:
        series = [l for l in levels if l["outer_ratio"] == outer_ratio]
        if len(series) < 2:
            continue
        fine, coarse = series[-1], series[-2]
        e_fine = abs(fine["current_A"] - fine["exact_current_A"])
        e_coarse = abs(coarse["current_A"] - coarse["exact_current_A"])
        p = (math.log(e_coarse / e_fine) / math.log(2.0)) if e_fine > 0 else float("nan")
        factor = (2.0**p - 1.0) if p > 0.2 else float("nan")
        extrapolated = fine["current_A"] + (fine["current_A"] - coarse["current_A"]) / factor
        richardson.append({
            "outer_ratio": float(outer_ratio),
            "observed_order_current": float(p),
            "richardson_current_A": float(extrapolated),
            "richardson_rel_error": _rel(extrapolated, i_exact),
        })
    return {
        "route": "skfem",
        "available": True,
        "richardson_current": richardson,
        "description": "axisymmetric P%d Galerkin in (r, mu) with exact Dirichlet "
                       "data on the truncation sphere" % order,
        "order": int(order),
        "scikit_fem_version": getattr(_sf, "__version__", "unknown"),
        "levels": levels,
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_xval(**kwargs: Any) -> dict[str, Any]:
    """Run every available route and return a JSON-serialisable comparison."""
    radius = kwargs.pop("radius", DEFAULT_RADIUS)
    phi_a = kwargs.pop("phi_a", DEFAULT_PHI_A)
    sigma = kwargs.pop("sigma", DEFAULT_SIGMA)
    subdivisions = kwargs.pop("subdivisions", (1, 2, 3, 4))
    quadrature_orders = kwargs.pop("quadrature_orders", (4, 6, 8, 12))
    fem_orders = kwargs.pop("fem_orders", (1, 2))
    fem_outer_ratios = kwargs.pop("fem_outer_ratios", (2.0, 4.0, 8.0))
    fem_resolutions = kwargs.pop("fem_resolutions", ((16, 12), (32, 24), (64, 48)))

    base = {"radius_m": radius, "phi_a_V": phi_a, "sigma_S_per_m": sigma,
            "exact_current_A": exact_current(radius, phi_a, sigma),
            "dimensionless_current_target": 1.0}

    bem = route_tensorbem(subdivisions, radius, phi_a, sigma)
    capy = route_capytaine(subdivisions, radius, phi_a, sigma)
    fem = {f"order{o}": route_skfem(radius, phi_a, sigma,
                                    outer_ratios=fem_outer_ratios,
                                    resolutions=fem_resolutions, order=o)
           for o in fem_orders}
    quadrature_sensitivity = []
    sensitivity_subdivisions = kwargs.pop("sensitivity_subdivisions", subdivisions[-2:])
    for order in quadrature_orders:
        route = route_tensorbem(sensitivity_subdivisions, radius, phi_a, sigma,
                               quadrature_order=order)
        quadrature_sensitivity.append({
            "quadrature_order": int(order),
            "finest_current_rel_error": route["levels"][-1]["current_rel_error"],
            "finest_density_linf_rel_error": route["levels"][-1]["density_linf_rel_error"],
            "finest_probe_l2_rel_error": route["levels"][-1]["probe_potential_l2_rel_error"],
        })
    return {
        "case": "exterior Laplace, sphere with phi = phi_a (1 + cos theta)",
        **base,
        "tensorbem": bem,
        "capytaine": capy,
        "skfem": fem,
        "quadrature_sensitivity": quadrature_sensitivity,
        "tensorbem_observed_order": {
            "current": _observed_order(bem["levels"], "current_rel_error"),
            "density_linf": _observed_order(bem["levels"], "density_linf_rel_error"),
            "probe_l2": _observed_order(bem["levels"], "probe_potential_l2_rel_error"),
        },
    }