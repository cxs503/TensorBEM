#!/usr/bin/env python3
"""Independent re-derivation of the CASE-BEM-RIGID-LID-HEMISPHERE reference.

Audit mission m17395 (verdict of m17338).  Nothing here imports the case module
``tensorbem.benchmarks.case_bem_rigid_lid_hemisphere``: the two routes below are
written from the BVP statement so that the checkpoint is independent of the
implementation being audited.

Physics (rigid-lid / zero-frequency endpoint of a floating hemisphere):

    Laplace in z>0 outside r=a,  dphi/dz = 0 on z=0 (rigid lid),
    dphi/dn = U mu on the wetted hemisphere (mu = cos(theta)),
    phi -> 0 at infinity.

Route R1 (series, analytic).  Even extension about z=0 gives a full sphere with
dphi/dr = -U|mu|; expanding in Legendre modes and projecting on the hemisphere
gives, with J_n = int_0^1 mu P_n(mu) dmu (J_n = 0 for odd n):

    A33/(rho*nabla) = 3 * sum_{n even} (2n+1) J_n^2 / (n+1).

    Odd extension (pressure-release plane, phi = 0) is the single mode
    A33/(rho*nabla) = 3 * J_1 = 0.5, used as an analytic control.

Route R2 (independent numerics).  Collocation boundary element on the wetted
panels with the rigid-lid Green function G = 1/|x-y| + 1/|x-y*| (y* = mirror in
z=0; same sign = rigid lid, opposite sign = pressure release).  A33 is taken
from the pressure integral A33 = -rho * int_S phi * n_z dS.  The code path is
self-validated on the closed-form full sphere (A33/(rho*nabla) = 0.5) and on the
pressure-release hemisphere (0.5) before the rigid-lid value is read.

Usage:  PYTHONPATH=src python scripts/audit_m17395_rigid_lid_hemisphere.py \
            [--outdir artifacts/m17395] [--levels 2 3]
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

RADIUS = 1.0
DENSITY = 1025.0
NABLA = (2.0 / 3.0) * math.pi * RADIUS**3
RHO_NABLA = DENSITY * NABLA

#: Value the audited case ships as the exact reference (results.json
#: reference.value_dimless) and the value the referee row names.
CASE_REFERENCE_DIMLESS = 0.8309512873124402
#: Exact digits quoted by the referee row (tensor_benchmark_cases.metric).
REFEREE_EXACT_DIMLESS = 0.830951320
#: Registered "got" of the referee row: 1024 panels, ka = 1e-3.
ROW_HEADLINE_DIMLESS = 0.833304


# ---------------------------------------------------------------------------
# route R1: series
# ---------------------------------------------------------------------------
def half_moments(degree_max: int) -> np.ndarray:
    """J_n = int_0^1 mu P_n(mu) dmu by exact (n+1)-point-free Gauss rule.

    mu*P_n is a polynomial of degree n+1, so n/2+2 Gauss-Legendre nodes on
    [0, 1] integrate it to machine precision.
    """
    moments = np.zeros(degree_max + 1)
    for n in range(degree_max + 1):
        order = n // 2 + 2
        x, w = np.polynomial.legendre.leggauss(order)
        mu = 0.5 * (x + 1.0)
        w = 0.5 * w
        p = np.polynomial.legendre.legval(mu, [0.0] * n + [1.0])
        moments[n] = float(np.dot(w, mu * p))
    return moments


def series_routes(degree_max: int = 240) -> dict:
    """Rigid-lid and pressure-release dimensionless added mass from the series."""
    J = half_moments(degree_max)
    even = np.arange(0, degree_max + 1, 2)
    terms = 3.0 * (2 * even + 1) * J[even] ** 2 / (even + 1)
    ladder = []
    for n_max in (20, 40, 60, 80, 120, degree_max):
        if n_max <= degree_max:
            sel = even <= n_max
            ladder.append({"degree_max": int(n_max), "dimless": float(terms[sel].sum())})
    return {
        "rigid_lid_dimless": float(terms.sum()),
        "rigid_lid_kg": float(terms.sum() * RHO_NABLA),
        "pressure_release_dimless": float(0.5 * (3.0 * J[1]) ** 2),
        "ladder": ladder,
        "n_modes": int(even.size),
        "moment_checks": {
            "J0_minus_1_2": float(abs(J[0] - 0.5)),
            "J1_minus_1_3": float(abs(J[1] - 1.0 / 3.0)),
            "J2_minus_1_8": float(abs(J[2] - 0.125)),
        },
    }


# ---------------------------------------------------------------------------
# route R2: independent collocation BEM with the rigid-lid image Green function
# ---------------------------------------------------------------------------
def _quad_points(order: int = 3):
    """3-point degree-2 symmetric rule on the reference triangle.

    The centroid point is deliberately excluded so that a collocation node at
    the panel centroid never coincides with a quadrature point; the three nodes
    lie in the panel plane, so the own-panel normal-derivative term vanishes
    identically and the jump is exactly the free term.  Weights sum to 1/2 (the
    reference area), so a physical panel of area A uses dS = 2*A*w.
    """
    bary = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
    wts = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    assert abs(wts.sum() - 0.5) < 1e-15
    return bary, wts


def image_bem(centroids, normals, areas, v0, v1, v2, sign: float, order: int = 3,
              jump: float = -0.5) -> dict:
    """Solve the single-layer problem with G = 1/|x-y| + sign/|x-y*|.

    sign = +1 rigid lid (dphi/dz = 0), sign = -1 pressure release (phi = 0),
    sign = 0 no mirror plane (closed body in unbounded fluid).
    Returns A33/(rho*nabla) from the pressure integral -rho*int phi n_z dS.
    """
    n = centroids.shape[0]
    bary, qw = _quad_points(order)
    mirror = np.array([1.0, 1.0, -1.0])

    def quad_of(j):
        return bary @ np.stack([v0[j], v1[j], v2[j]])

    op = np.zeros((n, n))
    phi = np.zeros(n)
    for j in range(n):
        y = quad_of(j)                       # (m,3)
        dS = 2.0 * areas[j] * qw              # (m,)
        dx = centroids[:, None, :] - y[None, :, :]
        r = np.maximum(np.linalg.norm(dx, axis=2), 1e-12)
        proj = (normals[:, None, :] * dx).sum(axis=2) / r**3
        projm = np.zeros_like(proj)
        if sign != 0.0:
            dxm = centroids[:, None, :] - (y * mirror)[None, :, :]
            rm = np.maximum(np.linalg.norm(dxm, axis=2), 1e-12)
            projm = (normals[:, None, :] * dxm).sum(axis=2) / rm**3
        op[:, j] = (-(proj + sign * projm) * dS[None, :]).sum(axis=1) / (4.0 * math.pi)
    # single layer: own-panel normal derivative vanishes identically (all nodes
    # in the panel plane) and the free term enters as -sigma/2 -- the sign is
    # fixed by the two closed-form controls below (sphere -> 1.0, pressure
    # release -> 0.5), never tuned on the audited quantity.
    np.fill_diagonal(op, 0.0)
    mat = op + jump * np.eye(n)
    rhs = normals[:, 2].copy()               # dphi/dn = U * n_z, U = 1
    sigma = np.linalg.solve(mat, rhs)

    for j in range(n):
        y = quad_of(j)
        dS = 2.0 * areas[j] * qw
        dx = centroids[:, None, :] - y[None, :, :]
        r = np.maximum(np.linalg.norm(dx, axis=2), 1e-12)
        kernel = 1.0 / r
        if sign != 0.0:
            rm = np.maximum(np.linalg.norm(centroids[:, None, :] - (y * mirror)[None, :, :], axis=2), 1e-12)
            kernel = kernel + sign / rm
        phi += sigma[j] * (kernel * dS[None, :]).sum(axis=1) / (4.0 * math.pi)

    a33 = -DENSITY * float((phi * normals[:, 2] * areas).sum())
    nabla = float(abs(np.sum(normals[:, 2] * centroids[:, 2] * areas)))
    return {
        "panels": int(n),
        "a33_kg": a33,
        "dimless": a33 / RHO_NABLA,
        "coeff_of_cubature_volume": nabla,
        "residual": float(np.max(np.abs(mat @ sigma - rhs))),
    }


def _arrays(mesh):
    """Body-outward normals (into the fluid) from the wetted-hull mesh."""
    return (
        mesh.centroids,
        mesh.normals,
        mesh.areas,
        mesh.v0,
        mesh.v1,
        mesh.v2,
    )


def sphere_self_test(level: int) -> dict:
    """Closed-form control: full sphere in unbounded fluid.

    A33/(rho*nabla_hemi) must be 1.0 because nabla_hemi = nabla_sphere/2 and
    A33 = 0.5 rho nabla_sphere.
    """
    from tensorbem.geometry3d import hemisphere_hull_mesh

    hemi = hemisphere_hull_mesh(level, radius=RADIUS)
    c, n, a, v0, v1, v2 = _arrays(hemi)
    lo = np.array([1.0, 1.0, -1.0])
    res = image_bem(
        np.vstack([c, c * lo]),
        np.vstack([n, n * lo]),
        np.concatenate([a, a]),
        np.vstack([v0, v0 * lo]),
        np.vstack([v1, v1 * lo]),
        np.vstack([v2, v2 * lo]),
        sign=0.0,
    )
    res["level"] = int(level)
    res["expected_dimless"] = 1.0
    return res


def hemisphere_panels(level: int) -> dict:
    from tensorbem.geometry3d import hemisphere_hull_mesh

    m = hemisphere_hull_mesh(level, radius=RADIUS)
    c, n, a, v0, v1, v2 = _arrays(m)
    rigid = image_bem(c, n, a, v0, v1, v2, sign=+1.0)
    release = image_bem(c, n, a, v0, v1, v2, sign=-1.0)
    rigid["level"] = int(level)
    rigid["pressure_release_dimless"] = release["dimless"]
    rigid["mesh_area_rel_error"] = float(m.surface_area / (2 * math.pi) - 1.0)
    return rigid


def hemisphere_panels(level: int) -> dict:
    from tensorbem.geometry3d import hemisphere_hull_mesh

    m = hemisphere_hull_mesh(level, radius=RADIUS)
    rigid = image_bem(m.centroids, m.normals, m.areas, m.v0, m.v1, m.v2, sign=+1.0)
    release = image_bem(m.centroids, m.normals, m.areas, m.v0, m.v1, m.v2, sign=-1.0)
    rigid["level"] = int(level)
    rigid["pressure_release_dimless"] = release["dimless"]
    rigid["mesh_area_rel_error"] = float(m.surface_area / (2 * math.pi) - 1.0)
    return rigid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="artifacts/m17395")
    ap.add_argument("--levels", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--degree-max", type=int, default=240)
    args = ap.parse_args(argv)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    r1 = series_routes(args.degree_max)
    r2 = {"sphere_selftest": [], "hemisphere": []}
    for level in args.levels:
        st = sphere_self_test(level)
        st["level"] = int(level)
        r2["sphere_selftest"].append(st)
        r2["hemisphere"].append(hemisphere_panels(level))

    payload = {
        "audit": "m17395 independent re-derivation of CASE-BEM-RIGID-LID-HEMISPHERE",
        "conventions": {
            "radius_m": RADIUS,
            "density": DENSITY,
            "nabla_m3": NABLA,
            "bem": "single-layer collocation, image Green function, pressure integral",
        },
        "route_R1_series": r1,
        "route_R2_image_bem": r2,
        "case_ships": CASE_REFERENCE_DIMLESS,
        "referee_exact": REFEREE_EXACT_DIMLESS,
        "row_headline_1024": ROW_HEADLINE_DIMLESS,
        "deltas": {
            "R1_vs_case_ships": float(r1["rigid_lid_dimless"] / CASE_REFERENCE_DIMLESS - 1.0),
            "R1_vs_referee_exact": float(r1["rigid_lid_dimless"] / REFEREE_EXACT_DIMLESS - 1.0),
            "case_ships_vs_referee_exact": float(CASE_REFERENCE_DIMLESS / REFEREE_EXACT_DIMLESS - 1.0),
            "row_headline_vs_referee_exact": float(ROW_HEADLINE_DIMLESS / REFEREE_EXACT_DIMLESS - 1.0),
            "r2_hemisphere_vs_case_ships": [
                {
                    "level": row["level"],
                    "dimless": row["dimless"],
                    "rel_error": float(row["dimless"] / CASE_REFERENCE_DIMLESS - 1.0),
                }
                for row in r2["hemisphere"]
            ],
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
        },
    }
    path = out / "independent_reference_check.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")

    print("R1 series rigid lid   : %.9f  (=%.4f kg)" % (r1["rigid_lid_dimless"], r1["rigid_lid_kg"]))
    print("R1 series press.release: %.9f (analytic control 0.5)" % r1["pressure_release_dimless"])
    print("case ships            : %.9f (delta %.2e)" % (CASE_REFERENCE_DIMLESS, payload["deltas"]["R1_vs_case_ships"]))
    for st in r2["sphere_selftest"]:
        print("R2 full sphere lvl=%d: dimless %.6f (analytic 1.0 = 0.5 rho*nabla_sphere)" % (st["level"], st["dimless"]))
    for row in r2["hemisphere"]:
        print(
            "R2 hemisphere lvl=%d panels=%4d: dimless %.6f (rho*nabla err %+.3f%%), "
            "pressure-release control %.6f, mesh area err %+.3f%%"
            % (
                row["level"],
                row["panels"],
                row["dimless"],
                100.0 * (row["dimless"] / CASE_REFERENCE_DIMLESS - 1.0),
                row["pressure_release_dimless"],
                100.0 * row["mesh_area_rel_error"],
            )
        )
    print("artifact:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())