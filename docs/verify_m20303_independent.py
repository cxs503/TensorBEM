"""Independent re-derivation of the CASE-BEM-LAPLACE-CYLINDER numbers (m20303).

Nothing here imports ``tensorbem.benchmarks.case_cylinder_stream``: the reference
flow, the exterior multipole solution and the panel routes are re-built from
:mod:`tensorbem.geometry` / :mod:`tensorbem.solver` and plain numpy so that a bug
in the case module cannot hide behind its own reference.

Checks
------
1. the committed probe table is the complex potential ``W = U(z + a^2/z)``
   (pure numpy, complex arithmetic, re-evaluated);
2. an *independent discretisation* -- exterior multipole least squares on the
   polygon boundary (no panels, no Green's function) -- reproduces the cylinder
   field, which separates "geometry/faceting residual" from "panel error";
3. the committed route D / route N error metrics are reproduced from scratch;
4. the committed near-wall table is reproduced from scratch.

Usage: PYTHONPATH=src python3 docs/verify_m20303_independent.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tensorbem.geometry import regular_polygon
from tensorbem.solver import (
    evaluate_exterior_field,
    solve_laplace_exterior_dirichlet,
    solve_laplace_exterior_neumann,
)

RADIUS = 1.0
SPEED = 1.0
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/verified/case_bem_laplace_cylinder/results.json"


def dWdz(points: np.ndarray) -> np.ndarray:
    z = points[:, 0] + 1j * points[:, 1]
    dW = SPEED * (1.0 - RADIUS**2 / z**2)
    return np.column_stack((dW.real, -dW.imag))


def phi_perturbation(points: np.ndarray) -> np.ndarray:
    r = np.hypot(points[:, 0], points[:, 1])
    return SPEED * RADIUS**2 * points[:, 0] / r**2


def multipole_field(mesh, data: np.ndarray, points: np.ndarray, modes: int = 6):
    """Exterior multipole least squares: phi = sum_k a_k Re(1/z^k), no panels."""
    def basis(pts: np.ndarray) -> np.ndarray:
        z = pts[:, 0] + 1j * pts[:, 1]
        return np.column_stack([np.real(z ** -k) for k in range(1, modes + 1)])

    def basis_grad(pts: np.ndarray) -> np.ndarray:
        z = pts[:, 0] + 1j * pts[:, 1]
        columns = []
        for k in range(1, modes + 1):
            dz = -k * z ** (-(k + 1))
            columns.append(np.column_stack((dz.real, -dz.imag)))
        return np.stack(columns, axis=1)

    matrix = basis(mesh.centers)
    coefficients, *_ = np.linalg.lstsq(matrix, data, rcond=None)
    multipole_field.last_fit = {  # type: ignore[attr-defined]
        "modes": modes,
        "condition_number": float(np.linalg.cond(matrix)),
        "relative_fit_residual": float(
            np.linalg.norm(matrix @ coefficients - data) / np.linalg.norm(data)
        ),
    }
    potential = basis(points) @ coefficients
    gradient = np.einsum("mkc,k->mc", basis_grad(points), coefficients)
    return potential, gradient


def relative_error(computed: np.ndarray, exact: np.ndarray) -> float:
    return float(
        np.linalg.norm(computed - exact) / np.linalg.norm(exact)
    )


def main() -> int:
    payload = json.loads(RESULTS.read_text())
    report: dict[str, object] = {"source_artifact": str(RESULTS.relative_to(ROOT))}

    # 1. probe table is the complex potential ---------------------------------
    table = payload["reference"]["probe_table"]
    points = np.array(
        [[row["x_over_a"], row["y_over_a"]] for row in table], dtype=float
    ) * RADIUS
    table_u = np.array(
        [[row["u_over_U"], row["v_over_U"]] for row in table], dtype=float
    ) * SPEED
    table_phi = np.array(
        [row["perturbation_potential_over_Ua"] for row in table], dtype=float
    ) * SPEED * RADIUS
    exact_u = dWdz(points)
    report["probe_table_velocity_max_abs_diff"] = float(
        np.max(np.abs(table_u - exact_u))
    )
    report["probe_table_potential_max_abs_diff"] = float(
        np.max(np.abs(table_phi - phi_perturbation(points)))
    )

    # 2. independent discretisation: multipole least squares ----------------
    budget = int(payload["parameters"]["budget_panels"])
    mesh = regular_polygon(budget, RADIUS)
    data = phi_perturbation(mesh.centers)
    probe = points
    potential, gradient = multipole_field(mesh, data, probe)
    report["multipole_fit"] = multipole_field.last_fit  # type: ignore[attr-defined]
    # the multipole fit carries the *perturbation* potential: add U x_hat for
    # the total velocity, exactly as the panel routes do.
    report["multipole_velocity_relative_l2"] = relative_error(
        gradient + np.array([SPEED, 0.0]), dWdz(probe)
    )
    report["multipole_potential_relative_l2"] = relative_error(
        potential, phi_perturbation(probe)
    )

    # 3. route D / route N metrics from scratch ------------------------------
    dirichlet = solve_laplace_exterior_dirichlet(mesh, data)
    field = evaluate_exterior_field(mesh, dirichlet, probe)
    report["route_d_velocity_relative_l2"] = relative_error(
        field.gradient + np.array([SPEED, 0.0]), dWdz(probe)
    )
    report["route_d_potential_relative_l2"] = relative_error(
        field.potential, phi_perturbation(probe)
    )
    centers = mesh.centers
    r = np.hypot(centers[:, 0], centers[:, 1])
    cos_theta = centers[:, 0] / r
    neumann = solve_laplace_exterior_neumann(
        mesh, SPEED * RADIUS**2 * cos_theta / r**2
    )
    field_n = evaluate_exterior_field(mesh, neumann, probe)
    report["route_n_velocity_relative_l2"] = relative_error(
        field_n.gradient + np.array([SPEED, 0.0]), dWdz(probe)
    )

    # 4. near-wall table from scratch ---------------------------------------
    h_max = float(np.max(mesh.lengths))
    theta = np.arctan2(centers[:, 1], centers[:, 0])
    near_rows = []
    wall_points = {}
    for factor in payload["near_wall"]["factors"]:
        delta = factor * h_max
        pts = np.column_stack(
            ((RADIUS + delta) * np.cos(theta), (RADIUS + delta) * np.sin(theta))
        )
        wall_points[factor] = pts
        fld = evaluate_exterior_field(mesh, neumann, pts)
        velocity = fld.gradient + np.array([SPEED, 0.0])
        exact = dWdz(pts)
        rel = np.linalg.norm(velocity - exact, axis=1) / np.linalg.norm(exact, axis=1)
        _, grad_multi = multipole_field(mesh, data, pts)
        rel_multi = np.linalg.norm(
            grad_multi + np.array([SPEED, 0.0]) - exact, axis=1
        ) / np.linalg.norm(exact, axis=1)
        near_rows.append(
            {
                "clearance_over_h_max": factor,
                "panel_bem_max_relative_error": float(np.max(rel)),
                "multipole_max_relative_error": float(np.max(rel_multi)),
            }
        )
    report["near_wall"] = near_rows

    committed = {
        row["clearance_over_h_max"]: row["max_relative_velocity_error"]
        for row in payload["near_wall"]["levels"][-1]["sweep"]
    }
    report["near_wall_committed_vs_recomputed_max_abs_diff"] = float(
        max(
            abs(row["panel_bem_max_relative_error"] - committed[row["clearance_over_h_max"]])
            for row in near_rows
        )
    )
    committed_routes = {
        "route_d_velocity_relative_l2": payload["routes"]["dirichlet"]["finest"][
            "velocity_relative_l2"
        ],
        "route_n_velocity_relative_l2": payload["routes"]["neumann"]["finest"][
            "velocity_relative_l2"
        ],
    }
    report["route_metrics_committed_vs_recomputed_max_abs_diff"] = float(
        max(
            abs(report[key] - value)  # type: ignore[arg-type]
            for key, value in committed_routes.items()
        )
    )
    print(json.dumps(report, indent=2))
    out = ROOT / "docs/verify_m20303_independent.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())