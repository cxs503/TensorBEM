"""Oblique-heading extension of CASE-BEM-HULME-HEMISPHERE (case-pool id 138).

The anchor case scores the floating hemisphere at one incident-wave heading
(``wave_angle = 0``). That leaves the whole ``wave_angle`` code path -- the
incident phase, the load projection and the sign convention of the transverse
force -- with no reference at all: a heading-sign error is invisible to a
head-on-only table (the existing third-party cross-check file states the same
worry in its own comments).

For a *body of revolution* whose axis is the vertical, the same observation
turns that gap into an exact reference. The fluid domain, the free-surface
condition, the body and the radiation condition are all invariant under
rotation about ``z``; so with

    phi_I(beta) = -(i g A / omega) exp(nu z + i nu (x cos beta + y sin beta))

being the rotation by ``beta`` of the ``beta = 0`` incident potential, the
complex load six-vector obeys, in the continuum,

    V(beta) = R_beta V(0),        V = (Fx, Fy, Fz, Mx, My, Mz)

with ``R_beta`` the rotation about ``z`` acting on the (x, y) components and
leaving ``Fz`` and ``Mz`` untouched. Two consequences are scored here:

* the transverse force is *not* free: ``Fy(beta) = Fx(0) sin beta`` exactly,
  and the horizontal magnitude ``|F_h|`` is the same at every heading;
* the yaw exciting moment vanishes identically, ``Mz(beta) = 0``.

Discretisation is what is measured, in three independent ways:

1. ``heading`` cells -- the case mesh (1024 constant panels) at ten headings
   and three dimensionless wavenumbers, compared with ``R_beta V(0)`` of the
   same mesh. The deviation is pure mesh anisotropy, and is gated at the case
   tolerance (5 %);
2. ``rotated`` cells -- the exact algebraic identity. Solving on the mesh
   rotated by ``-beta`` at ``wave_angle = 0`` and rotating the answer back must
   reproduce the ``wave_angle = beta`` solve on the unrotated mesh to
   round-off, because the two discrete problems are the same problem written
   in rotated coordinates. This isolates the heading *implementation* from the
   discretisation: it fails instantly on a wrong sign, a wrong phase prefactor
   or a transposed (x, y) pair;
3. ``haskind`` -- the Haskind/Kochin far-field route at the oblique heading,
   which uses radiation potentials only, against the direct diffraction solve.

Everything is written to JSON with provenance (git head, numpy, host, module
sha256) so that no number in the rendered document is typed by hand.

Usage (the task list is deterministic; one shard per task)::

    python benchmarks/verified/casepool_bem_hulme_heading/run_case.py --spec
    python benchmarks/verified/casepool_bem_hulme_heading/run_case.py --shard 0 --shards 5
    python benchmarks/verified/casepool_bem_hulme_heading/run_case.py --merge
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    haskind_excitation_force_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import WettedHullMesh, hemisphere_hull_mesh

RADIUS = 1.0
DENSITY = 1025.0
GRAVITY = 9.80665
AMPLITUDE = 1.0

CASE_SUBDIVISIONS = 4          # 1024 constant panels, the anchor case mesh
COARSE_SUBDIVISIONS = 3        # 256 panels, refinement control
CASE_KA = (0.5, 1.0, 2.0)
HEADINGS_DEG = (0.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
ROTATED_KA = 1.0
ROTATED_HEADINGS_DEG = (45.0, 90.0)

TOL = 0.05                     # the anchor case tolerance, unchanged
IDENTITY_TOL = 1.0e-3          # discrete-identity gate (convention errors are O(1))
YAW_NULL_TOL = 1.0e-3          # |Mz| / (rho g A a^3)

FORCE_SCALE = DENSITY * GRAVITY * AMPLITUDE * RADIUS**2
MOMENT_SCALE = FORCE_SCALE * RADIUS
VOLUME = (2.0 / 3.0) * math.pi * RADIUS**3

OUT_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# tasks
# --------------------------------------------------------------------------- #
def build_tasks() -> list[dict[str, Any]]:
    """Deterministic task list; one shard runs one task."""
    tasks: list[dict[str, Any]] = [
        {"kind": "heading", "ka": ka, "subdivisions": CASE_SUBDIVISIONS,
         "headings_deg": list(HEADINGS_DEG)}
        for ka in CASE_KA
    ]
    tasks.append({"kind": "rotated", "ka": ROTATED_KA,
                  "subdivisions": CASE_SUBDIVISIONS,
                  "headings_deg": list(ROTATED_HEADINGS_DEG)})
    tasks.append({"kind": "heading", "ka": ROTATED_KA,
                  "subdivisions": COARSE_SUBDIVISIONS,
                  "headings_deg": list(HEADINGS_DEG)})
    return tasks


def _slug(task: dict[str, Any]) -> str:
    return f"{task['kind']}_ka{task['ka']:g}_n{task['subdivisions']}"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _complex(value: complex) -> dict[str, float]:
    z = complex(value)
    return {"re": float(z.real), "im": float(z.imag)}


def _norm6(loads: np.ndarray) -> np.ndarray:
    """Normalised complex six-vector: forces / (rho g A a^2), moments / (... a)."""
    out = np.zeros(6, dtype=np.complex128)
    out[:3] = loads[:3] / FORCE_SCALE
    out[3:] = loads[3:] / MOMENT_SCALE
    return out


def rotate_loads(loads: np.ndarray, beta_rad: float) -> np.ndarray:
    """Apply ``R_beta`` about ``z`` to a normalised complex load six-vector."""
    c, s = math.cos(beta_rad), math.sin(beta_rad)
    out = np.array(loads, dtype=np.complex128, copy=True)
    out[0] = c * loads[0] - s * loads[1]
    out[1] = s * loads[0] + c * loads[1]
    out[3] = c * loads[3] - s * loads[4]
    out[4] = s * loads[3] + c * loads[4]
    return out


def rotate_mesh(mesh: WettedHullMesh, beta_rad: float) -> WettedHullMesh:
    """Rotate a wetted hull mesh about the ``z`` axis through the origin."""
    verts = np.asarray(mesh.vertices, dtype=float).copy()
    c, s = math.cos(beta_rad), math.sin(beta_rad)
    x, y = verts[:, 0].copy(), verts[:, 1].copy()
    verts[:, 0] = c * x - s * y
    verts[:, 1] = s * x + c * y
    # The waterline vertices stay exactly on z = 0: the rotation acts on x, y.
    verts[:, 2] = np.where(np.abs(verts[:, 2]) < 1.0e-14, 0.0, verts[:, 2])
    return WettedHullMesh(verts, np.asarray(mesh.faces))


def _shape(mesh: WettedHullMesh) -> dict[str, float]:
    exact_volume = VOLUME
    exact_area = math.pi * RADIUS**2
    return {
        "panels": int(mesh.faces.shape[0]),
        "mesh_volume": float(mesh.volume),
        "mesh_waterplane_area": float(mesh.waterplane_area),
        "volume_rel_error": float(mesh.volume) / exact_volume - 1.0,
        "waterplane_rel_error": float(mesh.waterplane_area) / exact_area - 1.0,
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    head = None
    git_dir = root / ".git"
    if git_dir.is_file():                     # git worktree: .git is a pointer file
        pointer = git_dir.read_text().strip()
        if pointer.startswith("gitdir: "):
            git_dir = (root / pointer[len("gitdir: "):]).resolve()
    head_file = git_dir / "HEAD"
    if head_file.is_file():
        ref = head_file.read_text().strip()
        if ref.startswith("ref: "):
            target = git_dir / ref[5:]
            head = target.read_text().strip() if target.exists() else None
        else:
            head = ref
    modules = {}
    for name in ("free_surface3d.py", "free_surface_solver3d.py",
                 "geometry3d.py", "analytic.py"):
        path = root / "src" / "tensorbem" / name
        if path.exists():
            modules[name] = _file_sha256(path)
    return {
        "git_head": head,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "host": socket.gethostname(),
        "arch": platform.machine(),
        "module_sha256": modules,
        "script_sha256": _file_sha256(Path(__file__).resolve()),
        "normalisation": {
            "force": "F / (rho g A a^2)",
            "moment": "M / (rho g A a^3)",
            "added_mass": "A / (rho nabla)",
            "damping": "B / (omega rho nabla)",
            "time_convention": "exp(-i omega t)",
        },
    }


# --------------------------------------------------------------------------- #
# cells
# --------------------------------------------------------------------------- #
def heading_cell(task: dict[str, Any]) -> dict[str, Any]:
    """Solve the case mesh at every requested heading on one (ka, mesh)."""
    ka = float(task["ka"])
    subdivisions = int(task["subdivisions"])
    mesh = hemisphere_hull_mesh(subdivisions, radius=RADIUS)
    nu = ka / RADIUS
    t0 = time.time()
    ops = assemble_free_surface_operators_3d(mesh, nu, chief_points="auto")
    assemble_seconds = time.time() - t0

    rad = solve_radiation_3d(
        mesh, nu, modes=(0, 1, 2), density=DENSITY, gravity=GRAVITY, operators=ops
    )
    norm_a = DENSITY * VOLUME
    norm_b = DENSITY * math.sqrt(GRAVITY * nu) * VOLUME

    rows = []
    for heading in task["headings_deg"]:
        beta = math.radians(float(heading))
        t1 = time.time()
        diff = solve_diffraction_3d(
            mesh, nu, amplitude=AMPLITUDE, wave_angle=beta,
            density=DENSITY, gravity=GRAVITY, operators=ops,
        )
        loads = _norm6(np.asarray(diff.excitation_force, dtype=np.complex128))
        hask = haskind_excitation_force_3d(
            mesh, rad, amplitude=AMPLITUDE, wave_angle=beta,
            density=DENSITY, gravity=GRAVITY,
        )
        kochin = np.asarray(hask["kochin_haskind"], dtype=np.complex128)[:3] / FORCE_SCALE
        boundary = np.asarray(hask["boundary_haskind"], dtype=np.complex128)[:3] / FORCE_SCALE
        rows.append({
            "heading_deg": float(heading),
            "loads": [_complex(v) for v in loads],
            "kochin_force": [_complex(v) for v in kochin],
            "boundary_haskind_force": [_complex(v) for v in boundary],
            "relative_residual": float(diff.relative_residual),
            "condition_number": float(diff.condition_number),
            "solve_method": str(getattr(diff, "solve_method", "")),
            "wall_seconds": time.time() - t1,
        })

    return {
        "kind": "heading",
        "slug": _slug(task),
        "ka": ka,
        "nu": nu,
        "subdivisions": subdivisions,
        "geometry": _shape(mesh),
        "chief": "auto",
        "assemble_seconds": assemble_seconds,
        "operator_condition_number": float(ops.condition_number),
        "radiation": {
            "A11_dimless": float(rad.added_mass[0, 0]) / norm_a,
            "A22_dimless": float(rad.added_mass[1, 1]) / norm_a,
            "A33_dimless": float(rad.added_mass[2, 2]) / norm_a,
            "A12_dimless": float(rad.added_mass[0, 1]) / norm_a,
            "A13_dimless": float(rad.added_mass[0, 2]) / norm_a,
            "B11_dimless": float(rad.damping[0, 0]) / norm_b,
            "B22_dimless": float(rad.damping[1, 1]) / norm_b,
            "B33_dimless": float(rad.damping[2, 2]) / norm_b,
            "B12_dimless": float(rad.damping[0, 1]) / norm_b,
            "relative_residual": float(rad.relative_residual),
            "condition_number": float(rad.condition_number),
        },
        "headings": rows,
        "wall_seconds": time.time() - t0,
    }


def rotated_cell(task: dict[str, Any]) -> dict[str, Any]:
    """Exact identity: solve(beta on M) == R_beta solve(0 on R_{-beta} M)."""
    ka = float(task["ka"])
    subdivisions = int(task["subdivisions"])
    mesh = hemisphere_hull_mesh(subdivisions, radius=RADIUS)
    nu = ka / RADIUS
    ops = assemble_free_surface_operators_3d(mesh, nu, chief_points="auto")
    base = solve_diffraction_3d(
        mesh, nu, amplitude=AMPLITUDE, wave_angle=0.0,
        density=DENSITY, gravity=GRAVITY, operators=ops,
    )
    base_loads = _norm6(np.asarray(base.excitation_force, dtype=np.complex128))

    rows = []
    for heading in task["headings_deg"]:
        beta = math.radians(float(heading))
        rot = rotate_mesh(mesh, -beta)
        rot_ops = assemble_free_surface_operators_3d(rot, nu, chief_points="auto")
        direct = solve_diffraction_3d(
            mesh, nu, amplitude=AMPLITUDE, wave_angle=beta,
            density=DENSITY, gravity=GRAVITY, operators=ops,
        )
        rebuilt = solve_diffraction_3d(
            rot, nu, amplitude=AMPLITUDE, wave_angle=0.0,
            density=DENSITY, gravity=GRAVITY, operators=rot_ops,
        )
        direct_loads = _norm6(np.asarray(direct.excitation_force, dtype=np.complex128))
        rebuilt_loads = _norm6(np.asarray(rebuilt.excitation_force, dtype=np.complex128))
        predicted = rotate_loads(rebuilt_loads, beta)
        scale = float(np.max(np.abs(rebuilt_loads)))
        dev = float(np.max(np.abs(direct_loads - predicted))) / scale
        rows.append({
            "heading_deg": float(heading),
            "direct_loads": [_complex(v) for v in direct_loads],
            "rotated_mesh_loads": [_complex(v) for v in rebuilt_loads],
            "rotated_mesh_loads_back_rotated": [_complex(v) for v in predicted],
            "max_deviation": dev,
            "direct_residual": float(direct.relative_residual),
            "rotated_residual": float(rebuilt.relative_residual),
            "rotated_volume_rel_error": _shape(rot)["volume_rel_error"],
        })

    return {
        "kind": "rotated",
        "slug": _slug(task),
        "ka": ka,
        "nu": nu,
        "subdivisions": subdivisions,
        "geometry": _shape(mesh),
        "base_loads_heading0": [_complex(v) for v in base_loads],
        "headings": rows,
    }


# --------------------------------------------------------------------------- #
# merge + gates
# --------------------------------------------------------------------------- #
def _c(entry: dict[str, float]) -> complex:
    return complex(entry["re"], entry["im"])


def _deviation(loads: np.ndarray, predicted: np.ndarray) -> float:
    scale = float(np.max(np.abs(predicted)))
    if scale == 0.0:
        return float(np.max(np.abs(loads)))
    return float(np.max(np.abs(loads - predicted))) / scale


def evaluate(cells: list[dict[str, Any]]) -> dict[str, Any]:
    heading_cells = [c for c in cells if c["kind"] == "heading"]
    # The primary (finest) heading mesh actually present in this bundle.  The
    # anchor case mesh is CASE_SUBDIVISIONS; a bundle that only carries the
    # coarse ladder is reported as such instead of pretending otherwise.
    primary_sub = max(c["subdivisions"] for c in heading_cells)
    case = [c for c in heading_cells if c["subdivisions"] == primary_sub]
    coarse = [c for c in heading_cells if c["subdivisions"] != primary_sub]
    rotated = [c for c in cells if c["kind"] == "rotated"]

    heading_rows: list[dict[str, Any]] = []
    for cell in sorted(case, key=lambda c: c["ka"]):
        base = np.array([_c(v) for v in cell["headings"][0]["loads"]])
        for row in cell["headings"]:
            beta = math.radians(row["heading_deg"])
            loads = np.array([_c(v) for v in row["loads"]])
            predicted = rotate_loads(base, beta)
            kochin = np.array([_c(v) for v in row["kochin_force"]])
            boundary = np.array([_c(v) for v in row["boundary_haskind_force"]])
            force_scale = float(np.max(np.abs(loads[:3])))
            heading_rows.append({
                "ka": cell["ka"],
                "panels": cell["geometry"]["panels"],
                "heading_deg": row["heading_deg"],
                "loads": row["loads"],
                "predicted": [_complex(v) for v in predicted],
                "max_deviation": _deviation(loads, predicted),
                "horizontal_magnitude": abs(complex(loads[0], loads[1])),
                "horizontal_magnitude_heading0": abs(base[0]),
                "heave_deviation": abs(loads[2] - base[2]) / abs(base[2]),
                "yaw_moment": abs(loads[5]),
                "kochin_deviation": float(np.max(np.abs(loads[:3] - kochin))) / force_scale,
                "boundary_haskind_deviation": float(np.max(np.abs(loads[:3] - boundary))) / force_scale,
                "relative_residual": row["relative_residual"],
                "condition_number": row["condition_number"],
            })

    coarse_rows: list[dict[str, Any]] = []
    for cell in sorted(coarse, key=lambda c: c["ka"]):
        base = np.array([_c(v) for v in cell["headings"][0]["loads"]])
        for row in cell["headings"]:
            beta = math.radians(row["heading_deg"])
            loads = np.array([_c(v) for v in row["loads"]])
            coarse_rows.append({
                "ka": cell["ka"],
                "panels": cell["geometry"]["panels"],
                "heading_deg": row["heading_deg"],
                "max_deviation": _deviation(loads, rotate_loads(base, beta)),
                "yaw_moment": abs(loads[5]),
            })

    rotated_rows: list[dict[str, Any]] = []
    for cell in rotated:
        for row in cell["headings"]:
            rotated_rows.append({
                "ka": cell["ka"],
                "panels": cell["geometry"]["panels"],
                "heading_deg": row["heading_deg"],
                "max_deviation": row["max_deviation"],
                "direct_residual": row["direct_residual"],
                "rotated_residual": row["rotated_residual"],
                "rotated_volume_rel_error": row["rotated_volume_rel_error"],
            })

    radiation_rows = [
        {"ka": cell["ka"], "panels": cell["geometry"]["panels"], **cell["radiation"]}
        for cell in sorted(case, key=lambda c: c["ka"])
    ]

    worst_heading = max((r["max_deviation"] for r in heading_rows), default=float("nan"))
    worst_coarse = max((r["max_deviation"] for r in coarse_rows), default=float("nan"))
    worst_rotated = max((r["max_deviation"] for r in rotated_rows), default=float("nan"))
    worst_yaw = max((r["yaw_moment"] for r in heading_rows + coarse_rows), default=float("nan"))
    worst_kochin = max((r["kochin_deviation"] for r in heading_rows), default=float("nan"))
    worst_boundary = max((r["boundary_haskind_deviation"] for r in heading_rows), default=float("nan"))
    worst_heave = max((r["heave_deviation"] for r in heading_rows), default=float("nan"))
    worst_horiz = max(
        (abs(r["horizontal_magnitude"] / r["horizontal_magnitude_heading0"] - 1.0)
         for r in heading_rows), default=float("nan"))
    axial = max(
        (max(abs(r["A22_dimless"] / r["A11_dimless"] - 1.0),
             abs(r["B22_dimless"] / r["B11_dimless"] - 1.0))
         for r in radiation_rows), default=float("nan"))
    cross = max((max(abs(r["A12_dimless"]), abs(r["B12_dimless"])) for r in radiation_rows),
                default=float("nan"))
    # The anchor's G2 geometry precondition is a *case-mesh* condition (the
    # anchor case documents that the 256-panel rung fails it by design); the
    # coarse ladder below is a refinement control, not a graded mesh.
    geometric = max(
        (max(abs(c["geometry"]["volume_rel_error"]), abs(c["geometry"]["waterplane_rel_error"]))
         for c in case), default=float("nan"))
    coarse_geometric = max(
        (max(abs(c["geometry"]["volume_rel_error"]), abs(c["geometry"]["waterplane_rel_error"]))
         for c in coarse), default=float("nan"))

    gates = [
        {"id": "G1_heading_rotation_invariance_case_mesh",
         "criterion": f"max over ka, heading of |V(beta) - R_beta V(0)| / |V(0)| < {TOL} "
                      "on the 1024-panel case mesh",
         "measured": worst_heading,
         "verdict": "PASS" if worst_heading < TOL else "FAIL"},
        {"id": "G2_transverse_and_heave_components",
         "criterion": f"| |F_h(beta)| / |F_h(0)| - 1 | < {TOL} and |Fz(beta)/Fz(0) - 1| < {TOL} "
                      "for every ka, heading",
         "measured": max(worst_horiz, worst_heave),
         "verdict": "PASS" if max(worst_horiz, worst_heave) < TOL else "FAIL"},
        {"id": "G3_yaw_exciting_moment_null",
         "criterion": f"|Mz| / (rho g A a^3) < {YAW_NULL_TOL} for every cell (exactly zero by "
                      "axial symmetry)",
         "measured": worst_yaw,
         "verdict": "PASS" if worst_yaw < YAW_NULL_TOL else "FAIL"},
        {"id": "G4_rotated_mesh_algebraic_identity",
         "criterion": f"|V_direct(beta) - R_beta V_rotated(0)| / |V| < {IDENTITY_TOL} "
                      "(the two discrete problems are the same problem in rotated coordinates)",
         "measured": worst_rotated,
         "verdict": "PASS" if worst_rotated < IDENTITY_TOL else "FAIL"},
        {"id": "G5_haskind_oblique_agreement",
         "criterion": f"|V_solve - V_Haskind| / |V_solve| < {TOL} on the three force components, "
                      "for the Kochin and for the boundary Haskind integral",
         "measured": max(worst_kochin, worst_boundary),
         "verdict": "PASS" if max(worst_kochin, worst_boundary) < TOL else "FAIL"},
        {"id": "G6_radiation_axial_symmetry",
         "criterion": f"|A22/A11 - 1| < {TOL}, |B22/B11 - 1| < {TOL} (exact by axial symmetry) and "
                      "|A12|, |B12| dimensionless cross terms < 5e-2",
         "measured": max(axial, cross),
         "verdict": "PASS" if max(axial, cross) < max(TOL, 5.0e-2) else "FAIL"},
        {"id": "G7_coarse_mesh_does_not_exceed_tolerance",
         "criterion": f"heading deviation on the 256-panel mesh < {TOL}",
         "measured": worst_coarse,
         "verdict": "PASS" if worst_coarse < TOL else "FAIL"},
        {"id": "G8_geometry_precondition_case_mesh",
         "criterion": "|nabla_mesh/nabla - 1| < 2% and |A_wp/(pi a^2) - 1| < 2% on the primary "
                      "(case) mesh, as in the anchor's own G2",
         "measured": geometric,
         "verdict": "PASS" if geometric < 2.0e-2 else "FAIL"},
    ]
    verdict = "PASS" if all(g["verdict"] == "PASS" for g in gates) else "FAIL"
    return {
        "gates": gates,
        "verdict": verdict,
        "primary_subdivisions": primary_sub,
        "primary_panels": max(c["geometry"]["panels"] for c in case),
        "coarse_geometry_rel_error": coarse_geometric,
        "coarse_geometry_note": (
            "the refinement-control rung (256 panels) has a mesh-volume error above the 2% "
            "geometry precondition; that is a property of the coarse rung, documented by the "
            "anchor case, and is why the graded mesh is the 1024-panel one"),
        "case_mesh_subdivisions_expected": CASE_SUBDIVISIONS,
        "case_mesh_present": primary_sub == CASE_SUBDIVISIONS,
        "heading_rows": heading_rows,
        "coarse_rows": coarse_rows,
        "rotated_rows": rotated_rows,
        "radiation_rows": radiation_rows,
    }


def merge(out_dir: Path) -> int:
    cells = []
    for path in sorted(out_dir.glob("cell_*.json")):
        cells.append(json.loads(path.read_text()))
    if not cells:
        print("FAIL: no cells", file=sys.stderr)
        return 2
    result = {
        "schema": "tensorbem.casepool_bem_hulme_heading/v1",
        "case": "BEM-HULME-HEMISPHERE",
        "case_pool_id": 138,
        "anchor": "CASE-BEM-HULME-HEMISPHERE",
        "extension": "oblique incident-wave heading (exact reference from axial symmetry)",
        "tolerance": TOL,
        "identity_tolerance": IDENTITY_TOL,
        "yaw_null_tolerance": YAW_NULL_TOL,
        "case_subdivisions": CASE_SUBDIVISIONS,
        "coarse_subdivisions": COARSE_SUBDIVISIONS,
        "case_ka": list(CASE_KA),
        "headings_deg": list(HEADINGS_DEG),
        "rotated_headings_deg": list(ROTATED_HEADINGS_DEG),
        "parameters": {
            "radius_m": RADIUS, "density": DENSITY, "gravity": GRAVITY,
            "amplitude_m": AMPLITUDE, "displaced_volume_m3": VOLUME,
            "force_scale_N": FORCE_SCALE, "moment_scale_Nm": MOMENT_SCALE,
        },
        "provenance": provenance(),
        "cells": cells,
    }
    present = {c.get("slug") for c in cells}
    expected = {_slug(t) for t in build_tasks()}
    result["tasks_missing"] = sorted(expected - present)
    result.update(evaluate(cells))
    (out_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"verdict: {result['verdict']}")
    for gate in result["gates"]:
        print(f"  {gate['id']}: {gate['measured']:.6g} {gate['verdict']}")
    return 0 if result["verdict"] == "PASS" else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--spec", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out).resolve()
    tasks = build_tasks()
    if args.spec:
        print(json.dumps({"tasks": tasks, "count": len(tasks)}, indent=2))
        return 0
    if args.merge:
        return merge(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    indices = range(len(tasks)) if args.shard is None else [args.shard]
    for idx in indices:
        if idx >= len(tasks):
            print(f"shard {idx} out of range", file=sys.stderr)
            return 2
        task = tasks[idx]
        t0 = time.time()
        cell = rotated_cell(task) if task["kind"] == "rotated" else heading_cell(task)
        cell["subtask_index"] = idx
        cell["subtask_total"] = len(tasks)
        cell["provenance"] = provenance()
        (out_dir / f"cell_{idx:02d}_{_slug(task)}.json").write_text(
            json.dumps(cell, indent=2) + "\n")
        print(f"[{idx}] {_slug(task)} done in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())