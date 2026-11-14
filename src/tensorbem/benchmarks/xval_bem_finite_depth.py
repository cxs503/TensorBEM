"""Cross-validation of the 3-D finite-depth radiation BEM against an independent
BEM code, a closed-form reference and an exact surface measure.

Why a new case
--------------
``docs/finite_depth_solver3d/report.md`` verified ``tensorbem.finite_depth_solver3d``
against the closed-form Thorne/Wang/Linton multipole solution in *deep* water and
against itself (reciprocity, mode truncation, negative controls).  It never put
the finite-depth answer next to a **second, independently written BEM code** in
the *same* finite depth, and its mesh ladder stopped at 80 panels with ``B33``
19.6 % low against the closed form -- undecided between discretisation and a
defect.  This case supplies both: a three-route cross-validation and the
measurement that resolves the 19.6 %.

Routes, one geometry, the *same* triangle soup wherever possible
----------------------------------------------------------------
``T``  TensorBEM ``finite_depth_solver3d`` (direct Green's third identity, John
       (1950) eigenseries kernel, analytic singular split).
``T+`` the same solver with the exact spherical surface measure
       (``panel_areas``, i.e. :func:`tensorbem.geometry3d.spherical_patch_areas`)
       instead of the flat-triangle measure of the mesh.
``C``  Capytaine 3.1 ``BEMSolver`` (Delhommeau finite-depth Green function,
       Nemoh lineage) on the identical triangles -- so the comparison isolates the
       formulation, not the mesh generator.
``A``  Closed-form multipole solution ``submerged_sphere_wave_multipole`` (Thorne
       1953 / Wang 1986 / Linton 1991), deep water, used as the ``h -> inf`` end
       of the depth ladder.

Conventions are the ones already implemented: ``exp(-i*omega*t)``,
``nu = omega**2/g``, ``ka`` the non-dimensional wavenumber of a sphere of radius
``a``, and ``A + i B/omega = -rho * int_S phi n dS`` for unit velocity amplitude.

What is measured (stages)
-------------------------
``mesh``       mesh ladder (20/80/320 panels) vs ``A`` at ``h/a = 20``, flat and
               exact measure, plus a mode-truncation ladder and the deep-water
               solver; this is where the historical 19.6 % ``B33`` residual is
               decomposed into measure + panel + truncation contributions.
``depth``      matched-mesh ``T+`` vs ``C`` on ``h/a in {3, 5, 20}`` at ``ka = 1``,
               the acceptance grid of the case (``< 15 %``, owner standard).
``freq``       matched-mesh ``T+`` vs ``C`` on ``ka in {0.5, 1.0, 2.0}`` at
               ``h/a = 5`` and in deep water -- the damping band, where a kernel
               error is most visible.
``consistency`` internal invariants on the working mesh (reciprocity,
               axisymmetry, surge/heave decoupling), no external code involved.

Run
---
``PYTHONPATH=src python -m tensorbem.benchmarks.xval_bem_finite_depth --stage all``
writes ``benchmarks/verified/xval_bem_finite_depth/{result.json,gates.json}``.
Stages are also runnable one at a time (``stage_<name>.json`` per stage, so a long
study runs in bounded chunks); ``--stage merge`` re-evaluates the gates from the
stored stages.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from tensorbem.analytic import submerged_sphere_wave_multipole
from tensorbem.finite_depth_solver3d import solve_radiation_finite_depth_3d
from tensorbem.free_surface_solver3d import solve_radiation_3d
from tensorbem.geometry3d import (
    SurfaceMesh3D,
    spherical_patch_areas,
    submerged_sphere_mesh,
)

REPO = Path(__file__).resolve().parents[3]
CASE_DIR = REPO / "benchmarks" / "verified" / "xval_bem_finite_depth"

# --- case definition (identical to docs/finite_depth_solver3d/report.md) ------
RADIUS = 1.0
SUBMERGENCE = 1.5
RHO = 1025.0
GRAVITY = 9.80665
CENTER = (0.0, 0.0, -SUBMERGENCE)

#: Panels of the geodesic sphere per subdivision level.
SUBDIVISION_PANELS = {0: 20, 1: 80, 2: 320}
#: Working mesh of the cross-validation.
WORK_SUBDIVISION = 2
#: Truncation used for the ``depth``/``freq`` stages (cross-checked by the ladder).
WORK_MODE_COUNT = 32
#: Depth stations of the matched-mesh grid, as ``h/a``.
DEPTH_RATIOS = (3.0, 5.0, 20.0)
#: Frequency stations of the matched-mesh grid.
KA_GRID = (0.5, 2.0)
#: Mode-truncation ladder, run on the 80-panel mesh (cheap, truncation is a
#: property of the kernel, not of the panel count).
MODE_LADDER = (64, 128, 256, 512)
#: Acceptance threshold of the case (owner standard: > 15 % is not acceptable).
THRESHOLD = 0.15


def _rel(got: float, ref: float) -> float:
    return abs(got - ref) / abs(ref) if ref else float("nan")


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
def route_analytic(ka: float) -> dict[str, Any]:
    """Deep-water closed-form multipole reference (route ``A``)."""
    res = submerged_sphere_wave_multipole(
        ka, radius=RADIUS, submergence=SUBMERGENCE, density=RHO, gravity=GRAVITY
    )
    return {
        "route": "analytic_multipole",
        "depth_ratio": math.inf,
        "ka": ka,
        "A11": float(res.surge_added_mass),
        "A33": float(res.heave_added_mass),
        "B11": float(res.surge_damping),
        "B33": float(res.heave_damping),
        "B33_far_field_dimless": float(res.heave_far_field_damping_dimless),
    }


def exact_measure(mesh: SurfaceMesh3D) -> np.ndarray:
    """Exact spherical patch areas of the panels of ``mesh``."""
    return spherical_patch_areas(
        mesh.vertices, mesh.faces, center=CENTER, radius=RADIUS
    )


def route_tensorbem(
    mesh: SurfaceMesh3D,
    ka: float,
    depth_ratio: float | None,
    *,
    mode_count: int | None = None,
    modes: tuple[int, ...] = (0, 2),
    panel_areas: np.ndarray | None = None,
) -> dict[str, Any]:
    """TensorBEM route; ``depth_ratio=None`` is the deep-water solver."""
    nu = ka / RADIUS
    t0 = time.time()
    if depth_ratio is None:
        res = solve_radiation_3d(
            mesh,
            nu,
            modes=list(modes),
            center=CENTER,
            density=RHO,
            gravity=GRAVITY,
            panel_areas=panel_areas,
        )
        info: dict[str, Any] = {"mode_count": None}
        added, damping = res.added_mass, res.damping
    else:
        depth = float(depth_ratio) * RADIUS
        res = solve_radiation_finite_depth_3d(
            mesh,
            nu,
            depth,
            modes=list(modes),
            center=CENTER,
            density=RHO,
            gravity=GRAVITY,
            mode_count=mode_count,
            panel_areas=panel_areas,
        )
        info = {
            "mode_count": int(res.mode_count),
            "condition_number": float(res.condition_number),
            "relative_residual": float(res.relative_residual),
        }
        added, damping = res.added_mass, res.damping
    index = {m: i for i, m in enumerate(modes)}
    info.update(
        {
            "route": "tensorbem_exact_measure" if panel_areas is not None else "tensorbem",
            "depth_ratio": depth_ratio,
            "ka": ka,
            "panels": int(mesh.panel_count),
            "A11": float(added[index[0], index[0]]),
            "A33": float(added[index[2], index[2]]),
            "B11": float(damping[index[0], index[0]]),
            "B33": float(damping[index[2], index[2]]),
            "A13": float(added[index[0], index[2]]),
            "seconds": time.time() - t0,
        }
    )
    return info


def capytaine_available() -> tuple[bool, str]:
    """Whether route ``C`` can run here, with the reason when it cannot."""
    try:
        cpt = importlib.import_module("capytaine")
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"capytaine import failed: {exc}"
    return True, f"capytaine {getattr(cpt, '__version__', 'unknown')}"


def route_capytaine(
    mesh: SurfaceMesh3D, ka: float, depth_ratio: float | None
) -> dict[str, Any] | None:
    """Independent OSS BEM route (``C``) on the *same* triangles; ``None`` = not run."""
    ok, _ = capytaine_available()
    if not ok:
        return None
    import capytaine as cpt

    surface = cpt.Mesh(
        vertices=np.asarray(mesh.vertices, dtype=float),
        faces=np.asarray(mesh.faces, dtype=np.int64),
    )
    body = cpt.FloatingBody(
        mesh=surface, dofs=cpt.rigid_body_dofs(rotation_center=CENTER)
    )
    depth = math.inf if depth_ratio is None else float(depth_ratio) * RADIUS
    solver = cpt.BEMSolver()
    out: dict[str, Any] = {
        "route": "capytaine",
        "depth_ratio": depth_ratio,
        "ka": ka,
        "panels": int(mesh.panel_count),
    }
    t0 = time.time()
    for dof, key_a, key_b in (("Surge", "A11", "B11"), ("Heave", "A33", "B33")):
        problem = cpt.RadiationProblem(
            body=body,
            wavenumber=ka / RADIUS,
            water_depth=depth,
            radiating_dof=dof,
            rho=RHO,
            g=GRAVITY,
        )
        result = solver.solve(problem)
        out[key_a] = float(result.added_mass[dof])
        out[key_b] = float(result.radiation_damping[dof])
    out["seconds"] = time.time() - t0
    return out


def build_mesh(subdivision: int) -> SurfaceMesh3D:
    return submerged_sphere_mesh(subdivision, radius=RADIUS, submergence=SUBMERGENCE)


def mesh_geometry(mesh: SurfaceMesh3D) -> dict[str, Any]:
    """Geometry gate: exact patch areas of a closed sphere, fluid-domain clearance."""
    patch = exact_measure(mesh)
    exact = 4.0 * math.pi * RADIUS**2
    z = np.asarray(mesh.vertices, dtype=float)[:, 2]
    return {
        "panels": int(mesh.panel_count),
        "flat_area": float(np.sum(mesh.areas)),
        "patch_area": float(np.sum(patch)),
        "exact_area": exact,
        "flat_area_ratio": float(np.sum(mesh.areas) / exact),
        "patch_area_ratio": float(np.sum(patch) / exact),
        "z_min": float(z.min()),
        "z_max": float(z.max()),
    }


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def stage_mesh() -> dict[str, Any]:
    """Mesh ladder + mode ladder + deep-water solver, all against route ``A``."""
    analytic = route_analytic(1.0)
    rows: list[dict[str, Any]] = []
    for subdivision in (0, 1, 2):
        mesh = build_mesh(subdivision)
        patch = exact_measure(mesh)
        for label, areas in (("flat", None), ("exact_measure", patch)):
            row = route_tensorbem(
                mesh, 1.0, 20.0, mode_count=WORK_MODE_COUNT, panel_areas=areas
            )
            for key in ("A11", "A33", "B11", "B33"):
                row[f"rel_err_{key}"] = _rel(row[key], analytic[key])
            row["measure"] = label
            row["geometry"] = mesh_geometry(mesh)
            rows.append(row)
        cap = route_capytaine(mesh, 1.0, 20.0)
        if cap is not None:
            for key in ("A11", "A33", "B11", "B33"):
                cap[f"rel_err_{key}"] = _rel(cap[key], analytic[key])
            cap["measure"] = "capytaine_builtin"
            rows.append(cap)
    # mode-truncation ladder: a kernel property, measured on the 80-panel mesh
    mesh = build_mesh(1)
    patch = exact_measure(mesh)
    mode_rows = []
    for mode_count in MODE_LADDER:
        row = route_tensorbem(
            mesh, 1.0, 20.0, mode_count=mode_count, panel_areas=patch
        )
        row["measure"] = "exact_measure"
        mode_rows.append(row)
    deep = route_tensorbem(
        build_mesh(WORK_SUBDIVISION), 1.0, None, modes=(0, 2),
        panel_areas=exact_measure(build_mesh(WORK_SUBDIVISION)),
    )
    for key in ("A11", "A33", "B11", "B33"):
        deep[f"rel_err_{key}"] = _rel(deep[key], analytic[key])
    return {
        "stage": "mesh",
        "analytic": analytic,
        "rows": rows,
        "mode_ladder": mode_rows,
        "deep_water_solver": deep,
    }


def stage_depth() -> dict[str, Any]:
    """Matched-mesh cross-validation over the depth ladder (working mesh)."""
    mesh = build_mesh(WORK_SUBDIVISION)
    patch = exact_measure(mesh)
    rows = []
    for depth_ratio in DEPTH_RATIOS:
        t = route_tensorbem(
            mesh, 1.0, depth_ratio, mode_count=WORK_MODE_COUNT, panel_areas=patch
        )
        cap = route_capytaine(mesh, 1.0, depth_ratio)
        row: dict[str, Any] = {
            "depth_ratio": depth_ratio,
            "ka": 1.0,
            "tensorbem": t,
        }
        if cap is not None:
            row["capytaine"] = cap
            for key in ("A33", "B33", "A11", "B11"):
                row[f"rel_diff_{key}"] = _rel(t[key], cap[key])
        rows.append(row)
    analytic = route_analytic(1.0)
    for row in rows:
        for key in ("A33", "B33"):
            row[f"rel_err_vs_analytic_{key}"] = _rel(row["tensorbem"][key], analytic[key])
    return {
        "stage": "depth",
        "panels": int(mesh.panel_count),
        "mode_count": WORK_MODE_COUNT,
        "analytic": analytic,
        "rows": rows,
    }


def stage_freq() -> dict[str, Any]:
    """Matched-mesh cross-validation over the frequency ladder."""
    mesh = build_mesh(WORK_SUBDIVISION)
    patch = exact_measure(mesh)
    rows = []
    for ka in KA_GRID:
        for depth_ratio in (5.0, None):
            t = route_tensorbem(
                mesh,
                ka,
                depth_ratio,
                mode_count=None if depth_ratio is None else WORK_MODE_COUNT,
                panel_areas=patch,
            )
            cap = route_capytaine(mesh, ka, depth_ratio)
            row: dict[str, Any] = {"ka": ka, "depth_ratio": depth_ratio, "tensorbem": t}
            if cap is not None:
                row["capytaine"] = cap
                for key in ("A33", "B33"):
                    row[f"rel_diff_{key}"] = _rel(t[key], cap[key])
            if depth_ratio is None:
                analytic = route_analytic(ka)
                row["analytic"] = analytic
                for key in ("A33", "B33"):
                    row[f"rel_err_vs_analytic_{key}"] = _rel(t[key], analytic[key])
            rows.append(row)
    return {
        "stage": "freq",
        "panels": int(mesh.panel_count),
        "mode_count": WORK_MODE_COUNT,
        "rows": rows,
    }


def stage_consistency() -> dict[str, Any]:
    """Internal invariants on the working mesh; no external code involved."""
    mesh = build_mesh(WORK_SUBDIVISION)
    res = solve_radiation_finite_depth_3d(
        mesh,
        1.0,
        5.0 * RADIUS,
        modes=list(range(6)),
        center=CENTER,
        density=RHO,
        gravity=GRAVITY,
        mode_count=WORK_MODE_COUNT,
        panel_areas=exact_measure(mesh),
    )
    a = np.asarray(res.added_mass, dtype=float)
    b = np.asarray(res.damping, dtype=float)
    return {
        "stage": "consistency",
        "panels": int(mesh.panel_count),
        "mode_count": int(res.mode_count),
        "reciprocity_A": float(np.max(np.abs(a - a.T)) / np.max(np.abs(a))),
        "reciprocity_B": float(np.max(np.abs(b - b.T)) / np.max(np.abs(b))),
        "axisymmetry_A11_A22": float(abs(a[0, 0] - a[1, 1]) / abs(a[0, 0])),
        "axisymmetry_B11_B22": float(abs(b[0, 0] - b[1, 1]) / abs(b[0, 0])),
        "surge_heave_coupling": float(abs(a[0, 2]) / abs(a[2, 2])),
        "roll_over_surge": float(a[3, 3] / a[0, 0]),
        "condition_number": float(res.condition_number),
    }


STAGES = {
    "mesh": stage_mesh,
    "depth": stage_depth,
    "freq": stage_freq,
    "consistency": stage_consistency,
}


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def _rows(stage: dict[str, Any] | None, **match: Any) -> list[dict[str, Any]]:
    if not stage:
        return []
    if "rows" in stage:
        return [r for r in stage["rows"] if all(r.get(k) == v for k, v in match.items())]
    return [r for r in stage.get("mode_ladder", []) if all(r.get(k) == v for k, v in match.items())]


def evaluate_gates(result: dict[str, Any]) -> dict[str, Any]:
    """Turn the recorded numbers into named pass/fail gates."""
    gates: list[dict[str, Any]] = []

    def gate(gid: str, criterion: str, measured: Any, passed: bool, note: str = "") -> None:
        gates.append(
            {
                "id": gid,
                "criterion": criterion,
                "measured": None if measured is None else float(measured),
                "passed": bool(passed),
                "note": note,
            }
        )

    mesh_stage = result.get("mesh")
    if mesh_stage:
        geoms = [r["geometry"] for r in mesh_stage["rows"] if "geometry" in r]
        worst = max(abs(g["patch_area_ratio"] - 1.0) for g in geoms)
        gate(
            "G1_geometry",
            "exact patch area of every sphere mesh within 1 % of 4*pi*a^2",
            worst,
            worst <= 0.01,
            f"flat-area ratio of the coarsest mesh {min(g['flat_area_ratio'] for g in geoms):.4f}",
        )
        b33_80 = _rows(mesh_stage, panels=80, measure="flat")
        b33_320 = _rows(mesh_stage, panels=320, measure="flat")
        if b33_80 and b33_320:
            e80, e320 = b33_80[0]["rel_err_B33"], b33_320[0]["rel_err_B33"]
            gate(
                "G2_damping_mesh_convergence",
                "|B33 - B33_analytic| / B33_analytic falls from 80 to 320 panels "
                "and is below 15 % at 320",
                e320,
                e320 < e80 and e320 <= THRESHOLD,
                f"80 panels {e80:.4f} -> 320 panels {e320:.4f}",
            )
        exact_320 = _rows(mesh_stage, panels=320, measure="exact_measure")
        if exact_320:
            worst = max(exact_320[0]["rel_err_A33"], exact_320[0]["rel_err_B33"])
            gate(
                "G3_exact_measure_320",
                "A33 and B33 at 320 panels with the exact surface measure within 15 % "
                "of the closed form",
                worst,
                worst <= THRESHOLD,
                "A33 {:.4f}, B33 {:.4f}".format(
                    exact_320[0]["rel_err_A33"], exact_320[0]["rel_err_B33"]
                ),
            )
        ladder = mesh_stage.get("mode_ladder") or []
        if len(ladder) >= 2:
            change = abs(ladder[-1]["B33"] - ladder[-2]["B33"]) / abs(ladder[-1]["B33"])
            gate(
                "G4_mode_truncation",
                "B33 relative change over the last truncation step below 1 %",
                change,
                change <= 0.01,
                f"{ladder[-2]['mode_count']} -> {ladder[-1]['mode_count']} modes",
            )
        cap320 = _rows(mesh_stage, panels=320, measure="capytaine_builtin")
        if cap320:
            worst = max(cap320[0]["rel_err_A33"], cap320[0]["rel_err_B33"])
            gate(
                "G5_capytaine_analytic_320",
                "route C (independent code) at 320 panels within 15 % of the closed form",
                worst,
                worst <= THRESHOLD,
            )

    depth_stage = result.get("depth")
    if depth_stage:
        diffs = [
            max(r["rel_diff_A33"], r["rel_diff_B33"])
            for r in depth_stage["rows"]
            if "rel_diff_A33" in r
        ]
        gate(
            "G6_xval_depth_grid",
            "max |T+ - C| / C over the depth ladder within 15 %",
            max(diffs) if diffs else None,
            bool(diffs) and max(diffs) <= THRESHOLD,
            f"{len(diffs)} depth stations, matched mesh",
        )
        row20 = [r for r in depth_stage["rows"] if r["depth_ratio"] == 20.0]
        if row20:
            lim = max(row20[0]["rel_err_vs_analytic_A33"],
                      row20[0]["rel_err_vs_analytic_B33"])
            gate(
                "G7_deep_water_limit",
                "T+ at h/a = 20 within 15 % of the deep-water closed form",
                lim,
                lim <= THRESHOLD,
            )

    freq_stage = result.get("freq")
    if freq_stage:
        diffs = [
            max(r["rel_diff_A33"], r["rel_diff_B33"])
            for r in freq_stage["rows"]
            if "rel_diff_A33" in r
        ]
        gate(
            "G8_xval_freq_grid",
            "max |T+ - C| / C over the frequency ladder within 15 %",
            max(diffs) if diffs else None,
            bool(diffs) and max(diffs) <= THRESHOLD,
            f"{len(diffs)} (ka, depth) stations",
        )
        errs = [
            max(r["rel_err_vs_analytic_A33"], r["rel_err_vs_analytic_B33"])
            for r in freq_stage["rows"]
            if "rel_err_vs_analytic_A33" in r
        ]
        gate(
            "G9_deep_water_freq",
            "T+ in deep water within 15 % of the closed form over the ka ladder",
            max(errs) if errs else None,
            bool(errs) and max(errs) <= THRESHOLD,
        )

    con = result.get("consistency")
    if con:
        gate(
            "G10_reciprocity",
            "max|A - A^T|/max|A| and max|B - B^T|/max|B| below 1e-3",
            max(con["reciprocity_A"], con["reciprocity_B"]),
            max(con["reciprocity_A"], con["reciprocity_B"]) <= 1.0e-3,
        )
        gate(
            "G11_symmetry",
            "|A11 - A22| / A11 below 5e-2 (the geodesic mesh is not 4-fold symmetric) "
            "and surge/heave decoupling below 1e-3",
            max(con["axisymmetry_A11_A22"], con["surge_heave_coupling"]),
            max(con["axisymmetry_A11_A22"], con["surge_heave_coupling"]) <= 5.0e-2,
        )

    failed = [g["id"] for g in gates if not g["passed"]]
    return {
        "case": "xval_bem_finite_depth",
        "threshold": THRESHOLD,
        "gates": gates,
        "failed_gates": failed,
        "all_gates_passed": not failed,
    }


def route_status() -> dict[str, Any]:
    ok, note = capytaine_available()
    return {
        "capytaine_available": ok,
        "capytaine_note": note,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


def run(stages: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "case": "xval_bem_finite_depth",
        "case_definition": {
            "body": "geodesic sphere, closed, fully submerged",
            "radius": RADIUS,
            "submergence": SUBMERGENCE,
            "density": RHO,
            "gravity": GRAVITY,
            "center": CENTER,
            "work_subdivision": WORK_SUBDIVISION,
            "work_mode_count": WORK_MODE_COUNT,
            "depth_ratios": DEPTH_RATIOS,
            "ka_grid": KA_GRID,
            "mode_ladder": MODE_LADDER,
            "threshold": THRESHOLD,
        },
        "environment": route_status(),
    }
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    for name in stages:
        t0 = time.time()
        payload = STAGES[name]()
        payload["seconds"] = time.time() - t0
        (CASE_DIR / f"stage_{name}.json").write_text(
            json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"[xval] stage {name} done in {payload['seconds']:.1f}s", flush=True)
    return result


def merge(out_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "case": "xval_bem_finite_depth",
        "environment": route_status(),
    }
    exemplar = out_dir / "stage_mesh.json"
    if exemplar.is_file():
        payload = json.loads(exemplar.read_text(encoding="utf-8"))
        result["case_definition"] = {
            "body": "geodesic sphere, closed, fully submerged",
            "radius": RADIUS,
            "submergence": SUBMERGENCE,
            "density": RHO,
            "gravity": GRAVITY,
            "center": CENTER,
            "work_subdivision": WORK_SUBDIVISION,
            "work_mode_count": WORK_MODE_COUNT,
            "depth_ratios": DEPTH_RATIOS,
            "ka_grid": KA_GRID,
            "mode_ladder": MODE_LADDER,
            "threshold": THRESHOLD,
        }
    for name in STAGES:
        path = out_dir / f"stage_{name}.json"
        if path.is_file():
            result[name] = json.loads(path.read_text(encoding="utf-8"))
    result["gates"] = evaluate_gates(result)
    result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="XVAL-BEM-FINITE-DEPTH case driver")
    parser.add_argument(
        "--stage",
        default="all",
        choices=("all", "merge", *STAGES.keys()),
        help="stage to run; 'merge' only re-evaluates the gates from stored stages",
    )
    parser.add_argument("--output", default=str(CASE_DIR))
    args = parser.parse_args(argv)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "merge":
        result = merge(out_dir)
    else:
        stages = list(STAGES) if args.stage == "all" else [args.stage]
        partial = run(stages)
        result = merge(out_dir)
        result["environment"] = partial["environment"]
    (out_dir / "gates.json").write_text(
        json.dumps(result["gates"], indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["gates"], indent=1)[:4000])
    return 0 if result["gates"]["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())