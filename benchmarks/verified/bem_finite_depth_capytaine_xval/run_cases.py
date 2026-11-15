#!/usr/bin/env python3
"""Per-case producer for the finite-depth cross-validation (TensorBEM vs capytaine).

Why a separate producer
-----------------------
The study is a (depth x frequency x mesh) grid of *independent* solves.  Running
it as one monolith makes it unshardable, uncacheable and, when a third-party
solver's return type changes, un-debuggable.  This producer solves one case at a
time, freezes the payload in ``cases/<case_id>.json`` and skips cases that are
already on disk, so a cluster run can be sharded, resumed, or interleaved with a
diagnostic probe.

Contract
--------
* ``--which tbem``: TensorBEM finite-depth radiation (heave/surge/sway) and, for
  the ``inf`` station, the infinite-depth free-surface solver.
* ``--which cpt``: capytaine radiation + diffraction at four mesh resolutions,
  plus the deep-water stations.
* ``--which all``: both.
* ``--shard I/N``: run only the cases with index ``I (mod N)``.
* ``--exact-areas``: weight TensorBEM panels by their exact spherical patch area
  instead of the flat-triangle area (suffix ``ea`` on the case id).

Every payload carries its own provenance (solver version, mode count, panel
count, wall time) so the comparison stage never has to guess.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

CASE_DIR = Path("benchmarks/verified/bem_finite_depth_capytaine_xval/cases")

RADIUS = 1.0
SUBMERGENCE = 2.0  # sphere centre at z = -2, lowest point z = -3
DENSITY = 1025.0
GRAVITY = 9.80665

#: ka grid (k = omega^2 / g, a = radius; a = 1 m so nu = ka)
KA_GRID: tuple[float, ...] = (0.3, 0.6, 1.0, 1.5, 2.0)
#: Depth stations h/R.  The shallowest station keeps a 1 m clearance between the
#: sphere's lowest point (z = -3) and the sea bed; h/R = 3 would put the sphere
#: exactly on the bottom (zero gap, degenerate BVP) and is therefore excluded.
DEPTH_RATIOS: tuple[float, ...] = (4.0, 5.0, 8.0, 12.0, 20.0)
#: capytaine mesh ladder (meridian, parallel) -> 128/288/512/1152 faces
CPT_LEVELS: tuple[tuple[int, int], ...] = ((8, 16), (12, 24), (16, 32), (24, 48))
#: icosphere subdivisions -> 320/1280 faces
TBEM_SUBDIVISIONS: tuple[int, ...] = (2, 3)


def _case_id(prefix: str, depth_ratio: Any, ka: float, extra: str) -> str:
    tag = "inf" if depth_ratio == "inf" else f"{float(depth_ratio):g}"
    return f"{prefix}-{extra}-h{tag}-ka{ka:g}"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# TensorBEM
# ---------------------------------------------------------------------------
def _tbem_solve(ka: float, depth_ratio: Any, subdivisions: int, exact_areas: bool,
                mode_count: int | None = None) -> dict[str, Any]:
    from tensorbem.geometry3d import submerged_sphere_mesh, spherical_patch_areas

    mesh = submerged_sphere_mesh(subdivisions, radius=RADIUS, submergence=SUBMERGENCE)
    nu = ka / RADIUS
    areas = None
    if exact_areas:
        areas = spherical_patch_areas(
            mesh.vertices, mesh.faces, center=(0.0, 0.0, -SUBMERGENCE), radius=RADIUS
        )

    if depth_ratio == "inf":
        from tensorbem.free_surface_solver3d import solve_radiation_3d

        res = solve_radiation_3d(
            mesh, nu, modes=[0, 1, 2], density=DENSITY, gravity=GRAVITY,
            panel_areas=areas,
        )
        ops = None
        depth = None
    else:
        from tensorbem.finite_depth_solver3d import (
            assemble_finite_depth_operators_3d,
            solve_radiation_finite_depth_3d,
        )

        depth = float(depth_ratio) * RADIUS
        ops = assemble_finite_depth_operators_3d(
            mesh, nu, depth, mode_count=mode_count, panel_areas=areas
        )
        res = solve_radiation_finite_depth_3d(
            mesh, nu, depth, modes=[0, 1, 2], density=DENSITY, gravity=GRAVITY,
            operators=ops, panel_areas=areas,
        )

    am = np.asarray(res.added_mass, dtype=float)
    bd = np.asarray(res.damping, dtype=float)
    return {
        "tool": "TensorBEM",
        "ka": ka,
        "depth_ratio": depth_ratio,
        "depth": depth,
        "subdivisions": int(subdivisions),
        "n_panels": int(mesh.panel_count),
        "exact_areas": bool(exact_areas),
        "mode_count": int(ops.mode_count) if ops is not None else None,
        "added_mass_surge": float(am[0]),
        "added_mass_sway": float(am[1]),
        "added_mass_heave": float(am[2]),
        "damping_surge": float(bd[0]),
        "damping_sway": float(bd[1]),
        "damping_heave": float(bd[2]),
        "condition_number": float(res.condition_number),
        "relative_residual": float(res.relative_residual),
    }


def tbem_case(ka, depth_ratio, subdivisions, exact_areas) -> dict[str, Any]:
    suffix = f"sub{subdivisions}" + ("ea" if exact_areas else "")
    cid = _case_id("tbem", depth_ratio, ka, suffix)
    t0 = time.time()
    try:
        payload = _tbem_solve(ka, depth_ratio, subdivisions, exact_areas)
        payload["status"] = "ok"
    except Exception:  # pragma: no cover - diagnostics
        payload = {
            "tool": "TensorBEM", "ka": ka, "depth_ratio": depth_ratio,
            "subdivisions": int(subdivisions), "exact_areas": bool(exact_areas),
            "error": traceback.format_exc(), "status": "error",
        }
    payload["case_id"] = cid
    payload["seconds"] = round(time.time() - t0, 1)
    payload["finished_utc"] = _now()
    payload["python"] = platform.python_version()
    return payload


# ---------------------------------------------------------------------------
# capytaine
# ---------------------------------------------------------------------------
def _cpt_scalar(field: Any, dof: str) -> float:
    """Extract one scalar from capytaine's ``{radiating: {influenced: value}}``.

    capytaine >= 3 returns plain nested dicts, capytaine 2.x returns xarray
    DataArrays indexed by dof names.  Both are accepted; the nested-dict route is
    the one the installed 3.1.dev0 takes.
    """
    if isinstance(field, dict):
        inner = field.get(dof, field)
        if isinstance(inner, dict):
            inner = inner.get(dof, next(iter(inner.values())))
        return float(np.asarray(inner, dtype=float).ravel()[0])
    # xarray / array fallback
    try:
        return float(np.asarray(field.sel(radiating_dof=dof, influenced_dof=dof)).ravel()[0])
    except Exception:  # pragma: no cover
        return float(np.asarray(field, dtype=float).ravel().max())


def cpt_solve(ka: float, depth_ratio: Any, resolution: tuple[int, int]) -> dict[str, Any]:
    import warnings

    import capytaine as cpt

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mesh = cpt.mesh_sphere(
            radius=RADIUS, center=(0.0, 0.0, -SUBMERGENCE), resolution=resolution
        )
        body = cpt.FloatingBody(mesh=mesh, name="submerged_sphere")
        body.add_translation_dof(name="Surge", direction=(1.0, 0.0, 0.0))
        body.add_translation_dof(name="Sway", direction=(0.0, 1.0, 0.0))
        body.add_translation_dof(name="Heave", direction=(0.0, 0.0, 1.0))

        omega = math.sqrt(GRAVITY * ka / RADIUS)
        water_depth = np.inf if depth_ratio == "inf" else float(depth_ratio) * RADIUS
        solver = cpt.BEMSolver()

        out: dict[str, Any] = {}
        for dof, key in (("Surge", "surge"), ("Sway", "sway"), ("Heave", "heave")):
            problem = cpt.RadiationProblem(
                body=body, omega=omega, water_depth=water_depth, rho=DENSITY,
                g=GRAVITY, radiating_dof=dof,
            )
            res = solver.solve(problem)
            out[f"added_mass_{key}"] = _cpt_scalar(res.added_mass, dof)
            out[f"damping_{key}"] = _cpt_scalar(res.radiation_damping, dof)

        # Diffraction: excitation = diffraction force + Froude-Krylov force
        diff = cpt.DiffractionProblem(
            body=body, omega=omega, water_depth=water_depth, rho=DENSITY,
            g=GRAVITY, wave_direction=0.0,
        )
        res_d = solver.solve(diff)
        try:
            from capytaine.bem.airy_waves import froude_krylov_force

            fk = froude_krylov_force(diff)
            fk_h = complex(fk["Heave"])
        except Exception:  # pragma: no cover - version drift
            fk = {}
            fk_h = 0.0j
        f_d = complex(res_d.forces["Heave"])
        out["excitation_heave_real"] = float((f_d + fk_h).real)
        out["excitation_heave_imag"] = float((f_d + fk_h).imag)
        out["excitation_heave_mag"] = float(abs(f_d + fk_h))
        out["fk_heave_mag"] = float(abs(fk_h))
        out["diffraction_heave_mag"] = float(abs(f_d))
        out["n_faces"] = int(mesh.nb_faces)
        out["n_vertices"] = int(mesh.nb_vertices)
        out["capytaine_version"] = getattr(cpt, "__version__", "unknown")
        return out


def cpt_case(ka, depth_ratio, resolution) -> dict[str, Any]:
    cid = _case_id("cpt", depth_ratio, ka, f"{resolution[0]}x{resolution[1]}")
    t0 = time.time()
    try:
        payload = cpt_solve(ka, depth_ratio, resolution)
        payload["status"] = "ok"
    except Exception:  # pragma: no cover - diagnostics
        payload = {
            "tool": "capytaine", "ka": ka, "depth_ratio": depth_ratio,
            "resolution": list(resolution), "error": traceback.format_exc(),
            "status": "error",
        }
    payload["case_id"] = cid
    payload["tool"] = "capytaine"
    payload["ka"] = ka
    payload["depth_ratio"] = depth_ratio
    payload["depth"] = None if depth_ratio == "inf" else float(depth_ratio) * RADIUS
    payload["resolution"] = list(resolution)
    payload["seconds"] = round(time.time() - t0, 2)
    payload["finished_utc"] = _now()
    return payload


# ---------------------------------------------------------------------------
# Plan + driver
# ---------------------------------------------------------------------------
def plan(which: str, exact_areas: bool, full_sub3: bool) -> list[tuple[str, tuple]]:
    stations: list[Any] = list(DEPTH_RATIOS) + ["inf"]
    cases: list[tuple[str, tuple]] = []
    if which in ("tbem", "all"):
        for depth_ratio in stations:
            for ka in KA_GRID:
                cases.append(("tbem", (ka, depth_ratio, 2, exact_areas)))
                if full_sub3 or ka == 1.0:
                    cases.append(("tbem", (ka, depth_ratio, 3, exact_areas)))
    if which in ("cpt", "all"):
        for level in CPT_LEVELS:
            for depth_ratio in stations:
                for ka in KA_GRID:
                    cases.append(("cpt", (ka, depth_ratio, level)))
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=("tbem", "cpt", "all"), default="all")
    ap.add_argument("--shard", default="0/1", help="I/N, I in [0, N)")
    ap.add_argument("--exact-areas", action="store_true")
    ap.add_argument("--full-sub3", action="store_true",
                    help="run subdivision 3 on every ka, not just ka = 1")
    ap.add_argument("--force", action="store_true", help="recompute cached cases")
    args = ap.parse_args()

    idx, nshard = (int(x) for x in args.shard.split("/"))
    if not 0 <= idx < nshard:
        raise SystemExit("shard index out of range")

    CASE_DIR.mkdir(parents=True, exist_ok=True)
    cases = plan(args.which, args.exact_areas, args.full_sub3)
    mine = [c for i, c in enumerate(cases) if i % nshard == idx]
    print(f"shard {idx}/{nshard}: {len(mine)} of {len(cases)} cases", flush=True)

    for kind, arg in mine:
        producer = tbem_case if kind == "tbem" else cpt_case
        cid = (
            _case_id("tbem", arg[1], arg[0],
                     f"sub{arg[2]}" + ("ea" if arg[3] else ""))
            if kind == "tbem"
            else _case_id("cpt", arg[1], arg[0], f"{arg[2][0]}x{arg[2][1]}")
        )
        target = CASE_DIR / f"{cid}.json"
        if target.exists() and not args.force:
            try:
                if json.loads(target.read_text()).get("status") == "ok":
                    print(f"  skip {cid} (cached)", flush=True)
                    continue
            except json.JSONDecodeError:  # pragma: no cover
                pass
        payload = producer(*arg)
        path = CASE_DIR / f"{payload['case_id']}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"  {payload['case_id']}: {payload['status']} {payload['seconds']}s",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())