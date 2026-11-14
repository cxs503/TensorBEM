#!/usr/bin/env python3
"""Produce the independent capytaine reference for the finite-depth cross-check.

This is the *only* place in the repository that asks capytaine for finite-depth
numbers.  Like ``scripts/capytaine_crosscheck_ref.py`` it is a producer, not a
test dependency: capytaine is optional (pure Python + a C extension built from
source for loongarch64), so its numbers are frozen into
``benchmarks/verified/capytaine_finite_depth/capytaine_ref.json`` together with
full provenance.

The plan (geometry, ``h/a x k0*a`` grid, panel ladders, criteria) lives in
:mod:`tensorbem.benchmarks.finite_depth_crossval_plan` and is shared with the
consumer so the two codes cannot drift apart.

Usage (needs capytaine importable)::

    PYTHONPATH=src python scripts/capytaine_finite_depth_ref.py \
        --out benchmarks/verified/capytaine_finite_depth/capytaine_ref.json

Reference ``nu`` is back-computed per cell from ``k0 tanh(k0 h) = nu`` so that
``k0*a`` is exact at every depth.  For each cell the reference is evaluated at
three resolutions with ``finite_depth_method="newer"`` plus once at the middle
resolution with ``finite_depth_method="legacy"``: the spread of the two
formulations is the reference's own uncertainty band.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from tensorbem.benchmarks import finite_depth_crossval_plan as plan


def _capytaine_provenance(cpt: Any) -> dict[str, Any]:
    return {
        "tool": "capytaine",
        "version": getattr(cpt, "__version__", "unknown"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "density": plan.DENSITY,
        "gravity": plan.GRAVITY,
        "water_depth": "finite (per cell)",
        "finite_depth_method": plan.CAPYTAINE_METHOD,
        "finite_depth_legacy_method": plan.CAPYTAINE_LEGACY_METHOD,
        "producer": "scripts/capytaine_finite_depth_ref.py",
        "plan_module": "tensorbem.benchmarks.finite_depth_crossval_plan",
        "resolutions": [list(r) for r in plan.CAPYTAINE_RESOLUTIONS],
        "legacy_resolution": list(plan.CAPYTAINE_LEGACY_RESOLUTION),
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "note": (
            "capytaine built from source (no loongarch64 wheel on the index); "
            "only this producer script imports it"
        ),
    }


def _mesh(cpt: Any, resolution: tuple[int, int]) -> Any:
    return cpt.mesh_sphere(
        radius=plan.SPHERE_RADIUS,
        center=(0.0, 0.0, -plan.SPHERE_SUBMERGENCE),
        resolution=tuple(int(v) for v in resolution),
    ).immersed_part()


def _solve_cell(
    cpt: Any,
    mesh: Any,
    cell: dict[str, float],
    method: str,
) -> dict[str, float]:
    """Heave + surge radiation coefficients of one cell with one GF formulation."""
    solver = cpt.BEMSolver(
        green_function=cpt.Delhommeau(finite_depth_method=method)
    )
    body = cpt.FloatingBody(mesh=mesh)
    body.add_all_rigid_body_dofs()
    out: dict[str, float] = {}
    for dof, name in ((0, "surge"), (2, "heave")):
        problem = cpt.RadiationProblem(
            body=body,
            omega=cell["omega"],
            radiating_dof=name.capitalize(),
            water_depth=cell["depth"],
            rho=plan.DENSITY,
            g=plan.GRAVITY,
        )
        result = solver.solve(problem)
        out[name] = {
            "added_mass": float(result.added_mass[name.capitalize()]),
            "damping": float(result.radiation_damping[name.capitalize()]),
        }
    return out


def _record(
    cpt: Any,
    cell: dict[str, float],
    resolutions: list[tuple[int, int]],
    legacy_resolution: tuple[int, int],
    method: str,
    legacy_method: str,
) -> dict[str, Any]:
    levels = []
    for resolution in resolutions:
        mesh = _mesh(cpt, resolution)
        t0 = time.perf_counter()
        values = _solve_cell(cpt, mesh, cell, method)
        levels.append(
            {
                "resolution": list(resolution),
                "panels": int(mesh.nb_faces),
                "seconds": round(time.perf_counter() - t0, 3),
                "values": values,
            }
        )
    legacy_mesh = _mesh(cpt, legacy_resolution)
    legacy_values = _solve_cell(cpt, legacy_mesh, cell, legacy_method)
    deep_values = _solve_infinite_depth(cpt, legacy_mesh, cell)
    return {
        "cell": plan.cell_id(cell["h_over_a"], cell["ka"]),
        "h_over_a": cell["h_over_a"],
        "ka": cell["ka"],
        "k0h": cell["k0h"],
        "depth": cell["depth"],
        "k0": cell["k0"],
        "nu": cell["nu"],
        "omega": cell["omega"],
        "levels": levels,
        "legacy": {
            "method": legacy_method,
            "resolution": list(legacy_resolution),
            "panels": int(legacy_mesh.nb_faces),
            "values": legacy_values,
        },
        "infinite_depth": {
            "note": "same mesh and omega, water_depth=inf -- the depth-free control",
            "resolution": list(legacy_resolution),
            "values": deep_values,
        },
    }


def _solve_infinite_depth(
    cpt: Any, mesh: Any, cell: dict[str, float]
) -> dict[str, dict[str, float]]:
    solver = cpt.BEMSolver()
    body = cpt.FloatingBody(mesh=mesh)
    body.add_all_rigid_body_dofs()
    out: dict[str, dict[str, float]] = {}
    for name in ("surge", "heave"):
        problem = cpt.RadiationProblem(
            body=body,
            omega=cell["omega"],
            radiating_dof=name.capitalize(),
            water_depth=math.inf,
            rho=plan.DENSITY,
            g=plan.GRAVITY,
        )
        result = solver.solve(problem)
        out[name] = {
            "added_mass": float(result.added_mass[name.capitalize()]),
            "damping": float(result.radiation_damping[name.capitalize()]),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/verified/capytaine_finite_depth/capytaine_ref.json"),
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=0,
        help="produce only the first N cells (smoke runs); 0 = all",
    )
    args = parser.parse_args(argv)

    import capytaine as cpt  # imported here: optional dependency

    payload: dict[str, Any] = {
        "provenance": _capytaine_provenance(cpt),
        "plan": {
            "depth_ratios": list(plan.DEPTH_RATIOS),
            "ka_grid": list(plan.KA_GRID),
            "sphere_radius": plan.SPHERE_RADIUS,
            "sphere_submergence": plan.SPHERE_SUBMERGENCE,
            "resolutions": [list(r) for r in plan.CAPYTAINE_RESOLUTIONS],
            "legacy_resolution": list(plan.CAPYTAINE_LEGACY_RESOLUTION),
        },
        "cells": [],
    }
    started = time.perf_counter()
    for index, cell in enumerate(plan.cells()):
        if args.max_cells and index >= args.max_cells:
            break
        record = _record(
            cpt,
            cell,
            list(plan.CAPYTAINE_RESOLUTIONS),
            plan.CAPYTAINE_LEGACY_RESOLUTION,
            plan.CAPYTAINE_METHOD,
            plan.CAPYTAINE_LEGACY_METHOD,
        )
        payload["cells"].append(record)
        print(
            f"[{index:2d}] {record['cell']} k0h={record['k0h']:.2f} "
            f"panels={[lv['panels'] for lv in record['levels']]}",
            flush=True,
        )
    payload["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n")
    print(f"wrote {args.out} ({len(payload['cells'])} cells, "
          f"{payload['elapsed_seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())