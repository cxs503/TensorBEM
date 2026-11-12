#!/usr/bin/env python3
"""Produce the independent capytaine reference for the TensorBEM cross-check.

This script is the *only* place in the repository that imports ``capytaine``.
It is a producer, not part of the test suite: capytaine is an optional external
dependency (pure Python + C extension, built here from source with
``meson-python`` on loongarch64), so the numbers it produces are frozen into
``benchmarks/verified/capytaine_crosscheck/capytaine_ref.json`` together with
the full provenance (version, git commit, Python, platform, mesh resolutions).

Usage (needs capytaine importable)::

    python scripts/capytaine_crosscheck_ref.py \
        --out benchmarks/verified/capytaine_crosscheck/capytaine_ref.json

The reference geometries are the smooth bodies where a low-order Rankine panel
method is trustworthy: a surface-piercing hemisphere (also anchored to Hulme's
1982 semi-analytical solution) and a floating sphere whose waterline is a
circle of radius sqrt(a^2 - z0^2).  The wall-sided barge is *kept* in the file
as an explicit negative control: at 576 panels capytaine still returns negative
heave added mass / damping, i.e. the case never leaves the spurious
(irregular-frequency) regime -- documenting that is more useful than silently
dropping the geometry.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

DENSITY = 1025.0
GRAVITY = 9.80665
NU_GRID = (0.5, 1.0, 2.0)


def _capytaine_provenance(module: Any, source: Path | None) -> dict[str, Any]:
    commit = None
    if source is not None and (source / ".git").exists():
        try:
            commit = subprocess.run(
                ["git", "-C", str(source), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:  # pragma: no cover - provenance only
            commit = None
    return {
        "tool": "capytaine",
        "version": getattr(module, "__version__", "unknown"),
        "git_commit": commit,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "density": DENSITY,
        "gravity": GRAVITY,
        "water_depth": "infinite",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "note": (
            "built from source (pip --no-build-isolation --no-deps) because no "
            "loongarch64 wheel exists on the configured index; only this "
            "producer script imports capytaine"
        ),
    }


def _mesh_hemisphere(cpt: Any, resolution: tuple[int, int]) -> Any:
    return cpt.mesh_sphere(
        radius=1.0, center=(0.0, 0.0, 0.0), resolution=resolution
    ).immersed_part()


def _mesh_sphere(cpt: Any, resolution: tuple[int, int]) -> Any:
    return cpt.mesh_sphere(
        radius=1.0, center=(0.0, 0.0, -0.5), resolution=resolution
    ).immersed_part()


def _mesh_barge(cpt: Any, resolution: tuple[int, int, int]) -> Any:
    return cpt.mesh_parallelepiped(
        size=(2.0, 1.0, 1.0),
        center=(0.0, 0.0, -0.5),
        resolution=resolution,
    ).immersed_part()


def _sweep(cpt: Any, solver: Any, mesh: Any, dofs: tuple[str, ...]) -> dict[str, Any]:
    body = cpt.FloatingBody(mesh=mesh)
    body.add_all_rigid_body_dofs()
    out: dict[str, Any] = {}
    for nu in NU_GRID:
        omega = math.sqrt(nu * GRAVITY)
        per_dof = {}
        for dof in dofs:
            res = solver.solve(
                cpt.RadiationProblem(
                    body=body,
                    omega=omega,
                    radiating_dof=dof,
                    water_depth=np.inf,
                    rho=DENSITY,
                    g=GRAVITY,
                )
            )
            per_dof[dof.lower()] = {
                "added_mass": float(res.added_mass[dof]),
                "damping": float(res.radiation_damping[dof]),
            }
        out[f"{nu:.3f}"] = per_dof
    return out


def _study(
    cpt: Any,
    solver: Any,
    mesh_factory: Any,
    levels: tuple[Any, ...],
    dofs: tuple[str, ...],
) -> dict[str, Any]:
    rows = []
    for level in levels:
        mesh = mesh_factory(cpt, level)
        rows.append(
            {
                "level": list(level) if isinstance(level, tuple) else [level],
                "n_faces": int(mesh.nb_faces),
                "volume": float(mesh.volume),
                "values": _sweep(cpt, solver, mesh, dofs),
            }
        )
    return {
        "levels": rows,
        "reference": rows[-1]["values"],
        "reference_level": rows[-1]["level"],
        "reference_faces": rows[-1]["n_faces"],
    }


def build(out_path: Path, capytaine_source: Path | None) -> dict[str, Any]:
    import capytaine as cpt

    solver = cpt.BEMSolver()
    payload: dict[str, Any] = {
        "metadata": _capytaine_provenance(cpt, capytaine_source),
        "nu_grid": list(NU_GRID),
        "cases": {},
    }
    payload["cases"]["hemisphere_r1"] = {
        "kind": "surface-piercing hemisphere, radius 1.0, centre (0,0,0)",
        "dofs": ["heave", "surge"],
        **_study(
            cpt,
            solver,
            _mesh_hemisphere,
            ((6, 12), (12, 24), (20, 40), (28, 56), (36, 72)),
            ("Heave", "Surge"),
        ),
    }
    payload["cases"]["sphere_r1_z-0.5"] = {
        "kind": "floating sphere, radius 1.0, centre (0,0,-0.5)",
        "dofs": ["heave"],
        **_study(
            cpt, solver, _mesh_sphere, ((12, 24), (20, 40), (28, 56)), ("Heave",)
        ),
    }
    barge = _study(
        cpt,
        solver,
        _mesh_barge,
        ((2, 2, 2), (4, 4, 2), (8, 8, 4), (12, 12, 6)),
        ("Heave", "Pitch"),
    )
    barge["excluded"] = True
    barge["exclusion_reason"] = (
        "wall-sided barge with sharp bilge/waterline corners never leaves the "
        "spurious irregular-frequency regime: even at 576 panels the heave "
        "added mass and damping change sign between frequencies, so no "
        "converged reference exists to compare against"
    )
    payload["cases"]["barge_L2_B1_T0.5"] = {
        "kind": "barge L=2.0 B=1.0 T=0.5 (negative control)",
        "dofs": ["heave", "pitch"],
        **barge,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "benchmarks/verified/capytaine_crosscheck/capytaine_ref.json"
        ),
    )
    parser.add_argument(
        "--capytaine-source",
        type=Path,
        default=None,
        help="optional git checkout of capytaine, recorded as provenance",
    )
    args = parser.parse_args(argv)
    payload = build(args.out, args.capytaine_source)
    print(f"wrote {args.out}")
    print(f"capytaine {payload['metadata']['version']} "
          f"commit={payload['metadata']['git_commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())