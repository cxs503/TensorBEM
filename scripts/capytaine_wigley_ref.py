#!/usr/bin/env python3
"""Producer of the *external-code* reference for CASE-BEM-CUMMINS-WIGLEY.

Why this file exists
--------------------
The Cummins/Wigley case has passed its own gates (C0..C9), but every one of them
is a **self-consistency** statement: geometry contract (C1), panel resolution
(C2), Kramers-Kronig closure of the same sweep (C3/C3a), Cummins transform round
trip (C4), time-domain decay against the same frequency-domain damping (C5),
negative controls (C6), the added-mass anchor of the same solver (C7).  None of
them can see a *shared* modelling error -- a wrong free-surface Green function,
a wrong $\\omega$ normalisation, a wrong density convention would pass all of
them.  The case documentation therefore records the external (L5) anchor as
``WITHHELD``.

This producer closes that hole by solving the **same radiation problems** in an
independent code, ``capytaine`` (a separate BEM implementation with its own
Green-function evaluation, its own quadrature and its own solver), on the
**identical panel geometry**: the vertices and triangles are read straight out
of :func:`tensorbem.wigley_hydrostatics.wigley_wetted_hull_mesh`, i.e. the exact
mesh the TensorBEM sweep used.  Comparing on the same panels isolates the
*numerical method* from the *discretisation*: any deviation left is method, not
mesh.  A second axis (``no_lid`` vs ``lid``) tests the one hypothesis that can
explain a top-of-band disagreement -- capytaine's irregular frequencies, which
TensorBEM suppresses with CHIEF points.

Only this producer imports ``capytaine`` (same optional-dependency pattern as
``scripts/capytaine_crosscheck_r2_ref.py``); the comparison, the gate and the
tests consume the frozen JSON and never import it.

Usage::

    PYTHONPATH=src python3 scripts/capytaine_wigley_ref.py \\
        --out benchmarks/verified/casepool_bem_cummins_wigley/capytaine_ref.json

The run is single-process and takes a few minutes (three mesh levels x 41
frequencies x 6 radiating dofs x 2 lid variants, with the influence matrix
factorisation cached per frequency by capytaine).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tensorbem.benchmarks import case_bem_cummins_wigley as case
from tensorbem.wigley_hydrostatics import wigley_wetted_hull_mesh

#: Panel levels, keyed exactly as the committed TensorBEM sweeps so the two
#: sides cannot be compared across different discretisations by accident.
LEVELS: dict[str, tuple[int, int]] = {
    "coarse": case.MESHES["coarse"],
    "medium": case.MESHES["medium"],
    "fine": (30, 9),
}

#: Lid variants.  ``no_lid`` is what a naive capytaine user runs; ``lid`` adds
#: the interior free-surface panel that removes the irregular frequencies.
VARIANTS = ("no_lid", "lid")

DOF_ORDER = case.DOF_NAMES  # ("surge", ..., "yaw")
_CAPYTAINE_DOF = {
    "surge": "Surge",
    "sway": "Sway",
    "heave": "Heave",
    "roll": "Roll",
    "pitch": "Pitch",
    "yaw": "Yaw",
}


def geometry_identity(n_sections: int, n_levels: int) -> dict[str, Any]:
    """Fingerprint the TensorBEM panel geometry the reference is built on.

    The comparison module recomputes this fingerprint from
    :func:`wigley_wetted_hull_mesh` before it scores anything, so a reference
    produced on a different body, a different rotation centre or a different
    panel layout cannot be silently reused.
    """
    mesh = wigley_wetted_hull_mesh(
        case.LENGTH, case.BEAM, case.DRAFT, n_sections=n_sections, n_levels=n_levels
    )
    verts = np.ascontiguousarray(np.asarray(mesh.vertices, dtype=float))
    faces = np.ascontiguousarray(np.asarray(mesh.faces, dtype=np.int64))
    digest = hashlib.sha256()
    digest.update(verts.tobytes())
    digest.update(faces.tobytes())
    return {
        "n_sections": int(n_sections),
        "n_levels": int(n_levels),
        "panel_count": int(mesh.panel_count),
        "vertex_count": int(verts.shape[0]),
        "face_count": int(faces.shape[0]),
        "surface_area_m2": float(np.asarray(mesh.areas, dtype=float).sum()),
        "sha256": digest.hexdigest(),
    }


def _capytaine_mesh(n_sections: int, n_levels: int):
    """capytaine mesh built from the *TensorBEM* vertices/faces, no re-meshing."""
    import capytaine as cpt

    mesh = wigley_wetted_hull_mesh(
        case.LENGTH, case.BEAM, case.DRAFT, n_sections=n_sections, n_levels=n_levels
    )
    return cpt.Mesh(
        np.asarray(mesh.vertices, dtype=float),
        np.asarray(mesh.faces, dtype=np.int64),
        name=f"wigley_{n_sections}x{n_levels}",
    )


def solve_level(
    n_sections: int,
    n_levels: int,
    omega_grid: np.ndarray,
    *,
    lid: bool,
) -> dict[str, Any]:
    """Solve the six radiation problems at every frequency of the grid."""
    import capytaine as cpt

    hull = _capytaine_mesh(n_sections, n_levels)
    lid_mesh = hull.generate_lid(z=0.0) if lid else None
    body = cpt.FloatingBody(
        mesh=hull,
        dofs=cpt.rigid_body_dofs(rotation_center=case.CENTER),
        lid_mesh=lid_mesh,
        name=hull.name,
    )
    names = [_CAPYTAINE_DOF[d] for d in DOF_ORDER]
    missing = [n for n in names if n not in body.dofs]
    if missing:
        raise SystemExit(f"capytaine did not expose the dofs {missing}")

    solver = cpt.BEMSolver()
    added = np.empty((omega_grid.size, 6, 6), dtype=float)
    damping = np.empty((omega_grid.size, 6, 6), dtype=float)
    for k, omega in enumerate(omega_grid):
        problems = [
            cpt.RadiationProblem(
                body=body,
                omega=float(omega),
                water_depth=float("inf"),
                rho=case.DENSITY,
                g=case.GRAVITY,
                radiating_dof=name,
            )
            for name in names
        ]
        results = solver.solve_all(problems)
        # results[j] is the field of the problem radiating in dof j; the whole
        # column A[:, j] / B[:, j] belongs to it.
        for j, res in enumerate(results):
            for i, name in enumerate(names):
                added[k, i, j] = float(res.added_mass[name])
                damping[k, i, j] = float(res.radiation_damping[name])
    return {
        "lid": bool(lid),
        "lid_panel_count": int(lid_mesh.nb_faces) if lid_mesh is not None else 0,
        "hull_panel_count": int(hull.nb_faces),
        "omega": omega_grid,
        "added_mass": added,
        "damping": damping,
    }


def _round(mat: np.ndarray, digits: int = 12) -> list[list[list[float]]]:
    return np.round(np.asarray(mat, dtype=float), digits).tolist()


def build_reference(
    omega_grid: np.ndarray | None = None,
    levels: dict[str, tuple[int, int]] | None = None,
    variants: tuple[str, ...] = VARIANTS,
) -> dict[str, Any]:
    """Assemble the frozen reference document (all levels, all lid variants)."""
    import capytaine as cpt

    grid = (
        np.asarray(omega_grid, dtype=float)
        if omega_grid is not None
        else case.sweep_grid(case.BAND, case.N_OMEGA)
    )
    table = levels or LEVELS
    payload: dict[str, Any] = {
        "case": case.CASE_SLUG,
        "kind": "external_code_reference",
        "code": "capytaine",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": {
            "capytaine_version": str(getattr(cpt, "__version__", "unknown")),
            "capytaine_commit": str(getattr(cpt, "__git_revision__", None)),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "settings": {
            "density": case.DENSITY,
            "gravity": case.GRAVITY,
            "water_depth": "inf",
            "rotation_center": list(case.CENTER),
            "dof_order": list(DOF_ORDER),
            "band_rad_s": list(case.BAND),
            "n_omega": int(grid.size),
            "lid": "capytaine generate_lid(z=0) for the variant named 'lid'",
        },
        "omega": np.round(grid, 12).tolist(),
        "levels": {},
    }
    for label, (n_sections, n_levels) in table.items():
        identity = geometry_identity(n_sections, n_levels)
        entry: dict[str, Any] = {"geometry": identity, "variants": {}}
        for variant in variants:
            solved = solve_level(
                n_sections, n_levels, grid, lid=(variant == "lid")
            )
            added = solved.pop("added_mass")
            damping = solved.pop("damping")
            solved.pop("omega", None)
            entry["variants"][variant] = {
                **solved,
                "added_mass": _round(added),
                "damping": _round(damping),
                "symmetry_error_added_mass": float(
                    np.max(np.abs(added - np.transpose(added, (0, 2, 1))))
                ),
                "symmetry_error_damping": float(
                    np.max(np.abs(damping - np.transpose(damping, (0, 2, 1))))
                ),
            }
        payload["levels"][label] = entry
        print(f"  level {label} ({identity['panel_count']} panels): done", flush=True)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "benchmarks/verified/casepool_bem_cummins_wigley/capytaine_ref.json"
        ),
    )
    parser.add_argument(
        "--level",
        action="append",
        choices=sorted(LEVELS),
        help="restrict the run to this level (repeatable); default: all levels",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=list(VARIANTS),
        help="restrict the run to this lid variant (repeatable); default: both",
    )
    parser.add_argument(
        "--merge",
        type=Path,
        nargs="+",
        help="merge previously written partial references and exit",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.merge:
        merged: dict[str, Any] = {}
        for path in args.merge:
            part = json.loads(path.read_text())
            for label, entry in part["levels"].items():
                slot = merged.setdefault(label, {"geometry": entry["geometry"], "variants": {}})
                if slot["geometry"]["sha256"] != entry["geometry"]["sha256"]:
                    raise SystemExit(f"geometry mismatch between partials at level {label}")
                slot["variants"].update(entry["variants"])
            base = {k: v for k, v in part.items() if k != "levels"}
            base.setdefault("levels", {})
            merged_base = base
        payload = {
            **merged_base,
            "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "levels": merged,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {args.out} ({len(merged)} levels)")
        return 0

    if args.quiet:
        logging.disable(logging.WARNING)
    levels = {k: LEVELS[k] for k in (args.level or sorted(LEVELS))}
    variants = tuple(args.variant or VARIANTS)
    payload = build_reference(levels=levels, variants=variants)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())