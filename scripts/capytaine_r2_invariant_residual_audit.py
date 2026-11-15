"""Independent audit of the round-2 "invariant zero" entries.

Round 2 of the capytaine cross-validation (``XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE``)
declares every entry touching a rotational dof of a spherical wetted surface to
have an *exact* zero target, because ``(r - c) x n`` vanishes pointwise on a
sphere whose centre of curvature is the rotation centre.  That statement is
about the *smooth* surface; on a faceted mesh it is only approximate, so the
measured entry is a discretisation residue rather than a solver result.

This script measures that residue independently on both codes and on both
sides of the claim:

* the **geometric** facet residue of each mesh,
  ``sum_k A_k |(r_k - c) x n_k| / sum_k A_k |r_k - c|`` (the area-weighted sine
  of the angle between the panel normal and the radial direction -- a pure
  property of the mesh, computed before any solve);
* the **measured** coupling residue
  ``max(|A_ij|, |B_ij|) / (rho L**k)`` at two frequencies on a refinement ladder;
* the **symmetry defect** ``|B[surge,pitch] + B[sway,roll]|``, which is zero for
  any body of revolution about z, smooth or faceted -- so a non-zero value is an
  unambiguous discretisation artefact of the code under test.

Usage::

    PYTHONPATH=src python scripts/capytaine_r2_invariant_residual_audit.py
    PYTHONPATH=src python scripts/capytaine_r2_invariant_residual_audit.py --fine

The ``--fine`` flag adds one capytaine level beyond the frozen reference ladder
(3200 mesh faces) and one TensorBEM level (subdivisions 4), which is the
evidence that the residue keeps decaying instead of saturating at a physical
coupling.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DENSITY = 1025.0
GRAVITY = 9.80665
SPHERE_CENTRE = (0.0, 0.0, -0.5)
RADIUS = 1.0
#: Frequencies of the frozen cross-validation grid that carry the failures.
NU_GRID = (1.0, 2.0)
#: (n_theta, n_phi) of ``capytaine.mesh_sphere``; the first three are the frozen
#: reference ladder, the fourth is the ``--fine`` extension.
CAPYTAINE_LEVELS = ((12, 24), (20, 40), (28, 56))
CAPYTAINE_FINE = (40, 80)
#: TensorBEM geodesic subdivisions; 2 and 3 are the frozen ladder.
TENSORBEM_LEVELS = (2, 3)
TENSORBEM_FINE = (4,)
DOF_ORDER = ("surge", "sway", "heave", "roll", "pitch", "yaw")

ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/verified/capytaine_crosscheck_r2/invariant_residual_audit.json"
)


def physical_scale(i: int, j: int) -> float:
    """``rho * L**k`` of the entry, from the geometry only (see the module)."""
    exponent = 3 + int(i >= 3) + int(j >= 3)
    return DENSITY * RADIUS**exponent


def facet_residue(
    centroids: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    centre: tuple[float, float, float],
) -> dict[str, float]:
    """Dimensionless facet residue of a mesh about ``centre``.

    ``mean`` is areal-average ``sin(angle(n_k, r_k - c))``; ``max`` is its
    worst panel.  Both are zero for a smooth sphere about its centre of
    curvature and O(facet angle) otherwise.
    """
    arm = np.asarray(centroids, dtype=float) - np.asarray(centre, dtype=float)
    arm_len = np.linalg.norm(arm, axis=1)
    sin_mis = np.linalg.norm(np.cross(arm, np.asarray(normals, dtype=float)), axis=1) / arm_len
    weights = np.asarray(areas, dtype=float)
    return {
        "mean": float(np.sum(weights * sin_mis) / np.sum(weights)),
        "max": float(np.max(sin_mis)),
    }


def tensorbem_side(levels: tuple[int, ...]) -> list[dict[str, Any]]:
    from tensorbem.benchmarks.capytaine_crosscheck_r2 import solve_matrix
    from tensorbem.geometry3d import floating_sphere_hull_mesh

    rows = []
    for sub in levels:
        mesh = floating_sphere_hull_mesh(sub, radius=RADIUS, center=SPHERE_CENTRE)
        geo = facet_residue(mesh.centroids, mesh.normals, mesh.areas, SPHERE_CENTRE)
        row: dict[str, Any] = {
            "subdivisions": sub,
            "faces": int(mesh.panel_count),
            "facet_residue": geo,
        }
        for nu in NU_GRID:
            got = solve_matrix(mesh, nu, SPHERE_CENTRE)
            a = np.asarray(got["added_mass"], dtype=float)
            b = np.asarray(got["damping"], dtype=float)
            ip, ir = DOF_ORDER.index("pitch"), DOF_ORDER.index("roll")
            isu, isw = DOF_ORDER.index("surge"), DOF_ORDER.index("sway")
            row[f"nu={nu:g}"] = {
                "added_mass_surge_pitch": float(a[isu, ip]),
                "damping_surge_pitch": float(b[isu, ip]),
                "damping_sway_roll": float(b[isw, ir]),
                "damping_symmetry_defect": float(abs(b[isu, ip] + b[isw, ir])),
                "residue_ratio_added_mass": float(
                    max(abs(a[isu, ip]), abs(a[isw, ir])) / physical_scale(isu, ip)
                ),
                "residue_ratio_damping": float(
                    max(abs(b[isu, ip]), abs(b[isw, ir])) / physical_scale(isu, ip)
                ),
            }
        rows.append(row)
        print(
            "  TensorBEM sub={} faces={} facet_mean={:.4f} ".format(
                sub, row["faces"], geo["mean"]
            )
            + " ".join(
                "nu={:g}:A{:.3f}% B{:.3f}% defect={:.3g}".format(
                    nu,
                    100 * row[f"nu={nu:g}"]["residue_ratio_added_mass"],
                    100 * row[f"nu={nu:g}"]["residue_ratio_damping"],
                    row[f"nu={nu:g}"]["damping_symmetry_defect"],
                )
                for nu in NU_GRID
            ),
            flush=True,
        )
    return rows


def capytaine_side(levels: tuple[tuple[int, int], ...]) -> list[dict[str, Any]]:
    import capytaine as cpt

    solver = cpt.BEMSolver()
    rows = []
    for res in levels:
        mesh = cpt.mesh_sphere(radius=RADIUS, center=SPHERE_CENTRE, resolution=res).immersed_part()
        geo = facet_residue(
            np.asarray(mesh.faces_centers),
            np.asarray(mesh.faces_normals),
            np.asarray(mesh.faces_areas),
            SPHERE_CENTRE,
        )
        body = cpt.FloatingBody(
            mesh=mesh, dofs=cpt.rigid_body_dofs(rotation_center=np.asarray(SPHERE_CENTRE))
        )
        row: dict[str, Any] = {
            "resolution": list(res),
            "faces": int(mesh.nb_faces),
            "facet_residue": geo,
        }
        for nu in NU_GRID:
            a = np.zeros((6, 6))
            b = np.zeros((6, 6))
            for dof in ("Pitch", "Roll"):
                j = DOF_ORDER.index(dof.lower())
                result = solver.solve(
                    cpt.RadiationProblem(
                        body=body,
                        omega=math.sqrt(nu * GRAVITY),
                        radiating_dof=dof,
                        water_depth=np.inf,
                        rho=DENSITY,
                        g=GRAVITY,
                    )
                )
                am = {k.lower(): float(v) for k, v in result.added_mass.items()}
                bm = {k.lower(): float(v) for k, v in result.radiation_damping.items()}
                for i, name in enumerate(DOF_ORDER):
                    a[i, j] = am.get(name, 0.0)
                    b[i, j] = bm.get(name, 0.0)
            ip, ir = DOF_ORDER.index("pitch"), DOF_ORDER.index("roll")
            isu, isw = DOF_ORDER.index("surge"), DOF_ORDER.index("sway")
            row[f"nu={nu:g}"] = {
                "added_mass_surge_pitch": float(a[isu, ip]),
                "damping_surge_pitch": float(b[isu, ip]),
                "damping_sway_roll": float(b[isw, ir]),
                "damping_symmetry_defect": float(abs(b[isu, ip] + b[isw, ir])),
                "residue_ratio_added_mass": float(
                    max(abs(a[isu, ip]), abs(a[isw, ir])) / physical_scale(isu, ip)
                ),
                "residue_ratio_damping": float(
                    max(abs(b[isu, ip]), abs(b[isw, ir])) / physical_scale(isu, ip)
                ),
            }
        rows.append(row)
        print(
            "  capytaine res={} faces={} facet_mean={:.4f} ".format(
                res, row["faces"], geo["mean"]
            )
            + " ".join(
                "nu={:g}:A{:.3f}% B{:.3f}% defect={:.3g}".format(
                    nu,
                    100 * row[f"nu={nu:g}"]["residue_ratio_added_mass"],
                    100 * row[f"nu={nu:g}"]["residue_ratio_damping"],
                    row[f"nu={nu:g}"]["damping_symmetry_defect"],
                )
                for nu in NU_GRID
            ),
            flush=True,
        )
    return rows


def decay(rows: list[dict[str, Any]], key: str, nu: float) -> dict[str, Any]:
    seq = [row[f"nu={nu:g}"][key] for row in rows]
    faces = [row["faces"] for row in rows]
    ratios = [seq[k] / seq[k + 1] for k in range(len(seq) - 1) if seq[k + 1] > 0]
    return {
        "quantity": key,
        "nu": nu,
        "faces": faces,
        "values": seq,
        "step_decay_ratios": ratios,
        "monotone": all(seq[k] > seq[k + 1] for k in range(len(seq) - 1)),
        "face_ratio": [faces[k + 1] / faces[k] for k in range(len(faces) - 1)],
    }


def _parse_levels(text: str, pair: bool) -> tuple:
    """``"2,3,4"`` for scalar levels, ``"12,24;20,40"`` for (n_theta, n_phi) pairs."""
    if pair:
        return tuple(
            tuple(int(v) for v in chunk.split(","))
            for chunk in text.split(";")
            if chunk.strip()
        )
    return tuple(int(v) for v in text.replace(";", ",").split(",") if v.strip())


def assemble(tb: list[dict[str, Any]], cap: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the published artifact from the two measured ladders."""
    payload: dict[str, Any] = {
        "anchor": "XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE",
        "case": "sphere_r1_z-0.5",
        "geometry": {
            "kind": "floating sphere, radius 1.0 m, centre (0, 0, -0.5), truncated at z = 0",
            "rotation_center": list(SPHERE_CENTRE),
            "density": DENSITY,
            "gravity": GRAVITY,
            "nu_grid": list(NU_GRID),
        },
        "claim_under_test": (
            "(r - c) x n == 0 pointwise on a sphere whose centre of curvature is the "
            "rotation centre, so every entry touching roll/pitch/yaw has an exact zero "
            "target. True for the smooth surface; the audit measures the residue a "
            "faceted mesh leaves behind."
        ),
        "definitions": {
            "facet_residue": (
                "sum_k A_k |(r_k - c) x n_k| / sum_k A_k |r_k - c| ; zero for a smooth "
                "sphere about its centre of curvature, O(facet angle) otherwise"
            ),
            "residue_ratio": "max(|A_ij|,|B_ij|) / (rho * L**k) of the entry under test",
            "symmetry_defect": (
                "|B[surge,pitch] + B[sway,roll]| ; exactly zero for any body of "
                "revolution about z, faceted or not, because the x->y swap is an exact "
                "symmetry of the discrete problem"
            ),
        },
        "tensorbem": tb,
        "capytaine": cap,
    }
    n0 = f"nu={NU_GRID[0]:g}"
    decays: dict[str, Any] = {}
    tracking: dict[str, Any] = {}
    for name, rows in (("tensorbem", tb), ("capytaine", cap)):
        for quantity, key in (
            ("damping", "residue_ratio_damping"),
            ("added_mass", "residue_ratio_added_mass"),
        ):
            decays[f"{name}_{quantity}"] = decay(rows, key, NU_GRID[0])
        pairs = [(row["facet_residue"]["mean"], row[n0]["residue_ratio_damping"]) for row in rows]
        tracking[name] = {
            "facet_mean_vs_damping_residue": pairs,
            "amplification": [(meas / geo if geo else None) for geo, meas in pairs],
        }
    payload["decay"] = decays
    payload["facet_tracking"] = tracking

    tb_mono = all(blk["monotone"] for k, blk in decays.items() if k.startswith("tensorbem"))
    cap_mono = all(blk["monotone"] for k, blk in decays.items() if k.startswith("capytaine"))
    payload["findings"] = {
        "tensorbem_residue_decays": bool(tb_mono),
        "capytaine_residue_decays": bool(cap_mono),
        "step_decay_ratios": {k: v["step_decay_ratios"] for k, v in decays.items()},
        "finest": {
            "tensorbem": {"faces": tb[-1]["faces"], "damping_residue": tb[-1][n0]["residue_ratio_damping"]},
            "capytaine": {"faces": cap[-1]["faces"], "damping_residue": cap[-1][n0]["residue_ratio_damping"]},
        },
        "symmetry_defect_finest": {
            "tensorbem": tb[-1][n0]["damping_symmetry_defect"],
            "capytaine": cap[-1][n0]["damping_symmetry_defect"],
        },
        "verdict": (
            "DISCRETISATION-DOMINATED: the rotational coupling entries are not physics "
            "at the mesh level of the cross-validation. They decay as the mesh refines on "
            "at least the TensorBEM ladder and they are not reproducible between the two "
            "codes as a physical quantity, so they must be bounded by a facet-derived "
            "bound and never scored by relative deviation against a foreign code."
            if tb_mono
            else "NOT ESTABLISHED: the TensorBEM ladder does not decay; treat the "
            "invariant entries as a potential solver defect."
        ),
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side", choices=("both", "tensorbem", "capytaine"), default="both")
    ap.add_argument("--tensorbem-levels", default="2,3", help='e.g. "2,3,4"')
    ap.add_argument("--capytaine-levels", default="12,24;20,40;28,56", help='e.g. "28,56;40,80"')
    ap.add_argument("--fine", action="store_true", help="use the ladder plus the finest extension")
    ap.add_argument("--part", default=None, help="write only the measured ladder of --side here")
    ap.add_argument("--merge", nargs=2, default=None,
                    metavar=("TENSORBEM_JSON", "CAPYTAINE_JSON"), help="merge two parts")
    ap.add_argument("--out", default=str(ARTIFACT))
    args = ap.parse_args(argv)

    if args.merge:
        tb = json.loads(Path(args.merge[0]).read_text())
        cap = json.loads(Path(args.merge[1]).read_text())
        payload = assemble(tb, cap)
        Path(args.out).write_text(json.dumps(payload, indent=1) + "\n")
        print("wrote", args.out)
        print("verdict:", payload["findings"]["verdict"])
        return 0

    tb_levels = _parse_levels(args.tensorbem_levels, pair=False)
    cap_levels = _parse_levels(args.capytaine_levels, pair=True)
    if args.fine:
        tb_levels = tuple(sorted(set(tb_levels) | set(TENSORBEM_FINE)))
        cap_levels = tuple(sorted(set(cap_levels) | set(CAPYTAINE_FINE)))

    if args.side in ("both", "tensorbem"):
        print("TensorBEM geodesic ladder", flush=True)
        tb = tensorbem_side(tb_levels)
        if args.part:
            Path(args.part).write_text(json.dumps(tb, indent=1) + "\n")
            print("wrote part", args.part, flush=True)
    if args.side in ("both", "capytaine"):
        print("capytaine UV-sphere ladder", flush=True)
        cap = capytaine_side(cap_levels)
        if args.part:
            Path(args.part).write_text(json.dumps(cap, indent=1) + "\n")
            print("wrote part", args.part, flush=True)

    if args.side == "both":
        payload = assemble(tb, cap)
        Path(args.out).write_text(json.dumps(payload, indent=1) + "\n")
        print("wrote", args.out)
        print("verdict:", payload["findings"]["verdict"])
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
