#!/usr/bin/env python3
"""Independent (capytaine) reference for the *rotational null* of a hemisphere.

Case-pool anchor ``CASE-BEM-HULME-HEMISPHERE`` grades only surge and heave,
because the rotational radiating modes of a floating hemisphere whose rotation
centre coincides with the sphere centre are *null modes*: every point of the
wetted hull satisfies ``n . (Omega x r) = 0`` identically, so

    A_44 = A_55 = A_66 = B_44 = B_55 = B_66 = 0   (exactly, a = R, centre = O)

and all rotational cross terms vanish as well.  Those zeros are analytic and do
not involve the free-surface Green function at all, which makes them a
kernel-free verification axis for the panel geometry, the panel normals and the
rigid-body moment arms -- the three things that a "<5 % agreement" test can
never see.

A discretised hull is a polyhedron, so its facet normals are not radial and the
residual is small but non-zero; it is therefore a *pure discretisation error*
that must fall to zero under refinement.  This script produces the foreign-code
half of that statement: it drives capytaine (third-party BEM, the only place in
the repository that imports it for this quantity) over a mesh ladder and freezes
the rotational added masses / dampings plus full provenance into

    benchmarks/verified/casepool_hulme_rotational_null/capytaine_rotational_null.json

Two measured findings are baked into the artefact:

1. capytaine's own rotational residual is non-zero on its polyhedral hull and
   shrinks as the hull is refined -- i.e. the null is a property of the exact
   geometry, not of one code.
2. capytaine's rotation degrees of freedom are taken about the origin
   ``(0, 0, 0)``, which is the sphere centre of the wetted hemisphere.  That is
   probed, not assumed: an explicit ``center_of_rotation`` offset below the
   origin must turn the null into the exact translation-superposition value
   ``d^2 * A_22``.

Usage::

    python scripts/capytaine_hemisphere_rotational_null.py \
        --out benchmarks/verified/casepool_hulme_rotational_null/capytaine_rotational_null.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

DENSITY = 1025.0
GRAVITY = 9.80665
RADIUS = 1.0

#: ka band of the case being extended (identical to the TensorBEM side).
KA_GRID: tuple[float, ...] = (0.2, 0.5, 0.8, 1.0, 1.5, 2.0)

#: capytaine lat-lon resolutions; the panel count of each level is read back
#: from the body at run time instead of being hard-coded.
MESH_LADDER: tuple[tuple[int, int], ...] = (
    (8, 16),
    (16, 32),
    (24, 48),
    (32, 64),
)

ROTATIONAL_DOFS: tuple[str, ...] = ("Roll", "Pitch", "Yaw")

#: Offset used by the rotation-centre probe: rolling about an axis ``d`` below
#: the origin must reproduce ``d^2 * A_22`` (the sway added mass times d^2) in
#: the limit of a vanishing rotational null, i.e. the probe turns a null mode
#: into a large, exactly predictable number.
OFFSET_D = 0.5


def _provenance(cpt: Any) -> dict[str, Any]:
    try:
        import capytaine  # noqa: F401

        version = getattr(capytaine, "__version__", "unknown")
    except Exception:  # pragma: no cover - provenance only
        version = "unknown"
    return {
        "tool": "capytaine",
        "version": version,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "density": DENSITY,
        "gravity": GRAVITY,
        "water_depth": "infinite",
        "rotation_centre": "origin (0, 0, 0) = centre of the sphere carrying the hull",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "note": (
            "optional dependency; only this producer script imports capytaine, "
            "the study itself reads the frozen json"
        ),
    }


def _body(cpt: Any, resolution: tuple[int, int]) -> Any:
    """Wetted hemisphere ``r = 1``, ``z <= 0``, open at the waterline."""
    mesh = cpt.mesh_sphere(
        radius=RADIUS, center=(0.0, 0.0, 0.0), resolution=resolution
    ).immersed_part()
    body = cpt.FloatingBody(mesh=mesh)
    # the sphere centre is the rotation centre: stated explicitly so the
    # convention is in the artefact and not in capytaine's default
    body.add_all_rigid_body_dofs(rotation_center=np.array([0.0, 0.0, 0.0]))
    return body


def _radiation(
    cpt: Any, solver: Any, body: Any, ka: float, dof: str
) -> dict[str, float]:
    res = solver.solve(
        cpt.RadiationProblem(
            body=body,
            omega=math.sqrt(ka * GRAVITY),
            radiating_dof=dof,
            water_depth=np.inf,
            rho=DENSITY,
            g=GRAVITY,
        )
    )
    return {
        "added_mass": float(res.added_mass[dof]),
        "damping": float(res.radiation_damping[dof]),
    }


def _dimensionless(value: float, ka: float, exponent: float) -> float:
    """``A / (rho a^exponent)`` and ``B / (rho omega a^exponent)`` with a = R."""
    a = RADIUS**exponent
    if exponent == 5.0:  # rotational self term: A / (rho a^5)
        return value / (DENSITY * a)
    omega = math.sqrt(ka * GRAVITY)
    return value / (DENSITY * omega * a)


def _offset_probe(cpt: Any, solver: Any, ka: float) -> dict[str, Any]:
    """Rotation centre moved ``OFFSET_D`` below the origin: exact superposition.

    Rolling about an axis ``d`` below the origin is, as a velocity field, the
    sum of rolling about the origin and a uniform sway ``-d``; linearity of the
    radiation problem therefore gives the exact relation

        A_44^c = A_44 - 2 d A_42 + d^2 A_22

    (and the same with ``B``).  Because the origin-roll term is a null mode,
    the probe is dominated by ``d^2 A_22``: the null becomes a large,
    predictable number, which is what makes the rotation-centre convention
    measurable rather than assumed.
    """
    mesh = cpt.mesh_sphere(
        radius=RADIUS, center=(0.0, 0.0, 0.0), resolution=MESH_LADDER[-1]
    ).immersed_part()
    body = cpt.FloatingBody(mesh=mesh)
    body.add_all_rigid_body_dofs(rotation_center=np.array([0.0, 0.0, 0.0]))
    body.add_rotation_dof(
        rotation_center=np.array([0.0, 0.0, -OFFSET_D]),
        direction=np.array([1.0, 0.0, 0.0]),
        name="RollOffset",
    )
    omega = math.sqrt(ka * GRAVITY)

    def radiation(dof: str) -> Any:
        return solver.solve(
            cpt.RadiationProblem(
                body=body,
                omega=omega,
                radiating_dof=dof,
                water_depth=np.inf,
                rho=DENSITY,
                g=GRAVITY,
            )
        )

    sway = radiation("Sway")
    roll_origin = radiation("Roll")
    roll_offset = radiation("RollOffset")

    a22 = float(sway.added_mass["Sway"])
    a44 = float(roll_origin.added_mass["Roll"])
    a42 = float(sway.added_mass["Roll"])
    a24 = float(roll_origin.added_mass["Sway"])
    a44c = float(roll_offset.added_mass["RollOffset"])
    d = OFFSET_D
    predicted = a44 - 2.0 * d * a42 + d * d * a22
    return {
        "n_faces": int(body.mesh.nb_faces),
        "offset_d": d,
        "A22_sway": a22,
        "A44_roll_about_origin": a44,
        "A42_roll_moment_from_sway": a42,
        "A24_sway_force_from_roll": a24,
        "A44c_roll_about_offset_axis": a44c,
        "predicted_d2_A22_only": d * d * a22,
        "predicted_full_superposition": predicted,
        "relative_dev_full_superposition": abs(a44c - predicted) / abs(a44c),
        "relative_dev_d2_A22_only": abs(a44c - d * d * a22) / abs(a44c),
    }


def build(cpt: Any) -> dict[str, Any]:
    cpt.set_logging("ERROR")
    solver = cpt.BEMSolver()

    ladder: list[dict[str, Any]] = []
    for resolution in MESH_LADDER:
        body = _body(cpt, resolution)
        values: dict[str, Any] = {}
        for ka in KA_GRID:
            values[f"{ka:.3f}"] = {
                dof: _radiation(cpt, solver, body, ka, dof) for dof in ROTATIONAL_DOFS
            }
        ladder.append(
            {
                "resolution": list(resolution),
                "n_faces": int(body.mesh.nb_faces),
                "volume": float(body.volume),
                "values": values,
            }
        )

    probes = {f"{ka:.3f}": _offset_probe(cpt, solver, ka) for ka in KA_GRID}

    return {
        "anchor": "CASE-BEM-HULME-HEMISPHERE",
        "quantity": (
            "rotational radiation added mass / damping of the floating hemisphere "
            "r = 1 m, waterplane z = 0, rotation axis through the sphere centre"
        ),
        "analytic_reference": {
            "added_mass": 0.0,
            "damping": 0.0,
            "reason": (
                "the wetted hull is a spherical surface concentric with the "
                "rotation axis, so n . (Omega x r) = 0 on the whole hull and the "
                "radiation potential vanishes identically"
            ),
        },
        "provenance": _provenance(cpt),
        "ka_grid": list(KA_GRID),
        "rotational_dofs": list(ROTATIONAL_DOFS),
        "mesh_ladder": ladder,
        "rotation_centre_probe": probes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "benchmarks/verified/casepool_hulme_rotational_null/"
            "capytaine_rotational_null.json"
        ),
    )
    args = parser.parse_args(argv)

    import capytaine as cpt  # imported here so the rest of the module is testable

    data = build(cpt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())