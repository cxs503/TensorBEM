"""Recompute the reference blocks, geometry gate and RAO of stored study shards.

Why this exists: the heavy part of the study (the BEM solve) is expensive while
the heave/pitch RAO is a 2x2 linear solve on the *stored* hydrodynamic
coefficients.  When a reference-frame or restoring-term convention is corrected,
the RAO must be re-derived, not re-solved.  This script does exactly that and
keeps the previous numbers in ``rao_previous_frame`` so the change is auditable.

Corrections applied (see ``docs/diffraction_drift_maturity.md`` §5):

1. rigid-body pitch block is taken about the **waterline origin**
   (``I_yy = rho T B L^3/45``), the frame of the BEM operators and of ``C55``;
2. pitch restoring is ``rho g nabla GM_L`` -- ``wigley_hydrostatic_reference``
   labels its ``C44``/``C55`` keys swapped with respect to the roll/pitch
   convention of ``compute_hydrostatic_restoring_matrix``;
3. the geometry gate is re-evaluated on the stored mesh resolution with the
   documented ``O(h^2)`` panelisation tolerance.

Usage::

    PYTHONPATH=src python3 tools/recompute_rao_m14571.py runs/m14571/shard_w*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tensorbem.diffraction_drift_maturity import (
    geometry_gate,
    rao_2dof,
    wigley_mesh,
    wigley_reference,
)


def patch(path: Path) -> dict:
    study = json.loads(path.read_text(encoding="utf-8"))
    ref = wigley_reference()
    settings = study.get("settings", {}).get("wigley_mesh") or {}
    rows = study["wigley"]["rows"]
    if rows:
        mesh = wigley_mesh(
            int(settings.get("n_sections", 32)), int(settings.get("n_levels", 16))
        )
        assert mesh.panel_count == rows[0]["panel_count"], (
            f"{path}: mesh resolution {settings} does not reproduce the stored "
            f"panel count {rows[0]['panel_count']}"
        )
        study["wigley"]["geometry"] = geometry_gate(mesh, ref)
    study["wigley"]["reference"] = _reference_payload(ref)
    for row in rows:
        near = row["excitation_routes"]["near_field"]
        excitation = [complex(near[0][0], near[0][1]), complex(near[1][0], near[1][1])]
        rao = rao_2dof(
            row["omega"], row["added_mass"], row["damping"],
            ref["mass_block"], ref["restoring_block"], excitation,
        )
        rao.pop("dynamic_matrix", None)
        xi = rao["displacement"]
        rao["displacement"] = [float(xi[0].real), float(xi[0].imag),
                              float(xi[1].real), float(xi[1].imag)]
        rao["rao_previous_frame"] = {
            k: row["rao"][k] for k in ("heave_rao", "pitch_rao_per_ka")
            if k in row["rao"]
        }
        row["rao"] = rao
    path.write_text(json.dumps(study, indent=2), encoding="utf-8")
    return {"path": str(path), "rows": len(rows),
            "C55": float(ref["restoring_block"][1, 1]),
            "I55": float(ref["mass_block"][1, 1])}


def _reference_payload(ref: dict) -> dict:
    hydro = ref["hydrostatic_reference"]
    return {
        "length": ref["length"], "beam": ref["beam"], "draft": ref["draft"],
        "density": ref["density"], "gravity": ref["gravity"],
        "volume": ref["volume"], "Cb": ref["Cb"],
        "waterplane_area": ref["waterplane_area"], "mass": ref["mass"],
        "z_g": ref["z_g"],
        "pitch_inertia_about_origin": ref["pitch_inertia_about_origin"],
        "pitch_inertia_about_cg": ref["pitch_inertia_about_cg"],
        "pitch_radius_of_gyration": ref["pitch_radius_of_gyration"],
        "pitch_radius_of_gyration_cg": ref["pitch_radius_of_gyration_cg"],
        "mass_block": ref["mass_block"].tolist(),
        "restoring_block": ref["restoring_block"].tolist(),
        "hydrostatic_reference": {
            k: (float(v) if isinstance(v, (int, float)) else
                (v.tolist() if hasattr(v, "tolist") else v))
            for k, v in hydro.items()
        },
        "modes": list(ref["modes"]),
    }


def main(argv: list[str] | None = None) -> int:
    paths = [Path(p) for p in (argv if argv is not None else sys.argv[1:])]
    if not paths:
        print("usage: recompute_rao_m14571.py <shard.json> [...]")
        return 2
    for p in paths:
        print(json.dumps(patch(p)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())