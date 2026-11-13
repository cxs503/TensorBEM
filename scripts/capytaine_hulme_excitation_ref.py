#!/usr/bin/env python3
"""Independent capytaine reference for the case-pool anchor BEM-HULME-HEMISPHERE.

Why this file exists
--------------------
The case's reference solution (``tensorbem.analytic.hulme_hemisphere_radiation``,
Hulme 1982's wave-free multipole series) and the panel solver share
``tensorbem.free_surface3d.wave_function_3d``: the *same* code produces the
free-surface Green function for the BEM assembly and the incident-wave / Kochin
machinery for the reference.  Agreement between the two therefore cannot detect
a defect in that shared kernel, and Hulme's printed tables are behind a paywall,
so no digit of them is transcribed anywhere in the repository.  A third code is
the only way to adjudicate.

This producer freezes such a third code for **all six case quantities**, in
particular the two excitation-force components, which the pre-existing
capytaine cross-check (``benchmarks/verified/capytaine_crosscheck``, radiation
only at nu = 0.5/1/2) never touched.

TRAP (cost one full debugging round, recorded for the next reader)
------------------------------------------------------------------
In capytaine >= 3.0 ``DiffractionResult.forces`` holds the **diffraction force
only**; the Froude-Krylov part lives in ``capytaine.bem.airy_waves.
froude_krylov_force(problem)`` and must be added by hand.  Comparing
``result.forces`` against a total excitation force silently loses the FK term:
on the heave force of this body that is ~60 % of the answer, and on the surge
force it turns the result into a 3.7x under-estimate.  :func:`_excitation`
below does the sum and the artifact stores both parts separately so the mistake
cannot be repeated invisibly.

Only this producer imports capytaine (same policy as
``scripts/capytaine_crosscheck_ref.py``): the numbers are frozen into
``benchmarks/verified/casepool_bem_hulme_hemisphere/capytaine_excitation.json``
and the test suite reads the artifact, so the library keeps zero dependency on
a third-party BEM code.

Usage
-----
    python scripts/capytaine_hulme_excitation_ref.py \
        --out benchmarks/verified/casepool_bem_hulme_hemisphere/capytaine_excitation.json
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

RADIUS = 1.0
DENSITY = 1025.0
GRAVITY = 9.80665
AMPLITUDE = 1.0
#: case band, identical to the anchor's ``case.band_ka`` (a = 1 m so ka == nu)
BAND_KA = (0.2, 0.5, 0.8, 1.0, 1.5, 2.0)
#: capytaine resolutions; the finest clips to 1024 panels, i.e. exactly the
#: anchor's case mesh, so the code-to-code comparison is apples-to-apples
LEVELS = ((16, 32), (24, 48), (32, 64), (40, 80))
DOFS = ("Surge", "Heave")


def _provenance(cpt: Any) -> dict[str, Any]:
    commit = None
    root = Path(cpt.__file__).resolve().parents[1]
    if (root / ".git").exists():
        try:
            commit = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:  # pragma: no cover - provenance only
            commit = None
    return {
        "tool": "capytaine",
        "version": getattr(cpt, "__version__", "unknown"),
        "git_commit": commit,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "water_depth": "infinite",
        "density": DENSITY,
        "gravity": GRAVITY,
        "wave_amplitude": AMPLITUDE,
        "dofs": list(DOFS),
        "band_ka": list(BAND_KA),
        "excitation_convention": (
            "total wave exciting force = DiffractionResult.forces (diffraction "
            "part only in capytaine >= 3.0) + froude_krylov_force(problem); "
            "the two parts are stored separately below"
        ),
        "note": (
            "only this producer script imports capytaine; the frozen JSON is "
            "what tests and documents read"
        ),
    }


def _norm_volume() -> float:
    return (2.0 / 3.0) * math.pi * RADIUS**3


def _excitation(cpt: Any, solver: Any, problem: Any) -> dict[str, Any]:
    """Total exciting force per dof: diffraction part + Froude-Krylov part."""
    from capytaine.bem.airy_waves import froude_krylov_force

    result = solver.solve(problem)
    fk = froude_krylov_force(problem)
    return {
        "diffraction": {d: complex(result.forces[d]) for d in DOFS},
        "froude_krylov": {d: complex(fk[d]) for d in DOFS},
        "total": {d: complex(result.forces[d]) + complex(fk[d]) for d in DOFS},
    }


def _sweep(cpt: Any, solver: Any, mesh: Any) -> dict[str, Any]:
    body = cpt.FloatingBody(mesh=mesh)
    body.add_all_rigid_body_dofs()
    volume = float(mesh.volume)
    nabla = _norm_volume()
    out: dict[str, Any] = {}
    for ka in BAND_KA:
        omega = math.sqrt(GRAVITY * ka)
        rho_omega_nabla = DENSITY * omega * nabla
        row: dict[str, Any] = {
            "A11": None,
            "B11": None,
            "A33": None,
            "B33": None,
            "Fx_dimless": None,
            "Fz_dimless": None,
            "phase_Fx_deg": None,
            "phase_Fz_deg": None,
        }
        for dof in DOFS:
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
            added = float(res.added_mass[dof]) / (DENSITY * nabla)
            damping = float(res.radiation_damping[dof]) / rho_omega_nabla
            if dof == "Surge":
                row["A11"], row["B11"] = added, damping
            else:
                row["A33"], row["B33"] = added, damping
        problem = cpt.DiffractionProblem(
            body=body,
            omega=omega,
            water_depth=np.inf,
            rho=DENSITY,
            g=GRAVITY,
            wave_direction=0.0,
        )
        force = _excitation(cpt, solver, problem)
        scale = DENSITY * GRAVITY * AMPLITUDE * RADIUS**2
        for dof, key, phase_key in (
            ("Surge", "Fx_dimless", "phase_Fx_deg"),
            ("Heave", "Fz_dimless", "phase_Fz_deg"),
        ):
            value = force["total"][dof]
            row[key] = abs(value) / scale
            row[phase_key] = math.degrees(math.atan2(value.imag, value.real))
        row["ka"] = ka
        row["omega"] = omega
        row["excitation_complex"] = {
            "diffraction": {
                d: [force["diffraction"][d].real, force["diffraction"][d].imag]
                for d in DOFS
            },
            "froude_krylov": {
                d: [force["froude_krylov"][d].real, force["froude_krylov"][d].imag]
                for d in DOFS
            },
            "total": {
                d: [force["total"][d].real, force["total"][d].imag] for d in DOFS
            },
        }
        out[f"{ka:g}"] = row
    return {
        "n_faces": int(mesh.nb_faces),
        "mesh_volume": volume,
        "volume_rel_error": volume / nabla - 1.0,
        "waterplane_radius": math.sqrt(RADIUS**2 - 0.0**2),
        "values": out,
    }


def _convergence(levels: list[dict[str, Any]]) -> dict[str, Any]:
    """Worst relative drift per quantity between successive mesh levels."""
    keys = ("A11", "B11", "A33", "B33", "Fx_dimless", "Fz_dimless")
    out: dict[str, dict[str, float]] = {}
    for coarse, fine in zip(levels, levels[1:]):
        drop: dict[str, float] = {}
        for key in keys:
            worst = 0.0
            for ka in BAND_KA:
                a = coarse["values"][f"{ka:g}"][key]
                b = fine["values"][f"{ka:g}"][key]
                worst = max(worst, abs(b - a) / abs(b))
            drop[key] = worst
        drop["max"] = max(drop.values())
        out[f"{coarse['n_faces']}->{fine['n_faces']}"] = drop
    return out


def build(out_path: Path) -> dict[str, Any]:
    import capytaine as cpt

    solver = cpt.BEMSolver()
    levels = []
    for resolution in LEVELS:
        mesh = cpt.mesh_sphere(
            radius=RADIUS, center=(0.0, 0.0, 0.0), resolution=resolution
        ).immersed_part()
        levels.append(
            {"resolution": list(resolution), **_sweep(cpt, solver, mesh)}
        )
    payload = {
        "metadata": _provenance(cpt),
        "case_code": "BEM-HULME-HEMISPHERE",
        "geometry": (
            "surface-piercing hemisphere, radius 1.0 m, flat face on the mean "
            "free surface (capytaine mesh_sphere(...).immersed_part())"
        ),
        "nabla_exact": _norm_volume(),
        "levels": levels,
        "convergence": _convergence(levels),
        "reference_level": levels[-1]["n_faces"],
        "reference": levels[-1]["values"],
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
            "benchmarks/verified/casepool_bem_hulme_hemisphere/"
            "capytaine_excitation.json"
        ),
    )
    args = parser.parse_args(argv)
    payload = build(args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "levels": [lvl["n_faces"] for lvl in payload["levels"]],
                "convergence": payload["convergence"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())