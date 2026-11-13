"""Convergence audit of the *independent* route of CASE-BEM-HULME-DIFF-VERIF.

Why this script exists
----------------------
The anchor scores the TensorBEM Kelvin-BIE hemisphere against Hulme (1982)'s
wave-free multipole series and closes with gate ``G4``: "third-party code
(capytaine) agrees with the reference to <= 5 %".  The frozen artefact reports

    worst over the band, capytaine's **finest** mesh (1024 panels) : 4.30 %
    worst over the band, capytaine after Richardson (assumed p = 2)  : 7.48 %

i.e. the verdict is taken on the *unconverged* level and the extrapolated
level breaks the gate.  A gate that passes only because the discretisation
error of the third-party code is taken at face value is not evidence that the
two codes agree; it is evidence that they differ by less than the gate *plus*
whatever the third code has not yet resolved.  This producer closes that hole
by measuring capytaine's own convergence instead of assuming it:

``stage uv``
    extend capytaine's lat-lon ladder (32, 64) -> (40, 80) -> (48, 96) so the
    observed order and the limit are *measured*;
``stage octa``
    run capytaine on the **same mesh family** as the TensorBEM side
    (``tensorbem.geometry3d.hemisphere_hull_mesh``), which separates
    "formulation disagreement" from "the UV-sphere mesh is a worse polyhedron";
``stage frozen``
    reanalyse the anchor's *frozen* third-party ladder (64/256/576/1024) with the
    corrected Richardson and quantify what the pre-fix helper published;
``stage repro``
    re-run the frozen resolution (32, 64) and compare with
    ``capytaine_scattered.json`` -- provenance gate: the frozen third-party
    route must be reproducible to 1e-12 before any of the above means anything.

Physics conventions are the anchor's, unchanged: surface-piercing hemisphere
``a = 1`` m, flat face on ``z = 0``, infinite depth, ``rho = 1025``,
``g = 9.80665``, incident wave amplitude 1 m, time convention ``e^{-i w t}``,
and the excitation force is the **total** force.  capytaine's
``DiffractionProblem.forces`` is the scattered part only, so the exact
Froude-Krylov quadrature of ``run_case.exact_froude_krylov`` is added back.

Nothing under ``src/`` imports capytaine; this is a producer script.

Usage::

    PYTHONPATH=src python scripts/capytaine_hulme_diff_convergence_audit.py \
        --out benchmarks/verified/casepool_bem_hulme_diff_verif/capytaine_convergence_audit.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tensorbem.analytic import hulme_hemisphere_radiation
from tensorbem.geometry3d import hemisphere_hull_mesh

DENSITY = 1025.0
GRAVITY = 9.80665
CASE = Path("benchmarks/verified/casepool_bem_hulme_diff_verif")

#: Frequencies audited (a subset of the anchor's band: the audit is about the
#: third-party route's convergence, which is frequency-smooth).
KA_GRID: tuple[float, ...] = (0.5, 1.0, 2.0)

#: capytaine lat-lon ladder: the frozen case mesh (32, 64) plus two refinements.
UV_LADDER: tuple[tuple[int, int], ...] = ((32, 64), (40, 80), (48, 96))

#: TensorBEM octahedral hemisphere subdivision -> panels 4*4**s.
OCTA_LEVELS: tuple[int, ...] = (3, 4, 5)

#: The anchor's original band, required by the frozen-ladder reanalysis.
FROZEN_KA_GRID: tuple[float, ...] = (0.2, 0.5, 0.8, 1.0, 1.5, 2.0)

REFERENCE_TERMS = 144
REFERENCE_QUAD = 288

#: Gate on the third-party route, unchanged from the anchor.
GATE_TOTAL = 0.05


def _c(value: complex) -> dict[str, float]:
    return {"re": float(np.real(value)), "im": float(np.imag(value))}


def _run_case_module() -> Any:
    """Import the case driver to reuse its *verified* Froude-Krylov quadrature."""
    path = CASE / "run_case.py"
    spec = importlib.util.spec_from_file_location("hulme_audit_run_case", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _provenance(cpt: Any) -> dict[str, Any]:
    commit = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # pragma: no cover - provenance only
        commit = None
    return {
        "tool": "capytaine",
        "version": getattr(cpt, "__version__", "unknown"),
        "repo_commit": commit,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "density": DENSITY,
        "gravity": GRAVITY,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }


def _reference(ka: float) -> dict[str, complex]:
    sol = hulme_hemisphere_radiation(
        ka,
        radius=1.0,
        density=DENSITY,
        gravity=GRAVITY,
        n_terms=REFERENCE_TERMS,
        quad_order=REFERENCE_QUAD,
    )
    return {
        "surge": complex(sol.excitation_force_x),
        "heave": complex(sol.excitation_force_z),
    }


def _fk(run_case: Any, ka: float) -> dict[str, complex]:
    fx, fz = run_case.exact_froude_krylov(ka)
    return {"surge": complex(fx), "heave": complex(fz)}


def _uv_body(cpt: Any, resolution: tuple[int, int]) -> Any:
    mesh = cpt.mesh_sphere(
        radius=1.0, center=(0.0, 0.0, 0.0), resolution=resolution
    ).immersed_part()
    body = cpt.FloatingBody(mesh=mesh)
    body.add_all_rigid_body_dofs()
    return body


def _octa_body(cpt: Any, subdivisions: int) -> Any:
    hull = hemisphere_hull_mesh(subdivisions, radius=1.0)
    vertices = np.asarray(hull.vertices, dtype=float)
    faces = np.asarray(hull.faces, dtype=int)
    body = cpt.FloatingBody(mesh=cpt.Mesh(vertices, faces))
    body.add_all_rigid_body_dofs()
    return body


def _scattered(cpt: Any, solver: Any, body: Any, ka: float) -> dict[str, complex]:
    problem = cpt.DiffractionProblem(
        body=body,
        omega=math.sqrt(ka * GRAVITY),
        wave_direction=0.0,
        water_depth=np.inf,
        rho=DENSITY,
        g=GRAVITY,
    )
    res = solver.solve(problem)
    return {
        "surge": complex(res.forces["Surge"]),
        "heave": complex(res.forces["Heave"]),
    }


def _cell_error(total: complex, reference: complex) -> float:
    return abs(total - reference) / abs(reference)


def _observed_order(errors: list[float], panels: list[int]) -> float | None:
    """Order from three levels with ``e ~ C h^p``, ``h ~ N^{-1/2}``."""
    if len(errors) < 3:
        return None
    e1, e2 = errors[-3], errors[-2]
    e3 = errors[-1]
    n1, n2, n3 = panels[-3], panels[-2], panels[-1]
    r12 = math.sqrt(n2 / n1)
    r23 = math.sqrt(n3 / n2)
    if not (e1 > 0 and e2 > 0 and e3 > 0) or e1 <= e2:
        return None
    ratio1 = math.log(e1 / e2) / math.log(r12)
    ratio2 = math.log(e2 / e3) / math.log(r23)
    return float(0.5 * (ratio1 + ratio2))


def _richardson(
    values: list[complex], panels: list[int], order: float, *, inverted: bool = False
) -> complex:
    """Two-level Richardson extrapolation in ``h ~ N^{-1/2}``.

    ``inverted=True`` reproduces the **pre-fix** helper of
    ``scripts/capytaine_hulme_diff_scattered.py`` (refinement ratio taken the
    wrong way round, ``r < 1``), which is kept here on purpose: the audit has to
    quantify the defect it reported, not just describe it.  With ``r < 1`` the
    denominator ``r**order - 1`` is negative, so the correction is applied with
    the wrong sign and divided by ``1 - r**order`` instead of ``r**order - 1``.
    """
    f2, f1 = values[-1], values[-2]
    n2, n1 = panels[-1], panels[-2]
    if not n2 > n1 > 0:
        raise ValueError("panel counts must be positive and strictly increasing")
    r = math.sqrt(n1 / n2) if inverted else math.sqrt(n2 / n1)
    return complex(f2 + (f2 - f1) / (r**order - 1.0))


def _ladder_summary(
    ladder: list[dict[str, Any]], run_case: Any
) -> dict[str, Any]:
    """Per (ka, dof): level errors, observed order, Richardson limit."""
    summary: dict[str, Any] = {}
    panels = [row["n_faces"] for row in ladder]
    for ka in KA_GRID:
        ref = _reference(ka)
        fk = _fk(run_case, ka)
        for dof in ("surge", "heave"):
            totals = [
                complex(
                    row["scattered"][f"{ka:.3f}"][dof]["re"],
                    row["scattered"][f"{ka:.3f}"][dof]["im"],
                )
                + fk[dof]
                for row in ladder
            ]
            errors = [_cell_error(t, ref[dof]) for t in totals]
            order = _observed_order(errors, panels)
            entry: dict[str, Any] = {
                "reference": _c(ref[dof]),
                "froude_krylov": _c(fk[dof]),
                "levels": [
                    {"panels": n, "total": _c(t), "relative_error": e}
                    for n, t, e in zip(panels, totals, errors)
                ],
                "worst_relative_error": max(errors),
                "finest_relative_error": errors[-1],
                "monotone": all(
                    errors[i] > errors[i + 1] for i in range(len(errors) - 1)
                ),
                "observed_order": order,
            }
            if order is not None:
                limit = _richardson(totals, panels, order)
                entry["richardson_observed_order"] = _c(limit)
                entry["richardson_observed_order_relative_error"] = _cell_error(
                    limit, ref[dof]
                )
            limit2 = _richardson(totals, panels, 2.0)
            entry["richardson_p2"] = _c(limit2)
            entry["richardson_p2_relative_error"] = _cell_error(limit2, ref[dof])
            summary[f"{ka:.3f}/{dof}"] = entry
    return summary


def _worst(summary: dict[str, Any], key: str) -> float:
    return max(entry[key] for entry in summary.values())


def build(cpt: Any, stages: set[str]) -> dict[str, Any]:
    cpt.set_logging("ERROR")
    solver = cpt.BEMSolver()
    run_case = _run_case_module()
    out: dict[str, Any] = {
        "anchor": "CASE-BEM-HULME-DIFF-VERIF",
        "purpose": "convergence audit of the third-party (capytaine) route",
        "provenance": _provenance(cpt),
        "ka_grid": list(KA_GRID),
        "gate_total": GATE_TOTAL,
        "reference_terms": REFERENCE_TERMS,
    }

    if "repro" in stages:
        frozen = json.loads((CASE / "capytaine_scattered.json").read_text())
        frozen_finest = frozen["mesh_ladder"][-1]
        body = _uv_body(cpt, (32, 64))
        got = _scattered(cpt, solver, body, KA_GRID[1])
        worst = 0.0
        for dof in ("surge", "heave"):
            stored = frozen_finest["values"][f"{KA_GRID[1]:.3f}"][dof]
            want = complex(stored["re"], stored["im"]) * (DENSITY * GRAVITY)
            worst = max(worst, abs(got[dof] - want) / abs(want))
        out["frozen_reproduction"] = {
            "resolution": [32, 64],
            "panels": int(body.mesh.nb_faces),
            "frozen_ladder_panels": frozen_finest["n_faces"],
            "ka": KA_GRID[1],
            "worst_relative_deviation": worst,
            "verdict": "REPRODUCED" if worst <= 1.0e-9 else "DEVIATES",
        }

    if "frozen" in stages:
        frozen = json.loads((CASE / "capytaine_scattered.json").read_text())
        rows = frozen["mesh_ladder"]
        panels = [row["n_faces"] for row in rows]
        cells: dict[str, Any] = {}
        for ka in FROZEN_KA_GRID:
            ref = _reference(ka)
            fk = _fk(run_case, ka)
            for dof in ("surge", "heave"):
                totals = [
                    complex(
                        row["values"][f"{ka:.3f}"][dof]["re"] * DENSITY * GRAVITY,
                        row["values"][f"{ka:.3f}"][dof]["im"] * DENSITY * GRAVITY,
                    )
                    + fk[dof]
                    for row in rows
                ]
                errors = [_cell_error(t, ref[dof]) for t in totals]
                order = _observed_order(errors, panels)
                limit = _richardson(totals, panels, 2.0)
                bad = _richardson(totals, panels, 2.0, inverted=True)
                stored = frozen["extrapolation"][dof]
                cells[f"{ka:.3f}/{dof}"] = {
                    "levels_relative_error": errors,
                    "observed_order": order,
                    "richardson_p2_relative_error": _cell_error(limit, ref[dof]),
                    "prefix_inverted_ratio_relative_error": _cell_error(bad, ref[dof]),
                    "finest_relative_error": errors[-1],
                    # the frozen file extrapolates the *scattered* force; the audit
                    # extrapolates the *total* (scattered + exact Froude-Krylov), so
                    # the stored limit has to be completed with the same FK before
                    # the two can be compared.
                    "stored_extrapolation_matches_fix": abs(
                        complex(
                            stored["richardson"]["order_2"][f"{ka:.3f}"]["re"] * DENSITY * GRAVITY,
                            stored["richardson"]["order_2"][f"{ka:.3f}"]["im"] * DENSITY * GRAVITY,
                        )
                        + fk[dof]
                        - limit
                    )
                    / abs(limit)
                    < 1.0e-9,
                }
        out["frozen_ladder_reanalysis"] = {
            "mesh_ladder_panels": panels,
            "cells": cells,
            "worst_finest_relative": max(
                c["finest_relative_error"] for c in cells.values()
            ),
            "worst_richardson_p2_relative": max(
                c["richardson_p2_relative_error"] for c in cells.values()
            ),
            "worst_prefix_inverted_ratio_relative": max(
                c["prefix_inverted_ratio_relative_error"] for c in cells.values()
            ),
            "all_stored_extrapolations_match_fix": all(
                c["stored_extrapolation_matches_fix"] for c in cells.values()
            ),
            "verdict_richardson_p2": (
                "PASS"
                if max(c["richardson_p2_relative_error"] for c in cells.values())
                <= GATE_TOTAL
                else "FAIL"
            ),
        }

    if "uv" in stages:
        ladder: list[dict[str, Any]] = []
        for resolution in UV_LADDER:
            body = _uv_body(cpt, resolution)
            ladder.append(
                {
                    "family": "capytaine lat-lon sphere (immersed part)",
                    "resolution": list(resolution),
                    "n_faces": int(body.mesh.nb_faces),
                    "volume": float(body.volume),
                    "scattered": {
                        f"{ka:.3f}": {k: _c(v) for k, v in _scattered(cpt, solver, body, ka).items()}
                        for ka in KA_GRID
                    },
                }
            )
        summary = _ladder_summary(ladder, run_case)
        out["uv_family"] = {
            "ladder": ladder,
            "cells": summary,
            "worst_finest_relative": _worst(summary, "finest_relative_error"),
            "worst_richardson_observed": _worst(
                summary, "richardson_observed_order_relative_error"
            ),
            "worst_richardson_p2": _worst(summary, "richardson_p2_relative_error"),
            "all_monotone": all(entry["monotone"] for entry in summary.values()),
            "verdict_finest": (
                "PASS"
                if _worst(summary, "finest_relative_error") <= GATE_TOTAL
                else "FAIL"
            ),
            "verdict_richardson_observed": (
                "PASS"
                if _worst(summary, "richardson_observed_order_relative_error")
                <= GATE_TOTAL
                else "FAIL"
            ),
        }

    if "octa" in stages:
        ladder = []
        for subdivisions in OCTA_LEVELS:
            body = _octa_body(cpt, subdivisions)
            ladder.append(
                {
                    "family": "tensorbem.geometry3d.hemisphere_hull_mesh",
                    "subdivisions": subdivisions,
                    "n_faces": int(body.mesh.nb_faces),
                    "volume": float(body.volume),
                    "scattered": {
                        f"{ka:.3f}": {k: _c(v) for k, v in _scattered(cpt, solver, body, ka).items()}
                        for ka in KA_GRID
                    },
                }
            )
        summary = _ladder_summary(ladder, run_case)
        out["octa_family"] = {
            "ladder": ladder,
            "cells": summary,
            "worst_finest_relative": _worst(summary, "finest_relative_error"),
            "worst_richardson_observed": _worst(
                summary, "richardson_observed_order_relative_error"
            ),
            "worst_richardson_p2": _worst(summary, "richardson_p2_relative_error"),
            "all_monotone": all(entry["monotone"] for entry in summary.values()),
            "verdict_finest": (
                "PASS"
                if _worst(summary, "finest_relative_error") <= GATE_TOTAL
                else "FAIL"
            ),
            "verdict_richardson_observed": (
                "PASS"
                if _worst(summary, "richardson_observed_order_relative_error")
                <= GATE_TOTAL
                else "FAIL"
            ),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    import capytaine as cpt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(CASE / "capytaine_convergence_audit.json"),
        help="JSON artefact to write",
    )
    parser.add_argument(
        "--stages",
        default="repro,frozen,uv,octa",
        help="comma separated subset of {repro,frozen,uv,octa}",
    )
    args = parser.parse_args(argv)
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    payload = build(cpt, stages)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())