#!/usr/bin/env python3
"""Lid referee for the *excitation* channel at the three unresolved cells.

``referee_lid_arm.py`` / ``referee_lid_adjudication.py`` close the attribution
question for added mass and damping.  The case, however, scores six quantities:
``A11, B11, A33, B33`` **and** ``|Fx|, |Fz|``.  The lid arm never touched the
excitation channel, so for the three cells the xval arm could not attribute
(``ka = 2.58, 3.0, 4.0``) the exciting force was still judged only by the
*unlidded* capytaine, i.e. by the referee whose interior Dirichlet irregular
frequency is the open question in the first place.

This driver solves the diffraction problem of the same lid-closed referee and
scores the total wave exciting force

    F = froude_krylov_force(problem) + DiffractionResult.forces

non-dimensionalised by ``rho g A a^2`` (``A = a = 1``), which is the convention
the case driver, ``band_extension/`` and the Hulme series all use.  It is
deliberately the *same* body mesh as the lid radiation arm (the TensorBEM
hemisphere hull mesh, closed on the waterplane by the Nemoh lid) so that the
radiation and excitation halves of the adjudication refer to one referee.

Gates are declared in :data:`GATES` before the cross-code numbers are formed.
``E1``/``E4`` are pre-registered; ``E2`` is a referee-accuracy statement whose
threshold is fixed here, before these solves were run.

Usage::

    PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_hemisphere/\\
        referee_lid_excitation.py --out referee_lid_excitation.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import referee_lid_arm as lidmod  # noqa: E402  (same-directory helper)

RADIUS = lidmod.RADIUS
GRAVITY = lidmod.GRAVITY
RHO = lidmod.RHO
AMPLITUDE = 1.0
NORM = RHO * GRAVITY * AMPLITUDE * RADIUS ** 2

SUBDIVISIONS = 5  # 4096 wetted panels, the rung the radiation arm needed
TARGET_KA = (2.0, 2.58, 3.0, 3.5, 4.0)
UNRESOLVED_CELLS = (2.58, 3.0, 4.0)
CONTROL_CELLS = (2.0,)

FORCE_QUANTITIES = ("Fx", "Fz")
TB_CELL_FILES = (
    "band_extension/cells/extension_ka2.58_n1024_chief_on.json",
    "band_extension/cells/extension_ka3_n1024_chief_on.json",
    "band_extension/cells/extension_ka3.5_n1024_chief_on.json",
    "band_extension/cells/extension_ka4_n1024_chief_on.json",
    "band_edge_refinement/cells/ka3.5_sub5.json",
    "band_edge_refinement/cells/ka4_sub5.json",
)

GATES: dict[str, dict[str, Any]] = {
    "E1_open_referee_excitation_polluted": {
        "criterion": "without the lid, the referee's |F| error at ka = 2.58 exceeds "
                     "the 5% case tolerance (the irregular frequency spoils the "
                     "exciting force as well as the radiation coefficients)",
        "threshold": 0.05,
        "note": "pre-registered; a FAIL here would mean the pole is confined to the "
                "radiation channel, which is itself a finding",
    },
    "E2_lid_referee_excitation_accuracy": {
        "criterion": "worst lid-closed referee |F| error against the Hulme series over "
                     "TARGET_KA <= 5% (the case tolerance)",
        "threshold": 0.05,
        "note": "pre-registered threshold, fixed before these solves were run",
    },
    "E3_cross_code_excitation_defined": {
        "criterion": "at ka = 2.58, 3.0, 4.0 the TensorBEM |F| agrees with the "
                     "lid-closed referee within 5%, so the attribution is defined",
        "threshold": 0.05,
        "note": "pre-registered",
    },
    "E4_lid_leaves_in_band_force_alone": {
        "criterion": "at ka = 2.0: |lid - open| / |open| <= 3% on both |Fx| and |Fz|",
        "threshold": 0.03,
        "note": "pre-registered",
    },
}


# --------------------------------------------------------------------------- #
# referee solve
# --------------------------------------------------------------------------- #
def capytaine_excitation(subdivisions: int, ka: float, lid: bool) -> dict[str, Any]:
    import capytaine as cpt

    mesh = lidmod.hull_mesh(subdivisions)
    kwargs: dict[str, Any] = {}
    if lid:
        lv, lf = lidmod.disk_lid()
        kwargs["lid_mesh"] = cpt.Mesh(vertices=lv, faces=lf)
    body = cpt.FloatingBody(
        mesh=cpt.Mesh(vertices=np.asarray(mesh.vertices, dtype=float),
                      faces=np.asarray(mesh.faces, dtype=np.int64)),
        dofs=cpt.rigid_body_dofs(rotation_center=(0.0, 0.0, 0.0)),
        **kwargs,
    )
    solver = cpt.BEMSolver()
    problem = cpt.DiffractionProblem(
        body=body,
        omega=math.sqrt(GRAVITY * ka / RADIUS),
        wave_direction=0.0,
        water_depth=math.inf,
        rho=RHO,
        g=GRAVITY,
    )
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = solver.solve(problem)
        fk = cpt.bem.airy_waves.froude_krylov_force(problem)
    out: dict[str, Any] = {
        "subdivisions": int(subdivisions),
        "panels": int(np.asarray(mesh.faces).shape[0]),
        "lid": bool(lid),
        "lid_panels": int(sum(len(f) - 2 for f in lidmod.disk_lid()[1])) if lid else 0,
    }
    total_c: dict[str, complex] = {}
    for dof, key in (("Surge", "Fx"), ("Heave", "Fz")):
        total = complex(fk[dof]) + complex(result.forces[dof])
        total_c[key] = total
        out[key] = abs(total) / NORM
        out[f"{key}_phase_deg"] = math.degrees(math.atan2(total.imag, total.real))
        out[f"{key}_fk"] = abs(complex(fk[dof])) / NORM
        out[f"{key}_scattering"] = abs(complex(result.forces[dof])) / NORM
    out["total_dimless"] = {
        k: [v.real / NORM, v.imag / NORM] for k, v in total_c.items()
    }
    out["seconds"] = time.time() - t0
    return out


def hulme_forces(ka: float) -> dict[str, float]:
    from tensorbem.benchmarks.hulme_hemisphere_case import reference as _ref

    got = _ref(float(ka), lidmod.REF_N_TERMS)
    return {q: float(got[f"{q}_dimless"]) for q in FORCE_QUANTITIES}


def tensorbem_forces(base: Path = HERE) -> dict[float, dict[str, Any]]:
    cells: dict[float, dict[str, Any]] = {}
    for name in TB_CELL_FILES:
        cell = json.loads((base / name).read_text())
        ka = float(cell["ka"])
        prev = cells.get(ka)
        if prev is None or cell["panels"] > prev["panels"]:
            cells[ka] = {
                "id": cell.get("id", f"ka{ka}_n{cell['panels']}"),
                "panels": int(cell["panels"]),
                "Fx": float(cell["Fx_dimless"]),
                "Fz": float(cell["Fz_dimless"]),
            }
    return cells


def _rel(a: float, b: float) -> float:
    return (a - b) / b if b else float("nan")


# --------------------------------------------------------------------------- #
# study
# --------------------------------------------------------------------------- #
def run(ka_list=TARGET_KA) -> dict[str, Any]:
    tb = tensorbem_forces()
    rows: list[dict[str, Any]] = []
    for ka in ka_list:
        ref = hulme_forces(ka)
        open_cell = capytaine_excitation(SUBDIVISIONS, ka, lid=False)
        lid_cell = capytaine_excitation(SUBDIVISIONS, ka, lid=True)
        row: dict[str, Any] = {
            "ka": ka,
            "role": ("unresolved_in_xval" if ka in UNRESOLVED_CELLS
                     else "in_band_control" if ka in CONTROL_CELLS
                     else "extension_control"),
            "reference": ref,
            "open_referee": open_cell,
            "lid_referee": lid_cell,
            "open_error": {q: abs(_rel(open_cell[q], ref[q])) for q in FORCE_QUANTITIES},
            "lid_error": {q: abs(_rel(lid_cell[q], ref[q])) for q in FORCE_QUANTITIES},
            "lid_vs_open": {
                q: abs(_rel(lid_cell[q], open_cell[q])) for q in FORCE_QUANTITIES
            },
        }
        cell = tb.get(ka)
        if cell is not None:
            row["tensorbem"] = cell
            row["tensorbem_error"] = {q: abs(_rel(cell[q], ref[q])) for q in FORCE_QUANTITIES}
            row["cross_code_vs_lid"] = {
                q: abs(_rel(cell[q], lid_cell[q])) for q in FORCE_QUANTITIES
            }
        row["open_worst"] = max(row["open_error"].values())
        row["lid_worst"] = max(row["lid_error"].values())
        rows.append(row)

    # ------------------------------------------------------------- gates
    first = next(r for r in rows if abs(r["ka"] - 2.58) < 1e-12)
    e1 = first["open_worst"] > GATES["E1_open_referee_excitation_polluted"]["threshold"]
    e2_worst = max(r["lid_worst"] for r in rows)
    e2 = e2_worst <= GATES["E2_lid_referee_excitation_accuracy"]["threshold"]
    unresolved = [r for r in rows if r["role"] == "unresolved_in_xval" and "cross_code_vs_lid" in r]
    e3_cells = {
        f"{r['ka']:.2f}": {
            "panels": r["tensorbem"]["panels"],
            "cross_code": r["cross_code_vs_lid"],
            "worst": max(r["cross_code_vs_lid"].values()),
        }
        for r in unresolved
    }
    e3 = bool(unresolved) and all(
        v["worst"] <= GATES["E3_cross_code_excitation_defined"]["threshold"]
        for v in e3_cells.values()
    )
    ctrl = [r for r in rows if r["ka"] in CONTROL_CELLS]
    e4_worst = max((max(r["lid_vs_open"].values()) for r in ctrl), default=float("nan"))
    e4 = bool(ctrl) and e4_worst <= GATES["E4_lid_leaves_in_band_force_alone"]["threshold"]

    gates = {
        "E1_open_referee_excitation_polluted": {
            **GATES["E1_open_referee_excitation_polluted"],
            "measured": {"ka": first["ka"], "open_error": first["open_error"],
                         "worst": first["open_worst"]},
            "verdict": "PASS" if e1 else "FAIL",
        },
        "E2_lid_referee_excitation_accuracy": {
            **GATES["E2_lid_referee_excitation_accuracy"],
            "measured": {"worst": e2_worst,
                         "per_ka": {f"{r['ka']:.2f}": r["lid_error"] for r in rows}},
            "verdict": "PASS" if e2 else "FAIL",
        },
        "E3_cross_code_excitation_defined": {
            **GATES["E3_cross_code_excitation_defined"],
            "measured": e3_cells,
            "verdict": "PASS" if e3 else "FAIL",
        },
        "E4_lid_leaves_in_band_force_alone": {
            **GATES["E4_lid_leaves_in_band_force_alone"],
            "measured": {"worst": e4_worst},
            "verdict": "PASS" if e4 else "FAIL",
        },
    }
    overall = "PASS" if all(g["verdict"] == "PASS" for g in gates.values()) else "FAIL"
    return {
        "case_code": "BEM-HULME-HEMISPHERE",
        "study": "referee_lid_excitation",
        "generated_at": None,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "capytaine": _capytaine_version(),
            "machine": platform.machine(),
            "host": platform.node(),
        },
        "constants": {
            "radius": RADIUS,
            "gravity": GRAVITY,
            "density": RHO,
            "amplitude": AMPLITUDE,
            "norm": NORM,
            "subdivisions": SUBDIVISIONS,
            "panels": "4096 wetted + 960 lid",
            "ref_n_terms": lidmod.REF_N_TERMS,
            "target_ka": list(ka_list),
            "force_convention": "total = froude_krylov_force + DiffractionResult.forces; "
                                "|F| / (rho g A a^2)",
        },
        "rows": rows,
        "gates": gates,
        "overall": overall,
        "repro_cmd": "PYTHONPATH=src python benchmarks/verified/"
                     "casepool_bem_hulme_hemisphere/referee_lid_excitation.py",
    }


def _capytaine_version() -> str:
    try:
        import capytaine

        return str(getattr(capytaine, "__version__", "unknown"))
    except Exception:  # pragma: no cover
        return "unavailable"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "referee_lid_excitation.json"))
    args = ap.parse_args(argv)
    import datetime as _dt

    payload = run()
    payload["generated_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    Path(args.out).write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {args.out}")
    for k, g in payload["gates"].items():
        print(f"  {k}: {g['verdict']}  {json.dumps(g['measured'])[:160]}")
    print(f"  overall: {payload['overall']}")
    return 0 if payload["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())