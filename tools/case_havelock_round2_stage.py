"""Staged runner for CASE-BEM-SUBMERGED-SPHERE-HAVELOCK round 2.

``run_case`` needs ~6 min of BEM solves (two 1280-panel levels); the agent shell
caps a single command at 120 s, so the case is assembled in two stages and the
judgement is built with the *same* helpers the one-shot ``run_case`` uses.

    PYTHONPATH=src python3 tools/case_havelock_round2_stage.py --stage a
    PYTHONPATH=src python3 tools/case_havelock_round2_stage.py --stage b

Stage ``a`` writes ``runs/m21069/_part_a.json`` (refinement at subdivisions 1-2,
low-frequency extrapolation, submergence sweep, reference integrity, geometry of
the production mesh), stage ``b`` adds the seven-point frequency sweep under both
surface measures, the comparison block and the judgement, and writes
``runs/m21069/case_havelock_round2.json``.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from tensorbem.benchmarks import case_havelock_submerged_sphere as case
from tensorbem.benchmarks import submerged_sphere_havelock_integrity as integrity
from tensorbem.benchmarks.submerged_sphere_havelock import (
    AREA_MEASURE_FLAT,
    AREA_MEASURE_SPHERICAL,
    run_convergence_study,
    run_frequency_sweep,
    run_low_frequency_limit,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "m21069"
PART_A = OUT / "_part_a.json"
FINAL = OUT / "case_havelock_round2.json"
KA_SWEEP = (0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
KA_LOW = (0.01, 0.02, 0.05, 0.1, 0.2)
SUBMERGENCE_RATIOS = (1.5, 2.0, 3.0)
SUBDIVISIONS = (1, 2)
RADIUS, SUBMERGENCE, DENSITY, GRAVITY = 1.0, 2.0, 1025.0, 9.80665


def _sweep_worker(ka: float) -> dict:
    """One frequency of the production-mesh sweep (pool worker)."""
    return run_frequency_sweep(
        ka_values=(ka,), radius=RADIUS, submergence=SUBMERGENCE,
        density=DENSITY, gravity=GRAVITY, area_measure=_WORKER_MEASURE[0],
    )[0]


_WORKER_MEASURE = [AREA_MEASURE_SPHERICAL]


def parallel_sweep(area_measure: str, ka_values=KA_SWEEP, processes: int = 4) -> list:
    """The 7-point sweep, one process per frequency (identical arithmetic)."""
    _WORKER_MEASURE[0] = area_measure
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=processes) as pool:
        rows = pool.map(_sweep_worker, list(ka_values))
    return sorted(rows, key=lambda row: row["ka"])


def stage_a() -> dict:
    study = dict(radius=RADIUS, submergence=SUBMERGENCE, density=DENSITY, gravity=GRAVITY)
    part = {
        "surface_measure_record": case._surface_measure_record(RADIUS, SUBMERGENCE, 2),
        "convergence_study": run_convergence_study(
            ka=0.8, subdivisions=SUBDIVISIONS, area_measure=AREA_MEASURE_SPHERICAL, **study
        ),
        "control_convergence_study": run_convergence_study(
            ka=0.8, subdivisions=SUBDIVISIONS, area_measure=AREA_MEASURE_FLAT, **study
        ),
        "low_frequency_limit": run_low_frequency_limit(
            ka_values=KA_LOW, area_measure=AREA_MEASURE_SPHERICAL, **study
        ),
        "submergence_sweep": case.run_submergence_sweep(
            submergence_ratios=SUBMERGENCE_RATIOS, ka=1.0, radius=RADIUS,
            density=DENSITY, gravity=GRAVITY, area_measure=AREA_MEASURE_SPHERICAL,
        ),
        "reference_integrity": integrity.run(
            ka_values=(0.1, 1.0, 5.0), radius=RADIUS, submergence=SUBMERGENCE,
            density=DENSITY, gravity=GRAVITY,
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    PART_A.write_text(json.dumps(part, indent=2, default=float))
    return part


PRIMARY_SWEEP = OUT / "_sweep_primary.json"


def stage_b1() -> None:
    """Primary (spherical-patch) frequency sweep -- 7 BEM solves."""
    sweep = parallel_sweep(AREA_MEASURE_SPHERICAL)
    PRIMARY_SWEEP.write_text(json.dumps(sweep, indent=2, default=float))
    print("primary sweep done:", [round(r["b33_rel_error"], 4) for r in sweep])


def stage_b() -> dict:
    part = json.loads(PART_A.read_text())
    study = dict(radius=RADIUS, submergence=SUBMERGENCE, density=DENSITY, gravity=GRAVITY)
    sweep = json.loads(PRIMARY_SWEEP.read_text())
    control = parallel_sweep(AREA_MEASURE_FLAT)
    items = case._collect_items(
        part["convergence_study"], sweep, part["low_frequency_limit"], part["submergence_sweep"]
    )
    rejection = [
        {"item": "refinement/sub%d" % level["subdivision"], "quantity": key,
         "rel_error": level["%s_rel_error" % key]}
        for level in part["convergence_study"]["levels"] for key in case.COEFFICIENTS
    ]
    artifact = {
        "case": "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
        "round": 2,
        "benchmark": "submerged_sphere_havelock_radiation_case",
        "geometry": {
            "body": "sphere", "radius": RADIUS, "submergence": SUBMERGENCE,
            "submergence_ratio": SUBMERGENCE / RADIUS, "density": DENSITY,
            "gravity": GRAVITY, "displaced_volume": (4.0 / 3.0) * np.pi * RADIUS**3,
        },
        "nondimensional_parameters": {
            "ka": list(KA_SWEEP), "ka_low": list(KA_LOW),
            "submergence_ratio_f_over_a": [SUBMERGENCE / RADIUS] + list(SUBMERGENCE_RATIOS),
            "normalisation": "A' = A/(rho*volume), B' = B/(rho*omega*volume)",
            "panel_subdivisions": list(SUBDIVISIONS),
        },
        "surface_measure": {
            "primary": AREA_MEASURE_SPHERICAL,
            "control": AREA_MEASURE_FLAT,
            "definition": {
                AREA_MEASURE_FLAT: "flat-triangle areas of the inscribed polyhedron (round 1)",
                AREA_MEASURE_SPHERICAL: (
                    "exact spherical-patch area each panel represents "
                    "(geometry3d.spherical_patch_areas); panels/normals/collocation/"
                    "singular kernels unchanged"
                ),
            },
            "production_mesh": part["surface_measure_record"],
        },
        "criteria": {
            "primary": "per-item relative error |got-ref|/|ref| < 5% (production mesh 320 panels)",
            "primary_area_measure": AREA_MEASURE_SPHERICAL,
            "control_area_measure": AREA_MEASURE_FLAT,
            "stopping_rule": (
                "one production mesh, one sweep, no adaptation after seeing errors; the "
                "round-2 surface measure is defined a priori (exact patch areas, no fitted "
                "constant) and the round-1 flat measure is judged in the same artifact by "
                "the same band"
            ),
        },
        "convergence_study": part["convergence_study"],
        "control_convergence_study": part["control_convergence_study"],
        "frequency_sweep": sweep,
        "control_frequency_sweep": control,
        "low_frequency_limit": part["low_frequency_limit"],
        "submergence_sweep": part["submergence_sweep"],
        "reference_integrity": part["reference_integrity"],
        "measure_comparison": case._measure_comparison(sweep, control),
        "judgement": {
            "primary_area_measure": AREA_MEASURE_SPHERICAL,
            "all_items": case._verdicts(items),
            "production_mesh": case._verdicts(case._sweep_items(sweep)),
            "control_area_measure": AREA_MEASURE_FLAT,
            "control_production_mesh": case._verdicts(case._sweep_items(control)),
            "refinement_levels": case._verdicts(rejection),
        },
        "supplementary_scale_aware_errors": case._supplementary_scale_aware(sweep),
    }
    FINAL.write_text(json.dumps(artifact, indent=2, default=float))
    prod, ctrl = artifact["judgement"]["production_mesh"], artifact["judgement"]["control_production_mesh"]
    summary = {
        "artifact": str(FINAL),
        "production_mesh": {"passed": prod["passed"], "max_rel_error": prod["max_rel_error"],
                            "n_failures": prod["n_failures"]},
        "control_production_mesh": {"passed": ctrl["passed"], "max_rel_error": ctrl["max_rel_error"],
                                    "n_failures": ctrl["n_failures"]},
        "all_items": {"passed": artifact["judgement"]["all_items"]["passed"],
                      "n_failures": artifact["judgement"]["all_items"]["n_failures"],
                      "n_items": len(items)},
        "refinement_levels": {"passed": artifact["judgement"]["refinement_levels"]["passed"],
                              "n_failures": artifact["judgement"]["refinement_levels"]["n_failures"]},
    }
    print(json.dumps(summary, indent=2))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("a", "b1", "b"), required=True)
    args = parser.parse_args()
    {"a": stage_a, "b1": stage_b1, "b": stage_b}[args.stage]()


if __name__ == "__main__":
    main()