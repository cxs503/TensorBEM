"""Staged runner for CASE-BEM-SUBMERGED-SPHERE-HAVELOCK round 3.

Round 2 (``docs/case_submerged_sphere_havelock_round2.md``) left two gaps it
declared honestly:

* the seven-point band sweep ``ka = 0.1 ... 5.0`` was never recomputed end to end
  under the spherical-patch area measure (the cluster artifact came back empty,
  and one shell command may not run 14 x 320-panel solves);
* the 1280-panel refinement was never produced under the spherical-patch measure;
* and its section 3.6 (reference-integrity audit) was injected with ``nan`` /
  an empty ``ka`` list -- a silent injection defect, not a physical result.

This module closes all three in one deterministic artifact.  Nothing is tuned:
criteria, mesh, measures and the multipole reference are exactly those of the
case pool entry (tolerance 5%, production mesh 320 panels).

    PYTHONPATH=src python3 tools/case_havelock_round3_stage.py --stage sweep --measure spherical_patch
    PYTHONPATH=src python3 tools/case_havelock_round3_stage.py --stage sweep --measure flat
    PYTHONPATH=src python3 tools/case_havelock_round3_stage.py --stage refine --ka 0.5 1.0 2.0
    PYTHONPATH=src python3 tools/case_havelock_round3_stage.py --stage integrity
    PYTHONPATH=src python3 tools/case_havelock_round3_stage.py --stage assemble
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np

from tensorbem.benchmarks import case_havelock_submerged_sphere as case
from tensorbem.benchmarks import submerged_sphere_havelock_integrity as integrity
from tensorbem.benchmarks.submerged_sphere_havelock import (
    AREA_MEASURE_FLAT,
    AREA_MEASURE_SPHERICAL,
    run_frequency_sweep,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "m21142"
FINAL = OUT / "case_havelock_round3.json"
SHARD_DIR = OUT / "shards"

KA_SWEEP = (0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
KA_REFINE = (0.5, 1.0, 2.0)
PRODUCTION_SUBDIVISION = 2
REFINE_SUBDIVISION = 3
RADIUS, SUBMERGENCE, DENSITY, GRAVITY = 1.0, 2.0, 1025.0, 9.80665

_WORKER = {"measure": AREA_MEASURE_SPHERICAL, "subdivision": PRODUCTION_SUBDIVISION}


def _worker_solve(ka: float) -> dict:
    """One frequency of one sweep (pool worker; identical arithmetic)."""
    return run_frequency_sweep(
        ka_values=(ka,), radius=RADIUS, submergence=SUBMERGENCE,
        density=DENSITY, gravity=GRAVITY,
        subdivision=int(_WORKER["subdivision"]),
        area_measure=_WORKER["measure"],
    )[0]


def parallel_sweep(measure: str, ka_values, subdivision: int = PRODUCTION_SUBDIVISION,
                   processes: int = 4) -> list[dict]:
    """Frequency sweep run one process per frequency."""
    _WORKER["measure"] = measure
    _WORKER["subdivision"] = subdivision
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=processes) as pool:
        rows = pool.map(_worker_solve, list(ka_values))
    return sorted(rows, key=lambda row: float(row["ka"]))


def _sweep_path(measure: str, subdivision: int) -> Path:
    return OUT / ("sweep_sub%d_%s.json" % (subdivision, measure))


def shard_path(measure: str, subdivision: int, ka: float) -> Path:
    return SHARD_DIR / ("sub%d_%s_ka%05.3f.json" % (subdivision, measure, float(ka)))


def collect_shards(measure: str, subdivision: int, ka_values) -> list[dict]:
    """Read the per-frequency shards back, in the declared ``ka`` order."""
    rows = []
    for ka in ka_values:
        path = shard_path(measure, subdivision, ka)
        if not path.exists():
            raise SystemExit("missing shard %s -- run that frequency first" % path)
        rows += json.loads(path.read_text())
    return rows


def stage_sweep(measure: str, ka_values, processes: int = 4) -> None:
    rows = parallel_sweep(measure, ka_values, PRODUCTION_SUBDIVISION, processes)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    for row in rows:
        shard_path(measure, PRODUCTION_SUBDIVISION, row["ka"]).write_text(
            json.dumps([row], indent=2, default=float))
    path = _sweep_path(measure, PRODUCTION_SUBDIVISION)
    path.write_text(json.dumps(sorted(rows, key=lambda r: r["ka"]), indent=2, default=float))
    worst = max(row["%s_rel_error" % k] for row in rows for k in case.COEFFICIENTS)
    print(json.dumps({"artifact": str(path), "measure": measure, "n_points": len(rows),
                      "worst_rel_error": worst,
                      "errors_b33": [round(row["b33_rel_error"] * 100, 3) for row in rows]},
                     indent=2))


def stage_refine(ka_values, processes: int = 3) -> None:
    rows = parallel_sweep(AREA_MEASURE_SPHERICAL, ka_values, REFINE_SUBDIVISION, processes)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    for row in rows:
        shard_path(AREA_MEASURE_SPHERICAL, REFINE_SUBDIVISION, row["ka"]).write_text(
            json.dumps([row], indent=2, default=float))
    path = _sweep_path(AREA_MEASURE_SPHERICAL, REFINE_SUBDIVISION)
    path.write_text(json.dumps(sorted(rows, key=lambda r: r["ka"]), indent=2, default=float))
    print(json.dumps({"artifact": str(path), "n_panels": rows[0]["n_panels"],
                      "ka": [row["ka"] for row in rows],
                      "errors": {k: [round(row["%s_rel_error" % k] * 100, 3) for row in rows]
                                 for k in case.COEFFICIENTS}}, indent=2))


def stage_integrity() -> None:
    block = integrity.run(ka_values=(0.1, 1.0, 5.0), radius=RADIUS, submergence=SUBMERGENCE,
                          density=DENSITY, gravity=GRAVITY)
    path = OUT / "reference_integrity.json"
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(block, indent=2, default=float))
    rows = block["truncation"]["rows"]
    deep = block["deep_submergence_limit"]["rows"]
    print(json.dumps({"artifact": str(path),
                      "truncation_ka": [row["ka"] for row in rows],
                      "truncation_max_rel_delta": max(row["max_truncation_rel_delta"] for row in rows),
                      "haskind_max_residual": max(row["haskind_error_z"] for row in rows),
                      "lamb_excess_at_f_over_a_20": [
                          row["a33_excess_over_lamb"] for row in deep
                          if row["submergence_ratio"] == max(r["submergence_ratio"] for r in deep)],
                      "low_frequency_b33_min": min(row["b33"] for row in block["low_frequency_damping"])},
                     indent=2))


# ---------------------------------------------------------------------------
# assembly + guards
# ---------------------------------------------------------------------------

def _find_nan(payload, path="") -> list[str]:
    """Every JSON path holding a NaN/inf -- injection must never hide one."""
    bad: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            bad += _find_nan(value, "%s.%s" % (path, key))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            bad += _find_nan(value, "%s[%d]" % (path, index))
    elif isinstance(payload, float) and not math.isfinite(payload):
        bad.append(path or "<root>")
    return bad


def _refine_order(rows: list[dict]) -> list[dict]:
    """Error ratio between the 320- and 1280-panel solutions at matched ka."""
    order = []
    for coarse, fine in zip(rows["sub2"], rows["sub3"]):
        assert abs(coarse["ka"] - fine["ka"]) < 1e-12, "ka mismatch in refinement pairing"
        entry = {"ka": coarse["ka"], "n_panels_coarse": coarse["n_panels"],
                 "n_panels_fine": fine["n_panels"]}
        for key in case.COEFFICIENTS:
            err_c = coarse["%s_rel_error" % key]
            err_f = fine["%s_rel_error" % key]
            entry["%s_error_ratio" % key] = err_c / err_f if err_f > 1e-15 else float("nan")
            entry["%s_error_drop" % key] = err_c - err_f
        order.append(entry)
    return order


def assemble() -> dict:
    integrity_path = OUT / "reference_integrity.json"
    if not integrity_path.exists():
        raise SystemExit("missing %s -- run --stage integrity first" % integrity_path)
    spherical = collect_shards(AREA_MEASURE_SPHERICAL, PRODUCTION_SUBDIVISION, KA_SWEEP)
    flat = collect_shards(AREA_MEASURE_FLAT, PRODUCTION_SUBDIVISION, KA_SWEEP)
    try:
        fine = collect_shards(AREA_MEASURE_SPHERICAL, REFINE_SUBDIVISION, KA_REFINE)
        refinement_status = "completed"
    except SystemExit:
        fine = []
        refinement_status = "not_completed"
    ref_integrity = json.loads(integrity_path.read_text())

    judgement = {
        "tolerance": case.CASE_TOLERANCE,
        "criteria": (
            "per-item relative error |got-ref|/|ref| < 5% against the in-repo multipole "
            "reference, production mesh 320 panels, spherical-patch area measure"
        ),
        "production_sweep_spherical": case._verdicts(case._sweep_items(spherical)),
        "production_sweep_flat_control": case._verdicts(case._sweep_items(flat)),
    }
    if fine:
        judgement["refined_1280_spherical"] = case._verdicts(case._sweep_items(fine))

    artifact = {
        "case": "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
        "round": 3,
        "benchmark": "submerged_sphere_havelock_radiation_case",
        "purpose": (
            "close round-2 declared gaps: full ka band under the spherical-patch measure, "
            "1280-panel refinement under the same measure, and a non-degenerate "
            "reference-integrity block"
        ),
        "geometry": {
            "body": "sphere", "radius": RADIUS, "submergence": SUBMERGENCE,
            "submergence_ratio": SUBMERGENCE / RADIUS, "density": DENSITY,
            "gravity": GRAVITY, "displaced_volume": (4.0 / 3.0) * np.pi * RADIUS**3,
        },
        "nondimensional_parameters": {
            "ka": list(KA_SWEEP), "ka_refinement": list(KA_REFINE),
            "submergence_ratio_f_over_a": SUBMERGENCE / RADIUS,
            "normalisation": "A' = A/(rho*volume), B' = B/(rho*omega*volume)",
            "production_subdivision": PRODUCTION_SUBDIVISION,
            "refinement_subdivision": REFINE_SUBDIVISION,
        },
        "surface_measure": {
            "primary": AREA_MEASURE_SPHERICAL,
            "control": AREA_MEASURE_FLAT,
            "production_mesh": case._surface_measure_record(RADIUS, SUBMERGENCE,
                                                          PRODUCTION_SUBDIVISION),
            "refinement_mesh": case._surface_measure_record(RADIUS, SUBMERGENCE,
                                                            REFINE_SUBDIVISION),
        },
        "frequency_sweep": spherical,
        "control_frequency_sweep": flat,
        "refinement_status": refinement_status,
        "refinement_sweep_1280": fine,
        "refinement_error_ratios": (
            _refine_order({"sub2": spherical_sorted(spherical, KA_REFINE), "sub3": fine})
            if fine else []),
        "reference_integrity": ref_integrity,
        "judgement": judgement,
    }

    nan_paths = _find_nan(artifact)
    if nan_paths:
        raise SystemExit("non-finite value(s) in artifact: %s" % ", ".join(sorted(nan_paths)[:10]))
    for row in artifact["frequency_sweep"]:
        assert row["n_panels"] == 320, "production sweep is not on the 320-panel mesh: %r" % row["n_panels"]
    for row in artifact["refinement_sweep_1280"]:
        assert row["n_panels"] == 1280, "refinement sweep is not on the 1280-panel mesh"
    if refinement_status != "completed":
        print("NOTE: 1280-panel refinement not completed; artifact records the gap honestly")
    truncation_rows = ref_integrity["truncation"]["rows"]
    assert truncation_rows, "reference-integrity truncation table is empty"
    assert [row["ka"] for row in truncation_rows] == [0.1, 1.0, 5.0], (
        "reference-integrity ka list is not the declared band: %r"
        % [row["ka"] for row in truncation_rows])

    OUT.mkdir(parents=True, exist_ok=True)
    FINAL.write_text(json.dumps(artifact, indent=2, default=float))
    summary = {name: {"passed": block["passed"], "n_items": block["n_items"],
                      "n_failures": block["n_failures"],
                      "max_rel_error": (block["worst_item"] or {}).get("rel_error")}
               for name, block in judgement.items() if isinstance(block, dict) and "passed" in block}
    print(json.dumps({"artifact": str(FINAL), "judgement": summary}, indent=2))
    return artifact


def spherical_sorted(rows: list[dict], ka_values) -> list[dict]:
    """Restrict a sweep to the refinement frequencies, order fixed by ``ka_values``."""
    by_ka = {round(float(row["ka"]), 12): row for row in rows}
    return [by_ka[round(float(ka), 12)] for ka in ka_values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("sweep", "refine", "integrity", "assemble"),
                        required=True)
    parser.add_argument("--measure", choices=(AREA_MEASURE_SPHERICAL, AREA_MEASURE_FLAT),
                        default=AREA_MEASURE_SPHERICAL)
    parser.add_argument("--ka", type=float, nargs="+", default=list(KA_REFINE))
    parser.add_argument("--processes", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "sweep":
        stage_sweep(args.measure, tuple(args.ka), min(args.processes, len(args.ka)))
    elif args.stage == "refine":
        stage_refine(tuple(args.ka), min(args.processes, len(args.ka)))
    elif args.stage == "integrity":
        stage_integrity()
    else:
        assemble()


if __name__ == "__main__":
    main()