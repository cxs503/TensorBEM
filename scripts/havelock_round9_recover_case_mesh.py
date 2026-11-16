#!/usr/bin/env python3
"""Round 9: recover and independently adjudicate the *already finished* case-mesh
surface-measure measurement of the Havelock submerged-sphere anchor.

Anchor: ``CASE-BEM-SUBMERGED-SPHERE-HAVELOCK`` (pool ``BEM-SUBMERGED-SPHERE-HAVELOCK``,
reference Havelock 1955 frequency-domain multipole solution, tolerance < 5 %).

Why this round exists
---------------------
The merged tree carries ``docs/case_bem_submerged_sphere_havelock/measure_correction/
verdicts.json``, which records in its own ``not_run`` list that the *case mesh*
(1280 panels, the mesh the pool gate is applied to) was never measured under the
exact-surface-measure intervention, and that the per-``ka`` rows of the study were
lost when the run was interrupted.  Rounds 5-8 of this anchor were built on that
statement: they re-derived T1/T2 rows (round 7), registered a quantitative
prediction for T3 (round 6: band-max 0.193 %) and adjudicated the deficit-share
mechanism on the coarse tiers (round 8), all the while recording T3 as
``not_run`` and re-submitting cluster jobs for it.

That statement is stale.  The study *was* run to completion: the artefact
``docs/case_bem_submerged_sphere_havelock/measure_correction/numbers.json``
(1890 s of wall clock on the loongarch cluster, tier T3 alone 2281 s) was
committed on the agent branch ``agent/m20156`` and never merged, so main never
saw it and every later round re-derived its absence from the *stale* T1/T2-only
``verdicts.json`` instead of from the branch.  Round 9 recovers that measurement
verbatim (see ``recovered/PROVENANCE.json``, sha256 pinned) and adjudicates it
with code that does not trust the branch's own verdicts.

What this script does (no new solves, ~1 min, no cluster)
--------------------------------------------------------
1. **Provenance gate**: the three recovered files must hash to the pinned sha256.
2. **Arithmetic**: every stored ``*_rel_error`` is recomputed from the raw
   coefficient and reference values (``|got-ref|/|ref|``) and must match.
3. **Independent verdicts**: per (tier, measure) the 5 % gate is re-applied item
   by item from the rows -- n_items, n_over_gate, band-max and its location are
   recomputed from scratch and compared against (a) the branch's committed
   ``verdict.json`` and (b) the merged study's own ``_verdict`` code object.
4. **Routing identity R1**: the ``flat`` rows of the recovered T3 must reproduce
   the *gated* case-mesh table ``docs/case_bem_submerged_sphere_havelock/
   numbers.json::band_sweep_case`` -- i.e. the recovered run is the same code
   path as the published case table, so its patch/uniform rows are comparable
   with it.  Without this check the recovery would be an anecdote.
5. **Registered prediction R-A4** (declared in round 6/8, *before* any T3
   measurement existed): the case-mesh band-max under the exact measure was
   predicted to be 0.19324 % with a factor-2 acceptance band, measured/predicted
   in [0.5, 2].  Judged here on the recovered measurement, not on an extrapolation.
6. **P1 / P2 / P3** at the case mesh, using the *pre-declared* definitions of the
   round-6 driver ``scripts/havelock_measure_case_mesh.py`` (its ``_decomposition``
   is imported, not re-implemented).

Honest scope
------------
The recovered run covered the *declared* case band (ka = 0.1 / 1.0 / 3.0).  The
widened 7-wavenumber band (0.1 ... 5.0) at the case mesh under patch/uniform was
never measured and stays ``not_run``; nothing here extrapolates to it.

    PYTHONPATH=src python3 scripts/havelock_round9_recover_case_mesh.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RECOVERED = ROOT / "docs/case_bem_submerged_sphere_havelock_round9" / "recovered"
PROVENANCE = RECOVERED / "PROVENANCE.json"
STUDY_NUMBERS = RECOVERED / "numbers_study_full.json"
STUDY_VERDICT = RECOVERED / "verdict_study.json"
STUDY_T3_CSV = RECOVERED / "table_T3_case_s3_band.csv"
COMMITTED_CASE_NUMBERS = ROOT / "docs" / "case_bem_submerged_sphere_havelock" / "numbers.json"
ROUND6_MECHANISM = ROOT / "docs" / "case_bem_submerged_sphere_havelock_round6" / "mechanism_scaling.json"
ROUND6_DRIVER = ROOT / "scripts" / "havelock_measure_case_mesh.py"
STUDY_MODULE = (
    ROOT / "benchmarks/verified/casepool_bem_submerged_sphere_havelock/measure_correction_study.py"
)
CASE_CODE = "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK"
COEFFICIENTS = ("a33", "b33", "a11", "b11")
DAMPING = ("b33", "b11")
MEASURES = ("flat", "patch", "uniform")
GATE = 0.05
ROUTING_TOL = 1e-12
ARITH_TOL = 1e-12
#: R-A4 acceptance band for the round-6 registered prediction (declared in round 8).
RA4_FACTOR = 2.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provenance_gate() -> dict[str, Any]:
    prov = json.loads(PROVENANCE.read_text())
    checks = {}
    ok = True
    for rel, expected in prov["sha256"].items():
        path = RECOVERED / rel.replace("recovered/", "")
        actual = _sha256(path)
        match = bool(actual == expected)
        ok = ok and match
        checks[rel] = {"expected": expected, "actual": actual, "match": match}
    return {
        "status": "PASS" if ok else "FAIL",
        "source_branch": prov["source_branch"],
        "source_commit": prov["source_commit"],
        "runtime_seconds_total": prov["runtime_seconds_total"],
        "files": checks,
    }


def _rows(study: dict[str, Any], tier: str) -> list[dict[str, Any]]:
    return list(study["tiers"][tier]["rows"])


def _recompute_errors(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Recompute |got-ref|/|ref| from the raw numbers; compare with the stored column."""
    worst = 0.0
    worst_at: dict[str, Any] = {}
    for row in rows:
        for c in COEFFICIENTS:
            got = float(row[c])
            ref = float(row[f"{c}_ref"])
            recomputed = abs(got - ref) / abs(ref)
            stored = float(row[f"{c}_rel_error"])
            diff = abs(recomputed - stored)
            if diff > worst:
                worst = diff
                worst_at = {"ka": float(row["ka"]), "measure": row["measure"], "quantity": c}
    return {
        "status": "PASS" if worst <= ARITH_TOL else "FAIL",
        "tolerance": ARITH_TOL,
        "worst_abs_diff": worst,
        "worst_at": worst_at,
        "n_compared": len(rows) * len(COEFFICIENTS),
    }


def _verdict_from_rows(rows: Sequence[dict[str, Any]], tier: str) -> dict[str, Any]:
    """Independent re-derivation of the 5 % gate, per measure, item by item."""
    out: dict[str, Any] = {}
    for measure in MEASURES:
        sub = [r for r in rows if r["measure"] == measure]
        if not sub:
            continue
        items = [
            {"ka": float(r["ka"]), "quantity": c, "rel_error": abs(float(r[c]) - float(r[f"{c}_ref"])) / abs(float(r[f"{c}_ref"]))}
            for r in sub
            for c in COEFFICIENTS
        ]
        best = max(items, key=lambda it: it["rel_error"])
        out[measure] = {
            "tier": tier,
            "n_items": len(items),
            "n_items_over_gate": sum(1 for it in items if it["rel_error"] > GATE),
            "band_max_rel_error": best["rel_error"],
            "band_max_at": {"ka": best["ka"], "quantity": best["quantity"]},
            "verdict": "FAIL" if any(it["rel_error"] > GATE for it in items) else "PASS",
            "items": items,
        }
    return out


def _verdict_agreement(mine: dict[str, Any], theirs: dict[str, Any], label: str) -> dict[str, Any]:
    diffs = []
    ok = True
    for measure in MEASURES:
        if measure not in mine or measure not in theirs:
            continue
        a, b = mine[measure], theirs[measure]
        same_n = int(a["n_items"]) == int(b["n_items"])
        same_over = int(a["n_items_over_gate"]) == int(b["n_items_over_gate"])
        same_verdict = a["verdict"] == b["verdict"]
        rel = abs(float(a["band_max_rel_error"]) - float(b["band_max_rel_error"])) / abs(
            float(b["band_max_rel_error"])
        )
        same_max = rel < 1e-12
        same_where = (
            round(float(a["band_max_at"]["ka"]), 6) == round(float(b["band_max_at"]["ka"]), 6)
            and a["band_max_at"]["quantity"] == b["band_max_at"]["quantity"]
        )
        good = same_n and same_over and same_verdict and same_max and same_where
        ok = ok and good
        diffs.append(
            {
                "measure": measure,
                "n_items": [a["n_items"], b["n_items"]],
                "n_over_gate": [a["n_items_over_gate"], b["n_items_over_gate"]],
                "band_max": [a["band_max_rel_error"], b["band_max_rel_error"]],
                "band_max_at": [a["band_max_at"], b["band_max_at"]],
                "verdict": [a["verdict"], b["verdict"]],
                "agree": good,
            }
        )
    return {"status": "PASS" if ok else "FAIL", "label": label, "per_measure": diffs}


def _routing_identity(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """R1: recovered T3 ``flat`` rows vs the gated case-mesh table on main."""
    if not COMMITTED_CASE_NUMBERS.exists():
        return {"status": "not_run", "reason": "committed case numbers.json unavailable"}
    committed = {
        round(float(r["ka"]), 6): r
        for r in json.loads(COMMITTED_CASE_NUMBERS.read_text())["band_sweep_case"]
    }
    worst = 0.0
    per_ka = []
    n_cmp = 0
    for row in rows:
        if row["measure"] != "flat":
            continue
        ka = round(float(row["ka"]), 6)
        if ka not in committed:
            continue
        n_cmp += 1
        diffs = {
            c: abs(float(row[c]) - float(committed[ka][f"{c}_bem"]))
            / abs(float(committed[ka][f"{c}_bem"]))
            for c in COEFFICIENTS
        }
        worst = max(worst, max(diffs.values()))
        per_ka.append({"ka": ka, "max_rel_diff": max(diffs.values())})
    return {
        "status": "PASS" if worst < ROUTING_TOL else "FAIL",
        "tolerance": ROUTING_TOL,
        "worst_rel_diff": worst,
        "n_ka_compared": n_cmp,
        "per_ka": per_ka,
        "note": "same mesh (subdivision 3, f/a = 2.0) and same code path as the gated table",
    }


def _registered_prediction(measured_patch_band_max: float) -> dict[str, Any]:
    """R-A4: round-6 prediction for the case mesh, judged on a real measurement."""
    if not ROUND6_MECHANISM.exists():
        return {"status": "not_run", "reason": "round-6 mechanism artefact unavailable"}
    mech = json.loads(ROUND6_MECHANISM.read_text())
    pred = mech["prediction_case_mesh_patch_measure"]
    predicted = float(pred["band_max_patch_predicted"])
    flat_committed = float(pred["band_max_flat"])
    ratio = measured_patch_band_max / predicted
    inside = bool(1.0 / RA4_FACTOR <= ratio <= RA4_FACTOR)
    per_coef = {}
    for c, entry in pred["per_coefficient"].items():
        per_coef[c] = {
            "flat_band_max": float(entry["flat_band_max_rel_error"]),
            "predicted_patch_band_max": float(entry["predicted_patch_band_max_rel_error"]),
            "expected_reduction_factor": float(entry["expected_reduction_factor"]),
        }
    return {
        "rule": f"measured/predicted inside [{1.0/RA4_FACTOR:g}, {RA4_FACTOR:g}] (declared in round 8, before any T3 measurement)",
        "predicted_patch_band_max": predicted,
        "measured_patch_band_max": measured_patch_band_max,
        "measured_over_predicted": ratio,
        "committed_flat_band_max": flat_committed,
        "realised_reduction_factor": flat_committed / measured_patch_band_max,
        "holds": inside,
        "verdict": "CONFIRMED" if inside else "FALSIFIED",
        "per_coefficient_prediction": per_coef,
    }


def _patch_vs_uniform(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Is the patch measure distinguishable from the scalar ``uniform`` control?"""
    by_ka: dict[float, dict[str, dict[str, float]]] = {}
    for row in rows:
        by_ka.setdefault(round(float(row["ka"]), 6), {})[row["measure"]] = {
            c: abs(float(row[c]) - float(row[f"{c}_ref"])) / abs(float(row[f"{c}_ref"]))
            for c in COEFFICIENTS
        }
    cells = []
    for ka in sorted(by_ka):
        for c in COEFFICIENTS:
            e_p = by_ka[ka]["patch"][c]
            e_u = by_ka[ka]["uniform"][c]
            cells.append(
                {
                    "ka": ka,
                    "quantity": c,
                    "err_patch": e_p,
                    "err_uniform": e_u,
                    "patch_minus_uniform": e_p - e_u,
                    "rel_gap": (e_p - e_u) / e_u if e_u else float("inf"),
                }
            )
    damping = [c for c in cells if c["quantity"] in DAMPING]
    return {
        "max_abs_rel_gap_any_cell": max(abs(c["rel_gap"]) for c in cells),
        "max_abs_rel_gap_damping": max(abs(c["rel_gap"]) for c in damping),
        "n_patch_better_damping": sum(1 for c in damping if c["patch_minus_uniform"] < 0),
        "n_damping_cells": len(damping),
        "cells": cells,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build(out_path: Path) -> dict[str, Any]:
    prov = _provenance_gate()
    study = json.loads(STUDY_NUMBERS.read_text())
    branch_verdict = json.loads(STUDY_VERDICT.read_text())

    verdicts: dict[str, Any] = {}
    arithmetic: dict[str, Any] = {}
    agreement_branch: dict[str, Any] = {}
    agreement_merged: dict[str, Any] = {}
    merged_study = _load_module(STUDY_MODULE, "_round9_mcs")

    for tier in study["tiers"]:
        rows = _rows(study, tier)
        verdicts[tier] = _verdict_from_rows(rows, tier)
        arithmetic[tier] = _recompute_errors(rows)
        if tier in branch_verdict:
            agreement_branch[tier] = _verdict_agreement(
                verdicts[tier], branch_verdict[tier], "branch verdict.json"
            )
        merged = merged_study._verdict(rows, tier)
        agreement_merged[tier] = _verdict_agreement(
            verdicts[tier], merged, "merged study _verdict()"
        )

    t3 = "T3:case_s3_band"
    t3_rows = _rows(study, t3)
    routing = _routing_identity(t3_rows)
    measured_patch = verdicts[t3]["patch"]["band_max_rel_error"]
    prediction = _registered_prediction(measured_patch)

    # P1/P2/P3 with the pre-declared round-6 definitions (imported, not re-implemented)
    round6 = _load_module(ROUND6_DRIVER, "_round9_round6_driver")
    decomposition = round6._decomposition(t3_rows, subdivision=3, submergence=2.0)

    declared_band = sorted({round(float(r["ka"]), 6) for r in t3_rows})
    band_full = (0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
    missing_band = [ka for ka in band_full if ka not in declared_band]

    # the CSV table must agree with the JSON rows (two committed views of one run)
    csv_rows = STUDY_T3_CSV.read_text().strip().splitlines()
    csv_n = len(csv_rows) - 1
    csv_ok = csv_n == len(t3_rows)

    checks = {
        "provenance": prov,
        "arithmetic": arithmetic,
        "verdict_vs_branch": agreement_branch,
        "verdict_vs_merged_code": agreement_merged,
        "routing_identity_R1": routing,
        "csv_row_count_matches_json": {
            "status": "PASS" if csv_ok else "FAIL",
            "csv_rows": csv_n,
            "json_rows": len(t3_rows),
        },
    }
    hard_fail = [
        name
        for name, chk in (
            ("provenance", prov["status"]),
            ("routing_identity_R1", routing["status"]),
            ("csv_row_count_matches_json", checks["csv_row_count_matches_json"]["status"]),
            *[
                (f"arithmetic[{t}]", arithmetic[t]["status"])
                for t in arithmetic
            ],
            *[
                (f"verdict_vs_branch[{t}]", agreement_branch[t]["status"])
                for t in agreement_branch
            ],
            *[
                (f"verdict_vs_merged_code[{t}]", agreement_merged[t]["status"])
                for t in agreement_merged
            ],
        )
        if chk not in ("PASS",)
    ]

    result = {
        "case_code": CASE_CODE,
        "round": "round9",
        "kind": "recovery_and_independent_adjudication_of_an_unmerged_measurement",
        "gate": GATE,
        "recovered_from": {
            "branch": prov["source_branch"],
            "commit": prov["source_commit"],
            "runtime_seconds_total": prov["runtime_seconds_total"],
            "tier_runtime_seconds": json.loads(PROVENANCE.read_text())["runtime_by_tier"],
        },
        "checks": checks,
        "checks_failed": hard_fail,
        "case_mesh_verdicts": verdicts[t3],
        "all_tier_verdicts": {
            tier: {m: {k: verdicts[tier][m][k] for k in ("n_items", "n_items_over_gate", "band_max_rel_error", "verdict")} | {"band_max_at": verdicts[tier][m]["band_max_at"]}
                   for m in verdicts[tier]}
            for tier in verdicts
        },
        "registered_prediction_RA4": prediction,
        "predeclared_P1_P2_P3": {
            "P1": {k: decomposition["prediction_P1"][k] for k in ("claim", "max_flat", "max_patch", "factor", "holds")},
            "P2": {k: decomposition["prediction_P2"][k] for k in ("claim", "deficit", "n_items", "n_holding", "min_frac_of_deficit", "median_frac_of_deficit", "holds")},
            "P3": {k: decomposition["prediction_P3"][k] for k in ("claim", "n_damping_cells", "n_patch_beats_uniform", "flat_minus_patch_min", "flat_minus_uniform_min", "holds")},
            "area_deficit_s3": decomposition["area_deficit_s3"],
        },
        "patch_vs_uniform_control": _patch_vs_uniform(t3_rows),
        "bands": {
            "declared_case_band": declared_band,
            "full_band_reference": list(band_full),
            "full_band_not_measured": missing_band,
        },
        "not_run": [
            "case mesh (1280 panels) under patch / uniform on the widened 7-wavenumber band "
            f"ka in {missing_band}: only the declared band {declared_band} was measured",
            "no new solves were started in this round (the recovered run is the measurement)",
        ],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "declared_band_from_merged_study": list(merged_study.CASE_BAND),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1, default=float))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default=str(ROOT / "docs/case_bem_submerged_sphere_havelock_round9" / "adjudicate.json"),
    )
    args = ap.parse_args(argv)
    res = build(Path(args.out))
    print(json.dumps(
        {
            "checks_failed": res["checks_failed"],
            "R1": res["checks"]["routing_identity_R1"]["worst_rel_diff"],
            "T3": res["all_tier_verdicts"]["T3:case_s3_band"],
            "RA4": {k: res["registered_prediction_RA4"][k] for k in ("predicted_patch_band_max", "measured_patch_band_max", "measured_over_predicted", "verdict")},
            "P1_factor": res["predeclared_P1_P2_P3"]["P1"]["factor"],
            "P2_holds": res["predeclared_P1_P2_P3"]["P2"]["holds"],
            "P3_holds": res["predeclared_P1_P2_P3"]["P3"]["holds"],
            "not_run": res["bands"]["full_band_not_measured"],
        },
        indent=1,
        default=float,
    ))
    return 0 if not res["checks_failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())