#!/usr/bin/env python3
"""Round 6 (analysis, no new solves): is the case-mesh error still deficit-proportional?

The surface-measure study claims the flat-panel bias of this case is the *global*
area deficit of the inscribed icosphere.  That claim was verified on the 80- and
320-panel meshes only; the 1280-panel case mesh -- the mesh the pool gate uses --
was never measured under the exact measure (the declared tier T3 was recorded
``not_run``).  Nothing below is a new solve: it re-reads the committed
``numbers.json`` (flat case rows + ``area_deficit_diagnostic`` + the ka=0.8
refinement ladder) and tests whether the *scaling law* that the study relies on
survives on the case mesh.

Predeclared before reading the numbers of this script: if the mechanism is a
global measure deficit, then err/flat-deficit must stay roughly constant across
meshes (80 -> 320 -> 1280) for each coefficient; a drift by more than a factor
1.5 would falsify the mechanism at the case mesh and make the "it will drop once
the measure is exact" expectation unsupported.

Writes ``docs/case_bem_submerged_sphere_havelock_round6/mechanism_scaling.json``
and prints a markdown table.
"""
from __future__ import annotations
import json
from pathlib import Path

CASEDIR = Path("docs/case_bem_submerged_sphere_havelock")
COEF = ("a33", "b33", "a11", "b11")
TOL = 0.05


def main() -> int:
    numbers = json.loads((CASEDIR / "numbers.json").read_text())
    diag = numbers["area_deficit_diagnostic"]
    case_rows = {round(float(r["ka"]), 6): r for r in numbers["band_sweep_case"]}
    ladder = numbers["convergence_ladder"]
    deficit_by_sub = {int(r["subdivision"]): float(r["area_deficit"]) for r in diag["rows"]}
    ratio_by_sub = {
        int(r["subdivision"]): {c: float(r.get(f"ratio_{c}_over_deficit", float("nan"))) for c in COEF}
        for r in diag["rows"]
    }

    # (1) ladder at ka=0.8: err / deficit per coefficient, per mesh
    ladder_rows = []
    for lvl in ladder["levels"]:
        sub = int(lvl["subdivision"])
        row = {"subdivision": sub, "n_panels": int(lvl["n_panels"]),
               "area_deficit": deficit_by_sub[sub], "ka": float(ladder["ka"])}
        for c in COEF:
            e = float(lvl[f"{c}_rel_error"])
            row[f"{c}_rel_error"] = e
            row[f"{c}_over_deficit"] = e / deficit_by_sub[sub]
        ladder_rows.append(row)

    # (2) case mesh (1280) flat band: err / deficit per (ka, coefficient)
    case_ratios = []
    for ka, r in sorted(case_rows.items()):
        for c in COEF:
            e = float(r[f"{c}_rel_error"])
            case_ratios.append({"ka": ka, "quantity": c, "rel_error": e,
                                "over_deficit": e / deficit_by_sub[3]})

    # (3) stability test of the ratio: ladder (3 meshes at one ka) + case band median
    stability = {}
    for c in COEF:
        vals = [lr[f"{c}_over_deficit"] for lr in ladder_rows]
        med_case = sorted(x["over_deficit"] for x in case_ratios if x["quantity"] == c)
        median_case = med_case[len(med_case) // 2]
        stability[c] = {
            "ladder_ratios": vals,
            "ladder_spread": (max(vals) - min(vals)) / (sum(vals) / len(vals)),
            "case_band_median_ratio": median_case,
            "case_vs_ladder_mean": median_case / (sum(vals) / len(vals)),
        }
    # (4) prediction for the pending exact-measure run on the case mesh
    predicted = {}
    for c in COEF:
        ratio = stability[c]["case_band_median_ratio"]
        resid = max(0.0, max(x["rel_error"] for x in case_ratios if x["quantity"] == c)
                    - ratio * deficit_by_sub[3])
        predicted[c] = {
            "flat_band_max_rel_error": max(x["rel_error"] for x in case_ratios if x["quantity"] == c),
            "predicted_patch_band_max_rel_error": resid,
            "expected_reduction_factor": (max(x["rel_error"] for x in case_ratios if x["quantity"] == c) / resid) if resid > 0 else None,
        }
    band_max_flat = max(x["rel_error"] for x in case_ratios)
    predicted_max_patch = max(v["predicted_patch_band_max_rel_error"] for v in predicted.values())

    out = {
        "case_code": "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
        "round": "round6",
        "kind": "reanalysis_of_committed_artefact (no new solves)",
        "source_artifact": "docs/case_bem_submerged_sphere_havelock/numbers.json",
        "deficit_by_subdivision": deficit_by_sub,
        "ladder_ka": float(ladder["ka"]),
        "ladder_rows": ladder_rows,
        "case_band_ratios": case_ratios,
        "ratio_stability": stability,
        "prediction_case_mesh_patch_measure": {
            "band_max_flat": band_max_flat,
            "band_max_patch_predicted": predicted_max_patch,
            "per_coefficient": predicted,
            "note": "prediction from err/deficit constancy; to be tested by cluster job cj4012 (T3), not yet returned",
        },
        "status": "prediction_registered_pending_measurement",
        "not_run": [
            "T3 case-mesh exact-measure solves (cluster job cj4012 queued, not returned in this session)",
            "per-ka rows of T1/T2 (declared in the study, never written: run interrupted before write_artifacts)",
            "5120-panel (subdivision 4) rung",
        ],
        "tolerance": TOL,
    }
    dest = CASEDIR.parent / "case_bem_submerged_sphere_havelock_round6" / "mechanism_scaling.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=float) + "\n")

    lines = ["| mesh (panels) | deficit | " + " | ".join(f"{c} err/deficit" for c in COEF) + " |",
             "|---|---|" + "---|" * len(COEF)]
    for lr in ladder_rows:
        lines.append(f"| s{lr['subdivision']} ({lr['n_panels']}) | {100*lr['area_deficit']:.4f} % | "
                     + " | ".join(f"{lr[f'{c}_over_deficit']:.2f}" for c in COEF) + " |")
    print("\n".join(lines))
    print()
    print("case-mesh (1280) flat band median err/deficit:",
          {c: round(stability[c]["case_band_median_ratio"], 3) for c in COEF})
    print("band-max flat %.4f%% -> predicted patch %.4f%%" % (100 * band_max_flat, 100 * predicted_max_patch))
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
