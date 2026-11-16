"""Round-8 adjudication of the surface-measure mechanism prediction (Havelock sphere).

CASE-BEM-SUBMERGED-SPHERE-HAVELOCK / pool code BEM-SUBMERGED-SPHERE-HAVELOCK.

What this closes
----------------
Round 6 registered a quantitative prediction of the "err proportional to the spherical-patch
area deficit" mechanism (``docs/case_bem_submerged_sphere_havelock_round6/mechanism_scaling.json``,
``prediction_case_mesh_patch_measure``) and P1/P2/P3 (``scripts/havelock_measure_case_mesh.py``),
but could only test it on committed *flat* data.  The mechanism claims (i) the flat-measure
error of a tier is the tier's area deficit mapped into the coefficients, and therefore
(ii) the exact patch measure removes (at least half of) that deficit from the damping error.
Round 7 wrote the per-ka T1 (320 panels) / T2 (80 panels) rows, so the *measured* halves of
that claim can now be adjudicated on two tiers.  The case mesh (T3, 1280 panels) still has no
measurement in this session, and is recorded as ``not_run`` -- never extrapolated into a pass.

Predeclared adjudication rules (fixed before the rows were read by this script)
------------------------------------------------------------------------------
R-P1  ``max_patch * 2 <= max_flat`` on the tier (patch must halve the band max).
R-P2  For *every* damping item (b33, b11) at *every* ka: ``(e_flat - e_patch)/deficit > 0.5``.
      One counterexample falsifies it.  This is the deficit-proportionality mechanism itself.
R-P3  patch must beat the ``uniform`` control (flat rescaled by sum(patch)/sum(flat)) on a
      strict majority of the damping cells; otherwise "the exact measure" is indistinguishable
      from a global area normalisation.
R-A4  (T3 only) ``0.5 <= measured/predicted <= 2.0`` for the band max and for each coefficient,
      against the round-6 registered values.  Not applied to T1/T2: the registration is
      case-mesh specific and using it elsewhere would be a post-hoc retarget.
R-A5  Read-only discipline: the 5 % gate, the honest economy/flat FAIL and the PASS of the
      gated flat table are inputs, never outputs of this script.

Outputs
-------
``<outdir>/adjudicate.json`` -- per-tier measured band statistics, R-P1/R-P2/R-P3 verdicts
with the rows they follow from, R-A4 when T3 exists, and the explicit ``not_run`` list.

Reproduce
---------
    python3 scripts/havelock_round8_adjudicate.py --job <cjN>
    python3 scripts/havelock_round8_figures.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "docs" / "case_bem_submerged_sphere_havelock"
TIERS = CASE_DIR / "measure_correction" / "numbers_t1t2_rows.json"
STUDY_T3 = CASE_DIR / "measure_correction" / "numbers_round6.json"
MECHANISM = ROOT / "docs" / "case_bem_submerged_sphere_havelock_round6" / "mechanism_scaling.json"
GATED = CASE_DIR / "numbers.json"
DEFAULT_OUT = ROOT / "docs" / "case_bem_submerged_sphere_havelock_round8"

COEFFICIENTS = ("a33", "b33", "a11", "b11")
DAMPING = ("b33", "b11")
MEASURES = ("flat", "patch", "uniform")
GATE = 0.05
FACTOR_BAND = (0.5, 2.0)
TIER_KEYS = {"T1": "T1:economy_s2_band", "T2": "T2:coarsest_s1_band"}
T3_KEY = "T3:case_s3_band_flat_patch_uniform"


def _gated_case_band_max(gated: dict[str, Any]) -> float:
    rows = gated["band_sweep_case"]
    return max(float(r[f"{c}_rel_error"]) for r in rows for c in COEFFICIENTS)


def _band_max(rows: Sequence[dict[str, Any]], measure: str, c: str = "band") -> tuple[float, dict[str, Any]]:
    sel = [r for r in rows if r["measure"] == measure]
    if not sel:
        raise SystemExit(f"no rows for measure {measure!r}")
    key = "band_max_rel_error" if c == "band" else f"{c}_rel_error"
    worst = max(sel, key=lambda r: float(r[key]))
    return float(worst[key]), {"ka": float(worst["ka"]), "quantity": c}


def adjudicate_tier(rows: Sequence[dict[str, Any]], deficit: float, label: str) -> dict[str, Any]:
    measured = {}
    for measure in MEASURES:
        value, where = _band_max(rows, measure)
        measured[measure] = {"band_max_rel_error": value, **where}

    per_coef = {}
    for c in COEFFICIENTS:
        entry = {}
        for measure in MEASURES:
            value, where = _band_max(rows, measure, c)
            entry[f"{measure}_band_max_rel_error"] = value
            entry[f"{measure}_at"] = where
        entry["reduction_factor_flat_over_patch"] = (
            entry["flat_band_max_rel_error"] / entry["patch_band_max_rel_error"]
            if entry["patch_band_max_rel_error"] > 0
            else math.inf
        )
        per_coef[c] = entry

    max_flat = measured["flat"]["band_max_rel_error"]
    max_patch = measured["patch"]["band_max_rel_error"]
    rp1 = {
        "rule": "R-P1: max_patch * 2 <= max_flat",
        "max_flat": max_flat,
        "max_patch": max_patch,
        "factor": (max_flat / max_patch) if max_patch > 0 else math.inf,
        "holds": bool(max_patch * 2.0 <= max_flat),
    }

    by_ka: dict[float, dict[str, dict[str, float]]] = {}
    for r in rows:
        by_ka.setdefault(round(float(r["ka"]), 6), {})[str(r["measure"])] = {
            c: float(r[f"{c}_rel_error"]) for c in COEFFICIENTS
        }
    cells = []
    for ka in sorted(by_ka):
        cell = by_ka[ka]
        for c in DAMPING:
            reduction = cell["flat"][c] - cell["patch"][c]
            cells.append(
                {
                    "ka": ka,
                    "quantity": c,
                    "err_flat": cell["flat"][c],
                    "err_patch": cell["patch"][c],
                    "err_uniform": cell["uniform"][c],
                    "reduction_abs": reduction,
                    "reduction_frac_of_deficit": reduction / deficit,
                    "holds": bool(reduction / deficit > 0.5),
                    "patch_beats_uniform": bool(cell["patch"][c] <= cell["uniform"][c]),
                }
            )
    fracs = sorted(c["reduction_frac_of_deficit"] for c in cells)
    rp2 = {
        "rule": "R-P2: (e_flat - e_patch)/deficit > 0.5 for every damping item",
        "deficit": deficit,
        "n_items": len(cells),
        "n_holding": sum(1 for c in cells if c["holds"]),
        "min_frac_of_deficit": fracs[0],
        "median_frac_of_deficit": fracs[len(fracs) // 2],
        "holds": bool(all(c["holds"] for c in cells)),
        "counterexamples": [
            {"ka": c["ka"], "quantity": c["quantity"], "frac": c["reduction_frac_of_deficit"]}
            for c in cells
            if not c["holds"]
        ],
        "rows": cells,
    }
    n_beats = sum(1 for c in cells if c["patch_beats_uniform"])
    rp3 = {
        "rule": "R-P3: patch beats uniform on a strict majority of damping cells",
        "n_damping_cells": len(cells),
        "n_patch_beats_uniform": n_beats,
        "holds": bool(n_beats > len(cells) / 2.0),
    }

    verdicts = {}
    for measure in MEASURES:
        sel = [r for r in rows if r["measure"] == measure]
        n_bad = sum(1 for r in sel for c in COEFFICIENTS if float(r[f"{c}_rel_error"]) >= GATE)
        worst = max(float(r[f"{c}_rel_error"]) for r in sel for c in COEFFICIENTS)
        verdicts[measure] = {
            "n_panels": int(sel[0]["n_panels"]) if sel else None,
            "n_items": 4 * len(sel),
            "n_items_over_gate": n_bad,
            "band_max_rel_error": worst,
            "verdict": "PASS" if n_bad == 0 else "FAIL",
        }

    return {
        "tier": label,
        "n_panels": int(rows[0]["n_panels"]),
        "band": sorted({round(float(r["ka"]), 6) for r in rows}),
        "area_deficit": deficit,
        "measured_band_max": measured,
        "per_coefficient": per_coef,
        "verdicts_per_measure": verdicts,
        "R_P1": rp1,
        "R_P2": rp2,
        "R_P3": rp3,
    }


def adjudicate(job: str | None = None) -> dict[str, Any]:
    mechanism = json.loads(MECHANISM.read_text())
    gated = json.loads(GATED.read_text())
    deficits = {int(k): float(v) for k, v in mechanism["deficit_by_subdivision"].items()}
    tiers_doc = json.loads(TIERS.read_text()) if TIERS.exists() else {"tiers": {}}
    published_t3 = json.loads(STUDY_T3.read_text()) if STUDY_T3.exists() else None

    blocks: dict[str, Any] = {}
    for label, key in TIER_KEYS.items():
        tier = tiers_doc["tiers"].get(key)
        if tier is None:
            continue
        subdivision = int(tier["rows"][0]["subdivision"])
        blocks[label] = adjudicate_tier(tier["rows"], deficits[subdivision], key)

    not_run = [
        "T3 case-mesh (1280 panels) flat/patch/uniform rows: no measurement in this session "
        "(round-7 job cj4350 ran against a stale node checkout without scripts/ -- head 104abe1; "
        "round-8 job cj4425 still queued at session end) -- the round-6 case-mesh prediction "
        "stays registered, not adjudicated",
        "T4/T5 (submergence off-design, low-ka limit) on the case mesh",
        "5120-panel (subdivision 4) rung",
    ]
    if published_t3 is not None:
        tier = published_t3["tiers"][T3_KEY]
        blocks["T3"] = adjudicate_tier(tier["rows"], deficits[3], T3_KEY)
        pred = mechanism["prediction_case_mesh_patch_measure"]
        band_ratio = (
            blocks["T3"]["measured_band_max"]["patch"]["band_max_rel_error"]
            / float(pred["band_max_patch_predicted"])
        )
        per = {}
        for c in COEFFICIENTS:
            measured = blocks["T3"]["per_coefficient"][c]["patch_band_max_rel_error"]
            predicted = float(pred["per_coefficient"][c]["predicted_patch_band_max_rel_error"])
            ratio = measured / predicted
            per[c] = {
                "measured": measured,
                "predicted": predicted,
                "measured_over_predicted": ratio,
                "within_factor_2": bool(FACTOR_BAND[0] <= ratio <= FACTOR_BAND[1]),
            }
        blocks["T3"]["R_A4_numeric_prediction"] = {
            "rule": "R-A4: 0.5 <= measured/predicted <= 2.0 (band and per coefficient)",
            "band": {
                "measured": blocks["T3"]["measured_band_max"]["patch"]["band_max_rel_error"],
                "predicted": float(pred["band_max_patch_predicted"]),
                "measured_over_predicted": band_ratio,
                "within_factor_2": bool(FACTOR_BAND[0] <= band_ratio <= FACTOR_BAND[1]),
            },
            "per_coefficient": per,
            "holds": bool(
                FACTOR_BAND[0] <= band_ratio <= FACTOR_BAND[1]
                and all(v["within_factor_2"] for v in per.values())
            ),
        }
        not_run = [item for item in not_run if not item.startswith("T3 case-mesh")]

    return {
        "case_code": "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
        "round": 8,
        "kind": "adjudication_of_the_registered_deficit_proportionality_mechanism_on_measured_tiers",
        "compute_job": job,
        "source_rows_T1_T2": str(TIERS.relative_to(ROOT)) if TIERS.exists() else None,
        "source_rows_T3": str(STUDY_T3.relative_to(ROOT)) if published_t3 is not None else None,
        "source_prediction": str(MECHANISM.relative_to(ROOT)),
        "tolerance": GATE,
        "tiers": blocks,
        "no_softening_A5": {
            "gated_case_mesh_verdict": gated["verdict"]["case_mesh"],
            "gated_economy_mesh_verdict": gated["verdict"]["economy_mesh"],
            "gated_case_mesh_band_max": _gated_case_band_max(gated),
            "untouched": bool(
                gated["verdict"]["case_mesh"] == "PASS"
                and gated["verdict"]["economy_mesh"] == "FAIL"
                and _gated_case_band_max(gated) < GATE
            ),
        },
        "not_run": not_run,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", default=None, help="cluster job id (informational)")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    doc = adjudicate(args.job)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "adjudicate.json"
    out.write_text(json.dumps(doc, indent=2, default=float) + "\n")
    print(f"  {out}")
    for label, block in doc["tiers"].items():
        print(
            f"  {label}: panels={block['n_panels']} P1={block['R_P1']['holds']} "
            f"P2={block['R_P2']['holds']} P3={block['R_P3']['holds']} "
            f"max_flat={block['measured_band_max']['flat']['band_max_rel_error']:.6f} "
            f"max_patch={block['measured_band_max']['patch']['band_max_rel_error']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())