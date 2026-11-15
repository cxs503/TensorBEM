#!/usr/bin/env python3
"""Apply the 1280-panel ka=4,5 computations to the case-pool artefact.

Takes the JSON produced by ``scripts/havelock_ka45_case_mesh_compute.py`` and

* replaces the two **GCI-scaled estimate** rows (``ka = 4.0``, ``5.0``) of
  ``docs/case_bem_submerged_sphere_havelock/numbers.json`` by the computed rows,
* keeps the superseded estimates under ``superseded_gci_estimate`` on the same
  row (audit trail: nothing is silently overwritten),
* tags every row with an explicit ``provenance`` marker
  (``computed_case_mesh_1280`` / ``computed_case_mesh_1280_control``) so that
  "estimated" rows can never masquerade as "computed" again,
* re-derives the tier gates, the refinement gate and the overall verdict from
  the updated table with the *pre-declared* 5 % criterion (unchanged), and
* appends explicit ``degenerate_items`` metadata: at ``ka >= 4`` the damping
  coefficients are O(1e-6), so a 5 % *relative* gate there is carried by
  discretisation noise.  This block is descriptive only -- it never relaxes or
  flips the declared criterion.

Refuses to write if the compute artefact does not show a bit-identical route
control (``ka = 3.0`` reproduced from the committed table) or if the geometry
precondition fails, i.e. the new numbers provably come from the same code path
as the rest of the table.

Usage
-----
    PYTHONPATH=src python3 scripts/havelock_ka45_apply.py \
        --computed docs/case_bem_submerged_sphere_havelock/ka45_case_mesh_computed.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

GATE = 0.05
COEFFICIENTS = ("a33", "b33", "a11", "b11")
DOCS = Path("docs/case_bem_submerged_sphere_havelock")
#: |ref| below this is treated as numerically degenerate for a relative gate.
DEGENERATE_SCALE = 1.0e-4


def _items(rows: Sequence[dict[str, Any]], tier: str) -> list[dict[str, Any]]:
    items = []
    for row in rows:
        for key in COEFFICIENTS:
            items.append({
                "tier": tier,
                "item": f"ka={row['ka']:.3g}",
                "quantity": key,
                "n_panels": row["n_panels"],
                "got": float(row[f"{key}_bem"]),
                "ref": float(row[f"{key}_exact"]),
                "rel_error": float(row[f"{key}_rel_error"]),
            })
    return items


def _tier_verdict(items: Sequence[dict[str, Any]], tier: str, gate: float) -> dict[str, Any]:
    failures = [it for it in items if not it["rel_error"] < gate]
    worst = max(items, key=lambda it: it["rel_error"])
    return {
        "tier": tier,
        "gate": gate,
        "n_items": len(items),
        "n_failures": len(failures),
        "max_rel_error": worst["rel_error"],
        "worst_item": {k: worst[k] for k in ("item", "quantity", "got", "ref", "rel_error")},
        "failures": [{k: it[k] for k in ("item", "quantity", "got", "ref", "rel_error")}
                     for it in sorted(failures, key=lambda it: -it["rel_error"])],
        "verdict": "PASS" if not failures else "FAIL",
    }


def _degenerate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Items whose reference magnitude makes the relative gate non-informative."""
    scale = max(abs(float(r[f"{k}_exact"])) for r in rows for k in COEFFICIENTS)
    flagged = []
    for row in rows:
        for key in COEFFICIENTS:
            ref = abs(float(row[f"{key}_exact"]))
            if ref < DEGENERATE_SCALE:
                got = float(row[f"{key}_bem"])
                flagged.append({
                    "item": f"ka={row['ka']:.3g}",
                    "quantity": key,
                    "ref": ref,
                    "got": got,
                    "rel_error": float(row[f"{key}_rel_error"]),
                    "scale_normalised_error": abs(got - ref) / scale,
                })
    return {
        "rule": f"|ref| < {DEGENERATE_SCALE:g} -> relative criterion is noise-dominated",
        "reference_scale": scale,
        "items": flagged,
        "note": ("descriptive only: the declared 5% relative criterion is applied "
                 "unchanged to every item; these rows carry the supplementary "
                 "scale-normalised error so the PASS is not over-read"),
    }


def _honest_note(case_tier: dict[str, Any], economy_tier: dict[str, Any],
                 n_degenerate: int) -> str:
    return (
        "Full band (ka=0.1-5.0, 7 wavenumbers) gated on the case mesh (1280 panels). "
        "All 28 items are now *computed* on that mesh: the previously GCI-scaled "
        "ka=4,5 rows were replaced by direct BEM solves "
        f"(case-mesh max relative error {case_tier['max_rel_error']*100:.3f}%, "
        f"{case_tier['n_failures']}/{case_tier['n_items']} failures). "
        f"Economy mesh (320 panels) FAILs {economy_tier['n_failures']}/{economy_tier['n_items']} "
        f"damping items (max {economy_tier['max_rel_error']*100:.2f}%) -- honest FAIL retained. "
        f"{n_degenerate} items at ka>=4 have |ref|<1e-4 (B' ~ 1e-6..1e-5): their 5% relative "
        "gate is noise-dominated, see degenerate_items."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--computed", type=Path, required=True)
    parser.add_argument("--numbers", type=Path, default=DOCS / "numbers.json")
    parser.add_argument("--verdict", type=Path, default=DOCS / "verdict.json")
    parser.add_argument("--tiers", type=Path, default=DOCS / "tiers.json")
    args = parser.parse_args(argv)

    computed = json.loads(args.computed.read_text())
    if not computed["route_control"]["route_identical"]:
        print("REFUSED: route control (ka=3.0) does not reproduce the committed row; "
              f"max rel diff {computed['route_control']['max_rel_diff']:.3e}", file=sys.stderr)
        return 2
    if computed["geometry_audit"]["verdict"] != "PASS":
        print("REFUSED: geometry audit failed", file=sys.stderr)
        return 2

    numbers = json.loads(args.numbers.read_text())
    table = numbers["band_sweep_case"]
    by_ka = {float(r["ka"]): i for i, r in enumerate(table)}

    replaced = []
    for row in computed["results"]:
        ka = float(row["ka"])
        idx = by_ka[ka]
        new_row = dict(row)
        new_row["provenance"] = "computed_case_mesh_1280"
        new_row["compute_job"] = computed.get("compute_job")
        old = table[idx]
        if old.get("provenance") != "computed_case_mesh_1280":
            new_row["superseded_gci_estimate"] = {
                key: {
                    "bem": float(old[f"{key}_bem"]),
                    "rel_error": float(old[f"{key}_rel_error"]),
                }
                for key in COEFFICIENTS
            } | {"condition_number": old.get("condition_number"),
                 "note": "GCI-scaled from the 320-panel economy mesh, r^p with r=2, p=1.91"}
        table[idx] = new_row
        replaced.append(ka)

    for row in table:
        row.setdefault("provenance", "computed_case_mesh_1280"
                       if float(row["ka"]) != float(computed["route_control"]["ka"])
                       else "computed_case_mesh_1280_control")

    numbers["band_sweep_case"] = table

    case_tier = _tier_verdict(_items(table, "case_mesh"), "case_mesh", GATE)
    economy_tier = _tier_verdict(_items(numbers["band_sweep_economy"], "economy_mesh"),
                                 "economy_mesh", GATE)
    numbers["tiers"]["case_mesh"] = case_tier
    numbers["tiers"]["economy_mesh"] = economy_tier

    refinement = {
        "economy_max_rel_error": economy_tier["max_rel_error"],
        "case_max_rel_error": case_tier["max_rel_error"],
        "observed_order_a33": numbers["convergence_ladder"].get("observed_order_a33"),
    }
    refinement["verdict"] = (
        "PASS" if refinement["case_max_rel_error"] <= refinement["economy_max_rel_error"]
        and (refinement["observed_order_a33"] or 0.0) > 0.0 else "FAIL")
    numbers["refinement"] = refinement

    degenerate = _degenerate(table)
    numbers["degenerate_items"] = degenerate
    numbers["ka45_computation"] = {
        "source": "scripts/havelock_ka45_case_mesh_compute.py",
        "artefact": str(args.computed),
        "compute_job": computed.get("compute_job"),
        "provenance": computed["provenance"],
        "elapsed_seconds": computed["elapsed_seconds"],
        "route_control": computed["route_control"],
        "geometry_audit": computed["geometry_audit"],
        "estimate_vs_computed": computed["estimate_vs_computed"],
        "replaced_ka": replaced,
    }

    numbers["verdict"] = {
        "case_mesh": case_tier["verdict"],
        "economy_mesh": economy_tier["verdict"],
        "geometry": numbers["geometry_precondition"]["verdict"],
        "refinement": refinement["verdict"],
        "overall": case_tier["verdict"],
        "honest_note": _honest_note(case_tier, economy_tier, len(degenerate["items"])),
    }

    args.numbers.write_text(json.dumps(numbers, indent=2) + "\n")
    args.verdict.write_text(json.dumps(numbers["verdict"], indent=2) + "\n")
    args.tiers.write_text(json.dumps({
        "tiers": numbers["tiers"],
        "refinement": numbers["refinement"],
        "geometry_precondition": numbers["geometry_precondition"],
        "degenerate_items": degenerate,
        "reference_gates": numbers.get("reference_integrity", {}).get("gates", {}),
    }, indent=2) + "\n")

    print(f"replaced rows: {replaced}")
    print(f"case_mesh: {case_tier['verdict']} max {case_tier['max_rel_error']*100:.3f}% "
          f"({case_tier['n_failures']}/{case_tier['n_items']} failures)")
    print(f"economy_mesh: {economy_tier['verdict']} max {economy_tier['max_rel_error']*100:.3f}% "
          f"({economy_tier['n_failures']}/{economy_tier['n_items']} failures)")
    print(f"degenerate items: {len(degenerate['items'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())