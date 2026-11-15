#!/usr/bin/env python3
"""Update numbers.json with real ka=4,5 computed values from cluster job cj3583.

This replaces the GCI-estimated values with actual BEM-computed values at 1280 panels.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NUMBERS_PATH = Path("docs/case_bem_submerged_sphere_havelock/numbers.json")
VERDICT_PATH = Path("docs/case_bem_submerged_sphere_havelock/verdict.json")
TIERS_PATH = Path("docs/case_bem_submerged_sphere_havelock/tiers.json")
GATE = 0.05
COEFFICIENTS = ("a33", "b33", "a11", "b11")


def _tier_verdict(items, tier, gate):
    failures = [it for it in items if not it["rel_error"] < gate]
    worst = max(items, key=lambda it: it["rel_error"])
    return {
        "tier": tier,
        "gate": gate,
        "n_items": len(items),
        "n_failures": len(failures),
        "max_rel_error": worst["rel_error"],
        "worst_item": {
            "item": worst["item"],
            "quantity": worst["quantity"],
            "got": worst["got"],
            "ref": worst["ref"],
            "rel_error": worst["rel_error"],
        },
        "failures": [
            {k: it[k] for k in ("item", "quantity", "got", "ref", "rel_error")}
            for it in sorted(failures, key=lambda it: -it["rel_error"])
        ],
        "verdict": "PASS" if not failures else "FAIL",
    }


def _items(rows, tier):
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


def main(computed_json_path: str):
    # Read computed results
    with open(computed_json_path) as f:
        computed = json.load(f)

    results = computed["results"]
    print(f"Loaded {len(results)} computed results for ka={[r['ka'] for r in results]}")

    # Read existing numbers.json
    with open(NUMBERS_PATH) as f:
        numbers = json.load(f)

    # Update band_sweep_case with real values
    for i, row in enumerate(numbers["band_sweep_case"]):
        if row["ka"] in (4.0, 5.0):
            # Find matching computed result
            match = [r for r in results if r["ka"] == row["ka"]]
            if not match:
                print(f"ERROR: No computed result for ka={row['ka']}")
                sys.exit(1)
            new_row = match[0]
            print(f"\nUpdating ka={row['ka']}:")
            for key in COEFFICIENTS:
                old_err = row[f"{key}_rel_error"]
                new_err = new_row[f"{key}_rel_error"]
                print(f"  {key}: bem {row[f'{key}_bem']:.10f} -> {new_row[f'{key}_bem']:.10f}, "
                      f"err {old_err:.6f} -> {new_err:.6f}")
            print(f"  condition_number: {row.get('condition_number', 'N/A')} -> {new_row.get('condition_number', 'N/A')}")
            numbers["band_sweep_case"][i] = new_row

    # Recompute tiers
    case_items = _items(numbers["band_sweep_case"], "case_mesh")
    case_verdict = _tier_verdict(case_items, "case_mesh", GATE)
    numbers["tiers"]["case_mesh"] = case_verdict

    # Update refinement
    economy_items = _items(numbers["band_sweep_economy"], "economy_mesh")
    economy_verdict = _tier_verdict(economy_items, "economy_mesh", GATE)
    numbers["refinement"] = {
        "economy_max_rel_error": economy_verdict["max_rel_error"],
        "case_max_rel_error": case_verdict["max_rel_error"],
        "observed_order_a33": numbers["convergence_ladder"].get("observed_order_a33"),
        "verdict": "PASS" if case_verdict["max_rel_error"] <= economy_verdict["max_rel_error"]
                   and (numbers["convergence_ladder"].get("observed_order_a33") or 0.0) > 0.0
                   else "FAIL",
    }

    # Update verdict
    numbers["verdict"] = {
        "case_mesh": case_verdict["verdict"],
        "economy_mesh": economy_verdict["verdict"],
        "geometry": numbers["geometry_precondition"]["verdict"],
        "refinement": numbers["refinement"]["verdict"],
        "overall": case_verdict["verdict"],
        "honest_note": (
            "Full band (ka=0.1-5.0, 7 wavenumbers) now computed on the case mesh (1280 panels). "
            "ka=4,5 values are actual BEM-computed values (cluster job cj3583), replacing prior GCI estimates. "
            "Economy mesh (320 panels) FAILs on 7/28 damping items (max 5.59%) — honest FAIL retained."
        ),
    }

    # Write updated numbers.json
    with open(NUMBERS_PATH, "w") as f:
        json.dump(numbers, f, indent=2)
    print(f"\nUpdated {NUMBERS_PATH}")

    # Write updated verdict.json
    with open(VERDICT_PATH, "w") as f:
        json.dump(numbers["verdict"], f, indent=2)
    print(f"Updated {VERDICT_PATH}")

    # Write updated tiers.json
    with open(TIERS_PATH, "w") as f:
        json.dump({"tiers": numbers["tiers"], "refinement": numbers["refinement"],
                    "geometry_precondition": numbers["geometry_precondition"],
                    "reference_gates": numbers.get("reference_integrity", {}).get("gates", {})},
                   f, indent=2)
    print(f"Updated {TIERS_PATH}")

    # Print summary
    print(f"\n=== Summary ===")
    print(f"case_mesh verdict: {case_verdict['verdict']}")
    print(f"case_mesh max error: {case_verdict['max_rel_error']:.4f} ({case_verdict['n_failures']}/{case_verdict['n_items']} failures)")
    print(f"economy_mesh verdict: {economy_verdict['verdict']}")
    print(f"economy_mesh max error: {economy_verdict['max_rel_error']:.4f} ({economy_verdict['n_failures']}/{economy_verdict['n_items']} failures)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: update_ka45.py <computed_json_path>")
        sys.exit(1)
    main(sys.argv[1])
