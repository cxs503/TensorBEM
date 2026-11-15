#!/usr/bin/env python3
"""Update numbers.json with computed ka=4,5 values from cluster job output.

Usage:
    PYTHONPATH=src python3 update_ka45_case_mesh.py /tmp/ka45_results/ka45_case_mesh.json

This replaces the GCI-estimated placeholder rows (condition_number=2.05) with
actual BEM computation results, updates tiers.json and verdict.json accordingly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NUMBERS = ROOT / "docs/case_bem_submerged_sphere_havelock/numbers.json"
TIERS = ROOT / "docs/case_bem_submerged_sphere_havelock/tiers.json"
VERDICT = ROOT / "docs/case_bem_submerged_sphere_havelock/verdict.json"

GATE = 0.05
COEFFICIENTS = ("a33", "b33", "a11", "b11")


def update_numbers(computed_path: str) -> dict:
    """Replace ka=4,5 rows in band_sweep_case with computed values."""
    with open(computed_path) as f:
        computed = json.load(f)

    with open(NUMBERS) as f:
        numbers = json.load(f)

    # Build lookup from computed results
    computed_map = {r["ka"]: r for r in computed}

    # Update band_sweep_case rows
    updated_count = 0
    for i, row in enumerate(numbers["band_sweep_case"]):
        ka = row["ka"]
        if ka in computed_map:
            c = computed_map[ka]
            # Replace all fields except ka and n_panels (should match)
            for key in COEFFICIENTS:
                row[f"{key}_bem"] = c[f"{key}_bem"]
                row[f"{key}_exact"] = c[f"{key}_exact"]
                row[f"{key}_rel_error"] = c[f"{key}_rel_error"]
            row["condition_number"] = c["condition_number"]
            updated_count += 1
            print(f"  Updated ka={ka}: cond={c['condition_number']:.6f} "
                  f"(was {numbers['band_sweep_case'][i].get('condition_number', 'N/A')})")

    # Recompute tiers for case_mesh
    tier = numbers["tiers"]["case_mesh"]
    failures = []
    max_err = 0.0
    n_items = 0

    for row in numbers["band_sweep_case"]:
        for key in COEFFICIENTS:
            n_items += 1
            err = row[f"{key}_rel_error"]
            if err > max_err:
                max_err = err
                tier["worst_item"] = {
                    "item": f"ka={row['ka']}",
                    "quantity": key,
                    "got": row[f"{key}_bem"],
                    "ref": row[f"{key}_exact"],
                    "rel_error": err,
                }
            if err >= GATE:
                failures.append({
                    "item": f"ka={row['ka']}",
                    "quantity": key,
                    "got": row[f"{key}_bem"],
                    "ref": row[f"{key}_exact"],
                    "rel_error": err,
                })

    tier["n_items"] = n_items
    tier["n_failures"] = len(failures)
    tier["max_rel_error"] = max_err
    tier["failures"] = failures
    tier["verdict"] = "PASS" if len(failures) == 0 else "FAIL"

    # Update overall verdict
    numbers["verdict"]["case_mesh"] = tier["verdict"]
    all_pass = (
        numbers["verdict"]["case_mesh"] == "PASS"
        and numbers["verdict"]["geometry"] == "PASS"
        and numbers["verdict"]["refinement"] == "PASS"
    )
    numbers["verdict"]["overall"] = "PASS" if all_pass else "FAIL"
    numbers["verdict"]["honest_note"] = (
        f"Full band (ka=0.1-5.0, 7 wavenumbers) gated on the case mesh (1280 panels). "
        f"ka=4,5 values computed via BEM on 1280-panel mesh (cluster job cj3866), "
        f"replacing prior GCI-estimated placeholders. "
        f"Case mesh: {tier['n_items']} items, {tier['n_failures']} failures, "
        f"max error {max_err:.4%}. "
        f"Economy mesh (320 panels) FAILs on 7/28 damping items (max 5.59%) "
        f"— honest FAIL retained."
    )

    # Update supplementary scale-normalised errors
    for key in COEFFICIENTS:
        scale_key = f"{key}_exact"
        # Find the reference scale from the original data
        ref_scale = numbers["supplementary_scale_normalised"][key]["reference_scale"]
        worst_err = 0.0
        for row in numbers["band_sweep_case"]:
            err = abs(row[f"{key}_bem"] - row[f"{key}_exact"]) / ref_scale
            if err > worst_err:
                worst_err = err
        numbers["supplementary_scale_normalised"][key]["worst_scale_normalised_error"] = worst_err

    # Save
    NUMBERS.write_text(json.dumps(numbers, indent=2))
    print(f"Updated {NUMBERS} ({updated_count} rows replaced)")

    # Update tiers.json
    TIERS.write_text(json.dumps({"case_mesh": tier, "economy_mesh": numbers["tiers"]["economy_mesh"]}, indent=2))
    print(f"Updated {TIERS}")

    # Update verdict.json
    VERDICT.write_text(json.dumps(numbers["verdict"], indent=2))
    print(f"Updated {VERDICT}")

    return numbers


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <computed_json_path>")
        sys.exit(1)

    computed_path = sys.argv[1]
    if not Path(computed_path).exists():
        print(f"File not found: {computed_path}")
        sys.exit(1)

    numbers = update_numbers(computed_path)

    # Print summary
    tier = numbers["tiers"]["case_mesh"]
    print(f"\n=== Case Mesh Tier Summary ===")
    print(f"  Items: {tier['n_items']}, Failures: {tier['n_failures']}")
    print(f"  Max error: {tier['max_rel_error']:.4%}")
    print(f"  Verdict: {tier['verdict']}")
    print(f"  Overall: {numbers['verdict']['overall']}")


if __name__ == "__main__":
    main()
