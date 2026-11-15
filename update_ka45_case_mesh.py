#!/usr/bin/env python3
"""Update ka=4,5 values in numbers.json with BEM-computed results from cluster job.

Usage:
    python3 update_ka45_case_mesh.py /tmp/ka45_case_mesh_result.json

The input JSON should be an array of two objects (ka=4.0 and ka=5.0) with keys:
    ka, n_panels, a33_bem, a33_exact, a33_rel_error,
    b33_bem, b33_exact, b33_rel_error,
    a11_bem, a11_exact, a11_rel_error,
    b11_bem, b11_exact, b11_rel_error,
    condition_number, relative_residual
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

NUMBERS_PATH = Path("docs/case_bem_submerged_sphere_havelock/numbers.json")
VERDICT_PATH = Path("docs/case_bem_submerged_sphere_havelock/verdict.json")
TIERS_PATH = Path("docs/case_bem_submerged_sphere_havelock/tiers.json")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <result_json>", file=sys.stderr)
        sys.exit(1)

    result_path = Path(sys.argv[1])
    assert result_path.is_file(), f"Result file not found: {result_path}"

    with open(result_path) as f:
        computed = json.load(f)

    # Ensure it's a list
    if isinstance(computed, dict):
        computed = [computed]

    # Build lookup by ka
    computed_by_ka = {row["ka"]: row for row in computed}
    assert 4.0 in computed_by_ka, "Missing ka=4.0 in results"
    assert 5.0 in computed_by_ka, "Missing ka=5.0 in results"

    # Load current numbers.json
    numbers = json.loads(NUMBERS_PATH.read_text())

    # Update band_sweep_case
    updated_count = 0
    for row in numbers["band_sweep_case"]:
        if row["ka"] in (4.0, 5.0):
            new = computed_by_ka[row["ka"]]
            for key in ("a33_bem", "a33_exact", "a33_rel_error",
                        "b33_bem", "b33_exact", "b33_rel_error",
                        "a11_bem", "a11_exact", "a11_rel_error",
                        "b11_bem", "b11_exact", "b11_rel_error",
                        "condition_number", "relative_residual"):
                if key in new:
                    row[key] = new[key]
            row["n_panels"] = 1280
            updated_count += 1
            print(f"Updated ka={row['ka']}: a33_err={row['a33_rel_error']:.4f}, "
                  f"b33_err={row['b33_rel_error']:.4f}, cond={row['condition_number']:.10f}")

    assert updated_count == 2, f"Expected 2 updates, got {updated_count}"

    # Recompute tiers for case_mesh
    coefficients = ("a33", "b33", "a11", "b11")
    gate = 0.05
    n_items = 0
    n_failures = 0
    max_err = 0.0
    for row in numbers["band_sweep_case"]:
        for key in coefficients:
            n_items += 1
            err = row[f"{key}_rel_error"]
            max_err = max(max_err, err)
            if err >= gate:
                n_failures += 1

    numbers["tiers"]["case_mesh"] = {
        "n_items": n_items,
        "n_failures": n_failures,
        "verdict": "PASS" if n_failures == 0 else "FAIL",
        "max_rel_error": max_err,
    }

    # Update verdict
    verdict = json.loads(VERDICT_PATH.read_text())
    verdict["case_mesh"] = "PASS" if n_failures == 0 else "FAIL"
    verdict["overall"] = "PASS" if verdict["case_mesh"] == "PASS" else "FAIL"
    verdict["honest_note"] = (
        f"Full band (ka=0.1-5.0, 7 wavenumbers) gated on case mesh (1280 panels). "
        f"ka=4,5 values now BEM-computed (cluster job cj3615). "
        f"Case mesh: {n_items} items, {n_failures} failures, max error {max_err:.4f}. "
        f"Economy mesh (320 panels) FAILs on damping items (max ~5.6%) — honest FAIL retained."
    )

    # Write back
    NUMBERS_PATH.write_text(json.dumps(numbers, indent=2) + "\n")
    VERDICT_PATH.write_text(json.dumps(verdict, indent=2) + "\n")

    print(f"\nUpdated {updated_count} rows in numbers.json")
    print(f"case_mesh tier: {n_items} items, {n_failures} failures, verdict={verdict['case_mesh']}")
    print(f"overall verdict: {verdict['overall']}")


if __name__ == "__main__":
    main()
