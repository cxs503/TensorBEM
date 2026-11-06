"""Recompute the rigid-body Neumann declared numbers in docs/marine_bem_benchmarks.md.

Tables 5 and 6 of ``docs/marine_bem_benchmarks.md`` (Section 4.7) quote
convergence ladders for the 2-D elliptic-cylinder and 3-D triaxial-ellipsoid
rigid-body Neumann added-mass tensors.  None of those numbers was guarded by
a test, so a change in the second-kind integral equation assembly could
silently invalidate the documentation.

This module re-runs the benchmark CLI and compares every declared number with
the recomputed value, emitting a strict JSON ledger and PASS/DRIFT verdicts.

Usage::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.rigid_body_doc_ledger

Exit status is 0 only when every declared number reproduces.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

#: Numbers transcribed from docs/marine_bem_benchmarks.md Tables 5 and 6.
#: (table, quantity, panel count / subdivision, declared value, relative tolerance)
DECLARED = [
    # --- Table 5: 2-D ellipse (a=1.5, b=0.75, rho=1025) -------------------
    ("table5", "condition_number", 16, 3.078, 1.0e-3),
    ("table5", "surge_x_relative_error", 16, 2.64e-3, 5.0e-3),
    ("table5", "heave_y_relative_error", 16, 5.15e-3, 5.0e-3),
    ("table5", "roll_xy_relative_error", 16, 3.23e-2, 5.0e-3),
    ("table5", "condition_number", 32, 3.019, 1.0e-3),
    ("table5", "surge_x_relative_error", 32, 7.42e-4, 5.0e-3),
    ("table5", "heave_y_relative_error", 32, 1.41e-3, 5.0e-3),
    ("table5", "roll_xy_relative_error", 32, 8.71e-3, 5.0e-3),
    ("table5", "condition_number", 64, 3.005, 1.0e-3),
    ("table5", "surge_x_relative_error", 64, 1.94e-4, 5.0e-3),
    ("table5", "heave_y_relative_error", 64, 3.75e-4, 5.0e-3),
    ("table5", "roll_xy_relative_error", 64, 2.25e-3, 5.0e-3),
    ("table5", "condition_number", 128, 3.001, 1.0e-3),
    ("table5", "surge_x_relative_error", 128, 4.94e-5, 5.0e-3),
    ("table5", "heave_y_relative_error", 128, 9.70e-5, 5.0e-3),
    ("table5", "roll_xy_relative_error", 128, 5.70e-4, 5.0e-3),
    # documented observed order of roll_xy error
    ("table5", "roll_xy_order", 32, 1.936, 1.0e-2),
    ("table5", "roll_xy_order", 64, 1.966, 1.0e-2),
    ("table5", "roll_xy_order", 128, 1.982, 1.0e-2),
    # --- Table 6: 3-D ellipsoid (a1=1.5, a2=1.0, a3=0.75, rho=1025) ------
    ("table6", "condition_number", 20, 2.447, 1.0e-3),
    ("table6", "max_translational_error", 20, 2.63e-1, 5.0e-3),
    ("table6", "max_rotational_error", 20, 5.19e-1, 5.0e-3),
    ("table6", "condition_number", 80, 2.246, 1.0e-3),
    ("table6", "max_translational_error", 80, 8.11e-2, 5.0e-3),
    ("table6", "max_rotational_error", 80, 1.78e-1, 5.0e-3),
    ("table6", "rotational_order", 80, 2.012, 1.0e-2),
    ("table6", "condition_number", 320, 2.130, 1.0e-3),
    ("table6", "max_translational_error", 320, 2.14e-2, 5.0e-3),
    ("table6", "max_rotational_error", 320, 4.85e-2, 5.0e-3),
    ("table6", "rotational_order", 320, 2.028, 1.0e-2),
]


def run_benchmark(panels: str = "16 32 64 128", subdivisions: str = "0 1 2") -> dict:
    """Run the rigid-body Neumann benchmark CLI in a fresh subprocess."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "1")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tensorbem.benchmarks.rigid_body_added_mass",
            "--panels",
            *panels.split(),
            "--subdivisions",
            *subdivisions.split(),
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def _observed_orders(levels: list[dict], key: str) -> dict[int, float]:
    """Order of accuracy from successive halvings of h_max."""
    orders: dict[int, float] = {}
    for coarse, fine in zip(levels, levels[1:]):
        ratio = coarse["diagonal_relative_errors"][key] / fine["diagonal_relative_errors"][key]
        orders[fine["panels"]] = math.log(ratio) / math.log(coarse["h_max"] / fine["h_max"])
    return orders


def _lookup(payload: dict, table: str, quantity: str, key) -> float:
    if table == "table5":
        levels = {row["panels"]: row for row in payload["ellipse_2d"]["levels"]}
        if quantity == "roll_xy_order":
            return _observed_orders(payload["ellipse_2d"]["levels"], "roll_xy")[key]
        if quantity in ("surge_x_relative_error", "heave_y_relative_error", "roll_xy_relative_error"):
            # Map quantity names to diagonal_relative_errors keys
            mode_map = {
                "surge_x_relative_error": "surge_x",
                "heave_y_relative_error": "heave_y",
                "roll_xy_relative_error": "roll_xy",
            }
            return float(levels[key]["diagonal_relative_errors"][mode_map[quantity]])
        return float(levels[key][quantity])
    if table == "table6":
        levels = {row["panels"]: row for row in payload["ellipsoid_3d"]["levels"]}
        if quantity == "rotational_order":
            return _observed_orders(payload["ellipsoid_3d"]["levels"], "pitch_y")[key]
        if quantity == "max_translational_error":
            errors = levels[key]["diagonal_relative_errors"]
            return max(errors["surge_x"], errors["sway_y"], errors["heave_z"])
        if quantity == "max_rotational_error":
            errors = levels[key]["diagonal_relative_errors"]
            return max(errors["roll_x"], errors["pitch_y"], errors["yaw_z"])
        return float(levels[key][quantity])
    raise ValueError(f"unknown table {table!r}")


def check(payload: dict) -> dict:
    checks = []
    for table, quantity, key, declared, tol in DECLARED:
        measured = _lookup(payload, table, quantity, key)
        scale = abs(declared) if declared != 0.0 else 1.0
        deviation = abs(measured - declared) / scale
        checks.append(
            {
                "table": table,
                "quantity": quantity,
                "key": key,
                "declared": declared,
                "recomputed": measured,
                "relative_deviation": deviation,
                "tolerance": tol,
                "verdict": "PASS" if deviation <= tol else "DRIFT",
            }
        )
    drift = [c for c in checks if c["verdict"] != "PASS"]
    return {
        "benchmark": "rigid_body_neumann_declared_numbers",
        "source": "docs/marine_bem_benchmarks.md Tables 5 and 6 (Section 4.7)",
        "checks": checks,
        "summary": {
            "registered": len(checks),
            "passed": len(checks) - len(drift),
            "drift": len(drift),
            "max_relative_deviation": max(c["relative_deviation"] for c in checks),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", default="16 32 64 128")
    parser.add_argument("--subdivisions", default="0 1 2")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run_benchmark(args.panels, args.subdivisions)
    ledger = check(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for row in ledger["checks"]:
        print(
            "%-9s %-38s %-6s declared=%-12.6g recomputed=%-12.6g dev=%.2e %s"
            % (
                row["table"],
                row["quantity"],
                row["key"],
                row["declared"],
                row["recomputed"],
                row["relative_deviation"],
                row["verdict"],
            )
        )
    summary = ledger["summary"]
    print(
        "\n%d/%d PASS, %d DRIFT, max deviation %.2e"
        % (
            summary["passed"],
            summary["registered"],
            summary["drift"],
            summary["max_relative_deviation"],
        )
    )
    return 0 if summary["drift"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
