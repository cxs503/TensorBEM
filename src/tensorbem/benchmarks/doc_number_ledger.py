"""Recompute the MacCamy-Fuchs declared numbers in docs/marine_bem_benchmarks.md.

Tables 8 and 9 of ``docs/marine_bem_benchmarks.md`` (Sections 4.9) quote a
convergence ladder for the vertical-cylinder diffraction force, a frequency
sweep of the effective inertia coefficient C_M(ka), and a study of the
irregular-frequency suppression by the Schenck CHIEF augmentation.  None of
those numbers was guarded by a test, so a change in the Helmholtz assembly or
in the Bessel/Hankel layer could silently invalidate the documentation.

This module re-runs the benchmark CLI and compares every declared number with
the recomputed value, emitting a strict JSON ledger and PASS/DRIFT verdicts.

Usage::

    PYTHONPATH=src python3 artifacts/doc_numbers/verify_doc_numbers.py \
        --output artifacts/doc_numbers/doc_numbers_check.json

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

#: Numbers transcribed from docs/marine_bem_benchmarks.md Tables 8 and 9.
#: (table, quantity, panel count / ka, declared value, relative tolerance)
DECLARED = [
    # --- Table 8: refinement at ka = 1.0, h/a = 3.0, rho = 1025 -----------
    ("table8", "condition_number", 16, 3.015, 1.0e-3),
    ("table8", "force_x_relative_error", 16, 8.71e-3, 5.0e-3),
    ("table8", "surface_potential_relative_l2_error", 16, 1.19e-2, 5.0e-3),
    ("table8", "condition_number", 32, 2.665, 1.0e-3),
    ("table8", "force_x_relative_error", 32, 2.14e-3, 5.0e-3),
    ("table8", "surface_potential_relative_l2_error", 32, 2.94e-3, 5.0e-3),
    ("table8", "condition_number", 64, 2.509, 1.0e-3),
    ("table8", "force_x_relative_error", 64, 5.31e-4, 5.0e-3),
    ("table8", "surface_potential_relative_l2_error", 64, 7.29e-4, 5.0e-3),
    ("table8", "condition_number", 128, 2.434, 1.0e-3),
    ("table8", "force_x_relative_error", 128, 1.32e-4, 5.0e-3),
    ("table8", "surface_potential_relative_l2_error", 128, 1.82e-4, 5.0e-3),
    # moment_y equals force_x by construction for a circular section
    ("table8", "moment_y_relative_error", 128, 1.32e-4, 5.0e-3),
    # documented observed order of the force error
    ("table8", "order_force_x_error", 32, 2.040, 1.0e-2),
    ("table8", "order_force_x_error", 64, 2.016, 1.0e-2),
    ("table8", "order_force_x_error", 128, 2.007, 1.0e-2),
    # --- Table 8: frequency sweep of the effective inertia coefficient ----
    ("sweep", "computed_effective_cm", 0.1, 2.0184, 1.0e-4),
    ("sweep", "exact_effective_cm", 0.1, 2.0192, 1.0e-4),
    ("sweep", "computed_effective_cm", 3.0, 0.3078, 1.0e-3),
    ("sweep", "exact_effective_cm", 3.0, 0.3077, 1.0e-3),
    # --- Table 9: irregular frequencies, N = 64 --------------------------
    ("irregular", "unaugmented_condition_number", "j01", 1002.5, 1.0e-3),
    ("irregular", "chief_condition_number", "j01", 2.31, 5.0e-3),
    ("irregular", "unaugmented_force_relative_error", "j01", 2.03e-3, 1.0e-2),
    ("irregular", "chief_force_relative_error", "j01", 2.02e-3, 1.0e-2),
    ("irregular", "unaugmented_condition_number", "j11", 665.0, 1.0e-3),
    ("irregular", "chief_condition_number", "j11", 3.34, 5.0e-3),
    ("irregular", "unaugmented_force_relative_error", "j11", 1.035e-1, 1.0e-2),
    ("irregular", "chief_force_relative_error", "j11", 2.81e-3, 1.0e-2),
]


def run_benchmark(panels: str = "16 32 64 128") -> dict:
    """Run the MacCamy-Fuchs benchmark CLI in a fresh subprocess."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "1")
    completed = subprocess.run(
        [sys.executable, "-m", "tensorbem.benchmarks.maccamy_fuchs", "--panels", *panels.split()],
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
        ratio = coarse[key] / fine[key]
        orders[fine["panels"]] = math.log(ratio) / math.log(coarse["h_max"] / fine["h_max"])
    return orders


def _lookup(payload: dict, table: str, quantity: str, key) -> float:
    if table == "table8":
        levels = {row["panels"]: row for row in payload["refinement"]["levels"]}
        if quantity == "order_force_x_error":
            return _observed_orders(payload["refinement"]["levels"], "force_x_relative_error")[key]
        return float(levels[key][quantity])
    if table == "sweep":
        rows = {round(row["ka"], 6): row for row in payload["frequency_sweep"]}
        return float(rows[round(float(key), 6)][quantity])
    if table == "irregular":
        labels = {
            "j01": "j01_axisymmetric_resonance",
            "j11": "j11_dipole_resonance",
        }
        rows = {row["resonance"]: row for row in payload["irregular_frequency_study"]}
        return float(rows[labels[key]][quantity])
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
        "benchmark": "maccamy_fuchs_declared_numbers",
        "source": "docs/marine_bem_benchmarks.md Tables 8 and 9 (Section 4.9)",
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run_benchmark(args.panels)
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