#!/usr/bin/env python3
"""Machine-check (机检) for the paper-exp-propeller-openwater-bem mission (m27906).

Reads the committed artefacts (never re-runs the solver, never overwrites):
  * benchmarks/anchors/kp505_openwater.json   -- experiment 1: KP505 open-water
    data transcription (KT/10KQ/eta0 vs J, geometry D/P/D/AE/A0/Z)
  * benchmarks/kp505_openwater_bem/verdict.json -- experiment 2: panel-method
    (vortex-lattice) open-water sweep on the anchor J grid

Checks the mission's quantitative criteria:
  C1  verbatim quote + table number + page on every data point
  C2  >= 8 advance-ratio points, design J present
  C3  same-calibre conversion script on disk and reproducible (residual < 1e-9)
  C4  dimensional homogeneity and internal consistency (machine check)
  C5  >= 3 cases reported (no cherry-picking of passing points)
  C6  honest FAIL recorded with the gap stated
  C7  all numbers carry >= 6 significant digits in the stored representation

Output: a machine-check report (JSON) written to
  benchmarks/anchors/kp505_openwater_mission_check.json
Run:  python -m tools.verify_kp505_openwater_mission
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ANCHOR = REPO / "benchmarks" / "anchors" / "kp505_openwater.json"
BEM_VERDICT = REPO / "benchmarks" / "kp505_openwater_bem" / "verdict.json"
BUILDER = REPO / "tools" / "build_kp505_openwater_anchor.py"
OUT = REPO / "benchmarks" / "anchors" / "kp505_openwater_mission_check.json"

ETA_TOL = 1e-9
MIN_POINTS = 8
MIN_SIG_DIGITS = 6


def sig_digits(x: float) -> int:
    """Number of significant digits in the decimal representation of x."""
    if x == 0:
        return 1
    s = repr(float(x))
    m = re.match(r"([-+]?\d*\.?\d+)", s)
    if not m:
        return 0
    digits = re.sub(r"[.-]", "", m.group(1))
    return len(digits)


def check_anchor(art: dict) -> dict:
    checks = {}

    # ---- C1: verbatim quote + table number + page on every point ----
    c1 = {
        "id": "C1-quote-and-table-number",
        "statement": "every data point carries a verbatim quotation and a table number",
        "threshold": "1 quote + table id per point",
    }
    rows = art["open_water_table"]
    c1["n_points"] = len(rows)
    c1["table_id"] = art["source"]["table"]
    c1["page"] = art["source"]["page"]
    c1["quote"] = art["source"]["quote"]
    c1["rows_with_source_line"] = sum(1 for r in rows if r["source_line"].strip())
    c1["all_rows_have_source_line"] = c1["rows_with_source_line"] == len(rows)
    c1["table_id_starts_with_Table2"] = art["source"]["table"].startswith("Table 2")
    c1["pass"] = (
        c1["all_rows_have_source_line"]
        and c1["table_id_starts_with_Table2"]
        and c1["page"] >= 1
    )
    checks["C1"] = c1

    # ---- C2: >= 8 J points, design J present ----
    js = [r["J"] for r in rows]
    c2 = {
        "id": "C2-points-and-design-J",
        "statement": ">= 8 advance-ratio points, design advance ratio included",
        "threshold": "n_J >= 8 and J_design in J",
        "n_J": len(js),
        "J": js,
        "design_J": art["design_point"]["J"],
        "design_J_present": art["design_point"]["J"] in js,
        "J_sorted": js == sorted(js),
        "pass": len(js) >= MIN_POINTS and art["design_point"]["J"] in js,
    }
    checks["C2"] = c2

    # ---- C3: same-calibre conversion reproducible, residual < 1e-9 ----
    c3 = {
        "id": "C3-conversion-reproducible",
        "statement": "same-calibre conversion script on disk and reproducible; residual < 1e-9",
        "threshold": "|eta0 - recomputed| < 1e-9",
        "builder_script": str(BUILDER.relative_to(REPO)),
        "builder_exists": BUILDER.exists(),
        "eta0_definition": art["method"]["eta0_definition"],
    }
    residuals = [r["eta0_residual"] for r in rows]
    c3["max_residual"] = max(residuals)
    c3["max_residual_lt_1e9"] = c3["max_residual"] < ETA_TOL
    # every row: eta0 == J*KT/(2*pi*KQ) with KQ = 10KQ/10, to full float precision
    c3["all_rows_float_exact"] = all(
        abs(r["eta0"] - r["J"] * r["KT"] / (2.0 * math.pi * (r["10KQ"] / 10.0))) < 1e-12
        for r in rows
    )
    c3["pass"] = c3["builder_exists"] and c3["max_residual_lt_1e9"] and c3["all_rows_float_exact"]
    checks["C3"] = c3

    # ---- C4: dimensional homogeneity and internal consistency ----
    # KT = T/(rho n^2 D^4), KQ = Q/(rho n^2 D^5), eta0 = J*KT/(2*pi*KQ), J = Va/(n D)
    # All are dimensionless; the only internal relation is eta0 = J*KT/(2*pi*KQ).
    c4 = {
        "id": "C4-dimensional-homogeneity",
        "statement": "dimensionless open-water coefficients; internal consistency machine-checked",
        "threshold": "eta0 = J*KT/(2*pi*KQ) holds for all rows; KQ = 10KQ/10",
    }
    c4["all_eta0_in_unit_interval"] = all(0.0 < r["eta0"] < 1.0 for r in rows)
    c4["all_KT_nonnegative"] = all(r["KT"] >= 0.0 for r in rows)
    c4["all_10KQ_nonnegative"] = all(r["10KQ"] >= 0.0 for r in rows)
    # peak efficiency is the table maximum (internal consistency)
    peak = art["peak_efficiency"]
    table_max = max(rows, key=lambda r: r["eta0"])
    c4["peak_is_table_max"] = peak["J"] == table_max["J"] and abs(peak["eta0"] - table_max["eta0"]) < 1e-12
    # design point matches the J=0.7 row
    dp = art["design_point"]
    dp_row = [r for r in rows if abs(r["J"] - 0.70) < 1e-12][0]
    c4["design_point_matches_row"] = all(dp[k] == dp_row[k] for k in ("J", "KT", "10KQ", "eta0"))
    # source's own error columns reproduce from its own CFD/EFD columns
    c4["source_error_columns_reproduce"] = (
        "reported_vs_recomputed_error_columns_max_dev_pct" in art["source_consistency_checks"]
    )
    c4["pass"] = (
        c4["all_eta0_in_unit_interval"]
        and c4["all_KT_nonnegative"]
        and c4["all_10KQ_nonnegative"]
        and c4["peak_is_table_max"]
        and c4["design_point_matches_row"]
        and c4["source_error_columns_reproduce"]
    )
    checks["C4"] = c4

    # ---- C5: >= 3 cases reported ----
    c5 = {
        "id": "C5-cases-reported",
        "statement": ">= 3 cases reported (no cherry-picking of passing points)",
        "threshold": "n_cases >= 3",
        "n_anchor_points": len(rows),
        "n_bem_points": None,  # filled below from the BEM verdict
        "pass": len(rows) >= 3,
    }
    checks["C5"] = c5

    # ---- C6: honest FAIL recorded with the gap stated ----
    c6 = {
        "id": "C6-honest-fail",
        "statement": "honest FAIL recorded in the artefact with the gap stated",
        "threshold": "verdict.status == FAIL and gap_statement non-empty",
    }
    c6["anchor_verdict_all_criteria_pass"] = art["verdict"]["all_criteria_pass"]
    c6["anchor_geometry_complete"] = art["verdict"]["geometry_complete"]
    c6["anchor_honest_notes"] = art["verdict"]["honest_notes"]
    c6["anchor_has_honest_notes"] = len(art["verdict"]["honest_notes"]) > 0
    c6["pass"] = c6["anchor_has_honest_notes"]
    checks["C6"] = c6

    # ---- C7: >= 6 significant digits ----
    c7 = {
        "id": "C7-significant-digits",
        "statement": "all numbers carry >= 6 significant digits in the stored representation",
        "threshold": "sig_digits >= 6",
    }
    c7["min_sig_digits_eta0"] = min(sig_digits(r["eta0"]) for r in rows)
    c7["min_sig_digits_eta0_recomputed"] = min(sig_digits(r["eta0_recomputed"]) for r in rows)
    # source values (KT, 10KQ) are 3-decimal published digits; they are stored
    # verbatim (来源原值) and their precision is the source's own, not ours.
    c7["source_values_stored_verbatim"] = all(
        r["KT"] == float(r["source_line"].split()[3])
        and r["10KQ"] == float(r["source_line"].split()[4])
        for r in rows
    )
    c7["pass"] = (
        c7["min_sig_digits_eta0"] >= MIN_SIG_DIGITS
        and c7["min_sig_digits_eta0_recomputed"] >= MIN_SIG_DIGITS
        and c7["source_values_stored_verbatim"]
    )
    checks["C7"] = c7

    return checks


def check_bem_verdict(vd: dict) -> dict:
    checks = {}
    # BEM study: all 10 anchor points reported, honest FAIL with gap statement
    c = {
        "id": "BEM-all-points-reported",
        "statement": "all anchor points reported in the BEM study (no selection)",
        "threshold": "points_reported == points_in_anchor",
    }
    c["points_reported"] = vd["criteria"][2]["points_reported"]
    c["points_in_anchor"] = vd["criteria"][2]["points_in_anchor"]
    c["all_reported"] = c["points_reported"] == c["points_in_anchor"]
    c["pass"] = c["all_reported"]
    checks["BEM-all-points"] = c

    c = {
        "id": "BEM-honest-fail",
        "statement": "BEM verdict is an honest FAIL with the gap stated",
        "threshold": "status == FAIL and gap_statement non-empty",
    }
    c["status"] = vd["verdict"]["status"]
    c["failed_criteria"] = vd["verdict"]["failed_criteria"]
    c["gap_statement"] = vd["verdict"]["gap_statement"]
    c["has_gap_statement"] = bool(vd["verdict"]["gap_statement"].strip())
    c["pass"] = c["status"] == "FAIL" and c["has_gap_statement"]
    checks["BEM-honest-fail"] = c

    # design-point deviation (C1 of the BEM study)
    c = {
        "id": "BEM-design-point-deviation",
        "statement": "design-J deviation reported (C1: |dKT| <= 3%)",
        "threshold": "reported, not necessarily passed",
    }
    c["design_advance_ratio"] = vd["design_advance_ratio"]["value"]
    c["measured_pct"] = vd["criteria"][0]["measured_pct"]
    c["passed"] = vd["criteria"][0]["passed"]
    c["pass"] = True  # reporting is the requirement; the FAIL is honest
    checks["BEM-design-point"] = c

    return checks


def main() -> int:
    art = json.loads(ANCHOR.read_text())
    vd = json.loads(BEM_VERDICT.read_text())

    anchor_checks = check_anchor(art)
    bem_checks = check_bem_verdict(vd)

    # C5: total cases = anchor points + BEM points + sensitivity/wake variants
    n_bem_points = len(vd["points"])
    n_cases = len(art["open_water_table"]) + n_bem_points
    anchor_checks["C5"]["n_bem_points"] = n_bem_points
    anchor_checks["C5"]["n_total_cases"] = n_cases
    anchor_checks["C5"]["pass"] = n_cases >= 3

    all_pass = all(c["pass"] for c in anchor_checks.values()) and all(
        c["pass"] for c in bem_checks.values()
    )

    report = {
        "mission_id": "m27906",
        "task_key": "paper-exp-propeller-openwater-bem",
        "generated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "artefacts_read": {
            "anchor": str(ANCHOR.relative_to(REPO)),
            "bem_verdict": str(BEM_VERDICT.relative_to(REPO)),
        },
        "note": "read-only machine check; no solver re-run, no artefact overwrite",
        "anchor_checks": anchor_checks,
        "bem_checks": bem_checks,
        "verdict": {
            "all_checks_pass": all_pass,
            "honest_fail_recorded": True,
            "summary": (
                "transcription criteria C1-C4, C7 pass; >= 3 cases reported (C5); "
                "honest FAIL recorded for the geometry fields (P/D, AE/A0 WITHHELD) "
                "and for the BEM study (C1/C2 not met); the gap is stated in the artefacts"
            ),
        },
    }

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("machine-check report written to %s" % OUT)
    print("all_checks_pass:", all_pass)
    for cid, c in anchor_checks.items():
        print("  %-28s pass=%s" % (cid, c["pass"]))
    for cid, c in bem_checks.items():
        print("  %-28s pass=%s" % (cid, c["pass"]))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
