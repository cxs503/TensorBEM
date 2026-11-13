#!/usr/bin/env python3
"""Independent re-derivation of the m17245 (CASE-BEM-KP505-GCI) audit numbers.

Reads only the delivered artefacts (levels.csv / gci.csv / case.json snapshot),
recomputes Roache's GCI, the observed order and the anchor errors from scratch
(no import of tensorbem), and prints a machine-checkable JSON verdict.

Usage: python tools/verify_m17245_numbers.py <snapshot_dir> [--json out.json]
"""
import csv
import json
import os
import sys


def load_levels(path):
    rows = list(csv.DictReader(open(path)))
    lv = {}
    for r in rows:
        key = (float(r["J"]), r["level"])
        lv[key] = {q: float(r[q]) for q in ("KT", "10KQ", "eta0")}
    return lv


def load_gci(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out[(float(r["J"]), r["quantity"])] = r
    return out


def roache(f_coarse, f_medium, f_fine, r=2.0, fs=1.25):
    """f1 coarse, f2 medium, f3 fine (Roache/Celik, exactly as the case documents)."""
    d21 = f_medium - f_coarse
    d32 = f_fine - f_medium
    if d32 == 0 or d21 == 0:
        return None
    import math
    p = abs(math.log(abs(d21 / d32))) / math.log(r)
    q = math.log(abs(d21 / d32)) / math.log(r)  # signed order
    ext = f_fine + (f_fine - f_medium) / (r ** q - 1) if q > 0 else float("nan")
    gci_f = fs * abs((f_medium - f_fine) / f_fine) / abs(r ** q - 1)
    gci_m = fs * abs((f_coarse - f_medium) / f_medium) / abs(r ** q - 1)
    return {"p_signed": q, "p_abs": p, "extrapolated": ext,
            "gci_fine_pct": 100 * gci_f, "gci_medium_pct": 100 * gci_m,
            "monotone": (d21 * d32) > 0,
            "asymptotic_ratio": (gci_m / (r ** q * gci_f)) if gci_f else float("nan")}


def main():
    snap = sys.argv[1]
    levels = load_levels(os.path.join(snap, "levels.csv"))
    dec = load_gci(os.path.join(snap, "gci.csv"))
    case = json.load(open(os.path.join(snap, "case.json")))
    res = {"cells": [], "declared_vs_recomputed_max_delta_pct_point": 0.0,
           "anchor": [], "checks": {}}

    for J in (0.5, 0.7, 0.9):
        for q in ("KT", "10KQ", "eta0"):
            got = roache(levels[(J, "D1")][q], levels[(J, "D2")][q], levels[(J, "D3")][q])
            row = dec[(J, q)]
            d_p = abs(got["p_signed"] - float(row["observed_order"]))
            d_g = abs(got["gci_fine_pct"] - float(row["gci_fine_pct"]))
            d_a = abs(got["asymptotic_ratio"] - float(row["asymptotic_ratio"]))
            res["declared_vs_recomputed_max_delta_pct_point"] = max(
                res["declared_vs_recomputed_max_delta_pct_point"], d_g, d_p, d_a)
            res["cells"].append({"J": J, "quantity": q, "p_signed": round(got["p_signed"], 6),
                                 "gci_fine_pct": round(got["gci_fine_pct"], 6),
                                 "under_3pct": got["gci_fine_pct"] <= 3.0,
                                 "delta_vs_declared": {"p": d_p, "gci_fine": d_g,
                                                       "asym_ratio": d_a}})

    # anchor errors recomputed from the delivered got values + transcribed anchor
    for item in case["anchor_comparison"]:
        q = item["quantity"]
        ref = float(item["ref_anchor"])
        for tag, val in (("finest_D3", float(item["got_finest"])),
                         ("extrapolated", float(item["got_extrapolated"]))):
            err = 100.0 * (val - ref) / ref
            declared = float(item["error_finest_pct"] if tag == "finest_D3"
                             else item["error_extrapolated_pct"])
            res["anchor"].append({"quantity": q, "got_kind": tag, "ref": ref, "got": val,
                                  "recomputed_error_pct": round(err, 4),
                                  "declared_error_pct": declared,
                                  "delta_pp": abs(err - declared),
                                  "over_15pct_platform_line": abs(err) > 15.0,
                                  "under_3pct_case_tolerance": abs(err) <= 3.0})

    v = case["verdict"]
    res["checks"] = {
        "gci_recompute_matches_declared(<=1e-6)":
            res["declared_vs_recomputed_max_delta_pct_point"] < 1e-6,
        "case_tolerance_3pct_met": bool(v["all_criteria_pass"]),
        "all_three_quantities_under_3pct_at_J0.70":
            all(v["case_tolerance_answer"][k]["under_3pct"] for k in ("KT", "10KQ", "eta0")),
        "fail_stage": v["fail_stage"],
        "worst_gci_fine_pct_J0.70": max(v["case_tolerance_answer"][k]["GCI_fine_pct"]
                                        for k in ("KT", "10KQ", "eta0")),
        "anchor_worst_abs_error_pct": max(a["recomputed_error_pct"] for a in res["anchor"]),
        "anchor_points_over_15pct": sum(a["over_15pct_platform_line"] for a in res["anchor"]),
        "anchor_points_under_3pct": sum(a["under_3pct_case_tolerance"] for a in res["anchor"]),
    }
    out = json.dumps(res, ensure_ascii=False, indent=1)
    if "--json" in sys.argv:
        open(sys.argv[sys.argv.index("--json") + 1], "w").write(out)
    print(out)


if __name__ == "__main__":
    main()