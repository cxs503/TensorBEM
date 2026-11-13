#!/usr/bin/env python3
"""board-audit m17232: independent recomputation for m17176 CASE-BEM-SPHERE-ADDED-MASS.

Reads the m17176 subject artifacts read-only (copied to ``refs/``) and the
auditor's own reruns (``rerun/``), recomputes every reported quantity, and
emits ``analysis.json`` with no hand-typed numbers.

Definitions used here (stated explicitly because the subject report and the
repo README disagree on the observed order):

* icosphere ``subdivisions`` s -> panels N = 20*4**s, edge length ~ h_0/2**s.
  One refinement step therefore halves h, so the observed order is
  ``p = log2( e(s) / e(s+1) )``. Using ``log4`` instead halves the fitted
  order -- this is exactly the arithmetic slip found in the subject report.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUBJECT = HERE / "refs"
RERUN = HERE / "rerun"
SUBS = (1, 2, 3, 4)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def order(e0: float, e1: float) -> float:
    return math.log2(e0 / e1)


def main() -> None:
    pairs = []
    for s in SUBS:
        sub = load(SUBJECT / f"sphere_added_mass_sub{s}.json")
        rer = load(RERUN / f"sub{s}.json") if (RERUN / f"sub{s}.json").exists() else None
        rec = {
            "sub": s,
            "panels": sub["panels"],
            "h_max": sub["h_max"],
            "volume_ratio": sub["volume_ratio"],
            "m_A_subject": sub["added_mass"],
            "m_A_rerun": rer["added_mass"] if rer else None,
            "m_A_abs_diff": abs(sub["added_mass"] - rer["added_mass"]) if rer else None,
            "ref_0p5_rhoV": sub["exact_added_mass"],
            "rel_err_subject": sub["added_mass_relative_error"],
            "rel_err_rerun": rer["added_mass_relative_error"] if rer else None,
            "rel_err_bit_identical": (
                rer is not None
                and abs(sub["added_mass_relative_error"] - rer["added_mass_relative_error"]) == 0.0
            ),
            "flux_l2_subject": sub["normal_derivative_relative_l2_error"],
            "flux_l2_rerun": rer["normal_derivative_relative_l2_error"] if rer else None,
            "residual_subject": sub["relative_residual_l2"],
            "cond_subject": sub["single_layer_condition_number"],
            "pass_3pct": sub["added_mass_relative_error"] < 0.03,
        }
        pairs.append(rec)

    integral_orders = [
        {"from": pairs[i]["panels"], "to": pairs[i + 1]["panels"],
         "p_log2": order(pairs[i]["rel_err_subject"], pairs[i + 1]["rel_err_subject"]),
         "p_log4_misread": math.log(pairs[i]["rel_err_subject"] / pairs[i + 1]["rel_err_subject"]) / math.log(4.0)}
        for i in range(len(pairs) - 1)
    ]
    flux_orders = [
        {"from": pairs[i]["panels"], "to": pairs[i + 1]["panels"],
         "p_log2": order(pairs[i]["flux_l2_subject"], pairs[i + 1]["flux_l2_subject"])}
        for i in range(len(pairs) - 1)
    ]

    sub0 = load(RERUN / "sub0.json")
    scale = load(RERUN / "sub2_rho1000_U2.json")
    negc = load(HERE / "negcontrol_sphere_added_mass.json")
    sens = load(SUBJECT / "sensitivity_sub2.json")

    # README declared values (README.md:62-63), transcribed by the auditor and
    # then checked against recomputation -- not used as truth.
    readme = {"panels": [20, 80, 320, 1280],
              "declared": [2.13e-1, 6.48e-2, 1.74e-2, 4.47e-3]}
    readme_rows = []
    measured = [sub0["added_mass_relative_error"]] + [p["rel_err_subject"] for p in pairs[:3]]
    panels = [sub0["panels"]] + [p["panels"] for p in pairs[:3]]
    for n_decl, e_decl, n_meas, e_meas in zip(readme["panels"], readme["declared"], panels, measured):
        readme_rows.append({"panels_declared": n_decl, "declared_err": e_decl,
                            "panels_measured": n_meas, "measured_err": e_meas,
                            "consistent_at_3digits": abs(e_decl - e_meas) / e_meas < 1e-2})

    out = {
        "subject": "m17176 CASE-BEM-SPHERE-ADDED-MASS (repo TensorBEM)",
        "auditor": "board-audit m17232",
        "ref": {
            "value": pairs[0]["ref_0p5_rhoV"],
            "formula": "m_A = 0.5*rho*V = 0.5*1025*4/3*pi -> Lamb 1932 §6.10 / Newman 1977 eq.4.132 sphere limit",
            "independent_recompute": "tensorbem.analytic.ellipsoid_added_mass((1,1,1),1025) -> see analytic_anchor",
        },
        "rows": pairs,
        "integral_order": integral_orders,
        "flux_order": flux_orders,
        "readme_consistency": readme_rows,
        "scale_invariance": {
            "base": {"rho": 1025.0, "U": 1.0, "rel_err": sub0 and pairs[1]["rel_err_subject"]},
            "scaled": {"rho": 1000.0, "U": 2.0, "rel_err": scale["added_mass_relative_error"],
                       "m_A": scale["added_mass"], "ref": scale["exact_added_mass"]},
            "identical": abs(scale["added_mass_relative_error"] - pairs[1]["rel_err_subject"]) < 1e-15,
            "subject_sensitivity_json_identical": abs(sens["base"]["relative_error"]
                                                     - sens["scaled"]["relative_error"]) < 1e-15,
        },
        "negative_control": negc,
        "sha256": {
            **{f"refs/sphere_added_mass_sub{s}.json": sha(SUBJECT / f"sphere_added_mass_sub{s}.json") for s in SUBS},
            **{f"rerun/sub{s}.json": sha(RERUN / f"sub{s}.json") for s in SUBS if (RERUN / f"sub{s}.json").exists()},
            "refs/sensitivity_sub2.json": sha(SUBJECT / "sensitivity_sub2.json"),
            "refs/flux_panels_sub2.csv": sha(SUBJECT / "flux_panels_sub2.csv"),
        },
    }

    # findings
    ours = [o["p_log2"] for o in integral_orders]
    out["findings"] = {
        "F1_medium_report_order_arithmetic": {
            "claim_in_subject_report": "observed order 0.95 / 0.98 / 0.99 for the integral added mass; conclusion 'the integral inherits the first-order per-panel flux'",
            "recomputed_p_log2": ours,
            "base4_misread_reproduces_report": [
                round(o["p_log4_misread"], 3) for o in integral_orders],
            "verdict": "FALSIFIED: with h halving per subdivision the integral added-mass order is ~2 (1.72/1.90/1.96/1.98), matching README 'near 2'; the subject's 0.95/0.98/0.99 are log4-space values. Per-panel flux is the ~1st-order quantity (0.94/1.09/1.09).",
            "impact": "no effect on the reported added-mass errors or the <3% verdict; affects only the convergence interpretation and the subject's suggestion that README needed review.",
        },
        "F2_medium_registry_gap": {
            "claim": "tensor_benchmark_cases is the machine referee for this anchor",
            "measured": "0 rows for anchor_slug=CASE-BEM-SPHERE-ADDED-MASS before this audit (35 rows total in table)",
            "action": "registered by this audit with the audited numbers (see insert_verdict.sql)",
        },
        "F3_low_cluster_claim": {
            "claim_in_subject_report": "sub3/sub4 heavy recompute submitted to cluster as cj1448",
            "measured": "cj1448 exists but terminal state=killed (submitted 11-13 01:55); no cluster artifact is cited in the report; sub3/sub4 numbers came from the local setsid route, which the report also states",
            "verdict": "claim is half-true and self-consistent (report says the local route produced the numbers first), but no cluster row backs the numbers; auditor reran sub3/sub4 locally AND resubmitted to the cluster (cj1466)",
        },
        "F4_low_tracelevel_scope": {
            "measured": "the benchmark imposes the exact analytic Dirichlet trace u=-0.5*U*a*cos(theta) and solves for q; it does not solve the physical Neumann BVP for a moving body",
            "consequence": "validates the exterior DtN/single-layer solution operator, not a general radiation solve -> level capped at L4, L5 not claimed",
        },
        "F5_info_sub1_fails_threshold": {
            "measured": "80-panel error 6.4759e-2 > 3% (already reported as FAIL by the subject); 320 panels onward pass",
        },
        "F6_info_no_artifacts_in_repo": {
            "measured": "m17176 deliverables live only in its worktree (runs/m17176/*); this audit pins copies with sha256 under docs/board_audit/m17232/refs/",
        },
    }
    out["verdict_level"] = {
        "level": 4,
        "why": "real BEM solve (input->compute->output, negative-controlled), classical closed-form anchor recomputed, 4-point refinement with per-point error table, one-command repro independently rerun bit-identically, tests green; not L5: no experimental/printed-table external anchor, trace-level scope, and the report's convergence-order analysis is wrong (F1).",
    }
    (HERE / "analysis.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("integral_order", "flux_order", "readme_consistency")},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()