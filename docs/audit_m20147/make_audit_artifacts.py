#!/usr/bin/env python3
"""Build the m20147 cross-audit evidence bundle for m20079 CASE-BEM-HELMHOLTZ-SPHERE.

Reads the audited mission's committed tables (read-only), the auditor's own
independent Mie series output, and emits:

  audit_metrics.json          every number quoted by the audit report
  fig_audit_sigma_vs_ka.png   BEM vs reference vs auditor's own series
  fig_audit_error_budget.png  per-ka error against the 3 % / 6 % bands + ladder
  fig_audit_pattern.png       far-field pattern, BEM vs reference

No number in the report is typed by hand: all of them come from this payload.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
AUDITED = Path("/root/tp-agent/work/m20079/docs/case_bem_helmholtz_sphere")


def rows(name):
    with open(AUDITED / name) as fh:
        return list(csv.DictReader(fh))


sweep = rows("table_sweep.csv")
ladder = rows("table_ladder.csv")
quad = rows("table_quadrature.csv")
chief = rows("table_chief.csv")
pattern = rows("table_pattern.csv")
refchk = rows("table_reference.csv")
drei = json.load(open(AUDITED / "three_checks.json"))
verdict = json.load(open(AUDITED / "verdict.json"))
indep = json.load(open(HERE / "independent_mie_metrics.json"))

# ---- independent series on a dense ka axis (auditor's own code) ------------
from independent_mie import sigma_hat  # noqa: E402

ka_dense = np.linspace(0.2, 3.0, 57)
sig_dense = np.array([sigma_hat(k) for k in ka_dense])

ka = np.array([float(r["ka"]) for r in sweep])
bem = np.array([float(r["sigma_hat_bem"]) for r in sweep])
ref = np.array([float(r["sigma_hat_ref"]) for r in sweep])
err_pct = 100.0 * np.abs(bem - ref) / ref
indep_err_pct = np.array([r["rel_bem_vs_auditor_pct"] for r in indep["rows"]])

lam = np.array([float(r["panel_count"]) for r in ladder])
lad_err = np.array([float(r["relative_error_pct"]) for r in ladder])
edge = np.array([float(r["edge_over_wavelength"]) for r in ladder])
quad_err = np.array([float(r["relative_error_pct"]) for r in quad])
chief_rows = {
    "with": [r for r in chief if r["use_chief"] == "True"][0],
    "without": [r for r in chief if r["use_chief"] == "False"][0],
}
pat_theta = np.array([float(r["theta_deg"]) for r in pattern])
pat_bem = np.array([float(r["abs_f_bem"]) for r in pattern])
pat_ref = np.array([float(r["abs_f_ref"]) for r in pattern])

checks = {c["name"]: c for c in verdict["acceptance"]["checks"]}
metrics = {
    "audited": {
        "mission": 20079,
        "case_code": "CASE-BEM-HELMHOLTZ-SPHERE",
        "worktree": str(AUDITED.parent.parent),
        "tolerance": verdict["tolerance"],
        "verdict_in_artifacts": verdict["verdict"],
    },
    "reference_independence": {
        "case_two_implementations_max_rel_diff": max(
            float(r["relative_difference"]) for r in refchk
        ),
        "auditor_third_implementation_max_rel_diff_vs_case_ref": max(
            r["rel_ref_vs_auditor"] for r in indep["rows"]
        ),
        "auditor_rayleigh_limit_7over9": indep["limits"]["rayleigh_ka0.02"],
        "auditor_extinction_ka100": indep["limits"]["extinction_ka100"],
        "case_declared_rayleigh_check_value": checks["reference_rayleigh_limit"]["value"],
        "case_declared_extinction_check_value": checks["reference_extinction_limit"]["value"],
    },
    "got_vs_ref": [
        {
            "ka": float(r["ka"]),
            "panels": int(r["panel_count"]),
            "panels_per_wavelength": float(r["panels_per_wavelength"]),
            "sigma_hat_bem": float(r["sigma_hat_bem"]),
            "sigma_hat_ref": float(r["sigma_hat_ref"]),
            "rel_error_pct": float(r["relative_error_pct"]),
            "auditor_rel_error_vs_own_series_pct": indep_err_pct[i],
            "within_tolerance": r["within_tolerance"] == "True",
            "boundary_residual": float(r["boundary_residual"]),
        }
        for i, r in enumerate(sweep)
    ],
    "worst_error_pct": float(max(err_pct)),
    "worst_error_pct_vs_auditor_series": indep["worst_bem_error_pct_vs_auditor"],
    "within_3pct": bool(max(err_pct) < 3.0),
    "ladder": {
        "rows": [
            {
                "panels": int(r["panel_count"]),
                "edge_over_lambda": float(r["edge_over_wavelength"]),
                "rel_error_pct": float(r["relative_error_pct"]),
                "order_vs_previous": (
                    None
                    if i == 0
                    else float(
                        math.log(lad_err[i - 1] / lad_err[i])
                        / math.log(edge[i - 1] / edge[i])
                    )
                ),
            }
            for i, r in enumerate(ladder)
        ],
        "finest_mesh_error_pct": float(lad_err[-1]),
        "declared_limit_pct": 100.0 * checks["mesh_ladder_converged"]["limit"],
        "check_passed": checks["mesh_ladder_converged"]["passed"],
        "blocking": checks["mesh_ladder_converged"]["blocking"],
    },
    "quadrature": {
        "orders": [int(r["n_theta"]) for r in quad],
        "spread_pct": float(quad_err.max() - quad_err.min()),
        "declared_limit": checks["quadrature_independence"]["limit"],
        "check_passed": checks["quadrature_independence"]["passed"],
    },
    "chief": {
        "ka": float(chief_rows["with"]["ka"]),
        "error_with_chief_pct": float(chief_rows["with"]["relative_error_pct"]),
        "error_without_chief_pct": float(chief_rows["without"]["relative_error_pct"]),
        "cond": float(chief_rows["with"]["condition_number"]),
        "residual_with_chief": float(chief_rows["with"]["boundary_residual"]),
        "residual_without_chief": float(chief_rows["without"]["boundary_residual"]),
    },
    "pattern": {
        "max_rel_deviation_pct": 100.0 * checks["angular_pattern"]["value"],
        "declared_limit_pct": 100.0 * checks["angular_pattern"]["limit"],
        "check_passed": checks["angular_pattern"]["passed"],
        "n_angles": len(pattern),
    },
    "declared_checks": [
        {"name": c["name"], "value": c["value"], "limit": c["limit"],
         "blocking": c["blocking"], "passed": c["passed"]}
        for c in verdict["acceptance"]["checks"]
    ],
    "failed_checks": [c["name"] for c in verdict["acceptance"]["checks"] if not c["passed"]],
    "knowledge_engine_at_case_time": {
        "available": drei["knowledge_base_capability"]["available"],
        "reason": drei["knowledge_base_capability"]["reason"],
    },
}
json.dump(metrics, open(HERE / "audit_metrics.json", "w"), indent=2)

# ---- figure 1: cross-section vs ka ----------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.plot(ka_dense, sig_dense, "-", color="0.55", lw=1.2,
        label="auditor partial-wave series (independent code)")
ax.plot(ref, ref, "s", ms=6, mfc="none", color="tab:blue", label="case reference column")
ax.plot(ka, bem, "o", ms=6, color="tab:red", label="BEM got (icosphere(2), 320 panels)")
for k, b, r in zip(ka, bem, ref):
    ax.annotate(f"{100 * abs(b - r) / r:.2f}%", (k, b), textcoords="offset points",
                xytext=(6, -12), fontsize=8, color="tab:red")
ax.set_xlabel("ka")
ax.set_ylabel(r"$\hat\sigma=\sigma_{sc}/(\pi a^2)$")
ax.set_title("Rigid-sphere scattering: BEM vs analytic series (m20079 re-audit)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "fig_audit_sigma_vs_ka.png", dpi=150)
plt.close(fig)

# ---- figure 2: error budget ------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2))
ax1.semilogy(ka, err_pct, "o-", color="tab:red", label="vs case reference column")
ax1.semilogy(ka, indep_err_pct, "x--", color="k", label="vs auditor's own series")
ax1.axhline(3.0, color="tab:green", ls=":", label="3 % question")
ax1.axhline(6.0, color="tab:orange", ls="--", label="6 % declared tolerance")
ax1.set_xlabel("ka")
ax1.set_ylabel("relative error [%]")
ax1.set_title("sweep: 3 ka points, 320 panels")
ax1.grid(alpha=0.3, which="both")
ax1.legend(fontsize=8)
for k, e, i in zip(ka, err_pct, indep_err_pct):
    ax1.annotate(f"{e:.2f}", (k, e), textcoords="offset points", xytext=(4, 5), fontsize=8)

ax2.loglog(lam, lad_err, "o-", color="tab:blue")
for l, e, ed in zip(lam, lad_err, edge):
    ax2.annotate(f"{e:.2f}% (edge/λ={ed:.4f})", (l, e), textcoords="offset points",
                 xytext=(4, 4), fontsize=8)
ax2.axhline(1.0, color="tab:orange", ls="--", label="declared ladder limit 1 %")
ax2.axhline(3.0, color="tab:green", ls=":", label="3 %")
ax2.set_xlabel("panels")
ax2.set_ylabel("error at ka=1 [%]")
ax2.set_title("mesh ladder (only 2 rungs committed)")
ax2.grid(alpha=0.3, which="both")
ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "fig_audit_error_budget.png", dpi=150)
plt.close(fig)

# ---- figure 3: far-field pattern ------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 4.2))
ax.plot(pat_theta, pat_ref, "-", color="0.5", label="reference |f(θ)|")
ax.plot(pat_theta, pat_bem, "o", ms=3, color="tab:red", label="BEM |f(θ)|")
ax.set_xlabel("θ [deg]")
ax.set_ylabel(r"$|f(\theta)|$ [m]")
ax.set_title(f"far field at ka=2: max dev {metrics['pattern']['max_rel_deviation_pct']:.2f} %")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "fig_audit_pattern.png", dpi=150)
plt.close(fig)

print(json.dumps({k: metrics[k] for k in
                  ("worst_error_pct", "within_3pct", "ladder", "pattern")},
                 indent=2)[:1200])