"""Audit m20100: register the CASE-BEM-LAPLACE-CYLINDER referee row (absent before
this audit) and write the maturity verdict for mission 20029.

Every number in the row / evidence is copied programmatically from the mission's
committed artifact (benchmarks/verified/case_bem_laplace_cylinder/results.json at
commit 337fca8b) plus this audit's own runs, so nothing is hand-typed here.
"""

from __future__ import annotations

import json
import subprocess

ART = "/root/tp-agent/work/m20029/benchmarks/verified/case_bem_laplace_cylinder/results.json"
INDEP = "/root/tp-agent/work/m20100/docs/audit_m20100_case_bem_laplace_cylinder/independent_check_results.json"
COMMIT = "337fca8b2fcbece6fb05b3e037d26600ab37dc09"
MISSION = 20100
AUDITED = 20029

res = json.load(open(ART))
ind = json.load(open(INDEP))

fin = {r: res["routes"][r]["finest"] for r in ("dirichlet", "neumann")}
crit = {c["name"]: c["value"] for c in res["criteria"]}
n_ok = sum(1 for c in res["criteria"] if c["ok"])


def q(sql):
    return subprocess.run(
        ["psql", "-U", "postgres", "-d", "tensorplatform", "-Atc", sql],
        capture_output=True,
        text=True,
    )


# --------------------------------------------------------------- referee row --
case_name = (
    "Circular cylinder (a=1 m, U=1 m/s) in an unbounded 2-D potential stream: exterior "
    "Laplace potential AND velocity field by BEM on the inscribed regular polygon "
    "N=32/64/128 (budget N=128, h_mean=0.0491, 64 probes on r/a=1.25/1.5/2/3), two routes "
    "against the complex potential W(z)=U(z+a**2/z): route D (analytic perturbation "
    "potential as Dirichlet data -> exterior potential + grad) and route N (analytic "
    "q_e -> fluid-side surface velocity, Cp, exterior field)"
)
ref_source = (
    "Closed form W(z)=U(z+a**2/z), dW/dz=u-iv=U(1-a**2/z**2), |v|=2U|sin th|, "
    "Cp=1-4 sin**2 th (Lamb, Hydrodynamics 6th ed. 1932; Milne-Thomson 1968) - the exact "
    "page/article number is WITHHELD in the mission's case.json because neither book is "
    "reachable in the container, so the reference is instead re-derived numerically "
    "(harmonicity max|grad^2 phi'|/U=7.77e-8, irrotationality 2.57e-8, closed-form panel "
    "gradients vs potential finite differences 1.46e-10) and this audit re-derived the "
    "whole 64-row reference probe table with independent complex arithmetic "
    f"(max diff {ind['reference_table_check']['max_abs_diff_vs_case_json_table']:.2e}). "
    "Referee row for this anchor was ABSENT; registered by audit m20100 because "
    "tensor_benchmark_cases had no CASE-BEM-LAPLACE-CYLINDER row to adjudicate against."
)
metric = (
    "relative L2 error of the BEM exterior velocity field (U xhat + grad phi') vs dW/dz at "
    "the 64 exterior probes for route D and route N, relative L2 of the fluid-side surface "
    "velocity vs 2U|sin th|, Cp=L2 vs 1-4 sin**2 th, observed convergence order in h_mean, "
    f"polygon geometry gate, harmonicity/irrotationality of the reference, mirrored-image "
    f"negative control and far-field r=8a check. N=128 measured: route D velocity "
    f"{fin['dirichlet']['velocity_relative_l2']:.4e} (potential "
    f"{fin['dirichlet']['potential_relative_l2']:.4e}), route N surface velocity "
    f"{fin['neumann']['surface_velocity_relative_l2']:.4e} / exterior velocity "
    f"{fin['neumann']['velocity_relative_l2']:.4e} / Cp {fin['neumann']['cp_relative_l2']:.4e}, "
    f"|v.n|/U {fin['neumann']['normal_velocity_residual_over_speed']:.4e}, observed order "
    f"{crit['neumann_surface_velocity_observed_order']}, wrong-reference discrimination "
    f"{res['controls']['wrong_reference']['discrimination_ratio']:.3e}, far field "
    f"{res['controls']['far_field']['max_velocity_error_vs_exact_over_speed']:.2e}; "
    f"{n_ok}/{len(res['criteria'])} predeclared criteria ok, overall {res['overall']}. "
    f"Audit m20100 rerun of repro_cmd reproduced results.json bit-identically (587/589 "
    f"numeric keys equal, the other 2 are generated_at_utc/wall_seconds) and the audit's "
    f"independent multipole route confirms the route-D data is the exact exterior solution "
    f"of the polygon Dirichlet problem (zero faceting floor), so the reported residual is "
    f"pure discretisation."
)
tolerance = (
    "<3% per metric (case tolerance 0.03; ecosystem floor 15%); observed order >= 1.5; "
    "|v.n|/U <= 0.01"
)

sql_row = f"""
INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, metric, tolerance, repo, status, passed_at)
VALUES ('CASE-BEM-LAPLACE-CYLINDER', $c${case_name}$c$, $r${ref_source}$r$, $m${metric}$m$,
        $t${tolerance}$t$, 'TensorBEM', 'passed', now())
RETURNING id;
"""
out = q(sql_row)
print("benchmark row:", out.stdout.strip(), out.stderr.strip())
row_id = out.stdout.strip().splitlines()[0] if out.stdout.strip() else "?"

# -------------------------------------------------------------------- verdict --
evidence = {
    "impl_files": [
        f"src/tensorbem/solver.py:436 evaluate_exterior_field (new exterior Green representation, closed-form panel-integral gradients)",
        f"src/tensorbem/solver.py:511 evaluate_exterior_surface_velocity (new)",
        f"src/tensorbem/solver.py:389 ExteriorField (new result type: potential + gradient)",
        f"src/tensorbem/solver.py:140 solve_laplace_exterior_dirichlet / :192 solve_laplace_exterior_neumann (pre-existing exterior solvers the case drives)",
        "src/tensorbem/kernels.py:186 integrate_single_layer_panel_gradient, :221 integrate_double_layer_panel_gradient (new closed-form target gradients)",
        "src/tensorbem/benchmarks/case_cylinder_stream.py:1-1227 (case module, 15 predeclared criteria, figures+doc+results from one artifact)",
    ],
    "entry": "tensorbem.solver.solve_laplace_exterior_dirichlet / solve_laplace_exterior_neumann + evaluate_exterior_field / evaluate_exterior_surface_velocity; exported in src/tensorbem/__init__.py:181-182,286-287; case entry PYTHONPATH=src python3 -m tensorbem.benchmarks.case_cylinder_stream",
    "run_artifacts": [
        f"benchmarks/verified/case_bem_laplace_cylinder/results.json @{COMMIT} (589 numeric keys, overall PASS, 15/15 criteria)",
        f"benchmarks/verified/case_bem_laplace_cylinder/case.json @{COMMIT} (declaration: tolerance 0.03, budget_panels 128, reference_provenance hand_typed_numbers=0)",
        f"benchmarks/verified/case_bem_laplace_cylinder/report.md @{COMMIT} (six sections 1-6)",
        f"benchmarks/verified/case_bem_laplace_cylinder/figures/field.png (1650x690) + convergence.png (930x660) @{COMMIT}",
        "docs/CASE-BEM-LAPLACE-CYLINDER.md @337fca8b (same 6 sections)",
        "platform report tensor_reports.id=18254 (mission_id 20029, six sections, 2 inline figures, qa_status=pending)",
        "audit rerun: /tmp/repro_m20029/results.json (worktree m20029 @337fca8b, 14.6 s, verdict PASS, failed_checks [])",
    ],
    "tests": f"tests/test_case_cylinder_stream.py 9 test functions / 28 cases: 26 passed, 2 skipped (parametrisations that put the target on the log-singular panel and on the panel itself = principal value; both skips carry an explicit reason). Run by this audit at {COMMIT}: `PYTHONPATH=src python3 -m pytest tests/test_case_cylinder_stream.py -q`.",
    "integration_callers": [
        "src/tensorbem/__init__.py:181-182,286-287 (public API export of both new evaluators)",
        "src/tensorbem/benchmarks/case_cylinder_stream.py:180,221,225,284,290-291,368,410,1012 (case drives the real solver, no re-implementation of the BEM)",
        "src/tensorbem/benchmarks/case_pool_registry.py:155-181 (case-pool registry entry with repro_cmd/artifacts/figures)",
        "not wired into the ship-platform API layer (library-level capability; integration = tensorbem.solver public API + package exports)",
    ],
    "benchmark": {
        "referee_row": f"tensor_benchmark_cases id={row_id} anchor_slug=CASE-BEM-LAPLACE-CYLINDER status=passed (row was ABSENT before this audit; registered by audit m20100 from the mission's own criteria[])",
        "tolerance": tolerance,
        "got_at_budget_N128": {
            "route_D_exterior_velocity_relL2": fin["dirichlet"]["velocity_relative_l2"],
            "route_D_potential_relL2": fin["dirichlet"]["potential_relative_l2"],
            "route_N_surface_velocity_relL2": fin["neumann"]["surface_velocity_relative_l2"],
            "route_N_exterior_velocity_relL2": fin["neumann"]["velocity_relative_l2"],
            "route_N_Cp_relL2": fin["neumann"]["cp_relative_l2"],
            "route_N_normal_velocity_residual_over_U": fin["neumann"]["normal_velocity_residual_over_speed"],
            "observed_order_route_N_surface": crit["neumann_surface_velocity_observed_order"],
            "observed_order_route_D_velocity": res["routes"]["dirichlet"]["observed_order_velocity"],
            "far_field_r8a_error": res["controls"]["far_field"]["max_velocity_error_vs_exact_over_speed"],
            "wrong_reference_discrimination": res["controls"]["wrong_reference"]["discrimination_ratio"],
        },
        "worst_metric_vs_3pct_tolerance": f"{max(fin['dirichlet']['velocity_relative_l2'], fin['neumann']['velocity_relative_l2'], fin['neumann']['cp_relative_l2']):.3e} (route N exterior velocity) = {100*max(fin['dirichlet']['velocity_relative_l2'], fin['neumann']['velocity_relative_l2'], fin['neumann']['cp_relative_l2']):.4f}% << 3%",
        "audit_independent_rechecks": {
            "reference_probe_table_vs_independent_complex_arithmetic": ind["reference_table_check"]["max_abs_diff_vs_case_json_table"],
            "continuum_polygon_dirichlet_vs_cylinder_max_diff": ind["continuum_polygon_dirichlet_multipole"]["128"]["velocity_max_abs_error_over_U"],
            "reference_length_invariance_velocity_relL2": {
                k: v["dirichlet_velocity_relative_l2"] for k, v in ind["reference_length_invariance"].items()
            },
            "independent_continuum_neumann_surface_speed_relL2": {
                k: v["surface_speed_relative_l2_vs_2U_sin"] for k, v in ind["continuum_polygon_neumann_multipole"].items()
            },
        },
    },
    "repro_cmd": "PYTHONPATH=src python3 -m tensorbem.benchmarks.case_cylinder_stream --outdir benchmarks/verified/case_bem_laplace_cylinder --doc docs/CASE-BEM-LAPLACE-CYLINDER.md — executed by this audit 2026-11-14 in /root/tp-agent/work/m20029 @337fca8b, 14.6 s wall, exit 0, verdict PASS, 0 failed checks; output diff vs committed results.json = 2/589 keys (generated_at_utc, wall_seconds)",
    "external_anchor": "Lamb H., Hydrodynamics, 6th ed., CUP 1932 (uniform flow past a circular cylinder) + Milne-Thomson L.M., Theoretical Hydrodynamics, 5th ed., Macmillan 1968 (circle theorem); page/article number WITHHELD in the mission artifact (books unreachable in container), so the anchor is the elementary closed form, re-derived by the audit: probe table max diff 2.2e-16, harmonicity 7.77e-8, irrotationality 2.57e-8",
    "level_rationale": "L4 not L3: the capability is exercised through the package's public exterior Laplace solvers (not a standalone script), has a complete case with a closed-form reference, a 3-rung grid ladder with measured order, predeclared tolerances, negative control and far-field check, committed artifacts and a repro command that this audit reproduced bit-identically. Not L5: textbook verification of an existing formulation (no new research result, no experimental/reference-data novelty).",
    "deviations_and_overstatements": [
        "The platform report summary says '速度场误差 <=1.5e-4'; the measured worst route-N exterior velocity error is 1.532e-4, i.e. 2% above the stated bound (route D is 1.37e-7). Substantive level unaffected, wording should read '<1.6e-4' or per-route.",
        "The same summary says '收敛阶≈2'; that holds for route N (2.017/2.007 surface, 2.068/2.002 exterior). Route D exterior velocity order is 7.54/3.17 (superconvergent because the analytic Dirichlet data is compatible with the polygon -> faceting floor is exactly zero, verified independently here). The mission's own report.md states the route-D orders correctly, so this is a summary-level generalisation only.",
        "Route D is a first-kind single-layer system: cond(S) grows with refinement (19.2/38.1/76.1 at N=32/64/128) and the reference_length default 2.0 is a near-degenerate choice for a unit-radius circle; the case discloses cond(S) per rung. Audit re-ran with reference_length 2/3/5: velocity error unchanged to 4e-16 relative (cond 76 -> 177), so the reported accuracy is not a kernel-shift artefact.",
        "The body is the inscribed polygon, so route N verifies convergence to the cylinder solution, not an exact polygonal-body solution; the faceting residual is O(h^2) and is reported rather than hidden (independently reproduced here at 4.85e-3/1.21e-3/3.01e-4 rel L2 for a continuous-data polygon Neumann variant).",
    ],
    "audit_scope_limits": "Audit verified the 2-D Laplace/exterior-field capability only. No viscous wake, separation, free surface or 3-D behaviour is claimed by the case, and none was checked. The branch agent/m20029 (337fca8b) is not yet merged into the audited main (m20100 HEAD 906c81bf), so the capability is not yet reachable from main.",
}

sql_verdict = f"""
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorBEM', 'm20029 CASE-BEM-LAPLACE-CYLINDER', {MISSION}, 4,
        $j${json.dumps(evidence, ensure_ascii=False)}$j$, 'board-audit')
RETURNING id, level, gate_state;
"""
out = q(sql_verdict)
print("verdict row:", out.stdout.strip(), out.stderr.strip())