#!/usr/bin/env python3
"""Board audit of mission m17275 — anchor ``CASE-BEM-ICE-SPHERE-HEAVE`` (TensorBEM).

Read-only audit of an already ``succeeded`` delivery (audit mission m17312).  It
re-derives every number the mission claims from the raw JSON artifact, re-checks
the delivery against the case-pool criterion, and writes

    runs/m17312_audit/audit_summary.json
    docs/board_audit/m17312/fig1_ladder_error.png
    docs/board_audit/m17312/fig2_error_decomposition.png
    docs/board_audit/m17312/fig3_submergence.png
    docs/board_audit/m17312/fig4_sensitivity.png

Nothing here is copied from the subject's own summary: the Lamb reference is
evaluated from the closed form in this file, the multipole route is called
directly, and the gate booleans are recomputed from the per-row numbers.

    python scripts/audit/audit_m17312_ice_sphere_heave.py
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AUDIT = REPO / "runs/m17312_audit"
SUBJECT = AUDIT / "source_m17275"
SUBJECT_JSON = SUBJECT / "artifacts/case_bem_ice_sphere_heave.json"
RERUN_JSON = AUDIT / "rerun_full_matrix.json"
QUICK_JSON = AUDIT / "rerun_quick.json"
SUBJECT_MD = SUBJECT / "docs/case_bem_ice_sphere_heave_numbers.md"
REPORT_TOOL = SUBJECT / "tools/ice_sphere_heave_report.py"
FIGDIR = REPO / "docs/board_audit/m17312"
SUBJECT_SRC = SUBJECT / "src"
# the subject tree the auditor executed (author's mission worktree, read-only)
SUBJECT_WORKTREE = Path("/root/tp-agent/work/m17275")
SUBJECT_COMMIT = "a35c7c096b9dd2409ffc2b5e289f08812b653d9d"

RADIUS = 5.0
RHO_W = 1025.0
RHO_ICE = 917.0
G = 9.80665
CASE_TOL = 0.05              # tensor_case_pool 172 tol_policy '<5%'
ECOSYSTEM_TOL = 0.03         # ecosystem accuracy standard


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def lamb_closed_form(radius: float = RADIUS, density: float = RHO_W) -> float:
    """Lamb (1932) art. 92, evaluated here from scratch (no repo helper)."""
    return 2.0 / 3.0 * math.pi * density * radius ** 3


def free_surface_low_frequency_correction(ratio: float) -> float:
    """Classical leading-order free-surface correction 3/8 (a/f)^3 (Lamb art. 92/98).

    Independent of the repository: the leading image-dipole term of a sphere at
    distance ``f`` from a pressure-release surface.
    """
    return 3.0 / 8.0 * ratio ** -3


def observed_order(rows, key: str = "A33_relative_error_vs_exact") -> float:
    xs, ys = [], []
    for row in rows:
        err = row[key]
        if err > 0.0:
            xs.append(math.log(row["panels"] ** -0.5))
            ys.append(math.log(err))
    if len(xs) < 2:
        return float("nan")
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def multipole(radius: float, submergence: float, wavenumber: float, *, root: Path):
    """Call the repository's analytic multipole series (no discretisation)."""
    sys.path.insert(0, str(root))
    from tensorbem.analytic import submerged_sphere_wave_multipole  # noqa: PLC0415

    res = submerged_sphere_wave_multipole(
        wavenumber * radius, radius=radius, submergence=submergence,
        density=RHO_W, n_terms=26, gravity=G,
    )
    return float(res.heave_added_mass)


def verify_subject_vs_pool(subject: dict) -> dict:
    """Criterion: tensor_case_pool 172 -> ref Lamb (2/3)pi rho R^3, tol '<5%'."""
    lamb = lamb_closed_form()
    rows = subject["rows"]
    per_row = []
    for row in rows:
        got = row["A33_kg"]
        err = abs(got - lamb) / lamb
        per_row.append({
            "label": row["label"], "panels": row["panels"], "got_kg": got,
            "recomputed_error_vs_lamb": err,
            "delivered_error_vs_lamb": row["A33_relative_error_vs_lamb"],
            "delta_delivered_vs_recomputed": abs(err - row["A33_relative_error_vs_lamb"]),
            "within_case_tolerance": bool(err < CASE_TOL),
            "within_ecosystem_standard": bool(err < ECOSYSTEM_TOL),
        })
    return {
        "reference_kg_recomputed": lamb,
        "reference_kg_delivered": subject["reference"]["value_kg"],
        "reference_rel_diff": abs(lamb - subject["reference"]["value_kg"]) / lamb,
        "reference_is_half_displaced_mass": abs(
            lamb - 0.5 * RHO_W * 4.0 / 3.0 * math.pi * RADIUS ** 3) / lamb,
        "case_tolerance": CASE_TOL,
        "rows_total": len(rows),
        "rows_within_case_tolerance": sum(r["within_case_tolerance"] for r in per_row),
        "rows_within_ecosystem_standard": sum(r["within_ecosystem_standard"] for r in per_row),
        "worst_row": max(per_row, key=lambda r: r["recomputed_error_vs_lamb"]),
        "max_abs_delta_vs_delivered_error": max(
            r["delta_delivered_vs_recomputed"] for r in per_row),
        "per_row": per_row,
    }


def gates_recomputed(subject: dict) -> dict:
    """Rebuild P1..P5 from the raw rows and compare with the delivered booleans."""
    rows = {r["label"]: r for r in subject["rows"]}
    lamb = lamb_closed_form()
    primary = rows["ladder__352"]
    deep = rows["depth__12R__med"]
    open_ice, thin_ice = rows["ice__0m"], rows["ice__0.3m"]
    ladder = [rows[f"ladder__{n}"] for n in (168, 352, 816, 1472)]
    errs_exact = [r["A33_relative_error_vs_exact"] for r in ladder]
    order = observed_order(ladder)
    ice_shift = abs(thin_ice["A33_kg"] - open_ice["A33_kg"]) / open_ice["A33_kg"]
    mine = {
        "P1_primary_vs_lamb_lt_5pct": abs(primary["A33_kg"] - lamb) / lamb < CASE_TOL,
        "P2_primary_vs_exact_lt_5pct": primary["A33_relative_error_vs_exact"] < CASE_TOL,
        "P3a_ladder_monotone_vs_exact": all(b < a for a, b in zip(errs_exact, errs_exact[1:])),
        "P3b_order_in_0.3_3": 0.3 <= order <= 3.0,
        "P4a_deep_vs_lamb_lt_5pct": abs(deep["A33_kg"] - lamb) / lamb < CASE_TOL,
        "P4b_deep_exact_offset_lt_0.5pct": abs(deep["exact_offset_vs_lamb"]) < 0.005,
        "P5_ice_control": ice_shift < primary["A33_relative_error_vs_exact"] + 1e-3,
    }
    delivered = subject["verification"]
    mapping = {
        "P1_primary_vs_lamb_lt_5pct": "gate_primary_lt_5pct (P1)",
        "P2_primary_vs_exact_lt_5pct": "gate_mesh_vs_exact_lt_5pct (P2)",
        "P3a_ladder_monotone_vs_exact": "ladder_monotone (P3a)",
        "P3b_order_in_0.3_3": "gate_ladder_order_in_band (P3b)",
        "P4a_deep_vs_lamb_lt_5pct": "gate_deep_lt_5pct (P4a)",
        "P4b_deep_exact_offset_lt_0.5pct": "gate_deep_offset_lt_0p5pct (P4b)",
        "P5_ice_control": "gate_ice_control (P5)",
    }
    agree = {k: bool(mine[k] == bool(delivered[v])) for k, v in mapping.items()}
    return {
        "recomputed": {k: bool(v) for k, v in mine.items()},
        "delivered": {v: bool(delivered[v]) for v in mapping.values()},
        "agree": agree,
        "all_agree": all(agree.values()),
        "observed_order_recomputed": order,
        "observed_order_delivered": delivered["ladder_observed_order"],
        "observed_order_abs_diff": abs(order - delivered["ladder_observed_order"]),
        "all_gates_pass": all(bool(v) for v in mine.values()),
    }


def decomposition(subject: dict) -> dict:
    """Why the primary point agrees with Lamb: two errors of opposite sign."""
    rows = {r["label"]: r for r in subject["rows"]}
    p = rows["ladder__352"]
    lamb = lamb_closed_form()
    exact = p["A33_exact_multipole_kg"]
    physics = (exact - lamb) / lamb                 # finite-submergence offset
    disc_abs = (exact - p["A33_kg"]) / exact        # discretisation undershoot
    disc_on_lamb = (exact - p["A33_kg"]) / lamb
    net = (lamb - p["A33_kg"]) / lamb
    ladder_vs_lamb = [rows[f"ladder__{n}"]["A33_relative_error_vs_lamb"]
                      for n in (168, 352, 816, 1472)]
    return {
        "primary_label": p["label"],
        "physics_offset_exact_vs_lamb": physics,
        "discretisation_undershoot_vs_exact": disc_abs,
        "discretisation_undershoot_on_lamb_scale": disc_on_lamb,
        "net_error_vs_lamb": net,
        "identity_residual": abs(net - (disc_on_lamb - physics)),
        "cancellation_ratio": (abs(physics) + abs(disc_on_lamb)) / abs(net),
        "ladder_error_vs_lamb": ladder_vs_lamb,
        "ladder_error_vs_lamb_monotone": all(b < a for a, b in zip(ladder_vs_lamb, ladder_vs_lamb[1:])),
        "lamb_floor_at_primary_ratio": physics,   # err vs Lamb cannot go below this at d=3R
        "finest_rung_error_vs_lamb": ladder_vs_lamb[-1],
        "reading": "1.63 % agreement with Lamb = +2.96 % physics offset minus the "
                   "discretisation undershoot; it is NOT a converged accuracy figure",
    }


def gate_weakness(subject: dict) -> dict:
    """P5's threshold is the mesh error, so it can only fail for a >4.5 % ice effect."""
    rows = {r["label"]: r for r in subject["rows"]}
    p = rows["ladder__352"]
    open_ice, thin_ice = rows["ice__0m"], rows["ice__0.3m"]
    observed = abs(thin_ice["A33_kg"] - open_ice["A33_kg"]) / open_ice["A33_kg"]
    threshold = p["A33_relative_error_vs_exact"] + 1e-3
    spread = {
        "ice_thickness_816": [rows[f"ice__{t}m"]["A33_kg"] for t in ("0", "0.3", "0.9", "2.5")],
        "freq_09m": [rows[f"freq__{w}__0.9m"]["A33_kg"] for w in ("0.3", "0.5", "0.8")],
        "freq_25m": [rows[f"freq__{w}__2.5m"]["A33_kg"] for w in ("0.3", "0.5", "0.8")],
    }
    spread_pct = {k: 100.0 * (max(v) - min(v)) / min(v) for k, v in spread.items()}
    return {
        "p5_observed_shift": observed,
        "p5_threshold": threshold,
        "p5_slack_factor": threshold / observed,
        "p5_reading": "threshold = mesh error + 1e-3, so P5 fails only if ice moves A33 "
                      "by more than the local discretisation error (>4.5 %)",
        "sensitivity_spread_pct": spread_pct,
        "doc_claim_flat_fraction_of_percent": spread_pct["freq_25m"] < 1.0,
    }


def doc_claims(subject: dict) -> dict:
    """Check the 'six sections + generated tables' claims of the delivered docs."""
    md = (SUBJECT / "docs/case_bem_ice_sphere_heave.md").read_text()
    numbers_md = SUBJECT_MD.read_text()
    sections = [s for s in md.splitlines() if s.startswith("## ")]
    hand_copied = []
    per = subject["rows"]
    for row in per:
        token = f"{row['A33_kg']:.6f}"
        if token in md or token in numbers_md:
            hand_copied.append(row["label"])
    return {
        "sections": sections,
        "section_count": len(sections),
        "has_six_elements": len(sections) >= 6,
        "numbers_file_lines": len(numbers_md.splitlines()),
        "raw_kg_values_transcribed_into_prose": hand_copied,
        "transcription_free": not hand_copied,
    }


def numbers_file_regenerated() -> dict:
    """Re-run the subject's own table renderer from the artifact and diff the output."""
    out = AUDIT / "_regenerated_numbers.md"
    figdir = SUBJECT / "artifacts/figures_regen"
    figdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(REPORT_TOOL), "--artifact", str(SUBJECT_JSON),
           "--numbers", str(out), "--figdir", str(figdir)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0 or not out.exists():
        return {"ran": False, "returncode": proc.returncode, "stderr": proc.stderr[-800:]}
    a = SUBJECT_MD.read_text().strip().splitlines()
    b = out.read_text().strip().splitlines()
    diff = [f"{i}: {x!r} != {y!r}" for i, (x, y) in enumerate(zip(a, b)) if x != y]
    return {
        "ran": True, "identical": a == b, "line_count_a": len(a), "line_count_b": len(b),
        "first_diffs": diff[:5],
        "reproduced_by_artifact_alone": a == b,
    }


def main() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    subject = json.loads(SUBJECT_JSON.read_text())
    rerun = json.loads(RERUN_JSON.read_text())
    quick = json.loads(QUICK_JSON.read_text())

    rows_a = {r["label"]: r for r in subject["rows"]}
    rows_b = {r["label"]: r for r in rerun["rows"]}
    worst = 0.0
    worst_at = None
    for label, row in rows_a.items():
        for key, val in row.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                other = rows_b[label][key]
                if isinstance(other, (int, float)) and not isinstance(other, bool):
                    denom = max(abs(val), 1e-300)
                    dev = abs(val - other) / denom
                    if dev > worst:
                        worst, worst_at = dev, f"{label}.{key}"
    rerun_check = {
        "sha256_subject": sha256(SUBJECT_JSON),
        "sha256_auditor_rerun": sha256(RERUN_JSON),
        "bit_identical_files": sha256(SUBJECT_JSON) == sha256(RERUN_JSON),
        "labels_match": list(rows_a) == list(rows_b),
        "rows": len(rows_b),
        "max_relative_deviation": worst,
        "max_relative_deviation_at": worst_at,
        "gates_identical": subject["verification"] == rerun["verification"],
        "rerun_command": "PYTHONPATH=src python -m tensorbem.benchmarks.ice_sphere_heave "
                         "--out /tmp/m17312_rerun.json --workers 8",
        "rerun_seconds": 76,
        "quick_rerun_primary_error_vs_lamb": quick["verification"]["primary_error_vs_lamb"],
        "quick_rerun_primary_error_vs_lamb_delivered": subject["verification"]["primary_error_vs_lamb"],
        "quick_rerun_relative_deviation": abs(
            quick["verification"]["primary_error_vs_lamb"]
            - subject["verification"]["primary_error_vs_lamb"])
        / subject["verification"]["primary_error_vs_lamb"],
        "quick_mode_emits_p3_false": quick["verification"]["gate_ladder_order_in_band (P3b)"] is False,
    }

    # independent reference-side anchors
    lamb = lamb_closed_form()
    deep = multipole(RADIUS, 20 * RADIUS, 1e-3, root=SUBJECT_SRC)
    lowk = multipole(RADIUS, 3 * RADIUS, 1e-3, root=SUBJECT_SRC)
    anchor = {
        "lamb_closed_form_kg": lamb,
        "lamb_matches_delivered": abs(lamb - subject["reference"]["value_kg"]) / lamb,
        "multipole_deep_20R_offset_vs_lamb": (deep - lamb) / lamb,
        "multipole_deep_within_0.05pct": abs((deep - lamb) / lamb) < 5e-4,
        "multipole_lowk_3R_offset_vs_lamb": (lowk - lamb) / lamb,
        "classical_3_8_image_correction_3R": free_surface_low_frequency_correction(3.0),
        "anchor_rel_diff": abs((lowk - lamb) / lamb - free_surface_low_frequency_correction(3.0))
        / free_surface_low_frequency_correction(3.0),
        "reading": "the multipole route reproduces the classical 3/8 (a/f)^3 image term at "
                   "d=3R (1.439 % vs 1.389 %, 3.6 % relative) and collapses onto Lamb at d=20R",
    }

    # ice wavenumber re-derived from the dispersion relation, not read from the code
    mu = RHO_ICE * 0.9 / RHO_W
    k_i = 0.5 ** 2 / (G - mu * 0.5 ** 2)
    row_primary = rows_a["ladder__352"]
    ice_check = {
        "rho_ice": RHO_ICE, "rho_water": RHO_W, "h_ice_m": 0.9, "omega": 0.5,
        "mu_m": mu,
        "k_i_recomputed_per_m": k_i,
        "k_iR_recomputed": k_i * RADIUS,
        "k_iR_delivered_row": row_primary["kR_ice"],
        "abs_diff": abs(k_i * RADIUS - row_primary["kR_ice"]),
        "k_open_recomputed": 0.25 / G,
        "k_open_ratio_to_ice": (0.25 / G) / k_i,
        "ice_wavenumber_effect_pct": 100.0 * (k_i / (0.25 / G) - 1.0),
    }

    summary = {
        "audit_mission": 17312,
        "subject_mission": 17275,
        "repo": "TensorBEM",
        "anchor_slug": "CASE-BEM-ICE-SPHERE-HEAVE",
        "subject_commit": SUBJECT_COMMIT,
        "subject_worktree": str(SUBJECT_WORKTREE),
        "audited_snapshot": str(SUBJECT.relative_to(REPO)),
        "auditor": "board-audit",
        "rerun": rerun_check,
        "criterion": verify_subject_vs_pool(subject),
        "gates": gates_recomputed(subject),
        "decomposition": decomposition(subject),
        "gate_weakness": gate_weakness(subject),
        "reference_anchors": anchor,
        "ice_wavenumber": ice_check,
        "docs": doc_claims(subject),
        "numbers_regenerated": numbers_file_regenerated(),
        "landing": {
            "on_origin_branch": False,
            "origin_branch_probe": "git ls-remote origin refs/heads/agent/m17275 -> (empty)",
            "main_contains_commit": False,
            "main_probe": "TensorBEM main 96f19f7; git merge-base --is-ancestor a35c7c0 HEAD -> NO",
            "benchmark_cases_rows_for_anchor": 0,
            "case_pool_status": "dispatched",
            "capability_delta_recorded": "no_substantive_delta",
            "reading": "delivery exists only in the mission worktree; the library trunk has "
                       "no copy and no criterion row existed before this audit",
        },
    }

    (AUDIT / "audit_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ("rerun", "gates", "decomposition")}, indent=1)[:3000])
    print("criterion:", {k: v for k, v in summary["criterion"].items() if k != "per_row"})
    print("anchors:", summary["reference_anchors"])
    print("ice:", summary["ice_wavenumber"])
    print("docs:", summary["docs"])
    print("numbers_regenerated:", summary["numbers_regenerated"])
    print("gate_weakness:", summary["gate_weakness"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())