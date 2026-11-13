#!/usr/bin/env python
"""Case-pool driver for ``BEM-URSELL-CYL-RAD``.

Ursell (1949) multipole solution for a submerged circular cylinder in deep
water, radiation problem (heave & sway).  The analytical reference is the
closed-form infinite system in ``tensorbem.analytic.ursell_cylinder_coefficients``.

Acceptance gate: relative error vs Ursell reference < 5 % for the case mesh.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Inject repo root so the driver works both in-tree and when copied to a node
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tensorbem.analytic import ursell_cylinder_added_mass, ursell_cylinder_coefficients
from tensorbem.benchmarks.ursell_cylinder import run as ursell_benchmark_run
from tensorbem.benchmarks.ursell_cylinder import submerged_circle_mesh

# ---------------------------------------------------------------------------
# Case constants
# ---------------------------------------------------------------------------
CASE_CODE = "BEM-URSELL-CYL-RAD"
CASE_POOL_ID = 201  # reserved for this case
MISSION_ID = 17530

RADIUS = 1.0          # m
SUBMERGENCE = 2.0     # m  (a/f = 0.5)
DENSITY = 1025.0      # kg/m³
GRAVITY = 9.80665     # m/s²

# Frequency band (ka values)
BAND_KA = (0.2, 0.4, 0.6, 0.8, 1.0)

# Mesh resolutions
CASE_PANELS = 128     # case mesh (the one that must pass the 5% gate)
COARSE_PANELS = 32    # coarse mesh for convergence check
FINE_PANELS = 256     # fine mesh for convergence check

PANEL_NODES = 24      # Gauss points per panel
SOURCE_COUNT = 64     # MFS source count for cross-check

# ---------------------------------------------------------------------------
# Reference computation
# ---------------------------------------------------------------------------

def reference(ka: float) -> dict:
    """Compute Ursell multipole reference for a given ka."""
    omega = math.sqrt(ka * GRAVITY / RADIUS)
    mu, nu = ursell_cylinder_coefficients(ka, RADIUS / SUBMERGENCE, terms=80)
    displaced = DENSITY * math.pi * RADIUS ** 2
    A = mu * displaced
    B = nu * displaced * omega
    return {
        "ka": ka,
        "omega": omega,
        "added_mass": A,
        "damping": B,
        "mu_dimless": mu,
        "nu_dimless": nu,
        "displaced_mass_per_unit_span": displaced,
    }


def reference_ladder(ka: float) -> dict:
    """Truncation convergence ladder for the Ursell multipole system."""
    terms_list = (20, 40, 60, 80, 100)
    results = []
    for terms in terms_list:
        mu, nu = ursell_cylinder_coefficients(ka, RADIUS / SUBMERGENCE, terms=terms)
        results.append({"terms": terms, "mu": mu, "nu": nu})

    deltas = {}
    for i in range(1, len(results)):
        t_prev = results[i - 1]["terms"]
        t_curr = results[i]["terms"]
        d_mu = abs(results[i]["mu"] - results[i - 1]["mu"]) / max(abs(results[i]["mu"]), 1e-30)
        d_nu = abs(results[i]["nu"] - results[i - 1]["nu"]) / max(abs(results[i]["nu"]), 1e-30)
        deltas[f"{t_prev}->{t_curr}"] = {"mu": d_mu, "nu": d_nu, "max": max(d_mu, d_nu)}

    return {
        "ka": ka,
        "steps": results,
        "deltas": deltas,
        "max_rel_delta": deltas.get(f"{terms_list[-2]}->{terms_list[-1]}", {"max": 0.0}),
    }


# ---------------------------------------------------------------------------
# Solver cell
# ---------------------------------------------------------------------------

def _panel_cell(ka: float, panels: int) -> dict:
    """Run one solver cell and return results with errors."""
    ref = reference(ka)
    result = ursell_benchmark_run(
        panels, RADIUS, SUBMERGENCE, ka, DENSITY, GRAVITY,
        panel_nodes=PANEL_NODES, source_count=SOURCE_COUNT,
    )
    return {
        "ka": ka,
        "panels": panels,
        "omega": result["omega"],
        "panel_added_mass": result["panel_added_mass"],
        "panel_damping": result["panel_damping"],
        "reference_added_mass": ref["added_mass"],
        "reference_damping": ref["damping"],
        "rel_error_added_mass": result["panel_added_mass_relative_error"],
        "rel_error_damping": result["panel_damping_relative_error"],
        "near_over_far": result["near_over_far_field_damping"],
        "heave_sway_diff": result["panel_heave_sway_added_mass_relative_difference"],
        "panel_residual": result["panel_relative_residual"],
        "condition_number": result["panel_condition_number"],
        "displaced_mass": ref["displaced_mass_per_unit_span"],
    }


# ---------------------------------------------------------------------------
# Task builder
# ---------------------------------------------------------------------------

def build_tasks() -> list[dict]:
    """Build the task list for the case pool run."""
    tasks = []
    for ka in BAND_KA:
        # Case mesh (must pass gate)
        tasks.append({"ka": ka, "panels": CASE_PANELS, "kind": "case"})
        # Coarse mesh (convergence check)
        tasks.append({"ka": ka, "panels": COARSE_PANELS, "kind": "coarse"})
        # Fine mesh (convergence check)
        tasks.append({"ka": ka, "panels": FINE_PANELS, "kind": "fine"})
    return tasks


# ---------------------------------------------------------------------------
# Gate evaluator
# ---------------------------------------------------------------------------

QUANTITIES = ("rel_error_added_mass", "rel_error_damping")
GATE_THRESHOLD = 0.05  # 5%


def evaluate(rows: list[dict], ref_ladders: dict) -> dict:
    """Evaluate the case pool results against acceptance criteria."""
    criteria = {}

    # G1: Case mesh error < 5% for all ka
    case_rows = [r for r in rows if r["kind"] == "case"]
    worst_case_err = 0.0
    for row in case_rows:
        for q in QUANTITIES:
            worst_case_err = max(worst_case_err, abs(row[q]))
    criteria["G1_case_error_below_5pct"] = {
        "pass": worst_case_err < GATE_THRESHOLD,
        "measured_worst": worst_case_err,
        "threshold": GATE_THRESHOLD,
    }

    # G2: Convergence (coarse > case > fine error)
    conv_ok = True
    for ka in BAND_KA:
        coarse = next((r for r in rows if r["kind"] == "coarse" and r["ka"] == ka), None)
        case = next((r for r in rows if r["kind"] == "case" and r["ka"] == ka), None)
        fine = next((r for r in rows if r["kind"] == "fine" and r["ka"] == ka), None)
        if coarse and case and fine:
            if not (coarse["rel_error_added_mass"] > case["rel_error_added_mass"] > fine["rel_error_added_mass"]):
                conv_ok = False
    criteria["G2_monotonic_convergence"] = {
        "pass": conv_ok,
        "note": "coarse > case > fine error for added mass at all ka",
    }

    # G3: Near/far field damping ratio close to 1
    worst_ratio_err = 0.0
    for row in case_rows:
        worst_ratio_err = max(worst_ratio_err, abs(row["near_over_far"] - 1.0))
    criteria["G3_near_far_identity"] = {
        "pass": worst_ratio_err < 0.03,
        "measured_worst": worst_ratio_err,
        "threshold": 0.03,
    }

    # G4: Heave/sway symmetry
    worst_hs_diff = 0.0
    for row in case_rows:
        worst_hs_diff = max(worst_hs_diff, row["heave_sway_diff"])
    criteria["G4_heave_sway_symmetry"] = {
        "pass": worst_hs_diff < 1e-8,
        "measured_worst": worst_hs_diff,
        "threshold": 1e-8,
    }

    # G5: Reference truncation converged
    ref_ok = True
    for ka in BAND_KA:
        ladder = ref_ladders.get(f"{ka:g}", {})
        max_delta = ladder.get("max_rel_delta", {}).get("max", 1.0)
        if max_delta > 1e-6:
            ref_ok = False
    criteria["G5_reference_truncation_converged"] = {
        "pass": ref_ok,
        "note": "80->100 term delta < 1e-6 at all ka",
    }

    all_pass = all(c["pass"] for c in criteria.values())
    return {
        "verdict": "PASS" if all_pass else "FAIL",
        "criteria": criteria,
        "n_case_rows": len(case_rows),
        "n_total_rows": len(rows),
    }


# ---------------------------------------------------------------------------
# Report & figure generation
# ---------------------------------------------------------------------------

def write_report(out_dir: Path, rows: list[dict], ref_ladders: dict, verdict: dict, meta: dict) -> Path:
    """Write a markdown report."""
    case_rows = sorted([r for r in rows if r["kind"] == "case"], key=lambda r: r["ka"])
    coarse_rows = sorted([r for r in rows if r["kind"] == "coarse"], key=lambda r: r["ka"])
    fine_rows = sorted([r for r in rows if r["kind"] == "fine"], key=lambda r: r["ka"])

    lines = [
        f"# Case Pool Report: {CASE_CODE}",
        "",
        "## Case Description",
        "",
        f"- **Case code**: {CASE_CODE}",
        f"- **Case pool ID**: {CASE_POOL_ID}",
        f"- **Mission ID**: {MISSION_ID}",
        f"- **Body**: Submerged circular cylinder in deep water",
        f"- **Radius**: {RADIUS} m",
        f"- **Submergence**: {SUBMERGENCE} m (a/f = {RADIUS/SUBMERGENCE})",
        f"- **Density**: {DENSITY} kg/m³",
        f"- **Gravity**: {GRAVITY} m/s²",
        f"- **Reference**: Ursell (1949) multipole solution, closed-form infinite system",
        f"- **Acceptance**: relative error < 5% vs Ursell reference",
        "",
        "## Reference Provenance",
        "",
        "- **Citation**: Ursell, F. (1949) On the heaving motion of a circular cylinder on the surface of a fluid. Quart. J. Mech. Appl. Math. 2:228-244.",
        "- **Method**: Multipole expansion reducing the body condition to an infinite linear system M·a = b, solved by truncation.",
        "- **Implementation**: `tensorbem.analytic.ursell_cylinder_coefficients`",
        "- **Truncation**: 80 terms, verified converged to < 1e-6 relative delta at 80→100 terms",
        "- **Independence**: The reference is a pure NumPy evaluation of the analytical system, sharing no code with the BEM solver kernel.",
        "",
        "## Verdict",
        "",
        f"**{verdict['verdict']}**",
        "",
        "| Gate | Pass | Measured | Threshold |",
        "|---|---|---|---|",
    ]
    for key, crit in verdict["criteria"].items():
        lines.append(
            f"| {key} | {'PASS' if crit['pass'] else 'FAIL'} | "
            f"{crit.get('measured_worst', crit.get('measured', '—'))} | "
            f"{crit.get('threshold', '—')} |"
        )

    lines += [
        "",
        "## Results: Case Mesh (128 panels)",
        "",
        "| ka | ω | A (panel) | A (ref) | ε_A | B (panel) | B (ref) | ε_B | near/far |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in case_rows:
        lines.append(
            f"| {row['ka']:.1f} | {row['omega']:.4f} | "
            f"{row['panel_added_mass']:.2f} | {row['reference_added_mass']:.2f} | "
            f"{row['rel_error_added_mass']:.4e} | "
            f"{row['panel_damping']:.2f} | {row['reference_damping']:.2f} | "
            f"{row['rel_error_damping']:.4e} | {row['near_over_far']:.5f} |"
        )

    lines += [
        "",
        "## Convergence Study",
        "",
        "| ka | panels | ε_A | ε_B |",
        "|---|---|---|---|",
    ]
    for ka in BAND_KA:
        for kind, label in [("coarse", "coarse"), ("case", "case"), ("fine", "fine")]:
            row = next((r for r in rows if r["ka"] == ka and r["kind"] == kind), None)
            if row:
                lines.append(
                    f"| {ka:.1f} | {row['panels']} ({label}) | "
                    f"{row['rel_error_added_mass']:.4e} | {row['rel_error_damping']:.4e} |"
                )

    lines += [
        "",
        "## Environment",
        "",
        f"- Python: {meta['python']}",
        f"- NumPy: {meta['numpy']}",
        f"- Host: {meta['host']}",
        f"- Machine: {meta['machine']}",
        "",
    ]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_figure(out_dir: Path, rows: list[dict], ref_ladders: dict) -> Path:
    """Write a comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case_rows = sorted([r for r in rows if r["kind"] == "case"], key=lambda r: r["ka"])
    coarse_rows = sorted([r for r in rows if r["kind"] == "coarse"], key=lambda r: r["ka"])
    fine_rows = sorted([r for r in rows if r["kind"] == "fine"], key=lambda r: r["ka"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: Added mass error vs ka
    ax = axes[0]
    kas = [r["ka"] for r in case_rows]
    ax.plot(kas, [r["rel_error_added_mass"] for r in case_rows], "o-", label="case (128)")
    ax.plot(kas, [r["rel_error_added_mass"] for r in coarse_rows], "s--", label="coarse (32)")
    ax.plot(kas, [r["rel_error_added_mass"] for r in fine_rows], "^:", label="fine (256)")
    ax.axhline(0.05, color="red", ls="--", alpha=0.5, label="5% gate")
    ax.set_xlabel("ka")
    ax.set_ylabel("relative error (added mass)")
    ax.set_title("Added mass error vs Ursell reference")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Damping error vs ka
    ax = axes[1]
    ax.plot(kas, [r["rel_error_damping"] for r in case_rows], "o-", label="case (128)")
    ax.plot(kas, [r["rel_error_damping"] for r in coarse_rows], "s--", label="coarse (32)")
    ax.plot(kas, [r["rel_error_damping"] for r in fine_rows], "^:", label="fine (256)")
    ax.axhline(0.05, color="red", ls="--", alpha=0.5, label="5% gate")
    ax.set_xlabel("ka")
    ax.set_ylabel("relative error (damping)")
    ax.set_title("Damping error vs Ursell reference")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Near/far field ratio
    ax = axes[2]
    ax.plot(kas, [r["near_over_far"] for r in case_rows], "o-", color="green")
    ax.axhline(1.0, color="black", ls="--", alpha=0.5, label="exact = 1")
    ax.set_xlabel("ka")
    ax.set_ylabel("near-field / far-field damping")
    ax.set_title("Radiation condition identity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "ursell_casepool.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_log(out_dir: Path, message: str) -> None:
    """Append a line to the run log."""
    (out_dir / "run.log").open("a", encoding="utf-8").write(message + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Case pool driver for {CASE_CODE}")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks()
    ref_ladders = {f"{ka:g}": reference_ladder(ka) for ka in BAND_KA}

    if not args.merge:
        if args.shards < 1 or not (0 <= args.shard < args.shards):
            raise SystemExit("invalid shard specification")
        mine = tasks[args.shard :: args.shards]
        if args.max_tasks:
            mine = mine[: args.max_tasks]
        rows = []
        for task in mine:
            started = time.time()
            cell = _panel_cell(task["ka"], task["panels"])
            cell["kind"] = task["kind"]
            rows.append(cell)
            message = (
                f"[shard {args.shard}/{args.shards}] {task['kind']} ka={task['ka']:.1f} "
                f"panels={cell['panels']} "
                f"ε_A={cell['rel_error_added_mass']:.4e} "
                f"ε_B={cell['rel_error_damping']:.4e} "
                f"({time.time() - started:.0f}s)"
            )
            print(message, flush=True)
            write_log(out_dir, message)
        (out_dir / f"shard_{args.shard}.json").write_text(
            json.dumps({"rows": rows}, indent=2), encoding="utf-8"
        )
        return 0

    # Merge phase
    rows = []
    for path in sorted(out_dir.glob("shard_*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
    if not rows:
        raise SystemExit(f"no shard results in {out_dir}")

    verdict = evaluate(rows, ref_ladders)
    meta = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "host": platform.node(),
        "machine": platform.machine(),
    }

    payload = {
        "case": {
            "case_code": CASE_CODE,
            "case_pool_id": CASE_POOL_ID,
            "mission_id": MISSION_ID,
            "repo": "TensorBEM",
            "tol_policy": "<5%",
            "body": "submerged circular cylinder, deep water, radiation (heave/sway)",
            "radius_m": RADIUS,
            "submergence_m": SUBMERGENCE,
            "a_over_f": RADIUS / SUBMERGENCE,
            "density_kg_m3": DENSITY,
            "gravity_m_s2": GRAVITY,
            "band_ka": list(BAND_KA),
            "case_panels": CASE_PANELS,
            "coarse_panels": COARSE_PANELS,
            "fine_panels": FINE_PANELS,
            "panel_nodes": PANEL_NODES,
        },
        "reference_provenance": {
            "citation": "Ursell, F. (1949) On the heaving motion of a circular cylinder on the surface of a fluid. Quart. J. Mech. Appl. Math. 2:228-244.",
            "method": "multipole expansion, closed-form infinite linear system M·a = b",
            "implementation": "tensorbem.analytic.ursell_cylinder_coefficients",
            "terms_used": 80,
            "truncation_ladder": {k: v["max_rel_delta"] for k, v in ref_ladders.items()},
            "independence": "pure NumPy evaluation of analytical system; shares no code with BEM solver kernel",
        },
        "environment": meta,
        "tasks": tasks,
        "reference": {f"{ka:g}": reference(ka) for ka in BAND_KA},
        "ref_ladder": ref_ladders,
        "rows": rows,
        "verdict": verdict,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "environment.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    report = write_report(out_dir, rows, ref_ladders, verdict, meta)
    figure = write_figure(out_dir, rows, ref_ladders)

    print(f"wrote {out_dir/'results.json'}, {report}, {figure}", flush=True)
    print(f"VERDICT: {verdict['verdict']}", flush=True)
    write_log(out_dir, f"merged {len(rows)} rows -> VERDICT {verdict['verdict']}")
    for key, crit in verdict["criteria"].items():
        write_log(out_dir, f"gate {key}: {'PASS' if crit['pass'] else 'FAIL'}")
    for key, crit in verdict["criteria"].items():
        print(f"  {key}: {'PASS' if crit['pass'] else 'FAIL'} measured={crit.get('measured_worst', crit.get('measured', '—'))}", flush=True)
    return 0 if verdict["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
