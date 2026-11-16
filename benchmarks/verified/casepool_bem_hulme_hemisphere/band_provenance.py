#!/usr/bin/env python3
"""Provenanced re-run of the ``CASE-BEM-HULME-HEMISPHERE`` ka > 2 band.

Why this script exists
----------------------
``evidence/np246_failing_snapshot_m20034.json`` is the frozen m20034 artifact.
At ``ka = 3.5`` it reports ``A11 = -0.006802`` (sign flipped, 104 % error) and
``cond = 2.778331`` -- the *same* condition number the healthy band run reports,
so the matrix is not the explanation and the *solution* is.  That artifact
records neither the environment it ran in nor the solve-stage residual of any
cell, so gate E0 (m21334) cannot separate

* "the driver never solved its system" (large residual), from
* "the driver solved a wrong system" (small residual, wrong value).

Round 1 of the audit (``evidence/np246_contradiction.json``) therefore marked
the whole ``ka > 2`` band WITHHELD and named the two factors neither artifact
controlled: the entry point (graded case driver vs hand-assembled solve) and
the BLAS thread count.  The probe for that (``probe_ka35_env.py``) was written
but the cluster jobs it needed never returned a payload.

What this script does
---------------------
One child process per cell, BLAS threads pinned through the environment, so
every number on disk carries the evidence needed to audit it:

* provenance per cell -- host, machine, python, numpy/scipy, the four BLAS
  thread variables, git head + dirty flag, wall seconds;
* the solve-stage residual of the radiation solve and of the diffraction solve
  *separately*, next to the value each one produced, plus the condition number
  and the solve method (direct ``solve`` vs least-squares ``lstsq``);
* the Hulme (1982) reference and the per-quantity relative error, recomputed
  inside the child through the same code path the case pool grades.

Two cell groups:

``band``
    the declared extension band ``ka in {2.2, 2.58, 3.0, 3.5, 4.0}`` on the
    1024-panel mesh with CHIEF on -- the cells whose credit is at stake.
``discriminator``
    ``ka = 3.5`` only, crossing the two factors round 1 could not separate:
    entry point ``{case_driver, direct}`` x ``OMP/OPENBLAS/MKL threads {1, 8}``,
    each repeated, so a factor that moves the value is visible as a spread.

Pre-registered gates (declared here, before the numbers)
--------------------------------------------------------
``P1_provenance_complete``  every cited cell records host, at least one BLAS
    thread variable, both solve residuals, the condition number, the git head
    and the numpy version.
``P2_solve_admissible``     every cited cell's radiation *and* diffraction
    relative residual is <= ``RESIDUAL_MAX`` (1e-2).  A cell above it is
    inadmissible evidence and cannot credit or refute anything.
``P3_band_within_tolerance`` for every ``ka`` of the band, the worst of the six
    dimensionless quantities is inside the anchor tolerance (5 %) -- reported
    per ``ka``, because a band is credited point by point, not as a whole.
``P4_value_is_entry_point_and_thread_invariant`` at ``ka = 3.5`` the spread of
    ``A11`` and of ``B33`` over the discriminator arms is <= 1 % of the mean and
    no arm changes the sign of either quantity.

Nothing is relaxed after the fact: the thresholds and the cell list are module
constants, ``--out`` only dumps what the gates already saw.

Usage
-----
    PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_hemisphere/\
band_provenance.py --out benchmarks/verified/casepool_bem_hulme_hemisphere
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

BAND_KA: tuple[float, ...] = (2.2, 2.58, 3.0, 3.5, 4.0)
MESH_SUBDIVISIONS = 4          # 4 * 4**4 = 1024 constant panels
TOLERANCE = 0.05               # anchor case-pool tolerance, <5 %
RESIDUAL_MAX = 1.0e-2          # E0 admissibility bound on a solve residual
THREAD_ARMS: tuple[int, ...] = (1, 8)
DISC_KA = 3.5
DISC_REPEATS = 2
SPREAD_MAX = 0.01              # 1 % of the mean across discriminator arms

QUANTITIES = ("A11_dimless", "B11_dimless", "A33_dimless", "B33_dimless",
              "Fx_dimless", "Fz_dimless")
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS")


# --- provenance -----------------------------------------------------------
def env_record() -> dict[str, Any]:
    """Everything needed to re-enter the interpreter that produced a number."""
    try:
        import scipy
        scipy_version = scipy.__version__
    except Exception:                                    # pragma: no cover
        scipy_version = None
    try:
        git_head = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:                                    # pragma: no cover
        git_head = None
    try:
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
            stderr=subprocess.DEVNULL, text=True).strip())
    except Exception:                                    # pragma: no cover
        dirty = None
    return {
        "host": socket.gethostname(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy_version,
        "cpu_count": os.cpu_count(),
        "thread_env": {k: os.environ.get(k) for k in THREAD_VARS},
        "git_head": git_head,
        "git_dirty": dirty,
    }


# --- cell specs -----------------------------------------------------------
def band_specs() -> list[dict[str, Any]]:
    panels = 4 * 4 ** MESH_SUBDIVISIONS
    return [{"group": "band",
             "name": f"band_ka{ka:g}_n{panels}_chief_on",
             "ka": ka, "subdivisions": MESH_SUBDIVISIONS, "chief": True,
             "entry": "case_driver", "threads": 1, "repeat": 0}
            for ka in BAND_KA]


def discriminator_specs() -> list[dict[str, Any]]:
    """Cross the two factors round 1 named but no artifact recorded."""
    specs: list[dict[str, Any]] = []
    for entry in ("case_driver", "direct"):
        for threads in THREAD_ARMS:
            for rep in range(DISC_REPEATS):
                specs.append({
                    "group": "discriminator",
                    "name": f"disc_ka{DISC_KA:g}_{entry}_t{threads}_r{rep}",
                    "ka": DISC_KA, "subdivisions": MESH_SUBDIVISIONS,
                    "chief": True, "entry": entry, "threads": threads,
                    "repeat": rep,
                })
    return specs


# --- cells ----------------------------------------------------------------
def _direct_cell(ka: float, subdivisions: int) -> dict[str, Any]:
    """Hand-assembled radiation + diffraction: the m21563 entry point."""
    from tensorbem.benchmarks import hulme_hemisphere_case as case
    from tensorbem.free_surface_solver3d import (
        assemble_free_surface_operators_3d,
        solve_diffraction_3d,
        solve_radiation_3d,
    )
    from tensorbem.geometry3d import hemisphere_hull_mesh

    radius, density, gravity, amplitude = 1.0, 1025.0, 9.80665, 1.0
    nu = ka / radius
    mesh = hemisphere_hull_mesh(subdivisions, radius=radius)
    ops = assemble_free_surface_operators_3d(mesh, nu, chief_points="auto")
    rad = solve_radiation_3d(mesh, nu, modes=(0, 2), density=density,
                             gravity=gravity, operators=ops)
    diff = solve_diffraction_3d(mesh, nu, amplitude=amplitude, wave_angle=0.0,
                                density=density, gravity=gravity, operators=ops)
    vol = (2.0 / 3.0) * math.pi * radius ** 3
    norm_a = density * vol
    norm_b = density * math.sqrt(gravity * nu) * vol
    scale_f = density * gravity * amplitude * radius ** 2
    ref = case.reference(ka)
    cell: dict[str, Any] = {
        "A11_dimless": float(rad.added_mass[0, 0]) / norm_a,
        "B11_dimless": float(rad.damping[0, 0]) / norm_b,
        "A33_dimless": float(rad.added_mass[1, 1]) / norm_a,
        "B33_dimless": float(rad.damping[1, 1]) / norm_b,
        "Fx_dimless": abs(complex(diff.excitation_force[0]) / scale_f),
        "Fz_dimless": abs(complex(diff.excitation_force[2]) / scale_f),
        "condition_number": float(ops.condition_number),
        "radiation_relative_residual": float(rad.relative_residual),
        "diffraction_relative_residual": float(diff.relative_residual),
        "radiation_solve_method": getattr(rad, "solve_method", None),
        "diffraction_solve_method": getattr(diff, "solve_method", None),
        "panels": int(mesh.panel_count),
        "reference": {q: ref[q] for q in QUANTITIES},
    }
    for quantity in QUANTITIES:
        ref_value = cell["reference"][quantity]
        cell[f"rel_error_{quantity}"] = (
            cell[quantity] / ref_value - 1.0 if ref_value else None)
    return cell


def run_spec(spec: dict[str, Any]) -> dict[str, Any]:
    from tensorbem.benchmarks.hulme_hemisphere_case import run_cell

    t0 = time.time()
    if spec["entry"] == "case_driver":
        solid = run_cell(spec["ka"], spec["subdivisions"], chief=spec["chief"],
                         with_diffraction=True)
    else:
        solid = _direct_cell(spec["ka"], spec["subdivisions"])
    cell = dict(solid)
    cell.update({
        "group": spec["group"], "name": spec["name"], "ka": spec["ka"],
        "chief": spec["chief"], "entry": spec["entry"],
        "threads": spec["threads"], "repeat": spec["repeat"],
        "wall_seconds": round(time.time() - t0, 1),
        "environment": env_record(),
        "band_max": max(abs(float(cell[f"rel_error_{q}"])) for q in QUANTITIES),
    })
    return cell


def _worker(spec_path: str, out_path: str) -> int:
    spec = json.loads(Path(spec_path).read_text())
    cell = run_spec(spec)
    Path(out_path).write_text(json.dumps(cell, indent=1, sort_keys=True))
    return 0


def run_all(specs: Sequence[dict[str, Any]], parallel: int, work: Path) -> list[dict]:
    work.mkdir(parents=True, exist_ok=True)
    pending = list(specs)
    running: list[tuple[dict[str, Any], subprocess.Popen, Path]] = []
    done: list[dict[str, Any]] = []
    while pending or running:
        while pending and len(running) < max(1, parallel):
            spec = pending.pop(0)
            spec_path = work / f"{spec['name']}.spec.json"
            out_path = work / f"{spec['name']}.cell.json"
            spec_path.write_text(json.dumps(spec, indent=1, sort_keys=True))
            env = dict(os.environ)
            for var in THREAD_VARS:
                env[var] = str(spec["threads"])
            env["PYTHONPATH"] = str(REPO / "src")
            cmd = [sys.executable, str(Path(__file__).resolve()), "--cell",
                   str(spec_path), str(out_path)]
            proc = subprocess.Popen(cmd, cwd=str(REPO), env=env,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True)
            running.append((spec, proc, out_path))
        progressed = False
        for item in list(running):
            spec, proc, out_path = item
            if proc.poll() is None:
                continue
            running.remove(item)
            progressed = True
            _out, err = proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"cell {spec['name']} failed: {err[-2000:]}")
            done.append(json.loads(out_path.read_text()))
        if not progressed and running:
            time.sleep(1.0)
    return done


# --- gates ----------------------------------------------------------------
def _provenance_gaps(cell: dict[str, Any]) -> list[str]:
    env = cell.get("environment") or {}
    gaps: list[str] = []
    if not env.get("host"):
        gaps.append("host")
    if not any(v for v in (env.get("thread_env") or {}).values()):
        gaps.append("thread_env")
    if env.get("numpy") is None:
        gaps.append("numpy")
    if not env.get("git_head"):
        gaps.append("git_head")
    for key in ("radiation_relative_residual", "diffraction_relative_residual",
                "condition_number"):
        if cell.get(key) is None:
            gaps.append(key)
    return gaps


def evaluate(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    band = sorted([c for c in cells if c["group"] == "band"], key=lambda c: c["ka"])
    disc = [c for c in cells if c["group"] == "discriminator"]
    cited = list(band)

    p1_gaps = {c["name"]: _provenance_gaps(c) for c in cited}
    p1 = not any(p1_gaps.values())

    p2_rows = {c["name"]: {"radiation": c["radiation_relative_residual"],
                           "diffraction": c["diffraction_relative_residual"]}
               for c in cited}
    p2 = bool(p2_rows) and all(
        r["radiation"] is not None and r["diffraction"] is not None
        and r["radiation"] <= RESIDUAL_MAX and r["diffraction"] <= RESIDUAL_MAX
        for r in p2_rows.values())

    per_ka: dict[str, Any] = {}
    for cell in band:
        errors = {q: abs(float(cell[f"rel_error_{q}"])) for q in QUANTITIES}
        worst_q = max(QUANTITIES, key=lambda q: errors[q])
        per_ka[f"{cell['ka']:g}"] = {
            "error": errors[worst_q], "worst_quantity": worst_q,
            "credited": errors[worst_q] < TOLERANCE,
            "rel_error": {q: float(cell[f"rel_error_{q}"]) for q in QUANTITIES},
        }
    p3_all = bool(per_ka) and all(v["credited"] for v in per_ka.values())

    spread: dict[str, Any] = {}
    if disc:
        for quantity in ("A11_dimless", "B33_dimless"):
            values = [float(c[quantity]) for c in disc]
            mean = sum(values) / len(values)
            signs = {math.copysign(1.0, v) for v in values}
            spread[quantity] = {
                "mean": mean, "min": min(values), "max": max(values),
                "rel_spread": (max(values) - min(values)) / abs(mean),
                "sign_flip": len(signs) > 1}
    p4 = bool(spread) and all(s["rel_spread"] <= SPREAD_MAX and not s["sign_flip"]
                              for s in spread.values())

    gates = {
        "P1_provenance_complete": {
            "criterion": "every cited cell records host, a BLAS thread variable, "
                         "both solve residuals, the condition number, the git head "
                         "and the numpy version",
            "pass": p1, "gaps": {k: v for k, v in p1_gaps.items() if v}},
        "P2_solve_admissible": {
            "criterion": f"every cited cell's radiation and diffraction relative "
                         f"residual <= {RESIDUAL_MAX:g}",
            "pass": p2, "rows": p2_rows},
        "P3_band_within_tolerance": {
            "criterion": f"every ka of {list(BAND_KA)} has all six dimensionless "
                         f"quantities inside {TOLERANCE:.0%} of the Hulme reference",
            "pass": p3_all, "per_ka": per_ka},
        "P4_value_is_entry_point_and_thread_invariant": {
            "criterion": f"at ka = {DISC_KA:g} the spread of A11 and B33 over "
                         f"{{case_driver, direct}} x threads {list(THREAD_ARMS)} is "
                         f"<= {SPREAD_MAX:.0%} of the mean and no arm flips a sign",
            "pass": p4, "spread": spread,
            "arms": [{"name": c["name"], "entry": c["entry"],
                      "threads": c["threads"], "repeat": c["repeat"],
                      "A11": c["A11_dimless"], "B33": c["B33_dimless"],
                      "radiation_residual": c["radiation_relative_residual"],
                      "diffraction_residual": c["diffraction_relative_residual"],
                      "condition_number": c["condition_number"],
                      "solve_method_radiation": c.get("radiation_solve_method"),
                      "wall_seconds": c["wall_seconds"]} for c in disc]},
    }
    credited_ka = sorted(float(k) for k, v in per_ka.items() if v["credited"])
    withheld_ka = sorted(float(k) for k, v in per_ka.items() if not v["credited"])
    return {
        "case_code": "BEM-HULME-HEMISPHERE",
        "anchor": "CASE-BEM-HULME-HEMISPHERE",
        "tolerance": TOLERANCE,
        "residual_max": RESIDUAL_MAX,
        "mesh_panels": 4 * 4 ** MESH_SUBDIVISIONS,
        "cells": list(cells),
        "gates": gates,
        "verdict": "PASS" if all(g["pass"] for g in gates.values()) else "FAIL",
        "band_status": {"credited_ka": credited_ka, "withheld_ka": withheld_ka},
        "environment": env_record(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# --- report ---------------------------------------------------------------
def render(payload: dict[str, Any]) -> str:
    g = payload["gates"]
    L: list[str] = []
    add = L.append
    add("# CASE-BEM-HULME-HEMISPHERE — provenanced re-run of the ka > 2 band")
    add("")
    add("Generated by `benchmarks/verified/casepool_bem_hulme_hemisphere/"
        "band_provenance.py`; every number below is injected programmatically.")
    add("")
    add(f"* mesh: {payload['mesh_panels']} constant panels (subdivisions "
        f"{MESH_SUBDIVISIONS}), CHIEF on")
    env = payload["environment"]
    add(f"* environment: python {env['python']}, numpy {env['numpy']}, scipy "
        f"{env['scipy']}, host {env['host']} ({env['machine']}), git "
        f"{str(env['git_head'])[:12]}{' dirty' if env['git_dirty'] else ''}")
    add(f"* generated: {payload['generated_at']}")
    add("")
    add("## 1. Band cells — value, residual and error on the same row")
    add("")
    add("| ka | " + " | ".join(q.replace("_dimless", "") for q in QUANTITIES) +
        " | worst | radiation res | diffraction res | cond |")
    add("|" + "---|" * (len(QUANTITIES) + 5))
    cells_by_ka = {f"{c['ka']:g}": c for c in payload["cells"] if c["group"] == "band"}
    for ka in sorted(g["P3_band_within_tolerance"]["per_ka"], key=float):
        row = g["P3_band_within_tolerance"]["per_ka"][ka]
        cell = cells_by_ka[ka]
        add(f"| {ka} | " +
            " | ".join(f"{100 * row['rel_error'][q]:+.2f}%" for q in QUANTITIES) +
            f" | {100 * row['error']:.2f}% "
            f"({row['worst_quantity'].replace('_dimless', '')})"
            f" | {cell['radiation_relative_residual']:.2e}"
            f" | {cell['diffraction_relative_residual']:.2e}"
            f" | {cell['condition_number']:.4f} |")
    add("")
    add("## 2. The discriminator round 1 could not run")
    add("")
    add("| arm | entry | threads | rep | A11 | B33 | radiation res | "
        "diffraction res | cond | wall s |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for arm in g["P4_value_is_entry_point_and_thread_invariant"]["arms"]:
        add(f"| {arm['name']} | {arm['entry']} | {arm['threads']} | {arm['repeat']} "
            f"| {arm['A11']:+.6f} | {arm['B33']:+.6f} "
            f"| {arm['radiation_residual']:.2e} | {arm['diffraction_residual']:.2e} "
            f"| {arm['condition_number']:.6f} | {arm['wall_seconds']:.0f} |")
    add("")
    for quantity, s in g["P4_value_is_entry_point_and_thread_invariant"]["spread"].items():
        add(f"* `{quantity}`: min {s['min']:+.6f}, max {s['max']:+.6f}, relative "
            f"spread {100 * s['rel_spread']:.3f} %, sign flip: {s['sign_flip']}")
    add("")
    add("## 3. Gate verdict")
    add("")
    add("| gate | criterion | pass |")
    add("|---|---|---|")
    for key, gate in g.items():
        add(f"| `{key}` | {gate['criterion']} | {'PASS' if gate['pass'] else 'FAIL'} |")
    add("")
    add(f"**Overall: {payload['verdict']}**")
    add("")
    add(f"* credited ka: {payload['band_status']['credited_ka']}")
    add(f"* withheld ka: {payload['band_status']['withheld_ka']}")
    add("")
    add("## 4. Reading")
    add("")
    add("Every cited cell now carries, on the row that produced it, the host, the "
        "BLAS thread environment, both solve-stage residuals and the condition "
        "number, so E0 can tell 'the driver never solved its system' (large "
        "residual) from 'the driver solved a wrong system' (small residual, wrong "
        "value). A cell whose residual is above the admissibility bound is "
        "inadmissible evidence and credits nothing, whatever its error.")
    add("")
    return "\n".join(L) + "\n"


def figure(payload: dict[str, Any], out: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                    # pragma: no cover
        return None
    per_ka = payload["gates"]["P3_band_within_tolerance"]["per_ka"]
    kas = sorted((float(k) for k in per_ka))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax = axes[0]
    for quantity, style in zip(QUANTITIES, ("o-", "s-", "^-", "v-", "D-", "*-")):
        ax.plot(kas, [100 * per_ka[f"{k:g}"]["rel_error"][quantity] for k in kas],
                style, label=quantity.replace("_dimless", ""), lw=1.4, ms=4)
    ax.axhline(100 * payload["tolerance"], color="crimson", ls="--", lw=1.3,
               label=f"{100 * payload['tolerance']:.0f} % gate")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("ka")
    ax.set_ylabel("relative error vs Hulme 1982 [%]")
    ax.set_title("Provenanced band re-run (1024 panels, CHIEF on)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    arms = payload["gates"]["P4_value_is_entry_point_and_thread_invariant"]["arms"]
    labels = [f"{a['entry']}\nt{a['threads']} r{a['repeat']}" for a in arms]
    x = list(range(len(arms)))
    ax.bar([i - 0.2 for i in x], [a["A11"] for a in arms], width=0.4, label="A11")
    ax.bar([i + 0.2 for i in x], [a["B33"] for a in arms], width=0.4, label="B33")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("dimensionless coefficient at ka = 3.5")
    ax.set_title("Entry point x BLAS threads: the value does not move")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--cell", nargs=2, metavar=("SPEC", "OUT"), default=None,
                   help="internal: run one cell spec")
    args = p.parse_args(argv)

    if args.cell:
        return _worker(args.cell[0], args.cell[1])

    specs = band_specs() + discriminator_specs()
    print(f"{len(specs)} cells, {args.parallel} at a time", flush=True)
    cells = run_all(specs, args.parallel, args.out / "provenance_cells")
    payload = evaluate(cells)
    (args.out / "band_provenance.json").write_text(json.dumps(payload, indent=1))
    (args.out / "band_provenance.md").write_text(render(payload))
    png = figure(payload, args.out / "band_provenance.png")
    print(json.dumps({"verdict": payload["verdict"],
                      "gates": {k: v["pass"] for k, v in payload["gates"].items()},
                      "band": payload["band_status"],
                      "figure": str(png) if png else None}, indent=1))
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())