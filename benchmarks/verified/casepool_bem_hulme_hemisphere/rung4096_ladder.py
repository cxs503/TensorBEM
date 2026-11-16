#!/usr/bin/env python3
"""CASE-BEM-HULME-HEMISPHERE -- measured 4096-panel rung at ka = 2.58 / 3.0.

Why this file exists
--------------------
``referee_lid_adjudication.md`` (session m23092) adjudicated the three
``referee_unresolved`` cells of ``radiation_capytaine_band_xval.json`` as
``agree``, but wrote its own limit down in section 5.6:

    ka = 2.58 and ka = 3.0 have only the 1024-panel rung.  That rung is inside
    tolerance (<= 2.31 %), but there is **no** 4096-panel measurement, so
    "these two cells are converged" is an inference, not a measurement.

This script converts that inference into a measurement.  It solves the same
case on the 4096-panel rung (``subdivisions = 5``) at the two ``ka`` values and
grades the rung against gates that were frozen in :data:`GATE_TEXT` **before**
any number of this session existed.

Every number in the rendered report is injected from ``results.json``; nothing
is transcribed by hand.

Usage (cluster: one shard per ka)::

    python rung4096_ladder.py cell --ka 2.58 --subdivisions 5 --out cells
    python rung4096_ladder.py merge --cells cells --out results.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.benchmarks import hulme_hemisphere_case as case

# --- design frozen before the first cell -----------------------------------
KA_TARGET = (2.58, 3.0)          # the two cells §5.6 called "inferred"
RUNGS = (4, 5)                   # 4 -> 1024 constant panels, 5 -> 4096
CASE_TOL = case.GATE             # 5 %, inherited from the case pool
GEOM_TOL = case.GEOM_GATE        # 2 %
RESIDUAL_TOL = 1e-2              # admissibility of a rung (solve-stage health)
COND_TOL = 50.0                  # a refined rung must not become near-singular
REF_TERMS = (96, 144, 192)       # reference truncation ladder
REF_DRIFT_TOL = CASE_TOL / 20.0  # 0.25 % -- 20x below the case gate
QUANTITIES = case.QUANTITIES
PHASE_KEYS = case.PHASE_KEYS

GATE_TEXT = {
    "P1_finest_rung_below_case_tolerance": (
        "at subdivisions = 5 (4096 panels) every one of the six quantities at "
        "ka = 2.58 and ka = 3.0 is within 5 % of the Hulme series"
    ),
    "P2_refinement_does_not_worsen": (
        "band-max |rel err| at 4096 panels <= band-max |rel err| at 1024 "
        "panels, for each ka"
    ),
    "P3_rung_is_admissible": (
        "every cell records radiation and diffraction relative residuals below "
        "1e-2 (a rung that cannot prove its own residual cannot be graded as "
        "physics; this closes audit gate A4 of the np246 contradiction audit)"
    ),
    "P4_reference_self_converged_at_these_ka": (
        "144 -> 192 term drift of the reference at each ka is at least 20x "
        "below the 5 % case gate"
    ),
    "P5_conditioning_does_not_explode": (
        "the free-surface operator condition number at 4096 panels stays below "
        "50 at both ka (a refined rung polluted by an irregular frequency "
        "would fail here)"
    ),
    "P6_geometry_precondition": (
        "|nabla_mesh/nabla - 1| < 2 % and |A_wp/(pi a^2) - 1| < 2 % on the "
        "4096-panel mesh"
    ),
}

REPO = Path(__file__).resolve().parents[3]


# --- cells -----------------------------------------------------------------
def cell_name(ka: float, subdivisions: int) -> str:
    return f"rung4096_ka{ka:g}_n{4 * 4 ** subdivisions}"


def run_one(ka: float, subdivisions: int) -> dict[str, Any]:
    cell = case.run_cell(ka, subdivisions, chief=True, with_diffraction=True)
    cell["name"] = cell_name(ka, subdivisions)
    cell["kind"] = "rung"
    return cell


def _worker(ka: float, subdivisions: int, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cell = run_one(ka, subdivisions)
    (out_dir / f"{cell['name']}.json").write_text(
        json.dumps(cell, indent=1, sort_keys=True)
    )
    print(json.dumps({"cell": cell["name"], "wall_seconds": cell["wall_seconds"]}))
    return 0


# --- reference ladder ------------------------------------------------------
def reference_ladder(ka: float) -> dict[str, Any]:
    vals = {n: case.reference(ka, n) for n in REF_TERMS}
    top = vals[REF_TERMS[-1]]
    drift = max(
        abs(vals[b][k] - vals[a][k]) / abs(top[k])
        for a, b in zip(REF_TERMS[:-1], REF_TERMS[1:])
        for k in QUANTITIES
    )
    return {
        "ka": ka,
        "terms": list(REF_TERMS),
        "max_rel_drift": drift,
        "values": {str(n): {k: vals[n][k] for k in QUANTITIES} for n in REF_TERMS},
    }


# --- verdict ---------------------------------------------------------------
def band_max(cell: dict[str, Any]) -> float:
    return max(abs(cell[f"rel_error_{q}"]) for q in QUANTITIES)


def evaluate(cells: list[dict[str, Any]], ladders: dict[str, Any]) -> dict[str, Any]:
    by_rung = {(c["ka"], c["subdivisions"]): c for c in cells}
    finest = [c for c in cells if c["subdivisions"] == RUNGS[-1]]
    gates: dict[str, Any] = {}

    if finest:
        worst = max(
            (abs(c[f"rel_error_{q}"]), q, c["ka"], c["subdivisions"])
            for c in finest
            for q in QUANTITIES
        )
        gates["P1_finest_rung_below_case_tolerance"] = {
            "measured": {
                "worst_rel_err": worst[0],
                "quantity": worst[1],
                "ka": worst[2],
                "subdivisions": worst[3],
            },
            "pass": bool(worst[0] < CASE_TOL),
        }
    else:
        # A missing rung is not a pass: it is `not_run` and the page says so.
        gates["P1_finest_rung_below_case_tolerance"] = {
            "measured": {"reason": "the 4096-panel rung produced no payload"},
            "pass": None,
        }

    rows = []
    for ka in KA_TARGET:
        coarse = by_rung.get((ka, RUNGS[0]))
        fine = by_rung.get((ka, RUNGS[1]))
        if coarse is None or fine is None:
            rows.append({"ka": ka, "coarse": None, "fine": None, "pass": None})
            continue
        bc, bf = band_max(coarse), band_max(fine)
        rows.append(
            {
                "ka": ka,
                "coarse_panels": coarse["panels"],
                "coarse_band_max": bc,
                "fine_panels": fine["panels"],
                "fine_band_max": bf,
                "improvement": bc - bf,
                "pass": bool(bf <= bc),
            }
        )
    gate_pass = (
        None if any(r["pass"] is None for r in rows)
        else bool(rows) and all(r["pass"] for r in rows)
    )
    gates["P2_refinement_does_not_worsen"] = {"measured": rows, "pass": gate_pass}

    resid = max(
        (c["radiation_relative_residual"], "radiation", c["ka"], c["subdivisions"])
        for c in cells
    )
    resid_d = max(
        (c["diffraction_relative_residual"], "diffraction", c["ka"], c["subdivisions"])
        for c in cells
    )
    worst_resid = max(resid, resid_d)
    gates["P3_rung_is_admissible"] = {
        "measured": {
            "worst_residual": worst_resid[0],
            "channel": worst_resid[1],
            "ka": worst_resid[2],
            "subdivisions": worst_resid[3],
            "warnings": sorted({w for c in cells for w in c["solve_warnings"]}),
        },
        "pass": bool(worst_resid[0] < RESIDUAL_TOL),
    }

    drift = max((l["max_rel_drift"], l["ka"]) for l in ladders.values())
    gates["P4_reference_self_converged_at_these_ka"] = {
        "measured": {"worst_drift": drift[0], "ka": drift[1], "tolerance": REF_DRIFT_TOL},
        "pass": bool(drift[0] < REF_DRIFT_TOL),
    }

    if finest:
        cond = max(
            (c["condition_number"], c["ka"]) for c in finest
        )
        gates["P5_conditioning_does_not_explode"] = {
            "measured": {"worst_condition_number": cond[0], "ka": cond[1], "panels": 4096},
            "pass": bool(cond[0] < COND_TOL),
        }
    else:
        gates["P5_conditioning_does_not_explode"] = {
            "measured": {"reason": "no 4096-panel rung"},
            "pass": None,
        }

    geom = [
        {
            "ka": c["ka"],
            "volume_error": abs(c["mesh_volume"] / c["nabla_exact"] - 1.0),
            "waterplane_error": abs(
                c["mesh_waterplane_area"] / (math.pi * case.RADIUS**2) - 1.0
            ),
        }
        for c in cells
        if c["subdivisions"] == RUNGS[-1]
    ]
    gates["P6_geometry_precondition"] = {
        "measured": geom,
        "pass": (
            None if not geom
            else all(
                g["volume_error"] < GEOM_TOL and g["waterplane_error"] < GEOM_TOL
                for g in geom
            )
        ),
    }

    failed = [k for k, v in gates.items() if v["pass"] is False]
    not_run = [k for k, v in gates.items() if v["pass"] is None]
    if failed:
        verdict = "FAIL"
    elif not_run:
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS"
    return {
        "gates": gates,
        "gate_text": GATE_TEXT,
        "failed": failed,
        "not_run": not_run,
        "verdict": verdict,
    }


# --- report ----------------------------------------------------------------
def _pct(x: float) -> str:
    return f"{100.0 * x:+.2f} %"


def render_report(results: dict[str, Any]) -> str:
    cells = sorted(results["cells"], key=lambda c: (c["ka"], c["subdivisions"]))
    verdict = results["verdict"]
    gates = results["gates"]
    lines = [
        "# CASE-BEM-HULME-HEMISPHERE — the 4096-panel rung at ka = 2.58 / 3.0 is now measured",
        "",
        f"Machine verdict: **{verdict}**"
        + ("" if verdict == "PASS" else (
            f" (failed: {', '.join(results['failed'])}; "
            f"not run: {', '.join(results.get('not_run', []))})")),
        "",
        "## 1. What was owed",
        "",
        "`referee_lid_adjudication.md` §5.6 recorded the debt in one sentence: at",
        "ka = 2.58 and ka = 3.0 the TensorBEM number existed only on the",
        "1024-panel rung, so *\"these two cells are converged\"* was an inference",
        "from one rung, not a measurement. This page pays that debt with the",
        "4096-panel rung at the same two `ka`.",
        "",
        "## 2. How",
        "",
        "One cell = `tensorbem.benchmarks.hulme_hemisphere_case.run_cell(ka,"
        " subdivisions)`; `subdivisions = 4` is 1024 constant panels,"
        "`subdivisions = 5` is 4096. The free-surface operators are assembled",
        "once per cell (`chief_points=\"auto\"`), radiation and diffraction share",
        "them, and the reference is the in-repo Hulme wave-free multipole series",
        f"truncated at {REF_TERMS[-1]} terms (self-convergence ladder"
        f" {REF_TERMS[0]}/{REF_TERMS[1]}/{REF_TERMS[2]} re-run in the same job).",
        "",
        "## 3. Results, per cell",
        "",
        "| ka | panels | cond | rad residual | diff residual | A11 err | B11 err | A33 err | B33 err | Fx err | Fz err | wall (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cells:
        lines.append(
            "| {ka:g} | {panels} | {cond:.2f} | {res_r:.2e} | {res_d:.2e} | ".format(
                ka=c["ka"],
                panels=c["panels"],
                cond=c["condition_number"],
                res_r=c["radiation_relative_residual"],
                res_d=c["diffraction_relative_residual"],
            )
            + " | ".join(_pct(c[f"rel_error_{q}"]) for q in QUANTITIES)
            + f" | {c['wall_seconds']:.0f} |"
        )
    lines += [
        "",
        "## 4. Ladder (the measurement that §5.6 asked for)",
        "",
        "| ka | 1024 band-max err | 4096 band-max err | improvement |",
        "|---:|---:|---:|---:|",
    ]
    for r in gates["P2_refinement_does_not_worsen"]["measured"]:
        if r.get("coarse_band_max") is None or r.get("fine_band_max") is None:
            if r.get("coarse_band_max") is not None:
                lines.append(
                    f"| {r['ka']:g} | {_pct(r['coarse_band_max'])} | "
                    "not measured | -- |"
                )
            continue
        lines.append(
            f"| {r['ka']:g} | {_pct(r['coarse_band_max'])} | "
            f"{_pct(r['fine_band_max'])} | {_pct(r['improvement'])} |"
        )
    lines += [
        "",
        "## 5. Gates (frozen in `GATE_TEXT` before the first cell)",
        "",
        "| gate | criterion | measured | verdict |",
        "|---|---|---|---|",
    ]
    for name, text in GATE_TEXT.items():
        g = gates[name]
        lines.append(
            f"| `{name}` | {text} | `{json.dumps(g['measured'], sort_keys=True)}` | "
            f"{'PASS' if g['pass'] else ('not run' if g['pass'] is None else '**FAIL**')} |"
        )
    lines += [
        "",
        "### 5.1 Measurement status of the missing rungs",
        "",
        results.get("not_run_reason") or "not recorded",
        "",
        "## 6. Limits",
        "",
        "1. The reference series and the solver share",
        "   `tensorbem.free_surface3d.wave_function_3d`, so this page measures",
        "   *convergence on the rung*, not the physical kernel; the third-party",
        "   axis for these `ka` is the lid-closed capytaine referee of",
        "   `referee_lid_adjudication.md`.",
        "2. `P2` is a non-worsening gate, not a Richardson order estimate: two",
        "   rungs cannot determine an observed order honestly.",
        "3. Wall clock is one core of the ecosystem cluster; it is recorded",
        "   because a rung that is expensive is a rung that gets skipped.",
        "",
        "## 7. Error statement (vs the Hulme series, per quantity)",
        "",
        "| key quantity | basis | error | < 5 % |",
        "|---|---|---:|---|",
    ]
    fine = [c for c in cells if c["subdivisions"] == RUNGS[-1]]
    if fine:
        worst_by_q = {
            q: max((abs(c[f"rel_error_{q}"]), c["ka"]) for c in fine)
            for q in QUANTITIES
        }
        for q in QUANTITIES:
            w = worst_by_q[q]
            lines.append(
                f"| `{q}` | Hulme 144-term series | {_pct(w[0])} (ka = {w[1]:g}) | "
                f"{'yes' if w[0] < CASE_TOL else '**no**'} |"
            )
    else:
        lines.append(
            "| _none_ | -- | the 4096-panel rung has no payload in this artifact, "
            "so no error is stated. See section 5. | **not measured** |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cell", help="run one (ka, mesh) cell")
    c.add_argument("--ka", type=float, required=True)
    c.add_argument("--subdivisions", type=int, required=True)
    c.add_argument("--out", type=Path, default=Path("cells"))

    m = sub.add_parser("merge", help="grade the cells and render the report")
    m.add_argument("--cells", type=Path, default=Path("cells"))
    m.add_argument("--out", type=Path, default=Path("results.json"))
    m.add_argument("--report", type=Path, default=Path("report_rung4096.md"))
    m.add_argument("--not-run-reason", default="")

    a = p.parse_args(argv)
    if a.cmd == "cell":
        return _worker(a.ka, a.subdivisions, a.out)

    cells = [
        json.loads(f.read_text()) for f in sorted(a.cells.glob("rung4096_*.json"))
    ]
    if not cells:
        print("no cells", file=sys.stderr)
        return 1
    ladders = {str(ka): reference_ladder(ka) for ka in KA_TARGET}
    verdict = evaluate(cells, ladders)
    results = {
        "case": "BEM-HULME-HEMISPHERE",
        "ka_target": list(KA_TARGET),
        "rungs": list(RUNGS),
        "cells": cells,
        "reference_ladders": ladders,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": os.uname().sysname + " " + os.uname().machine,
            "git_head": _git_head(),
        },
        **verdict,
        "not_run_reason": a.not_run_reason,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    a.out.write_text(json.dumps(results, indent=1, sort_keys=True))
    a.report.write_text(render_report(results))
    print(json.dumps({"verdict": results["verdict"], "failed": results["failed"]}))
    return 0 if results["verdict"] == "PASS" else 1


def _git_head() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "-C", shlex.quote(str(REPO)), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # pragma: no cover - provenance is best effort
        return ""


if __name__ == "__main__":
    raise SystemExit(main())

def note_unmeasured(results: dict[str, Any], reason: str) -> dict[str, Any]:
    """Attach the honest reason why a gate is ``not_run`` to the artifact."""
    results["not_run_reason"] = reason
    return results
