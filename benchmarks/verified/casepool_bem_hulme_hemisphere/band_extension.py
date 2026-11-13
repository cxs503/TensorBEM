#!/usr/bin/env python3
"""CASE-BEM-HULME-HEMISPHERE, round 2: predeclared band extension.

Why this file exists
--------------------
The case-pool anchor (``tensor_case_pool.id = 138``, ``case_code =
BEM-HULME-HEMISPHERE``) is declared on ``ka in (0.2, 0.5, 0.8, 1.0, 1.5, 2.0)``
and the committed case run (``results.json``) stops the band at ``ka = 2`` on
purpose, "well below the first irregular frequency ``ka ~ 2.58`` of this
waterplane".  That leaves two questions open, and this driver answers them in
one recorded run:

1. **Band extension.** Does the deep-water Kelvin Green-function BIE keep the
   same ``<5 %`` agreement against the Hulme (1982) wave-free multipole
   reference for ``ka in [2.2, 4.0]``?  The reference series is checked for
   self-convergence up to ``ka = 4`` in the same run (96/144/192 terms), so a
   degradation cannot be blamed on the reference by default.
2. **Irregular-frequency control.** ``ka = 2.58`` is inside the sweep.  With the
   interior CHIEF null-field equations on, the solve must stay regular and stay
   inside tolerance; the same cell is also run with ``chief = off`` as a control
   so the reader can see what the null-field equations buy.

Everything below is predeclared: the ka grid, the meshes, the gates and the
stopping rule were fixed before the first cell ran, and nothing is relaxed
afterwards.  A gate that fails is reported as a failure with its measured
number -- the band limit of the solver is a result, not an embarrassment.

Predeclared experiment
----------------------
Decision
    Is the ``<5 %`` agreement of TensorBEM's Kelvin-BEM with the Hulme (1982)
    reference a property of the *declared* band only, or does it survive up to
    ``ka = 4`` (three times the wavelength-to-radius ratio of the low end)?

Alternatives considered
    (a) push panels until the extended band passes (tested here);
    (b) keep the declared band and declare ``ka > 2`` NOT_SUPPORTED for this
        route (the honest-FAIL branch -- taken only if (a) fails);
    (c) switch route (Rankine + numerical beach) for the high band -- out of
        scope, the case is defined on the Kelvin Green-function route.

Hypothesis
    The error grows with ``ka`` (panel size in wavelengths) but stays inside
    5 % at 1024 panels up to ``ka = 4`` except possibly at the irregular
    frequency, where CHIEF is what saves the cell.

Metrics (unit velocity amplitude, ``exp(-i*omega*t)``)
    ``A11/(rho*nabla)``, ``B11/(rho*omega*nabla)``, ``A33/(rho*nabla)``,
    ``B33/(rho*omega*nabla)``, ``|Fx|/(rho*g*A*a^2)``, ``|Fz|/(rho*g*A*a^2)``
    plus both force phases.  ``nabla = 2*pi*a^3/3`` (exact, never the mesh
    volume, so a coefficient table cannot hide a geometry error).

Unit of analysis
    One ``(ka, mesh, chief)`` cell = one BEM assembly + one radiation solve
    (surge + heave) + one diffraction solve.  Deterministic; no seeds.

Gates (predeclared, not relaxed afterwards)
    E1 ``extension_band_error_below_5pct``
        |relative error| < 5 % for all six magnitudes at every ka of
        ``KA_EXTENDED`` on the 1024-panel mesh.
    E2 ``geometry_precondition``
        |nabla_mesh/nabla - 1| < 2 % and |A_wp,mesh/(pi a^2) - 1| < 2 % on that
        mesh (same precondition as the declared case; a coefficient table is
        unreadable if the body is mis-represented).
    E3 ``refinement_does_not_worsen_beyond_ka2``
        band-max error at 1024 panels <= band-max error at 256 panels, for
        ``ka = 3`` and ``ka = 4`` (the two rungs of the extension ladder).
    E4 ``reference_self_converged_up_to_ka4``
        reference 144 -> 192 term drift at ``ka = 4`` at least 20x below the
        5 % gate (i.e. < 2.5e-3), so the extension cannot be an artefact of a
        truncated multipole series.
    E5 ``chief_keeps_irregular_frequency_cell_inside_gate``
        at ``ka = 2.58`` the CHIEF-on cell is finite, ``cond`` finite and every
        magnitude inside 5 %; the CHIEF-off cell is reported next to it as the
        control (no pass/fail, it is a control).
    Stopping rule
        All cells of the task list are computed even if E1 fails at an early
        ka: the band limit needs the full grid to be located.  No threshold is
        touched after the first number exists.

Confounders / controls
    * panel resolution (256 vs 1024) -- E3;
    * reference truncation (96/144/192) -- E4;
    * CHIEF on/off at the irregular frequency -- E5;
    * the diffraction solve shares the assembled operator with the radiation
      solve, so an operator error would move both; the independent axis for
      that is Capytaine (``capytaine_excitation.json`` in this directory,
      produced by ``scripts/capytaine_hulme_excitation_ref.py``).

Reproduce
---------
    python benchmarks/verified/casepool_bem_hulme_hemisphere/band_extension.py \\
        --out artifacts/hulme_band_extension --parallel 7

``--parallel`` fans the deterministic cell list over worker processes; each
worker writes ``cells/<name>.json`` and the merge step is byte-stable.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.benchmarks import hulme_hemisphere_case as case

# --- predeclared design (frozen before the first cell) ---------------------
KA_EXTENDED = (2.2, 2.58, 3.0, 3.5, 4.0)
MESH_EXT = 4                     # 4 * 4**4 = 1024 constant panels
MESH_COARSE = 3                  # 4 * 4**3 = 256 panels
LADDER_EXT = ((3.0, 4.0), (4.0, 4.0))    # (ka, coarse-mesh ka rung) -> E3
CHIEF_CONTROL_KA = (2.58, 3.0)
REF_LADDER_TERMS = (96, 144, 192)         # E4 at ka = 4 (and at ka = 2.58)
E_GATE = case.GATE                        # 5 %, inherited from the case pool
GEOM_GATE = case.GEOM_GATE                # 2 %
QUANTITIES = case.QUANTITIES
PHASE_KEYS = case.PHASE_KEYS
REPO = Path(__file__).resolve().parents[3]


# --- cells -----------------------------------------------------------------
def cell_spec() -> list[dict[str, Any]]:
    """Deterministic cell list (kind, ka, subdivisions, chief)."""
    specs: list[dict[str, Any]] = []
    for ka in KA_EXTENDED:
        specs.append({"kind": "extension", "ka": ka, "subdivisions": MESH_EXT,
                      "chief": True})
    for ka, _rung in LADDER_EXT:
        specs.append({"kind": "ladder_coarse", "ka": ka, "subdivisions": MESH_COARSE,
                      "chief": True})
    for ka in CHIEF_CONTROL_KA:
        specs.append({"kind": "chief_off", "ka": ka, "subdivisions": MESH_EXT,
                      "chief": False})
    return specs


def cell_name(spec: dict[str, Any]) -> str:
    chief = "chief_on" if spec["chief"] else "chief_off"
    return f"{spec['kind']}_ka{spec['ka']:g}_n{4 * 4 ** spec['subdivisions']}_{chief}"


def run_one(spec: dict[str, Any]) -> dict[str, Any]:
    """One cell: BEM solve + reference + per-quantity relative errors."""
    cell = case.run_cell(
        spec["ka"], spec["subdivisions"], chief=spec["chief"], with_diffraction=True
    )
    cell.update({"kind": spec["kind"], "chief": spec["chief"],
                 "name": cell_name(spec)})
    return cell


def _worker(spec_path: str, cells_dir: str) -> int:
    spec = json.loads(Path(spec_path).read_text())
    cell = run_one(spec)
    out = Path(cells_dir) / f"{cell['name']}.json"
    out.write_text(json.dumps(cell, indent=1, sort_keys=True))
    return 0


# --- reference truncation ladder ------------------------------------------
def reference_ladder(ka: float) -> dict[str, Any]:
    """Reference self-convergence 96 -> 144 -> 192 terms at ``ka``."""
    vals = {n: case.reference(ka, n) for n in REF_LADDER_TERMS}
    top = vals[REF_LADDER_TERMS[-1]]
    deltas = {
        f"{a}->{b}": max(abs(vals[b][k] - vals[a][k]) / abs(top[k]) for k in QUANTITIES)
        for a, b in zip(REF_LADDER_TERMS[:-1], REF_LADDER_TERMS[1:])
    }
    return {"ka": ka, "terms": list(REF_LADDER_TERMS), "max_rel_delta": deltas,
            "values": {str(n): {k: vals[n][k] for k in QUANTITIES}
                       for n in REF_LADDER_TERMS}}


# --- verdict ---------------------------------------------------------------
def band_max(cell: dict[str, Any]) -> float:
    return max(abs(cell[f"rel_error_{q}"]) for q in QUANTITIES)


def evaluate(cells: list[dict[str, Any]], ladders: dict[str, Any]) -> dict[str, Any]:
    """Apply E1..E5 to the merged cells.  No relaxation after the fact."""
    ext = sorted([c for c in cells if c["kind"] == "extension"], key=lambda c: c["ka"])
    coarse = {c["ka"]: c for c in cells if c["kind"] == "ladder_coarse"}
    chief_off = {c["ka"]: c for c in cells if c["kind"] == "chief_off"}
    crit: dict[str, Any] = {}

    worst = max((abs(c[f"rel_error_{q}"]), q, c["ka"])
                for c in ext for q in QUANTITIES)
    per_ka = {f"{c['ka']:g}": {"band_max": band_max(c),
                               "worst_quantity": max(
                                   QUANTITIES,
                                   key=lambda q: abs(c[f"rel_error_{q}"]))}
              for c in ext}
    crit["E1_extension_band_error_below_5pct"] = {
        "criterion": f"|rel err| < {E_GATE:.0%} for all six magnitudes, every ka "
                     f"of {list(KA_EXTENDED)} on the {4 * 4 ** MESH_EXT}-panel mesh",
        "measured": worst[0], "worst_quantity": worst[1], "worst_ka": worst[2],
        "per_ka": per_ka,
        "pass": bool(worst[0] < E_GATE),
    }

    geom = max((abs(c["mesh_volume"] / c["nabla_exact"] - 1.0),
                abs(c["mesh_waterplane_area"] / (math.pi * case.RADIUS ** 2) - 1.0))
               for c in ext)
    crit["E2_geometry_precondition"] = {
        "criterion": f"|nabla_mesh/nabla-1| < {GEOM_GATE:.0%} and "
                     f"|A_wp/(pi a^2)-1| < {GEOM_GATE:.0%} on the extension mesh",
        "measured_volume_error": geom[0], "measured_waterplane_error": geom[1],
        "pass": bool(max(geom) < GEOM_GATE),
    }

    rungs = {}
    for ka, _ in LADDER_EXT:
        if ka in coarse:
            rungs[f"{ka:g}"] = {"coarse_mesh_max": band_max(coarse[ka]),
                                "extension_mesh_max": band_max(
                                    next(c for c in ext if c["ka"] == ka))}
    crit["E3_refinement_does_not_worsen_beyond_ka2"] = {
        "criterion": "band-max error at 1024 panels <= band-max error at 256 panels "
                     "for ka = 3 and ka = 4",
        "measured": rungs,
        "pass": bool(rungs) and all(
            r["extension_mesh_max"] <= r["coarse_mesh_max"] for r in rungs.values()
        ),
    }

    drift = ladders[f"{max(KA_EXTENDED):g}"]["max_rel_delta"]["144->192"]
    crit["E4_reference_self_converged_up_to_ka4"] = {
        "criterion": "reference 144->192 term drift at ka = 4 at least 20x below "
                     "the 5 % gate (< 2.5e-3)",
        "measured_drift": drift, "margin_factor": (0.2 * E_GATE) / max(drift, 1e-300),
        "pass": bool(drift * 20.0 < E_GATE),
    }

    irr = next((c for c in ext if abs(c["ka"] - 2.58) < 1e-9), None)
    off = chief_off.get(2.58)
    finite = bool(irr) and math.isfinite(irr["condition_number"]) and all(
        math.isfinite(irr[q]) for q in QUANTITIES)
    crit["E5_chief_keeps_irregular_frequency_cell_inside_gate"] = {
        "criterion": "at ka = 2.58 the CHIEF-on cell is finite and inside 5 % on "
                     "every magnitude; the CHIEF-off cell is the control",
        "measured_chief_on_band_max": band_max(irr) if irr else None,
        "measured_chief_on_cond": irr["condition_number"] if irr else None,
        "control_chief_off_band_max": band_max(off) if off else None,
        "control_chief_off_cond": off["condition_number"] if off else None,
        "pass": bool(finite and band_max(irr) < E_GATE),
    }

    return {"criteria": crit,
            "verdict": "PASS" if all(c["pass"] for c in crit.values()) else "FAIL"}


# --- writers ---------------------------------------------------------------
def write_report(out_dir: Path, cells: list[dict[str, Any]], ladders: dict[str, Any],
                 verdict: dict[str, Any], meta: dict[str, Any]) -> Path:
    ext = sorted([c for c in cells if c["kind"] == "extension"], key=lambda c: c["ka"])
    coarse = sorted([c for c in cells if c["kind"] == "ladder_coarse"],
                    key=lambda c: c["ka"])
    off = sorted([c for c in cells if c["kind"] == "chief_off"], key=lambda c: c["ka"])
    L: list[str] = []
    L.append("# CASE-BEM-HULME-HEMISPHERE -- round 2 band extension (ka 2.2 -> 4.0)")
    L.append("")
    L.append("Generated by `benchmarks/verified/casepool_bem_hulme_hemisphere/"
             "band_extension.py`. Regenerated numbers only; no number in this "
             "file is typed by hand.")
    L.append("")
    L.append(f"Environment: python {meta['python']}, numpy {meta['numpy']}, "
             f"host {meta['host']}, arch {meta['machine']}.")
    L.append("")
    L.append("## 1. Reference vs computed, extension band "
             f"({4 * 4 ** MESH_EXT} panels)")
    L.append("")
    L.append("| ka | A11 ref | got | err | B11 ref | got | err | A33 ref | got | err "
             "| B33 ref | got | err |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in ext:
        r = c["reference"]
        row = [f"{c['ka']:g}"]
        for key, err in (("A11_dimless", "rel_error_A11_dimless"),
                         ("B11_dimless", "rel_error_B11_dimless"),
                         ("A33_dimless", "rel_error_A33_dimless"),
                         ("B33_dimless", "rel_error_B33_dimless")):
            row += [f"{r[key]:.5f}", f"{c[key]:.5f}",
                    f"{c[err]:+.2%} {'OK' if abs(c[err]) < E_GATE else 'X'}"]
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append("| ka | Fx ref | got | err | Fz ref | got | err | dphase Fx (deg) "
             "| dphase Fz (deg) | cond | wall (s) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in ext:
        r = c["reference"]
        L.append("| " + " | ".join([
            f"{c['ka']:g}", f"{r['Fx_dimless']:.5f}", f"{c['Fx_dimless']:.5f}",
            f"{c['rel_error_Fx_dimless']:+.2%}",
            f"{r['Fz_dimless']:.5f}", f"{c['Fz_dimless']:.5f}",
            f"{c['rel_error_Fz_dimless']:+.2%}",
            f"{c['delta_phase_Fx_deg']:+.3f}", f"{c['delta_phase_Fz_deg']:+.3f}",
            f"{c['condition_number']:.2f}", f"{c['wall_seconds']:.1f}"]) + " |")
    L.append("")
    L.append(f"## 2. Mesh control ({4 * 4 ** MESH_COARSE} panels, same ka)")
    L.append("")
    L.append("| ka | panels | band-max err | A11 err | A33 err | B33 err |")
    L.append("|---|---|---|---|---|---|")
    for c in coarse:
        L.append("| " + " | ".join([
            f"{c['ka']:g}", str(c["panels"]), f"{band_max(c):.2%}",
            f"{c['rel_error_A11_dimless']:+.2%}", f"{c['rel_error_A33_dimless']:+.2%}",
            f"{c['rel_error_B33_dimless']:+.2%}"]) + " |")
    L.append("")
    L.append("## 3. CHIEF control at / near the irregular frequency")
    L.append("")
    L.append("| ka | chief | cond | band-max err | A33 | B33 |")
    L.append("|---|---|---|---|---|---|")
    for c in sorted(ext + off, key=lambda c: (c["ka"], not c["chief"])):
        if c["ka"] not in CHIEF_CONTROL_KA:
            continue
        L.append("| " + " | ".join([
            f"{c['ka']:g}", "on" if c["chief"] else "off",
            f"{c['condition_number']:.2f}", f"{band_max(c):.2%}",
            f"{c['A33_dimless']:.5f}", f"{c['B33_dimless']:.5f}"]) + " |")
    L.append("")
    L.append("## 4. Reference truncation ladder")
    L.append("")
    L.append("| ka | 96->144 | 144->192 |")
    L.append("|---|---|---|")
    for k, v in sorted(ladders.items(), key=lambda kv: float(kv[0])):
        L.append("| " + " | ".join([k, f"{v['max_rel_delta']['96->144']:.3e}",
                                    f"{v['max_rel_delta']['144->192']:.3e}"]) + " |")
    L.append("")
    L.append("## 5. Gates")
    L.append("")
    L.append("| gate | criterion | measured | verdict |")
    L.append("|---|---|---|---|")
    for name, c in verdict["criteria"].items():
        m = c.get("measured", c.get("measured_drift"))
        if isinstance(m, float):
            m = f"{m:.3e}" if m < 1e-2 else f"{m:.4f}"
        L.append(f"| `{name}` | {c['criterion']} | {m} | "
                 f"{'PASS' if c['pass'] else 'FAIL'} |")
    L.append("")
    L.append(f"**Overall: {verdict['verdict']}**")
    L.append("")
    path = out_dir / "report.md"
    path.write_text("\n".join(L) + "\n")
    return path


def write_figure(out_dir: Path, cells: list[dict[str, Any]]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = sorted([c for c in cells if c["kind"] == "extension"], key=lambda c: c["ka"])
    ka = [c["ka"] for c in ext]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for key, label in (("A33_dimless", "A33/(rho*nabla)"),
                       ("B33_dimless", "B33/(rho*omega*nabla)")):
        ax[0].plot(ka, [c["reference"][key] for c in ext], "o-", label=f"{label} ref")
        ax[0].plot(ka, [c[key] for c in ext], "x--", label=f"{label} BEM")
    ax[0].set_xlabel("ka")
    ax[0].set_ylabel("heave coefficient")
    ax[0].set_title("Heave coefficients, extension band")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    for q, style in (("A11_dimless", "o-"), ("B11_dimless", "s-"),
                     ("A33_dimless", "^--"), ("B33_dimless", "v--"),
                     ("Fx_dimless", "d-"), ("Fz_dimless", "P-")):
        ax[1].plot(ka, [100 * abs(c[f"rel_error_{q}"]) for c in ext], style,
                   label=q.split("_")[0])
    ax[1].axhline(100 * E_GATE, color="k", ls=":", label="5 % gate")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("ka")
    ax[1].set_ylabel("|relative error| (%)")
    ax[1].set_title("Reference vs BEM (1024 panels)")
    ax[1].legend(fontsize=7, ncol=2)
    ax[1].grid(alpha=0.3, which="both")

    ax[2].semilogy(ka, [c["condition_number"] for c in ext], "o-", label="CHIEF on")
    for c in cells:
        if c["kind"] == "chief_off":
            ax[2].semilogy([c["ka"]], [c["condition_number"]], "rx",
                           label="CHIEF off (control)")
    ax[2].set_xlabel("ka")
    ax[2].set_ylabel("condition number")
    ax[2].set_title("Operator conditioning; ka = 2.58 = first irregular frequency")
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = out_dir / "figure.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# --- driver ----------------------------------------------------------------
def build(out_dir: Path, parallel: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(exist_ok=True)
    specs = cell_spec()
    todo = [s for s in specs if not (cells_dir / f"{cell_name(s)}.json").exists()]
    spec_dir = out_dir / "specs"
    spec_dir.mkdir(exist_ok=True)
    for s in todo:
        (spec_dir / f"{cell_name(s)}.json").write_text(json.dumps(s))
    if todo:
        ctx = mp.get_context("fork")
        with ctx.Pool(min(max(1, parallel), len(todo))) as pool:
            pool.starmap(_worker,
                         [(str(spec_dir / f"{cell_name(s)}.json"), str(cells_dir))
                          for s in todo])
    cells = [json.loads(p.read_text()) for p in sorted(cells_dir.glob("*.json"))]
    ladders = {f"{ka:g}": reference_ladder(ka)
               for ka in (max(KA_EXTENDED), 2.58)}
    verdict = evaluate(cells, ladders)
    meta = {"python": platform.python_version(), "numpy": np.__version__,
            "host": platform.node(), "machine": platform.machine(),
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": _git_head()}
    (out_dir / "results.json").write_text(json.dumps(
        {"case_code": "BEM-HULME-HEMISPHERE", "study": "band_extension",
         "ka_extended": list(KA_EXTENDED), "mesh_panels": 4 * 4 ** MESH_EXT,
         "gate": E_GATE, "cells": cells, "reference_ladder": ladders,
         "verdict": verdict, "metadata": meta,
         "repro_cmd": "python benchmarks/verified/casepool_bem_hulme_hemisphere/"
                      f"band_extension.py --out {out_dir} --parallel 7"},
        indent=1, sort_keys=True))
    write_report(out_dir, cells, ladders, verdict, meta)
    write_figure(out_dir, cells)
    print(json.dumps({k: v["pass"] for k, v in verdict["criteria"].items()}, indent=1))
    print("VERDICT", verdict["verdict"])
    return 0 if verdict["verdict"] == "PASS" else 1


def _git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="artifacts/hulme_band_extension")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--spec", action="store_true", help="print the cell list and exit")
    args = p.parse_args(argv)
    if args.spec:
        print(json.dumps({"cells": cell_spec(), "ka_extended": list(KA_EXTENDED),
                          "gate": E_GATE, "geometry_gate": GEOM_GATE}, indent=2))
        return 0
    return build(Path(args.out).resolve(), args.parallel)


if __name__ == "__main__":
    raise SystemExit(main())