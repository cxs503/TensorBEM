#!/usr/bin/env python3
"""Referee robustness for CASE-BEM-HULME-HEMISPHERE: the three unresolved cells.

Why this file exists
--------------------
``radiation_capytaine_band_xval.json`` (the third-party axis of this anchor)
registers ``**Overall: FAIL**`` for a reason that is entirely on the *referee*
side: ``G2_referee_self_converged`` -- capytaine's own 1024 -> 4096 panel drift
-- exceeds 5 % at ``ka = 2.58`` (37.4 %), ``3.0`` (6.8 %) and ``4.0`` (8.6 %), so
``G3_attribution_defined`` cannot vote on those cells and returns
``referee_unresolved``.  A ``G5_three_way_agreement_in_band`` gate was left
``NOT_RUN`` because no TensorBEM cell existed in band.

That file's own diagnosis (its ``Limitations`` section) names two candidate
causes and does not separate them:

* ``H_irregular`` -- the ``ka = 2.58`` cell sits on a *discrete irregular
  frequency* of the referee's panel system (its ``B33`` error is +108 % there
  while every neighbouring cell is < 6 %).  Irregular frequencies are a property
  of the discretised interior Dirichlet problem: they **move with the mesh** and
  are confined to a narrow band of ``ka``.
* ``H_under_resolved`` -- the referee is simply too coarse above ``ka ~ 3`` on
  all quantities, so its drift is a general resolution deficit.

This producer separates them with two cheap, pre-registered probes, and then
closes the missing in-band gate.

Pre-registered gates (declared *before* the run; the earlier file's gates
``G1..G5`` are **not** touched -- a gate is never relaxed after the fact, so any
new metric is a *new* gate with its own name and its own justification)
-----------------------------------------------------------------------
* ``R1_irregular_localised`` -- if ``H_irregular`` holds, the referee's error
  versus the Hulme series must form a *spike*: at ``ka = 2.58`` the error
  exceeds 20 % on at least one magnitude, while every probe at
  ``|ka - 2.58| >= 0.08`` stays under 10 % on the same magnitude.  A broad
  under-resolution deficit fails this gate (the error would be large everywhere).
* ``R2_irregular_mesh_dependent`` -- the ``ka`` at which the spike sits must
  **move** when the referee mesh changes (1024 -> 4096 panels) by at least one
  probe spacing.  A physical resonance of the hemisphere cannot do that; an
  irregular frequency of the discretisation must.
* ``R3_referee_scaled_drift`` -- referee self-drift (1024 -> 4096 panels)
  measured in the *physical* scales ``|dA|/(rho*nabla)`` and
  ``|dB|/(rho*omega*nabla)`` <= 1e-2 at every ``ka``.  Justification: ``B33``
  falls to ``0.0219`` at ``ka = 4`` where the exciting/radiating magnitudes
  cancel, so a relative drift there is dominated by the smallness of the
  denominator -- the same metric defect that the diffraction anchor
  (``casepool_bem_hulme_diff_verif``) already fixed on the force side.  The
  relative metric of the earlier file (``G2``) is still reported alongside, and
  its FAIL verdict stands unchanged.
* ``R4_in_band_three_way`` -- at the accepted band ``ka = 1.0, 2.0``,
  TensorBEM(1024) and capytaine(4096) must both agree with the Hulme series
  within 5 % on all four magnitudes, and with each other within 5 %.  This is
  the ``G5`` gate of the earlier file, run here for the first time.

Usage::

    PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_hemisphere/\\
        referee_robustness.py --part sweep --ka-start 2.30 --ka-stop 2.70
    PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_hemisphere/\\
        referee_robustness.py --part merge      # writes the JSON + md
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "referee_robustness.json"
OUT_DOC = HERE / "referee_robustness.md"

RHO = 1025.0
GRAVITY = 9.80665
RADIUS = 1.0
NABLA = 2.0 / 3.0 * math.pi * RADIUS**3
MAGNITUDES = ("A11", "B11", "A33", "B33")

SUBDIVISIONS_TO_PANELS = {3: 256, 4: 1024, 5: 4096}
REF_N_TERMS = 144

#: Irregular-frequency probe: a fine sweep through the ``ka = 2.58`` cell.  The
#: spacing (0.04) is small against the observed 37 % drift, and the design has
#: four points on each side of the suspect cell at two meshes.
SWEEP_CENTRE = 2.58
SWEEP_HALF_WIDTH = 0.28
SWEEP_STEP = 0.04
SWEEP_MESHES = (4, 5)

#: Referee drift band (same ``ka`` grid as the earlier file's ``G2``).
DRIFT_KA: tuple[float, ...] = (1.0, 2.0, 2.2, 2.58, 3.0, 3.5, 4.0)
DRIFT_MESHES = (4, 5)

#: In-band three-way cell (``G5`` of the earlier file).
IN_BAND_KA: tuple[float, ...] = (1.0, 2.0)
IN_BAND_TENSORBEM_SUBDIVISIONS = 4
IN_BAND_REFEREE_SUBDIVISIONS = 5

#: Pre-registered thresholds.
R1_SPIKE = 0.20
R1_NEIGHBOUR = 0.10
R1_EXCLUSION = 0.08
R2_MIN_SHIFT = SWEEP_STEP
R3_SCALED_DRIFT = 1.0e-2
R4_AGREEMENT = 0.05


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
def hulme_reference(ka: float, n_terms: int = REF_N_TERMS) -> dict[str, float]:
    from tensorbem.benchmarks.hulme_hemisphere_case import reference as _ref

    got = _ref(float(ka), int(n_terms))
    return {m: float(got[f"{m}_dimless"]) for m in MAGNITUDES}


def capytaine_available() -> tuple[bool, str]:
    try:
        cpt = importlib.import_module("capytaine")
    except Exception as exc:  # pragma: no cover
        return False, f"capytaine import failed: {exc}"
    return True, f"capytaine {getattr(cpt, '__version__', 'unknown')}"


def capytaine_radiation(subdivisions: int, ka: float) -> dict[str, Any]:
    """Same normalisation as the case: ``A/(rho*nabla)``, ``B/(rho*omega*nabla)``."""
    import capytaine as cpt

    from tensorbem.geometry3d import hemisphere_hull_mesh

    mesh = hemisphere_hull_mesh(int(subdivisions), radius=RADIUS)
    body = cpt.FloatingBody(
        mesh=cpt.Mesh(vertices=np.asarray(mesh.vertices, dtype=float),
                      faces=np.asarray(mesh.faces, dtype=np.int64)),
        dofs=cpt.rigid_body_dofs(rotation_center=(0.0, 0.0, 0.0)),
    )
    solver = cpt.BEMSolver()
    omega = math.sqrt(GRAVITY * ka / RADIUS)
    out: dict[str, Any] = {
        "subdivisions": int(subdivisions),
        "panels": int(np.asarray(mesh.faces).shape[0]),
    }
    t0 = time.time()
    for dof, ka_key, kb_key in (("Surge", "A11", "B11"), ("Heave", "A33", "B33")):
        problem = cpt.RadiationProblem(
            body=body, wavenumber=ka / RADIUS, water_depth=math.inf,
            radiating_dof=dof, rho=RHO, g=GRAVITY,
        )
        result = solver.solve(problem)
        out[ka_key] = float(result.added_mass[dof]) / (RHO * NABLA)
        out[kb_key] = float(result.radiation_damping[dof]) / (RHO * omega * NABLA)
    out["seconds"] = time.time() - t0
    # capytaine's own irregular-frequency estimate for this discretisation: this
    # is the referee *declaring* where its discrete interior-Dirichlet pole sits,
    # so it is independent evidence, not something this script infers.
    try:
        out["first_irregular_wavenumber"] = float(body.first_irregular_frequency_estimate)
        out["first_irregular_ka"] = float(body.first_irregular_frequency_estimate) / RADIUS
    except Exception:  # pragma: no cover - optional diagnostic
        pass
    out["minimal_computable_wavelength_panels"] = (
        float(body.minimal_computable_wavelength / (2.0 * math.pi / ka)))
    return out


def tensorbem_radiation(subdivisions: int, ka: float) -> dict[str, Any]:
    from tensorbem.free_surface_solver3d import solve_radiation_3d
    from tensorbem.geometry3d import hemisphere_hull_mesh

    mesh = hemisphere_hull_mesh(int(subdivisions), radius=RADIUS)
    res = solve_radiation_3d(mesh, nu=ka / RADIUS, modes=(0, 2),
                             density=RHO, gravity=GRAVITY)
    omega = math.sqrt(GRAVITY * ka / RADIUS)
    a = np.asarray(res.added_mass, dtype=float)
    b = np.asarray(res.damping, dtype=float)
    return {
        "subdivisions": int(subdivisions),
        "panels": int(np.asarray(mesh.faces).shape[0]),
        "A11": float(a[0, 0]) / (RHO * NABLA),
        "B11": float(b[0, 0]) / (RHO * omega * NABLA),
        "A33": float(a[1, 1]) / (RHO * NABLA),
        "B33": float(b[1, 1]) / (RHO * omega * NABLA),
        "relative_residual": float(res.relative_residual),
        "condition_number": float(res.condition_number),
    }


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------
def sweep_kas() -> list[float]:
    n = int(round(2 * SWEEP_HALF_WIDTH / SWEEP_STEP))
    return [round(SWEEP_CENTRE - SWEEP_HALF_WIDTH + i * SWEEP_STEP, 10) for i in range(n + 1)]


def run_sweep(ka_list: Sequence[float]) -> dict[str, Any]:
    rows = []
    for ka in ka_list:
        ref = hulme_reference(ka)
        row: dict[str, Any] = {"ka": ka, "reference": ref}
        for sub in SWEEP_MESHES:
            cell = capytaine_radiation(sub, ka)
            errs = {m: (cell[m] - ref[m]) / ref[m] for m in MAGNITUDES}
            cell["rel_error"] = errs
            row[f"sub{sub}"] = cell
        rows.append(row)
        print(json.dumps({"ka": ka,
                          "err4": {m: round(rows[-1]["sub4"]["rel_error"][m], 4) for m in MAGNITUDES},
                          "err5": {m: round(rows[-1]["sub5"]["rel_error"][m], 4) for m in MAGNITUDES}}),
              flush=True)
    return {"probe": "irregular_frequency_sweep",
            "switches": {"centre": SWEEP_CENTRE, "half_width": SWEEP_HALF_WIDTH,
                         "step": SWEEP_STEP, "meshes": list(SWEEP_MESHES)},
            "rows": rows}


def run_drift() -> dict[str, Any]:
    rows = []
    for ka in DRIFT_KA:
        ref = hulme_reference(ka)
        row: dict[str, Any] = {"ka": ka, "reference": ref}
        for sub in DRIFT_MESHES:
            cell = capytaine_radiation(sub, ka)
            cell["rel_error"] = {m: (cell[m] - ref[m]) / ref[m] for m in MAGNITUDES}
            cell["scaled_error"] = {m: cell[m] - ref[m] for m in MAGNITUDES}
            row[f"sub{sub}"] = cell
        fine, coarse = row[f"sub{DRIFT_MESHES[-1]}"], row[f"sub{DRIFT_MESHES[0]}"]
        row["self_drift_relative"] = {
            m: abs(fine[m] - coarse[m]) / abs(fine[m]) for m in MAGNITUDES}
        row["self_drift_scaled"] = {m: abs(fine[m] - coarse[m]) for m in MAGNITUDES}
        rows.append(row)
        print(json.dumps({"ka": ka,
                          "rel": {m: round(row["self_drift_relative"][m], 4) for m in MAGNITUDES},
                          "scaled": {m: round(row["self_drift_scaled"][m], 5) for m in MAGNITUDES}}),
              flush=True)
    return {"probe": "referee_self_drift", "meshes": list(DRIFT_MESHES), "rows": rows}


def run_in_band(ka_list: Sequence[float] | None = None) -> dict[str, Any]:
    rows = []
    for ka in (IN_BAND_KA if ka_list is None else ka_list):
        ref = hulme_reference(ka)
        bem = tensorbem_radiation(IN_BAND_TENSORBEM_SUBDIVISIONS, ka)
        ref_c = capytaine_radiation(IN_BAND_REFEREE_SUBDIVISIONS, ka)
        row = {
            "ka": ka,
            "reference": ref,
            "tensorbem": bem,
            "capytaine": ref_c,
            "tensorbem_rel_error": {m: (bem[m] - ref[m]) / ref[m] for m in MAGNITUDES},
            "capytaine_rel_error": {m: (ref_c[m] - ref[m]) / ref[m] for m in MAGNITUDES},
            "code_vs_code": {m: abs(bem[m] - ref_c[m]) / abs(ref_c[m]) for m in MAGNITUDES},
        }
        rows.append(row)
        print(json.dumps({"ka": ka,
                          "bem": {m: round(row["tensorbem_rel_error"][m], 4) for m in MAGNITUDES},
                          "cpt": {m: round(row["capytaine_rel_error"][m], 4) for m in MAGNITUDES}}),
              flush=True)
    return {"probe": "in_band_three_way", "rows": rows}


# ---------------------------------------------------------------------------
# merge / verdicts
# ---------------------------------------------------------------------------
def _spike(summary: dict[str, Any]) -> dict[str, Any]:
    rows = summary["rows"]
    worst: dict[str, Any] = {"magnitude": None, "abs_error": -1.0, "ka": None}
    for row in rows:
        for m in MAGNITUDES:
            e = abs(row["sub5"]["rel_error"][m])
            if e > worst["abs_error"]:
                worst = {"magnitude": m, "abs_error": e, "ka": row["ka"],
                         "coarse_abs_error": abs(row["sub4"]["rel_error"][m])}
    mag = worst["magnitude"]
    neighbours = [
        {"ka": row["ka"],
         "fine": row["sub5"]["rel_error"][mag],
         "coarse": row["sub4"]["rel_error"][mag]}
        for row in rows
        if abs(row["ka"] - SWEEP_CENTRE) >= R1_EXCLUSION - 1e-12
    ]
    near = [n for n in neighbours if abs(n["ka"] - SWEEP_CENTRE) <= 0.12]
    out = {
        "magnitude": mag,
        "peak": worst,
        "neighbour_worst_fine": max((abs(n["fine"]) for n in neighbours), default=float("nan")),
        "neighbour_worst_coarse": max((abs(n["coarse"]) for n in neighbours), default=float("nan")),
        "neighbours": neighbours,
        "near_neighbour_worst_fine": max((abs(n["fine"]) for n in near), default=float("nan")),
    }
    # R1: spike must be localised (peak > 20 %, neighbours < 10 %).
    out["R1_irregular_localised"] = bool(
        worst["abs_error"] > R1_SPIKE and out["neighbour_worst_fine"] < R1_NEIGHBOUR)
    # R2: the peak must move with the mesh.
    shift = abs(float(worst["ka"]) - float(SWEEP_CENTRE)) if False else None
    # Locate the coarse-mesh peak of the same magnitude and compare positions.
    coarse_rows = sorted(rows, key=lambda r: -abs(r["sub4"]["rel_error"][mag]))
    fine_rows = sorted(rows, key=lambda r: -abs(r["sub5"]["rel_error"][mag]))
    out["peak_ka_fine"] = fine_rows[0]["ka"]
    out["peak_ka_coarse"] = coarse_rows[0]["ka"]
    shift = abs(out["peak_ka_fine"] - out["peak_ka_coarse"])
    out["peak_shift_ka"] = shift
    out["R2_irregular_mesh_dependent"] = bool(shift >= R2_MIN_SHIFT - 1e-12)
    return out


def merge(parts: list[Path]) -> dict[str, Any]:
    summary: dict[str, Any] = {"parts": [], "sweep": None, "drift": None, "in_band": None}
    for path in parts:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        summary["parts"].append({"path": Path(path).name,
                                 "probe": data.get("probe"),
                                 "generated_at": data.get("generated_at")})
        key = {"irregular_frequency_sweep": "sweep",
               "referee_self_drift": "drift",
               "in_band_three_way": "in_band"}[data["probe"]]
        if summary[key] is None:
            summary[key] = {"rows": []}
        summary[key]["rows"].extend(data["rows"])
        for k, v in data.items():
            if k != "rows":
                summary[key].setdefault(k, v)
    ok, why = capytaine_available()
    gates: dict[str, Any] = {}
    if summary["sweep"]:
        spike = _spike(summary["sweep"])
        summary["sweep"]["spike"] = spike
        gates["R1_irregular_localised"] = {
            "criterion": f"referee error peaks > {R1_SPIKE:.0%} at ka={SWEEP_CENTRE} on some "
                         f"magnitude while every probe |ka-{SWEEP_CENTRE}| >= {R1_EXCLUSION} "
                         f"stays < {R1_NEIGHBOUR:.0%}",
            "measured": {"magnitude": spike["magnitude"],
                         "peak_abs_error": spike["peak"]["abs_error"],
                         "peak_ka": spike["peak"]["ka"],
                         "neighbour_worst": spike["neighbour_worst_fine"]},
            "verdict": "PASS" if spike["R1_irregular_localised"] else "FAIL",
        }
        gates["R2_irregular_mesh_dependent"] = {
            "criterion": f"the peak ka moves by >= {R2_MIN_SHIFT} (one probe spacing) "
                         f"between the 1024 and 4096 panel referee meshes",
            "measured": {"peak_ka_fine": spike["peak_ka_fine"],
                         "peak_ka_coarse": spike["peak_ka_coarse"],
                         "shift": spike["peak_shift_ka"]},
            "verdict": "PASS" if spike["R2_irregular_mesh_dependent"] else "FAIL",
        }
    if summary["drift"]:
        worst_scaled = {"value": -1.0, "ka": None, "magnitude": None}
        worst_relative = {"value": -1.0, "ka": None, "magnitude": None}
        for row in summary["drift"]["rows"]:
            for m in MAGNITUDES:
                s = row["self_drift_scaled"][m]
                if s > worst_scaled["value"]:
                    worst_scaled = {"value": s, "ka": row["ka"], "magnitude": m}
                r = row["self_drift_relative"][m]
                if r > worst_relative["value"]:
                    worst_relative = {"value": r, "ka": row["ka"], "magnitude": m}
        gates["R3_referee_scaled_drift"] = {
            "criterion": f"|dA|/(rho*nabla) and |dB|/(rho*omega*nabla) <= {R3_SCALED_DRIFT} "
                         f"for the 1024 -> 4096 panel referee ladder",
            "measured": {"worst_scaled": worst_scaled, "worst_relative_for_reference": worst_relative},
            "verdict": "PASS" if worst_scaled["value"] <= R3_SCALED_DRIFT else "FAIL",
        }
    if summary["in_band"]:
        worst_bem = {"value": -1.0}
        worst_cpt = {"value": -1.0}
        worst_code = {"value": -1.0}
        for row in summary["in_band"]["rows"]:
            for m in MAGNITUDES:
                for src, box in ((row["tensorbem_rel_error"], worst_bem),
                                 (row["capytaine_rel_error"], worst_cpt),
                                 (row["code_vs_code"], worst_code)):
                    if abs(src[m]) > box["value"]:
                        box.clear()
                        box.update({"value": abs(src[m]), "ka": row["ka"], "magnitude": m})
        gates["R4_in_band_three_way"] = {
            "criterion": f"ka = {IN_BAND_KA}: TensorBEM(1024) and capytaine(4096) each within "
                         f"{R4_AGREEMENT:.0%} of Hulme on all four magnitudes, and within "
                         f"{R4_AGREEMENT:.0%} of each other",
            "measured": {"worst_tensorbem_vs_reference": worst_bem,
                         "worst_capytaine_vs_reference": worst_cpt,
                         "worst_code_vs_code": worst_code},
            "verdict": "PASS" if max(worst_bem["value"], worst_cpt["value"],
                                     worst_code["value"]) <= R4_AGREEMENT else "FAIL",
        }
    overall = "PASS" if gates and all(g["verdict"] == "PASS" for g in gates.values()) else "FAIL"
    summary["gates"] = gates
    summary["verdict"] = overall
    summary["environment"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "machine": platform.machine(),
        "capytaine": why,
        "capytaine_available": ok,
    }
    summary["constants"] = {"rho": RHO, "g": GRAVITY, "radius": RADIUS, "nabla": NABLA,
                            "ref_n_terms": REF_N_TERMS}
    summary["generated_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    return summary


def _table(rows, cols, fmt) -> list[str]:
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = [fmt(r) for r in rows]
    return [head, sep, *body]


def render_doc(summary: dict[str, Any]) -> str:
    L: list[str] = []
    L.append("# CASE-BEM-HULME-HEMISPHERE — 裁判侧健壮性：三格 `referee_unresolved` 的定性")
    L.append("")
    L.append(f"generated_at: `{summary['generated_at']}` 环境: "
             f"python {summary['environment']['python']} / numpy {summary['environment']['numpy']} / "
             f"{summary['environment']['capytaine']}")
    L.append("")
    L.append("复现：见 `referee_robustness.py` 文件头；本文件由 `--part merge` 程序化生成，数字零手抄。")
    L.append("")
    if summary.get("sweep"):
        sp = summary["sweep"]["spike"]
        L.append("## 1. 不规则频率探针（R1/R2）")
        L.append("")
        L.append(f"焦点量 `{sp['magnitude']}`：细网格(4096)峰值 |err| = {sp['peak']['abs_error']:.2%} "
                 f"@ ka={sp['peak']['ka']}；同量在 |ka-2.58|>=0.08 的各探测点最坏仅 "
                 f"{sp['neighbour_worst_fine']:.2%}（细）/ {sp['neighbour_worst_coarse']:.2%}（粗）。")
        L.append(f"峰值位置：细网格 ka={sp['peak_ka_fine']} vs 粗网格 ka={sp['peak_ka_coarse']}"
                 f"（位移 {sp['peak_shift_ka']:.3f}，一个探测步长 {SWEEP_STEP}）。")
        L.append("")
        L.extend(_table(
            summary["sweep"]["rows"],
            ["ka", f"{sp['magnitude']} 1024", f"{sp['magnitude']} 4096", "其余量最坏(4096)"],
            lambda r: "| {:.2f} | {:+.2%} | {:+.2%} | {:.2%} |".format(
                r["ka"], r["sub4"]["rel_error"][sp["magnitude"]],
                r["sub5"]["rel_error"][sp["magnitude"]],
                max(abs(r["sub5"]["rel_error"][m]) for m in MAGNITUDES if m != sp["magnitude"]))))
        L.append("")
    if summary.get("drift"):
        L.append("## 2. 裁判自收敛：相对度量 vs 物理缩放度量（R3）")
        L.append("")
        L.extend(_table(
            summary["drift"]["rows"],
            ["ka", "自漂移(相对)最坏", "自漂移(缩放)最坏", "自漂移(相对) B33", "自漂移(缩放) B33"],
            lambda r: "| {:.2f} | {:.2%} ({}) | {:.2e} ({}) | {:.2%} | {:.2e} |".format(
                r["ka"],
                max(r["self_drift_relative"].values()),
                max(r["self_drift_relative"], key=r["self_drift_relative"].get),
                max(r["self_drift_scaled"].values()),
                max(r["self_drift_scaled"], key=r["self_drift_scaled"].get),
                r["self_drift_relative"]["B33"], r["self_drift_scaled"]["B33"])))
        L.append("")
    if summary.get("in_band"):
        L.append("## 3. 带内三方一致（R4，原 G5 `NOT_RUN` 补跑）")
        L.append("")
        L.extend(_table(
            summary["in_band"]["rows"],
            ["ka", "量", "Hulme", "TensorBEM(1024)", "err", "capytaine(4096)", "err", "码间差"],
            lambda r: "\n".join(
                "| {:.2f} | {} | {:.5f} | {:.5f} | {:+.2%} | {:.5f} | {:+.2%} | {:.2%} |".format(
                    r["ka"], m, r["reference"][m], r["tensorbem"][m],
                    r["tensorbem_rel_error"][m], r["capytaine"][m],
                    r["capytaine_rel_error"][m], r["code_vs_code"][m]) for m in MAGNITUDES)))
        L.append("")
    L.append("## 4. 判据")
    L.append("")
    L.append("| 判据 | 判定 | 实测 |")
    L.append("|---|---|---|")
    for name, g in (summary.get("gates") or {}).items():
        L.append(f"| `{name}` | **{g['verdict']}** | `{json.dumps(g['measured'], default=str)}` |")
    L.append("")
    L.append(f"**总判：{summary['verdict']}**")
    L.append("")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", required=True, choices=["sweep", "drift", "in-band", "merge"])
    ap.add_argument("--out-dir", type=Path, default=HERE)
    ap.add_argument("--parts", nargs="*", default=None)
    ap.add_argument("--ka-start", type=float, default=None)
    ap.add_argument("--ka-stop", type=float, default=None)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.part == "merge":
        parts = args.parts or sorted(str(p) for p in args.out_dir.glob("referee_robustness_part_*.json"))
        if not parts:
            ap.error("no parts found to merge")
        summary = merge([Path(p) for p in parts])
        OUT_JSON.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        OUT_DOC.write_text(render_doc(summary), encoding="utf-8")
        print(f"wrote {OUT_JSON} and {OUT_DOC}; verdict={summary['verdict']}")
        return 0 if summary["verdict"] == "PASS" else 1

    if args.part == "sweep":
        kas = sweep_kas()
        if args.ka_start is not None:
            kas = [k for k in kas if k >= args.ka_start - 1e-12]
        if args.ka_stop is not None:
            kas = [k for k in kas if k <= args.ka_stop + 1e-12]
        payload = run_sweep(kas)
        name = f"referee_robustness_part_sweep_{args.ka_start or 0:g}_{args.ka_stop or 99:g}.json"
    elif args.part == "drift":
        payload = run_drift()
        name = "referee_robustness_part_drift.json"
    else:
        kas_in = list(IN_BAND_KA)
        if args.ka_start is not None:
            kas_in = [k for k in kas_in if k >= args.ka_start - 1e-12]
        if args.ka_stop is not None:
            kas_in = [k for k in kas_in if k <= args.ka_stop + 1e-12]
        payload = run_in_band(kas_in)
        name = (f"referee_robustness_part_in_band_{kas_in[0]:g}_{kas_in[-1]:g}.json"
                if kas_in else "referee_robustness_part_in_band.json")

    payload["generated_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    payload["environment"] = {"python": platform.python_version(),
                              "numpy": np.__version__,
                              "machine": platform.machine()}
    path = args.out_dir / name
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())