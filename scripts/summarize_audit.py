#!/usr/bin/env python3
"""Summarise the m17235 accountability audit of CASE-BEM-HULME-HEMISPHERE.

Consumes

* ``runs/casepool_bem_hulme_hemisphere/results.json`` of mission m17181 (read-only)
* ``artifacts/independent_hemisphere_check.json`` (this branch, capytaine ladder)

and produces

* ``artifacts/audit_summary.json`` -- the machine-readable audit table
* ``docs/board_audit/m17235/hulme_hemisphere_errors.png`` -- figure for the report

The independent limit is a 3-point Richardson extrapolation of the capytaine
ladder; it is reported with the raw ladder so a reader can judge the
extrapolation instead of trusting it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

GRAVITY = 9.80665
DENSITY = 1025.0
VOLUME = (2.0 / 3.0) * math.pi
CASE_DIR = Path("/root/tp-agent/work/m17181/runs/casepool_bem_hulme_hemisphere")
HERE = Path(__file__).resolve().parents[1]


def richardson(ladder: list[float]) -> tuple[float, float]:
    """3-point Richardson limit + estimated geometric ratio of the residuals."""
    if len(ladder) < 3:
        return ladder[-1], float("nan")
    v1, v2, v3 = ladder[-3], ladder[-2], ladder[-1]
    d1, d2 = v2 - v1, v3 - v2
    if d1 == 0 or d2 == 0 or (d2 / d1) >= 1 or (d2 / d1) <= 0:
        return v3, float("nan")
    ratio = d2 / d1
    return v3 + d2 * ratio / (1.0 - ratio), ratio


def main() -> int:
    results = json.loads((CASE_DIR / "results.json").read_text())
    ind = json.loads((HERE / "artifacts/independent_hemisphere_check.json").read_text())

    ref = results["reference"]
    got = {
        f"{float(r['ka']):.3f}": r
        for r in results["rows"]
        if r.get("kind") == "band" and int(r["subdivisions"]) == 3
    }
    levels = ind["independent"]["levels"]
    finest_nf = levels[-1]["n_faces"]

    cells = []
    for nu_key, block in levels[-1]["values"].items():
        ka = float(nu_key)
        omega = math.sqrt(ka * GRAVITY)
        rho_vol = DENSITY * VOLUME
        rho_omega_vol = DENSITY * omega * VOLUME
        for dof, suffix in (("heave", "33"), ("surge", "11")):
            for quantity, key, scale in (
                ("added_mass", f"A{suffix}_dimless", rho_vol),
                ("damping", f"B{suffix}_dimless", rho_omega_vol),
            ):
                ladder = [
                    lv["values"][nu_key][dof][quantity] / scale for lv in levels
                ]
                limit, ratio = richardson(ladder)
                ref_v = ref[f"{ka:g}"][key] if f"{ka:g}" in ref else ref[f"{ka:.1f}"][key]
                got_v = got[f"{ka:.3f}"][key]
                cells.append(
                    {
                        "ka": ka,
                        "dof": dof,
                        "quantity": quantity,
                        "ref_selfderived": ref_v,
                        "got_tensorbem_256panels": got_v,
                        "ind_capytaine_finest": ladder[-1],
                        "ind_capytaine_richardson": limit,
                        "ind_richardson_ratio": ratio,
                        "ind_ladder_faces": [lv["n_faces"] for lv in levels],
                        "ind_ladder": ladder,
                        "err_got_vs_ref": (got_v - ref_v) / abs(ref_v),
                        "err_got_vs_ind_limit": (got_v - limit) / abs(limit),
                        "err_ref_vs_ind_limit": (ref_v - limit) / abs(limit),
                    }
                )

    def worst(rows, key):
        row = max(rows, key=lambda r: abs(r[key]))
        return {"value": row[key], "ka": row["ka"], "dof": row["dof"], "quantity": row["quantity"]}

    verdict = results["verdict"]["criteria"]
    summary = {
        "audited_mission": 17181,
        "anchor": "CASE-BEM-HULME-HEMISPHERE (tensor_case_pool.id=138, tol <5%)",
        "case_mesh": {"subdivisions": 3, "panels": 256},
        "capytaine_finest_faces": finest_nf,
        "cells": cells,
        "worst": {
            "got_vs_ref": worst(cells, "err_got_vs_ref"),
            "got_vs_independent_limit": worst(cells, "err_got_vs_ind_limit"),
            "ref_vs_independent_limit": worst(cells, "err_ref_vs_ind_limit"),
        },
        "mission_gates": verdict,
        "mission_verdict": results["verdict"]["verdict"],
        "reference_provenance": results["reference_provenance"],
        "geometry": {
            "mesh_volume_error": verdict["G2_geometry_precondition"]["measured_volume_error"],
            "mesh_waterplane_error": verdict["G2_geometry_precondition"]["measured_waterplane_error"],
            "gate": verdict["G2_geometry_precondition"]["criterion"],
        },
    }
    (HERE / "artifacts/audit_summary.json").write_text(json.dumps(summary, indent=1))

    # ---- table to stdout (numbers are injected into the doc programmatically) --
    print(f"{'ka':>4} {'dof':>6} {'qty':>10} {'got':>9} {'ref':>9} {'ind':>9} "
          f"{'got/ref':>8} {'got/ind':>8} {'ref/ind':>8} {'r':>5}")
    for c in cells:
        print(
            f"{c['ka']:4.1f} {c['dof']:>6} {c['quantity']:>10} "
            f"{c['got_tensorbem_256panels']:9.4f} {c['ref_selfderived']:9.4f} "
            f"{c['ind_capytaine_richardson']:9.4f} "
            f"{100*c['err_got_vs_ref']:+7.2f}% {100*c['err_got_vs_ind_limit']:+7.2f}% "
            f"{100*c['err_ref_vs_ind_limit']:+7.2f}% {c['ind_richardson_ratio']:5.2f}"
        )
    print(json.dumps(summary["worst"], indent=1))

    # ---- figure ---------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, dof, title in (
        (axes[0], "heave", "heave (A33, B33)"),
        (axes[1], "surge", "surge (A11, B11)"),
    ):
        for quantity, marker in (("added_mass", "o"), ("damping", "s")):
            rows = [c for c in cells if c["dof"] == dof and c["quantity"] == quantity]
            rows.sort(key=lambda c: c["ka"])
            kas = [c["ka"] for c in rows]
            ax.plot(kas, [100 * c["err_got_vs_ref"] for c in rows], marker=marker,
                    ls="-", color="tab:blue" if quantity == "added_mass" else "tab:red",
                    label=f"TensorBEM vs self-derived ref ({quantity})")
            ax.plot(kas, [100 * c["err_got_vs_ind_limit"] for c in rows], marker=marker,
                    ls="--", mfc="none",
                    color="tab:blue" if quantity == "added_mass" else "tab:red",
                    label=f"TensorBEM vs capytaine limit ({quantity})")
        ax.axhline(0, color="k", lw=0.6)
        ax.axhline(5, color="grey", ls=":", lw=0.8)
        ax.axhline(-5, color="grey", ls=":", lw=0.8)
        ax.set_title(f"{title}: deviation at 256 panels")
        ax.set_xlabel("ka")
        ax.set_ylabel("relative deviation [%]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(
        "CASE-BEM-HULME-HEMISPHERE audit (m17235): the 5% gate holds vs the "
        "self-derived series, not vs an independent code (dotted = ±5%)",
        fontsize=9,
    )
    fig.tight_layout()
    out_png = HERE / "docs/board_audit/m17235/hulme_hemisphere_errors.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"[figure] {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())