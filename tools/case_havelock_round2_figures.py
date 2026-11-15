"""Figures for CASE-BEM-SUBMERGED-SPHERE-HAVELOCK round 2 (surface measures).

Consumes the case artifact written by
``tools/run_case_havelock_round2.py`` -- every plotted number comes from that
JSON, nothing is re-typed -- and additionally runs **one** 320-panel BEM solve
at ``ka = 1`` with the exact spherical-patch measure to draw the heave-potential
field on the wetted surface.

    PYTHONPATH=src python3 tools/case_havelock_round2_figures.py \\
        --artifact runs/m21069/case_havelock_round2.json

Outputs (``docs/figures/``)::

    case_havelock_round2_sweep.png        A33'/B33'/A11'/B11' vs ka, ref + both measures
    case_havelock_round2_measure.png      worst relative error per ka, both measures, 5 % band
    case_havelock_round2_refinement.png   error vs panel size, both measures (log-log)
    case_havelock_round2_field.png        heave potential on the sphere, patch measure
and ``runs/m21069/field_heave_ka1_patch.json`` (the field behind the figure).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorbem.benchmarks.submerged_sphere_havelock import (
    AREA_MEASURE_SPHERICAL,
    panel_area_measure,
)
from tensorbem.benchmarks.case_havelock_submerged_sphere import CASE_TOLERANCE
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import submerged_sphere_mesh

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "docs" / "figures"
RUNS = ROOT / "runs" / "m21069"
COEFFICIENTS = ("a33", "b33", "a11", "b11")
LABELS = {
    "a33": r"$A'_{33}$",
    "b33": r"$B'_{33}$",
    "a11": r"$A'_{11}$",
    "b11": r"$B'_{11}$",
}


def sweep_figure(artifact: dict, out: Path) -> None:
    primary = artifact["frequency_sweep"]
    control = artifact["control_frequency_sweep"]
    ka = np.array([row["ka"] for row in primary])
    ctrl = {float(row["ka"]): row for row in control} if control else {}
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), sharex=True)
    for ax, key in zip(axes.ravel(), COEFFICIENTS):
        ref = np.array([row["%s_exact" % key] for row in primary])
        got = np.array([row["%s_bem" % key] for row in primary])
        ax.fill_between(
            ka, ref * (1 - CASE_TOLERANCE), ref * (1 + CASE_TOLERANCE),
            color="0.85", label="±5 % band",
        )
        ax.plot(ka, ref, "k-", lw=1.4, label="Havelock multipole (ref)")
        ax.plot(ka, got, "o-", color="tab:blue", ms=4.5,
                label="BEM, spherical patch measure")
        if ctrl:
            cg = np.array([ctrl[float(row["ka"])]["%s_bem" % key] for row in primary])
            ax.plot(ka, cg, "s--", color="tab:red", ms=4, alpha=0.85,
                    label="BEM, flat measure (round 1, control)")
        ax.set_xlabel(r"$ka$")
        ax.set_ylabel(LABELS[key] + r" $[\varrho\nabla]$")
        ax.set_yscale("log" if key.startswith("b") else "linear")
        ax.set_title("ka dependence -- %s" % LABELS[key], fontsize=10)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7.5, loc="best")
    fig.suptitle(
        "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK round 2: %d-panel production mesh, $f/a=%.1f$"
        % (primary[0]["n_panels"], artifact["geometry"]["submergence_ratio"])
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def measure_figure(artifact: dict, out: Path) -> None:
    rows = artifact["measure_comparison"]
    ka = np.array([row["ka"] for row in rows])
    prim = np.array([row["max_rel_error_primary"] for row in rows]) * 100.0
    ctrl = np.array([row["max_rel_error_control"] for row in rows]) * 100.0
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.semilogy(ka, ctrl, "s--", color="tab:red", label="flat measure (round 1)")
    ax.semilogy(ka, prim, "o-", color="tab:blue", label="spherical patch measure (round 2)")
    ax.axhline(CASE_TOLERANCE * 100.0, color="k", ls=":", lw=1.2, label="5 % case criterion")
    for x, p, c in zip(ka, prim, ctrl):
        ax.annotate("%.2f→%.2f %%" % (c, p), (x, p), textcoords="offset points",
                    xytext=(0, -13), ha="center", fontsize=7, color="0.25")
    ax.set_xlabel(r"$ka$")
    ax.set_ylabel("worst relative error over $A'_{33},B'_{33},A'_{11},B'_{11}$ [%]")
    ax.set_title("Surface measure fixes the round-1 damping bias")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def refinement_figure(artifact: dict, out: Path) -> None:
    prim = artifact["convergence_study"]
    ctrl = artifact.get("control_convergence_study")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for ax, key in zip(axes, ("a33", "b33")):
        for study, style, colour, name in (
            (prim, "o-", "tab:blue", "spherical patch measure"),
            (ctrl, "s--", "tab:red", "flat measure (control)"),
        ):
            if not study:
                continue
            h = np.array([level["h_mean"] for level in study["levels"]])
            err = np.array([level["%s_rel_error" % key] for level in study["levels"]]) * 100.0
            n = [level["n_panels"] for level in study["levels"]]
            ax.loglog(h, err, style, color=colour, label=name)
            for hi, ei, ni in zip(h, err, n):
                ax.annotate("%d" % ni, (hi, ei), textcoords="offset points",
                            xytext=(3, 4), fontsize=7, color=colour)
        ax.axhline(CASE_TOLERANCE * 100.0, color="k", ls=":", lw=1.2, label="5 % criterion")
        ax.set_xlabel(r"mean panel edge $h$ [m]")
        ax.set_ylabel("%s relative error [%%]" % LABELS[key])
        ax.set_title("refinement, $ka=%.1f$" % prim["ka"], fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("Flat measure converges to a biased limit; patch measure to the analytic value")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def field_figure(artifact: dict, out: Path, field_json: Path) -> dict:
    geom = artifact["geometry"]
    radius, submergence = geom["radius"], geom["submergence"]
    subdivision = artifact["surface_measure"]["production_mesh"]["subdivision"]
    ka = 1.0
    mesh = submerged_sphere_mesh(subdivision, radius=radius, submergence=submergence)
    areas = panel_area_measure(
        mesh, radius=radius, submergence=submergence, area_measure=AREA_MEASURE_SPHERICAL
    )
    nu = ka / radius
    ops = assemble_free_surface_operators_3d(mesh, nu, panel_areas=areas)
    rad = solve_radiation_3d(
        mesh, nu, modes=(0, 2), center=(0.0, 0.0, -submergence),
        density=geom["density"], gravity=geom["gravity"],
        operators=ops, panel_areas=areas,
    )
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    centers = verts[faces].mean(axis=1)
    phi = np.asarray(rad.potentials, dtype=complex)[:, 1] / radius
    payload = {
        "case": "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
        "round": 2,
        "field": "heave radiation potential phi_3 on the wetted sphere surface",
        "area_measure": AREA_MEASURE_SPHERICAL,
        "parameters": {
            "radius": radius, "submergence": submergence, "ka": ka,
            "subdivision": subdivision, "n_panels": int(mesh.panel_count),
            "density": geom["density"], "gravity": geom["gravity"],
            "normalisation": "phi / (a * U)",
        },
        "panel_centers": centers.tolist(),
        "phi_real": phi.real.tolist(),
        "phi_imag": phi.imag.tolist(),
        "phi_abs": np.abs(phi).tolist(),
        "condition_number": float(rad.condition_number),
        "relative_residual": float(rad.relative_residual),
    }
    field_json.parent.mkdir(parents=True, exist_ok=True)
    field_json.write_text(json.dumps(payload))

    fig = plt.figure(figsize=(11.0, 4.2))
    for i, (values, title) in enumerate(
        ((phi.real, r"$\mathrm{Re}\,\phi_3/(aU)$"), (np.abs(phi), r"$|\phi_3|/(aU)$"))
    ):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        sc = ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=values,
                        cmap="coolwarm" if i == 0 else "viridis", s=9)
        fig.colorbar(sc, ax=ax, shrink=0.72, label=title)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        ax.set_box_aspect((1, 1, 1))
        ax.set_title("%s, $ka=%.1f$, $f/a=%.1f$, %d panels, patch measure"
                     % ("Re" if i == 0 else "|.|", ka, submergence / radius, mesh.panel_count),
                     fontsize=8.5)
    fig.suptitle("Heave radiation potential on the wetted sphere (round-2 surface measure)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path,
                        default=RUNS / "case_havelock_round2.json")
    parser.add_argument("--skip-field", action="store_true")
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text())
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for fn, name in (
        (sweep_figure, "case_havelock_round2_sweep.png"),
        (measure_figure, "case_havelock_round2_measure.png"),
        (refinement_figure, "case_havelock_round2_refinement.png"),
    ):
        fn(artifact, FIGDIR / name)
        print("wrote", FIGDIR / name)
    if not args.skip_field:
        payload = field_figure(
            artifact, FIGDIR / "case_havelock_round2_field.png",
            RUNS / "field_heave_ka1_patch.json",
        )
        print("wrote", FIGDIR / "case_havelock_round2_field.png",
              "cond=%.3f res=%.1e" % (payload["condition_number"], payload["relative_residual"]))


if __name__ == "__main__":
    main()