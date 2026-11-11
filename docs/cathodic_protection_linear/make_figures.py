"""Regenerate the figures and numbers of ``docs/cathodic_protection_linear``.

Every number in ``REPORT.md`` is produced by this script (no hand copying):

* the hull potential field of the 200 m bulk carrier at a 5 % coating-damage
  ratio (side view, wetted panels only, holidays and anodes marked);
* the conducting-sphere refinement history against the exact
  ``I = 4 pi sigma a phi_a`` and the exterior field ``phi_a a / r``;
* the BEM-versus-DNV current-density comparison over the damage ratio.

Run from the repository root::

    PYTHONPATH=src python docs/cathodic_protection_linear/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorbem.cathodic_protection import (  # noqa: E402
    BulkCarrierHull,
    assign_coating_damage,
    generate_closed_hull_mesh,
    run_verification_study,
    solve_cathodic_protection,
    verify_conducting_sphere,
)

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"


def hull_field(damage_ratio: float = 0.05, n_sections: int = 13) -> dict:
    """Solve one hull case and persist the fields used by figure 1."""
    hull = BulkCarrierHull()
    mesh = generate_closed_hull_mesh(hull, n_sections=n_sections, n_side=3, n_bottom=2)
    coated, damaged, anodes = assign_coating_damage(mesh, damage_ratio, seed=42)
    result = solve_cathodic_protection(mesh, coated, damaged, anodes)
    c = mesh.centroids
    wetted = ~result.lid_panels
    data = {
        "case": {
            "length_m": hull.length,
            "breadth_m": hull.breadth,
            "draft_m": hull.draft,
            "damage_ratio": damage_ratio,
            "n_sections": n_sections,
            "panels": int(mesh.panel_count),
            "wetted_area_m2": result.wetted_area,
            "lid_area_m2": result.lid_area,
            "anode_panels": int(len(anodes)),
            "holiday_panels": int(len(damaged)),
            "anode_current_A": result.anode_current_total,
            "protection_coverage": result.protection_coverage,
            "min_potential_V": result.min_potential_hull,
            "max_potential_V": result.max_potential_hull,
            "condition_number": result.condition_number,
        },
        "panels": {
            "x": c[:, 0].tolist(),
            "z": c[:, 2].tolist(),
            "potential": result.potential.tolist(),
            "wetted": wetted.tolist(),
            "is_holiday": np.isin(np.arange(mesh.panel_count), damaged).tolist(),
            "is_anode": np.isin(np.arange(mesh.panel_count), anodes).tolist(),
        },
    }
    return data


def figure_hull(data: dict) -> str:
    c = data["panels"]
    x = np.asarray(c["x"])
    z = np.asarray(c["z"])
    phi = np.asarray(c["potential"])
    wetted = np.asarray(c["wetted"])
    holiday = np.asarray(c["is_holiday"])
    anode = np.asarray(c["is_anode"])

    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    order = np.argsort(phi[wetted])
    sc = ax.scatter(
        x[wetted][order] / data["case"]["length_m"],
        z[wetted][order] / data["case"]["draft_m"],
        c=phi[wetted][order], s=26, cmap="viridis", marker="s", linewidths=0,
    )
    ax.scatter(x[holiday] / data["case"]["length_m"], z[holiday] / data["case"]["draft_m"],
               marker="o", facecolors="none", edgecolors="crimson", s=70,
               label=f"coating holiday ({int(holiday.sum())} panels)")
    ax.scatter(x[anode] / data["case"]["length_m"], z[anode] / data["case"]["draft_m"],
               marker="*", c="white", edgecolors="black", s=110,
               label=f"anode ({int(anode.sum())} panels)")
    ax.set_xlabel("x / L")
    ax.set_ylabel("z / T")
    ax.set_title(
        "Linear CP BEM: electrolyte potential on the wetted hull "
        f"(L=200 m, damage {100 * data['case']['damage_ratio']:.0f} %, "
        f"{data['case']['panels']} panels)"
    )
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("phi [V vs Ag/AgCl]")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
    ax.set_aspect(0.25)
    fig.tight_layout()
    path = FIG / "hull_potential_field.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def figure_sphere(report: dict) -> str:
    levels = report["levels"]
    n = np.array([lv["n_panels"] for lv in levels], dtype=float)
    h = n ** -0.5
    err_current = np.array([lv["current_rel_error"] for lv in levels])
    err_phi = np.array([lv["max_potential_rel_error"] for lv in levels])

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.loglog(h, err_current, "o-", label="total current vs $4\\pi\\sigma a\\phi_a$")
    ax.loglog(h, err_phi, "s--", label="max exterior $\\phi$ error vs $\\phi_a a/r$")
    guide = err_current[0] * (h / h[0]) ** 2
    ax.loglog(h, guide, "k:", label="$O(h^2)$ guide")
    ax.set_xlabel("$h \\propto N^{-1/2}$ (icosphere level)")
    ax.set_ylabel("relative error")
    ax.set_title(
        "Conducting-sphere anchor "
        f"(r = {report['radius_m']:g} m, $\\sigma$ = {report['conductivity_S_m']:g} S/m)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = FIG / "sphere_convergence.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def figure_dnv(study: dict) -> str:
    cases = study["cases"]
    dr = np.array([c["damage_ratio"] for c in cases])
    j_bem = np.array([c["current_per_wetted_area_bem_A_m2"] for c in cases])
    j_dnv = np.array([c["current_per_wetted_area_dnv_A_m2"] for c in cases])

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(100 * dr, j_bem, "o-", label="BEM (linear, prescribed potentials)")
    ax.plot(100 * dr, j_dnv, "s--", label="DNV-RP-B401 simplified demand")
    ax.set_xlabel("coating damage ratio [% of wetted area]")
    ax.set_ylabel("anode current per wetted area [A/m$^2$]")
    ax.set_title("Resistance-limited linear model vs DNV demand")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "dnv_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    field = hull_field()
    (OUT / "hull_field.json").write_text(json.dumps(field, indent=1))
    print("figure:", figure_hull(field))

    sphere = verify_conducting_sphere((1, 2, 3))
    (OUT / "sphere-spreading-resistance.json").write_text(json.dumps(sphere, indent=2))
    print("figure:", figure_sphere(sphere))

    study = run_verification_study(
        damage_ratios=(0.02, 0.05, 0.10),
        mesh_levels=(9, 11, 13),
        # The C8 verdict quoted in REPORT.md section 3.3 is the same three-level
        # icosphere ladder as the standalone benchmark, so the study artefact
        # cannot disagree with the anchor it cites.
        sphere_levels=(1, 2, 3),
        output_dir=OUT / "artifacts",
    )
    (OUT / "results.json").write_text(
        json.dumps(
            {
                "hull": {
                    "length_m": 200.0,
                    "breadth_m": 32.0,
                    "draft_m": 12.0,
                    "wetted_area_m2": study["cases"][0]["wetted_area_m2"],
                    "lid_area_m2": study["cases"][0]["lid_area_m2"],
                },
                "sphere_verification": sphere,
                "mesh_convergence": study["mesh_convergence"],
                "cases": study["cases"],
                "verification": study["verification"],
            },
            indent=2,
        )
    )
    print("figure:", figure_dnv(study))


if __name__ == "__main__":
    main()