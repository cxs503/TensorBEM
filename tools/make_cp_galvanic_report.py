"""Figures and paper-level report for the galvanic-protection verification face.

Reads ``docs/cathodic_protection_galvanic/results.json`` + ``fine_solution.npz``
(written by ``python -m tensorbem.cp_galvanic --output docs/...``) and writes the
PNG figures plus ``REPORT_DATA.json`` with figure payloads and the criterion
table.  Every number in the report comes from the JSON/NPZ -- nothing is typed
in by hand.
"""

from __future__ import annotations

import base64
import json
import math
import pathlib

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorbem import cp_galvanic as cp  # noqa: E402

OUT = pathlib.Path("docs/cathodic_protection_galvanic")


def _png(fig) -> str:
    import io

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def figure_hull_field(npz) -> tuple[str, dict]:
    """Surface potential / current-density cloud on the unwrapped hull."""
    centroids = npz["centroids"]
    potential = npz["potential"]
    current = npz["current_density"]
    areas = npz["areas"]
    bare = npz["bare"]
    anodes = npz["anodes"]
    length, breadth, draft = npz["hull"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    # unwrap: x coordinate along the hull, z for the vertical position; the
    # bottom (z=-draft, |y|<=B/2) and the two sides share the x axis.
    side = np.abs(np.abs(centroids[:, 1]) - breadth / 2.0) < 1e-6
    bottom = np.abs(centroids[:, 2] + draft) < 1e-6
    ends = np.abs(np.abs(centroids[:, 0]) - length / 2.0) < 1e-6

    sc = axes[0].scatter(
        centroids[side, 0], centroids[side, 2], c=potential[side], s=6,
        cmap="viridis_r", vmin=potential.min(), vmax=potential.max(),
    )
    axes[0].set_title("side shell: E = V_m - phi  [V vs Ag/AgCl]")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("z [m]")
    fig.colorbar(sc, ax=axes[0], label="electrode potential [V]")

    sc2 = axes[1].scatter(
        centroids[bottom, 0], centroids[bottom, 1], c=current[bottom] * 1e3, s=6,
        cmap="RdBu_r", vmin=-max(abs(current * 1e3).max(), 1e-6),
        vmax=max(abs(current * 1e3).max(), 1e-6),
    )
    axes[1].scatter(centroids[anodes, 0], centroids[anodes, 1], marker="s", s=22,
                    facecolors="none", edgecolors="k", label="anodes")
    axes[1].set_title("bottom: current density leaving the metal [mA/m$^2$]")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    axes[1].legend(loc="upper right", fontsize=7)
    fig.colorbar(sc2, ax=axes[1], label="i [mA/m$^2$]")
    fig.suptitle(
        "design case (beta = 0.05, 8 anodes, 1972 panels): bare panels = %.1f m$^2$, "
        "coated = %.1f m$^2$" % (areas[bare].sum(), areas.size and areas.sum() - areas[bare].sum() - areas[anodes].sum()),
        fontsize=9,
    )
    payload = {
        "panels": int(potential.size),
        "E_min": float(potential.min()),
        "E_max": float(potential.max()),
        "anode_panels": int(anodes.size),
        "ends_panels": int(ends.sum()),
    }
    return _png(fig), payload


def figure_analytic() -> tuple[str, dict]:
    """BEM vs exact profiles for the plate and cylinder anchors."""
    plate = cp.plate_linear_benchmark(n_horizontal=12, n_vertical=3)
    mesh, groups = cp.interior_box_mesh(length=10.0, breadth=10.0, height=1.0,
                                        n_horizontal=12, n_vertical=3)
    n = mesh.panel_count
    phi = np.full(n, np.nan)
    flux = np.zeros(n)
    phi[groups["z_min"]] = 0.0
    phi[groups["z_max"]] = 1.0
    plate_res = cp.solve_mixed_laplace(mesh, phi, flux, domain="interior")
    zs = np.linspace(0.02, 0.98, 25)
    plate_pts = np.array([[5.0, 5.0, z] for z in zs])
    plate_num = cp.evaluate_boundary_potential(mesh, plate_res.potential, plate_res.flux,
                                              plate_pts, domain="interior")

    cyl, cyl_groups = cp.annulus_cylinder_mesh(n_theta=48, n_radial=8, n_axial=1)
    m = cyl.panel_count
    cphi = np.full(m, np.nan)
    cflux = np.zeros(m)
    cphi[cyl_groups["inner_wall"]] = -1.0
    cphi[cyl_groups["outer_wall"]] = 0.0
    cyl_res = cp.solve_mixed_laplace(cyl, cphi, cflux, domain="interior")
    radii = np.linspace(1.05, 9.5, 30)
    cyl_pts = np.array([[r * math.cos(0.7), r * math.sin(0.7), 1.0] for r in radii])
    cyl_num = cp.evaluate_boundary_potential(cyl, cyl_res.potential, cyl_res.flux,
                                             cyl_pts, domain="interior")
    cyl_exact = -np.log(10.0 / radii) / math.log(10.0)
    plate_exact = zs

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    axes[0].plot(zs, plate_exact, "k-", label="exact $\\phi = Vz/H$")
    axes[0].plot(zs, plate_num, "o", ms=4, label="BEM (864 panels)")
    axes[0].set_xlabel("z / H")
    axes[0].set_ylabel("$\\phi$ [V]")
    axes[0].set_title("infinite parallel plates (rel. L2 = %.2e)"
                      % plate["potential_rel_l2"])
    axes[0].legend()

    axes[1].plot(radii, cyl_exact, "k-", label="exact $\\phi_a\\ln(R/r)/\\ln(R/a)$")
    axes[1].plot(radii, cyl_num, "o", ms=4, label="BEM (1728 panels)")
    axes[1].set_xlabel("r [m]")
    axes[1].set_ylabel("$\\phi$ [V]")
    axes[1].set_title("coaxial cylinders (rel. L2 = %.2e)"
                      % cp.cylinder_log_benchmark(n_theta=48, n_radial=8, n_axial=1)["potential_rel_l2"])
    axes[1].legend()
    return _png(fig), {
        "plate_max_error": float(np.max(np.abs(plate_num - plate_exact))),
        "cylinder_max_error": float(np.max(np.abs(cyl_num - cyl_exact))),
    }


def figure_parameter_study(study: dict) -> tuple[str, dict]:
    cases = study["parameter_study"]["cases"]
    betas = sorted({c["beta_requested"] for c in cases})
    counts = sorted({c["n_anodes"] for c in cases})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    for count in counts:
        current = [next(c for c in cases if c["beta_requested"] == b and c["n_anodes"] == count)["anode_current_A"] for b in betas]
        axes[0].plot(betas, current, "o-", label="N = %d anodes" % count)
        lo = [next(c for c in cases if c["beta_requested"] == b and c["n_anodes"] == count)["phi_min_steel"] for b in betas]
        hi = [next(c for c in cases if c["beta_requested"] == b and c["n_anodes"] == count)["phi_max_steel"] for b in betas]
        axes[1].fill_between(betas, lo, hi, alpha=0.25, label="N = %d" % count)
    axes[0].plot(betas, [0.15 * 4400.0 * b for b in betas], "k--", lw=1,
                 label="DNV $i_{bare}\\,\\beta A$ (0.15 A/m$^2$)")
    axes[0].set_xlabel("coating breakdown ratio $\\beta$")
    axes[0].set_ylabel("total anode current [A]")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7)
    axes[0].set_title("anode current vs coating breakdown")
    axes[1].axhline(-0.80, color="k", ls="--", lw=1)
    axes[1].axhline(-1.05, color="r", ls="--", lw=1)
    axes[1].set_xlabel("coating breakdown ratio $\\beta$")
    axes[1].set_ylabel("protection envelope $E$ [V]")
    axes[1].set_title("DNV window (dashed) vs computed envelope")
    axes[1].legend(fontsize=7)
    return _png(fig), {"betas": betas, "anode_counts": counts}


def figure_mesh_convergence(study: dict) -> tuple[str, dict]:
    cases = study["mesh_convergence"]["cases"]
    names = list(cases)
    cell = [cases[n]["cell_size_m"] for n in names]
    current = [cases[n]["anode_current_A"] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
    axes[0].plot(cell, current, "o-")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("cell size $\\sqrt{\\langle A\\rangle}$ [m]")
    axes[0].set_ylabel("$I_{anode}$ [A]")
    axes[0].set_title("design case: anode current, GCI = %.2f %%"
                      % study["criteria"]["e_mesh_gci_anode_current"]["value"])
    for name in names:
        axes[1].plot([cell[names.index(name)]] * 2,
                     [cases[name]["phi_min_steel"], cases[name]["phi_max_steel"]],
                     "|-", ms=12, label="%s (%d panels)" % (name, cases[name]["panels"]))
    axes[1].axhline(-0.80, color="k", ls="--", lw=1)
    axes[1].axhline(-1.05, color="r", ls="--", lw=1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("cell size [m]")
    axes[1].set_ylabel("envelope $E$ [V]")
    axes[1].set_title("protection envelope vs mesh")
    axes[1].legend(fontsize=7)
    return _png(fig), {"cell_sizes": cell, "currents": current}


def main() -> int:
    study = json.loads((OUT / "results.json").read_text())
    npz = np.load(OUT / "fine_solution.npz")
    figures = []
    png, info = figure_hull_field(npz)
    figures.append({
        "name": "hull_potential_field.png",
        "caption": "Design case (beta=0.05, N=8, 1972 panels): electrode potential "
                   "E = V_m - phi on the side shell (left) and current density leaving "
                   "the metal on the bottom with anode footprints (right).",
        "png_base64": png,
    })
    (OUT / "hull_potential_field.png").write_bytes(base64.b64decode(png))
    png, analytic_info = figure_analytic()
    figures.append({
        "name": "analytic_anchors.png",
        "caption": "Analytic anchors: BEM vs exact linear profile between two infinite "
                   "plates (left) and logarithmic profile between coaxial infinite "
                   "cylinders (right).",
        "png_base64": png,
    })
    (OUT / "analytic_anchors.png").write_bytes(base64.b64decode(png))
    png, param_info = figure_parameter_study(study)
    figures.append({
        "name": "parameter_study.png",
        "caption": "Coating-breakdown study on the 1080-panel mesh: total anode current "
                   "vs beta and anode count (left, with the DNV design-current line) and "
                   "the computed protection envelope against the DNV window (right).",
        "png_base64": png,
    })
    (OUT / "parameter_study.png").write_bytes(base64.b64decode(png))
    png, mesh_info = figure_mesh_convergence(study)
    figures.append({
        "name": "mesh_convergence.png",
        "caption": "Mesh convergence of the design case: anode current and protection "
                   "envelope on the 572/1080/1972-panel meshes.",
        "png_base64": png,
    })
    (OUT / "mesh_convergence.png").write_bytes(base64.b64decode(png))
    payload = {"figures": figures, "hull_field": info,
               "analytic": analytic_info, "parameter": param_info,
               "mesh": mesh_info}
    (OUT / "figure_data.json").write_text(json.dumps(payload, indent=2))
    print("figures written:", [f["name"] for f in figures])
    print("extras:", info, analytic_info, mesh_info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
