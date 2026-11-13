#!/usr/bin/env python3
"""CASE-BEM-NEUMANN-KELVIN -- steady Neumann-Kelvin wave resistance vs Michell Cw(Fr).

Case-pool item (owner 2026-11-12 算例池令), mission 17246, repo TensorBEM.

What is computed
----------------
Wigley III parabolic hull (L=10 m, B=1.0 m, T=0.625 m, B/L = 0.1) advancing at
constant speed ``U`` on deep calm water.  Two things are produced:

* **Route A -- steady-Kelvin far field (got)**: the Neumann-Kelvin steady wave
  problem is solved in its thin-ship form, ``sigma = -2 U db/dx`` on the
  centreplane, and the wave resistance is evaluated with the *steady Kelvin*
  far-field (Havelock/Kochin) operator of ``tensorbem.nk_panel3d``

      Rw = rho kappa^2/(2 pi) int_{-pi/2}^{pi/2} |H(theta)|^2 sec^3(theta) dtheta

  with ``kappa = g/U^2`` the steady wavenumber.
* **Route B -- NK panel method on the real hull (got, sub-route)**: the
  single-layer NK integral equation of ``nk_panel3d.solve_nk_panel`` on the
  wetted hull, same far-field operator.  Reported honestly, not as evidence
  when it fails.

**ref (zero hand-copy)**: Michell thin-ship theory, ``W. Michell, "The
wave-resistance of a ship", Philosophical Magazine 45 (1898) 106-123``, in the
form ``Rw = 4 rho g^2/(pi U^2) int_0^{pi/2} sec^3(theta) |A|^2 dtheta``.  The
numbers come from the repository's arbitrated implementation
``tensorbem.wigley.michell_wave_resistance_canonical`` (arbitration of the
competing Michell exits: ``docs/michell_exit_arbitration.md``, mission m16345,
criteria = steady dispersion feasibility + Fr-preserving similarity exponent +
amplitude consistency + quadrature convergence) and are cross-checked against
the independent ``tensorbem.thin_ship_michell`` lambda-form quadrature.

Usage
-----
``PYTHONPATH=src python scripts/case_bem_neumann_kelvin_michell.py [--stage cheap|full]``
Writes ``docs/case_bem_neumann_kelvin/{numbers.json,table.md,*.png}``.
No number in the report is typed by hand: every quoted value is read back from
``numbers.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorbem import nk_panel3d as nk
from tensorbem import thin_ship_michell as tsm
from tensorbem import wigley

OUT = Path("docs/case_bem_neumann_kelvin")
L, B, T = 10.0, 1.0, 0.625
RHO, G = 1025.0, 9.80665
FROUDES = (0.20, 0.25, 0.30, 0.35, 0.40)
S_REF = L * T  # case convention (matches benchmarks/verified/bem_wigley_wave_resistance)
TOL = 0.05
#: the module's own radiation referee (`verify_radiation_direction`) selects this
#: mode: waves behind the source, ratio 3.02.  `radiation="downstream"` (the
#: `solve_nk_panel` default) gives ratio ~0.99, i.e. no Kelvin wake asymmetry.
RADIATION = {"radiation": "halfplane", "sign": -1.0}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def speed(fr: float) -> float:
    return fr * math.sqrt(G * L)


def cw(rw: float, u: float, s_ref: float = S_REF) -> float:
    return rw / (0.5 * RHO * u * u * s_ref)


def michell_canonical(u: float) -> float:
    """Arbitrated Michell reference (m16345)."""
    return float(wigley.michell_wave_resistance_canonical(L, B, T, u, RHO, G))


def michell_lambda_form(u: float, lam_max: float = 40.0, n_lam: int = 480) -> float:
    """Independent lambda-form quadrature of the same Michell integral."""
    hull = tsm.WigleyHull(L=L, B=B, T=T)
    amp = tsm.wigley_amplitude_function(hull)
    return float(tsm.michell_wave_resistance(amp, u, lam_max=lam_max, n_lam=n_lam, rho=RHO, g=G)["Rw"])


def thin_ship_sigma(mesh, u: float):
    """NK thin-ship body condition on the centreplane: sigma = -2 U db/dx."""
    x = mesh.centroids[:, 0]
    z = mesh.centroids[:, 2]
    b_x = 0.5 * B * (-8.0 * x / L**2) * (1.0 - (z / T) ** 2)
    return -2.0 * u * b_x, b_x


def far_resistance(sigma, mesh, kappa: float, u: float, n_theta: int = 2001, mirror: bool = False):
    """Steady Kelvin far-field operator, exact prefactor kappa/(2 pi g)."""
    rw, th, H = nk.wave_resistance_far(
        sigma, mesh, kappa, rho=RHO, g=G, n_theta=n_theta, mirror=mirror
    )
    return rw, th, H


def far_from_amplitude(u: float, n_theta: int = 4001):
    """Same operator applied to the closed-form Michell amplitude (exact hull integrals)."""
    kappa = G / u**2
    th = np.linspace(-0.5 * math.pi + 1e-7, 0.5 * math.pi - 1e-7, n_theta)
    lam = 1.0 / np.cos(th)
    hull = tsm.WigleyHull(L=L, B=B, T=T)
    A = tsm.wigley_amplitude_function(hull)(kappa * lam**2, kappa * lam)
    H = -2.0 * u * A
    rw = nk.kochin_prefactor(kappa, G) * RHO * G * kappa * float(
        np.trapezoid(np.abs(H) ** 2 / np.cos(th) ** 3, th)
    )
    return float(rw), th, H


def richardson(values):
    """Two-level Richardson estimate + observed order from a factor-2 ladder."""
    if len(values) < 3:
        return float("nan"), float("nan")
    f1, f2, f3 = values[-3], values[-2], values[-1]
    d1, d2 = f2 - f1, f3 - f2
    if d2 == 0.0 or d1 == 0.0 or d1 / d2 <= 1.0:
        return float("nan"), float("nan")
    p = math.log(d1 / d2) / math.log(2.0)
    return float(f3 + d2 / (2.0**p - 1.0)), float(p)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def geometry_block() -> dict:
    """Geometry first: closed-form hydrostatics + panel-area convergence."""
    vol = 4.0 * L * B * T / 9.0
    rows = []
    for nx, nz in ((8, 4), (16, 8), (32, 16), (64, 32)):
        mesh = nk.wigley_half_panels(L, B, T, nx, nz)
        rows.append({"nx": nx, "nz": nz, "n_panels": mesh.n_panels,
                     "half_area": mesh.total_area, "full_area": 2.0 * mesh.total_area})
    areas = [r["full_area"] for r in rows]
    return {
        "L": L, "B": B, "T": T, "B_over_L": B / L, "T_over_L": T / L, "L_over_B": L / B,
        "Cb_closed_form": 4.0 / 9.0,
        "Cm_closed_form": 2.0 / 3.0,
        "Cp_closed_form": 1.0,
        "volume_closed_form": vol,
        "waterplane_area_closed_form": 2.0 * L * B / 3.0,
        "notes": "y = (B/2)(1-(2x/L)^2)(1-(z/T)^2): Cb = 4/9 (NOT 0.5625; that "
                 "figure in the campaign metadata does not belong to this hull, see "
                 "docs/m10003/REPORT.md), Cm = 2/3, Cp = Cb/Cm = 1 (fine form).",
        "panel_area_ladder": rows,
        "area_drift_last_pair": abs(areas[-1] - areas[-2]) / areas[-1],
    }


def kernel_block() -> dict:
    """NK-specific checks that do not involve Michell at all."""
    fs = nk.verify_free_surface_condition(kappa=1.0, n_theta=64)
    rad = nk.verify_radiation_direction(kappa=1.0, n_theta=64)
    return {"free_surface_condition": fs, "radiation": rad}


def route_a(stage: str) -> dict:
    """Steady-Kelvin far field with the thin-ship NK source sheet (primary route)."""
    meshes = ((30, 15), (60, 30)) if stage == "cheap" else ((30, 15), (60, 30), (120, 60))
    rows = []
    for fr in FROUDES:
        u = speed(fr)
        kappa = G / u**2
        ref = michell_canonical(u)
        ref_lam = michell_lambda_form(u)
        got_amp, _, _ = far_from_amplitude(u)
        mesh_rows = []
        for nx, nz in meshes:
            mesh = nk.centreplane_panels(L, B, T, nx, nz)
            sigma, b_x = thin_ship_sigma(mesh, u)
            t0 = time.time()
            rw, th, H = far_resistance(sigma, mesh, kappa, u, n_theta=2001, mirror=False)
            mesh_rows.append({"nx": nx, "nz": nz, "n_panels": mesh.n_panels,
                              "rw": rw, "cw": cw(rw, u),
                              "rel_error_ref": (rw - ref) / ref,
                              "seconds": time.time() - t0})
        theta_sens = []
        for n_theta in (501, 1001, 2001, 4001, 8001):
            rw_th, _, _ = far_from_amplitude(u, n_theta=n_theta)
            theta_sens.append({"n_theta": n_theta, "rw": rw_th, "rel_error_ref": (rw_th - ref) / ref})
        lam_sens = [{"lam_max": lm, "rw": michell_lambda_form(u, lam_max=lm)}
                    for lm in (20.0, 40.0, 60.0)]
        finest = mesh_rows[-1]
        rows.append({
            "Fr": fr, "U": u, "kappa": kappa,
            "ref_michell_canonical": ref, "ref_Cw": cw(ref, u),
            "ref_michell_lambda_form": ref_lam,
            "ref_route_spread": abs(ref_lam - ref) / ref,
            "got_farfield_closed_form_amplitude": got_amp, "got_Cw_closed_form": cw(got_amp, u),
            "err_closed_form": (got_amp - ref) / ref,
            "got_farfield_panel_finest": finest["rw"], "got_Cw_panel_finest": finest["cw"],
            "err_panel_finest": finest["rel_error_ref"],
            "panel_mesh_ladder": mesh_rows,
            "theta_sensitivity": theta_sens,
            "lam_max_sensitivity": lam_sens,
            "passed_finest": abs(finest["rel_error_ref"]) < TOL,
            "passed_closed_form": abs(got_amp - ref) / ref < TOL,
        })
    rws = [r["got_farfield_panel_finest"] for r in rows]
    rich, order = richardson(rws)
    return {
        "froudes": list(FROUDES), "s_ref": S_REF, "tolerance": TOL,
        "rows": rows,
        "richardson_rw_extrapolated": rich,
        "richardson_observed_order": order,
        "max_abs_error": max(abs(r["err_panel_finest"]) for r in rows),
        "all_passed": all(r["passed_finest"] for r in rows),
    }


def route_b(stage: str) -> dict:
    """NK single-layer panel method on the real hull (sub-route, honest)."""
    ladder = ((16, 8),) if stage == "cheap" else ((16, 8), (24, 12), (32, 16))
    sweep_mesh = (16, 8) if stage == "cheap" else (24, 12)
    rows = []
    for fr in FROUDES:
        u = speed(fr)
        kappa = G / u**2
        ref = michell_canonical(u)
        mesh = nk.wigley_half_panels(L, B, T, *sweep_mesh)
        t0 = time.time()
        res = nk.solve_nk_panel(mesh, u, rho=RHO, g=G, n_theta=48, **RADIATION)
        rw, _, H = far_resistance(res.sigma, mesh, kappa, u, n_theta=2001, mirror=True)
        rows.append({"Fr": fr, "U": u, "mesh": list(sweep_mesh), "n_panels": mesh.n_panels,
                     "rw_far": rw, "cw_far": cw(rw, u), "ref": ref, "ref_Cw": cw(ref, u),
                     "rel_error": (rw - ref) / ref, "bc_residual": res.bc_residual,
                     "condition_number": res.extra["condition_number"],
                     "rw_near_field": res.wave_resistance, "seconds": time.time() - t0})
    ladder_rows = []
    fr = 0.30
    u = speed(fr)
    kappa = G / u**2
    ref = michell_canonical(u)
    for nx, nz in ladder:
        mesh = nk.wigley_half_panels(L, B, T, nx, nz)
        t0 = time.time()
        res = nk.solve_nk_panel(mesh, u, rho=RHO, g=G, n_theta=48, **RADIATION)
        rw, _, _ = far_resistance(res.sigma, mesh, kappa, u, n_theta=2001, mirror=True)
        ladder_rows.append({"nx": nx, "nz": nz, "n_panels": mesh.n_panels, "rw_far": rw,
                            "cw_far": cw(rw, u), "rel_error": (rw - ref) / ref,
                            "bc_residual": res.bc_residual, "seconds": time.time() - t0})
    theta_rows = []
    mesh = nk.wigley_half_panels(L, B, T, sweep_mesh[0], sweep_mesh[1])
    for n_theta in ((24, 48) if stage == "cheap" else (24, 32, 48, 64)):
        t0 = time.time()
        res = nk.solve_nk_panel(mesh, u, rho=RHO, g=G, n_theta=n_theta, **RADIATION)
        rw, _, _ = far_resistance(res.sigma, mesh, kappa, u, n_theta=2001, mirror=True)
        theta_rows.append({"n_theta": n_theta, "rw_far": rw, "rel_error": (rw - ref) / ref,
                           "seconds": time.time() - t0})
    bl_rows = []
    for b_l in (0.2, 0.1, 0.05):
        b = b_l * L
        mesh = nk.wigley_half_panels(L, b, T, 16, 8)
        res = nk.solve_nk_panel(mesh, u, rho=RHO, g=G, n_theta=48, **RADIATION)
        rw, _, _ = far_resistance(res.sigma, mesh, kappa, u, n_theta=2001, mirror=True)
        ref_bl = float(wigley.michell_wave_resistance_canonical(L, b, T, u, RHO, G))
        # distance from a hull panel to its own mirror image (thin-hull conditioning)
        d_img = 2.0 * float(np.min(np.abs(mesh.centroids[:, 1])))
        bl_rows.append({"B_over_L": b_l, "B": b, "rw_far": rw, "ref": ref_bl,
                        "rel_error": (rw - ref_bl) / ref_bl,
                        "min_mirror_distance": d_img,
                        "condition_number": res.extra["condition_number"]})
    rad_rows = []
    mesh_r = nk.wigley_half_panels(L, B, T, sweep_mesh[0], sweep_mesh[1])
    for tag, kw in (("halfplane,sign=-1 (verified)", RADIATION),
                    ("downstream,sign=-1 (module default)", {"radiation": "downstream", "sign": -1.0})):
        res = nk.solve_nk_panel(mesh_r, speed(0.30), rho=RHO, g=G, n_theta=48, **kw)
        rw, _, _ = far_resistance(res.sigma, mesh_r, G / speed(0.30) ** 2, speed(0.30),
                                  n_theta=2001, mirror=True)
        rad_rows.append({"mode": tag, "rw_far": rw,
                         "rel_error": (rw - michell_canonical(speed(0.30))) / michell_canonical(speed(0.30)),
                         "bc_residual": res.bc_residual})
    return {
        "radiation_mode_comparison_Fr0.30": rad_rows,
        "sweep_mesh": list(sweep_mesh), "rows": rows,
        "mesh_ladder": ladder_rows, "theta_sensitivity": theta_rows,
        "thin_hull_ladder": bl_rows,
        "max_abs_error": max(abs(r["rel_error"]) for r in rows),
        "any_passed": any(abs(r["rel_error"]) < TOL for r in rows),
    }


def wave_field(stage: str) -> dict:
    """Free-surface elevation of the NK thin-ship sheet (field figure + Kelvin angle)."""
    OUT.mkdir(parents=True, exist_ok=True)
    nx, nz = (24, 12) if stage == "cheap" else (40, 20)
    mesh = nk.centreplane_panels(L, B, T, nx, nz)
    fr = 0.30
    u = speed(fr)
    sigma, _ = thin_ship_sigma(mesh, u)
    # far-field elevations scale with kappa, so use K = 1 units for a compact grid
    kappa = 1.0
    u_unit = math.sqrt(G / kappa)
    sigma_u = sigma * (u_unit / u)
    n_grid = (41, 25) if stage == "cheap" else (101, 61)
    xs = np.linspace(-3.0, 12.0, n_grid[0]) / kappa
    ys = np.linspace(-7.0, 7.0, n_grid[1]) / kappa
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    t0 = time.time()
    eta = nk.free_surface_elevation(pts, mesh, sigma_u, u_unit, g=G, n_theta=32, **RADIATION)
    zeta = eta.reshape(X.shape)
    # Kelvin wake half-angle from the outermost crest inside the wedge apex
    deg = np.degrees(np.arctan2(np.abs(Y), np.maximum(X, 1e-9)))
    mask = (X > 1.0 / kappa) & (np.abs(zeta) > 0.4 * np.abs(zeta).max())
    measured = float(np.percentile(deg[mask], 98.0)) if np.any(mask) else float("nan")
    np.savez_compressed(OUT / "wave_field.npz", x=X, y=Y, zeta=zeta, kappa=kappa)
    return {"Fr": fr, "mesh": [nx, nz], "n_panels": mesh.n_panels, "grid": list(n_grid),
            "eta_min": float(zeta.min()), "eta_max": float(zeta.max()),
            "measured_half_angle_deg": measured,
            "theory_half_angle_deg": math.degrees(math.asin(1.0 / 3.0)),
            "seconds": time.time() - t0}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def figures(numbers: dict) -> list[dict]:
    figs = []
    a = numbers["route_a"]["rows"]
    fr = [r["Fr"] for r in a]
    ref = [r["ref_Cw"] for r in a]
    got = [r["got_Cw_panel_finest"] for r in a]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(fr, ref, "k-o", label="Michell thin-ship (ref, m16345 arbitration)")
    ax.plot(fr, got, "s--", color="#d84b3b", label="NK steady far field (got)")
    ax2 = ax.twinx()
    ax2.semilogy(fr, [max(abs(r["err_panel_finest"]), 1e-12) * 100 for r in a], ":", color="#3b7dd8")
    ax2.axhline(TOL * 100, color="#3b7dd8", lw=0.7, ls="--")
    ax2.set_ylabel("|error| [%]  (dotted)", color="#3b7dd8")
    ax.set_xlabel("Froude number $Fr = U/\\sqrt{gL}$")
    ax.set_ylabel("$C_w = R_w/(0.5\\rho U^2 LT)$")
    ax.set_title("Wigley III: NK steady wave resistance vs Michell")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = OUT / "fig_cw_vs_froude.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append({"file": p.name, "caption": "Cw(Fr): NK steady-Kelvin far field (got) vs Michell "
                                            "thin-ship theory (ref); dotted = relative error, dashed "
                                            "line = 5% case tolerance."})

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for r in a:
        lad = r["panel_mesh_ladder"]
        ax.semilogy([m["n_panels"] for m in lad],
                    [max(abs(m["rel_error_ref"]), 1e-8) * 100 for m in lad], "o-", label=f"Fr={r['Fr']:.2f}")
    ax.axhline(TOL * 100, color="k", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("centreplane panels")
    ax.set_ylabel("|error vs Michell| [%]")
    ax.set_title("Route A convergence (panel quadrature of the source sheet)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = OUT / "fig_route_a_convergence.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append({"file": p.name, "caption": "Route A mesh convergence of the far-field resistance "
                                            "against the Michell reference."})

    data = np.load(OUT / "wave_field.npz")
    X, Y, Z = data["x"], data["y"], data["zeta"]
    lim = np.percentile(np.abs(Z), 99.0)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    im = ax.pcolormesh(X, Y, Z, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
    ang = math.radians(math.degrees(math.asin(1.0 / 3.0)))
    ax.plot(X[:, 0], np.maximum(X[:, 0], 0) * math.tan(ang), "k--", lw=0.9)
    ax.plot(X[:, 0], -np.maximum(X[:, 0], 0) * math.tan(ang), "k--", lw=0.9)
    ax.set_xlabel("$\\kappa x$")
    ax.set_ylabel("$\\kappa y$")
    ax.set_title("NK thin-ship free-surface elevation, $Fr=0.30$ (dashed: Kelvin cusp $\\pm19.47^\\circ$)")
    fig.colorbar(im, ax=ax, label="$\\zeta\\,[U^2/g]$")
    fig.tight_layout()
    p = OUT / "fig_kelvin_wake.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append({"file": p.name, "caption": "NK thin-ship wave elevation with the Kelvin cusp marked; "
                                            "measured outermost-crest half angle is reported in the table."})
    return figs


def table_md(numbers: dict) -> str:
    a = numbers["route_a"]["rows"]
    lines = [
        "| Fr | U [m/s] | ref Rw [N] | ref Cw | got Rw [N] | got Cw | rel err | <5% |",
        "|---:|---:|---:|---:|---:|---:|---:|:--:|",
    ]
    for r in a:
        lines.append(
            f"| {r['Fr']:.2f} | {r['U']:.4f} | {r['ref_michell_canonical']:.4f} | "
            f"{r['ref_Cw']:.6e} | {r['got_farfield_panel_finest']:.4f} | "
            f"{r['got_Cw_panel_finest']:.6e} | {100 * r['err_panel_finest']:+.3f}% | "
            f"{'PASS' if r['passed_finest'] else 'FAIL'} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="full", choices=("cheap", "full"))
    ap.add_argument("--only", default="geometry,kernel,route_a,route_b,wave_field",
                    help="comma list of blocks to run; results merge into numbers.json")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    blocks = [b.strip() for b in args.only.split(",") if b.strip()]
    existing = {}
    if (OUT / "numbers.json").exists():
        existing = json.loads((OUT / "numbers.json").read_text())
    numbers: dict = {
        "case": "CASE-BEM-NEUMANN-KELVIN",
        "mission": 17246,
        "stage": args.stage,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo": "TensorBEM",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                   text=True).stdout.strip(),
        "reference": {
            "theory": "Michell thin-ship, Phil. Mag. 45 (1898) 106-123",
            "evaluated_by": "tensorbem.wigley.michell_wave_resistance_canonical",
            "arbitration": "docs/michell_exit_arbitration.md (mission m16345)",
            "cross_check": "tensorbem.thin_ship_michell lambda-form quadrature",
            "tolerance": TOL,
            "s_ref": "L*T (case convention of benchmarks/verified/bem_wigley_wave_resistance)",
        },
    }
    for key, fn in (("geometry", geometry_block), ("kernel", kernel_block),
                    ("route_a", lambda: route_a(args.stage)),
                    ("route_b", lambda: route_b(args.stage)),
                    ("wave_field", lambda: wave_field(args.stage))):
        if key in blocks:
            numbers[key] = fn()
        elif key in existing:
            numbers[key] = existing[key]
    numbers["blocks_run"] = blocks
    if "route_a" in numbers:
        numbers["table_md"] = table_md(numbers)
    figs = figures(numbers) if ("route_a" in numbers and (OUT / "wave_field.npz").exists()) else existing.get("figures", [])
    numbers["figures"] = figs
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1))
    (OUT / "table.md").write_text(numbers["table_md"])

    brief = {"blocks_run": blocks}
    if "route_a" in numbers:
        brief["route_a_all_passed"] = numbers["route_a"]["all_passed"]
        brief["route_a_max_err"] = numbers["route_a"]["max_abs_error"]
        brief["route_a_richardson"] = numbers["route_a"]["richardson_rw_extrapolated"]
    if "route_b" in numbers:
        brief["route_b_max_err"] = numbers["route_b"]["max_abs_error"]
        brief["route_b_any_passed"] = numbers["route_b"]["any_passed"]
    if "wave_field" in numbers:
        brief["wave_field_angle"] = numbers["wave_field"]["measured_half_angle_deg"]
    if "kernel" in numbers:
        brief["kernel_fs_residual"] = numbers["kernel"]["free_surface_condition"]
    print(json.dumps(brief, indent=1))
    print("sha256(numbers.json) =", sha256_file(OUT / "numbers.json"))


if __name__ == "__main__":
    sys.exit(main())