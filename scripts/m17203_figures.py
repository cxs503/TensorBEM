"""CASE-BEM-DEAN-DIFFRACTION (m17203) — figures.

Reads artifacts/m17203_dean_diffraction/stage_*.json and the capytaine
cross-check, rebuilds the scattered field from the MFS strengths, and writes
PNGs into artifacts/m17203_dean_diffraction/figures/.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorbem.benchmarks.ursell_cylinder import submerged_circle_mesh
from tensorbem.free_surface import green_function
from tensorbem.free_surface_solver import diffraction_reference

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "m17203_dean_diffraction"
FIG = ART / "figures"
FIG.mkdir(parents=True, exist_ok=True)

A, F, KA, AMP, RHO, G = 1.0, 2.0, 0.6, 1.0, 1025.0, 9.80665
NORMALISER = RHO * G * A * AMP
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight"})


def load(stage: str) -> list[dict]:
    return json.loads((ART / f"stage_{stage}.json").read_text())["rows"]


def modulus(pair) -> float:
    return math.hypot(pair[0], pair[1])


# ---------------------------------------------------------------- figure 1
def fig_force_vs_ka() -> Path:
    rows = sorted(load("ka"), key=lambda r: r["ka"])
    ka = np.array([r["ka"] for r in rows])
    ref = np.array([r["exact_force_x_modulus"] for r in rows]) / NORMALISER
    panel = np.array([modulus(r["panel_force_x"]) for r in rows]) / NORMALISER
    hask = np.array([modulus(r["haskind_force_x"]) for r in rows]) / NORMALISER

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(ka, ref, "k-", lw=1.6, label="Ursell 1950 / Ogilvie 1963 multipole (ref)")
    ax.plot(ka, panel, "o", ms=5, mfc="none", label="TensorBEM 2-D panel, N=128")
    ax.plot(ka, hask, "x", ms=6, label="TensorBEM Haskind reciprocity (radiation only)")
    ax.set_xlabel(r"$ka$")
    ax.set_ylabel(r"$|F_x|/(\rho g a A)$")
    ax.set_title("2-D deep-water wave load on a submerged cylinder, f/a = 2")
    ax.legend(fontsize=8)
    ax2 = ax.twinx()
    ax2.semilogy(ka, [r["panel_force_x_relative_error"] * 100 for r in rows], "s--",
                 ms=3, color="tab:red", alpha=0.7)
    ax2.set_ylabel("panel route relative error [%] (log)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.grid(False)
    path = FIG / "fig1_force_vs_ka.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- figure 2
def fig_convergence() -> Path:
    rows = sorted(load("panels"), key=lambda r: r["panels"])
    n = np.array([r["panels"] for r in rows], dtype=float)
    fx = np.array([r["panel_force_x_relative_error"] for r in rows]) * 100
    ph = np.array([r["panel_transmission_phase_error_degrees"] for r in rows])
    mfs = np.array([r["mfs_force_x_relative_error"] for r in rows]) * 100

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.loglog(n, fx, "o-", label=r"$|F_x|$ relative error")
    ax.loglog(n, ph, "s-", label=r"transmission phase error [deg]")
    ax.loglog(n, mfs, "^--", label=r"MFS route $|F_x|$ error")
    ax.axhline(5.0, color="tab:red", ls=":", lw=1.2, label="mission tolerance 5%")
    ax.set_xlabel("panels on the circle")
    ax.set_ylabel("error  [%]  /  [deg]  (log)")
    ax.set_title(r"Mesh convergence at $ka=0.6$, $f/a=2$ (nodes/panel = 24)")
    ax.legend(fontsize=8)
    path = FIG / "fig2_convergence.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- figure 3
def fig_error_study() -> Path:
    ka_rows = sorted(load("ka"), key=lambda r: r["ka"])
    f_rows = sorted(load("submergence"), key=lambda r: r["submergence_over_radius"])
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.6))

    ax = axs[0]
    ax.semilogy([r["ka"] for r in ka_rows],
                [r["panel_force_x_relative_error"] * 100 for r in ka_rows], "o-",
                label=r"$|F_x|$ error")
    ax.semilogy([r["ka"] for r in ka_rows],
                [r["panel_transmission_phase_error_degrees"] / 26.655 for r in ka_rows],
                "s--", label="phase error / 26.655 deg")
    ax.axhline(5.0, color="tab:red", ls=":", label="5% tolerance")
    ax.set_xlabel(r"$ka$")
    ax.set_ylabel("error [%]")
    ax.set_title("frequency sweep (N=128)")

    ax = axs[1]
    kf = [r["wavenumber_times_submergence"] for r in f_rows]
    ax.plot(kf, [r["panel_force_x_relative_error"] * 100 for r in f_rows], "o-",
            label=r"$|F_x|$ error [%]")
    ax.plot(kf, [r["morison_over_exact"] for r in f_rows], "s--",
            label=r"Morison $C_M=2$ limit / exact")
    ax.axhline(5.0, color="tab:red", ls=":", label="5% tolerance")
    ax.set_xlabel(r"$kf$  (submergence sweep, $f/a$ = 1.5 ... 8)")
    ax.set_title("submergence sweep (N=128)")
    ax.legend(fontsize=8)
    path = FIG / "fig3_error_sweeps.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- figure 4
def fig_capytaine() -> Path:
    data = json.loads((ART / "capytaine_crosscheck.json").read_text())
    rows = sorted(data["rows"], key=lambda r: r["length"])
    ll = [r["length_over_lambda"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.semilogx(ll, [r["force_x_relative_error"] * 100 for r in rows], "o-",
                label=r"capytaine 3-D $|F_x|/L$ vs 2-D reference")
    ax.semilogx(ll, [r["force_z_relative_error"] * 100 for r in rows], "s-",
                label=r"capytaine 3-D $|F_z|/L$ vs 2-D reference")
    ax.axhline(5.0, color="tab:red", ls=":", label="5% tolerance")
    ax.set_xlabel(r"cylinder length $L/\lambda$")
    ax.set_ylabel("relative error [%]")
    ax.set_title("independent software cross-check: 3-D BEM cylinder -> 2-D limit")
    ax.legend(fontsize=8)
    path = FIG / "fig4_capytaine_limit.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- figure 5
def fig_wave_field() -> Path:
    nu = KA / A
    omega = math.sqrt(G * nu)
    mesh = submerged_circle_mesh(128, A, F)
    mfs = diffraction_reference(mesh, nu, AMP, source_count=32, density=RHO, gravity=G)
    centre = np.mean(mesh.centers, axis=0)
    radius = float(np.mean(np.linalg.norm(mesh.centers - centre, axis=1)))
    n_src = 32
    ang = 2.0 * np.pi * np.arange(n_src) / n_src
    src = centre + 0.55 * radius * np.column_stack((np.cos(ang), np.sin(ang)))
    q = np.asarray(mfs.source_density, dtype=complex)

    # (a) free-surface elevation envelope sampled 5 cm below the mean level
    xs = np.linspace(-15.0, 15.0, 1501)
    zs = np.full_like(xs, -0.05)
    phi_s = np.zeros_like(xs, dtype=complex)
    for j in range(n_src):
        phi_s += q[j] * green_function(xs, zs, np.full_like(xs, src[j, 0]),
                                       np.full_like(xs, src[j, 1]), nu)
    eta = AMP * np.exp(1j * nu * xs) + (1j * omega / G) * phi_s
    up = (xs > -15) & (xs < -6)
    dn = (xs > 6) & (xs < 15)

    # (b) field map of |phi_total| in the fluid
    gx = np.linspace(-12.0, 12.0, 361)
    gz = np.linspace(-8.0, -0.05, 241)
    X, Z = np.meshgrid(gx, gz)
    phi_total = AMP * (-1j * G / omega) * np.exp(nu * Z + 1j * nu * X)
    for j in range(n_src):
        phi_total = phi_total + q[j] * green_function(X, Z, np.full_like(X, src[j, 0]),
                                                     np.full_like(X, src[j, 1]), nu)
    inside = (X ** 2 + (Z + F) ** 2) < A ** 2
    mag = np.abs(phi_total)
    mag = np.ma.masked_where(inside, mag)

    fig, axs = plt.subplots(2, 1, figsize=(7.0, 6.6),
                            gridspec_kw={"height_ratios": [1.0, 1.15]})
    ax = axs[0]
    ax.plot(xs, np.abs(eta), lw=1.0, label=r"$|\eta(x)|$ total elevation")
    ax.axhline(np.mean(np.abs(eta[up])), color="tab:green", ls="--", lw=1,
               label="upstream mean = %.4f (Dean: R=0)" % np.mean(np.abs(eta[up])))
    ax.axhline(np.mean(np.abs(eta[dn])), color="tab:orange", ls=":", lw=1.2,
               label="downstream mean = %.4f (|T|=1)" % np.mean(np.abs(eta[dn])))
    ax.set_ylim(0.0, 1.6)
    ax.set_xlabel("x [m]  (sampled 5 cm below the free surface)")
    ax.set_ylabel(r"$|\eta|$ [m]")
    ax.set_title("Dean 1948 zero reflection: no upstream standing-wave pattern")
    ax.legend(fontsize=7.5, loc="lower center", ncol=3)

    ax = axs[1]
    cs = ax.contourf(X, Z, mag, levels=24, cmap="viridis")
    ax.add_patch(plt.Circle((0.0, -F), A, color="0.85", ec="k", lw=1.0, zorder=3))
    ax.axhline(0.0, color="w", lw=1.4)
    fig.colorbar(cs, ax=ax, label=r"$|\phi_{tot}|$ [m$^2$/s]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(r"total potential magnitude, $ka=0.6$, $f/a=2$ (incident + scattered, MFS)")
    fig.savefig(FIG / "fig5_wave_field.png")
    plt.close(fig)
    return FIG / "fig5_wave_field.png"


if __name__ == "__main__":
    for path in (fig_force_vs_ka(), fig_convergence(), fig_error_study(),
                 fig_capytaine(), fig_wave_field()):
        print("wrote", path)