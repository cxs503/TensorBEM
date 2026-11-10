"""Figures for the NK panel study (mission m10003).

Reads ``artifacts/nk_panel3d_study.json`` (written by
``scripts/nk_panel3d_study.py``) and produces four PNGs:

* ``fig_nk_kernel_checks.png``   -- kernel verification (radiation + FS residual)
* ``fig_kelvin_wake.png``        -- free-surface elevation of a point source
* ``fig_wigley_wave_cuts.png``   -- panel vs thin-ship wave cuts, Fr = 0.3
* ``fig_wigley_resistance.png``  -- resistance vs mesh vs Michell (honest gap)

Every caption number is read from the JSON, nothing is hand-typed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
DOCS = ROOT / "docs" / "m10003"


def fig_kernel_checks(report: dict) -> Path:
    kv = report["kernel_verification"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))

    # (a) theta convergence
    conv = kv["theta_convergence"]
    nt = [c["n_theta"] for c in conv]
    err = [max(c.get("rel_error", 0.0), 1e-12) for c in conv]
    axes[0].semilogy(nt, err, "o-")
    axes[0].set_xlabel(r"$n_\theta$ (Gauss-Legendre)")
    axes[0].set_ylabel("relative error of $G_W$")
    axes[0].set_title("(a) direction-integral convergence")
    axes[0].grid(True, which="both", alpha=0.3)

    # (b) upstream/downstream asymmetry
    labels, ratios = [], []
    for key, val in kv["radiation"].items():
        labels.append(key)
        ratios.append(val["ratio_down_over_up"])
    axes[1].bar(range(len(labels)), ratios, color=["#3b7dd8", "#d84b3b", "#3b7dd8", "#d84b3b"])
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[1].axhline(1.0, color="k", lw=0.8, ls="--")
    axes[1].set_ylabel("amplitude ratio (behind / ahead)")
    axes[1].set_title("(b) radiation direction")
    fs = kv["free_surface_condition"]["halfplane"]
    axes[1].text(
        0.02,
        0.95,
        f"FS residual (max)\n{fs['max_relative_residual']:.2e}",
        transform=axes[1].transAxes,
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    out = DOCS / "fig_nk_kernel_checks.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_kelvin_wake() -> Path:
    data = np.load(ART / "kelvin_wake_field.npz")
    x, y, zeta = data["x"], data["y"], data["zeta"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    lim = np.percentile(np.abs(zeta), 99.0)
    im = ax.pcolormesh(x, y, zeta.T, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
    kappa = 1.0
    for deg in (19.47, -19.47):
        ax.plot(x, np.abs(x) * math.tan(math.radians(deg)), "k--", lw=0.9)
    ax.set_xlabel(r"$x\kappa$  (downstream $\leftarrow$)")
    ax.set_ylabel(r"$y\kappa$")
    ax.set_title("steady NK wave kernel: elevation field of a point source\n"
                 "dashed: classical Kelvin caustic $\\pm19.47^\\circ$")
    fig.colorbar(im, ax=ax, label=r"$G_W\kappa$")
    fig.tight_layout()
    out = DOCS / "fig_kelvin_wake.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_wave_cuts(report: dict) -> Path:
    wc = report["wave_cuts"]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.4), sharex=True)
    for ax, key in zip(axes, wc["cuts_panel"]):
        p = wc["cuts_panel"][key]
        t = wc["cuts_thin_ship"][key]
        ax.plot(p["x"], p["zeta"], label="NK panel solution", lw=1.2)
        ax.plot(t["x"], t["zeta"], "--", label="thin-ship (same kernel)", lw=1.1)
        ax.set_ylabel(r"$\zeta$ [m]")
        ax.set_title(f"cut {key}", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("x [m]")
    fig.suptitle(
        f"Wigley III wave cuts, Fr = {wc['froude']:.2f}, mesh {wc['mesh'][0]}x{wc['mesh'][1]}\n"
        f"max |deviation| = {100 * wc['max_rel_error_vs_thin_ship']:.1f} % of the reference peak",
        fontsize=9,
    )
    fig.tight_layout()
    out = DOCS / "fig_wigley_wave_cuts.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_resistance(report: dict) -> Path:
    ms = report["mesh_study"]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for case in ms["levels"]:
        panels = [m["n_panels"] for m in case["meshes"]]
        cw = [m["cw_near"] for m in case["meshes"]]
        ax.plot(panels, cw, "o-", label=f"NK panel, Fr={case['froude']:.1f}")
        ax.axhline(case["michell_cw"], ls="--", lw=0.9,
                   color=ax.lines[-1].get_color(),
                   label=f"Michell, Fr={case['froude']:.1f}")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlabel("number of hull panels (half body)")
    ax.set_ylabel(r"$C_w = R_w / (\frac{1}{2}\rho U^2 LT)$")
    ax.set_title("Wigley III wave resistance: near-field pressure integral\n"
                 "the panel values do not converge to Michell (honest FAIL)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = DOCS / "fig_wigley_resistance.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    report = json.loads((ART / "nk_panel3d_study.json").read_text())
    outs = [fig_kernel_checks(report), fig_kelvin_wake(), fig_wave_cuts(report), fig_resistance(report)]
    for o in outs:
        print("wrote", o)


if __name__ == "__main__":
    main()