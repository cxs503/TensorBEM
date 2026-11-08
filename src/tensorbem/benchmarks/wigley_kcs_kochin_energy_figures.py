"""Figures for the Wigley/KCS radiation-diffraction verification study (m6656).

Every panel is drawn from the numbers in ``study.json`` only -- no quantity is
recomputed here -- so a figure can never disagree with the artefact that
supports the report.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_DPI = 140


def _save(fig: plt.Figure, path: Path) -> dict[str, str]:
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    return {"name": path.stem, "file": path.name}


def fig_wigley_cw(study: dict[str, Any], out: Path) -> dict[str, str]:
    block = study["wigley_wave_resistance"]
    rows = block["rows"]
    froude = sorted({r["froude"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    michel = [next(r["cw_michell"] for r in rows if r["froude"] == fn) for fn in froude]
    ax.plot(froude, michel, "k--", marker="s", label="Michell (closed form)")
    for count in sorted({r["panel_count"] for r in rows}):
        sel = sorted([r for r in rows if r["panel_count"] == count], key=lambda r: r["froude"])
        ax.plot(
            [r["froude"] for r in sel],
            [r["cw_panel"] for r in sel],
            marker="o",
            ls="-",
            label=f"panel Kochin, N={count}",
        )
    ax.set_xlabel(r"$F_n$")
    ax.set_ylabel(r"$C_w$")
    ax.set_title("Wigley wave resistance: panel Kochin vs Michell")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for fn in froude:
        sel = sorted([r for r in rows if r["froude"] == fn], key=lambda r: r["panel_count"])
        hs = [r["panel_count"] ** -0.5 for r in sel]
        errs = [r["cw_relative_error"] for r in sel]
        ax.loglog(hs, errs, marker="o", label=rf"$F_n={fn:g}$")
        order = block["orders_vs_h"].get(str(fn))
        if order is not None and len(hs) >= 2:
            ref = errs[-1] * (np.asarray(hs) / hs[-1]) ** order
            ax.loglog(hs, ref, ":", lw=1, label=rf"order $h^{{{order:.2f}}}$")
    ax.set_xlabel(r"panel size $h \propto N^{-1/2}$")
    ax.set_ylabel(r"$|C_w^{panel}-C_w^{Michell}|/C_w^{Michell}$")
    ax.set_title("Panel convergence of the Kochin quadrature")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return _save(fig, out / "fig1_wigley_cw_panel_vs_michell.png")


def fig_wigley_energy(study: dict[str, Any], out: Path) -> dict[str, str]:
    block = study["wigley_radiation"]
    rows = block["rows"]
    labels = {(r["n_sections"], r["n_levels"]) for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(labels)))

    ax = axes[0]
    for color, (ns, nl) in zip(colors, sorted(labels)):
        sel = sorted(
            [r for r in rows if r["n_sections"] == ns and r["n_levels"] == nl],
            key=lambda r: r["omega_sqrt_L_over_g"],
        )
        om = [r["omega_sqrt_L_over_g"] for r in sel]
        ax.plot(om, [r["damping_near"][0][0] for r in sel], "o-", color=color,
                label=f"near N={2*sel[0]['half_panels']}")
        ax.plot(om, [r["damping_far"][0][0] for r in sel], "x--", color=color, alpha=0.8)
    ax.set_xlabel(r"$\omega\sqrt{L/g}$")
    ax.set_ylabel(r"$B_{33}$")
    ax.set_title("Wigley heave damping: near field (o) vs Kochin far field (x)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    first = min(r["omega_sqrt_L_over_g"] for r in rows)
    chain = sorted(
        [r for r in rows if abs(r["omega_sqrt_L_over_g"] - first) < 1e-9],
        key=lambda r: r["full_panels"],
    )
    hs = [r["full_panels"] ** -0.5 for r in chain]
    near = [r["damping_near"][0][0] for r in chain]
    diffs = [abs(near[i + 1] - near[i]) for i in range(len(near) - 1)]
    ax.loglog(hs[1:], diffs, "o-", label=r"$|B_{33}^{(k+1)}-B_{33}^{(k)}|$")
    order = block["criteria"]["convergence_orders"].get("heave_near")
    if order is not None:
        ref = diffs[-1] * (np.asarray(hs[1:]) / hs[-1]) ** order
        ax.loglog(hs[1:], ref, "k:", label=rf"order $h^{{{order:.2f}}}$")
    ax.set_xlabel(r"panel size $h \propto N^{-1/2}$")
    ax.set_ylabel("successive damping difference")
    ax.set_title("Mesh convergence of the BEM damping")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return _save(fig, out / "fig2_wigley_kochin_energy_and_convergence.png")


def fig_kcs_excitation(study: dict[str, Any], out: Path) -> dict[str, str]:
    block = study["kcs_diffraction"]
    rows = block["rows"]
    dofs = ("surge", "sway", "heave", "roll", "pitch", "yaw")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for ax, heading, names in (
        (axes[0], "head", ("surge", "heave", "pitch")),
        (axes[1], "beam", ("sway", "roll", "yaw")),
    ):
        sel = sorted([r for r in rows if r["heading"] == heading], key=lambda r: r["lambda_over_L"])
        if not sel:
            continue
        for name in names:
            idx = dofs.index(name)
            ratios = [r["lambda_over_L"] for r in sel]
            ax.plot(ratios, [r["amplitude_pressure"][idx] for r in sel], "o-", label=f"{name} (pressure)")
            ax.plot(ratios, [r["amplitude_haskind_boundary"][idx] for r in sel], "x:",
                    label=f"{name} (Haskind near)")
            ax.plot(ratios, [r["amplitude_kochin"][idx] for r in sel], "+--", alpha=0.7,
                    label=f"{name} (Kochin far)")
        ax.set_xlabel(r"$\lambda/L$")
        ax.set_ylabel("exciting-force amplitude (N or N m)")
        ax.set_yscale("log")
        ax.set_title(f"KCS-class hull, {heading} seas")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=6)

    ax = axes[2]
    for heading, marker in (("head", "o"), ("beam", "s")):
        sel = [r for r in rows if r["heading"] == heading]
        if not sel:
            continue
        worst = [
            max(
                e
                for e, keep in zip(r["relative_error_pressure_vs_haskind"], r["significant_dof"])
                if keep
            )
            for r in sel
        ]
        ax.semilogy([r["lambda_over_L"] for r in sel], worst, marker + "-", label=heading)
    ax.axhline(0.02, color="r", ls=":", label="2 % gate")
    ax.set_xlabel(r"$\lambda/L$")
    ax.set_ylabel("max relative offset (significant DOF)")
    ax.set_title("Pressure integral vs Haskind near field")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return _save(fig, out / "fig3_kcs_6dof_excitation_reciprocity.png")


def fig_kcs_fk_limit(study: dict[str, Any], out: Path) -> dict[str, str]:
    block = study["kcs_diffraction"]
    rows = block["long_wave_rows"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    dofs = ("surge", "heave", "pitch", "sway", "roll", "yaw")
    ratios = [r["lambda_over_L"] for r in rows]
    for idx, name in enumerate(dofs):
        vals = [
            r["relative_error_pressure_vs_fk"][idx] if r["significant_dof"][idx] else np.nan
            for r in rows
        ]
        if np.all(np.isnan(vals)):
            continue
        ax.loglog(ratios, vals, "o-", label=name)
    ax.axhline(0.02, color="r", ls=":", label="2 % gate")
    ax.set_xlabel(r"$\lambda/L$ (long-wave limit $\to \infty$)")
    ax.set_ylabel(r"$|F_j^{BEM}-F_j^{FK}|/|F_j^{FK}|$")
    ax.set_title("Long-wave Froude-Krylov limit of the exciting force")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return _save(fig, out / "fig4_kcs_long_wave_fk_limit.png")


def fig_kcs_energy(study: dict[str, Any], out: Path) -> dict[str, str]:
    block = study["kcs_kochin_energy"]
    rows = block["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    om = [r["omega_sqrt_L_over_g"] for r in rows]
    ax.plot(om, [r["damping_near"][0][0] for r in rows], "o-", label=r"$B_{33}$ near")
    ax.plot(om, [r["damping_far"][0][0] for r in rows], "x--", label=r"$B_{33}$ Kochin")
    ax.plot(om, [r["damping_near"][1][1] for r in rows], "s-", label=r"$B_{55}$ near")
    ax.plot(om, [r["damping_far"][1][1] for r in rows], "+--", label=r"$B_{55}$ Kochin")
    ax.set_xlabel(r"$\omega\sqrt{L/g}$")
    ax.set_ylabel("radiation damping")
    ax.set_title("KCS-class hull: near-field vs Kochin damping")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    names = []
    vals = []
    for r in rows:
        for i, mode in enumerate(r["modes"]):
            names.append(f"{mode}/{r['omega_sqrt_L_over_g']:g}")
            vals.append(r["damping_relative_error"][i])
    ax.bar(range(len(vals)), vals, color="steelblue")
    ax.axhline(0.02, color="r", ls=":", label="2 % gate")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("relative offset")
    ax.set_title("Energy-identity error per mode / frequency")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    return _save(fig, out / "fig5_kcs_kochin_energy_identity.png")


def make_all_figures(study: dict[str, Any], out: Path) -> list[dict[str, str]]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    figures = [
        fig_wigley_cw(study, out),
        fig_wigley_energy(study, out),
        fig_kcs_excitation(study, out),
        fig_kcs_fk_limit(study, out),
        fig_kcs_energy(study, out),
    ]
    (out / "figures.json").write_text(json.dumps(figures, indent=2), encoding="utf-8")
    return figures
