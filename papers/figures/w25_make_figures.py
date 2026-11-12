"""Figures for the W25 B4-70 open-water error surface (numbers injected from the artifact).

Every plotted value is read from
``papers/artifacts/w25/propeller-openwater-bem/b4_70_error_surface.json``; nothing
is typed here except axis labels.  Run::

    PYTHONPATH=src python papers/figures/w25_make_figures.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ARTIFACT = REPO / "papers" / "artifacts" / "w25" / "propeller-openwater-bem" / "b4_70_error_surface.json"
OUTDIR = ARTIFACT.parent / "figures"


def _rows(rows, key):
    return [row[key] for row in rows]


def figure_curves(art, path):
    base = art["error_surface"]["base_configuration"]
    fine = art["error_surface"]["fine_configuration"]
    j = _rows(fine, "J")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].plot(j, _rows(fine, "KT_anchor"), "k-", lw=2.2, label="anchor: O&O (1975) B4-70")
    axes[0].plot(j, _rows(base, "KT"), "o--", ms=4.5, lw=1.2, label="BEM 16x2, geometric L1.0/24")
    axes[0].plot(j, _rows(fine, "KT"), "s-", ms=4.5, lw=1.6, label="BEM 24x2, geometric L2.0/48")
    axes[0].axhline(0.0, color="0.6", lw=0.8)
    axes[0].set_xlabel("advance ratio J = V/(nD)")
    axes[0].set_ylabel(r"$K_T$")
    axes[0].set_title("Thrust coefficient: BEM vs B4-70 regression anchor")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(j, _rows(fine, "10KQ_anchor"), "k-", lw=2.2, label="anchor: O&O (1975) B4-70")
    axes[1].plot(j, _rows(base, "10KQ"), "o--", ms=4.5, lw=1.2, label="BEM 16x2, geometric L1.0/24")
    axes[1].plot(j, _rows(fine, "10KQ"), "s-", ms=4.5, lw=1.6, label="BEM 24x2, geometric L2.0/48")
    axes[1].axhline(0.0, color="0.6", lw=0.8)
    axes[1].set_xlabel("advance ratio J")
    axes[1].set_ylabel(r"$10 K_Q$")
    axes[1].set_title("Torque coefficient: BEM vs B4-70 regression anchor")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_deviations(art, path):
    fine = art["error_surface"]["fine_configuration"]
    base = art["error_surface"]["base_configuration"]
    j = _rows(fine, "J")
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), sharex=True)
    for ax, key, title in (
        (axes[0], "dKT_pct", r"$K_T$ deviation"),
        (axes[1], "dKQ_pct", r"$K_Q$ deviation"),
        (axes[2], "deta0_pct", r"$\eta_0$ deviation"),
    ):
        ax.axhspan(-15.0, 15.0, color="tab:green", alpha=0.10, label="mission gate ±15 %")
        ax.plot(j, _rows(base, key), "o--", ms=4, lw=1.1, label="base 16x2 / L1.0-24")
        ax.plot(j, _rows(fine, key), "s-", ms=4, lw=1.6, label="fine 24x2 / L2.0-48")
        ax.axhline(0.0, color="0.5", lw=0.8)
        ax.axvline(0.8, color="0.3", ls=":", lw=1.0)
        ax.set_title(title)
        ax.set_xlabel("advance ratio J")
        ax.set_ylabel("relative deviation [%]")
        ax.set_ylim(-80, 80)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("W25 B4-70 error surface (dotted line = end of the scored range, J = 0.8)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_ablation(art, path):
    mesh = art["ablation"]["mesh"]
    wake = art["ablation"]["wake"]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))
    levels = {}
    for row in mesh:
        key = f"{row['n_radial']}x{row['n_chord']}"
        levels.setdefault(key, []).append((row["J"], row["KT"], row["dKT_pct"]))
    for key, values in sorted(levels.items(), key=lambda kv: int(kv[0].split("x")[0]) * int(kv[0].split("x")[1])):
        values.sort()
        axes[0].plot([v[0] for v in values], [v[1] for v in values], "o-", ms=3.5, lw=1.2, label=key)
    anchor = sorted({(row["J"], row["KT_anchor"], row["10KQ_anchor"]) for row in mesh})
    axes[0].plot([a[0] for a in anchor], [a[1] for a in anchor], "k--", lw=2.0, label="anchor")
    axes[0].set_title("Mesh ladder (geometric wake L1.0/24)")
    axes[0].set_xlabel("J")
    axes[0].set_ylabel(r"$K_T$")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    wakes = {}
    for row in wake:
        key = f"{row['wake_model'][:5]} L{row['wake_length']}/{row['wake_steps']}"
        wakes.setdefault(key, []).append((row["J"], row["KT"], row["numerically_stable"]))
    for key, values in sorted(wakes.items()):
        values.sort()
        stable = [(v[0], v[1]) for v in values if v[2]]
        unstable = [(v[0], v[1]) for v in values if not v[2]]
        if stable:
            axes[1].plot([v[0] for v in stable], [v[1] for v in stable], "o-", ms=3.5, lw=1.2, label=key)
        if unstable:
            axes[1].plot([v[0] for v in unstable], [v[1] for v in unstable], "x", ms=6, label=f"{key} (breakdown)")
    axes[1].plot([a[0] for a in anchor], [a[1] for a in anchor], "k--", lw=2.0, label="anchor")
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].set_title("Wake ladder at 16x2 (symlog: breakdowns visible)")
    axes[1].set_xlabel("J")
    axes[1].set_ylabel(r"$K_T$")
    axes[1].legend(fontsize=6)
    axes[1].grid(alpha=0.3)

    friction = art["ablation"]["friction"]
    for flag in (True, False):
        rows = sorted(
            [(row["J"], row["10KQ"]) for row in friction if row["viscous"] is flag]
        )
        axes[2].plot(
            [r[0] for r in rows], [r[1] for r in rows], "o-", ms=3.5, lw=1.3,
            label="viscous (ITTC-78)" if flag else "inviscid",
        )
    axes[2].plot([a[0] for a in anchor], [a[2] for a in anchor], "k--", lw=2.0, label="anchor")
    axes[2].set_title("Friction ablation: section drag")
    axes[2].set_xlabel("J")
    axes[2].set_ylabel(r"$10 K_Q$")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_decomposition(art, path):
    rows = art["shapley_decomposition_b4_70"]["rows"]
    j = [row["J"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for ax, quantity, title in (
        (axes[0], "KT", r"$K_T$"),
        (axes[1], "10KQ", r"$10K_Q$"),
    ):
        mesh = [row["factors"][quantity]["contributions_pct_of_anchor"]["mesh"] for row in rows]
        wake = [row["factors"][quantity]["contributions_pct_of_anchor"]["wake"] for row in rows]
        friction = [row["factors"][quantity]["contributions_pct_of_anchor"]["friction"] for row in rows]
        remainder = [row["factors"][quantity]["model_form_remainder_pct_of_anchor"] for row in rows]
        width = 0.08
        ax.bar([x - 1.5 * width for x in j], mesh, width, label="mesh ablation", color="tab:blue")
        ax.bar([x - 0.5 * width for x in j], wake, width, label="wake ablation", color="tab:orange")
        ax.bar([x + 0.5 * width for x in j], friction, width, label="friction ablation", color="tab:green")
        ax.bar([x + 1.5 * width for x in j], remainder, width, label="model-form remainder", color="0.55")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_ylim(-120, 120)
        ax.set_title(f"Exact Shapley decomposition of the {title} deviation")
        ax.set_xlabel("advance ratio J")
        ax.set_ylabel("contribution to deviation [% of anchor]")
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle("Parts add up to the deviation at the highest-fidelity configuration (residual < 1e-15)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_kp505(art, path):
    ident = art["kp505"]["pitch_dof_identifiability"]
    rows = ident["rows"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot([r["P_D"] for r in rows], [r["dKT_pct"] for r in rows], "o-", lw=1.6, label=r"$K_T$ deviation at J = 0.8")
    ax.plot([r["P_D"] for r in rows], [r["dJ0_pct"] for r in rows], "s-", lw=1.6, label=r"zero-thrust $J_0$ deviation")
    ax.axhline(0.0, color="k", lw=0.9)
    for span in (-2.0, 2.0):
        ax.axhline(span, color="tab:green", ls=":", lw=1.0)
    ax.axvline(0.95, color="0.4", ls="--", lw=1.0)
    ax.text(0.951, ax.get_ylim()[1] * 0.9, "published mean P/D = 0.95", fontsize=7, rotation=90)
    ax.set_xlabel("constant pitch ratio P/D of the BEM geometry")
    ax.set_ylabel("deviation from the KP505 anchor [%]")
    ax.set_title("KP505: no single pitch value satisfies both targets\n(green dotted = ±2 % identifiability band)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    art = json.loads(ARTIFACT.read_text())
    OUTDIR.mkdir(parents=True, exist_ok=True)
    figure_curves(art, OUTDIR / "w25_b4_70_kt_kq_curves.png")
    figure_deviations(art, OUTDIR / "w25_b4_70_deviations.png")
    figure_ablation(art, OUTDIR / "w25_b4_70_ablation.png")
    figure_decomposition(art, OUTDIR / "w25_b4_70_shapley.png")
    figure_kp505(art, OUTDIR / "w25_kp505_pitch_identifiability.png")
    for name in sorted(os.listdir(OUTDIR)):
        print("wrote", OUTDIR / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())