#!/usr/bin/env python3
"""Figures for the finite-depth cross-validation refinement study (XVAL anchor).

Reads the two producer artefacts and writes PNGs; no numbers are typed by hand.

* ``truncation_cap_study.json`` -- A33/B33 vs the evanescent-mode cap
* ``refinement_spine.json``      -- A33/B33 vs mesh level + Richardson limit

    python3 make_figures_refinement.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / ".." / ".." / ".." / "docs" / "finite_depth_xval_refinement"
OUT = OUT.resolve()


def _load(name: str) -> dict | None:
    path = HERE / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def fig_mode_cap(cap: dict) -> Path:
    verdicts = cap.get("verdicts") or []
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, key, label in zip(axes, ("am_vs_cap", "b_vs_cap"),
                              ("heave added mass A33 [kg]", "heave damping B33 [kg s$^{-1}$]")):
        for v in verdicts:
            caps = v["caps"]
            vals = v[key]
            ax.plot(caps, vals, "o-", label=f"h/R={v['depth_ratio']:g}, ka={v['ka']:g}")
            ref = v.get("cpt") or {}
            if ref.get("added_mass_heave") and key == "am_vs_cap":
                ax.axhline(ref["added_mass_heave"], ls="--", lw=0.9,
                           color="0.30")
            if ref.get("damping_heave") and key == "b_vs_cap":
                ax.axhline(ref["damping_heave"], ls="--", lw=0.9,
                           color="0.30")
            ax.axvline(v["auto_mode_count"], ls=":", lw=0.8, color="0.55")
        ax.set_xscale("log")
        ax.set_xlabel("evanescent-mode cap $n_{modes}$")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Finite-depth sphere: eigenseries truncation vs automatic mode budget\n"
                 "(dashed = capytaine 24x48 reference, dotted = automatic requirement)",
                 fontsize=9)
    fig.tight_layout()
    path = OUT / "fig_mode_cap.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_refinement(spine: dict) -> Path:
    stations = [s for s in spine["stations"] if len(s["points"]) >= 2]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, key, label in zip(axes, ("added_mass_heave", "damping_heave"),
                              ("A33 [kg]", "B33 [kg s$^{-1}$]")):
        for st in stations:
            labels = [f"h/R={st['depth_ratio']:g} ka={st['ka']:g}"]
            edge = [p["n_panels"] ** -0.5 for p in st["points"]]
            vals = [p[key] for p in st["points"]]
            ax.plot(edge, vals, "o-", label=labels[0])
            ex = st.get("extrapolated") or {}
            if ex.get(key):
                ax.plot([0.0], [ex[key]], "s", mfc="none",
                        label=labels[0] + " Richardson")
            ref = st.get("cpt") or {}
            if ref.get(key):
                ax.axhline(ref[key], ls="--", lw=0.9, color="0.30")
        ax.set_xlabel("panel edge $\\propto n_{panels}^{-1/2}$")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Finite-depth sphere: mesh refinement at the automatic mode count\n"
                 "(hollow square = Richardson limit, dashed = capytaine 24x48)",
                 fontsize=9)
    fig.tight_layout()
    path = OUT / "fig_refinement_spine.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    cap = _load("truncation_cap_study.json")
    if cap and cap.get("verdicts"):
        made.append(fig_mode_cap(cap))
    spine = _load("refinement_spine.json")
    if spine and spine.get("stations"):
        made.append(fig_refinement(spine))
    for p in made:
        print("wrote", p)
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())