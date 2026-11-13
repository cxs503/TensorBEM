#!/usr/bin/env python3
"""Figures for the m17318 board audit of m17262 (CASE-BEM-SPHERE-ADDED-MASS).

Data sources (no hand-typed numbers):
  references/sphere_added_mass_sub3.json   subject artifact, sub3 (1280 panels)
  references/sphere_added_mass_sub4.json   subject artifact, sub4 (5120 panels)
  artifacts/auditor_rerun_sub1.json        auditor independent rerun (80 panels)
  artifacts/auditor_rerun_sub2.json        auditor independent rerun (320 panels)
  artifacts/auditor_sub4_cluster.json      auditor cluster rerun (optional, if fetched)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
REFS = HERE / "references"
ART = HERE / "artifacts"


def load_levels() -> list[dict]:
    rows = []
    for p in (ART / "auditor_rerun_sub1.json", ART / "auditor_rerun_sub2.json",
              REFS / "sphere_added_mass_sub3.json", REFS / "sphere_added_mass_sub4.json",
              ART / "auditor_sub4_cluster.json"):
        if p.exists():
            d = json.loads(p.read_text())
            d["_source"] = str(p.relative_to(HERE))
            rows.append(d)
    # one row per panel count, auditor rerun preferred
    best: dict[int, dict] = {}
    for r in rows:
        n = int(r["panels"])
        if n not in best or "auditor" in r["_source"]:
            best[n] = r
    return [best[n] for n in sorted(best)]


def fig_convergence(levels: list[dict]) -> Path:
    panels = [r["panels"] for r in levels]
    err = [r["added_mass_relative_error"] for r in levels]
    l2 = [r["normal_derivative_relative_l2_error"] for r in levels]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(panels, err, "o-", color="#1f4e79", lw=1.8, label="added-mass rel. error vs $0.5\\rho V$")
    ax.loglog(panels, l2, "s--", color="#c00000", lw=1.2, alpha=0.8,
              label=r"flux $\partial\phi/\partial n$ rel. $L_2$ error")
    ax.axhline(0.03, color="k", ls=":", lw=1.2)
    ax.annotate("mission tolerance 3%", xy=(panels[0], 0.03), xytext=(panels[0], 0.037),
                fontsize=8)
    # slope-2 reference guide anchored at the finest level
    n0, e0 = panels[-1], err[-1]
    guide = [e0 * (n / n0) ** -1.0 for n in (panels[0], panels[-1])]
    ax.loglog([panels[0], panels[-1]], guide, color="0.5", lw=1.0, ls="-.",
              label=r"$O(N^{-1})$ guide (panel area $\propto N^{-1}$)")
    for n, e in zip(panels, err):
        ax.annotate(f"{e*100:.3f}%", xy=(n, e), xytext=(4, 5), textcoords="offset points",
                    fontsize=7)
    ax.set_xlabel("panels (icosphere, flat faceted)")
    ax.set_ylabel("relative error")
    ax.set_title("CASE-BEM-SPHERE-ADDED-MASS: refinement ladder (auditor-verified)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower left")
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "fig_convergence.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_parity(levels: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ref = [r["exact_added_mass"] for r in levels]
    got = [r["added_mass"] for r in levels]
    ax.plot([2000, 2200], [2000, 2200], color="k", lw=1, ls="--", label="parity $got=ref$")
    ax.scatter(ref, got, s=[18 + 0.02 * r["panels"] for r in levels], color="#1f4e79", zorder=3)
    for r in levels:
        ax.annotate(f"{r['panels']}p  {r['added_mass_relative_error']*100:.3f}%",
                    xy=(r["exact_added_mass"], r["added_mass"]),
                    xytext=(-96, -14), textcoords="offset points", fontsize=7)
    ax.set_xlabel(r"ref: $m_A = \frac{1}{2}\rho V$  [kg]  (Lamb 1932 §6.10)")
    ax.set_ylabel("got: BEM added mass [kg]")
    ax.set_title("ref vs got, flat-faceted icosphere, $\\rho$=1025, R=1 m")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "fig_parity.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    levels = load_levels()
    for f in (fig_convergence(levels), fig_parity(levels)):
        print("wrote", f.relative_to(HERE))
    print("levels:", [(r["panels"], r["added_mass_relative_error"], r["_source"]) for r in levels])


if __name__ == "__main__":
    main()