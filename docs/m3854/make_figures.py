#!/usr/bin/env python3
"""Figures for the m3236 -> main heal mission (m3854).

Two figures, both derived from committed evidence only:

``fig_m3236_convergence.png``
    From ``docs/m3236/outputs/m3236/dean_{32,64,128}.json`` (the benchmark
    output that m3236 shipped): the Dean-Ursell-Ogilvie diffraction errors
    against panel count.  The transmission-coefficient error and the
    Haskind-Newman reciprocity residual both fall with the panel count; a
    second-order reference line (slope -2) is drawn through the coarsest point
    so the observed slope can be read off the log-log axes.

``fig_m3236_registry_deviation.png``
    From the merged registry check (``artifacts/m3236_fast_tier.json``): the
    signed relative deviation of each of the eleven Dean rows against the value
    the documentation quotes, with the registry tolerance band.  This is the
    figure that decides whether the merged (hand-resolved) registry still
    reproduces the numbers it declares.

Run with ``PYTHONPATH=src python3 docs/m3854/make_figures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "m3854"
EVIDENCE = ROOT / "docs" / "m3236" / "outputs" / "m3236"
FAST_TIER = ROOT / "artifacts" / "m3236_fast_tier.json"


def convergence_figure() -> Path:
    panels = [32, 64, 128]
    rows = {
        n: json.loads((EVIDENCE / f"dean_{n}.json").read_text()) for n in panels
    }
    transmission = [rows[n]["circular_cylinder"]["panel_transmission_error"] for n in panels]
    haskind = [
        rows[n]["circular_cylinder"]["haskind_diffraction_force_x_relative_difference"]
        for n in panels
    ]
    ellipse_reflection = [rows[n]["elliptic_cylinder"]["reflection_modulus"] for n in panels]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))

    ax = axes[0]
    ax.loglog(panels, transmission, "o-", label=r"transmission error $|T-T_{\rm exact}|$")
    ax.loglog(panels, haskind, "s--", label="Haskind-Newman reciprocity residual")
    ref = transmission[0] * (np.asarray(panels) / panels[0]) ** -2.0
    ax.loglog(panels, ref, ":", color="0.5", label=r"$N^{-2}$ reference")
    ax.set_xlabel("panels on the submerged circle")
    ax.set_ylabel("relative error")
    ax.set_title("Dean cylinder, ka=0.6: discretisation error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xticks(panels)
    ax.set_xticklabels([str(n) for n in panels])

    ax = axes[1]
    ax.semilogx(panels, ellipse_reflection, "^-", color="tab:red")
    ax.axhline(0.09751, color="0.4", ls=":", label="registry value (64 panels)")
    ax.set_xlabel("panels on the elliptic cylinder")
    ax.set_ylabel(r"reflected wave modulus $|R|$")
    ax.set_title("Elliptic cylinder a=1.4 b=0.7: symmetry breaking survives")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xticks(panels)
    ax.set_xticklabels([str(n) for n in panels])
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = OUT / "fig_m3236_convergence.png"
    fig.savefig(path, dpi=95)
    plt.close(fig)
    return path


def deviation_figure() -> Path:
    report = json.loads(FAST_TIER.read_text())
    rows = [row for row in report["checks"] if row["key"].startswith("dean")]
    report_entries = {
        entry["key"]: entry
        for entry in json.loads(
            (ROOT / "artifacts" / "m3236_registry_entries.json").read_text()
        )["entries"]
    }
    labels = [row["key"].removeprefix("dean_") for row in rows]
    deviations = [abs(row["relative_deviation"]) for row in rows]
    tolerances = [report_entries[row["key"]]["tolerance"] for row in rows]
    kinds = [report_entries[row["key"]]["kind"] for row in rows]

    ypos = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = ["tab:blue" if kind == "value" else "tab:gray" for kind in kinds]
    ax.barh(ypos, deviations, color=colors, label="|measured - declared| / declared")
    ax.scatter(tolerances, ypos, marker="|", s=180, color="crimson", label="registry tolerance")
    ax.set_xscale("log")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("relative deviation (log scale)")
    ax.set_title(
        "Merged registry: 11 Dean rows vs the numbers table 7 quotes\n"
        f"verdict {report['summary']['verdict']}, "
        f"{report['summary']['checked']} rows, {report['summary']['drifted']} drift"
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="x", which="both", alpha=0.3)
    fig.tight_layout()
    path = OUT / "fig_m3236_registry_deviation.png"
    fig.savefig(path, dpi=95)
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (convergence_figure(), deviation_figure()):
        print(path, path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()