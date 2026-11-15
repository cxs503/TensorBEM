#!/usr/bin/env python3
"""Figures for the CASE-BEM-CUMMINS-WIGLEY verdict-coverage closure (m21960).

Panel A: gate C10's residual against the foreign code's own panel dependence --
the number that decides whether the external anchor's PASS can be read as
"the two codes reproduce one another".
Panel B: gate C5b, the free-decay bar on every restoring mode, with the
estimator-undecidable mode hatched (reported, not scored).
Panel C: the verdict tally before/after this mission (gates 13 -> 15).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorbem.benchmarks import case_bem_cummins_wigley as case  # noqa: E402
from tensorbem.benchmarks import case_bem_cummins_wigley_xval as xval  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"
OUT = ROOT / "docs" / "fig_m21960_cummins_verdict_closure.png"


def main() -> int:
    sweep = case.load_sweep(xval.SWEEP_PATH)
    reference = xval.load_reference()
    floor = xval.reference_noise_floor(sweep, reference)
    gate = xval.external_anchor_gate(sweep, reference)
    decay = case.decay_modes_gate(sweep, audit=case.dof_decay_audit(sweep))
    published = json.loads((ARTIFACT / "case_result.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # --- A: residual vs the reference's own panel dependence -----------------
    ax = axes[0]
    omega = np.asarray(sweep.omega)
    styles = {"added_mass": ("#1f77b4", "added mass"), "damping": ("#d62728", "damping")}
    for quantity, (colour, label) in styles.items():
        series = floor["quantities"][quantity]["per_frequency"]
        ax.semilogx(omega, 100 * np.asarray(series["residual"]), "-", color=colour, lw=1.8,
                    label=f"C10 residual, {label}")
        ax.semilogx(omega, 100 * np.asarray(series["reference_ladder_spread"]), "--", color=colour,
                    lw=1.2, alpha=0.75, label=f"capytaine own panel spread, {label}")
    ax.axhline(100 * case.TOLERANCES["kk"], color="black", lw=1.0, ls=":",
               label=f"case criterion {100 * case.TOLERANCES['kk']:.0f}%")
    ax.set_xlabel("omega [rad/s]")
    ax.set_ylabel("deviation / band amplitude [%]")
    ax.set_title("A. gate C10 vs the reference's own mesh noise")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, loc="upper left")
    worst = gate["values"]["worst"]
    ax.text(
        0.98, 0.02,
        f"worst residual {100 * worst['relative']:.2f}%\n"
        f"({worst['dof_to']}<-{worst['dof_from']}, "
        f"{gate['values']['worst_frequency_rad_s']:.2f} rad/s)\n"
        f"worst ratio to reference noise "
        f"{max(row['worst']['ratio'] for row in floor['quantities'].values()):.2f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        bbox=dict(fc="white", ec="#999999", alpha=0.9),
    )

    # --- B: free decay on every restoring mode ------------------------------
    ax = axes[1]
    names = list(decay["values"]["modes"])
    zeta_d = [decay["values"]["modes"][name]["zeta_decay"] for name in names]
    zeta_f = [decay["values"]["modes"][name]["zeta_frequency_domain"] for name in names]
    x = np.arange(len(names))
    bars = ax.bar(x - 0.2, zeta_d, width=0.4, label="Cummins time-domain decay",
                  color=["#2ca02c" if decay["values"]["modes"][n]["scored"] else "#bbbbbb" for n in names])
    ax.bar(x + 0.2, zeta_f, width=0.4, label="frequency-domain impedance", color="#7f7f7f")
    bar = case.TOLERANCES["free_decay"]
    for index, name in enumerate(names):
        row = decay["values"]["modes"][name]
        rel = row["relative_error"]
        ax.text(
            index, max(zeta_d[index], zeta_f[index]) + 0.004,
            f"{100 * rel:.2f}%", ha="center", fontsize=8,
            fontweight="bold" if row["scored"] else "normal",
        )
        if not row["scored"]:
            bars[index].set_hatch("//")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(scored)" if decay["values"]["modes"][n]["scored"] else f"{n}\n(reported)"
                        for n in names])
    ax.set_ylabel("damping ratio")
    ax.set_title(f"B. gate C5b: free decay, bar {100 * bar:.0f}% (hatched = not scored)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    # --- C: verdict tally ---------------------------------------------------
    ax = axes[2]
    tally = published["verdict"]
    labels = ["gates in verdict", "gates passed"]
    values = [tally["gates_total"], tally["gates_passed"]]
    ax.bar(labels, values, color=["#4c72b0", "#55a868"])
    for index, value in enumerate(values):
        ax.text(index, value + 0.1, str(value), ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(values) + 2)
    ax.set_title("C. published verdict (m21960 folds C10 + C5b in)")
    ax.grid(alpha=0.3, axis="y")
    new = [key for key in ("C10", "C5b")]
    ax.text(0.5, 0.08, "added by this mission: " + ", ".join(new), transform=ax.transAxes,
            ha="center", fontsize=9,
            bbox=dict(fc="#fff7cc", ec="#cc9900"))

    fig.suptitle(
        "CASE-BEM-CUMMINS-WIGLEY: external anchor and wider decay gate folded into the verdict",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())