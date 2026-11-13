#!/usr/bin/env python3
"""Board-audit m17339 figure: registered-benchmark numbers vs their references.

Panel (a): BEM panel ladder of CASE-BEM-CYL-DRIFT-ASYMP (208/456/800/1776 panels,
ka = 0.5) -- far-field (Kochin -> Maruo) and near-field (pressure integral)
relative error against the MacCamy-Fuchs/Maruo closed form, with the case's 5 %
gate and the ecosystem 3 % line.  The 208-panel point is *this audit's* replay.
Panel (b): the (ka)^3 law -- F/(rho g A^2 a (ka)^3) from the m17296 artifact vs
the closed-form plateau 5*pi^2/16 = 3.084251 with +-3 % / +-5 % bands.

All numbers are read from the snapshots in ./snapshots (no hand transcription).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> Path:
    res = json.loads((HERE / "snapshots" / "m17204_results.json").read_text())
    study = json.loads((HERE / "snapshots" / "study.json").read_text())
    replay = json.loads((HERE / "replay" / "bem_cell_208.json").read_text())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    panels = [r["panels"] for r in res["convergence"]["rows"]]
    far = [100 * r["far_error"] for r in res["convergence"]["rows"]]
    near = [100 * r["near_error"] for r in res["convergence"]["rows"]]
    ax1.loglog(panels, far, "o-", label="far field (BEM Kochin -> Maruo)")
    ax1.loglog(panels, near, "s-", label="near field (BEM pressure integral)")
    ax1.loglog([replay["panels"]], [100 * replay["replay_far_error"]], "o",
               mfc="none", ms=11, mew=1.6, color="C0", label="audit m17339 replay (far)")
    ax1.axhline(5.0, color="r", ls="--", lw=1.2, label="case gate 5 %")
    ax1.axhline(3.0, color="0.4", ls=":", lw=1.2, label="ecosystem 3 %")
    ax1.set_xlabel("panels")
    ax1.set_ylabel("relative error vs closed form  [%]")
    ax1.set_title("(a) BEM ladder, ka = 0.5")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=7, loc="lower left")

    ka = study["long_wave"]["ka_values"]
    coef = study["long_wave"]["normalised_coefficient"]
    ref = study["long_wave"]["closed_form_coefficient"]
    ax2.semilogx(ka, coef, "o-", label=r"artifact $F/(\rho g A^2 a (ka)^3)$")
    ax2.axhline(ref, color="k", ls="--", lw=1.2, label=r"$5\pi^2/16 = %.6f$" % ref)
    ax2.fill_between([ka[0], ka[-1]], ref * 0.97, ref * 1.03, color="C2", alpha=0.15,
                     label=r"$\pm 3\%$")
    ax2.fill_between([ka[0], ka[-1]], ref * 0.95, ref * 0.97, color="C1", alpha=0.15)
    ax2.fill_between([ka[0], ka[-1]], ref * 1.03, ref * 1.05, color="C1", alpha=0.15,
                     label=r"$\pm 5\%$")
    ax2.set_xlabel("ka")
    ax2.set_ylabel(r"$F_x/(\rho g A^2 a (ka)^3)$")
    ax2.set_title("(b) long-wave law vs 5*pi^2/16")
    ax2.set_ylim(2.7, 3.2)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=7, loc="lower left")

    fig.suptitle("CASE-BEM-CYL-DRIFT-ASYMP: registered benchmark numbers vs references (audit m17339)")
    p = HERE / "fig_m17339_benchmark_vs_ref.png"
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print(p)
    return p


if __name__ == "__main__":
    main()