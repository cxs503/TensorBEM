#!/usr/bin/env python3
"""Figures for the ka=4,5 case-mesh computation (report material).

Two panels, both built from the committed artefact (numbers.json) and the
compute artefact -- nothing is hand-copied:

1. relative error of the four radiation coefficients versus ``ka``; the two
   rebuilt wavenumbers are marked, and the superseded GCI estimates are drawn
   as hollow markers so the size of the correction is visible;
2. the damping magnitude ``B'`` on a log axis, with the ``|ref| < 1e-4`` region
   shaded: there the 5 % relative gate is noise-dominated (degenerate items).

Usage
-----
    PYTHONPATH=src python3 scripts/havelock_ka45_figures.py --outdir <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DOCS = Path("docs/case_bem_submerged_sphere_havelock")
COEFFICIENTS = ("a33", "b33", "a11", "b11")
LABEL = {
    "a33": r"$A'_{33}$",
    "b33": r"$B'_{33}$",
    "a11": r"$A'_{11}$",
    "b11": r"$B'_{11}$",
}
REPLACED_KA = (4.0, 5.0)
DEGENERATE_SCALE = 1.0e-4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--numbers", type=Path, default=DOCS / "numbers.json")
    parser.add_argument("--outdir", type=Path, default=DOCS)
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    numbers = json.loads(args.numbers.read_text())
    rows = sorted(numbers["band_sweep_case"], key=lambda r: r["ka"])
    ka = np.array([r["ka"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # --- panel 1: relative error vs ka, computed rows vs superseded estimates
    ax = axes[0]
    colours = {"a33": "C0", "a11": "C1", "b33": "C2", "b11": "C3"}
    for key in COEFFICIENTS:
        err = np.array([r[f"{key}_rel_error"] for r in rows]) * 100.0
        ax.plot(ka, err, "o-", color=colours[key], ms=4.5, lw=1.4, label=LABEL[key])
        for i, r in enumerate(rows):
            if r["ka"] not in REPLACED_KA:
                continue
            old = r.get("superseded_gci_estimate", {}).get(key)
            if old is None:
                continue
            ax.plot(r["ka"], old["rel_error"] * 100.0, "x", color=colours[key],
                    ms=9, mew=1.9, alpha=0.85)
    ax.axhline(5.0, color="k", ls="--", lw=1.0, label="case gate 5%")
    ax.axvspan(3.5, 5.2, color="grey", alpha=0.10)
    ax.set_xlabel(r"$ka$ [-]")
    ax.set_ylabel("relative error vs multipole reference [%]")
    ax.set_title("case mesh (1280 panels): full band, all rows computed\n"
                 r"$\times$ = superseded GCI estimate at $ka=4,5$")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)

    # --- panel 2: damping magnitude with the degenerate band shaded
    ax = axes[1]
    for key in ("b33", "b11"):
        ref = np.array([abs(r[f"{key}_exact"]) for r in rows])
        got = np.array([abs(r[f"{key}_bem"]) for r in rows])
        ax.plot(ka, ref, "o-", color=colours[key], ms=4.5, lw=1.4,
                label=LABEL[key] + " reference")
        ax.plot(ka, got, "s--", color=colours[key], ms=3.6, lw=1.0, alpha=0.75,
                label=LABEL[key] + " BEM (1280 panels)")
    ax.axhspan(1e-8, DEGENERATE_SCALE, color="tab:red", alpha=0.10)
    ax.axhline(DEGENERATE_SCALE, color="tab:red", ls=":", lw=1.0)
    ax.text(0.13, DEGENERATE_SCALE * 1.6, r"$|ref| < 10^{-4}$: 5% relative gate "
            "noise-dominated", fontsize=8, color="tab:red")
    ax.set_xlabel(r"$ka$ [-]")
    ax.set_ylabel(r"damping coefficient $B'$ [-]")
    ax.set_yscale("log")
    ax.set_title("damping magnitude: the high-$ka$ tail is a degenerate relative criterion")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out1 = args.outdir / "fig11_ka45_computed_error.png"
    fig.savefig(out1, dpi=150)
    print(f"wrote {out1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())