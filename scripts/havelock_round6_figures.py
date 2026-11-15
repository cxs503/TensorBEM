#!/usr/bin/env python3
"""Figures for round 6 of CASE-BEM-SUBMERGED-SPHERE-HAVELOCK (surface measure on the case mesh).

All numbers are read from the merged artefact -- nothing is transcribed by hand.

    python3 scripts/havelock_round6_figures.py \
        --numbers docs/case_bem_submerged_sphere_havelock/measure_correction/numbers_round6.json \
        --outdir  docs/case_bem_submerged_sphere_havelock/measure_correction
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COEFFICIENTS = ("a33", "b33", "a11", "b11")
LABEL = {"a33": "$A'_{33}$", "b33": "$B'_{33}$", "a11": "$A'_{11}$", "b11": "$B'_{11}$"}
STYLE = {"flat": ("-", "o", "#b2182b"), "patch": ("-", "s", "#2166ac"), "uniform": ("--", "^", "#4d9221")}
GATE = 0.05


def _rows(study: dict[str, Any]) -> list[dict[str, Any]]:
    return study["tiers"]["T3:case_s3_band_flat_patch_uniform"]["rows"]


def fig_band(study: dict[str, Any], out: Path) -> Path:
    rows = _rows(study)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), sharex=True)
    for ax, c in zip(axes.ravel(), COEFFICIENTS):
        for measure, (ls, marker, colour) in STYLE.items():
            sel = sorted((r for r in rows if r["measure"] == measure), key=lambda r: r["ka"])
            ax.plot([r["ka"] for r in sel], [100.0 * r[f"{c}_rel_error"] for r in sel],
                    ls, marker=marker, ms=4.5, lw=1.4, color=colour, label=measure)
        ax.axhline(100.0 * GATE, color="0.35", lw=1.0, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(LABEL[c])
        ax.set_xlabel("$ka$")
        ax.set_ylabel("relative error  [%]")
        ax.grid(alpha=0.3, which="both")
    axes[0, 0].legend(frameon=False, fontsize=8, title="surface measure")
    fig.suptitle("case mesh (1280 panels, $f/a=2$): measure sensitivity of the radiation coefficients\n"
                 "dotted = 5 % pool gate", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_decomposition(study: dict[str, Any], out: Path) -> Path:
    dec = study["decomposition"]
    cells = dec["cells"]
    deficit = dec["area_deficit_s3"] * 100.0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    kas = sorted({c["ka"] for c in cells})
    width = 0.8 / len(COEFFICIENTS)
    for i, c in enumerate(COEFFICIENTS):
        sel = [c0 for c0 in cells if c0["quantity"] == c]
        xs = [kas.index(r["ka"]) + (i - 1.5) * width for r in sel]
        ax1.bar(xs, [100.0 * r["reduction_frac_of_deficit"] for r in sel], width=width,
                label=LABEL[c], alpha=0.85)
    ax1.axhline(100.0, color="0.35", lw=1.0, ls=":")
    ax1.set_xticks(range(len(kas)))
    ax1.set_xticklabels([f"{k:g}" for k in kas])
    ax1.set_xlabel("$ka$")
    ax1.set_ylabel("error removed by the exact measure\nas fraction of the area deficit  [%]")
    ax1.set_title(f"attribution test (deficit = {deficit:.4f} %)")
    ax1.legend(frameon=False, fontsize=8, ncol=2)
    ax1.grid(alpha=0.3, axis="y")

    for c in COEFFICIENTS:
        sel = sorted((r for r in cells if r["quantity"] == c), key=lambda r: r["ka"])
        ax2.plot([r["ka"] for r in sel], [r["flat_over_patch"] for r in sel],
                 "-o", ms=4, lw=1.2, label=LABEL[c])
    ax2.axhline(1.0, color="0.35", lw=1.0, ls=":")
    ax2.set_xscale("log")
    ax2.set_xlabel("$ka$")
    ax2.set_ylabel("$\\epsilon_{flat} / \\epsilon_{patch}$")
    ax2.set_title("error ratio removed by the measure")
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_controls(study: dict[str, Any], out: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    labels, vals = [], []
    for key, payload in study["tiers"].items():
        if not key.startswith("T4"):
            continue
        worst = {}
        for row in payload["rows"]:
            m = row["measure"]
            worst[m] = max(worst.get(m, 0.0), row["band_max_rel_error"])
        labels.append(key.split(":")[1])
        vals.append([100.0 * worst[m] for m in ("flat", "patch", "uniform")])
    vals_arr = np.array(vals)
    n = len(labels)
    x = np.arange(n)
    for i, (m, (ls, marker, colour)) in enumerate(STYLE.items()):
        ax1.bar(x + (i - 1) * 0.26, vals_arr[:, i], 0.26, color=colour, label=m)
    ax1.axhline(100.0 * GATE, color="0.35", lw=1.0, ls=":")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_yscale("log")
    ax1.set_ylabel("band-max relative error [%]")
    ax1.set_title("T4 off-design submergence ($ka=1$)")
    ax1.legend(frameon=False, fontsize=8)

    low = study.get("low_frequency_limit") or []
    if low:
        kas = [r["ka"] for r in low]
        ax2.loglog(kas, [r["b33_ref"] for r in low], "-", color="0.3", lw=1.2, label="reference")
        ax2.loglog(kas, [r["b33_flat"] for r in low], "o", color=STYLE["flat"][2], ms=4, label="flat")
        ax2.loglog(kas, [r["b33_patch"] for r in low], "s", color=STYLE["patch"][2], ms=4, label="patch")
        ax2.set_xlabel("$ka$")
        ax2.set_ylabel("$B'_{33}$")
        ax2.set_title("T5 low-frequency limit of the damping")
        ax2.grid(alpha=0.3, which="both")
        ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args(argv)
    study = json.loads(args.numbers.read_text())
    args.outdir.mkdir(parents=True, exist_ok=True)
    for fn, name in ((fig_band, "fig3_round6_band_measures.png"),
                     (fig_decomposition, "fig4_round6_decomposition.png"),
                     (fig_controls, "fig5_round6_controls.png")):
        print(fn(study, args.outdir / name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())