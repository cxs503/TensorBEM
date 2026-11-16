#!/usr/bin/env python3
"""Figures for round 7 of CASE-BEM-SUBMERGED-SPHERE-HAVELOCK.

Two artefacts are read, nothing is transcribed by hand:

* ``docs/case_bem_submerged_sphere_havelock/measure_correction/numbers_t1t2_rows.json``
  (this round: the per-``ka`` rows behind the committed T1/T2 verdicts);
* ``.../measure_correction/numbers_round6.json`` (case mesh T3), if present, for the
  cross-mesh comparison of the band-maximum error under the three measures.

    python3 scripts/havelock_round7_figures.py \
        --t1t2 docs/case_bem_submerged_sphere_havelock/measure_correction/numbers_t1t2_rows.json \
        --t3   docs/case_bem_submerged_sphere_havelock/measure_correction/numbers_round6.json \
        --outdir docs/case_bem_submerged_sphere_havelock_round7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COEFFICIENTS = ("a33", "b33", "a11", "b11")
LABEL = {"a33": "$A'_{33}$", "b33": "$B'_{33}$", "a11": "$A'_{11}$", "b11": "$B'_{11}$"}
STYLE = {
    "flat": ("-", "o", "#b2182b"),
    "patch": ("-", "s", "#2166ac"),
    "uniform": ("--", "^", "#4d9221"),
}
GATE = 0.05
TIERS = (
    ("T1:economy_s2_band", "economy mesh — 320 panels"),
    ("T2:coarsest_s1_band", "coarsest mesh — 80 panels"),
)


def _rows(doc: dict[str, Any], label: str) -> list[dict[str, Any]]:
    return doc["tiers"][label]["rows"]


def fig_t1t2_band(doc: dict[str, Any], out: Path) -> Path:
    """Relative error vs ka for both coarse meshes, three measures, four coefficients."""
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharex=True)
    for row_ax, (label, title) in zip(axes, TIERS):
        rows = _rows(doc, label)
        for ax, c in zip(row_ax, ("a33", "b33", "a11", "b11")):
            for measure, (ls, marker, colour) in STYLE.items():
                sel = sorted((r for r in rows if r["measure"] == measure), key=lambda r: r["ka"])
                ax.plot([r["ka"] for r in sel],
                        [100.0 * r[f"{c}_rel_error"] for r in sel],
                        ls, marker=marker, ms=4.5, lw=1.4, color=colour, label=measure)
            ax.axhline(100.0 * GATE, color="0.35", lw=1.0, ls=":")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{LABEL[c]} — {title}", fontsize=9)
            ax.set_xlabel("$ka$")
            ax.set_ylabel("relative error [%]")
            ax.grid(alpha=0.3, which="both")
    axes[0, 0].legend(frameon=False, fontsize=8, title="surface measure")
    fig.suptitle("T1/T2 row tables (rows behind the committed verdicts): dotted = 5 % gate",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_cross_mesh(doc: dict[str, Any], t3: dict[str, Any] | None, out: Path) -> Path:
    """Band-maximum error per measure, coarse meshes here and the case mesh if available."""
    entries: list[tuple[str, dict[str, float]]] = []
    for label, title in TIERS:
        v = doc["verdicts"][label]
        entries.append((title, {m: 100.0 * v[m]["band_max_rel_error"] for m in STYLE}))
    if t3 is not None:
        v3 = t3["verdicts"]["T3:case_s3_band_flat_patch_uniform"]
        entries.append(("case mesh — 1280 panels",
                        {m: 100.0 * v3[m]["band_max_rel_error"] for m in STYLE}))
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    width = 0.8 / len(entries)
    for i, (title, vals) in enumerate(entries):
        xs = [j + (i - (len(entries) - 1) / 2) * width for j in range(len(STYLE))]
        ax.bar(xs, [vals[m] for m in STYLE], width=width, label=title, alpha=0.9)
        for x, m in zip(xs, STYLE):
            ax.text(x, vals[m], f"{vals[m]:.2f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(100.0 * GATE, color="0.35", lw=1.0, ls=":")
    ax.set_xticks(range(len(STYLE)))
    ax.set_xticklabels(list(STYLE))
    ax.set_yscale("log")
    ax.set_ylabel("band-max relative error [%]")
    ax.set_title("band-maximum error by surface measure and mesh (dotted = 5 % gate)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--t1t2", type=Path, required=True)
    parser.add_argument("--t3", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)
    doc = json.loads(args.t1t2.read_text())
    t3 = json.loads(args.t3.read_text()) if args.t3 and args.t3.exists() else None
    args.outdir.mkdir(parents=True, exist_ok=True)
    for path in (
        fig_t1t2_band(doc, args.outdir / "fig1_t1t2_band.png"),
        fig_cross_mesh(doc, t3, args.outdir / "fig2_cross_mesh_bandmax.png"),
    ):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())