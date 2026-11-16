"""Round-8 figures: is the flat-panel error the area deficit, at the tiers we measured?

All numbers come from committed artefacts (no value is typed in):
``docs/case_bem_submerged_sphere_havelock/measure_correction/numbers_t1t2_rows.json`` (rows)
and ``docs/case_bem_submerged_sphere_havelock_round8/adjudicate.json`` (R-P1/R-P2/R-P3).

Usage: python3 scripts/havelock_round8_figures.py [--outdir ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ADJ = ROOT / "docs" / "case_bem_submerged_sphere_havelock_round8" / "adjudicate.json"
LABELS = {"a33": r"$A'_{33}$", "b33": r"$B'_{33}$", "a11": r"$A'_{11}$", "b11": r"$B'_{11}$"}


def fig_deficit_share(adj: dict, outdir: Path) -> Path:
    tiers = [k for k in ("T1", "T2", "T3") if k in adj["tiers"]]
    fig, axes = plt.subplots(1, len(tiers), figsize=(5.0 * len(tiers), 4.2), squeeze=False)
    for ax, label in zip(axes[0], tiers):
        block = adj["tiers"][label]
        cells = block["R_P2"]["rows"]
        kas = sorted({c["ka"] for c in cells})
        width = 0.38
        for i, quantity in enumerate(("b33", "b11")):
            sel = {c["ka"]: c for c in cells if c["quantity"] == quantity}
            ax.bar([j + (i - 0.5) * width for j in range(len(kas))],
                   [sel[k]["reduction_frac_of_deficit"] for k in kas], width=width,
                   label=LABELS[quantity], color=("#1f6f8b", "#e0a800")[i])
        ax.axhline(0.5, color="k", ls="--", lw=1.2, label="R-P2 bar (0.5 x deficit)")
        ax.axhline(1.0, color="#c0392b", ls=":", lw=1.2, label="exactly the deficit")
        ax.set_xticks(range(len(kas)))
        ax.set_xticklabels([f"{k:g}" for k in kas], fontsize=8)
        ax.set_xlabel(r"$ka$")
        ax.set_ylabel(r"$(e_\mathrm{flat}-e_\mathrm{patch})/\mathrm{deficit}$")
        ax.set_title(f"{label} ({block['n_panels']} panels, deficit {100*block['area_deficit']:.3f}%)",
                     fontsize=10)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Share of the spherical-patch area deficit removed by the exact measure (damping)",
                 fontsize=11)
    fig.tight_layout()
    path = outdir / "fig1_deficit_share.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_flat_vs_patch(adj: dict, outdir: Path) -> Path:
    tiers = [k for k in ("T1", "T2", "T3") if k in adj["tiers"]]
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    markers = {"T1": "o", "T2": "s", "T3": "^"}
    colours = {"T1": "#1f6f8b", "T2": "#e0a800", "T3": "#c0392b"}
    xs, ys = [], []
    for label in tiers:
        per = adj["tiers"][label]["per_coefficient"]
        for c, entry in per.items():
            x = 100 * entry["flat_band_max_rel_error"]
            y = 100 * entry["patch_band_max_rel_error"]
            xs.append(x)
            ys.append(y)
            ax.plot(x, y, markers[label], ms=7, color=colours[label])
    lo = min(xs + ys) / 3.0
    hi = max(xs + ys) * 3.0
    ax.plot([lo, hi], [lo, hi], "k-", lw=1.0, label="patch = flat (no gain)")
    ax.plot([lo, hi], [lo / 2.0, hi / 2.0], "k--", lw=1.0, label="patch halves the error")
    ax.fill_between([lo, hi], [lo / 2, hi / 2], [lo, hi], color="grey", alpha=0.12)
    for label in tiers:
        ax.plot([], [], markers[label], color=colours[label], label=f"{label} ({adj['tiers'][label]['n_panels']} panels)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("flat-measure band-max relative error per coefficient [%]")
    ax.set_ylabel("exact-patch band-max relative error per coefficient [%]")
    ax.set_title("R-P1/R-P3: exact patch measure vs flat, per coefficient", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    path = outdir / "fig2_flat_vs_patch.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=ROOT / "docs" / "case_bem_submerged_sphere_havelock_round8")
    args = ap.parse_args()
    adj = json.loads(ADJ.read_text())
    args.outdir.mkdir(parents=True, exist_ok=True)
    for path in (fig_deficit_share(adj, args.outdir), fig_flat_vs_patch(adj, args.outdir)):
        print(f"  {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())