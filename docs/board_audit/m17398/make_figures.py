#!/usr/bin/env python3
"""Audit figures for m17398 (board audit of m17336 / CASE-BEM-MARUO-DRIFT).

All numbers are read from the delivered study JSON and from the auditor's own
independent reference re-derivation -- nothing is typed by hand.

Usage: python3 make_figures.py [outdir]
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
STUDY = HERE / "refs/docs/case_bem_maruo_drift/study.json"
INDEP = HERE / "artifacts/independent_ref_v2.json"


def main() -> int:
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    study = json.loads(STUDY.read_text())
    cells = study["cells"]
    kas = sorted({c["ka"] for c in cells})
    prod = study["parameters"]["production_resolution"]

    # ---- figure 1: error vs mesh, with the two tolerance lines -------------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for ka in kas:
        row = sorted((c for c in cells if c["ka"] == ka), key=lambda c: c["panel_count"])
        ax.plot([c["panel_count"] for c in row], [100 * c["rel_error"] for c in row],
                marker="o", label=f"ka = {ka:g}")
        p = [c for c in row if c["resolution"] == prod][0]
        ax.annotate(f"{100 * p['rel_error']:.2f}%", (p["panel_count"], 100 * p["rel_error"]),
                    textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.axhline(5.0, color="tab:red", ls="--", lw=1.0, label="case criterion 5%")
    ax.axhline(3.0, color="tab:orange", ls=":", lw=1.0, label="ecosystem target 3%")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("panels (triangle count)")
    ax.set_ylabel("|F_BEM - F_closed| / F_closed  [%]")
    ax.set_title("CASE-BEM-MARUO-DRIFT: BEM far-field drift vs closed form\n"
                 "(audit m17398, delivered study reproduced cell for cell)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    f1 = out / "fig_m17398_error_vs_panels.png"
    fig.savefig(f1, dpi=140)
    plt.close(fig)

    # ---- figure 2: production-mesh got vs ref, plus reference validation --
    indep = json.loads(INDEP.read_text())
    prod_cells = [c for c in cells if c["resolution"] == prod]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    xs = range(len(prod_cells))
    a1.bar([x - 0.2 for x in xs], [c["bem_force"] for c in prod_cells], width=0.4,
           label="BEM (got)")
    a1.bar([x + 0.2 for x in xs], [c["ref_force"] for c in prod_cells], width=0.4,
           label="closed form (ref)")
    a1.set_xticks(list(xs))
    a1.set_xticklabels([f"ka={c['ka']:g}\n{c['panel_count']} panels" for c in prod_cells], fontsize=8)
    a1.set_ylabel("mean drift force F_x [N]")
    a1.set_title(f"production mesh '{prod}': got vs ref")
    a1.legend(fontsize=8)
    a1.grid(True, axis="y", alpha=0.3)

    rel = [100 * r["rel_diff_vs_delivered"] for r in indep["rows"]]
    kas_i = [r["ka"] for r in indep["rows"]]
    a2.bar([str(k) for k in kas_i], [max(v, 1e-12) for v in rel], color="tab:green")
    a2.set_yscale("log")
    a2.set_ylabel("|F_indep - F_delivered| / F  [%]")
    a2.set_xlabel("ka")
    a2.set_title("auditor's independent MacCamy-Fuchs re-derivation\n"
                 f"(max {indep['max_rel_diff_vs_delivered']:.2e}; "
                 "modal vs quadrature "
                 f"{indep['max_modal_vs_quadrature_rel']:.1e})")
    a2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    f2 = out / "fig_m17398_ref_validation.png"
    fig.savefig(f2, dpi=140)
    plt.close(fig)
    print(f1)
    print(f2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())