"""m13811: verify the finite-depth 3-D kernel off-axis after the coordinate-clamp fix.

Emits (programmatically, no hand transcription):
  outputs/m13811/fix_verification.json   -- every number used in the report
  outputs/m13811/table.md                -- the markdown table in docs/
  docs/figures/fig_m13811_fd3d_offaxis_fix.png

"before" = pre-fix behaviour reproduced exactly: the buggy clamp set x=y=0 for
target and source, so every pair was evaluated at R = |x_h - xi_h| = 0.  The
shipped kernel is therefore asked for the projected target; that is bit-for-bit
what the old code computed (checked against the recorded 0.1495+0.1219j at
h=8, target (0.5,0,-0.5)).
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorbem.finite_depth3d import green_function_3d as eigenseries
from tensorbem.finite_depth3d_quadrature import (
    green_function_finite_depth_3d as quadrature,
)
from tensorbem.free_surface3d import green_function_3d as deep_water

NU = 0.6
SOURCE = np.array([0.0, 0.0, -1.0])
TARGETS = [(0.5, 0.0, -0.5), (2.0, 1.0, -1.5), (1.5, -0.7, -0.2), (5.0, 3.0, -0.9)]
DEPTHS = (4.0, 8.0, 20.0, 60.0, 128.0)


def rel(value, ref):
    return float(np.abs(value - ref) / np.abs(ref))


def main():
    rows = []
    for tgt in TARGETS:
        t = np.array([tgt])
        src = np.array([SOURCE])
        projected = np.array([[0.0, 0.0, tgt[2]]])
        ref_dw = deep_water(t, SOURCE, NU)[0]
        for h in DEPTHS:
            after = quadrature(t, src, NU, h)[0]
            before = quadrature(projected, src, NU, h)[0]
            eig = eigenseries(t, src, NU, h)[0]
            rows.append(
                dict(
                    target=list(tgt),
                    radius=float(np.hypot(tgt[0], tgt[1])),
                    depth=h,
                    before_complex=[before.real, before.imag],
                    after_complex=[after.real, after.imag],
                    eigenseries_complex=[eig.real, eig.imag],
                    deepwater_complex=[ref_dw.real, ref_dw.imag],
                    err_before_vs_deepwater=rel(before, ref_dw),
                    err_after_vs_deepwater=rel(after, ref_dw),
                    err_after_vs_eigenseries=rel(after, eig),
                )
            )

    os.makedirs("outputs/m13811", exist_ok=True)
    with open("outputs/m13811/fix_verification.json", "w") as fh:
        json.dump(dict(nu=NU, source=list(SOURCE), depths=list(DEPTHS),
                       targets=[list(t) for t in TARGETS], rows=rows), fh, indent=1)

    # ---- markdown table (injected into the doc verbatim) -------------------
    lines = ["| target (R) | h | before: err vs deep water | after: err vs deep water "
             "| after: err vs eigenseries |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append("| ({:.1f},{:.1f},{:.1f}) R={:.2f} | {:g} | {:.3e} | {:.3e} | {:.3e} |".format(
            *r["target"], r["radius"], r["depth"],
            r["err_before_vs_deepwater"], r["err_after_vs_deepwater"],
            r["err_after_vs_eigenseries"]))
    with open("outputs/m13811/table.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    for tgt in TARGETS:
        sel = [r for r in rows if r["target"] == list(tgt)]
        ax.loglog([r["depth"] for r in sel], [r["err_before_vs_deepwater"] for r in sel],
                  "--o", label=f"before fix, R={sel[0]['radius']:.2f}")
        ax.loglog([r["depth"] for r in sel], [r["err_after_vs_deepwater"] for r in sel],
                  "-s", label=f"after fix, R={sel[0]['radius']:.2f}")
    ax.axhline(1e-2, color="k", lw=0.6, ls=":")
    ax.text(4.2, 1.4e-2, "1 % band", fontsize=7)
    ax.axhline(1e-8, color="firebrick", lw=0.6, ls=":")
    ax.set_xlabel("water depth h [m]  (nu*h = 0.6 h)")
    ax.set_ylabel("relative error vs deep-water Kelvin kernel")
    ax.set_title("finite-depth 3-D kernel, off-axis: before / after the clamp fix")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6, ncol=2)

    ax = axes[1]
    for tgt in TARGETS:
        sel = [r for r in rows if r["target"] == list(tgt)]
        ax.semilogy([r["depth"] for r in sel], [r["err_after_vs_eigenseries"] for r in sel],
                    "-o", label=f"R={sel[0]['radius']:.2f}")
    ax.axhline(1e-8, color="firebrick", lw=0.6, ls=":")
    ax.text(4.2, 1.2e-8, "1e-8 gate", fontsize=7)
    ax.set_xlabel("water depth h [m]")
    ax.set_ylabel("|G_quadrature - G_eigenseries| / |G_eigenseries|")
    ax.set_title("after fix: two independent implementations, roundoff agreement")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs("docs/figures", exist_ok=True)
    fig.savefig("docs/figures/fig_m13811_fd3d_offaxis_fix.png", dpi=160)

    worst_before = max(r["err_before_vs_deepwater"] for r in rows)
    worst_after_dw = max(r["err_after_vs_deepwater"] for r in rows
                         if r["depth"] >= 20.0)
    worst_eig = max(r["err_after_vs_eigenseries"] for r in rows)
    print(json.dumps(dict(worst_before=worst_before, worst_after_vs_deepwater_h20plus=worst_after_dw,
                          worst_after_vs_eigenseries=worst_eig), indent=1))
    print(open("outputs/m13811/table.md").read())


if __name__ == "__main__":
    main()
