#!/usr/bin/env python3
"""m17547 — figures for the rigid-lid reference pin + independent-route study.

All three panels are drawn from ``results.json`` only (no hand-copied number).

Panel A  MFS ring-source ladder (two source contractions) against the pinned
         reference and a +/-0.1 % band.
Panel B  Deviation of the exact-rational partial sums from the pin vs truncation
         degree (log y): the O(n**-4) tail, plus the Gauss-quadrature value of
         m16802/m17243 sitting 3.4 ppm above the pin (kink bias, not tail).
Panel C  Deviations of every value in circulation from the pin (ppm, log scale).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
PAYLOAD = json.loads((HERE / "results.json").read_text(encoding="utf-8"))

PIN = PAYLOAD["pin"]["value"]
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

# --- Panel A: independent route ladder -------------------------------------
ax = axes[0]
rows = PAYLOAD["mfs_ladder"]
for c, style in ((0.6, "o-"), (0.8, "s-")):
    sel = [r for r in rows if abs(r["contraction"] - c) < 1e-12]
    ax.plot(
        [r["n_sources"] for r in sel],
        [r["added_mass_dimless"] for r in sel],
        style,
        label=f"MFS ring sources, contraction {c:.1f}",
    )
ax.axhline(PIN, color="k", lw=1.6, label=f"pinned reference {PIN:.9f}")
ax.axhspan(PIN * (1 - 1e-3), PIN * (1 + 1e-3), color="0.85", zorder=0,
           label=r"$\pm 0.1\,\%$ band")
anchor = [r for r in PAYLOAD["anchor_rows"] if r["panels"] == 256]
if anchor:
    ax.axhline(anchor[0]["added_mass_dimless"], color="C3", ls="--", lw=1.2,
               label=f"3-D Kelvin BEM, 256 panels ({anchor[0]['added_mass_dimless']:.6f})")
ax.set_xscale("log")
ax.set_xlabel("number of ring sources")
ax.set_ylabel(r"$A_{33}/(\rho \nabla)$")
ax.set_title("A. Independent route (MFS) vs pinned reference")
ax.legend(fontsize=7.5, loc="lower right")
ax.grid(alpha=0.3)

# --- Panel B: series tail vs quadrature bias -------------------------------
ax = axes[1]
degrees = PAYLOAD["pin"]["degrees"]
sums = PAYLOAD["pin"]["partial_sums"]
dev = [abs(s - PIN) for s in sums]
ax.loglog(degrees, dev, "o-", label="exact-rational partial sums (tail)")
deg_fit = np.array(degrees, dtype=float)
ax.loglog(
    deg_fit,
    dev[-1] * (deg_fit / degrees[-1]) ** -4.0,
    "k:",
    label=r"$O(n^{-4})$ guide",
)
gauss = next(
    r for r in PAYLOAD["reference_audit"]["rows"]
    if r["key"].startswith("gauss_quadrature")
)
ax.axhline(abs(gauss["deviation"]), color="C3", ls="--",
           label=f"Gauss-quadrature series: {gauss['deviation_ppm']:+.2f} ppm")
ax.set_xlabel("truncation degree (max even Legendre degree)")
ax.set_ylabel(r"$|S(n) - S_\infty|$")
ax.set_title("B. Series truncation vs quadrature bias")
ax.legend(fontsize=7.5)
ax.grid(alpha=0.3, which="both")

# --- Panel C: values in circulation ---------------------------------------
ax = axes[2]
audit = sorted(PAYLOAD["reference_audit"]["rows"], key=lambda r: abs(r["deviation_ppm"]))
labels = [r["key"].replace("_", "\n", 1) for r in audit]
vals = [max(abs(r["deviation_ppm"]), 1e-3) for r in audit]
colors = ["C2" if abs(r["deviation_ppm"]) < 10 else "C3" for r in audit]
bars = ax.barh(labels, vals, color=colors)
ax.set_xscale("log")
for bar, row in zip(bars, audit):
    ax.text(bar.get_width() * 1.3, bar.get_y() + bar.get_height() / 2,
            f"{row['deviation_ppm']:+.3f} ppm", va="center", fontsize=8)
ax.axvline(1.0, color="k", ls=":", label="1 ppm")
ax.set_xlabel("|deviation from pin| [ppm]")
ax.set_title("C. Values in circulation vs the pin")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="x", which="both")

fig.suptitle(
    "CASE-BEM-RIGID-LID-HEMISPHERE (m17547): reference pinned to "
    f"{PIN:.9f}, independent MFS route agrees to "
    f"{100 * abs(PAYLOAD['mfs_best']['rel_error_vs_pin']):.4f} %",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = HERE / "figure.png"
fig.savefig(out, dpi=150)
print("wrote", out, out.stat().st_size, "bytes")

# integrity: the figure must be non-trivial
assert out.stat().st_size > 20_000, out.stat().st_size
assert all(math.isfinite(v) for v in dev)
