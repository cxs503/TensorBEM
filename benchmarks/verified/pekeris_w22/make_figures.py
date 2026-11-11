#!/usr/bin/env python
"""Figures for the W22 Pekeris / ice-cover report.

All curves are read from the two JSON artifacts (``tl_compare.json`` and
``ice_cover_sensitivity.json``) -- no number is recomputed or typed here, so a figure can
always be traced back to the artifact entry that produced it.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

tl = json.loads((HERE / "tl_compare.json").read_text())
ice = json.loads((HERE / "ice_cover_sensitivity.json").read_text())

ranges = tl["inputs"]["ranges_m"]
freqs = tl["inputs"]["frequencies_hz"]

# ----------------------------------------------------------------------------------
# figure 1: TL vs range, pressure release vs ice levels, at every frequency
# ----------------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(freqs), figsize=(4.2 * len(freqs), 3.8), sharey=False)
for ax, f in zip(axes, freqs):
    pr = [p["tl_db"] for p in tl["pressure_release_points"] if p["freq_hz"] == f]
    ax.plot(ranges, pr, "o-", lw=2, label="pressure release (Pekeris)", color="k")
    for lvl, style in zip(ice["levels"], ("-", "--", ":")):
        rows = [t for t in lvl["tl"] if t["freq_hz"] == f and t.get("route") != "finite_difference_crosscheck"]
        ax.plot([t["range_m"] for t in rows], [t["tl_ice_db"] for t in rows], style, marker="s",
                label=lvl["label"].replace("_", " "))
    ax.set_xscale("log")
    ax.set_xlabel("range r [m]")
    ax.set_ylabel("transmission loss TL [dB]")
    ax.set_title(f"f = {f:g} Hz, source/receiver z = {tl['inputs']['source_depth_m']:g} m")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
fig.suptitle("Pekeris two-layer waveguide: TL with pressure-release top vs thin elastic ice cover")
fig.tight_layout()
fig.savefig(FIG / "tl_vs_range.png", dpi=140)
plt.close(fig)

# ----------------------------------------------------------------------------------
# figure 2: analytic vs finite-difference, point by point (criterion 2)
# ----------------------------------------------------------------------------------
pts = tl["criteria"]["C2"]["points"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.8))
labels = [f"{p['freq_hz']:g}Hz/{p['range_m']/1000:g}km" for p in pts]
x = range(len(pts))
ax1.plot(list(x), [p["tl_analytic_db"] for p in pts], "o", label="analytic Pekeris")
ax1.plot(list(x), [p["tl_finite_difference_db"] for p in pts], "x", label="finite difference")
ax1.set_xticks(list(x))
ax1.set_xticklabels(labels, rotation=60, fontsize=7)
ax1.set_ylabel("TL [dB]")
ax1.set_title("criterion 2: 9-point TL comparison")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)
ax2.bar(list(x), [p["deviation_db"] for p in pts], color="tab:blue")
ax2.axhline(tl["criteria"]["C2"]["threshold_db"], color="r", ls="--", label="criterion 2 = 1 dB")
ax2.axhline(-tl["criteria"]["C2"]["threshold_db"], color="r", ls="--")
ax2.set_xticks(list(x))
ax2.set_xticklabels(labels, rotation=60, fontsize=7)
ax2.set_ylabel("FD - analytic [dB]")
ax2.set_ylim(-1.2, 1.2)
ax2.set_title(f"max |deviation| = {tl['criteria']['C2']['measured_max_abs_deviation_db']*1000:.1f} m dB")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "criterion2_compare.png", dpi=140)
plt.close(fig)

# ----------------------------------------------------------------------------------
# figure 3: modal dispersion shift and TL deviation vs ice cover
# ----------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.9))
for lvl in ice["levels"]:
    disp = lvl["dispersion"]
    ax1.plot([d["freq_hz"] for d in disp], [d["max_delta_phase_velocity_m_s"] for d in disp],
             "o-", label=lvl["label"].replace("_", " "))
    ax1.plot([d["freq_hz"] for d in disp], [d["max_delta_k_r_ppm"] / 1000.0 for d in disp],
             "s--", alpha=0.6)
ax1.set_xlabel("frequency [Hz]")
ax1.set_ylabel(r"max $|\Delta c_{ph}|$ [m/s]  (solid)" + "\n" + r"max $|\Delta k_r|/k_r$ [1e-3] (dashed)")
ax1.set_title("criterion 3: modal dispersion shift vs ice level")
ax1.grid(alpha=0.3)
ax1.legend(fontsize=8)

width = 0.2
for i, f in enumerate(freqs):
    vals = []
    for lvl in ice["levels"]:
        rows = [t for t in lvl["tl"] if t["freq_hz"] == f and t.get("route") != "finite_difference_crosscheck"]
        vals.append(max(abs(t["deviation_db"]) for t in rows))
    ax2.bar([j + i * width for j in range(len(ice["levels"]))], vals, width=width,
            label=f"{f:g} Hz")
ax2.axhline(1.0, color="r", ls="--", lw=1, label="1 dB reference")
ax2.set_xticks([j + width for j in range(len(ice["levels"]))])
ax2.set_xticklabels([lvl["label"].replace("_", "\n") for lvl in ice["levels"]], fontsize=8)
ax2.set_ylabel(r"max $|TL_{ice} - TL_{PR}|$ [dB]")
ax2.set_title("criterion 3: TL deviation induced by the ice cover")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "ice_cover_sensitivity.png", dpi=140)
plt.close(fig)

# ----------------------------------------------------------------------------------
# figure 4: convergence evidence (criterion 1 / criterion 4)
# ----------------------------------------------------------------------------------
c4 = tl["criteria"]["C4"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
ns = [g["n_nodes"] for g in c4["grid_doubling"]["rows"]]
for i, r in enumerate(ranges):
    ax1.semilogy(ns, [g["abs_change_vs_finest_db"][i] + 1e-8 for g in c4["grid_doubling"]["rows"]],
                 "o-", label=f"r = {r:g} m")
ax1.set_xlabel("finite-difference nodes N")
ax1.set_ylabel("|TL(N) - TL(1600)| [dB]")
ax1.set_title("criterion 4a: grid doubling (2nd order)")
ax1.grid(alpha=0.3, which="both")
ax1.legend(fontsize=8)
tr = c4["domain_doubling"]["rows"]
ax2.semilogx([t["bottom_layer_thickness_m"] for t in tr],
             [max(t["abs_change_vs_half_space_db"]) for t in tr], "o-")
ax2.axhline(1.0, color="r", ls="--", label="1 dB")
ax2.set_xlabel("modelled bottom-domain thickness [m]")
ax2.set_ylabel("|TL(truncated) - TL(half-space)| [dB]")
ax2.set_title("criterion 4b: domain-length truncation")
ax2.grid(alpha=0.3, which="both")
ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "convergence.png", dpi=140)
plt.close(fig)

print("figures written to", FIG)
for p in sorted(FIG.glob("*.png")):
    print(" ", p.name, p.stat().st_size, "bytes")