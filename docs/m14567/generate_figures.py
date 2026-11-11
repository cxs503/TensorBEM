#!/usr/bin/env python3
"""Generate figures for seakeeping verification report."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load results
with open("runs/seakeeping_verification_results.json") as f:
    results = json.load(f)

output_dir = Path("runs")
output_dir.mkdir(exist_ok=True)

# Figure 1: Wigley III RAO frequency response
print("Generating Figure 1: Wigley III heave RAO...")
fig, ax = plt.subplots(figsize=(8, 5))
rao = results["wigley_rao"]
ax.plot(rao["omegas"], rao["heave_rao_amplitude"], "b-o", linewidth=2, markersize=8)
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect tracking")
ax.axhspan(0.95, 1.05, alpha=0.2, color="green", label="Long-wave criterion [0.95, 1.05]")
ax.axhline(0.2, color="red", linestyle="--", alpha=0.5, label="High-freq criterion (<0.2)")
ax.set_xlabel("ω (rad/s)", fontsize=12)
ax.set_ylabel("Heave RAO amplitude", fontsize=12)
ax.set_title("Wigley III Heave RAO (L=3m, B=0.3m, T=0.1875m)", fontsize=14)
ax.legend(loc="best")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 3.5)
ax.set_ylim(0, 1.2)
fig.tight_layout()
fig.savefig(output_dir / "fig_wigley_rao.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {output_dir / 'fig_wigley_rao.png'}")

# Figure 2: FK excitation force
print("Generating Figure 2: FK excitation force...")
fig, ax = plt.subplots(figsize=(8, 5))
fk = results["fk_excitation"]
omegas = [f["omega"] for f in fk["frequencies"]]
mags = [f["fk_heave_magnitude"] for f in fk["frequencies"]]
ax.semilogy(omegas, mags, "r-s", linewidth=2, markersize=8)
ax.set_xlabel("ω (rad/s)", fontsize=12)
ax.set_ylabel("|F_FK| (N)", fontsize=12)
ax.set_title("Froude-Krylov Heave Excitation (Box Barge L=100m, B=20m, T=8m)", fontsize=14)
ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(output_dir / "fig_fk_excitation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {output_dir / 'fig_fk_excitation.png'}")

# Figure 3: Mesh convergence
print("Generating Figure 3: Mesh convergence...")
fig, ax = plt.subplots(figsize=(8, 5))
conv = results["mesh_convergence"]
panels = conv["panel_counts"]
rao_amps = conv["heave_rao_amplitudes"]
ax.semilogx(panels, rao_amps, "g-^", linewidth=2, markersize=10)
ax.axhline(rao_amps[-1], color="gray", linestyle="--", alpha=0.5, label=f"Converged: {rao_amps[-1]:.4f}")
ax.set_xlabel("Number of panels", fontsize=12)
ax.set_ylabel("Heave RAO amplitude (ω=1.0 rad/s)", fontsize=12)
ax.set_title("Mesh Convergence: Wigley III Heave RAO", fontsize=14)
ax.legend(loc="best")
ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(output_dir / "fig_mesh_convergence.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {output_dir / 'fig_mesh_convergence.png'}")

# Figure 4: Hemisphere added mass (summary)
print("Generating Figure 4: Hemisphere added mass anchor...")
fig, ax = plt.subplots(figsize=(6, 4))
anchor = results["hemisphere_anchor"]
categories = ["BEM", "Hulme (1982)", "Zero-freq limit"]
values = [anchor["a33_bem_dimless"], anchor["hulme_a33_dimless"], anchor["a33_zero_freq_dimless"]]
colors = ["blue", "orange", "green"]
bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor="black")
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("A33' (dimensionless)", fontsize=12)
ax.set_title("Hemisphere Heave Added Mass (a=1m, Ka=0.01)", fontsize=14)
ax.set_ylim(0, 1.2)
ax.grid(True, alpha=0.3, axis="y")
# Add value labels on bars
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f"{val:.4f}", ha="center", va="bottom", fontsize=10)
fig.tight_layout()
fig.savefig(output_dir / "fig_hemisphere_anchor.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {output_dir / 'fig_hemisphere_anchor.png'}")

print("\nAll figures generated successfully!")
