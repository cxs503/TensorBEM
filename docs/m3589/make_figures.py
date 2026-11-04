"""Figures for the finite-depth declared-number gate (m3589).

Run with ``PYTHONPATH=src python3 docs/m3589/make_figures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorbem.benchmarks import finite_depth3d_pins as pins  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "m3589" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


ledger = pins.evaluate()
measured = pins.measure()

claims = ledger["claims"]

fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.4))

# --- (a) declared vs recomputed: deviation in units of the printed tolerance
value_claims = [c for c in claims if c["kind"] == "value" and c["tolerance"] > 0]
value_claims.sort(key=lambda c: c["relative"] / c["tolerance"])
ratio = np.array([c["relative"] / c["tolerance"] for c in value_claims])
ax = axes[0, 0]
ax.semilogy(ratio, "o", ms=2.6, color="#1f77b4")
ax.axhline(1.0, color="crimson", lw=1.2, ls="--", label="printed tolerance")
ax.set_xlabel(f"{len(value_claims)} value-class claims (sorted)")
ax.set_ylabel(r"$|\mathrm{measured}-\mathrm{declared}|/\mathrm{tol}$")
ax.set_title("every documented number reproduces\n(gap in units of its printed half-ulp)")
ax.set_ylim(1e-6, 10.0)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=7, loc="upper left")

# --- (b) singularity strength vs the analytic value 1
radii = np.array(pins.SING_RADII)
strength = np.array([measured["singularity"][r]["four_pi_r_G"] for r in radii])
declared_strength = np.array([c["declared"] for c in claims if c["id"].startswith("singularity/")])
ax = axes[0, 1]
ax.semilogx(radii, np.abs(strength - 1.0), "s-", ms=4, label="recomputed (this run)")
ax.semilogx(radii, np.abs(declared_strength - 1.0), "x", ms=6, color="darkorange",
            label="declared in doc")
ax.axhline(3e-2, color="crimson", ls="--", lw=1.0, label="3 % acceptance line")
ax.invert_xaxis()
ax.set_xlabel("source distance $r$ [m]")
ax.set_ylabel(r"$|4\pi r\,\Re G-1|$")
ax.set_title(r"singularity strength $\to 1$ (analytic reference)")
ax.set_yscale("log")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=7)

# --- (c) deep-water collapse against the independent Kelvin kernel
depths = np.array(pins.DEEP_DEPTHS)
k0h = np.array([measured["deep_water_limit"][d]["k0_h"] for d in depths])
dev = np.array([measured["deep_water_limit"][d]["max_rel_error"] for d in depths])
declared_dev = np.array(
    [c["declared"] for c in claims if c["id"].endswith("/deviation")]
)
ax = axes[1, 0]
ax.semilogy(k0h, dev, "o-", ms=4, label="recomputed")
ax.semilogy(k0h, declared_dev, "x", ms=6, color="darkorange", label="declared in doc")
ax.annotate(
    f"last row: {dev[-1]:.3e}\n(at $k_0h$={k0h[-1]:.1f}$)",
    xy=(k0h[-1], dev[-1]),
    xytext=(0.35, 0.55),
    textcoords="axes fraction",
    fontsize=7,
    arrowprops=dict(arrowstyle="->", lw=0.7),
)
ax.set_xlabel("$k_0 h$")
ax.set_ylabel(r"max relative deviation from `free_surface3d`")
ax.set_title("deep-water limit against the independent Kelvin kernel")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=7)

# --- (d) evanescent tail of the far field
k0 = measured["far_field"]["k0"]
ratios = measured["far_field"]["evanescent"]
k0R = np.array(sorted(ratios))
vals = np.array([ratios[k] for k in k0R])
declared_e = {c["id"]: c["declared"] for c in claims if c["id"].startswith("far_field/evanescent/")}
ax = axes[1, 1]
ax.semilogy(k0R, vals, "o-", ms=3, label="recomputed")
ax.set_xlabel("$k_0 R$  ($h=4$ m)")
ax.set_ylabel(r"$|G_{\mathrm{evan}}|/|G|$")
ax.set_title("evanescent tail decays exponentially (h = 4 m)")
for cid, value in declared_e.items():
    radius = float(cid.rsplit("=", 1)[1])
    ax.plot([radius], [value], "x", ms=7, color="darkorange")
ax.plot([], [], "x", ms=7, color="darkorange", label="declared in doc")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=7)

fig.suptitle(
    "TensorBEM docs/finite_depth3d.md: 110 declared numbers re-measured "
    f"({ledger['coverage']['protected_fraction']:.1%} of the document's numeric tokens)",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
path = OUT / "finite_depth3d_doc_pins.png"
fig.savefig(path, dpi=76, optimize=True, compress_level=9)
plt.close(fig)
print("wrote", path, path.stat().st_size)
print(json.dumps(ledger["coverage"], indent=1))