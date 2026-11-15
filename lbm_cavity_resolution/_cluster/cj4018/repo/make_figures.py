"""Figures for the cavity resolution-convergence study (GAP-DIFFLBM-CAV-RES)."""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d2q9_cavity import GHIA100_U, GHIA100_V, GHIA100_X, GHIA100_Y  # noqa: E402

J = json.load(open("results/cavity_resolution_study.json"))
rows = sorted(J["rows"], key=lambda r: r["n"])
fits = J["order_fits"]
figs = []

# ---- Fig 1: log-log convergence -------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 4.6))
h = np.array([r["h"] for r in rows])
for key, lab, mk in (("max_uv", "max-point (u and v)", "o"),
                     ("max_u", "max-point, u only", "s"),
                     ("l2_u", "L2, u", "^"),
                     ("max_v", "max-point, v only", "d")):
    e = np.array([r["errors"][key] for r in rows])
    ax.loglog(h, e, mk + "-", label=f"{lab}  (p={fits[key]['slope']:.2f})")
e_leg = np.array([r["errors_legacy_v"]["max_v"] for r in rows])
ax.loglog(h, e_leg, "x--", color="0.5",
          label=f"v vs legacy TensorLBM Table II (invalid)")
p2 = h[0] * (np.array([rows[0]["errors"]["max_uv"], rows[-1]["errors"]["max_uv"]]) /
             (h / h[0]) ** 0) if False else None
ax.axhline(0.03, color="r", lw=1, ls=":")
ax.text(h[0], 0.0315, "3 % acceptance gate", color="r", fontsize=8)
ref = 0.06 * (h / h[0]) ** 2.0
ax.loglog(h, ref, "k:", lw=1)
ax.text(h[-1], ref[-1], " slope 2", fontsize=8, rotation=0)
ax.axvline(1 / 96, color="g", lw=0.8, ls="--")
ax.text(1 / 96, ax.get_ylim()[0] * 1.2, " n=97", color="g", fontsize=8)
ax.set_xlabel("h = 1/(n-1)"); ax.set_ylabel("error / u_lid")
ax.set_title("D2Q9 LBM lid-driven cavity, Re=100: grid convergence")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7.5)
fig.tight_layout(); f1 = "figures/fig1_convergence_loglog.png"; fig.savefig(f1, dpi=160)
plt.close(fig); figs.append(f1)

# ---- Fig 2: centreline profiles vs Ghia ------------------------------------
P = np.load("results/cavity_profiles.npz")
fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.2))
for n, key in [(r["n"], f"n{r['n']}") for r in rows]:
    y, u, x, v = P[key]
    axs[0].plot(u, y, "-", lw=1.0, label=f"LBM n={n}")
    axs[1].plot(x, v, "-", lw=1.0, label=f"LBM n={n}")
axs[0].plot(GHIA100_U, GHIA100_Y, "ko", ms=4, label="Ghia Re=100 (Tbl I)")
axs[1].plot(GHIA100_V, GHIA100_X, "ko", ms=4, label="Ghia Re=100 (Tbl II)")
axs[0].set_xlabel("u / u_lid"); axs[0].set_ylabel("y / L")
axs[0].set_title("Vertical centreline (x = L/2)")
axs[1].set_xlabel("v / u_lid"); axs[1].set_ylabel("x / L")
axs[1].set_title("Horizontal centreline (y = L/2)")
for a in axs:
    a.grid(alpha=0.3); a.legend(fontsize=7)
fig.tight_layout(); f2 = "figures/fig2_centreline_profiles.png"; fig.savefig(f2, dpi=160)
plt.close(fig); figs.append(f2)

# ---- Fig 3: error map over the two centrelines (finest grid) ---------------
r = rows[-1]
y, u, x, v = P[f"n{r['n']}"]
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.4))
axs[0].semilogy(y, np.abs(u - np.interp(y, GHIA100_Y, GHIA100_U)) + 1e-16, "b-", lw=1)
axs[0].semilogy(x, np.abs(v - np.interp(x, GHIA100_X, GHIA100_V)) + 1e-16, "r-", lw=1)
axs[0].set_xlabel("normalised wall coordinate"); axs[0].set_ylabel("|error| / u_lid")
axs[0].set_title(f"Pointwise error, n={r['n']}  (blue u, red v)")
axs[0].grid(alpha=0.3)
axs[1].bar([r["n"] for r in rows], [100 * r["errors"]["max_uv"] for r in rows],
           color="steelblue")
axs[1].axhline(3.0, color="r", ls=":", label="3 % gate")
axs[1].set_xlabel("n"); axs[1].set_ylabel("max-point error [%]")
axs[1].set_title("max-point error vs resolution"); axs[1].legend(fontsize=8)
axs[1].grid(alpha=0.3, axis="y")
fig.tight_layout(); f3 = "figures/fig3_error_map.png"; fig.savefig(f3, dpi=160)
plt.close(fig); figs.append(f3)
print(json.dumps({"figures": figs}, indent=2))
