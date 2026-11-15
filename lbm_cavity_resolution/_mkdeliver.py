import json, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from d2q9_cavity import GHIA100_U, GHIA100_V, GHIA100_X, GHIA100_Y
J = json.load(open("results/cavity_resolution_study.json"))
rows = sorted(J["rows"], key=lambda r: r["n"])
P = np.load("results/cavity_profiles.npz")
os.makedirs("figures", exist_ok=True); os.makedirs("../docs", exist_ok=True)
figs = []
for r in rows:
    n = r["n"]; y, u, x, v = P[f"n{n}"]
    f, axs = plt.subplots(1, 2, figsize=(9.0, 3.8))
    axs[0].plot(u, y, "b-", lw=1.2, label=f"LBM n={n} ({r['steps']} steps)")
    axs[0].plot(GHIA100_U, GHIA100_Y, "ko", ms=4, label="Ghia 1982 Re=100 Tbl I")
    axs[0].set_xlabel("u/u_lid"); axs[0].set_ylabel("y/L"); axs[0].grid(alpha=.3)
    axs[0].set_title(f"vertical centreline (max err {100*r['errors']['max_u']:.2f}%)")
    axs[1].plot(x, v, "r-", lw=1.2, label=f"LBM n={n}")
    axs[1].plot(GHIA100_V, GHIA100_X, "ko", ms=4, label="Ghia 1982 Re=100 Tbl II (corrected)")
    axs[1].set_xlabel("v/u_lid"); axs[1].set_ylabel("x/L"); axs[1].grid(alpha=.3)
    axs[1].set_title(f"horizontal centreline (max err {100*r['errors']['max_v']:.2f}%)")
    for a in axs: a.legend(fontsize=7)
    f.tight_layout(); p = f"figures/fig_n{n}_profiles_vs_ghia.png"; f.savefig(p, dpi=150); plt.close(f)
    figs.append(p); axs = None
print(json.dumps(figs))
