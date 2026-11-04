"""Figures + machine-readable summary for the m3134 (Ka, f/a) envelope study."""
from __future__ import annotations
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROWS = [json.loads(l) for l in (HERE / "outputs" / "points.jsonl").read_text().splitlines() if l.strip()]
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

kas = sorted({r["ka"] for r in ROWS})
ratios = sorted({r["f_over_a"] for r in ROWS})
fine = {(r["ka"], r["f_over_a"]): r for r in ROWS if r["panels"] == 320}

def grid(key):
    m = np.full((len(ratios), len(kas)), np.nan)
    for (ka, ra), row in fine.items():
        m[ratios.index(ra), kas.index(ka)] = row[key]
    return m

# --- Fig 1: error map (A33 and B33) -------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, key, title in zip(axes, ("err_a33_scaled", "err_b33_scaled"),
                          (r"heave added mass  $|A_{33}-A_{33}^{ex}|/(0.5\rho V)$",
                           r"heave damping  $|B_{33}-B_{33}^{ex}|/(\rho\omega V)$")):
    m = grid(key)
    im = ax.imshow(m, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(kas)), [f"{k:g}" for k in kas])
    ax.set_yticks(range(len(ratios)), [f"{r:g}" for r in ratios])
    ax.set_xlabel(r"$Ka$"); ax.set_ylabel(r"$f/a$"); ax.set_title(title, fontsize=9)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if not math.isnan(m[i, j]):
                ax.text(j, i, f"{m[i,j]*100:.1f}%", ha="center", va="center",
                        color="w" if m[i, j] < np.nanmax(m) else "k", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("3-D submerged sphere: BEM vs multipole reference, 320 panels", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "envelope_map.png", dpi=150)
plt.close(fig)

# --- Fig 2: mesh convergence at the three budget corners ----------------------
fig, ax = plt.subplots(figsize=(5.2, 4.0))
corner_rows = defaultdict(list)
for r in ROWS:
    if (round(r["ka"], 6), round(r["f_over_a"], 6)) in {(0.2, 1.5), (1.2, 1.1), (2.0, 3.0)}:
        corner_rows[(r["ka"], r["f_over_a"])].append(r)
budget = []
for (ka, ra), rows in sorted(corner_rows.items()):
    rows.sort(key=lambda r: -r["h_max"])
    h = [r["h_max"] for r in rows]; e = [r["err_a33_scaled"] for r in rows]
    ax.loglog(h, e, "o-", label=fr"$Ka={ka:g}$, $f/a={ra:g}$")
    order = None
    if len(e) > 1 and e[-1] > 0 and e[-2] > 0:
        order = math.log(e[-1] / e[-2]) / math.log(h[-1] / h[-2])
    budget.append({"ka": ka, "f_over_a": ra,
                   "errors": e, "h_max": h, "observed_order": order,
                   "finest_err": e[-1]})
ax.set_xlabel(r"$h_{max}$ / a"); ax.set_ylabel(r"$|A_{33}-A^{ex}_{33}|/(0.5\rho V)$")
ax.set_title("Mesh convergence at envelope corners", fontsize=10)
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "mesh_budget.png", dpi=150); plt.close(fig)

# --- Fig 3: A33 vs Ka (BEM vs exact) at three submergences -------------------
fig, ax = plt.subplots(figsize=(5.4, 4.0))
for ra in (1.1, 1.5, 3.0):
    pts = sorted([r for r in ROWS if r["panels"] == 320 and r["f_over_a"] == ra],
                 key=lambda r: r["ka"])
    if not pts:
        continue
    ks = [p["ka"] for p in pts]
    ax.plot(ks, [p["a33"] for p in pts], "o", label=fr"BEM $f/a={ra:g}$")
    ax.plot(ks, [p["exact_a33"] for p in pts], "-", color=ax.lines[-1].get_color(), lw=1)
ax.set_xlabel(r"$Ka$"); ax.set_ylabel(r"$A_{33}/(\rho V)$")
ax.set_title("Heave added mass: markers BEM, lines multipole (320 panels)", fontsize=9)
ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "added_mass_vs_ka.png", dpi=150); plt.close(fig)

summary = {
    "panels_fine": 320,
    "ka_values": kas, "ratios": ratios,
    "err_a33_scaled_map": np.nan_to_num(grid("err_a33_scaled"), nan=-1).tolist(),
    "err_b33_scaled_map": np.nan_to_num(grid("err_b33_scaled"), nan=-1).tolist(),
    "max_err_a33": float(np.nanmax(grid("err_a33_scaled"))),
    "max_err_b33": float(np.nanmax(grid("err_b33_scaled"))),
    "worst_cell": None,
    "budget": budget,
    "rows_fine": [fine[(ka, ra)] for ra in ratios for ka in kas if (ka, ra) in fine],
    "n_rows": len(ROWS),
}
mg = grid("err_a33_scaled"); i, j = np.unravel_index(np.nanargmax(mg), mg.shape)
summary["worst_cell"] = {"ka": kas[j], "f_over_a": ratios[i], "err_a33_scaled": float(mg[i, j])}
(HERE / "outputs" / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: summary[k] for k in ("max_err_a33", "max_err_b33", "worst_cell", "n_rows")}, indent=2))
for b in budget:
    print("corner ka=%.2f f/a=%.2f  errs=%s  order=%.2f" % (b["ka"], b["f_over_a"], ["%.3e" % x for x in b["errors"]], b["observed_order"]))
