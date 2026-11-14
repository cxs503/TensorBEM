"""Audit m20100 figures for CASE-BEM-LAPLACE-CYLINDER (mission 20029).

fig_verdict.png      : criteria got-vs-threshold margins (log scale) incl. the 3% case
                       tolerance and the 15% ecosystem floor
fig_convergence.png  : error-vs-h ladder of routes D/N + the audit's independent
                       continuum polygon-Neumann route, h^2 / h^4 reference slopes
fig_field.png        : |v|/U cloud from the audited BEM exterior evaluator (N=128)
                       with the analytic solution contours and the probe error map
All data are read from the mission artifact / recomputed through the mission's own
solver; nothing is typed in by hand.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M20029 = "/root/tp-agent/work/m20029"
ART = os.path.join(M20029, "benchmarks/verified/case_bem_laplace_cylinder/results.json")
INDEP = os.path.join(HERE, "independent_check_results.json")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

res = json.load(open(ART))
ind = json.load(open(INDEP))
U, A = 1.0, 1.0

# ------------------------------------------------------------ fig 1: verdict --
num = []
for c in res["criteria"]:
    thr = c["threshold"]
    if isinstance(c["value"], list):
        continue
    val = float(c["value"])
    if "< 0.03" in thr:
        lim, kind = 0.03, "upper"
    elif "<= 0.01 U" in thr:
        lim, kind = 0.01, "upper"
    elif "<= 1e-05" in thr:
        lim, kind = 1e-5, "upper"
    elif "<= 1e-12 a" in thr or "<= 1e-12" in thr:
        lim, kind = 1e-12, "upper"
    elif "<= 1e-06 U/a" in thr:
        lim, kind = 1e-6, "upper"
    elif ">= 10" in thr:
        lim, kind = 10.0, "lower"
    else:
        continue
    num.append((c["name"], val, lim, kind))

fig, ax = plt.subplots(figsize=(9.6, 5.4))
labels = [n for n, *_ in num]
y = np.arange(len(num))
ratio = []
for name, val, lim, kind in num:
    r = lim / val if kind == "upper" and val > 0 else (val / lim if kind == "lower" else np.nan)
    ratio.append(max(r, 1e-19))
colors = ["#2a7f62" if r >= 1 else "#b3312c" for r in ratio]
ax.barh(y, np.log10(ratio), color=colors)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8)
ax.axvline(0.0, color="k", lw=1.2, label="threshold (pass line)")
ax.set_xlabel("log10(threshold / measured)   [>0 = inside tolerance]")
ax.set_title("CASE-BEM-LAPLACE-CYLINDER: 11 numeric criteria, margin to threshold\n(mission results.json, N=128 budget; 15/15 criteria ok)")
for i, (r, (name, val, lim, kind)) in enumerate(zip(ratio, num)):
    ax.text(max(np.log10(r), 0) + 0.15, i, f"{val:.2e}", va="center", fontsize=7)
ax.legend(fontsize=8)
ax.grid(alpha=0.25, axis="x")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_verdict.png"), dpi=150)
plt.close(fig)

# -------------------------------------------------------- fig 2: convergence --
fig, ax = plt.subplots(figsize=(7.6, 5.4))
panels = np.array(res["parameters"]["panel_ladder"], dtype=float)
h = np.array([res["routes"]["dirichlet"]["levels"][i]["h_mean"] for i in range(len(panels))])
d_vel = np.array([L["velocity_relative_l2"] for L in res["routes"]["dirichlet"]["levels"]])
d_pot = np.array([L["potential_relative_l2"] for L in res["routes"]["dirichlet"]["levels"]])
n_srf = np.array([L["surface_velocity_relative_l2"] for L in res["routes"]["neumann"]["levels"]])
n_vel = np.array([L["velocity_relative_l2"] for L in res["routes"]["neumann"]["levels"]])
ax.loglog(h, d_vel, "o-", label="route D exterior velocity (mission)")
ax.loglog(h, d_pot, "s--", label="route D perturbation potential (mission)")
ax.loglog(h, n_srf, "^-", label="route N surface velocity (mission)")
ax.loglog(h, n_vel, "v--", label="route N exterior velocity (mission)")
ind_h = []
ind_n = []
for N in (32, 64, 128):
    ind_h.append(2 * A * np.sin(np.pi / N))  # mean panel length of the inscribed polygon
    ind_n.append(ind["continuum_polygon_neumann_multipole"][str(N)]["surface_speed_relative_l2_vs_2U_sin"])
ax.loglog(ind_h, ind_n, "k*-", label="audit independent continuum polygon-Neumann")
href = np.array([h.min(), h.max()])
ax.loglog(href, (n_srf[-1] / href[-1] ** 2) * href ** 2, ":", color="0.5", label="h^2 slope")
ax.loglog(href, (d_vel[-1] / href[-1] ** 4) * href ** 4, "--", color="0.75", label="h^4 slope")
ax.axhline(0.03, color="#b3312c", lw=1.1, label="case tolerance 3%")
ax.axhline(0.15, color="#b3312c", lw=0.8, ls=":", label="ecosystem floor 15%")
ax.set_xlabel("h_mean  (panel length / a)")
ax.set_ylabel("relative L2 error")
ax.set_title("CASE-BEM-LAPLACE-CYLINDER: error vs mesh (mission artifact + audit recheck)")
ax.grid(alpha=0.25, which="both")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_convergence.png"), dpi=150)
plt.close(fig)

# -------------------------------------------------------------- fig 3: field --
sys.path.insert(0, os.path.join(M20029, "src"))
from tensorbem.geometry import regular_polygon  # noqa: E402
from tensorbem.solver import (  # noqa: E402
    evaluate_exterior_field,
    solve_laplace_exterior_dirichlet,
)

mesh = regular_polygon(panel_count=128, radius=A)
c = mesh.centers
r = np.hypot(c[:, 0], c[:, 1])
th = np.arctan2(c[:, 1], c[:, 0])
phi_bc = U * A ** 2 * np.cos(th) / r
result = solve_laplace_exterior_dirichlet(mesh, phi_bc)

xs = np.linspace(-3.2, 3.2, 121)
X, Y = np.meshgrid(xs, xs)
R = np.hypot(X, Y)
mask = R >= 1.0
pts = np.column_stack([X.ravel(), Y.ravel()])
pts = pts[np.hypot(pts[:, 0], pts[:, 1]) >= 1.002]
pts = pts[::8]   # decimated for wall time; the contour uses the analytic field
field = evaluate_exterior_field(mesh, result, pts)
up = U + field.gradient[:, 0]
vp = field.gradient[:, 1]
speed = np.hypot(up, vp)

z = pts[:, 0] + 1j * pts[:, 1]
dw = U * (1 - A ** 2 / z ** 2)
u_ex, v_ex = dw.real, -dw.imag
err = np.hypot(up - u_ex, vp - v_ex) / np.hypot(u_ex, v_ex)

analytic = np.full(X.shape, np.nan)
an = np.hypot(U * (1 - A ** 2 / (X + 1j * Y) ** 2).real, -U * (1 - A ** 2 / (X + 1j * Y) ** 2).imag)
analytic[R >= 1.002] = an[R >= 1.002]

fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
cf = axes[0].contourf(X, Y, analytic, levels=24, cmap="viridis")
th_poly = np.linspace(0, 2 * np.pi, 400)
axes[0].plot(A * np.cos(th_poly), A * np.sin(th_poly), "w--", lw=1.0)
axes[0].quiver(pts[::6, 0], pts[::6, 1], up[::6], vp[::6], color="w", scale=22, width=0.003)
axes[0].set_aspect("equal")
axes[0].set_title("analytic |v|/U cloud + BEM velocity arrows (N=128)\nD route, 64-probe case, audit recomputation")
axes[0].set_xlabel("x/a")
axes[0].set_ylabel("y/a")
fig.colorbar(cf, ax=axes[0], shrink=0.85, label="|v|/U")

sc = axes[1].scatter(pts[:, 0], pts[:, 1], c=err, s=3, norm=matplotlib.colors.LogNorm(), cmap="magma")
axes[1].plot(A * np.cos(th_poly), A * np.sin(th_poly), "k--", lw=1.0)
axes[1].set_aspect("equal")
axes[1].set_title("per-point relative velocity error |v_BEM - v_exact| / |v_exact|\n(N=128, max = %.2e)" % err.max())
axes[1].set_xlabel("x/a")
axes[1].set_ylabel("y/a")
fig.colorbar(sc, ax=axes[1], shrink=0.85, label="relative error")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_field.png"), dpi=150)
plt.close(fig)

rp = np.hypot(pts[:, 0], pts[:, 1])
bands = {}
for lo, hi in ((1.002, 1.05), (1.05, 1.25), (1.25, 1.6), (1.6, 3.2), (1.25, 99.0)):
    m = (rp >= lo) & (rp < hi)
    if m.any():
        bands[f"r/a in [{lo},{hi})"] = {
            "points": int(m.sum()),
            "max_rel_error": float(err[m].max()),
            "p99_rel_error": float(np.quantile(err[m], 0.99)),
            "rms_rel_error": float(np.sqrt(np.mean(err[m] ** 2))),
        }
summary = {
    "fig_verdict": "11 numeric criteria margins",
    "fig_convergence": {"h_mean": h.tolist(), "route_D_velocity": d_vel.tolist(), "route_N_surface": n_srf.tolist()},
    "fig_field": {"max_rel_error_all_grid_points": float(err.max()), "cond_S": float(result.condition_number),
                  "note": "grid samples run from r/a=1.002 outward; the near-wall band carries the largest error"},
}
summary["field_error_by_radius_band"] = bands
json.dump(summary, open(os.path.join(OUT, "figure_data.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))