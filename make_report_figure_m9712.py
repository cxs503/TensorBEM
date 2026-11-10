"""Report figure + tables for the non-affine geometry axis (m9712).

Reads the committed study.json (no hand-copied numbers) and measures the
quadrature-order behaviour of the solid-angle residual on the spot.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorbem.benchmarks.quadratic_panel_verification import (
    affine_image_residual,
    radially_modulated_mesh,
)
from tensorbem.panel_quadratic import (
    assemble_quadratic_matrices,
    ellipsoid_quadratic,
    icosphere_quadratic,
)

ROOT = Path(__file__).resolve().parent
STUDY = json.loads((ROOT / "docs/quadratic_panel_verification/study.json").read_text())
OUT = ROOT / "docs/quadratic_panel_verification/fig_non_affine_geometry_m9712.png"
TABLE = ROOT / "docs/quadratic_panel_verification/non_affine_geometry_m9712.json"

colours = {"sphere": "#1f77b4", "ellipsoid_1p00_1p60_0p70": "#d62728",
           "sheared_ellipsoid_1p00_1p60_0p70": "#2ca02c", "wavy_r3_e0p15": "#9467bd"}

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.5))

# --- (a) flux convergence on the four geometries -------------------------
h_ref = None
for geom in STUDY["geometry"]["geometries"]:
    h = np.array([r["h"] for r in geom["levels"]])
    err = np.array([r["flux_rel_error"] for r in geom["levels"]])
    ax.loglog(h, err, "o-", color=colours.get(geom["name"], "grey"),
              label=f"{geom['name']} (order {geom['observed_order']:.2f})")
ax.axhline(0.03, color="#2ca02c", ls="--", lw=1.0, label="3% acceptance")
ax.axhline(0.15, color="#d62728", ls=":", lw=1.0, label="15% hard limit")
h_all = np.array([r["h"] for g in STUDY["geometry"]["geometries"] for r in g["levels"]])
ref = STUDY["geometry"]["geometries"][0]["levels"][-1]
ax.loglog(h_all, ref["flux_rel_error"] * (h_all / ref["h"]) ** 2, "k-.", lw=1.0,
          label=r"$O(h^2)$ reference")
ax.set_xlabel("mesh size h [-]")
ax.set_ylabel(r"relative $L_2$ flux error [-]")
ax.set_title("(a) flux recovery: affine vs non-affine geometry")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=7)

# --- (b) solid-angle residual vs quadrature order (crease, not quadrature) --
orders = [(8, 16), (12, 24), (16, 32)]
rows = []
for name, mesh in (
    ("unit sphere (affine)", icosphere_quadratic(1)),
    ("ellipsoid 1.0/1.6/0.7 (affine)", ellipsoid_quadratic(1, axes=(1.0, 1.6, 0.7))),
    ("wavy r=P3, eps=0.15 (non-affine)", radially_modulated_mesh(1, 3, 0.15)),
):
    ones = np.ones(mesh.node_count)
    res = []
    for qo, so in orders:
        _S, D = assemble_quadratic_matrices(mesh, quadrature_order=qo, self_order=so)
        res.append(float(np.max(np.abs(D @ ones + 0.5))))
    rows.append({"geometry": name, "residuals": res,
                 "relative_spread": float((max(res) - min(res)) / min(res))})
    ax2.semilogy([o[0] for o in orders], res, "o-", label=name)

ax2.set_xlabel("Gauss order for off-panel rows (self-panel order = 2x)")
ax2.set_ylabel(r"$\max_i |\sum_k D_{ik} + 1/2|$ [-]")
ax2.set_title("(b) solid-angle residual is quadrature-order invariant")
ax2.set_xticks([o[0] for o in orders])
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(fontsize=7)

fig.tight_layout()
fig.savefig(OUT, dpi=160)
plt.close(fig)

classif = {}
ref_nodes = icosphere_quadratic(1).vertices
base = ellipsoid_quadratic(1, axes=(1.0, 1.6, 0.7))
shear = np.array([[1.0, 0.55, 0.0], [0.0, 1.0, 0.0], [0.0, 0.30, 1.0]])
from tensorbem.panel_quadratic import QuadraticSurfaceMesh

classif["sphere"] = affine_image_residual(icosphere_quadratic(1), ref_nodes)
classif["ellipsoid"] = affine_image_residual(base, ref_nodes)
classif["sheared_ellipsoid"] = affine_image_residual(
    QuadraticSurfaceMesh((shear @ base.vertices.T).T, base.faces.copy()), ref_nodes
)
classif["wavy_r3_e0p15"] = affine_image_residual(
    radially_modulated_mesh(1, 3, 0.15), ref_nodes
)

TABLE.write_text(json.dumps(
    {"figure": OUT.name, "quadrature_order_ladder": {"orders": orders, "rows": rows},
     "affine_image_residual_sub1": classif}, indent=2))
print(json.dumps({"figure": str(OUT.relative_to(ROOT)), "ladder": rows,
                  "affine": classif}, indent=2))