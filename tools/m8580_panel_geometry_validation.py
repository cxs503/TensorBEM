"""Generate m8580 panel-geometry validation artifacts (numbers + figures).

Programmatic source of every number in the m8580 report:

* watertightness / orientation check results (edge incidence, directed-edge
  ownership, outward-normal statistics) for icosphere levels 0..3;
* icosphere area-convergence table vs the analytic ``4*pi`` (levels 0..4,
  the extra level 4 row documents the asymptotic second-order ratio);
* figures: area-error convergence curve vs panel count (comparison against
  the analytic solution), edge-incidence histogram with the open-mesh
  negative control, and the outward normal field on the sphere.

Run:  python tools/m8580_panel_geometry_validation.py
Out:  artifacts/m8580_panel_geometry_validation/{results.json, fig*.png}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorbem.geometry3d import icosphere  # noqa: E402

OUT = REPO / "artifacts" / "m8580_panel_geometry_validation"
EXACT_AREA = 4.0 * np.pi
TABLE_LEVELS = [0, 1, 2, 3]  # acceptance table (finest = 1280 panels)
EXTRA_LEVEL = 4  # asymptotic-ratio documentation row


def undirected_edge_incidence(faces):
    incidence = {}
    for f, (a, b, c) in enumerate(np.asarray(faces)):
        for x, y in ((a, b), (b, c), (c, a)):
            key = (int(x), int(y)) if x < y else (int(y), int(x))
            incidence.setdefault(key, []).append(f)
    return incidence


def directed_edge_owner(faces):
    owner = {}
    for f, (a, b, c) in enumerate(np.asarray(faces)):
        for x, y in ((a, b), (b, c), (c, a)):
            owner[(int(x), int(y))] = f
    return owner


def watertightness_report(faces, vertices):
    incidence = undirected_edge_incidence(faces)
    counts = np.array([len(f) for f in incidence.values()])
    return {
        "edge_count": int(counts.size),
        "edges_with_exactly_two_panels": int(np.sum(counts == 2)),
        "isolated_edges": int(np.sum(counts == 1)),
        "edges_with_more_than_two_panels": int(np.sum(counts > 2)),
        "euler_v_minus_e_plus_f": int(
            len(np.asarray(vertices)) - counts.size + len(np.asarray(faces))
        ),
    }


def orientation_report(mesh):
    owner = directed_edge_owner(mesh.faces)
    violations = 0
    for f, (a, b, c) in enumerate(mesh.faces):
        for x, y in ((a, b), (b, c), (c, a)):
            if owner[(int(y), int(x))] == f:
                violations += 1
    dot = np.einsum("ij,ij->i", mesh.normals, mesh.centroids)
    return {
        "directed_edges": 3 * mesh.panel_count,
        "opposite_direction_violations": violations,
        "panels_with_positive_normal_centroid_dot": int(np.sum(dot > 0.0)),
        "min_normal_centroid_dot": float(np.min(dot)),
        "mean_normal_centroid_dot": float(np.mean(dot)),
        "signed_volume": float(mesh.volume),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    convergence = []
    meshes = {}
    for level in TABLE_LEVELS + [EXTRA_LEVEL]:
        mesh = icosphere(level)
        meshes[level] = mesh
        area = mesh.surface_area
        rel_err = (EXACT_AREA - area) / EXACT_AREA
        convergence.append(
            {
                "subdivisions": level,
                "panels": mesh.panel_count,
                "vertices": mesh.vertex_count,
                "surface_area": area,
                "exact_area": EXACT_AREA,
                "absolute_error": EXACT_AREA - area,
                "relative_error": rel_err,
                "volume": mesh.volume,
                "h_max": mesh.h_max,
            }
        )
    for prev, cur in zip(convergence, convergence[1:]):
        cur["error_ratio_vs_coarser"] = prev["relative_error"] / cur["relative_error"]

    watertightness = {}
    orientation = {}
    for level in TABLE_LEVELS:
        mesh = meshes[level]
        watertightness[f"level_{level}"] = watertightness_report(
            mesh.faces, mesh.vertices
        )
        orientation[f"level_{level}"] = orientation_report(mesh)

    # Negative control: remove one panel from level 1 and re-run the check.
    open_mesh = meshes[1]
    open_faces = open_mesh.faces[1:]
    watertightness["level_1_one_panel_removed"] = watertightness_report(
        open_faces, open_mesh.vertices
    )

    results = {
        "task": "m8580 panel-mesh geometry validation",
        "exact_sphere_area": EXACT_AREA,
        "area_error_gate": 0.01,
        "convergence": convergence,
        "acceptance": {
            "finest_level_relative_error": convergence[-2]["relative_error"],
            "below_one_percent": bool(convergence[-2]["relative_error"] < 0.01),
            "monotone_decrease": bool(
                np.all(
                    np.diff(
                        [row["relative_error"] for row in convergence[:4]]
                    )
                    < 0.0
                )
            ),
            "second_order_ratio_band": [3.0, 5.0],
            "ratios": [
                row["error_ratio_vs_coarser"] for row in convergence[1:4]
            ],
        },
        "watertightness": watertightness,
        "orientation": orientation,
    }
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- fig 1: area-error convergence vs panel count (log-log) ----------
    rows = convergence
    panels = np.array([r["panels"] for r in rows], dtype=float)
    rel_err = np.array([r["relative_error"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=150)
    ax.loglog(panels, rel_err, "o-", color="tab:blue",
              label="icosphere area error vs $4\\pi$")
    ref = rel_err[0] * (panels / panels[0]) ** -2
    ax.loglog(panels, ref, "--", color="tab:gray", label="second-order $O(h^2)$")
    ax.axhline(0.01, color="tab:red", ls=":", lw=1.2, label="1% gate")
    ax.set_xlabel("panel count $F$")
    ax.set_ylabel("relative area error $(4\\pi - A)/4\\pi$")
    ax.set_title("Unit-sphere area convergence (inscribed icosphere)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig1_area_convergence.png")
    plt.close(fig)

    # ---- fig 2: edge-incidence histogram + open-mesh negative control ----
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), dpi=150)
    for ax, (label, key) in zip(
        axes,
        [
            ("closed icosphere (level 1)", "level_1"),
            ("one panel removed (negative control)", "level_1_one_panel_removed"),
        ],
    ):
        wt = watertightness[key]
        values = [
            wt["edges_with_exactly_two_panels"],
            wt["isolated_edges"],
            wt["edges_with_more_than_two_panels"],
        ]
        bars = ax.bar(
            ["2 panels", "1 panel\n(isolated)", ">2 panels"],
            values,
            color=["tab:green", "tab:red", "tab:orange"],
        )
        for bar, value in zip(bars, values):
            ax.annotate(
                str(value),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
            )
        ax.set_title(label)
        ax.set_ylabel("edge count")
    fig.suptitle("Edge incidence: every edge of a closed mesh is shared by exactly 2 panels")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_edge_incidence.png")
    plt.close(fig)

    # ---- fig 3: outward normal field on the sphere ------------------------
    mesh = meshes[2]
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=150,
                           subplot_kw={"projection": "3d"})
    tri = mesh.vertices[mesh.faces]
    ax.add_collection3d(
        Poly3DCollection(tri, facecolors="lightsteelblue",
                         edgecolors="navy", linewidths=0.3)
    )
    c = mesh.centroids
    n = mesh.normals
    ax.quiver(c[:, 0], c[:, 1], c[:, 2],
              0.25 * n[:, 0], 0.25 * n[:, 1], 0.25 * n[:, 2],
              color="crimson", linewidth=0.6)
    ax.set_title("icosphere(2): 320 panels, outward normals (red)")
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    fig.savefig(OUT / "fig3_outward_normals.png")
    plt.close(fig)

    print(json.dumps(results["acceptance"], indent=2))
    print(f"artifacts written to {OUT}")


if __name__ == "__main__":
    main()
