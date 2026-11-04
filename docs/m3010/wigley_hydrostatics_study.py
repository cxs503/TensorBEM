"""Wigley hull hydrostatics: mesh study, JSON artifact and figure.

Run from the worktree root:

    PYTHONPATH=src python3 docs/m3010/wigley_hydrostatics_study.py

Writes ``outputs/wigley_hydrostatics/{study.json,convergence.png}``.  Every
number in the report is read from that JSON; nothing is transcribed by hand.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorbem.hydrostatic import (  # noqa: E402
    compute_displaced_volume_and_buoyancy,
    compute_hydrostatic_restoring_matrix,
    compute_waterplane_properties,
)
from tensorbem.wigley_hydrostatics import (  # noqa: E402
    wigley_hydrostatic_reference,
    wigley_wetted_area_reference,
    wigley_wetted_hull_mesh,
)

L, B, T = 10.0, 1.0, 0.625
RHO, G = 1025.0, 9.80665
LEVELS = [(8, 4), (12, 6), (16, 8), (24, 12), (32, 16), (48, 24), (64, 32)]


def main() -> dict:
    out = Path("outputs/wigley_hydrostatics")
    out.mkdir(parents=True, exist_ok=True)

    ref = wigley_hydrostatic_reference(L, B, T, density=RHO, gravity=G)
    area_ref = wigley_wetted_area_reference(L, B, T)

    rows = []
    for ns, nt in LEVELS:
        mesh = wigley_wetted_hull_mesh(L, B, T, ns, nt)
        volume, centre = compute_displaced_volume_and_buoyancy(mesh)
        awp, awp_c, moments = compute_waterplane_properties(mesh)
        props = compute_hydrostatic_restoring_matrix(
            mesh,
            RHO * ref["volume"],
            np.array([0.0, 0.0, -T + 0.5 * T]),
            density=RHO,
            gravity=G,
        )
        c = props.restoring_matrix
        rows.append(
            {
                "n_sections": ns,
                "n_levels": nt,
                "panels": int(mesh.faces.shape[0]),
                "h_mean": float(mesh.h_mean),
                "volume": volume,
                "volume_rel_err": abs(volume - ref["volume"]) / ref["volume"],
                "Cb_mesh": volume / (L * B * T),
                "z_B": float(centre[2]),
                "z_B_rel_err": abs(centre[2] - ref["center_of_buoyancy"][2])
                / abs(ref["center_of_buoyancy"][2]),
                "A_wp": awp,
                "A_wp_rel_err": abs(awp - ref["waterplane_area"]) / ref["waterplane_area"],
                "I_xx": float(moments[0]),
                "I_xx_rel_err": abs(moments[0] - ref["I_xx"]) / ref["I_xx"],
                "I_yy": float(moments[1]),
                "I_yy_rel_err": abs(moments[1] - ref["I_yy"]) / ref["I_yy"],
                "wetted_area": float(mesh.surface_area),
                "wetted_area_rel_err": abs(mesh.surface_area - area_ref) / area_ref,
                "C33": float(c[2, 2]),
                "C44_roll": float(c[3, 3]),
                "C55_pitch": float(c[4, 4]),
                "C33_rel_err": abs(c[2, 2] - ref["C33"]) / ref["C33"],
                "C44_roll_rel_err": abs(c[3, 3] - ref["C55"]) / ref["C55"],
                "C55_pitch_rel_err": abs(c[4, 4] - ref["C44"]) / ref["C44"],
            }
        )

    keys = ["volume_rel_err", "z_B_rel_err", "A_wp_rel_err", "I_xx_rel_err", "I_yy_rel_err"]
    orders = {}
    for key in keys:
        series = [r[key] for r in rows]
        orders[key] = [
            float(math.log2(a / b)) for a, b in zip(series, series[1:]) if b > 0
        ]

    summary = {
        "case": "parabolic Wigley hull, wetted-surface hydrostatics",
        "L": L,
        "B": B,
        "T": T,
        "density": RHO,
        "gravity": G,
        "KG_above_keel": 0.5 * T,
        "closed_form": {
            k: (v.tolist() if isinstance(v, np.ndarray) else float(v))
            for k, v in ref.items()
            if k not in {"density", "gravity"}
        },
        "wetted_area_reference": area_ref,
        "levels": rows,
        "observed_orders": orders,
        "finest": rows[-1],
    }
    (out / "study.json").write_text(json.dumps(summary, indent=2) + "\n")

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    h = np.array([r["h_mean"] for r in rows])
    for key, label in [
        ("volume_rel_err", "displaced volume"),
        ("z_B_rel_err", "centre of buoyancy z_B"),
        ("A_wp_rel_err", "waterplane area"),
        ("I_xx_rel_err", "I_xx (roll)"),
        ("I_yy_rel_err", "I_yy (pitch)"),
    ]:
        ax.loglog(h, [r[key] for r in rows], "o-", ms=3.5, label=label)
    hh = np.array([h[0], h[-1]])
    ax.loglog(hh, 0.25 * (hh / hh[0]) ** 2 * rows[0]["volume_rel_err"], "k--", lw=1.0,
              label=r"$\propto h^2$")
    ax.set_xlabel("mean panel size $h$ (m)")
    ax.set_ylabel("relative error vs closed form")
    ax.set_title("Wigley hull hydrostatics: mesh convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5)

    ax = axes[1]
    mesh = wigley_wetted_hull_mesh(L, B, T, 32, 16)
    v = mesh.vertices
    for x_target in np.linspace(-0.5 * L, 0.5 * L, 9):
        mask = np.isclose(v[:, 0], x_target)
        section = v[mask]
        if section.size == 0:
            continue
        order = np.argsort(section[:, 1])
        ax.plot(section[order, 1], section[order, 2], "-", lw=0.8, color="C0")
    wl = np.abs(v[:, 2]) < 1e-12
    order = np.argsort(v[wl, 0])
    ax.plot(v[wl][order, 0], 100.0 * v[wl][order, 1], "C3-", lw=1.4,
            label="waterline $y_{wl}(x)\\times100$")
    ax.set_xlabel("y (m)  [body plan sections] / x (m) for the waterline")
    ax.set_ylabel("z (m)")
    ax.set_title("Meshed wetted surface (32x16, %d panels)" % mesh.faces.shape[0])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=150)

    print(json.dumps({"finest": rows[-1], "orders": orders}, indent=2))
    return summary


if __name__ == "__main__":
    main()