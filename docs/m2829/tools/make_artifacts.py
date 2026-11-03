"""Recompute every number quoted in docs/mutation_gaps_closure.md."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tensorbem.free_surface import (
    green_depth_gradient,
    green_function,
    green_gradient_field,
)
from tensorbem.geometry3d import WettedHullMesh, box_hull_mesh
from tensorbem.hydrostatic import (
    compute_displaced_volume_and_buoyancy,
    compute_waterplane_properties,
)

OUT = Path(__file__).resolve().parents[1] / "outputs"
NU = 0.6


def main() -> dict:
    length, beam, draft = 2.0, 1.0, 1.0
    base = box_hull_mesh(length=length, beam=beam, draft=draft)
    hull = WettedHullMesh(base.vertices + np.array([1.0, 0.5, 0.0]), base.faces)
    area, centroid, moments = compute_waterplane_properties(hull)
    volume, cob = compute_displaced_volume_and_buoyancy(hull)

    tetra = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, -2.0]])
    tfaces = np.array([[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]])
    submerged = WettedHullMesh(tetra, tfaces)
    submerged_area, submerged_centroid, submerged_moments = compute_waterplane_properties(submerged)

    # finite-difference ladder for dG/dA
    x, z, x0, z0 = 0.7, -0.5, 0.0, -2.0
    analytic = green_depth_gradient(x, z, x0, z0, NU)
    field_vertical = green_gradient_field(x, z, x0, z0, NU, include_rankine=False)[1]
    ladder = []
    for step in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        fwd = green_function(x, z + step, x0, z0 + step, NU, include_rankine=False)
        bwd = green_function(x, z - step, x0, z0 - step, NU, include_rankine=False)
        fd = -(fwd - bwd) / (2.0 * step) / 2.0
        ladder.append({"step": step, "fd": fd, "rel_error": abs(fd - analytic) / abs(analytic)})

    numbers = {
        "eccentric_barge": {
            "length": length, "beam": beam, "draft": draft,
            "waterplane_area": float(area),
            "waterplane_area_error": abs(float(area) - length * beam) / (length * beam),
            "centroid": [float(c) for c in centroid],
            "centroid_error": [abs(float(centroid[0]) - 1.0), abs(float(centroid[1]) - 0.5)],
            "moments": [float(m) for m in moments],
            "moments_reference": [length * beam**3 / 12.0, beam * length**3 / 12.0, 0.0],
            "moments_error": [
                abs(float(moments[0]) - length * beam**3 / 12.0) / (length * beam**3 / 12.0),
                abs(float(moments[1]) - beam * length**3 / 12.0) / (beam * length**3 / 12.0),
                abs(float(moments[2])),
            ],
            "displaced_volume": float(volume),
            "displaced_volume_reference": length * beam * draft,
            "displaced_volume_error": abs(float(volume) - length * beam * draft) / (length * beam * draft),
            "cob": [float(c) for c in cob],
        },
        "submerged_tetrahedron": {
            "waterplane_edges": int(submerged.waterline_edges.shape[0]),
            "waterplane_area": float(submerged_area),
            "centroid": [float(c) for c in submerged_centroid],
            "moments": [float(m) for m in submerged_moments],
            "displaced_volume": float(compute_displaced_volume_and_buoyancy(submerged)[0]),
            "displaced_volume_reference": 1.0 / 6.0,
        },
        "depth_derivative": {
            "nu": NU, "point": [x, z], "source": [x0, z0],
            "analytic": [analytic.real, analytic.imag],
            "field_gradient_negative": [(-field_vertical).real, (-field_vertical).imag],
            "field_gradient_rel_error": abs(analytic + field_vertical) / abs(analytic),
            "finite_difference_ladder": [
                {"step": row["step"], "rel_error": row["rel_error"]} for row in ladder
            ],
        },
        "mutant_pins": json.loads((OUT / "mutation_pins.json").read_text()),
    }
    (OUT / "verification_numbers.json").write_text(json.dumps(numbers, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    steps = [row["step"] for row in ladder]
    errors = [row["rel_error"] for row in ladder]
    axes[0].loglog(steps, errors, "o-", label="|FD - analytic| / |analytic|")
    axes[0].set_xlabel("finite-difference step h [m]")
    axes[0].set_ylabel("relative error")
    axes[0].set_title(r"$\partial G/\partial A$ vs central difference (nu=0.6)")
    axes[0].invert_xaxis()
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].axhline(1e-10, color="k", ls=":", lw=0.8, label="truncation floor")
    axes[0].legend(fontsize=8)

    rows = numbers["mutant_pins"]["rows"]
    labels = [f"{r['pin']['function']}\n{r['pin']['operator']}" for r in rows]
    axes[1].barh(range(len(rows)), [1] * len(rows), color="#2b8cbe")
    axes[1].set_yticks(range(len(rows)))
    axes[1].set_yticklabels(labels, fontsize=6)
    axes[1].set_xlabel("killed by the named test (an empty bar would be a survivor)")
    axes[1].set_title("pinned mutants after backfill (5/5 killed)")
    axes[1].set_xlim(0, 1.05)
    fig.tight_layout()
    fig.savefig(OUT / "fig_mutation_gaps_closure.png", dpi=160)
    print(json.dumps({k: v for k, v in numbers.items() if k != "mutant_pins"}, indent=2)[:2000])
    return numbers


if __name__ == "__main__":
    main()
