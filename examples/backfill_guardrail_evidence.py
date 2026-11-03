"""Evidence generator for the mutation-backfill guardrails.

Produces, with no hand-copied numbers:

* ``outputs/mutation_backfill/guardrail_evidence.json`` — measured values and
  relative errors for the eccentric waterplane, the finite-difference order of
  the depth derivative, and the mutant tables of the two audit sweeps;
* ``outputs/mutation_backfill/guardrail_evidence.png`` — left: the eccentric
  waterplane with its measured centroid; right: log-log finite-difference error
  of ``smooth_depth_derivative`` against the step size (second-order
  truncation), which is the evidence that the depth-derivative path is now
  guarded by a *numerical* reference rather than by a recorded constant.

Run with::

    PYTHONPATH=src python3 examples/backfill_guardrail_evidence.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tensorbem.free_surface import green_function, smooth_depth_derivative
from tensorbem.geometry3d import WettedHullMesh, box_hull_mesh
from tensorbem.hydrostatic import compute_waterplane_properties

OUT = Path("outputs/mutation_backfill")
REPORT_FILES = {
    "combined sweep (hydrostatic + free_surface)": OUT / "mutation_report.json",
    "hydrostatic re-run (sub-threshold waterline added)": (
        OUT.parent / "mutation_backfill_hydrostatic" / "mutation_report.json"
    ),
}
NU = 1.0
FIELD = (0.7, -1.3)
SOURCE = (0.0, -2.1)


def eccentric_waterplane() -> dict:
    length, beam, draft = 2.0, 1.0, 1.0
    base = box_hull_mesh(length, beam, draft, n_length=3, n_beam=2, n_draft=2)
    shift = np.array([1.0, 0.5, 0.0])
    hull = WettedHullMesh(base.vertices + shift, base.faces)
    area, centroid, moments = compute_waterplane_properties(hull)
    analytic = {
        "area": length * beam,
        "centroid": [1.0, 0.5],
        "i_xx": length * beam**3 / 12.0,
        "i_yy": beam * length**3 / 12.0,
        "i_xy": 0.0,
    }
    measured = {
        "area": float(area),
        "centroid": [float(centroid[0]), float(centroid[1])],
        "i_xx": float(moments[0]),
        "i_yy": float(moments[1]),
        "i_xy": float(moments[2]),
    }
    rel = {
        "area": abs(measured["area"] - analytic["area"]) / analytic["area"],
        "centroid_x": abs(measured["centroid"][0] - analytic["centroid"][0]),
        "centroid_y": abs(measured["centroid"][1] - analytic["centroid"][1]),
        "i_xx": abs(measured["i_xx"] - analytic["i_xx"]) / analytic["i_xx"],
        "i_yy": abs(measured["i_yy"] - analytic["i_yy"]) / analytic["i_yy"],
        "i_xy": abs(measured["i_xy"] - analytic["i_xy"]),
    }
    # Waterline polygon in the shifted frame, for the figure.
    loops = [
        hull.vertices[edge, :2].tolist() for edge in hull.waterline_edges
    ]
    return {
        "length": length,
        "beam": beam,
        "draft": draft,
        "shift": shift.tolist(),
        "analytic": analytic,
        "measured": measured,
        "absolute_or_relative_error": rel,
        "waterline_edge_midpoints": loops,
    }


def finite_difference_study() -> dict:
    x, z = FIELD
    sx, sz = SOURCE
    analytic = smooth_depth_derivative(x, z, sx, sz, NU)
    steps = [1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5, 3.0e-6, 1.0e-6]
    rows = []
    for step in steps:
        plus = green_function(x, z, sx, sz + step, NU, include_rankine=False)
        minus = green_function(x, z, sx, sz - step, NU, include_rankine=False)
        fd = (plus - minus) / (2.0 * step)
        err = abs(analytic - (-fd)) / abs(analytic)
        rows.append({"step": step, "relative_error": float(err)})
    return {"analytic_value": [analytic.real, analytic.imag], "rows": rows}


def mutant_tables() -> dict:
    tables = {}
    for label, path in REPORT_FILES.items():
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        rows = []
        for row in payload.get("rows", []):
            rows.append(
                {
                    "mutant": row.get("key", ""),
                    "status": row.get("status", ""),
                    "killed_by": row.get("failing_test") or row.get("killed_by") or "",
                    "tests": list(row.get("tests", [])),
                }
            )
        tables[label] = {
            "summary": payload.get("summary", payload.get("score", {})),
            "rows": rows,
        }
    return tables


def render(mutants: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    waterplane = eccentric_waterplane()
    fd = finite_difference_study()
    payload = {
        "eccentric_waterplane": waterplane,
        "depth_derivative_finite_difference": fd,
        "mutation_sweeps": mutants,
    }
    (OUT / "guardrail_evidence.json").write_text(json.dumps(payload, indent=2))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    shift = np.asarray(waterplane["shift"])
    hull = box_hull_mesh(
        waterplane["length"], waterplane["beam"], waterplane["draft"],
        n_length=3, n_beam=2, n_draft=2,
    )
    verts = hull.vertices + shift
    on_waterline = np.abs(verts[:, 2]) < 1.0e-12
    # Draw the waterline loop polygon from the hull's boundary edges.
    order = []
    edges = [tuple(e) for e in hull.waterline_edges]
    if edges:
        order = [edges[0][0], edges[0][1]]
        remaining = edges[1:]
        while remaining:
            for idx, (a, b) in enumerate(remaining):
                if a == order[-1]:
                    order.append(b)
                    remaining.pop(idx)
                    break
                if b == order[-1]:
                    order.append(a)
                    remaining.pop(idx)
                    break
            else:
                break
    poly = verts[order][:, :2]
    ax0.fill(poly[:, 0], poly[:, 1], alpha=0.25, color="tab:blue",
             label="waterplane (shifted by (1.0, 0.5))")
    ax0.plot(poly[:, 0], poly[:, 1], color="tab:blue")
    measured = waterplane["measured"]["centroid"]
    analytic = waterplane["analytic"]["centroid"]
    ax0.plot(*analytic, "k+", markersize=14, label="analytic centroid")
    ax0.plot(*measured, "ro", markersize=6, mfc="none", label="measured centroid")
    ax0.set_aspect("equal")
    ax0.set_xlabel("x [m]")
    ax0.set_ylabel("y [m]")
    ax0.set_title(
        "Eccentric waterplane: centroid error "
        f"{waterplane['absolute_or_relative_error']['centroid_x']:.1e} m"
    )
    ax0.legend(fontsize=8, loc="lower right")
    ax0.grid(alpha=0.3)

    steps = np.array([row["step"] for row in fd["rows"]])
    errs = np.array([row["relative_error"] for row in fd["rows"]])
    ax1.loglog(steps, errs, "o-", label="|analytic - FD| / |analytic|")
    ref = errs[-3] * (steps / steps[-3]) ** 2
    ax1.loglog(steps, ref, "k--", label="slope 2 reference")
    ax1.set_xlabel("finite-difference step in source_z [m]")
    ax1.set_ylabel("relative error")
    ax1.set_title("dG/dA: analytic vs central difference")
    ax1.invert_xaxis()
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(OUT / "guardrail_evidence.png", dpi=140)
    print(json.dumps(
        {
            "waterplane_error": waterplane["absolute_or_relative_error"],
            "fd_min_relative_error": float(errs.min()),
            "sweeps": list(mutants),
            "figure": str(OUT / "guardrail_evidence.png"),
        },
        indent=2,
    ))


if __name__ == "__main__":  # pragma: no cover
    render(mutant_tables())