"""Figures + machine-readable numbers for ``docs/m2970/mutation_gap_closure.md``.

Nothing in the report is transcribed by hand: this script reads
``outputs/targeted_mutation.json`` (written by ``tools/targeted_mutation.py``) and
recomputes the solver-side quantities it plots, dumping everything to
``outputs/figure_data.json``.

Panels
------
A. mutant verdict before/after  -- the audit's survivor list vs the closed list.
B. eccentric waterplane         -- geometry and the exact integral of the guard.
C. finite-difference error of dG/ds_z -- why the 1e-7 tolerance is safe.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorbem.benchmarks.hydrostatic_analytic import box_hull_mesh  # noqa: E402
from tensorbem.free_surface import green_function, green_gradient  # noqa: E402
from tensorbem.geometry3d import WettedHullMesh  # noqa: E402
from tensorbem.hydrostatic import compute_waterplane_properties  # noqa: E402

OUT = REPO_ROOT / "docs" / "m2970" / "outputs"


def _verdicts() -> tuple[list[dict], list[float]]:
    payload = json.loads((OUT / "targeted_mutation.json").read_text())
    return payload["rows"], payload["rows"]


def panel_a(ax, rows) -> None:
    names = [
        f"{r['function'].split('.')[-1]}  L{r['lineno']}  {r['operator']}  [{r['source']}]"
        for r in rows
    ]
    y = np.arange(len(rows))[::-1]
    codes_before = [0 if r["baseline"]["verdict"] == "survived" else 1 for r in rows]
    codes_after = [0 if r["with_new_tests"]["verdict"] == "survived" else 1 for r in rows]
    ax.barh(y + 0.2, codes_before, height=0.36, color="#d9534f", label="pre-existing suite")
    ax.barh(y - 0.2, codes_after, height=0.36, color="#2e8540", label="+ new guards")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["survived (blind spot)", "killed (guard works)"])
    ax.set_xlim(0, 1.05)
    ax.set_title("A. mutant verdict, before vs after", fontsize=9, loc="left")
    for yi, (b, a) in zip(y, zip(codes_before, codes_after)):
        ax.text(0.02, yi + 0.2, "survived" if b == 0 else "killed", va="center", fontsize=6)
        ax.text(0.02, yi - 0.2, "survived" if a == 0 else "killed", va="center", fontsize=6)
    ax.legend(fontsize=6, loc="lower right")


def panel_b(ax) -> dict:
    hull = box_hull_mesh(2.0, 1.0, 1.0)
    shifted = WettedHullMesh(hull.vertices + np.array([1.0, 0.5, 0.0]), hull.faces)
    area, centroid, moments = compute_waterplane_properties(shifted)
    verts = shifted.vertices[shifted.waterline_edges.ravel()]
    loop = np.vstack([verts, verts[0]])
    # order the waterline vertices around the loop: use the traced edges instead
    order, _ = _ordered_loop(shifted)
    poly = shifted.vertices[order][:, :2]
    ax.fill(poly[:, 0], poly[:, 1], alpha=0.18, color="#2b6cb0", label="waterplane (z = 0)")
    ax.plot(
        np.append(poly[:, 0], poly[0, 0]),
        np.append(poly[:, 1], poly[0, 1]),
        "-",
        color="#2b6cb0",
        lw=1.2,
    )
    ax.plot(
        [centroid[0]],
        [centroid[1]],
        marker="+",
        ms=12,
        mew=2,
        color="#c53030",
        label=f"centroid = ({centroid[0]:.3f}, {centroid[1]:.3f})",
    )
    ax.plot(0.0, 0.0, marker="o", ms=4, color="0.4", label="origin (0, 0)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("B. eccentric waterplane: pinning x_moment", fontsize=9, loc="left")
    ax.legend(fontsize=6, loc="upper left")
    ax.text(
        0.98,
        0.04,
        f"A = {area:.6f} m$^2$\n"
        f"I$_{{xx}}$ = {moments[0]:.6f} m$^4$\n"
        f"I$_{{yy}}$ = {moments[1]:.6f} m$^4$\n"
        f"I$_{{xy}}$ = {moments[2]:.2e} m$^4$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
    )
    return {
        "area": float(area),
        "centroid": [float(c) for c in centroid],
        "moments": [float(m) for m in moments],
        "loop_vertices": int(len(order)),
        "unused_edge_vertices": int(verts.shape[0]),
        "loop": loop.tolist(),
    }


def _ordered_loop(hull: WettedHullMesh) -> tuple[list[int], list[int]]:
    """Walk the waterline edges into a single closed loop (test geometry only)."""
    adjacency: dict[int, list[int]] = {}
    for a, b in np.asarray(hull.waterline_edges):
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    start = min(adjacency)
    order = [start]
    previous, current = -1, start
    while True:
        a, b = adjacency[current]
        nxt = b if a == previous else a
        if nxt == start:
            break
        order.append(nxt)
        previous, current = current, nxt
    return order, order


def panel_c(ax) -> dict:
    cases = [
        (0.9, -0.3, 0.3, -2.0, 0.6),
        (0.0, -1.0, 0.0, -1.5, 1.2),
        (-2.5, -0.2, 1.1, -0.9, 0.35),
    ]
    steps = np.logspace(-8, -2, 13)
    data = []
    for field_x, field_z, source_x, source_z, nu in cases:
        _, analytic = green_gradient(field_x, field_z, source_x, source_z, nu)
        errors = []
        for step in steps:
            numeric = (
                green_function(field_x, field_z, source_x, source_z + step, nu)
                - green_function(field_x, field_z, source_x, source_z - step, nu)
            ) / (2.0 * step)
            errors.append(float(abs(numeric - analytic) / abs(analytic)))
        ax.loglog(steps, errors, "o-", ms=2.5, lw=1.0, label=f"nu={nu}")
        data.append(
            {
                "nu": nu,
                "steps": [float(s) for s in steps],
                "relative_error": errors,
                "best_step": float(steps[int(np.argmin(errors))]),
                "min_relative_error": float(np.min(errors)),
            }
        )
    ax.axhline(1.0e-7, color="#c53030", ls="--", lw=0.9, label="guard tolerance 1e-7")
    ax.set_xlabel("central-difference step [m]")
    ax.set_ylabel("|FD - analytic| / |analytic|")
    ax.set_title("C. dG/ds_z finite-difference check", fontsize=9, loc="left")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=6)
    return {"cases": data}


def main() -> dict:
    rows, _ = _verdicts()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    panel_a(axes[0], rows)
    data_b = panel_b(axes[1])
    data_c = panel_c(axes[2])
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "mutation_gap_closure.png", dpi=160)
    plt.close(fig)

    summary = {
        "mutants": [
            {
                "key": r["mutant"],
                "source": r["source"],
                "baseline": r["baseline"]["verdict"],
                "after": r["with_new_tests"]["verdict"],
                "baseline_failing_test": r["baseline"]["failing_test"],
                "after_failing_test": r["with_new_tests"]["failing_test"],
            }
            for r in rows
        ],
        "waterplane_eccentric": data_b,
        "source_depth_derivative_fd": data_c,
    }
    (OUT / "figure_data.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["mutants"], indent=2))
    return summary


if __name__ == "__main__":
    main()