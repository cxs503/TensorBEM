"""Evidence for m4529: MultiComponentMesh domain-guard protocol (PR #283 red).

Background
----------
``tensorbem.solver.evaluate_interior_field`` screens its targets before applying
the interior Green representation:

* ``mesh.boundary_clearance(points)``  -- reject targets on/too near a panel,
* ``mesh.contains_point(points)``      -- reject targets outside the domain.

``PanelMesh`` implements both.  ``MultiComponentMesh`` (outer loop + hole loops,
used by ``annulus_mesh`` and by ``multicomponent.evaluate_...``) implemented
neither, so every guarded annulus evaluation died with
``AttributeError: 'MultiComponentMesh' object has no attribute
'boundary_clearance'`` -- 3 failed + 5 errors in ``tests/test_multicomponent.py``.

This script measures the two methods against closed forms that share none of
their implementation:

* clearance: for an inscribed-polygon annulus mesh the numeric distance is
  compared with the analytic annulus distance ``min(|rho - r_in|, |r_out - rho|)``
  (``rho`` = radius of the probe point).  An inscribed polygon lies inside the
  true boundary, so the error must be one-signed and shrink like the chord
  sagitta, i.e. second order in the panel size.
* containment: the loop-parity test is compared with the analytic indicator
  ``r_in < rho < r_out`` on the same probe set.

Usage (works on both trees; pre-fix raises ``AttributeError``)::

    PYTHONPATH=src python docs/m4529/guard_evidence.py --out artifacts/m4529/guard_evidence.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensorbem.multicomponent import annulus_mesh

R_IN = 0.5
R_OUT = 1.0
PROBE_PER_RING = 720


def analytic_clearance(points: np.ndarray) -> np.ndarray:
    rho = np.hypot(points[:, 0], points[:, 1])
    return np.minimum(np.abs(rho - R_IN), np.abs(R_OUT - rho))


def analytic_inside(points: np.ndarray) -> np.ndarray:
    rho = np.hypot(points[:, 0], points[:, 1])
    return (rho > R_IN) & (rho < R_OUT)


def probe_points() -> np.ndarray:
    """Rings at mid radius, just outside the hole and just inside the outer loop."""
    theta = 2.0 * math.pi * np.arange(PROBE_PER_RING) / PROBE_PER_RING
    radii = (0.75, R_IN + 0.05, R_OUT - 0.05)
    return np.array([[r * math.cos(t), r * math.sin(t)] for r in radii for t in theta])


def measure(panel_counts: tuple[int, ...]) -> dict:
    points = probe_points()
    exact = analytic_clearance(points)
    rows = []
    for panels in panel_counts:
        mesh = annulus_mesh(R_IN, R_OUT, panels, reference_length=R_OUT)
        clearance = np.asarray(mesh.boundary_clearance(points), dtype=float)
        error = np.abs(clearance - exact)
        rows.append(
            {
                "panels_per_loop": int(panels),
                "panel_size": float(2.0 * math.pi * R_OUT / panels),
                "max_abs_error": float(np.max(error)),
                "max_rel_error": float(np.max(error / exact)),
                "error_within_chord_sagitta": bool(
                    np.max(error) <= 3.0 * 0.5 * (1.0 - math.cos(math.pi / panels))
                ),
                "contains_point_agreement": float(
                    np.mean(
                        np.asarray(mesh.contains_point(points), dtype=bool)
                        == analytic_inside(points)
                    )
                ),
            }
        )
    order = math.log(rows[0]["max_abs_error"] / rows[-1]["max_abs_error"]) / math.log(
        rows[-1]["panels_per_loop"] / rows[0]["panels_per_loop"]
    )
    return {"rows": rows, "observed_order": float(order)}


def figure(payload: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = 256
    mesh = annulus_mesh(R_IN, R_OUT, panels, reference_length=R_OUT)
    grid = np.linspace(-1.25, 1.25, 241)
    xx, yy = np.meshgrid(grid, grid)
    flat = np.column_stack((xx.ravel(), yy.ravel()))
    numeric = np.asarray(mesh.boundary_clearance(flat), dtype=float).reshape(xx.shape)
    exact = analytic_clearance(flat).reshape(xx.shape)
    band = 0.05 * float(np.max(mesh.lengths))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    ax = axes[0]
    image = ax.pcolormesh(
        xx,
        yy,
        np.ma.masked_array(numeric, mask=~analytic_inside(flat).reshape(xx.shape)),
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=0.5,
    )
    ax.contour(xx, yy, numeric, levels=[band], colors="red", linewidths=1.2,
               linestyles="--")
    for label, loop in (("outer", mesh.outer), ("hole", mesh.holes[0])):
        ax.plot(loop[:, 0], loop[:, 1], "w-", lw=0.8, label=label)
    ax.set_aspect("equal")
    ax.set_title(
        f"MultiComponentMesh.boundary_clearance (annulus, {panels} panels/loop)\n"
        f"dashed red = guard band 0.05*h_max = {band:.4f}",
        fontsize=10,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(image, ax=ax, label="distance to nearest panel")

    ax = axes[1]
    rows = payload["rows"]
    counts = np.array([row["panels_per_loop"] for row in rows], dtype=float)
    errors = np.array([row["max_abs_error"] for row in rows], dtype=float)
    ax.loglog(counts, errors, "o-", label="numeric vs analytic clearance")
    anchor = errors[0] * (counts / counts[0]) ** -2.0
    ax.loglog(counts, anchor, "k--", lw=1.0, label=r"second order ($N^{-2}$)")
    ax.set_xlabel("panels per loop")
    ax.set_ylabel(r"$\max |\Delta|$  (m)")
    ax.set_title(
        "Inscribed-polygon clearance error vs panel count\n"
        f"observed order = {payload['observed_order']:.2f}; "
        "containment agreement = "
        f"{min(row['contains_point_agreement'] for row in rows):.3f}",
        fontsize=10,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/m4529/guard_evidence.json"))
    parser.add_argument("--figure", type=Path, default=Path("artifacts/m4529/guard_clearance.png"))
    args = parser.parse_args()
    payload = {
        "case": "annulus r_in=0.5 m, r_out=1.0 m, MultiComponentMesh domain-guard protocol",
        "panel_counts": [32, 64, 128, 256],
    }
    payload.update(measure((32, 64, 128, 256)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    figure(payload, args.figure)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())