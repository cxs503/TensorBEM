"""Reproduce the 2-D interior-field domain-guard study (P0-2).

Run from the repository root::

    PYTHONPATH=src python3 docs/m3351/field_guard_study.py

Outputs (written next to this file, programmatically -- no hand-copied numbers
appear in the report doc):

* ``outputs/field_guard.json`` -- every number quoted in the report;
* ``outputs/fig_field_guard.png`` -- two panels: accuracy versus distance to
  the boundary inside the domain, and the guard's rejection map on a grid.

The "legacy" values (how wrong the pre-guard evaluator was outside the domain)
are recomputed from the previous commit's ``src/tensorbem/solver.py`` rather
than quoted from prose: ``--legacy-rev`` selects the revision (default
``HEAD~1``, the commit before the guard landed).  If that revision cannot be
read, the script records ``legacy_available: false`` and the report says so
instead of inventing numbers.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import types

import numpy as np

from tensorbem.geometry import PanelMesh, regular_polygon
from tensorbem.solver import evaluate_interior_field, solve_laplace_dirichlet

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "outputs"
CLEARANCE_FACTOR_3D_STYLE = 0.05


def _load_legacy_solver(revision: str) -> types.ModuleType | None:
    """Execute ``revision``'s ``solver.py`` as ``tensorbem.solver_legacy``."""
    try:
        source = subprocess.check_output(
            ["git", "show", f"{revision}:src/tensorbem/solver.py"],
            cwd=HERE.parent.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    module = types.ModuleType("tensorbem.solver_legacy")
    module.__package__ = "tensorbem"
    sys.modules[module.__name__] = module  # dataclasses look their module up
    exec(compile(source, f"<{revision}:solver.py>", "exec"), module.__dict__)
    if not hasattr(module, "evaluate_interior_field"):
        return None
    return module


def _problem(panel_count: int):
    """Harmonic Dirichlet problem: u = x on a regular polygon, exact u = x."""
    mesh = regular_polygon(panel_count)
    result = solve_laplace_dirichlet(mesh, mesh.centers[:, 0].copy())
    return mesh, result


def _ray_points(mesh: PanelMesh, angle: float, inner: float, outer: float, count: int):
    """Points along one ray, ordered from the interior outwards."""
    radii = np.linspace(inner, outer, count)
    return np.column_stack((radii * np.cos(angle), radii * np.sin(angle)))


def interior_profile(panel_count: int, angle_deg: float = 30.0) -> dict:
    """Error against the analytic field along one ray (guard-honest targets)."""
    mesh, result = _problem(panel_count)
    points = _ray_points(mesh, np.deg2rad(angle_deg), 0.03, 0.999, 40)
    # Add targets at controlled distances from one facet, so the near-boundary
    # band that a 0.05 h_max guard would refuse is actually sampled.
    offsets = np.array([1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1])
    facet = mesh.centers[0] - offsets[:, None] * mesh.normals[0][None, :]
    points = np.vstack((points, facet))
    distance = mesh.boundary_clearance(points)
    keep = (distance > 1.0e-12) & mesh.contains_point(points)
    points, distance = points[keep], distance[keep]
    values = evaluate_interior_field(mesh, result, points)
    error = np.abs(values - points[:, 0])
    h_max = float(np.max(mesh.lengths))
    band = CLEARANCE_FACTOR_3D_STYLE * h_max
    outside_band = distance > band
    in_band = ~outside_band
    return {
        "panels": int(mesh.panel_count),
        "h_max": h_max,
        "band_targets": int(np.sum(in_band)),
        "distance": distance.tolist(),
        "abs_error": error.tolist(),
        "max_abs_error_inside_band": float(np.max(error[in_band])) if np.any(in_band) else None,
        "max_abs_error_outside_band": float(np.max(error[outside_band])),
        "rms_abs_error": float(np.sqrt(np.mean(error**2))),
    }


def legacy_outside(panel_count: int, legacy_rev: str) -> dict:
    """Values the pre-guard evaluator returned outside / on the boundary."""
    legacy = _load_legacy_solver(legacy_rev)
    mesh, _ = _problem(panel_count)
    targets = np.array(
        [
            [2.0, 0.0],  # far outside
            [1.2, 0.3],  # just outside
            [1.0, 0.0],  # on a vertex
            [0.5, 0.0],  # interior control
        ]
    )
    record: dict = {"legacy_available": legacy is not None, "legacy_rev": legacy_rev}
    if legacy is None:
        return record
    legacy_result = legacy.solve_laplace_dirichlet(mesh, mesh.centers[:, 0].copy())
    legacy_values = legacy.evaluate_interior_field(mesh, legacy_result, targets)
    detail = []
    for target, value in zip(targets, legacy_values):
        exact = float(target[0])
        detail.append(
            {
                "target": target.tolist(),
                "legacy_value": float(value),
                "exact": exact,
                "rel_error": float(abs(value - exact) / abs(exact)),
                "inside": bool(mesh.contains_point(target[None, :])[0]),
                "clearance": float(mesh.boundary_clearance(target[None, :])[0]),
            }
        )
    record["points"] = detail
    return record


def guard_behaviour(panel_count: int, grid_size: int = 61) -> dict:
    """Which grid targets the guard refuses, and the cost of accepting them."""
    mesh, result = _problem(panel_count)
    axis = np.linspace(-1.2, 1.2, grid_size)
    gx, gy = np.meshgrid(axis, axis)
    grid = np.column_stack((gx.ravel(), gy.ravel()))
    inside = mesh.contains_point(grid)
    clearance = mesh.boundary_clearance(grid)
    h_max = float(np.max(mesh.lengths))
    rejected_outside = int(np.sum(~inside))
    rejected_boundary = int(np.sum(inside & (clearance <= 1.0e-14 * h_max)))
    accepted = inside & (clearance > 1.0e-14 * h_max)
    values = evaluate_interior_field(mesh, result, grid[accepted])
    error = np.abs(values - grid[accepted][:, 0])
    # A 0.05*h_max band costs how many otherwise-usable targets?
    band_rejected = int(np.sum(accepted & (clearance < CLEARANCE_FACTOR_3D_STYLE * h_max)))
    return {
        "panels": int(mesh.panel_count),
        "grid_points": int(len(grid)),
        "rejected_outside": rejected_outside,
        "rejected_on_boundary": rejected_boundary,
        "accepted": int(np.sum(accepted)),
        "band_rejected_if_factor_005": band_rejected,
        "accepted_max_abs_error": float(np.max(error)),
        "accepted_rms_abs_error": float(np.sqrt(np.mean(error**2))),
        "accepted_min_clearance": float(np.min(clearance[accepted])),
        "h_max": h_max,
    }


def make_figure(profiles: list[dict], legacy: dict, guard: dict, path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)

    axis = axes[0]
    for profile in profiles:
        axis.loglog(
            profile["distance"],
            np.maximum(profile["abs_error"], 1.0e-16),
            marker=".",
            linewidth=1.0,
            label=f"BEM, {profile['panels']} panels",
        )
    if legacy.get("legacy_available"):
        legacy_outside_points = [p for p in legacy["points"] if not p["inside"]]
        if legacy_outside_points:
            axis.loglog(
                [max(p["clearance"], 1.0e-3) for p in legacy_outside_points],
                [max(p["rel_error"], 1.0e-16) for p in legacy_outside_points],
                "rx",
                markersize=9,
                label="pre-guard value, exterior target",
            )
    band = CLEARANCE_FACTOR_3D_STYLE * profiles[0]["h_max"]
    axis.axvline(band, color="0.4", linestyle="--", linewidth=1.0)
    axis.text(
        band,
        3.0e-2,
        " 0.05 h_max (3-D default band)",
        rotation=90,
        fontsize=8,
        color="0.35",
        va="bottom",
    )
    axis.set_xlabel("distance to boundary  [m]")
    axis.set_ylabel("|u_BEM - u_exact|  (relative for red crosses)")
    axis.set_title("(a) interior accuracy vs. clearance; exterior error")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8, loc="lower left")

    axis = axes[1]
    mesh, result = _problem(guard["panels"])
    axis_g = np.linspace(-1.2, 1.2, 241)
    gx, gy = np.meshgrid(axis_g, axis_g)
    grid = np.column_stack((gx.ravel(), gy.ravel()))
    inside = mesh.contains_point(grid)
    clearance = mesh.boundary_clearance(grid)
    accepted = inside & (clearance > 1.0e-14 * guard["h_max"])
    field = np.full(len(grid), np.nan)
    field[accepted] = evaluate_interior_field(mesh, result, grid[accepted])
    field = field.reshape(gx.shape)
    image = axis.pcolormesh(gx, gy, field, shading="auto", cmap="viridis")
    fig.colorbar(image, ax=axis, label="u_BEM")
    rejected = (~inside).reshape(gx.shape)
    axis.contourf(
        gx, gy, rejected.astype(float), levels=[0.5, 1.5], colors="none", hatches=["//"]
    )
    axis.plot(
        np.append(mesh.vertices[:, 0], mesh.vertices[0, 0]),
        np.append(mesh.vertices[:, 1], mesh.vertices[0, 1]),
        color="white",
        linewidth=1.0,
    )
    axis.set_aspect("equal")
    axis.set_title(f"(b) {guard['rejected_outside']} exterior grid targets refused")
    axis.set_xlabel("x  [m]")
    axis.set_ylabel("y  [m]")

    fig.suptitle(
        "2-D interior field evaluator: domain guard (harmonic u = x, "
        f"{guard['panels']}-panel polygon)"
    )
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-rev", default="HEAD~1")
    parser.add_argument("--panels", type=int, nargs="+", default=[32, 64, 128])
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    profiles = [interior_profile(n) for n in args.panels]
    legacy = legacy_outside(32, args.legacy_rev)
    guard = guard_behaviour(32)

    make_figure(profiles, legacy, guard, OUT / "fig_field_guard.png")

    payload = {
        "case": {
            "field": "u = x (harmonic, exact on the boundary)",
            "domain": "regular polygon, radius 1 m, CCW",
            "evaluator": "constant-panel analytic Green representation",
        },
        "interior_profiles": profiles,
        "legacy_outside": legacy,
        "guard": guard,
        "clearance_factor_3d_style": CLEARANCE_FACTOR_3D_STYLE,
    }
    (OUT / "field_guard.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("guard",)}, indent=2))
    for profile in profiles:
        print(
            f"panels={profile['panels']:4d} h_max={profile['h_max']:.4e} "
            f"rms={profile['rms_abs_error']:.3e} "
            f"max_in_band={profile['max_abs_error_inside_band']:.3e} "
            f"max_outside_band={profile['max_abs_error_outside_band']:.3e}"
        )
    if legacy["legacy_available"]:
        for point in legacy["points"]:
            print(
                f"legacy target={point['target']} value={point['legacy_value']:+.6e} "
                f"exact={point['exact']:+.1f} rel_err={point['rel_error']:.3e} "
                f"inside={point['inside']}"
            )
    else:
        print("legacy solver unavailable at", args.legacy_rev)


if __name__ == "__main__":
    main()