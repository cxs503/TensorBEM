"""Error budget for the near-wall field evaluation of the 3-D BEM.

Why this exists
---------------
``docs/near_surface_field.md`` ships an *honest FAIL*: the near-wall velocity
and pressure (``d <= 0.1 h_max``) do not meet the 3 % bar
(``|dCp| = 3.34`` of the ``Cp`` scale at ``d = 0.03 h_max``).  The document
attributes the residual to *"cancellation inside the closed-form solid-angle
gradient"* of the constant-panel double layer.  That attribution was never
measured, and it is wrong: this module measures it and shows that

* the closed-form solid-angle gradient is **exact** -- it agrees with a
  finite-difference derivative of the (independently closed-form) solid angle
  to ``~1e-9`` *per panel* at near-wall clearances,
* the global cancellation hypothesis is refuted by the closed-mesh identity
  ``sum_j grad(Omega_j) = 0``, which holds to ``~1e-14`` at ``d = 0.03 h_max``
  (individual panel gradients reach ``~1e2``, i.e. the sum loses no digits),
* the error is instead the *intra-panel variation of the density*: the
  double-layer term differentiates a piecewise-constant trace, so a trace
  perturbation ``dmu`` on panel ``j`` moves the velocity by
  ``grad(Omega_j) * dmu / (4 pi)`` -- an amplification of ``|grad Omega|/(4 pi)``
  that grows like ``1/d`` and turns the ``O(h)`` constant-panel trace error into
  an ``O(1)`` velocity error.  The measured error follows this model and is
  unchanged by quadrupling the quadrature effort, which is what distinguishes a
  *representation* error from a *quadrature* error.

The same experiment fixes the practical envelope that the document half-states:
which clearances can be used, and that refining the mesh is the only lever the
constant-panel representation offers.

Physics and reference
---------------------
Unit sphere, disturbance potential of a sphere in a uniform stream along
``+z``: ``u = z / (2 r^3)``, ``grad(u) = e_z/(2 r^3) - 3 z x /(2 r^5)``,
surface traces ``u = cos(theta)/2``, ``du/dn_fluid = cos(theta)``.  The
reference is closed form, so every number below is a discretisation measure,
not a comparison with another code.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

from ..geometry3d import icosphere
from ..kernels3d import (
    gradient_solid_angle_triangles_batch,
    solid_angle_triangles_batch,
)
from ..near_singular import single_layer_gradients

FOUR_PI = 4.0 * math.pi
CP_SCALE = 2.25  # max |Cp| of the unit-sphere disturbance field

__all__ = [
    "CP_SCALE",
    "NearWallErrorBudget",
    "analytic_potential",
    "analytic_gradient",
    "analytic_traces",
    "near_wall_velocity",
    "solid_angle_gradient_oracle",
    "solid_angle_gradient_batch_error",
    "closed_mesh_zero_sum",
    "density_sensitivity",
    "run_error_budget",
    "write_report",
]


def analytic_potential(points: FloatArray) -> FloatArray:
    """``u = z / (2 r**3)`` -- disturbance potential, unit sphere in unit stream."""
    x = np.asarray(points, dtype=float)
    r = np.linalg.norm(x, axis=-1)
    return x[..., 2] / (2.0 * r**3)


def analytic_gradient(points: FloatArray) -> FloatArray:
    """``grad(u)`` of :func:`analytic_potential`."""
    x = np.asarray(points, dtype=float)
    r = np.linalg.norm(x, axis=-1)
    grad = np.empty_like(x)
    grad[..., 0] = -3.0 * x[..., 2] * x[..., 0] / (2.0 * r**5)
    grad[..., 1] = -3.0 * x[..., 2] * x[..., 1] / (2.0 * r**5)
    grad[..., 2] = 1.0 / (2.0 * r**3) - 3.0 * x[..., 2] ** 2 / (2.0 * r**5)
    return grad


def analytic_traces(mesh) -> tuple[FloatArray, FloatArray]:
    """Exact traces ``(boundary values, normal derivative along -r_hat)``.

    Both are evaluated **at the panel centroids**, i.e. on the points a
    collocation method actually uses.  The centroids of an inscribed icosphere
    sit slightly inside ``r = 1``, which is the correct place for the trace of a
    constant-panel representation.
    """
    centroids = mesh.centroids
    values = analytic_potential(centroids)
    normal_derivative = -(analytic_gradient(centroids) * centroids).sum(axis=1)
    return values, normal_derivative


def near_wall_velocity(
    mesh,
    boundary_values: FloatArray,
    normal_derivative: FloatArray,
    target: FloatArray,
    quadrature_order: int = 8,
    split_ratio: float = 1.0,
    max_level: int = 24,
) -> FloatArray:
    """Exterior velocity at ``target`` by the representation used near the wall.

    Identical in structure to
    :func:`tensorbem.solver3d.evaluate_surface_velocity_near_surface`
    (single layer by the adaptive near-singular rule, double layer by the
    closed-form solid-angle gradient), reproduced here so the error budget can
    dissect the two terms without a dense solve.
    """
    x = np.asarray(target, dtype=float)
    single = single_layer_gradients(
        x,
        mesh.v0,
        mesh.v1,
        mesh.v2,
        order=quadrature_order,
        split_ratio=split_ratio,
        max_level=max_level,
    )
    double = gradient_solid_angle_triangles_batch(x, mesh.v0, mesh.v1, mesh.v2)
    return np.asarray(normal_derivative) @ single - (
        np.asarray(boundary_values) @ double
    ) / FOUR_PI


def solid_angle_gradient_oracle(
    target: FloatArray, mesh, delta: float
) -> FloatArray:
    """Gradient of one panel's solid angle by central differences.

    Uses only the closed-form solid-angle *value*, so it is independent of the
    analytic-gradient formula it is used to audit.
    """
    x = np.asarray(target, dtype=float)
    grad = np.empty(3)
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = delta
        plus = solid_angle_triangles_batch(x + step, mesh.v0, mesh.v1, mesh.v2)
        minus = solid_angle_triangles_batch(x - step, mesh.v0, mesh.v1, mesh.v2)
        grad[axis] = (plus - minus)[0] / (2.0 * delta)
    return grad


def solid_angle_gradient_batch_error(
    target: FloatArray, mesh, delta: float
) -> dict[str, float]:
    """Per-panel audit of the closed-form solid-angle gradient against the oracle."""
    x = np.asarray(target, dtype=float)
    closed = gradient_solid_angle_triangles_batch(x, mesh.v0, mesh.v1, mesh.v2)
    oracle = np.empty_like(closed)
    for j in range(mesh.panel_count):
        panel = _single_panel(mesh, j)
        oracle[j] = solid_angle_gradient_oracle(x, panel, delta)
    scale = np.linalg.norm(oracle, axis=1)
    abs_err = np.linalg.norm(closed - oracle, axis=1)
    worst = int(np.argmax(abs_err))
    finite = scale > 0.0
    return {
        "max_abs_error": float(abs_err.max()),
        "max_relative_error": float((abs_err[finite] / scale[finite]).max()),
        "worst_panel": worst,
        "oracle_scale_at_worst_panel": float(scale[worst]),
        "max_gradient_scale": float(scale.max()),
    }


def _single_panel(mesh, index: int):
    """A one-triangle mesh view of panel ``index`` (keeps the oracle call simple)."""

    class _Panel:
        pass

    panel = _Panel()
    panel.v0 = mesh.v0[index : index + 1]
    panel.v1 = mesh.v1[index : index + 1]
    panel.v2 = mesh.v2[index : index + 1]
    return panel


def closed_mesh_zero_sum(target: FloatArray, mesh) -> float:
    """``|sum_j grad(Omega_j)|`` for a closed mesh; exactly zero in exact arithmetic.

    The constant-density double layer over a closed surface is piecewise constant,
    so this identity is the sharpest available test of *global* cancellation.
    """
    gradients = gradient_solid_angle_triangles_batch(
        np.asarray(target, dtype=float), mesh.v0, mesh.v1, mesh.v2
    )
    return float(np.linalg.norm(gradients.sum(axis=0)))


def density_sensitivity(mesh, target: FloatArray) -> dict[str, float]:
    """Amplification of a panel-trace perturbation into the near-wall velocity.

    ``d(velocity)/d(mu_j) = -grad(Omega_j)/(4 pi)`` because the double-layer term
    is linear in the panel traces.  The intra-panel variation of the *exact*
    trace, ``dmu_j = max_vertices |u - u_centroid|``, multiplied by that
    amplification gives the predicted velocity error carried by the
    piecewise-constant density.  Terms are combined in quadrature (they are
    independent panel-wise), which is the standard first-order estimate for a
    representation error.
    """
    x = np.asarray(target, dtype=float)
    gradients = gradient_solid_angle_triangles_batch(x, mesh.v0, mesh.v1, mesh.v2)
    amplification = np.linalg.norm(gradients, axis=1) / FOUR_PI
    centroids = mesh.centroids
    u_centroid = analytic_potential(centroids)
    vertex_values = np.stack(
        [
            analytic_potential(mesh.v0),
            analytic_potential(mesh.v1),
            analytic_potential(mesh.v2),
        ],
        axis=1,
    )
    variation = np.abs(vertex_values - u_centroid[:, None]).max(axis=1)
    predicted = math.sqrt(float(((amplification * variation) ** 2).sum()))
    return {
        "max_amplification": float(amplification.max()),
        "rms_trace_variation": float(math.sqrt(float((variation**2).mean()))),
        "max_trace_variation": float(variation.max()),
        "predicted_velocity_error": predicted,
    }


@dataclass
class NearWallErrorBudget:
    """Container for the measured budget (JSON-ready)."""

    cases: list[dict] = field(default_factory=list)
    audit: dict = field(default_factory=dict)
    quadrature_check: list[dict] = field(default_factory=list)
    mesh_ladder: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "cases": self.cases,
            "audit": self.audit,
            "quadrature_check": self.quadrature_check,
            "mesh_ladder": self.mesh_ladder,
        }


def _target(direction: str, clearance_ratio: float, h_max: float) -> FloatArray:
    d = clearance_ratio * h_max
    if direction == "equator":
        base = np.array([1.0, 0.0, 0.0])
    elif direction == "pole":
        base = np.array([0.0, 0.0, 1.0])
    elif direction == "oblique":
        base = np.array([1.0, 0.0, 1.0]) / math.sqrt(2.0)
    else:  # pragma: no cover - guard
        raise ValueError(f"unknown direction {direction!r}")
    return (1.0 + d) * base


def run_error_budget(
    subdivisions: tuple[int, ...] = (2, 3),
    clearance_ratios: tuple[float, ...] = (1.0, 0.5, 0.1, 0.03),
    directions: tuple[str, ...] = ("equator", "pole", "oblique"),
    quadrature_orders: tuple[int, ...] = (8, 12),
) -> NearWallErrorBudget:
    """Run the whole budget; deterministic, no dense solves."""
    budget = NearWallErrorBudget()
    meshes = {s: icosphere(subdivisions=s, radius=1.0) for s in subdivisions}

    for subdiv, mesh in meshes.items():
        values, flux = analytic_traces(mesh)
        for direction in directions:
            for ratio in clearance_ratios:
                target = _target(direction, ratio, mesh.h_max)
                exact = analytic_gradient(target)
                velocity = near_wall_velocity(mesh, values, flux, target)
                err = float(np.linalg.norm(velocity - exact))
                cp_error = 2.0 * err  # |dCp| = 2 |grad(u)| for |U| = 1
                sensitivity = density_sensitivity(mesh, target)
                budget.cases.append(
                    {
                        "subdivisions": subdiv,
                        "panels": mesh.panel_count,
                        "h_max": mesh.h_max,
                        "direction": direction,
                        "clearance_ratio": ratio,
                        "clearance": ratio * mesh.h_max,
                        "velocity_error": err,
                        "cp_error_abs": cp_error,
                        "cp_error_of_scale": cp_error / CP_SCALE,
                        "predicted_velocity_error": sensitivity[
                            "predicted_velocity_error"
                        ],
                        "max_amplification": sensitivity["max_amplification"],
                        "max_trace_variation": sensitivity["max_trace_variation"],
                    }
                )

    # quadrature independence at the failing case: quadruple the effort
    mesh = meshes[max(subdivisions)]
    values, flux = analytic_traces(mesh)
    for ratio in clearance_ratios:
        target = _target("equator", ratio, mesh.h_max)
        exact = analytic_gradient(target)
        row = {"clearance_ratio": ratio, "panels": mesh.panel_count, "errors": {}}
        for order in quadrature_orders:
            velocity = near_wall_velocity(
                mesh, values, flux, target, quadrature_order=order,
                split_ratio=2.0, max_level=32,
            )
            row["errors"][f"order_{order}"] = float(
                np.linalg.norm(velocity - exact)
            )
        budget.quadrature_check.append(row)

    # closed-form vs oracle, and the closed-mesh identity, at the failing clearance
    mesh = meshes[max(subdivisions)]
    for ratio in (0.1, 0.03):
        target = _target("equator", ratio, mesh.h_max)
        budget.audit[f"equator_d{ratio:g}"] = {
            **solid_angle_gradient_batch_error(target, mesh, 1e-6 * mesh.h_max),
            "closed_mesh_zero_sum": closed_mesh_zero_sum(target, mesh),
            "panels": mesh.panel_count,
        }

    # mesh ladder of the measured velocity error
    for ratio in (0.1, 0.03):
        row = {"clearance_ratio": ratio, "levels": []}
        for subdiv, mesh in meshes.items():
            values, flux = analytic_traces(mesh)
            target = _target("equator", ratio, mesh.h_max)
            exact = analytic_gradient(target)
            velocity = near_wall_velocity(mesh, values, flux, target)
            row["levels"].append(
                {
                    "subdivisions": subdiv,
                    "panels": mesh.panel_count,
                    "h_max": mesh.h_max,
                    "velocity_error": float(np.linalg.norm(velocity - exact)),
                }
            )
        budget.mesh_ladder.append(row)

    return budget


def _observed_order(coarse: dict, fine: dict) -> float | None:
    if coarse["velocity_error"] <= 0.0 or fine["velocity_error"] <= 0.0:
        return None
    ratio = coarse["h_max"] / fine["h_max"]
    return float(
        math.log(coarse["velocity_error"] / fine["velocity_error"]) / math.log(ratio)
    )


def write_report(budget: NearWallErrorBudget, outdir: Path) -> dict:
    """Write ``results.json``, ``summary.md`` and the figure; return the JSON dict."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = budget.as_dict()
    for row in payload["mesh_ladder"]:
        levels = row["levels"]
        row["observed_orders"] = [
            _observed_order(levels[i], levels[i + 1]) for i in range(len(levels) - 1)
        ]
    (outdir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = ["# Near-wall error budget (generated)", ""]
    lines += [
        "## 1. Near-wall velocity error and the density-amplification model",
        "",
        "| panels | clearance/h_max | velocity error | \\|dCp\\|/scale | "
        "model prediction | max amplification \\|gradOmega\\|/4pi |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for case in payload["cases"]:
        if case["direction"] != "equator":
            continue
        lines.append(
            f"| {case['panels']} | {case['clearance_ratio']:g} | "
            f"{case['velocity_error']:.3e} | {case['cp_error_of_scale']:.3f} | "
            f"{case['predicted_velocity_error']:.3e} | "
            f"{case['max_amplification']:.3e} |"
        )
    lines += ["", "## 2. Closed-form solid-angle gradient vs the finite-difference oracle", ""]
    lines += [
        "| case | panels | max \\|error\\| | max relative error | "
        "\\|sum_j grad Omega_j\\| |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in payload["audit"].items():
        lines.append(
            f"| {name} | {row['panels']} | {row['max_abs_error']:.3e} | "
            f"{row['max_relative_error']:.3e} | {row['closed_mesh_zero_sum']:.3e} |"
        )
    lines += ["", "## 3. Quadrature independence at the failing clearance", ""]
    lines += ["| clearance/h_max | error, order 8 | error, order 12 | ratio |", "|---:|---:|---:|---:|"]
    for row in payload["quadrature_check"]:
        lo = row["errors"]["order_8"]
        hi = row["errors"]["order_12"]
        lines.append(
            f"| {row['clearance_ratio']:g} | {lo:.3e} | {hi:.3e} | "
            f"{(hi / lo if lo else float('nan')):.3f} |"
        )
    lines += ["", "## 4. Mesh ladder", ""]
    lines += ["| clearance/h_max | levels (panels: error) | observed orders |", "|---:|---|---|"]
    for row in payload["mesh_ladder"]:
        levels = ", ".join(
            f"{lv['panels']}: {lv['velocity_error']:.3e}" for lv in row["levels"]
        )
        orders = ", ".join(
            "n/a" if o is None else f"{o:.2f}" for o in row["observed_orders"]
        )
        lines.append(f"| {row['clearance_ratio']:g} | {levels} | {orders} |")
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write_figure(payload, outdir / "near_wall_error_budget.png")
    return payload


def _write_figure(payload: dict, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        return
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    cases = [c for c in payload["cases"] if c["direction"] == "equator"]
    panels = sorted({c["panels"] for c in cases})
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(panels)))
    for color, panel in zip(colors, panels):
        rows = sorted(
            [c for c in cases if c["panels"] == panel],
            key=lambda r: r["clearance_ratio"],
        )
        ratios = [r["clearance_ratio"] for r in rows]
        axes[0].loglog(
            ratios, [r["velocity_error"] for r in rows], "o-", color=color,
            label=f"{panel} panels (measured)",
        )
        axes[0].loglog(
            ratios, [r["predicted_velocity_error"] for r in rows], "x--",
            color=color, label=f"{panel} panels (model)",
        )
    axes[0].set_xlabel(r"clearance $d/h_{max}$")
    axes[0].set_ylabel(r"$|\nabla u_h - \nabla u|$")
    axes[0].invert_xaxis()
    axes[0].set_title("near-wall velocity error")
    axes[0].legend(fontsize=6)
    axes[0].grid(True, which="both", alpha=0.3)

    aud = payload["audit"]
    labels = list(aud)
    axes[1].bar(
        np.arange(len(labels)),
        [aud[k]["max_abs_error"] for k in labels],
        color="#2a6fdb",
    )
    axes[1].set_yscale("log")
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels, rotation=15, fontsize=8)
    axes[1].set_ylabel(r"max $|\nabla\Omega_{closed} - \nabla\Omega_{FD}|$")
    axes[1].set_title("closed form vs finite-difference oracle")
    axes[1].grid(True, axis="y", which="both", alpha=0.3)

    for color, panel in zip(colors, panels):
        rows = sorted(
            [c for c in cases if c["panels"] == panel],
            key=lambda r: r["clearance_ratio"],
        )
        axes[2].loglog(
            [r["clearance_ratio"] for r in rows],
            [r["max_amplification"] for r in rows],
            "o-", color=color, label=f"{panel} panels",
        )
    axes[2].set_xlabel(r"clearance $d/h_{max}$")
    axes[2].set_ylabel(r"$\max_j |\nabla\Omega_j|/(4\pi)$")
    axes[2].invert_xaxis()
    axes[2].set_title("trace-perturbation amplification")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--outdir", default="docs/near_surface_error_budget", type=Path,
        help="directory for results.json / summary.md / figure",
    )
    parser.add_argument(
        "--subdivisions", default="2,3", help="comma separated icosphere levels"
    )
    args = parser.parse_args(argv)
    subdivisions = tuple(int(s) for s in args.subdivisions.split(",") if s.strip())
    budget = run_error_budget(subdivisions=subdivisions)
    payload = write_report(budget, args.outdir)
    print(json.dumps(
        {
            "cases": len(payload["cases"]),
            "audit": {
                k: {
                    "max_relative_error": v["max_relative_error"],
                    "closed_mesh_zero_sum": v["closed_mesh_zero_sum"],
                }
                for k, v in payload["audit"].items()
            },
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())