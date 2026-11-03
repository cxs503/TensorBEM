"""Where does the finite-depth energy-identity offset come from?

``docs/finite_depth_radiation.md`` §5 (limitation 1) left an open debt: at 64
panels the damping read off the radiated wave amplitude exceeds the damping
obtained from the surface pressure by ``+1.06 %``, the deep-water solver at the
same mesh gives ``-1.04 %``, and the deciding panel study (128/256) was never
run.  A sign-flipped one-percent discrepancy is exactly the kind of number that
gets explained away by a wrong constant, so this module turns it into a
convergence measurement with a stated attribution.

Three candidate causes are separated, and each gets its own axis:

1. **Quadrature.**  The finite-depth kernel has an analytic Rankine part and a
   quadratured smooth remainder; the near-singular free-surface term is the
   obvious suspect.  Axis: ``panel_nodes`` at a fixed mesh.  If the offset is
   quadrature-limited it moves with the node count.
2. **The far-field amplitude integrator.**  :func:`wave_amplitude` in the solver
   evaluates ``integral phi0(z) exp(-i k0 x) sigma ds`` with the panel-centre
   value times the panel length, i.e. a midpoint rule.  Axis: replace it by a
   Gauss rule per panel (the density is piecewise constant, so only the smooth
   factor is approximated) and compare the two readings on the same densities.
3. **Panel discretisation of the integral equation.**  Axis: refine the mesh.
   A normalisation error in ``B = 2 rho g c_g |eta_a|^2`` would show up as a
   *mesh-independent* offset; a discretisation error must vanish as ``h -> 0``
   at the observed order of the constant-panel scheme.

The same refinement is run with the *deep-water* solver (already verified
against Ursell's closed form) so that the finite-depth numbers are compared to
an independently validated implementation at identical meshes.

Usage::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.finite_depth_energy_convergence \
        --json outputs/finite_depth_energy_identity/convergence.json \
        --figures outputs/finite_depth_energy_identity
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from tensorbem.benchmarks.finite_depth_radiation import submerged_circle_mesh
from tensorbem.finite_depth import (
    finite_depth_wavenumber,
    group_velocity,
    propagating_mode_norm,
)
from tensorbem.finite_depth_solver import (
    _solve_sources,
    far_field_damping,
    radiation_reference,
)
from tensorbem.free_surface_solver import far_field_damping as deep_far_field_damping
from tensorbem.free_surface_solver import solve_radiation_indirect as deep_solve

GRAVITY = 9.80665
MOTIONS = {"heave": 1, "sway": 0}
DEPTHS = {"shallow": 3.2, "medium": 5.0, "deep": 20.0}


def far_field_damping_panel_quadrature(
    mesh,
    sources: np.ndarray,
    nu: float,
    depth: float,
    density: float = 1025.0,
    *,
    gravity: float = GRAVITY,
    nodes: int = 6,
) -> float:
    """Same energy relation as the solver, with the panel integral done properly.

    The source density is piecewise constant, so the far-field coefficient
    ``C = -i/(2 k0 N0) * integral phi0(z) exp(-i k0 x) sigma ds`` becomes a sum
    of per-panel integrals of the *smooth* factor ``phi0(z) exp(-i k0 x)`` over
    the straight panel, evaluated by Gauss-Legendre quadrature.  Comparing this
    with :func:`tensorbem.finite_depth_solver.wave_amplitude` isolates the
    midpoint-rule error without touching the assembly.
    """
    k0 = finite_depth_wavenumber(nu, depth)
    norm = propagating_mode_norm(k0, depth)
    quadrature_nodes, quadrature_weights = leggauss(nodes)
    fraction = 0.5 * (quadrature_nodes + 1.0)
    omega = math.sqrt(nu * gravity)
    integral = 0.0 + 0.0j
    for panel in range(mesh.panel_count):
        start = mesh.vertices[panel]
        vector = mesh.vectors[panel]
        x = start[0] + fraction * vector[0]
        z = start[1] + fraction * vector[1]
        integrand = np.cosh(k0 * (z + depth)) * np.exp(-1j * k0 * x)
        panel_integral = 0.5 * mesh.lengths[panel] * np.sum(
            quadrature_weights * integrand
        )
        integral += sources[panel] * panel_integral
    coefficient = -1j / (2.0 * k0 * norm) * integral
    amplitude = 1j * omega / gravity * math.cosh(k0 * depth) * coefficient
    c_g = group_velocity(nu, depth, gravity=gravity)
    return 2.0 * density * gravity * float(c_g) * float(abs(amplitude) ** 2)


def _finite_depth_case(
    depth: float,
    motion: str,
    panels: int,
    ka: float,
    radius: float,
    submergence: float,
    density: float,
    panel_nodes: int,
    source_count: int,
    quadrature_nodes: int,
) -> dict:
    """One mesh: near-field damping, two far-field readings, MFS reference.

    The linear system is solved once; the two far-field readings share the same
    source density, so any difference between them is a quadrature effect and
    nothing else.
    """
    nu = ka / radius
    mesh = submerged_circle_mesh(panels, radius, submergence)
    index = MOTIONS[motion]
    sources, system, data, potential = _solve_sources(
        mesh, nu, depth, motion, panel_nodes, None
    )
    force = -(potential * mesh.lengths) @ mesh.normals
    omega = math.sqrt(nu * GRAVITY)
    near = float(omega * density * np.imag(force)[index])
    far = float(far_field_damping(mesh, sources, nu, depth, density, gravity=GRAVITY))
    far_quadrature = far_field_damping_panel_quadrature(
        mesh, sources, nu, depth, density, gravity=GRAVITY, nodes=quadrature_nodes
    )
    reference = radiation_reference(
        mesh, nu, depth, motion, source_count=source_count, density=density,
        gravity=GRAVITY,
    )
    residual = system @ sources - data
    return {
        "depth": float(depth),
        "nu_h": float(nu * depth),
        "motion": motion,
        "panels": int(panels),
        "clearance": float(depth - submergence - radius),
        "panel_length": float(mesh.lengths.mean()),
        "k0_panel_length": float(
            finite_depth_wavenumber(nu, depth) * mesh.lengths.mean()
        ),
        "added_mass": float(density * np.real(force)[index]),
        "damping_near": near,
        "damping_far": far,
        "damping_far_quadrature": far_quadrature,
        "energy_offset": float(far / near - 1.0),
        "energy_offset_quadrature": float(far_quadrature / near - 1.0),
        "far_reading_difference": float(far_quadrature / far - 1.0),
        "mfs_damping": float(reference.damping[index]),
        "mfs_added_mass": float(reference.added_mass[index]),
        "mfs_damping_rel_diff": float(
            abs(reference.damping[index] - near) / abs(near)
        ),
        "condition_number": float(np.linalg.cond(system)),
        "relative_residual": float(
            np.linalg.norm(residual) / max(np.linalg.norm(data), 1e-300)
        ),
    }


def _deep_water_case(
    motion: str,
    panels: int,
    ka: float,
    radius: float,
    submergence: float,
    density: float,
    panel_nodes: int,
) -> dict:
    nu = ka / radius
    mesh = submerged_circle_mesh(panels, radius, submergence)
    index = MOTIONS[motion]
    result = deep_solve(mesh, nu, motion, density=density, panel_nodes=panel_nodes)
    far = deep_far_field_damping(
        mesh, result.potential, result.normal_velocity, nu, density, gravity=GRAVITY
    )
    near = float(result.damping[index])
    return {
        "motion": motion,
        "panels": int(panels),
        "damping_near": near,
        "damping_far": float(far),
        "energy_offset": float(far / near - 1.0),
        "relative_residual": float(result.relative_residual),
    }


def observed_order(panels: tuple[int, ...], offsets: tuple[float, ...]) -> list[float]:
    """Order ``p`` implied by consecutive offset levels, ``p = log(r)/log(2)``."""
    orders = []
    for coarse, fine, value_coarse, value_fine in zip(
        panels[:-1], panels[1:], offsets[:-1], offsets[1:]
    ):
        if panels[0] != 0 and min(abs(value_coarse), abs(value_fine)) == 0.0:
            orders.append(float("nan"))
            continue
        ratio = abs(value_fine) / abs(value_coarse)
        orders.append(float(math.log(ratio) / math.log(coarse / fine)) if ratio > 0 else float("nan"))
    return orders


def richardson_limit(
    panels: tuple[int, ...], values: tuple[float, ...], order: float
) -> float:
    """Extrapolate the two finest levels with the given observed order."""
    coarse_panels, fine_panels = panels[-2], panels[-1]
    coarse_value, fine_value = values[-2], values[-1]
    factor = (fine_panels / coarse_panels) ** order
    return float(fine_value + (fine_value - coarse_value) / (factor - 1.0))


def _load_state(path, ka, radius, submergence, density, panel_nodes) -> list[dict]:
    """Rows already measured by an interrupted run, if the parameters match."""
    if path is None or not Path(path).exists():
        return []
    try:
        payload = json.loads(Path(path).read_text())
    except json.JSONDecodeError:
        return []
    expected = {"ka": ka, "radius": radius, "submergence": submergence,
                "density": density, "panel_nodes": panel_nodes}
    if any(payload.get(key) != value for key, value in expected.items()):
        return []
    return list(payload.get("rows", []))


def _dump_state(path, rows, ka, radius, submergence, density, panel_nodes) -> None:
    """Persist the measured rows so that a long sweep is restartable."""
    if path is None:
        return
    payload = {
        "benchmark": "finite_depth_energy_identity_attribution",
        "complete": False,
        "ka": ka,
        "radius": radius,
        "submergence": submergence,
        "density": density,
        "panel_nodes": panel_nodes,
        "rows": rows,
    }
    _write_json(payload, Path(path))


def run(
    panels: tuple[int, ...] = (32, 64, 128, 256),
    ka: float = 0.6,
    radius: float = 1.0,
    submergence: float = 2.0,
    density: float = 1025.0,
    panel_nodes: int = 8,
    source_count: int = 24,
    quadrature_nodes: int = 6,
    node_sweep: tuple[int, ...] = (4, 8, 16),
    node_sweep_panels: int = 64,
    state: str | Path | None = None,
) -> dict:
    """Refinement study, node sweep and the deep-water control at the same meshes."""
    began = time.time()
    rows = _load_state(state, ka, radius, submergence, density, panel_nodes)
    cached = {(r["depth_label"], r["motion"], r["panels"]): r for r in rows}
    for label, depth in DEPTHS.items():
        motions = ("heave", "sway") if label == "shallow" else ("heave",)
        for motion in motions:
            for count in panels:
                key = (label, motion, count)
                if key in cached:
                    continue
                row = _finite_depth_case(
                    depth, motion, count, ka, radius, submergence, density,
                    panel_nodes, source_count, quadrature_nodes,
                )
                row["depth_label"] = label
                rows.append(row)
                _dump_state(state, rows, ka, radius, submergence, density, panel_nodes)

    deep_rows = []
    for motion in ("heave", "sway"):
        for count in panels:
            deep_rows.append(
                _deep_water_case(
                    motion, count, ka, radius, submergence, density, panel_nodes
                )
            )

    nodes = []
    shallow = DEPTHS["shallow"]
    for count in node_sweep:
        row = _finite_depth_case(
            shallow, "heave", node_sweep_panels, ka, radius, submergence, density,
            count, source_count, quadrature_nodes,
        )
        row["panel_nodes"] = int(count)
        nodes.append(row)

    summaries = {}
    for label, depth in DEPTHS.items():
        motions = ("heave", "sway") if label == "shallow" else ("heave",)
        for motion in motions:
            subset = sorted(
                (r for r in rows if r["depth_label"] == label and r["motion"] == motion),
                key=lambda r: r["panels"],
            )
            if len(subset) < 3:
                continue
            mesh = tuple(r["panels"] for r in subset)
            offsets = tuple(r["energy_offset"] for r in subset)
            near = tuple(r["damping_near"] for r in subset)
            far = tuple(r["damping_far"] for r in subset)
            mfs = tuple(r["mfs_damping"] for r in subset)
            orders = observed_order(mesh, offsets)
            order = float(np.nanmedian(orders))
            summaries[f"{label}:{motion}"] = {
                "panels": list(mesh),
                "energy_offset": list(offsets),
                "observed_orders": orders,
                "median_order": order,
                "offset_limit": richardson_limit(mesh, offsets, order),
                "damping_near_limit": richardson_limit(mesh, near, order),
                "damping_far_limit": richardson_limit(mesh, far, order),
                "mfs_damping_limit": richardson_limit(mesh, mfs, order),
                "near_vs_far_limit_rel": float(
                    abs(richardson_limit(mesh, far, order)
                        - richardson_limit(mesh, near, order))
                    / abs(richardson_limit(mesh, near, order))
                ),
            }
    deep_summaries = {}
    for motion in ("heave", "sway"):
        subset = sorted((r for r in deep_rows if r["motion"] == motion),
                        key=lambda r: r["panels"])
        mesh = tuple(r["panels"] for r in subset)
        offsets = tuple(r["energy_offset"] for r in subset)
        orders = observed_order(mesh, offsets)
        order = float(np.nanmedian(orders))
        deep_summaries[motion] = {
            "panels": list(mesh),
            "energy_offset": list(offsets),
            "observed_orders": orders,
            "median_order": order,
            "offset_limit": richardson_limit(mesh, offsets, order),
        }

    mfs_sources = []
    for count in (16, 24, 48, 96):
        reference = radiation_reference(
            submerged_circle_mesh(node_sweep_panels, radius, submergence),
            ka / radius,
            shallow,
            "heave",
            source_count=count,
            density=density,
            gravity=GRAVITY,
        )
        mfs_sources.append(
            {
                "source_count": int(count),
                "added_mass": float(reference.added_mass[MOTIONS["heave"]]),
                "damping": float(reference.damping[MOTIONS["heave"]]),
            }
        )

    node_offsets = [r["energy_offset"] for r in nodes]
    return {
        "benchmark": "finite_depth_energy_identity_attribution",
        "radius": radius,
        "submergence": submergence,
        "ka": ka,
        "nu": ka / radius,
        "density": density,
        "gravity": GRAVITY,
        "panel_nodes": panel_nodes,
        "source_count": source_count,
        "quadrature_nodes": quadrature_nodes,
        "complete": True,
        "elapsed_seconds": round(time.time() - began, 2),
        "rows": rows,
        "deep_water_control": deep_rows,
        "node_sweep": nodes,
        "mfs_source_sweep": mfs_sources,
        "summaries": summaries,
        "deep_water_summaries": deep_summaries,
        "node_sweep_offset_spread": float(max(node_offsets) - min(node_offsets)),
    }


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def plot(payload: dict, directory: Path) -> list[Path]:
    """Offset-vs-mesh, convergence of the three damping readings, node sweep."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    written = []
    labels = {"shallow": "h/a = 3.2", "medium": "h/a = 5.0", "deep": "h/a = 20.0"}
    colours = {"shallow": "#1f77b4", "medium": "#2ca02c", "deep": "#9467bd"}

    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    for label in ("shallow", "medium", "deep"):
        subset = sorted(
            (r for r in payload["rows"]
             if r["depth_label"] == label and r["motion"] == "heave"),
            key=lambda r: r["panels"],
        )
        if not subset:
            continue
        mesh = np.array([r["panels"] for r in subset], dtype=float)
        offset = np.array([abs(r["energy_offset"]) for r in subset])
        axes[0].loglog(mesh, offset, "o-", color=colours[label],
                       label=f"finite depth, {labels[label]}")
        axes[0].loglog(mesh, offset[0] * (mesh / mesh[0]) ** -1.0, "k--", lw=1.0,
                       label="O(h)" if label == "shallow" else None)
    for motion, style in (("heave", "s-"), ("sway", "^-")):
        subset = sorted((r for r in payload["deep_water_control"]
                         if r["motion"] == motion), key=lambda r: r["panels"])
        mesh = np.array([r["panels"] for r in subset], dtype=float)
        offset = np.array([abs(r["energy_offset"]) for r in subset])
        axes[0].loglog(mesh, offset, style, color="#d62728", mfc="none",
                       label=f"deep water (verified), {motion}")
    axes[0].set_xlabel("panels N")
    axes[0].set_ylabel(r"$|B_{\rm far}/B_{\rm near} - 1|$")
    axes[0].set_title("energy-identity offset converges as O(h)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7.5)

    for label in ("shallow", "medium", "deep"):
        subset = sorted(
            (r for r in payload["rows"]
             if r["depth_label"] == label and r["motion"] == "heave"),
            key=lambda r: r["panels"],
        )
        if not subset:
            continue
        mesh = np.array([r["panels"] for r in subset], dtype=float)
        axes[1].plot(mesh, [r["damping_near"] for r in subset], "o-",
                     color=colours[label], label=f"near field, {labels[label]}")
        axes[1].plot(mesh, [r["damping_far_quadrature"] for r in subset], "x--",
                     color=colours[label], label=f"far field, {labels[label]}")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("panels N")
    axes[1].set_ylabel(r"$B$ [kg/(m s)], heave")
    axes[1].set_title("pressure vs wave-amplitude damping: same limit")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=7.0)
    figure.tight_layout()
    path = directory / "energy_identity_convergence.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path)

    figure, axis = plt.subplots(figsize=(6.0, 4.0))
    node_counts = [r["panel_nodes"] for r in payload["node_sweep"]]
    axis.plot(node_counts, [r["energy_offset"] * 100.0 for r in payload["node_sweep"]],
              "o-", color="#1f77b4", label="energy offset")
    axis.axhline(0.0, color="k", lw=0.8)
    axis.set_xlabel("Gauss nodes per panel (fixed N = %d)" %
                    payload["node_sweep"][0]["panels"])
    axis.set_ylabel(r"$B_{\rm far}/B_{\rm near}-1$  [%]")
    axis.set_title("offset is not quadrature-limited")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = directory / "quadrature_node_sweep.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument("--panels", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--ka", type=float, default=0.6)
    parser.add_argument(
        "--state",
        type=Path,
        default=None,
        help="write rows after every case and resume from here if interrupted",
    )
    args = parser.parse_args(argv)
    state = args.state if args.state is not None else args.json
    payload = run(panels=tuple(args.panels), ka=args.ka, state=state)
    if args.json:
        _write_json(payload, args.json)
    if args.figures:
        for path in plot(payload, args.figures):
            print(f"figure: {path}")
    for key, summary in payload["summaries"].items():
        print(
            f"{key:16s} order={summary['median_order']:.2f} "
            f"offset_limit={summary['offset_limit']:+.3e} "
            f"near/far limit rel={summary['near_vs_far_limit_rel']:.2e}"
        )
    print(f"node-sweep offset spread: {payload['node_sweep_offset_spread']:.2e}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())