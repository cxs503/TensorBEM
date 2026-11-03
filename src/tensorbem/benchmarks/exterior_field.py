"""Verification harness for :func:`tensorbem.solver.evaluate_exterior_field`.

The 2-D exterior potential is reconstructed from the boundary traces with the
analytically integrated single- and double-layer panels and compared against
closed-form exterior harmonics of the circle,

    u_m(r, theta) = r**(-m) cos(m theta),        m = 1, 2,

which satisfy Laplace's equation, decay as ``O(r**(-m))`` and, on the circle
``r = a``, reduce to ``cos(m theta)``. Because the reference solution is a
closed-form exterior harmonic and not another discretisation, the comparison
measures the reconstruction error directly. The same identities are used for
the Neumann path (a translating cylinder, ``q_e = -cos(theta)``) and for the
far-field decay rate.

Two failure modes are quantified as well:

* far-field decay: for a dipole the product ``r * u`` must tend to ``1``;
* non-decaying boundary data: constant Dirichlet data have a non-zero boundary
  mean and admit no decaying exterior harmonic, so the exterior Dirichlet
  equation returns a pure capacity flux and the reconstructed field grows like
  ``-(Q/2) log r``. The study records the net flux, the decay ratio and the
  residual of that log law, which is what the guard in
  ``evaluate_exterior_field`` reports instead of returning a silent
  non-solution.

Run ``python -m tensorbem.benchmarks.exterior_field --out <path.json>``; pass
``--figure <path.png>`` to redraw the comparison figure. Every number in
``docs/exterior_field_verification.md`` is read from that JSON.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np

from tensorbem.geometry import regular_polygon
from tensorbem.solver import (
    evaluate_exterior_field,
    exterior_dirichlet_decay_ratio,
    exterior_net_flux,
    solve_laplace_exterior_dirichlet,
    solve_laplace_exterior_neumann,
)

PANEL_COUNTS = (32, 64, 128, 256)
ANGLES_PER_RING = 48
RADII = (1.1, 1.5, 2.0, 4.0, 8.0)
DECAY_RADII = (1.5, 2.0, 4.0, 8.0, 16.0, 32.0)


@dataclass(frozen=True)
class ModeError:
    """Error of one exterior harmonic mode on one mesh."""

    mode: int
    panels: int
    max_abs_error: float
    max_rel_error: float


@dataclass(frozen=True)
class NeumannError:
    """Error of the Neumann-driven translating-cylinder field."""

    panels: int
    radius: float
    max_abs_error: float
    rel_error: float


def _ring(radius: float, angles: np.ndarray) -> np.ndarray:
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def _probe_points(radii=RADII, angles_per_ring: int = ANGLES_PER_RING) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, angles_per_ring, endpoint=False)
    return np.vstack([_ring(r, angles) for r in radii])


def _polar(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.hypot(points[:, 0], points[:, 1]), np.arctan2(points[:, 1], points[:, 0])


def _panel_angles(mesh) -> np.ndarray:
    return np.arctan2(mesh.centers[:, 1], mesh.centers[:, 0])


def _observed_order(coarse: float, fine: float) -> float:
    if fine <= 0.0 or coarse <= 0.0:
        return float("nan")
    return float(math.log(coarse / fine) / math.log(2.0))


def dipole_study(panels=PANEL_COUNTS) -> list[ModeError]:
    """Dirichlet data ``cos(theta)`` against the exterior dipole ``cos(theta)/r``."""
    points = _probe_points()
    radius, theta = _polar(points)
    records = []
    for n in panels:
        mesh = regular_polygon(n)
        result = solve_laplace_exterior_dirichlet(mesh, np.cos(_panel_angles(mesh)))
        numerical = evaluate_exterior_field(mesh, result, points)
        exact = np.cos(theta) / radius
        error = np.abs(numerical - exact)
        records.append(
            ModeError(
                mode=1,
                panels=n,
                max_abs_error=float(np.max(error)),
                max_rel_error=float(np.max(error) / np.max(np.abs(exact))),
            )
        )
    return records


def quadrupole_study(panels=PANEL_COUNTS) -> list[ModeError]:
    """Dirichlet data ``cos(2 theta)`` against the exterior quadrupole ``cos(2 theta)/r**2``."""
    points = _probe_points()
    radius, theta = _polar(points)
    records = []
    for n in panels:
        mesh = regular_polygon(n)
        result = solve_laplace_exterior_dirichlet(mesh, np.cos(2.0 * _panel_angles(mesh)))
        numerical = evaluate_exterior_field(mesh, result, points)
        exact = np.cos(2.0 * theta) / radius**2
        error = np.abs(numerical - exact)
        records.append(
            ModeError(
                mode=2,
                panels=n,
                max_abs_error=float(np.max(error)),
                max_rel_error=float(np.max(error) / np.max(np.abs(exact))),
            )
        )
    return records


def neumann_study(panels=PANEL_COUNTS) -> list[NeumannError]:
    """Translating cylinder: ``q_e = -cos(theta)`` gives ``u = -cos(theta)/r``."""
    records = []
    angles = np.linspace(0.0, 2.0 * math.pi, ANGLES_PER_RING, endpoint=False)
    for n in panels:
        mesh = regular_polygon(n)
        flux = -np.cos(_panel_angles(mesh))
        result = solve_laplace_exterior_neumann(mesh, flux)
        for r in (1.2, 2.0, 5.0):
            points = _ring(r, angles)
            numerical = evaluate_exterior_field(mesh, result, points)
            exact = -np.cos(angles) / r
            error = np.abs(numerical - exact)
            records.append(
                NeumannError(
                    panels=n,
                    radius=r,
                    max_abs_error=float(np.max(error)),
                    rel_error=float(np.max(error) / np.max(np.abs(exact))),
                )
            )
    return records


def far_field_study(panels: int = 128) -> dict[str, float]:
    """Fit the decaying dipole tail ``u ~ C r**(-p)`` and report ``p`` and ``C``."""
    mesh = regular_polygon(panels)
    result = solve_laplace_exterior_dirichlet(mesh, np.cos(_panel_angles(mesh)))
    angles = np.linspace(0.0, 2.0 * math.pi, ANGLES_PER_RING, endpoint=False)
    radii = DECAY_RADII
    peak = []
    for r in radii:
        values = evaluate_exterior_field(mesh, result, _ring(r, angles))
        peak.append(float(np.max(np.abs(values))))
    log_r = np.log(np.asarray(radii))
    log_u = np.log(np.asarray(peak))
    slope, intercept = np.polyfit(log_r, log_u, 1)
    scaled = np.asarray(peak) * np.asarray(radii)
    return {
        "panels": float(panels),
        "fitted_decay_exponent": float(slope),
        "fitted_amplitude": float(math.exp(intercept)),
        "r_times_u_min": float(np.min(scaled)),
        "r_times_u_max": float(np.max(scaled)),
        "r_times_u_max_deviation": float(np.max(np.abs(scaled - 1.0))),
    }


def nondecaying_study(panels: int = 128) -> dict[str, float]:
    """Constant Dirichlet data: record the spurious capacity flux and its log tail."""
    mesh = regular_polygon(panels)
    result = solve_laplace_exterior_dirichlet(mesh, np.ones(mesh.panel_count))
    flux = result.normal_derivative
    net = exterior_net_flux(mesh, result)
    angles = np.linspace(0.0, 2.0 * math.pi, 4, endpoint=False)
    means = [
        float(np.mean(evaluate_exterior_field(mesh, result, _ring(r, angles), require_decaying=False)))
        for r in DECAY_RADII
    ]
    prediction = [
        means[0] - net / (2.0 * math.pi) * math.log(r / DECAY_RADII[0]) for r in DECAY_RADII
    ]
    guarded = 0.0
    try:
        evaluate_exterior_field(mesh, result, _ring(2.0, angles))
    except ValueError:
        guarded = 1.0
    return {
        "panels": float(panels),
        "boundary_mean": 1.0,
        "net_flux": net,
        "flux_uniformity_spread": float(np.std(flux)),
        "flux_to_equilibrium_ratio": float(flux.mean()),
        "decay_ratio": float(exterior_dirichlet_decay_ratio(mesh, result)),
        "log_law_max_deviation": float(np.max(np.abs(np.asarray(means) - np.asarray(prediction)))),
        "guard_raised": guarded,
        "dirichlet_condition_number": float(result.condition_number),
        "bounded_solution_is_one_max_error": float(np.max(np.abs(np.asarray(means) - 1.0))),
    }


def run_study() -> dict:
    dipole = dipole_study()
    quadrupole = quadrupole_study()
    neumann = neumann_study()
    far_field = far_field_study()
    nondecaying = nondecaying_study()
    dipole_128 = [r for r in dipole if r.panels == 128][0]
    dipole_finest = dipole[-1]
    quad_128 = [r for r in quadrupole if r.panels == 128][0]
    quad_finest = quadrupole[-1]
    finest_neumann = [r for r in neumann if r.panels == 128 and r.radius == 2.0][0]
    return {
        "case": {
            "body": "regular polygon approximating a unit circle",
            "panels": list(PANEL_COUNTS),
            "probe_radii": list(RADII),
            "angles_per_ring": ANGLES_PER_RING,
            "reference_solution": "closed-form exterior harmonics r^(-m) cos(m theta)",
            "reference_length": 1.0,
        },
        "dipole": [asdict(r) for r in dipole],
        "dipole_observed_orders": [
            _observed_order(dipole[i].max_abs_error, dipole[i + 1].max_abs_error)
            for i in range(len(dipole) - 1)
        ],
        "quadrupole": [asdict(r) for r in quadrupole],
        "quadrupole_observed_orders": [
            _observed_order(quadrupole[i].max_rel_error, quadrupole[i + 1].max_rel_error)
            for i in range(len(quadrupole) - 1)
        ],
        "neumann": [asdict(r) for r in neumann],
        "far_field": far_field,
        "nondecaying": nondecaying,
        "verdict": {
            "dipole_rel_error_128": dipole_128.max_rel_error,
            "dipole_rel_error_256": dipole_finest.max_rel_error,
            "quadrupole_rel_error_128": quad_128.max_rel_error,
            "quadrupole_rel_error_256": quad_finest.max_rel_error,
            "neumann_rel_error_128_r2": finest_neumann.rel_error,
            "dipole_below_1pct": dipole_128.max_rel_error < 1.0e-2,
            "quadrupole_below_1pct": quad_128.max_rel_error < 1.0e-2,
            "neumann_below_1pct": finest_neumann.rel_error < 1.0e-2,
            "decay_exponent_within_1pct": abs(far_field["fitted_decay_exponent"] + 1.0) < 1.0e-2,
        },
    }


def make_figure(study: dict, path: Path) -> None:
    """Draw the convergence and far-field comparison figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ax = axes[0]
    for key, label, marker in (
        ("dipole", "dipole  cos$\\theta$/$r$", "o"),
        ("quadrupole", "quadrupole  cos2$\\theta$/$r^2$", "s"),
    ):
        panels = [r["panels"] for r in study[key]]
        rel = [r["max_rel_error"] for r in study[key]]
        ax.loglog(panels, rel, marker=marker, label=label)
    first = study["dipole"][0]
    ref = [first["max_rel_error"] * (first["panels"] / len_) ** 2 for len_ in
           [r["panels"] for r in study["dipole"]]]
    ax.loglog([r["panels"] for r in study["dipole"]], ref, "k--", lw=1.0, label="$O(N^{-2})$")
    ax.set_xlabel("panels per circle")
    ax.set_ylabel("max relative error vs analytic")
    ax.set_title("Exterior field reconstruction convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    mesh_rings = DECAY_RADII
    ff = study["far_field"]
    theoretical = [1.0 / r for r in mesh_rings]
    ax.loglog(mesh_rings, theoretical, "k--", lw=1.0, label="exact peak $1/r$")
    ax.loglog(mesh_rings, [ff["r_times_u_max"] / r for r in mesh_rings], "o",
              label=f"numerical, fit exponent {ff['fitted_decay_exponent']:.4f}")
    ax.set_xlabel("probe radius $r/a$")
    ax.set_ylabel("peak $|u|$")
    ax.set_title("Far-field decay of the dipole mode")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/exterior_field/study.json"))
    parser.add_argument("--figure", type=Path, default=None)
    args = parser.parse_args(argv)
    study = run_study()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(study, indent=2) + "\n")
    if args.figure is not None:
        make_figure(study, args.figure)
    verdict = study["verdict"]
    print(json.dumps(verdict, indent=2))
    print(f"wrote {args.out}")
    if args.figure is not None:
        print(f"wrote {args.figure}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())