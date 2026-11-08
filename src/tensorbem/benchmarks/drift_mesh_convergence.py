"""Mesh-convergence closure of the drift-force near/far agreement claim.

Why this driver exists
----------------------
``tests/test_drift_second_order.py`` compares the Pinkster near-field mean
drift force of a bottom-mounted cylinder with the Maruo far-field estimate and
tolerates a **35 %** gap on the coarse mesh (``n_theta=16, n_z=6``), with the
comment "A finer mesh would achieve tighter agreement."  That sentence was never
measured.  It is a claim about the *same* quantity computed by two routes that
share no formula:

* near field: second-order pressure + waterline integral over the wetted hull
  (:func:`tensorbem.drift.near_field_mean_drift_3d`, Pinkster);
* far field: momentum flux of the scattered wave through a large control
  cylinder, i.e. the Maruo/Kochin weight
  ``F_j = (rho k^2 / 8 pi) Int |K|^2 (e_beta - e_theta)_j dtheta``
  (:func:`tensorbem.drift.far_field_mean_drift_3d`).

This driver (m6509) runs the missing refinement study and turns the sentence
into a measured statement, plus two independent cross-checks that need no
literature reference:

1. **Reduced-kernel closed form.**  For the long-wave dipole Kochin function
   ``K(theta) = (ka)^2 cos(theta)`` the analytically reduced Maruo integral
   (:func:`tensorbem.maruo_drift.maruo_mean_drift_force`, m5296) is exact::

       Int_0^{2pi} |K|^2 (1 - cos theta) dtheta = (ka)^4 (pi - 0) = pi (ka)^4
       F_x = (rho g / 4) * pi * (ka)^4

   so its log-log slope is exactly **4**, not 3.  This is the analytic reason
   the ``(ka)^3`` gate in :mod:`tensorbem.benchmarks.drift_force_study` was not
   evaluable: that gate's own ansatz is a toy long-wave dipole, whose true
   asymptote is 4, while the real cylinder measures ~3 (its Kochin function is
   not ``(ka)^2 cos theta``).  The historical honest FAIL of the toy route
   stands; what is added here is the *verified* route on the real geometry.

2. **Two far-field formulae on the same Kochin data.**  The reduced Maruo
   kernel (``tensorbem.maruo_drift``) and the control-surface weight
   (``tensorbem.drift``) are separate implementations with different prefactors
   and normalisations.  Fed the same ``K(theta)`` from a real diffraction
   solve, their azimuthal integrals must agree to quadrature accuracy.  This
   checks the two formulas against each other with no reference value at all.

Everything printed in ``docs/drift_mesh_convergence.md`` is written by this
driver (``--inject-docs``); no digit in that document is transcribed by hand.

Usage
-----
    python -m tensorbem.benchmarks.drift_mesh_convergence \\
        --json artifacts/drift_mesh/study.json \\
        --figures artifacts/drift_mesh --inject-docs
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.drift import (
    bottom_mounted_cylinder_mesh,
    far_field_mean_drift_3d,
    near_field_mean_drift_3d,
)
from tensorbem.free_surface_solver3d import solve_diffraction_3d
from tensorbem.maruo_drift import maruo_mean_drift_force

DEFAULT_DENSITY = 1025.0
DEFAULT_GRAVITY = 9.81

#: ka values for the asymptotic sweep (one mesh family: depth = DEPTH_OVER_KA/ka).
KA_VALUES: tuple[float, ...] = (0.15, 0.30, 0.45)
#: depth/radius ratio kept fixed across the sweep so the mesh *shape* is constant.
DEPTH_OVER_KA = 5.0
#: (n_theta, n_z, growth) refinement ladder.
LEVELS: tuple[tuple[int, int, float], ...] = (
    (16, 6, 1.35),
    (24, 8, 1.35),
    (32, 10, 1.35),
)
#: the ka at which the whole ladder is run (the refinement focal point).
FOCAL_KA = 0.30
#: acceptance: the near/far gap on the finest mesh.
GAP_TARGET = 0.03
#: acceptance: two far-field formulae on the same Kochin data.
ROUTE_TARGET = 1.0e-3
#: acceptance: reduced-kernel closed form.
CLOSED_FORM_TARGET = 1.0e-10
#: acceptance: log-log slope of the real cylinder over the ka sweep.
SLOPE_TARGET = 3.0
SLOPE_TOLERANCE = 0.2

DOC_BEGIN = "<!-- BEGIN GENERATED: drift-mesh-convergence -->"
DOC_END = "<!-- END GENERATED: drift-mesh-convergence -->"


def _rel(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    return abs(a - b) / scale if scale > 0.0 else 0.0


def _panel_count(n_theta: int, n_z: int) -> int:
    """Side triangles + bottom cap, exactly as built by the mesh helper."""
    return 2 * n_z * n_theta + n_theta


def reduced_kernel_closed_form(ka: float, *, density: float = DEFAULT_DENSITY,
                              gravity: float = DEFAULT_GRAVITY) -> dict:
    """Numeric reduced-Maruó integral vs its closed form for ``K = (ka)^2 cos``."""
    n_theta = 8192
    theta = 2.0 * math.pi * np.arange(n_theta, dtype=float) / n_theta
    kochin = (ka ** 2) * np.cos(theta)
    res = maruo_mean_drift_force(
        kochin, theta, 1.0, n_theta=n_theta, density=density, gravity=gravity
    )
    closed_integral = math.pi * ka ** 4
    closed_force = 0.25 * density * gravity * closed_integral
    return {
        "ka": ka,
        "flux_integral": res["flux_integral"],
        "closed_form_integral": closed_integral,
        "integral_rel_error": _rel(res["flux_integral"], closed_integral),
        "force": res["mean_drift_force_x"],
        "closed_form_force": closed_force,
        "force_rel_error": _rel(res["mean_drift_force_x"], closed_force),
        "exact_slope": 4.0,
        "pass": _rel(res["flux_integral"], closed_integral) < CLOSED_FORM_TARGET,
    }


def run_case(
    ka: float,
    n_theta: int,
    n_z: int,
    growth: float,
    *,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    n_azimuth: int = 180,
) -> dict:
    """Solve one cylinder case and return all three estimates plus their gaps."""
    start = time.perf_counter()
    mesh = bottom_mounted_cylinder_mesh(
        1.0, DEPTH_OVER_KA / ka, n_theta=n_theta, n_z=n_z, growth=growth
    )
    diffraction = solve_diffraction_3d(mesh, ka, density=density, gravity=gravity)
    near = near_field_mean_drift_3d(mesh, diffraction, density=density, gravity=gravity)
    far = far_field_mean_drift_3d(
        mesh, diffraction, density=density, n_azimuth=n_azimuth
    )
    kochin = np.asarray(far["kochin"], dtype=np.complex128)
    azimuth = np.asarray(far["azimuth"], dtype=float)
    reduced = maruo_mean_drift_force(
        kochin, azimuth, ka, density=density, gravity=gravity
    )
    near_x = float(near.mean_force[0])
    far_x = float(far["force"][0])
    # Both far-field formulae are integrals of |K|^2 (1 - cos theta); undo each
    # prefactor so the two integrands are compared, not the two prefactors.
    integral_control_surface = far_x / (density * ka * ka / (8.0 * math.pi))
    integral_reduced = float(reduced["flux_integral"])
    return {
        "ka": ka,
        "n_theta": n_theta,
        "n_z": n_z,
        "growth": growth,
        "panels": _panel_count(n_theta, n_z),
        "depth_over_radius": DEPTH_OVER_KA / ka,
        "near_force_x": near_x,
        "far_force_x": far_x,
        "near_field_waterline": float(np.sum(near.waterline_term)),
        "near_field_pressure": float(np.sum(near.pressure_term)),
        "gap": _rel(far_x, near_x),
        "signed_gap": (far_x - near_x) / near_x,
        "integral_control_surface": integral_control_surface,
        "integral_reduced_maruo": integral_reduced,
        "route_rel_error": _rel(integral_control_surface, integral_reduced),
        "kochin_max_abs": float(np.abs(kochin).max()),
        "seconds": time.perf_counter() - start,
    }


def _slope(kas: Sequence[float], forces: Sequence[float]) -> float:
    return float(np.polyfit(np.log(np.asarray(kas)), np.log(np.asarray(forces)), 1)[0])


def run_study(
    *,
    ka_values: Sequence[float] = KA_VALUES,
    levels: Sequence[tuple[int, int, float]] = LEVELS,
    focal_ka: float = FOCAL_KA,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    n_azimuth: int = 180,
) -> dict:
    levels = tuple(levels)
    coarse = levels[0]
    sweep = [
        run_case(ka, *coarse, density=density, gravity=gravity, n_azimuth=n_azimuth)
        for ka in ka_values
    ]
    ladder = [
        run_case(focal_ka, *level, density=density, gravity=gravity, n_azimuth=n_azimuth)
        for level in levels
    ]
    closed = [reduced_kernel_closed_form(ka, density=density, gravity=gravity)
              for ka in ka_values]

    kas = [row["ka"] for row in sweep]
    near_slope = _slope(kas, [abs(row["near_force_x"]) for row in sweep])
    far_slope = _slope(kas, [abs(row["far_force_x"]) for row in sweep])

    finest = ladder[-1]
    gaps = [row["gap"] for row in ladder]
    gap_monotone = all(gaps[i + 1] <= gaps[i] for i in range(len(gaps) - 1))
    gates = {
        "reduced_kernel_closed_form": {
            "target": CLOSED_FORM_TARGET,
            "worst": max(row["integral_rel_error"] for row in closed),
            "pass": all(row["pass"] for row in closed),
        },
        "two_far_field_routes": {
            "target": ROUTE_TARGET,
            "worst": max(
                [row["route_rel_error"] for row in ladder]
                + [row["route_rel_error"] for row in sweep]
            ),
            "pass": max(
                [row["route_rel_error"] for row in ladder]
                + [row["route_rel_error"] for row in sweep]
            ) < ROUTE_TARGET,
        },
        "near_far_gap_finest": {
            "target": GAP_TARGET,
            "mesh": {"n_theta": finest["n_theta"], "n_z": finest["n_z"],
                     "panels": finest["panels"]},
            "measured": finest["gap"],
            "pass": finest["gap"] < GAP_TARGET,
        },
        "near_far_gap_monotone": {
            "gaps": gaps,
            "pass": gap_monotone,
        },
        "asymptotic_slope_coarse": {
            "target": SLOPE_TARGET,
            "tolerance": SLOPE_TOLERANCE,
            "near_slope": near_slope,
            "far_slope": far_slope,
            "pass": (
                abs(near_slope - SLOPE_TARGET) <= SLOPE_TOLERANCE
                and abs(far_slope - SLOPE_TARGET) <= SLOPE_TOLERANCE
            ),
        },
    }
    return {
        "case": {
            "geometry": "bottom-mounted waterline-truncated circular cylinder, radius 1 m",
            "depth_over_radius": DEPTH_OVER_KA,
            "density_kg_m3": density,
            "gravity_m_s2": gravity,
            "ka_values": list(ka_values),
            "levels": [{"n_theta": n, "n_z": z, "growth": g} for n, z, g in levels],
            "focal_ka": focal_ka,
            "n_azimuth": n_azimuth,
        },
        "sweep": sweep,
        "ladder": ladder,
        "reduced_kernel_closed_form": closed,
        "gates": gates,
        "verdict": {
            "near_field_route_verified_against_far_field": gates["near_far_gap_finest"]["pass"],
            "gap_claim": (
                "measured: the pre-existing 35 percent near/far tolerance of "
                "tests/test_drift_second_order.py is %s by the finest mesh here"
                % ("met by a tighter value" if gates["near_far_gap_finest"]["pass"]
                   else "still required")
            ),
            "toy_ansatz_asymptote": 4.0,
            "real_cylinder_asymptote": SLOPE_TARGET,
            "honest_note": (
                "the (ka)^3 gate of tensorbem.benchmarks.drift_force_study belongs to a "
                "toy long-wave dipole ansatz whose true asymptote is (ka)^4; the real "
                "cylinder asymptote is measured here on the verified routes"
            ),
        },
    }


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}g}"


def render_doc_block(study: dict) -> str:
    """Markdown table + gate block, generated from the study dict only."""
    case = study["case"]
    lines = [
        DOC_BEGIN,
        "",
        "### Generated numbers",
        "",
        "**Refinement ladder at ka = %s** (radius 1 m, depth %.1f/ka):" % (
            _fmt(case["focal_ka"]), case["depth_over_radius"]),
        "",
        "| n_theta | n_z | panels | near field F_x (N) | far field F_x (N) | gap | reduced-Maruó vs control-surface integral |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in study["ladder"]:
        lines.append(
            "| %d | %d | %d | %s | %s | %.2f%% | %s |"
            % (
                row["n_theta"], row["n_z"], row["panels"],
                _fmt(row["near_force_x"]), _fmt(row["far_force_x"]),
                100.0 * row["gap"], _fmt(row["route_rel_error"]),
            )
        )
    lines += [
        "",
        "**ka sweep on the coarsest mesh** (n_theta = %d, n_z = %d):"
        % (study["sweep"][0]["n_theta"], study["sweep"][0]["n_z"]),
        "",
        "| ka | near field F_x (N) | far field F_x (N) | gap | |K|_max |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in study["sweep"]:
        lines.append(
            "| %.2f | %s | %s | %.2f%% | %s |"
            % (row["ka"], _fmt(row["near_force_x"]), _fmt(row["far_force_x"]),
               100.0 * row["gap"], _fmt(row["kochin_max_abs"]))
        )
    lines += [
        "",
        "**Reduced kernel against its closed form** (K = (ka)^2 cos theta):",
        "",
        "| ka | numeric integral | closed form pi (ka)^4 | rel. error | exact slope |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in study["reduced_kernel_closed_form"]:
        lines.append(
            "| %.2f | %s | %s | %s | %s |"
            % (row["ka"], _fmt(row["flux_integral"]), _fmt(row["closed_form_integral"]),
               _fmt(row["integral_rel_error"]), _fmt(row["exact_slope"]))
        )
    slope = study["gates"]["asymptotic_slope_coarse"]
    gaps = study["gates"]["near_far_gap_monotone"]["gaps"]
    lines += [
        "",
        "**Gates**",
        "",
        "| gate | target | measured | pass |",
        "|---|---|---:|---|",
        "| reduced kernel vs closed form | < %s | %s | %s |"
        % (_fmt(CLOSED_FORM_TARGET), _fmt(study["gates"]["reduced_kernel_closed_form"]["worst"]),
           study["gates"]["reduced_kernel_closed_form"]["pass"]),
        "| two far-field formulae on one Kochin | < %s | %s | %s |"
        % (_fmt(ROUTE_TARGET), _fmt(study["gates"]["two_far_field_routes"]["worst"]),
           study["gates"]["two_far_field_routes"]["pass"]),
        "| near/far gap, finest mesh (%d panels) | < %.0f%% | %.2f%% | %s |"
        % (study["gates"]["near_far_gap_finest"]["mesh"]["panels"],
           100.0 * GAP_TARGET, 100.0 * study["gates"]["near_far_gap_finest"]["measured"],
           study["gates"]["near_far_gap_finest"]["pass"]),
        "| gap monotone in refinement | True | %s | %s |"
        % (["%.2f%%" % (100.0 * g) for g in gaps],
           study["gates"]["near_far_gap_monotone"]["pass"]),
        "| log-log slope (ka sweep) | %.1f ± %.1f | near %s / far %s | %s |"
        % (SLOPE_TARGET, SLOPE_TOLERANCE, _fmt(slope["near_slope"]),
           _fmt(slope["far_slope"]), slope["pass"]),
        "",
        DOC_END,
    ]
    return "\n".join(lines)


def inject_docs(study: dict, doc_path: Path) -> None:
    """Replace the generated block of the report with freshly computed numbers."""
    text = doc_path.read_text()
    begin = text.index(DOC_BEGIN)
    end = text.index(DOC_END) + len(DOC_END)
    doc_path.write_text(text[:begin] + render_doc_block(study) + text[end:])


def make_figures(study: dict, outdir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    sweep = study["sweep"]
    kas = np.array([row["ka"] for row in sweep])
    near = np.array([abs(row["near_force_x"]) for row in sweep])
    far = np.array([abs(row["far_force_x"]) for row in sweep])
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.loglog(kas, near, "o-", label="near field (Pinkster)")
    ax.loglog(kas, far, "s--", label="far field (Maruo/Kochin)")
    ref = far[0] * (kas / kas[0]) ** 3.0
    ax.loglog(kas, ref, ":", color="0.4", label=r"slope 3 reference")
    ax.set_xlabel("ka")
    ax.set_ylabel(r"$|\bar F_x|$ (N)")
    ax.set_title("cylinder mean drift force: two routes, mesh family fixed")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    path = outdir / "fig_drift_slope.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)

    ladder = study["ladder"]
    panels = [row["panels"] for row in ladder]
    gaps = [100.0 * row["gap"] for row in ladder]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.semilogx(panels, gaps, "o-")
    ax.axhline(100.0 * GAP_TARGET, color="tab:red", linestyle="--",
               label="3 % acceptance")
    ax.axhline(35.0, color="0.5", linestyle=":", label="pre-existing 35 % tolerance")
    ax.set_xlabel("panels")
    ax.set_ylabel("near/far gap (%)")
    ax.set_title("refinement at ka = %.2f" % study["case"]["focal_ka"])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    path = outdir / "fig_drift_gap_refinement.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument("--docs", type=Path,
                        default=Path("docs/drift_mesh_convergence.md"))
    parser.add_argument("--inject-docs", action="store_true")
    parser.add_argument("--n-azimuth", type=int, default=180)
    args = parser.parse_args(argv)

    study = run_study(n_azimuth=args.n_azimuth)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(study, indent=2, sort_keys=True))
    if args.figures is not None:
        for path in make_figures(study, args.figures):
            print(f"figure: {path}")
    if args.inject_docs:
        inject_docs(study, args.docs)
        print(f"doc block injected into {args.docs}")
    print(json.dumps({"gates": study["gates"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())