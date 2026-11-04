"""The 2-D limit of the 3-D Kelvin free-surface solver (cross-dimension check).

Every 3-D verification in this package so far compares the solver with a
*three-dimensional* reference: the Hulme hemisphere multipoles, the
Wang/Linton submerged sphere, or an interior-harmonic identity. Nothing then
tests whether the 3-D Kelvin Green function, its Kochin far field and its
solid-angle assembly carry the right **dimensionless scaling** -- a factor of
``pi`` or a wrong ``1/(4 pi r)`` prefactor can survive a same-dimension
comparison if the reference happens to share the mistake.

The sharpest available reference that shares no code at all is the two-
dimensional Ursell solution (:func:`tensorbem.analytic.ursell_cylinder_added_mass`,
a truncated multipole series with an exponential integral). A long submerged
capsule :math:`-L/2 \\le x \\le L/2` of radius ``a`` in deep water must satisfy

    A_3D(L) / L  ->  A_2D ,      B_3D(L) / L  ->  B_2D    as L/a -> infinity,

because the end contribution is a fixed ``O(rho a^3)`` while the wetted area
grows like ``rho a L``. The convergence is therefore ``O(a/L)``, and the
per-unit-span value plotted against ``1/L`` must extrapolate linearly onto the
2-D Ursell value. That is a one-parameter, falsifiable statement: it fixes the
3-D normalisation, the Kochin damping prefactor ``omega*rho*k/(4 pi)`` and the
radiation-force convention ``A + i B/omega = -rho * int phi n dS`` at once.

Run::

    python -m tensorbem.benchmarks.cylinder_3d_limit --ratios 4 8 12 16 \\
        --panel-sizes 0.5 0.35 --output artifacts/cylinder-3d-limit.json

The capsule geometry (:func:`tensorbem.geometry3d.capsule_mesh`) is closed form
in volume and area, so the mesh itself is audited on the way (a geometry that is
wrong by a factor cannot reproduce the limit).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.analytic import ursell_cylinder_added_mass
from tensorbem.free_surface_solver3d import (
    far_field_damping_matrix_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import capsule_mesh

__all__ = [
    "PARAMETERS",
    "REQUIRED_PANEL_SIZE",
    "panel_refinement",
    "length_ladder",
    "run_case",
    "verify",
]

#: Case shared by every row: unit radius, axis two radii down, ``ka = 0.6``.
#: ``f/a = 2`` keeps the cylinder inside Ursell's deeply-submerged regime while
#: still letting the free-surface part of the kernel matter (the wave field at
#: the body is not negligible: the damping is 6.8 % of ``rho*pi*a^2*omega``).
PARAMETERS: dict[str, float] = {
    "radius": 1.0,
    "submergence": 2.0,
    "ka": 0.6,
    "density": 1025.0,
    "gravity": 9.80665,
}

#: Edge length used for the length ladder (the resolution the refinement study
#: shows to be converged to the percent level).
REQUIRED_PANEL_SIZE = 0.35


def _reference(params: dict[str, float]) -> dict[str, float]:
    a = params["radius"]
    omega = math.sqrt(params["gravity"] * params["ka"])
    added_mass, damping = ursell_cylinder_added_mass(
        a,
        params["submergence"],
        omega,
        params["density"],
        gravity=params["gravity"],
    )
    displaced = params["density"] * math.pi * a * a
    return {
        "omega": omega,
        "added_mass": added_mass,
        "damping": damping,
        "added_mass_coefficient": added_mass / displaced,
        "damping_coefficient": damping / (displaced * omega),
    }


def run_case(
    length_ratio: float,
    panel_size: float,
    params: dict[str, float] | None = None,
    *,
    kochin: bool = True,
) -> dict[str, Any]:
    """Solve heave radiation of one capsule and report per-unit-span loads."""
    par = dict(PARAMETERS if params is None else params)
    a = par["radius"]
    length = float(length_ratio) * a
    mesh = capsule_mesh(
        length,
        a,
        panel_size=panel_size,
        submergence=par["submergence"],
    )
    exact_volume = math.pi * a * a * length + 4.0 / 3.0 * math.pi * a**3
    exact_area = 2.0 * math.pi * a * length + 4.0 * math.pi * a * a

    started = time.perf_counter()
    radiation = solve_radiation_3d(
        mesh,
        par["ka"],
        modes=[2],
        density=par["density"],
        gravity=par["gravity"],
    )
    seconds = time.perf_counter() - started

    ref = _reference(par)
    added_mass = float(radiation.added_mass[0, 0]) / length
    damping = float(radiation.damping[0, 0]) / length
    row: dict[str, Any] = {
        "length_over_radius": float(length_ratio),
        "length": length,
        "panel_size": float(panel_size),
        "panels": mesh.panel_count,
        "mesh_volume_relative_error": mesh.volume / exact_volume - 1.0,
        "mesh_area_relative_error": mesh.surface_area / exact_area - 1.0,
        "added_mass_per_length": added_mass,
        "damping_per_length": damping,
        "added_mass_coefficient": added_mass / (par["density"] * math.pi * a * a),
        "damping_coefficient": damping / (par["density"] * math.pi * a * a * ref["omega"]),
        "added_mass_relative_error": added_mass / ref["added_mass"] - 1.0,
        "damping_relative_error": damping / ref["damping"] - 1.0,
        "condition_number": radiation.condition_number,
        "relative_residual": radiation.relative_residual,
        "seconds": seconds,
    }
    if kochin:
        far = far_field_damping_matrix_3d(
            mesh, radiation, density=par["density"], gravity=par["gravity"]
        )
        near = float(radiation.damping[0, 0])
        row["kochin_to_direct_damping"] = float(far[0, 0]) / near if near else float("nan")
    return row


def panel_refinement(
    length_ratio: float,
    panel_sizes: Sequence[float],
    params: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Fix the body, refine the panels: separation of discretisation from ends."""
    rows = [run_case(length_ratio, size, params, kochin=False) for size in panel_sizes]
    for idx in range(1, len(rows)):
        h0, h1 = rows[idx - 1]["panel_size"], rows[idx]["panel_size"]
        e0 = abs(rows[idx - 1]["added_mass_relative_error"])
        e1 = abs(rows[idx]["added_mass_relative_error"])
        if e0 > 0.0 and e1 > 0.0 and h0 > h1:
            rows[idx]["observed_order"] = math.log(e0 / e1) / math.log(h0 / h1)
        else:
            rows[idx]["observed_order"] = None
    return rows


def length_ladder(
    ratios: Sequence[float],
    panel_size: float,
    params: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Extend the body at fixed panel size: the end effect must wash out as 1/L."""
    return [run_case(ratio, panel_size, params) for ratio in ratios]


def extrapolate(
    rows: Sequence[dict[str, Any]], key: str, reference: float
) -> dict[str, float]:
    """Least-squares fit of ``per-length value = limit + c/L`` over the ladder."""
    x = np.array([1.0 / row["length_over_radius"] for row in rows], dtype=float)
    y = np.array([row[key] for row in rows], dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    limit, slope = float(coefficients[0]), float(coefficients[1])
    residual = y - design @ coefficients
    denominator = float(np.sum((y - y.mean()) ** 2))
    return {
        "limit": limit,
        "slope": slope,
        "relative_error_vs_2d": limit / reference - 1.0,
        "residual_max_relative": float(np.max(np.abs(residual)) / abs(reference)),
        "r_squared": 1.0 - float(np.sum(residual**2)) / denominator if denominator else 1.0,
    }


def verify(
    ratios: Sequence[float] = (4.0, 8.0, 12.0, 16.0),
    panel_sizes: Sequence[float] = (0.5, REQUIRED_PANEL_SIZE),
    params: dict[str, float] | None = None,
    refinement_ratio: float = 8.0,
) -> dict[str, Any]:
    """Full report: reference, panel refinement, length ladder, extrapolation."""
    par = dict(PARAMETERS if params is None else params)
    ref = _reference(par)
    refinement = panel_refinement(refinement_ratio, panel_sizes, par)
    ladder = length_ladder(ratios, REQUIRED_PANEL_SIZE, par)
    fine = [row for row in ladder if row["length_over_radius"] >= min(ratios[1:], default=min(ratios))]
    extrapolation = {
        "added_mass": extrapolate(ladder, "added_mass_per_length", ref["added_mass"]),
        "damping": extrapolate(ladder, "damping_per_length", ref["damping"]),
        "added_mass_tail": extrapolate(fine, "added_mass_per_length", ref["added_mass"]),
        "damping_tail": extrapolate(fine, "damping_per_length", ref["damping"]),
    }
    monotone = all(
        abs(ladder[idx]["added_mass_relative_error"])
        < abs(ladder[idx - 1]["added_mass_relative_error"])
        for idx in range(1, len(ladder))
    )
    kochin_ok = all(abs(row["kochin_to_direct_damping"] - 1.0) < 0.02 for row in ladder)
    worst = max(abs(extrapolation["added_mass"]["relative_error_vs_2d"]),
                abs(extrapolation["damping"]["relative_error_vs_2d"]))
    verdict = {
        "added_mass_error_met": abs(extrapolation["added_mass"]["relative_error_vs_2d"]) < 0.03,
        "damping_error_met": abs(extrapolation["damping"]["relative_error_vs_2d"]) < 0.03,
        "ladder_monotone": monotone,
        "kochin_closure_within_2pct": kochin_ok,
        "worst_extrapolated_error": worst,
        "consistent_one_over_length": (
            abs(extrapolation["added_mass"]["slope"] - extrapolation["added_mass_tail"]["slope"])
            < 0.25 * abs(extrapolation["added_mass"]["slope"]) if extrapolation["added_mass"]["slope"] else False
        ),
    }
    return {
        "parameters": par,
        "reference_2d": ref,
        "panel_refinement": refinement,
        "length_ladder": ladder,
        "extrapolation": extrapolation,
        "verdict": verdict,
        "machine": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }


def make_figures(report: dict[str, Any], outdir: Path) -> list[Path]:
    """Two panels: per-span value against 1/L, and the panel refinement."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ref = report["reference_2d"]
    ladder = report["length_ladder"]
    refinement = report["panel_refinement"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=150)
    ax = axes[0]
    inv = [1.0 / row["length_over_radius"] for row in ladder]
    for key, label, ref_key, colour in (
        ("added_mass_coefficient", r"added mass  $A/(\rho\pi a^2)$", "added_mass_coefficient", "tab:blue"),
        ("damping_coefficient", r"damping  $B/(\rho\pi a^2\omega)$", "damping_coefficient", "tab:orange"),
    ):
        values = [row[key] for row in ladder]
        ax.plot(inv, values, "o", color=colour, label=f"3-D capsule, {label}")
        fit = report["extrapolation"]["added_mass" if key.startswith("added") else "damping"]
        limit_coefficient = (
            fit["limit"] / (report["parameters"]["density"] * math.pi * report["parameters"]["radius"] ** 2)
            if key.startswith("added")
            else fit["limit"]
            / (
                report["parameters"]["density"]
                * math.pi
                * report["parameters"]["radius"] ** 2
                * ref["omega"]
            )
        )
        ax.axhline(ref[ref_key], color=colour, ls="--", lw=1.0, label=f"Ursell 2-D, {label}")
        ax.plot([0.0], [limit_coefficient], "s", color=colour, mfc="none", ms=8)
    ax.set_xlabel(r"$a/L$")
    ax.set_ylabel("per unit span, normalised")
    ax.set_title(r"2-D limit: squares = extrapolation at $a/L\to0$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")

    ax = axes[1]
    h = [row["panel_size"] for row in refinement]
    for key, label, colour in (
        ("added_mass_relative_error", "added mass", "tab:blue"),
        ("damping_relative_error", "damping", "tab:orange"),
    ):
        ax.loglog(h, [abs(row[key]) for row in refinement], "o-", color=colour, label=f"|error| {label}")
    ax.set_xlabel("panel edge length  $h/a$")
    ax.set_ylabel("relative error vs Ursell 2-D")
    ax.set_title(f"panel refinement at $L/a = {refinement[0]['length_over_radius']:g}$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = outdir / "cylinder_3d_limit.png"
    fig.savefig(path)
    plt.close(fig)
    return [path]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratios", type=float, nargs="+", default=[4.0, 8.0, 12.0, 16.0])
    parser.add_argument("--panel-sizes", type=float, nargs="+", default=[0.5, REQUIRED_PANEL_SIZE])
    parser.add_argument("--refinement-ratio", type=float, default=8.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    report = verify(
        ratios=tuple(args.ratios),
        panel_sizes=tuple(args.panel_sizes),
        refinement_ratio=args.refinement_ratio,
    )
    if args.figure_dir is not None and not args.no_figures:
        report["figures"] = [str(path) for path in make_figures(report, args.figure_dir)]
    text = json.dumps(report, indent=2, sort_keys=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0 if report["verdict"]["added_mass_error_met"] and report["verdict"]["damping_error_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())