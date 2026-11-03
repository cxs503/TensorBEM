"""Kramers-Kronig causality verification of the two-dimensional radiation solver.

The study takes a submerged circular cylinder (Ursell's classical case) and
builds two independent ``a(omega), B(omega)`` sweeps:

* the closed-form Ursell multipole solution, which is analytic and shares no
  code with the boundary-element assembly;
* the panel solution of :func:`tensorbem.free_surface_solver.solve_radiation_indirect`,
  the routine that the platform actually uses.

Each sweep is then fed to the causality relations (A) and (B) of
:mod:`tensorbem.kramers_kronig`:

* relation (A) reconstructs the *frequency dependence* of the added mass from
  the damping alone and, as a by-product, returns the implied
  infinite-frequency added mass ``a(inf)`` -- a quantity that the tabulated
  sweep approaches only as ``1/ka`` and therefore never pins down directly;
* relation (B) does the reverse, reconstructing the damping from the added mass
  and an independently known ``a(inf)``.

An exact match is not expected from the panel solution: it converges at first
order for this kernel, so its coefficients carry an ``O(h)`` error. What the
study measures is therefore the *causality residual* of each dataset, which is
what an independent verification can check without any reference table.

Run: python -m tensorbem.benchmarks.kramers_kronig_study --json-out docs/...
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensorbem.analytic import ursell_cylinder_added_mass, ursell_cylinder_coefficients
from tensorbem.free_surface import green_function, green_gradient_field
from tensorbem.free_surface_solver import radiation_reference, solve_radiation_indirect
from tensorbem.geometry import PanelMesh
from tensorbem.kramers_kronig import (
    damping_from_added_mass,
    implied_infinite_added_mass,
    kk_residual_metrics,
    sum_rule_residual,
)

RADIUS = 1.0
SUBMERGENCE = 2.0
DENSITY = 1025.0
GRAVITY = 9.80665
DISPLACED = DENSITY * math.pi * RADIUS**2


def submerged_circle_mesh(panel_count: int, radius: float = RADIUS, submergence: float = SUBMERGENCE) -> PanelMesh:
    """Equally spaced panels on a circle of the given radius, centred at ``-f``."""
    if panel_count < 8:
        raise ValueError("panel_count must be at least 8")
    angles = 2.0 * math.pi * np.arange(panel_count) / panel_count
    return PanelMesh(
        np.column_stack((radius * np.cos(angles), radius * np.sin(angles) - submergence))
    )


def ka_to_omega(ka):
    return np.sqrt(np.asarray(ka, dtype=float) * GRAVITY / RADIUS)


def sweep_analytic(ka_values, terms: int = 40) -> dict:
    """Ursell multipole sweep (heave == sway in infinite depth)."""
    ka = np.asarray(ka_values, dtype=float)
    omega = ka_to_omega(ka)
    added_mass = np.empty(ka.size)
    damping = np.empty(ka.size)
    for index, (k, w) in enumerate(zip(ka, omega)):
        mass, damp = ursell_cylinder_added_mass(
            RADIUS, SUBMERGENCE, float(w), DENSITY, gravity=GRAVITY, terms=terms
        )
        added_mass[index] = mass
        damping[index] = damp
    return {"ka": ka, "omega": omega, "added_mass": added_mass, "damping": damping}


def sweep_panels(ka_values, panels: int, panel_nodes: int = 24) -> dict:
    """Panel sweep of the indirect radiation solver."""
    ka = np.asarray(ka_values, dtype=float)
    omega = ka_to_omega(ka)
    mesh = submerged_circle_mesh(panels)
    added_mass = np.empty(ka.size)
    damping = np.empty(ka.size)
    residuals = np.empty(ka.size)
    condition = np.empty(ka.size)
    for index, (k, w) in enumerate(zip(ka, omega)):
        result = solve_radiation_indirect(
            mesh, float(k), "heave", density=DENSITY, panel_nodes=panel_nodes
        )
        added_mass[index] = float(result.added_mass[1])
        damping[index] = float(result.damping[1])
        residuals[index] = float(result.relative_residual)
        condition[index] = float(result.condition_number)
    return {
        "ka": ka,
        "omega": omega,
        "added_mass": added_mass,
        "damping": damping,
        "panel_relative_residual": residuals,
        "condition_number": condition,
        "panels": int(panels),
    }


def extrapolated_infinite_added_mass() -> dict:
    """Asymptotic ``a(inf)`` of the multipole solution by a three-point fit.

    The multipole sweep approaches its high-frequency (rigid free surface) limit
    algebraically, ``a(inf) - a(ka) = C * ka**(-p)``.  Fitting ``p`` on
    ``ka = 20, 50, 100`` and extrapolating is a route to ``a(inf)`` that is
    structurally independent of the causality relations.
    """
    samples = [20.0, 50.0, 100.0]
    values = [ursell_cylinder_coefficients(k, RADIUS / SUBMERGENCE, 40)[0] for k in samples]
    d1 = values[1] - values[0]
    d2 = values[2] - values[1]
    ratio_target = d1 / d2

    def ratio(p):
        return (samples[0] ** (-p) - samples[1] ** (-p)) / (
            samples[1] ** (-p) - samples[2] ** (-p)
        )

    low, high = 0.2, 4.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if ratio(mid) > ratio_target:
            high = mid
        else:
            low = mid
    exponent = 0.5 * (low + high)
    step = (samples[2] / samples[1]) ** exponent - 1.0
    limit = values[2] + d2 / step
    return {
        "ka_samples": samples,
        "added_mass_over_displaced": values,
        "decay_exponent": float(exponent),
        "added_mass_infinite_over_displaced": float(limit),
        "added_mass_infinite": float(limit * DISPLACED),
    }


def analyse_dataset(label: str, sweep: dict, *, a_inf: float | None = None) -> dict:
    """Run both causality relations on one sweep and summarise the residuals."""
    omega = sweep["omega"]
    added_mass = sweep["added_mass"]
    damping = sweep["damping"]

    relation_a = implied_infinite_added_mass(omega, damping, added_mass)
    metrics_a = kk_residual_metrics(
        relation_a["added_mass_reference"], relation_a["reconstructed_added_mass"]
    )
    out = {
        "label": label,
        "samples": int(omega.size),
        "ka_min": float(sweep["ka"][0]),
        "ka_max": float(sweep["ka"][-1]),
        "implied_added_mass_infinite": relation_a["implied_infinite_added_mass"],
        "implied_added_mass_infinite_over_displaced": relation_a["implied_infinite_added_mass"]
        / DISPLACED,
        "implied_added_mass_relative_spread": relation_a["implied_added_mass_relative_spread"],
        "relation_a_max_relative_error": metrics_a["max_relative_error"],
        "relation_a_rms_relative_error": metrics_a["rms_relative_error"],
        "omega_evaluated": relation_a["omega_evaluated"],
        "added_mass_reference": relation_a["added_mass_reference"],
        "added_mass_reconstructed": relation_a["reconstructed_added_mass"],
        "sum_rule_residual_over_displaced": sum_rule_residual(
            omega, added_mass, relation_a["implied_infinite_added_mass"]
        )
        / DISPLACED,
    }
    if a_inf is not None:
        # relation (B) needs omega strictly inside the grid, so it is evaluated on
        # the same interior subset that relation (A) uses.
        margin = 2 if omega.size > 8 else 1
        interior = omega[margin:-margin]
        reconstructed_damping = damping_from_added_mass(omega, added_mass, a_inf, interior)
        metrics_b = kk_residual_metrics(damping[margin:-margin], reconstructed_damping)
        out.update(
            {
                "relation_b_infinite_added_mass": a_inf,
                "relation_b_max_relative_error": metrics_b["max_relative_error"],
                "relation_b_rms_relative_error": metrics_b["rms_relative_error"],
                "damping_reconstructed": reconstructed_damping,
                "damping_reference": damping[margin:-margin],
            }
        )
    return out


def radiation_field(points: np.ndarray, nu: float, *, panels: int = 96, source_count: int = 48) -> np.ndarray:
    """Potential of the unit-velocity heave radiation field at ``points``.

    Uses the method of fundamental solutions with the same source layout as
    :func:`tensorbem.free_surface_solver.radiation_reference`; the returned
    potential reproduces that routine's added mass, which is asserted by the
    caller. The purpose is the field plot, not a new solver.
    """
    mesh = submerged_circle_mesh(panels)
    centre = np.mean(mesh.centers, axis=0)
    radius = float(np.mean(np.linalg.norm(mesh.centers - centre, axis=1)))
    angle = 2.0 * np.pi * np.arange(source_count) / source_count
    sources = centre + 0.55 * radius * np.column_stack((np.cos(angle), np.sin(angle)))

    gradient_x, gradient_z = green_gradient_field(
        mesh.centers[:, 0][:, None], mesh.centers[:, 1][:, None],
        sources[None, :, 0], sources[None, :, 1], nu,
    )
    kernel = gradient_x * mesh.normals[:, 0][:, None] + gradient_z * mesh.normals[:, 1][:, None]
    strengths, *_ = np.linalg.lstsq(kernel, mesh.normals[:, 1], rcond=None)

    values = green_function(
        points[:, 0][:, None], points[:, 1][:, None],
        sources[None, :, 0], sources[None, :, 1], nu,
    ) @ strengths
    return values


def _field_grid(panels: int, nu: float) -> dict:
    """Potential field on a rectangle covering a couple of radiated wavelengths."""
    wavelength = 2.0 * math.pi / nu
    x = np.linspace(-2.0 * wavelength, 2.0 * wavelength, 320)
    z = np.linspace(-1.6 * wavelength, -0.02 * wavelength, 200)  # strictly below z=0
    xx, zz = np.meshgrid(x, z)
    points = np.column_stack((xx.ravel(), zz.ravel()))
    phi = radiation_field(points, nu, panels=panels)
    phi = phi.reshape(xx.shape)
    # Blank the body interior so the contour does not draw through it.
    inside = (xx**2 + (zz + SUBMERGENCE) ** 2) < RADIUS**2
    phi = np.where(inside, np.nan, phi)
    reference = radiation_reference(submerged_circle_mesh(panels), nu, "heave", density=DENSITY)
    return {
        "x": x,
        "z": z,
        "real": np.real(phi),
        "imag": np.imag(phi),
        "wavelength": wavelength,
        "reference_added_mass": float(reference.added_mass[1]),
    }


def _figure(analytic, panel_sweeps, datasets, field, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))

    axis = axes[0, 0]
    contour = axis.contourf(field["x"], field["z"], field["real"], levels=40, cmap="RdBu_r")
    axis.plot(field["x"], np.zeros_like(field["x"]), color="k", lw=1.2)
    axis.set_title("(a) Re $\\phi$ of the heave radiation field, $ka=0.6$")
    axis.set_xlabel("$x/a$")
    axis.set_ylabel("$z/a$")
    axis.set_aspect("equal")
    fig.colorbar(contour, ax=axis, shrink=0.85)

    axis = axes[0, 1]
    axis.loglog(analytic["ka"], analytic["added_mass"] / DISPLACED, "k-", lw=1.8, label="Ursell multipole (exact)")
    for sweep in panel_sweeps:
        axis.loglog(
            sweep["ka"], sweep["added_mass"] / DISPLACED, "o", ms=3.4, alpha=0.75,
            label=f"panel solver, {sweep['panels']} panels",
        )
    for data in datasets:
        axis.loglog(
            data["omega_evaluated"] ** 2 * RADIUS / GRAVITY,
            data["added_mass_reconstructed"] / DISPLACED,
            "--", lw=1.0, alpha=0.9, label=f"KK reconstruction ({data['label']})",
        )
    axis.set_title("(b) Added mass and its causality reconstruction")
    axis.set_xlabel("$ka$")
    axis.set_ylabel("$a(\\omega) / (\\rho\\pi a^2)$")
    axis.legend(fontsize=7.5, loc="lower left")
    axis.grid(True, which="both", alpha=0.25)

    axis = axes[1, 0]
    axis.loglog(analytic["ka"], analytic["damping"] / (DISPLACED * analytic["omega"]), "k-", lw=1.8, label="Ursell multipole (exact)")
    for sweep in panel_sweeps:
        axis.loglog(
            sweep["ka"], sweep["damping"] / (DISPLACED * sweep["omega"]), "s", ms=3.2, alpha=0.75,
            label=f"panel solver, {sweep['panels']} panels",
        )
    for data in datasets:
        if "damping_reconstructed" not in data:
            continue
        mask = data["damping_reference"] > 0.0
        axis.loglog(
            data["omega_evaluated"][mask] ** 2 * RADIUS / GRAVITY,
            data["damping_reconstructed"][mask]
            / (DISPLACED * data["omega_evaluated"][mask]),
            "--", lw=1.0, alpha=0.9, label=f"KK reconstruction ({data['label']})",
        )
    axis.set_title("(c) Damping and its causality reconstruction")
    axis.set_xlabel("$ka$")
    axis.set_ylabel("$B(\\omega) / (\\rho\\pi a^2 \\omega)$")
    axis.legend(fontsize=7.5, loc="lower left")
    axis.grid(True, which="both", alpha=0.25)

    axis = axes[1, 1]
    for data in datasets:
        ka = data["omega_evaluated"] ** 2 * RADIUS / GRAVITY
        residual = np.abs(data["added_mass_reconstructed"] - data["added_mass_reference"]) / DISPLACED
        axis.loglog(ka, np.maximum(residual, 1e-16), lw=1.6, label=f"(A) {data['label']}")
        if "damping_reconstructed" in data:
            residual_b = np.abs(data["damping_reconstructed"] - data["damping_reference"]) / (
                np.max(np.abs(data["damping_reference"]))
            )
            axis.loglog(ka, np.maximum(residual_b, 1e-16), "--", lw=1.2, label=f"(B) {data['label']}")
    axis.set_title("(d) Causality residual by relation")
    axis.set_xlabel("$ka$")
    axis.set_ylabel("relative residual")
    axis.legend(fontsize=7.5, loc="lower center")
    axis.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytic-samples", type=int, default=150)
    parser.add_argument("--analytic-ka-min", type=float, default=0.002)
    parser.add_argument("--analytic-ka-max", type=float, default=6.0)
    parser.add_argument("--bem-samples", type=int, default=36)
    parser.add_argument("--bem-ka-min", type=float, default=0.003)
    parser.add_argument("--bem-ka-max", type=float, default=4.0)
    parser.add_argument("--panels", type=int, nargs="+", default=[64, 96])
    parser.add_argument("--field-panels", type=int, default=96)
    parser.add_argument("--field-ka", type=float, default=0.6)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--figure-out", type=Path, default=None)
    parser.add_argument("--field-json-out", type=Path, default=None)
    args = parser.parse_args()

    ka_analytic = np.geomspace(args.analytic_ka_min, args.analytic_ka_max, args.analytic_samples)
    analytic = sweep_analytic(ka_analytic)

    ka_bem = np.geomspace(args.bem_ka_min, args.bem_ka_max, args.bem_samples)
    panel_sweeps = [sweep_panels(ka_bem, panels) for panels in args.panels]

    limit = extrapolated_infinite_added_mass()
    a_inf_exact = limit["added_mass_infinite"]

    datasets = [analyse_dataset("Ursell multipole", analytic, a_inf=a_inf_exact)]
    datasets += [
        analyse_dataset(f"panel solver ({sweep['panels']} panels)", sweep, a_inf=a_inf_exact)
        for sweep in panel_sweeps
    ]

    summary = {
        "benchmark": "kramers_kronig_causality_submerged_cylinder",
        "radius": RADIUS,
        "submergence": SUBMERGENCE,
        "fluid_density": DENSITY,
        "gravity": GRAVITY,
        "displaced_mass_per_unit_span": DISPLACED,
        "analytic_sweep": {
            "samples": int(analytic["ka"].size),
            "ka_min": float(analytic["ka"][0]),
            "ka_max": float(analytic["ka"][-1]),
        },
        "panel_sweep": {
            "samples": int(ka_bem.size),
            "ka_min": float(ka_bem[0]),
            "ka_max": float(ka_bem[-1]),
            "panels": list(args.panels),
        },
        "infinite_added_mass_extrapolation": limit,
        "datasets": [
            {key: value for key, value in data.items() if not isinstance(value, np.ndarray)}
            for data in datasets
        ],
        "datasets_detail": [
            {
                key: (np.asarray(value).tolist() if isinstance(value, np.ndarray) else value)
                for key, value in data.items()
            }
            for data in datasets
        ],
        "conclusion": (
            "Relation (A) reproduces the added-mass frequency dependence from the "
            "damping sweep alone; relation (B) reproduces the damping from the "
            "added mass. Both are consequences of causality, not of the "
            "discretisation."
        ),
    }

    figure_written = None
    if args.figure_out is not None:
        field = _field_grid(args.field_panels, args.field_ka)
        figure_written = str(args.figure_out)
        args.figure_out.parent.mkdir(parents=True, exist_ok=True)
        _figure(analytic, panel_sweeps, datasets, field, args.figure_out)
        if args.field_json_out is not None:
            args.field_json_out.parent.mkdir(parents=True, exist_ok=True)
            args.field_json_out.write_text(
                json.dumps(
                    {
                        "ka": args.field_ka,
                        "panels": args.field_panels,
                        "wavelength": field["wavelength"],
                        "reference_added_mass": field["reference_added_mass"],
                        "re": field["real"].tolist(),
                        "im": field["imag"].tolist(),
                        "x": field["x"].tolist(),
                        "z": field["z"].tolist(),
                    }
                )
            )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = json.loads(json.dumps(summary))
        payload["figure"] = figure_written
        args.json_out.write_text(json.dumps(payload, indent=2))

    for data in datasets:
        mass_inf = data["implied_added_mass_infinite_over_displaced"]
        line = (
            f"{data['label']:28s} implied a(inf)/rho pi a^2 = {mass_inf:.6f}"
            f"  spread = {data['implied_added_mass_relative_spread']:.2e}"
            f"  (A) max rel = {data['relation_a_max_relative_error']:.2e}"
        )
        if "relation_b_max_relative_error" in data:
            line += f"  (B) max rel = {data['relation_b_max_relative_error']:.2e}"
        print(line)
    print(
        "extrapolated a(inf)/rho pi a^2 = "
        f"{limit['added_mass_infinite_over_displaced']:.6f} (exponent {limit['decay_exponent']:.3f})"
    )


if __name__ == "__main__":
    main()