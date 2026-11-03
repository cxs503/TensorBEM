"""Mean second-order wave-drift force on a submerged 2-D body: far vs near field.

The study answers one question with two independent answers: what is the mean
(time-averaged) second-order horizontal force on a fixed, fully submerged
two-dimensional body in deep-water regular waves?

* the **far-field** route uses the momentum theorem,
  ``F_x = (rho*g/2)*|R*A|^2``, with ``R`` the reflection coefficient from the
  Kochin integrals of :func:`tensorbem.free_surface_solver.solve_diffraction_indirect`;
* the **near-field** route integrates the mean second-order pressure over the
  body, ``F_j = (rho/2)*oint <|grad phi|^2> n_j ds``, which needs the surface
  tangential velocity recovered from the panel trace
  (:func:`tensorbem.drift.surface_tangential_velocity`).

Cases (all deep water, ``ka = 0.6`` unless stated, ``rho = 1025``,
``g = 9.80665``, incident elevation amplitude ``A``):

1. ``circle``: a circular cylinder of radius 1 m at 2 m submergence.  Dean's
   (1948) / Ursell's (1950) zero-reflection theorem makes the exact drift force
   zero at every frequency, so both routes are checked against a closed-form
   value of zero.
2. ``ellipse``: a strongly reflecting ellipse (semi-axes 1.4 m / 0.7 m,
   submergence 1.2 m) refined over panel count; the two routes must converge to
   the same non-zero value.
3. ``amplitude``: the force must scale as ``A^2``.
4. ``submergence``: a deeper body reflects less, so ``|R|^2`` and the drift force
   must decay together.
5. ``band``: sensitivity of the near-field route to the Fourier band limit of
   the tangential-velocity recovery.
6. ``recovery``: the recovery layer alone against an analytic harmonic field
   whose surface integral is computed by high-order quadrature (exact truth, no
   boundary-element solve involved).

Run::

    python -m tensorbem.benchmarks.drift_force --output outputs/drift_force_2d/study.json

The documentation page ``docs/drift_force_2d.md`` is rendered from that JSON by
``python docs/tools/gen_drift_force_doc.py``, so no number in it is retyped.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np

from ..drift import (
    DEFAULT_MODE_FRACTION,
    mean_drift_force,
    mean_drift_force_near_field,
)
from ..free_surface_solver import solve_diffraction_indirect
from .dean_diffraction import submerged_circle_mesh, submerged_ellipse_mesh

SCHEMA = "tensorbem-drift-force/1"
DENSITY = 1025.0
GRAVITY = 9.80665
KA = 0.6
AMPLITUDE = 1.0

#: Two-mode analytic field used by the exact-truth recovery check.  The
#: wavenumbers differ (``0.6`` and ``0.9``) and the coefficient is complex, so
#: ``|grad phi|^2`` is not symmetric in ``x`` and the surface integral does not
#: vanish identically.
RECOVERY_FIELD = {"nu": 0.6, "mu": 0.9, "coefficient_abs": 0.7, "coefficient_phase": math.pi / 3.0}


def _analytic_field(x, z, *, nu: float, mu: float, coefficient: complex):
    first = np.exp(nu * z + 1.0j * nu * x)
    second = np.exp(mu * z + 1.0j * mu * x)
    potential = first + coefficient * second
    grad_x = 1.0j * nu * first + 1.0j * mu * coefficient * second
    grad_z = nu * first + mu * coefficient * second
    return potential, grad_x, grad_z


def _analytic_surface_force(radius: float, submergence: float) -> float:
    """``(rho/4)*oint |dphi/ds|^2 n_x ds`` for :data:`RECOVERY_FIELD`.

    Evaluated on the exact circle with the trapezoidal rule, which is
    spectrally accurate for a smooth periodic integrand; the number of points is
    large enough that the reference is converged to machine precision.
    """
    nu = RECOVERY_FIELD["nu"]
    mu = RECOVERY_FIELD["mu"]
    coefficient = RECOVERY_FIELD["coefficient_abs"] * np.exp(
        1.0j * RECOVERY_FIELD["coefficient_phase"]
    )
    theta = np.linspace(0.0, 2.0 * math.pi, 65537)[:-1]
    x = radius * np.cos(theta)
    z = radius * np.sin(theta) - submergence
    _, grad_x, grad_z = _analytic_field(x, z, nu=nu, mu=mu, coefficient=coefficient)
    tangent_x = -np.sin(theta)
    tangent_z = np.cos(theta)
    tangential = grad_x * tangent_x + grad_z * tangent_z
    step = 2.0 * math.pi / theta.size
    return float(0.25 * DENSITY * np.sum(np.abs(tangential) ** 2 * np.cos(theta)) * step)


def naive_surface_force(mesh, potential, density=DENSITY):
    """Near-field force from an unfiltered central difference of the trace.

    This is the failure mode that motivates the band-limited recovery: the
    constant-panel trace carries a panel-scale sawtooth component, and
    differencing it amplifies that component by ``1/h``.  It is kept here as a
    measured diagnostic so the report can quote the damage instead of asserting
    it.
    """
    from ..drift import panel_arclength

    s, perimeter = panel_arclength(mesh)
    derivative = np.empty_like(potential)
    derivative[0] = (potential[1] - potential[-1]) / (s[1] - s[-1] + perimeter)
    derivative[-1] = (potential[0] - potential[-2]) / (s[0] - s[-1] + perimeter)
    derivative[1:-1] = (potential[2:] - potential[:-2]) / (s[2:] - s[:-2])
    kinetic = 0.5 * np.abs(derivative) ** 2
    integrand = kinetic[:, None] * mesh.normals * mesh.lengths[:, None]
    return 0.5 * density * np.sum(integrand, axis=0)


def recovery_study(panel_counts=(16, 32, 64, 128, 256)) -> dict:
    """Exact-truth check of the tangential-velocity recovery and surface quadrature."""
    nu = RECOVERY_FIELD["nu"]
    mu = RECOVERY_FIELD["mu"]
    coefficient = RECOVERY_FIELD["coefficient_abs"] * np.exp(
        1.0j * RECOVERY_FIELD["coefficient_phase"]
    )
    reference = _analytic_surface_force(1.0, 2.0)
    rows = []
    for panels in panel_counts:
        mesh = submerged_circle_mesh(int(panels), 1.0, 2.0)
        potential, _, _ = _analytic_field(
            mesh.centers[:, 0], mesh.centers[:, 1], nu=nu, mu=mu, coefficient=coefficient
        )
        force = mean_drift_force_near_field(mesh, potential, density=DENSITY)
        rows.append(
            {
                "panels": int(panels),
                "force_x": float(force[0]),
                "relative_error": float(abs(force[0] - reference) / abs(reference)),
            }
        )
    orders = [
        float(
            math.log(rows[i]["relative_error"] / rows[i + 1]["relative_error"]) / math.log(2.0)
        )
        for i in range(len(rows) - 1)
    ]
    return {
        "field": {
            "nu": nu,
            "mu": mu,
            "coefficient_abs": RECOVERY_FIELD["coefficient_abs"],
            "coefficient_phase": RECOVERY_FIELD["coefficient_phase"],
        },
        "reference_force_x": reference,
        "panel_counts": [int(p) for p in panel_counts],
        "rows": rows,
        "observed_orders": orders,
        "note": (
            "Analytic two-mode harmonic field (nu = 0.6, mu = 0.9, complex coefficient) "
            "on a circle; the reference integral is a spectrally accurate trapezoidal "
            "evaluation of the exact surface tangential derivative, so no boundary-element "
            "discretisation enters the truth value."
        ),
    }


def _drift_row(mesh, nu, amplitude, density, gravity, mode_fraction, panel_nodes) -> dict:
    result = solve_diffraction_indirect(
        mesh, nu, amplitude, density=density, gravity=gravity, panel_nodes=panel_nodes
    )
    force = mean_drift_force(
        mesh,
        result,
        amplitude=amplitude,
        density=density,
        gravity=gravity,
        mode_fraction=mode_fraction,
    )
    return {
        "panels": mesh.panel_count,
        "reflection_modulus": abs(result.reflection_coefficient),
        "transmission_modulus": abs(result.transmission_coefficient),
        "energy_balance_error": result.energy_balance_error,
        "far_field_force_x": force.far_field,
        "near_field_force_x": float(force.near_field[0]),
        "near_field_force_z": float(force.near_field[1]),
        "relative_discrepancy": force.relative_discrepancy,
        "discrepancy_normalised": force.discrepancy_normalised,
        "force_ratio_far": force.far_field / force.energy_scale,
        "force_ratio_near": force.force_ratio,
        "condition_number": result.condition_number,
        "relative_residual": result.relative_residual,
    }


def run(
    panel_counts=(32, 64, 128, 256),
    *,
    ka: float = KA,
    amplitude: float = AMPLITUDE,
    density: float = DENSITY,
    gravity: float = GRAVITY,
    mode_fraction: float = DEFAULT_MODE_FRACTION,
    panel_nodes: int = 24,
    radius: float = 1.0,
    circle_submergence: float = 2.0,
    ellipse_semi_major: float = 1.4,
    ellipse_semi_minor: float = 0.7,
    ellipse_submergence: float = 1.2,
    amplitudes=(0.5, 1.0, 2.0),
    submergences=(1.2, 1.6, 2.0, 2.6, 3.5),
    band_fractions=(0.10, 0.15, 0.25, 0.35),
) -> dict:
    """Run the full drift-force verification study and return a JSON-able dict."""
    nu = ka / float(radius)
    counts = [int(p) for p in panel_counts]

    circle_rows = []
    for panels in counts:
        mesh = submerged_circle_mesh(panels, radius, circle_submergence)
        circle_rows.append(
            _drift_row(mesh, nu, amplitude, density, gravity, mode_fraction, panel_nodes)
        )

    ellipse_rows = []
    for panels in counts:
        mesh = submerged_ellipse_mesh(
            panels, ellipse_semi_major, ellipse_semi_minor, ellipse_submergence
        )
        ellipse_rows.append(
            _drift_row(mesh, nu, amplitude, density, gravity, mode_fraction, panel_nodes)
        )

    amplitude_rows = []
    for amp in amplitudes:
        mesh = submerged_ellipse_mesh(
            counts[len(counts) // 2 - 1] if len(counts) > 1 else counts[0],
            ellipse_semi_major,
            ellipse_semi_minor,
            ellipse_submergence,
        )
        row = _drift_row(mesh, nu, amp, density, gravity, mode_fraction, panel_nodes)
        row["amplitude"] = float(amp)
        row["force_ratio_far_over_amplitude_squared"] = row["far_field_force_x"] / float(amp) ** 2
        row["force_ratio_near_over_amplitude_squared"] = row["near_field_force_x"] / float(amp) ** 2
        amplitude_rows.append(row)

    submergence_rows = []
    for submergence in submergences:
        if submergence <= ellipse_semi_minor:
            raise ValueError("every submergence must exceed the ellipse semi-minor axis")
        mesh = submerged_ellipse_mesh(
            counts[len(counts) // 2 - 1] if len(counts) > 1 else counts[0],
            ellipse_semi_major,
            ellipse_semi_minor,
            float(submergence),
        )
        row = _drift_row(mesh, nu, amplitude, density, gravity, mode_fraction, panel_nodes)
        row["submergence"] = float(submergence)
        row["clearance_over_semi_minor"] = float(
            (submergence - ellipse_semi_minor) / ellipse_semi_minor
        )
        submergence_rows.append(row)

    band_rows = []
    submergence = float(submergences[0])
    mesh = submerged_ellipse_mesh(
        counts[len(counts) // 2 - 1] if len(counts) > 1 else counts[0],
        ellipse_semi_major,
        ellipse_semi_minor,
        submergence,
    )
    result = solve_diffraction_indirect(
        mesh, nu, amplitude, density=density, gravity=gravity, panel_nodes=panel_nodes
    )
    reference_force = mean_drift_force(
        mesh, result, amplitude=amplitude, density=density, gravity=gravity
    ).far_field
    for fraction in band_fractions:
        force = mean_drift_force(
            mesh,
            result,
            amplitude=amplitude,
            density=density,
            gravity=gravity,
            mode_fraction=float(fraction),
        )
        band_rows.append(
            {
                "mode_fraction": float(fraction),
                "mode_count": int(math.floor(float(fraction) * mesh.panel_count)),
                "near_field_force_x": float(force.near_field[0]),
                "relative_discrepancy": force.relative_discrepancy,
                "relative_change_vs_default": float(
                    (force.near_field[0] - reference_force) / reference_force
                ),
            }
        )
    raw_force = naive_surface_force(mesh, result.total_potential, density=density)

    return {
        "schema": SCHEMA,
        "benchmark": "two_dimensional_deep_water_mean_drift_force",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "settings": {
            "ka": float(ka),
            "wavenumber": float(nu),
            "wave_amplitude": float(amplitude),
            "fluid_density": float(density),
            "gravity": float(gravity),
            "mode_fraction": float(mode_fraction),
            "panel_nodes": int(panel_nodes),
            "radius": float(radius),
            "circle_submergence": float(circle_submergence),
            "ellipse_semi_major": float(ellipse_semi_major),
            "ellipse_semi_minor": float(ellipse_semi_minor),
            "ellipse_submergence": float(ellipse_submergence),
            "energy_scale": float(density * gravity * amplitude**2),
            "fluid_density_units": "kg/m^3",
            "force_units": "N/m (per unit crest length)",
        },
        "exact_references": {
            "circle_drift_force": 0.0,
            "circle_basis": "Dean (1948) / Ursell (1950) zero reflection in deep water",
            "far_field_formula": "F_x = (rho*g/2)*|R*A|^2",
            "near_field_formula": "F_j = (rho/2)*oint<|grad phi|^2> n_j ds",
        },
        "circle": circle_rows,
        "ellipse": ellipse_rows,
        "amplitude_scaling": amplitude_rows,
        "submergence_decay": submergence_rows,
        "band_sensitivity": band_rows,
        "unfiltered_difference": {
            "panels": mesh.panel_count,
            "submergence": submergence,
            "raw_force_x": float(raw_force[0]),
            "far_field_force_x": reference_force,
            "relative_error": float((raw_force[0] - reference_force) / reference_force),
            "note": (
                "central difference of the panel trace without band limiting; "
                "the panel-scale sawtooth is amplified by 1/h and swamps the force"
            ),
        },
        "recovery": recovery_study(),
        "note": (
            "Both routes are dimensionless-normalised by rho*g*A^2, the mean "
            "momentum flux scale of the incident wave train. The circular cylinder "
            "must give exactly zero; the ellipse gives the non-zero case where the "
            "two routes can be compared with each other."
        ),
    }


def figure(study: dict, path: Path) -> None:
    """Render the three-panel verification figure of the study."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_obj, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    ellipse = study["ellipse"]
    panels = np.array([row["panels"] for row in ellipse], dtype=float)
    far = np.array([row["force_ratio_far"] for row in ellipse])
    near = np.array([row["force_ratio_near"] for row in ellipse])
    axis = axes[0]
    axis.plot(1.0 / panels, far, "o-", label="far field (momentum theorem)")
    axis.plot(1.0 / panels, near, "s--", label="near field (pressure integration)")
    axis.set_xlabel("1 / panels")
    axis.set_ylabel(r"$F_x/(\rho g A^2)$")
    axis.set_title("Submerged ellipse: both routes converge")
    axis.grid(True, alpha=0.3)
    axis.legend()

    submergence_rows = study["submergence_decay"]
    clearance = np.array([row["submergence"] for row in submergence_rows])
    reflection_sq = np.array([row["reflection_modulus"] ** 2 for row in submergence_rows])
    ratio_far = np.array([row["force_ratio_far"] for row in submergence_rows])
    ratio_near = np.array([row["force_ratio_near"] for row in submergence_rows])
    axis = axes[1]
    axis.semilogy(clearance, reflection_sq, "o-", label=r"$|R|^2$")
    axis.semilogy(clearance, np.abs(ratio_far), "s--", label=r"$|F^{far}_x|/(\rho g A^2)$")
    axis.semilogy(clearance, np.abs(ratio_near), "^:", label=r"$|F^{near}_x|/(\rho g A^2)$")
    axis.set_xlabel("axis submergence [m]")
    axis.set_ylabel("dimensionless")
    axis.set_title("Deeper body: reflection and force decay together")
    axis.grid(True, alpha=0.3)
    axis.legend()

    recovery = study["recovery"]
    panels = np.array([row["panels"] for row in recovery["rows"]], dtype=float)
    error = np.array([row["relative_error"] for row in recovery["rows"]])
    axis = axes[2]
    axis.loglog(panels, error, "o-", label="recovery + surface quadrature")
    axis.loglog(
        panels,
        error[0] * (panels[0] / panels) ** 2,
        "k--",
        alpha=0.6,
        label=r"$O(h^2)$ reference",
    )
    axis.set_xlabel("panels")
    axis.set_ylabel("relative error")
    axis.set_title("Tangential velocity vs analytic field")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()

    figure_obj.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure_obj.savefig(path, dpi=150)
    plt.close(figure_obj)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panels",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256],
        help="panel counts for the refinement series",
    )
    parser.add_argument("--ka", type=float, default=KA)
    parser.add_argument("--amplitude", type=float, default=AMPLITUDE)
    parser.add_argument("--density", type=float, default=DENSITY)
    parser.add_argument("--gravity", type=float, default=GRAVITY)
    parser.add_argument("--mode-fraction", type=float, default=DEFAULT_MODE_FRACTION)
    parser.add_argument("--panel-nodes", type=int, default=24)
    parser.add_argument("--output", type=Path, help="write the study JSON here")
    parser.add_argument("--plot", type=Path, help="write the verification figure here")
    args = parser.parse_args()
    try:
        study = run(
            args.panels,
            ka=args.ka,
            amplitude=args.amplitude,
            density=args.density,
            gravity=args.gravity,
            mode_fraction=args.mode_fraction,
            panel_nodes=args.panel_nodes,
        )
    except ValueError as error:
        parser.error(str(error))
    rendered = json.dumps(study, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.plot:
        figure(study, args.plot)
    print(rendered)


if __name__ == "__main__":
    main()