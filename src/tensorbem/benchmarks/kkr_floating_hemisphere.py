"""Kramers-Kronig causality audit of the 3-D floating-hemisphere panel solutions.

:mod:`tensorbem.benchmarks.kkr_causality` audits the *two-dimensional* Ursell
cylinder: it feeds the panel ``(A, B)`` sweeps of the indirect 2-D solver into
the hydrodynamic Kramers-Kronig relations of :mod:`tensorbem.causality` and
reads the residual against the analytic floor. The three-dimensional free
surface layer (Kelvin/Delhommeau kernel, :mod:`tensorbem.free_surface_solver3d`)
has so far only been checked *one frequency at a time*: every documented number
for the floating hemisphere is a value at a single ``Ka`` (see
``docs/marine_bem_benchmarks.md`` section 4.10, table 10). A single value can be
right in the third digit and still come from a non-causal pair of curves; a
wrong wave kernel, a mis-set radiation condition or a residual irregular
frequency shows up as ``A(omega)`` and ``B(omega)`` that cannot both belong to
one causal impulse response.

This module closes that gap. It sweeps a frequency band and reports, for each
mesh level,

* the pointwise error of the panel solution against Hulme's (1982)
  semi-analytical floating-hemisphere coefficients on the same grid,
* the Kramers-Kronig residual of the panel sweep, the residual of the Hulme
  sweep on the same grid (the *floor* of the transform at this band and
  sampling), and their ratio,
* a negative control: the panel damping is deliberately scaled by a few percent
  and the same transform has to notice.

``A_inf`` -- the infinite-frequency added mass that relation (1) needs and a
sweep does not contain -- is taken from the *Hulme* sweep fitted with the same
``A_inf + c/omega^2`` tail model (:func:`tensorbem.causality.
fit_added_mass_asymptote`) and then handed to **both** the panel and the
reference check. Any bias of the fit therefore cancels in the panel/floor ratio,
which is the number this benchmark judges.

Run::

    python -m tensorbem.benchmarks.kkr_floating_hemisphere --subdivisions 2 3 \
        --points 14 --ka-min 1.0 --ka-max 5.0 --output outputs/m3594/kkr3d.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.benchmarks.floating_hemisphere import (
    hemisphere_hull_mesh,
    hulme_hemisphere_radiation,
    solve_radiation_3d,
)
from tensorbem.causality import fit_added_mass_asymptote, kkr_check

__all__ = [
    "MODES",
    "audit_sweep",
    "run",
]

#: Rigid-body modes audited: (BEM mode index, name, Hulme added-mass / damping
#: attribute stems). Surge and heave are the two modes for which Hulme (1982)
#: tabulates both the added mass and the wave damping.
MODES = (
    (0, "surge", "surge_added_mass", "surge_damping"),
    (1, "heave", "heave_added_mass", "heave_damping"),
)

_DEFAULT_DENSITY = 1025.0
_DEFAULT_GRAVITY = 9.80665

#: Damping perturbations of the negative control. Each one has to raise the
#: Kramers-Kronig residual well above the unperturbed one; a ``1.0`` entry is
#: the control itself.
_NEGATIVE_CONTROL_FACTORS = (1.02, 1.10, 0.90, 1.30)


def _kkr_payload(report: Any) -> dict[str, Any]:
    return {
        "sample_omegas": np.asarray(report.sample_omegas).tolist(),
        "measured_damping": np.asarray(report.measured_damping).tolist(),
        "predicted_damping": np.asarray(report.predicted_damping).tolist(),
        "measured_added_mass": np.asarray(report.measured_added_mass).tolist(),
        "predicted_added_mass": np.asarray(report.predicted_added_mass).tolist(),
        "damping_relative_error": np.asarray(report.damping_relative_error).tolist(),
        "added_mass_relative_error": np.asarray(report.added_mass_relative_error).tolist(),
        "max_damping_relative_error": float(report.max_damping_relative_error),
        "max_added_mass_relative_error": float(report.max_added_mass_relative_error),
        "added_mass_inf": float(report.added_mass_inf),
        "tail_coefficient": float(report.tail_coefficient),
        "asymptote_relative_residual": float(report.asymptote_relative_residual),
    }


def _reference_sweep(
    ka: np.ndarray,
    omega: np.ndarray,
    radius: float,
    density: float,
    gravity: float,
) -> dict[str, Any]:
    """Hulme (1982) coefficients on the sweep grid, one block per mode."""
    solutions = [
        hulme_hemisphere_radiation(float(value), radius=radius, density=density, gravity=gravity)
        for value in ka
    ]
    block: dict[str, Any] = {}
    for _, name, mass_attr, damper_attr in MODES:
        mass = np.array([float(getattr(item, mass_attr)) for item in solutions])
        damper = np.array([float(getattr(item, damper_attr)) for item in solutions])
        fit = fit_added_mass_asymptote(omega, mass)
        block[name] = {
            "added_mass": mass.tolist(),
            "damping": damper.tolist(),
            "added_mass_inf_fit": float(fit.added_mass_inf),
            "tail_coefficient_fit": float(fit.tail_coefficient),
            "asymptote_relative_residual": float(fit.relative_residual),
            "kkr_with_fitted_limit": _kkr_payload(kkr_check(omega, mass, damper)),
        }
    block["displaced_volume"] = float(solutions[0].displaced_volume)
    return block


def audit_sweep(
    *,
    subdivision: int = 3,
    ka_min: float = 1.0,
    ka_max: float = 5.0,
    points: int = 14,
    radius: float = 1.0,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One mesh level of the 3-D causality audit (panel sweep + Kramers-Kronig)."""
    if points < 12:
        raise ValueError("points must be at least 12 to resolve the band")
    if not 0.0 < ka_min < ka_max:
        raise ValueError("require 0 < ka_min < ka_max")

    ka = np.geomspace(ka_min, ka_max, points)
    omega = np.sqrt(ka * gravity / radius)
    if reference is None:
        reference = _reference_sweep(ka, omega, radius, density, gravity)

    mesh = hemisphere_hull_mesh(int(subdivision), radius=radius)
    panels = int(np.asarray(mesh.centroids).shape[0])
    mass_panel = np.empty((points, 2))
    damper_panel = np.empty((points, 2))
    condition = np.empty(points)
    body_residual = np.empty(points)
    for index, wavenumber in enumerate(ka / radius):
        result = solve_radiation_3d(
            mesh,
            float(wavenumber),
            modes=tuple(mode for mode, _, _, _ in MODES),
            density=density,
            gravity=gravity,
        )
        positions = {mode: column for column, (mode, _, _, _) in enumerate(MODES)}
        for column, (mode, _, _, _) in enumerate(MODES):
            position = positions[mode]
            mass_panel[index, column] = float(result.added_mass[position, position])
            damper_panel[index, column] = float(result.damping[position, position])
        condition[index] = float(result.condition_number)
        body_residual[index] = float(result.relative_residual)

    modes: dict[str, Any] = {}
    for column, (_, name, _, _) in enumerate(MODES):
        mass = mass_panel[:, column]
        damper = damper_panel[:, column]
        reference_mass = np.asarray(reference[name]["added_mass"], dtype=float)
        reference_damper = np.asarray(reference[name]["damping"], dtype=float)
        shared_limit = float(reference[name]["added_mass_inf_fit"])

        shared = kkr_check(omega, mass, damper, added_mass_inf=shared_limit)
        floor = kkr_check(omega, reference_mass, reference_damper, added_mass_inf=shared_limit)
        self_fit = kkr_check(omega, mass, damper)

        controls: dict[str, Any] = {}
        for factor in _NEGATIVE_CONTROL_FACTORS:
            check = kkr_check(omega, mass, float(factor) * damper, added_mass_inf=shared_limit)
            controls[f"damping_x{factor:g}"] = {
                "max_damping_relative_error": float(check.max_damping_relative_error),
                "ratio_to_unperturbed": float(
                    check.max_damping_relative_error / max(shared.max_damping_relative_error, 1e-300)
                ),
            }

        modes[name] = {
            "added_mass": mass.tolist(),
            "damping": damper.tolist(),
            "added_mass_over_rho_displaced": float(
                np.max(mass) / (density * float(reference["displaced_volume"]))
            ),
            "added_mass_relative_error_max": float(
                np.max(np.abs(mass - reference_mass)) / np.max(np.abs(reference_mass))
            ),
            "damping_relative_error_max": float(
                np.max(np.abs(damper - reference_damper)) / np.max(np.abs(reference_damper))
            ),
            "added_mass_relative_error_by_ka": (
                np.abs(mass - reference_mass) / np.max(np.abs(reference_mass))
            ).tolist(),
            "damping_relative_error_by_ka": (
                np.abs(damper - reference_damper) / np.max(np.abs(reference_damper))
            ).tolist(),
            "kkr_shared_limit": _kkr_payload(shared),
            "kkr_reference_floor": _kkr_payload(floor),
            "kkr_self_fit": _kkr_payload(self_fit),
            "damping_residual_ratio_panel_over_reference": float(
                shared.max_damping_relative_error
                / max(floor.max_damping_relative_error, 1e-300)
            ),
            "added_mass_residual_ratio_panel_over_reference": float(
                shared.max_added_mass_relative_error
                / max(floor.max_added_mass_relative_error, 1e-300)
            ),
            "negative_controls": controls,
        }

    return {
        "subdivision": int(subdivision),
        "panels": panels,
        "body_condition_relative_residual_max": float(np.max(body_residual)),
        "condition_number_max": float(np.max(condition)),
        "condition_number_min": float(np.min(condition)),
        "modes": modes,
    }


def run(
    *,
    subdivisions: Sequence[int] = (2, 3),
    ka_min: float = 1.0,
    ka_max: float = 5.0,
    points: int = 14,
    radius: float = 1.0,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> dict[str, Any]:
    """Full 3-D causality audit across mesh levels (JSON-serialisable)."""
    if not subdivisions:
        raise ValueError("at least one mesh subdivision is required")
    ka = np.geomspace(ka_min, ka_max, points)
    omega = np.sqrt(ka * gravity / radius)
    reference = _reference_sweep(ka, omega, radius, density, gravity)

    cases: dict[str, Any] = {}
    for subdivision in subdivisions:
        levels = audit_sweep(
            subdivision=int(subdivision),
            ka_min=ka_min,
            ka_max=ka_max,
            points=points,
            radius=radius,
            density=density,
            gravity=gravity,
            reference=reference,
        )
        cases[f"subdivision_{int(subdivision)}_panels_{levels['panels']}"] = levels

    return {
        "benchmark": "kramers_kronig_causality_audit_3d",
        "case": "hulme_1982_floating_hemisphere_deep_water_radiation",
        "reference": "Hulme (1982) wave-free-potential semi-analytical coefficients",
        "method": "3-D constant-panel Kelvin/Delhommeau free-surface BEM "
        "(tensorbem.free_surface_solver3d.solve_radiation_3d)",
        "radius": radius,
        "density": density,
        "gravity": gravity,
        "ka_min": ka_min,
        "ka_max": ka_max,
        "points": points,
        "ka": ka.tolist(),
        "omega": omega.tolist(),
        "displaced_volume": reference["displaced_volume"],
        "displaced_mass": density * reference["displaced_volume"],
        "reference_sweep": reference,
        "cases": cases,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivisions", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--ka-min", type=float, default=1.0)
    parser.add_argument("--ka-max", type=float, default=5.0)
    parser.add_argument("--points", type=int, default=14)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--density", type=float, default=_DEFAULT_DENSITY)
    parser.add_argument("--gravity", type=float, default=_DEFAULT_GRAVITY)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    record = run(
        subdivisions=tuple(args.subdivisions),
        ka_min=args.ka_min,
        ka_max=args.ka_max,
        points=args.points,
        radius=args.radius,
        density=args.density,
        gravity=args.gravity,
    )
    text = json.dumps(record, indent=2, sort_keys=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())