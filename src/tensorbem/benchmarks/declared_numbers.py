"""Recompute every headline number that the documentation quotes, and judge it.

The documentation of this package states concrete numbers: convergence errors at
a given panel count, a reference-solution value, a near/far-field closure. Those
sentences are the contract with a reader who does not want to run anything. They
also rot: a kernel revision, a mesh change or a different ``g`` moves them, and
nobody notices because a single number in a table is not executable.

This module makes them executable. Each quoted number is registered with

* the file and section that quotes it (:attr:`DeclaredNumber.source`),
* the benchmark that produces it and the exact parametric setting,
* the tolerance at which a deviation counts as drift.

:func:`check` re-runs the benchmarks and returns one row per quoted number with
``PASS`` or ``DRIFT``. ``python -m tensorbem.benchmarks.declared_numbers`` prints
the same table as strict JSON; ``--strict`` turns a drift into a non-zero exit
status so a CI step or a reviewer can gate on it.

Provenance note (2026-11-03). The Ursell rows were re-derived during a
verification audit: the previously quoted ``7.7e-3`` (added mass) and
``1.3e-2`` (damping) at 128 panels are *not reproducible from the released
code* -- the same release that introduced them already returns ``8.51e-3`` and
``1.34e-2``, and no documented setting (panel node count, angular offset,
gravity) recovers the older pair. The registry and the documents now carry the
reproducible values, and :func:`ursell_refinement` records the evidence that the
reference itself is right: the panel family extrapolated to zero mesh size
agrees with Ursell's multipole solution to about ``1e-5``. The source strings
keep the citation; the original values are recorded in
``dev_report_tensorbem_declared_numbers.md`` rather than silently overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from tensorbem.analytic import ursell_cylinder_added_mass
from tensorbem.benchmarks import (
    cylinder_added_mass as cylinder_benchmark,
    ellipsoid_added_mass as ellipsoid_benchmark,
    floating_hemisphere as hulme_benchmark,
    interior_harmonics as interior_harmonics_benchmark,
    submerged_sphere_wave as wang_linton_benchmark,
    ursell_cylinder as ursell_benchmark,
)

__all__ = [
    "DECLARED_NUMBERS",
    "DeclaredNumber",
    "TIERS",
    "check",
    "richardson_limit",
    "ursell_refinement",
]

#: Cost classes, cheapest first. ``check`` runs the tiers it is asked for; the
#: heavy tier exists because a 1280-panel three-dimensional assembly costs
#: minutes while a 16-panel two-dimensional one costs milliseconds.
TIERS = ("fast", "standard", "heavy")

#: A quoted number is considered reproduced when the measured value is within
#: this relative deviation. Well above floating-point noise (the benchmarks are
#: deterministic) and well below the 1e-2-level effects a code change produces.
DEFAULT_TOLERANCE = 1.5e-2


@dataclass(frozen=True)
class DeclaredNumber:
    """One number quoted in the documentation, with the way to recompute it."""

    key: str
    description: str
    source: str
    group: str
    """Which benchmark to run: ``cylinder``, ``ellipsoid``, ``ursell`` or
    ``ursell_reference``."""
    quantity: str
    """Key into the dictionary returned by the group's runner."""
    declared: float
    tier: str = "fast"
    parameters: dict = field(default_factory=dict)
    tolerance: float = DEFAULT_TOLERANCE
    kind: str = "value"
    """``value`` compares for equality, ``upper`` only checks ``measured <=
    declared`` (for sentences of the form "within 6e-3")."""


# --------------------------------------------------------------------------
# 3-D deep-water free-surface tables (Hulme 1982 / Wang & Linton)
# --------------------------------------------------------------------------
# docs/marine_bem_benchmarks.md sections 4.10-4.11 quote three convergence
# tables and one irregular-frequency (CHIEF) study for the three-dimensional
# Kelvin-Green-function solver. Until now none of those numbers was executable:
# a revision of ``tensorbem.free_surface3d`` or of the multipole reference could
# have invalidated the tables without a single test turning red. They are
# registered below as one heavy-tier group per study, so that every row of a
# table shares one solve (the cache key is ``(group, parameters)``).

#: Hulme (1982) floating hemisphere, Ka = 1.0, a = 1 m, rho = 1025,
#: octahedral hull subdivisions 1/2/3 -> 16/64/256 panels.
_HULME_REFINEMENT_PARAMETERS = {"subdivisions": [1, 2, 3], "ka": 1.0}
#: 3-D interior-CHIEF study at the first waterplane resonance.
_HULME_CHIEF_PARAMETERS = {"subdivision": 3, "ka_values": [2.58]}
#: Wang (1986) / Linton (1991) submerged sphere, f/a = 1.5, Ka = 0.8.
_WANG_LINTON_PARAMETERS = {"subdivisions": [0, 1, 2], "ka": 0.8}

#: Table 10: (subdivisions, panels, quantity, documented value).
#: Columns A33 / B33 / A11 / B11 / Fz are the relative errors of the heave and
#: surge added mass and damping and of the vertical wave force; the B11 order is
#: quoted in the table but is not emitted by the benchmark, so the runner
#: derives it from the consecutive-level error ratio (an exact log2 identity
#: for a pure power law).
_TABLE_HULME = (
    (1, 16, "condition_number", 3.893),
    (1, 16, "heave_added_mass_rel_error", 1.20e-1),
    (1, 16, "heave_damping_rel_error", 1.22e-1),
    (1, 16, "surge_added_mass_rel_error", 1.80e-1),
    (1, 16, "surge_damping_rel_error", 4.02e-1),
    (1, 16, "excitation_force_z_rel_error", 6.67e-2),
    (2, 64, "condition_number", 3.052),
    (2, 64, "heave_added_mass_rel_error", 2.97e-2),
    (2, 64, "heave_damping_rel_error", 4.50e-2),
    (2, 64, "surge_added_mass_rel_error", 5.05e-2),
    (2, 64, "surge_damping_rel_error", 1.30e-1),
    (2, 64, "excitation_force_z_rel_error", 2.21e-2),
    (2, 64, "observed_order_a33", 2.23),
    (2, 64, "observed_order_b33", 1.59),
    (2, 64, "observed_order_a11", 2.04),
    (2, 64, "observed_order_b11", 1.62),
    (2, 64, "observed_order_fz", 1.77),
    (3, 256, "condition_number", 2.783),
    (3, 256, "heave_added_mass_rel_error", 7.41e-3),
    (3, 256, "heave_damping_rel_error", 1.18e-2),
    (3, 256, "surge_added_mass_rel_error", 1.30e-2),
    (3, 256, "surge_damping_rel_error", 3.53e-2),
    (3, 256, "excitation_force_z_rel_error", 5.67e-3),
    (3, 256, "observed_order_a33", 2.06),
    (3, 256, "observed_order_b33", 1.98),
    (3, 256, "observed_order_a11", 2.01),
    (3, 256, "observed_order_b11", 1.88),
    (3, 256, "observed_order_fz", 2.02),
)

#: Table 11: (block, quantity, documented value) at Ka = 2.58, 256 panels.
#: ``exact`` is the Hulme semi-analytical reference; ``unregularized`` is the
#: plain exterior BIE (which produces an unphysical negative heave damping);
#: ``chief_regularized`` is the five-point 3-D interior null-field (CHIEF)
#: augmentation, without waterplane lid panels.
_TABLE_HULME_CHIEF = (
    ("exact", "a33_dimless", 0.4007),
    ("exact", "b33_dimless", 0.0631),
    ("unregularized", "condition_number", 145.66),
    ("unregularized", "a33_dimless", 0.3481),
    ("unregularized", "b33_dimless", -0.1042),
    ("chief_regularized", "chief_point_count", 5.0),
    ("chief_regularized", "condition_number", 4.24),
    ("chief_regularized", "a33_dimless", 0.3919),
    ("chief_regularized", "b33_dimless", 0.0687),
)

#: Table 12: (subdivisions, panels, quantity, documented value).
#: ``rotational_added_mass_max_ratio`` is quoted at two significant digits only
#: (2.5e-32 / 2.2e-4 / 2.6e-5), i.e. with a rounding half-width of ~1.9 %; the
#: 20-panel cell is a machine-zero claim and is therefore registered as an upper
#: bound, while the two finite cells get a tolerance equal to that band.
_TABLE_WANG_LINTON = (
    (0, 20, "condition_number", 2.407),
    (0, 20, "heave_added_mass_rel_error", 2.55e-1),
    (0, 20, "surge_added_mass_rel_error", 2.68e-1),
    (0, 20, "excitation_force_x_rel_error", 3.60e-1),
    (0, 20, "excitation_force_z_rel_error", 3.70e-1),
    (0, 20, "rotational_added_mass_max_ratio", 1.0e-20),
    (1, 80, "condition_number", 2.259),
    (1, 80, "heave_added_mass_rel_error", 7.30e-2),
    (1, 80, "surge_added_mass_rel_error", 8.02e-2),
    (1, 80, "excitation_force_x_rel_error", 1.16e-1),
    (1, 80, "excitation_force_z_rel_error", 1.19e-1),
    (1, 80, "rotational_added_mass_max_ratio", 2.2e-4),
    (1, 80, "observed_order_a33", 2.12),
    (1, 80, "observed_order_b33", 1.72),
    (1, 80, "observed_order_a11", 2.04),
    (1, 80, "observed_order_fz", 1.91),
    (2, 320, "condition_number", 2.188),
    (2, 320, "heave_added_mass_rel_error", 1.85e-2),
    (2, 320, "surge_added_mass_rel_error", 2.08e-2),
    (2, 320, "excitation_force_x_rel_error", 3.10e-2),
    (2, 320, "excitation_force_z_rel_error", 3.20e-2),
    (2, 320, "rotational_added_mass_max_ratio", 2.6e-5),
    (2, 320, "observed_order_a33", 2.06),
    (2, 320, "observed_order_b33", 1.93),
    (2, 320, "observed_order_a11", 2.03),
    (2, 320, "observed_order_fz", 1.98),
)

#: Human-readable names for the quantities the tables quote.
_QUANTITY_LABEL = {
    "condition_number": "condition number",
    "heave_added_mass_rel_error": "A33 relative error",
    "heave_damping_rel_error": "B33 relative error",
    "surge_added_mass_rel_error": "A11 relative error",
    "surge_damping_rel_error": "B11 relative error",
    "excitation_force_z_rel_error": "Fz relative error",
    "excitation_force_x_rel_error": "Fx relative error",
    "observed_order_a33": "observed order of A33",
    "observed_order_b33": "observed order of B33",
    "observed_order_a11": "observed order of A11",
    "observed_order_b11": "observed order of B11",
    "observed_order_fz": "observed order of Fz",
    "rotational_added_mass_max_ratio": "rotational coupling about the body centre",
    "a33_dimless": "A33 dimensionless",
    "b33_dimless": "B33 dimensionless",
    "chief_point_count": "number of interior CHIEF points",
}

_REFINEMENT_TOLERANCE = 5.0e-3
"""Tables 10 and 12 quote three to four significant digits. The values are
deterministic (fixed mesh, fixed solver), so a 5e-3 relative band is still five
orders of magnitude above the reproducibility floor while catching any kernel
revision that would move a quoted digit."""


def _refinement_rows(
    prefix, table, *, group, source, parameters, upper=(), tolerances=None
):
    """One :class:`DeclaredNumber` per quoted table cell."""
    tolerances = tolerances or {}
    rows = []
    for subdivisions, panels, quantity, declared in table:
        is_upper = quantity in upper and declared <= 1.0e-15  # machine-zero cell
        rows.append(
            DeclaredNumber(
                key=f"{prefix}_{quantity}_{panels}",
                description=(
                    f"{_QUANTITY_LABEL.get(quantity, quantity)}, "
                    f"{panels} panels"
                ),
                source=source,
                group=group,
                quantity=f"level{subdivisions}_{quantity}",
                parameters=dict(parameters),
                declared=declared,
                tier="heavy",
                tolerance=tolerances.get(quantity, _REFINEMENT_TOLERANCE),
                kind="upper" if is_upper else "value",
            )
        )
    return tuple(rows)


def _chief_rows():
    """One :class:`DeclaredNumber` per cell of the irregular-frequency table."""
    rows = []
    for block, quantity, declared in _TABLE_HULME_CHIEF:
        rows.append(
            DeclaredNumber(
                key=f"hulme_chief_{block}_{quantity}",
                description=(
                    f"Ka = 2.58, 256 panels, {block}: "
                    f"{_QUANTITY_LABEL.get(quantity, quantity)}"
                ),
                source="docs/marine_bem_benchmarks.md section 4.10 table 11",
                group="hulme3d_chief",
                quantity=f"ka2p58_{block}_{quantity}",
                parameters=dict(_HULME_CHIEF_PARAMETERS),
                declared=declared,
                tier="heavy",
                tolerance=_REFINEMENT_TOLERANCE,
            )
        )
    return tuple(rows)


def _free_surface_3d_rows() -> tuple[DeclaredNumber, ...]:
    """The 3-D Kelvin-Green-function tables of docs/marine_bem_benchmarks.md."""
    return (
        _refinement_rows(
            "hulme",
            _TABLE_HULME,
            group="hulme3d",
            source="docs/marine_bem_benchmarks.md section 4.10 table 10",
            parameters=_HULME_REFINEMENT_PARAMETERS,
        )
        + _chief_rows()
        + _refinement_rows(
            "wang_linton",
            _TABLE_WANG_LINTON,
            group="wang_linton3d",
            source="docs/marine_bem_benchmarks.md section 4.11 table 12",
            parameters=_WANG_LINTON_PARAMETERS,
            upper=("rotational_added_mass_max_ratio",),
            tolerances={"rotational_added_mass_max_ratio": 2.5e-2},
        )
    )


#: The registry. ``source`` cites the current wording; every value below is the
#: number a reader of the docs will find there after the 2026-11-03 audit.
DECLARED_NUMBERS: tuple[DeclaredNumber, ...] = (
    # --- 2-D translating cylinder (Laplace, no free surface) ---------------
    DeclaredNumber(
        key="cylinder_added_mass_16",
        description="relative error of the cylinder added mass at 16 panels",
        source="README.md (Milestone 3); docs/classical_benchmarks.md section 2",
        group="cylinder",
        quantity="added_mass_relative_error",
        parameters={"panels": 16},
        declared=2.623278e-2,
    ),
    DeclaredNumber(
        key="cylinder_added_mass_128",
        description="relative error of the cylinder added mass at 128 panels",
        source="README.md (Milestone 3); docs/classical_benchmarks.md section 2",
        group="cylinder",
        quantity="added_mass_relative_error",
        parameters={"panels": 128},
        declared=4.023222e-4,
    ),
    # --- 3-D translating ellipsoid (Laplace, no free surface) --------------
    DeclaredNumber(
        key="ellipsoid_added_mass_1280_x",
        description="relative error, longest semi-axis, 1280 panels",
        source="README.md (Milestone 3); docs/classical_benchmarks.md section 3 table",
        group="ellipsoid",
        quantity="added_mass_relative_error_x",
        parameters={"subdivisions": 3},
        declared=4.1679e-3,
        tier="heavy",
    ),
    DeclaredNumber(
        key="ellipsoid_added_mass_1280_y",
        description="relative error, intermediate semi-axis, 1280 panels",
        source="README.md (Milestone 3); docs/classical_benchmarks.md section 3 table",
        group="ellipsoid",
        quantity="added_mass_relative_error_y",
        parameters={"subdivisions": 3},
        declared=4.4621e-3,
        tier="heavy",
    ),
    DeclaredNumber(
        key="ellipsoid_added_mass_1280_z",
        description="relative error, shortest semi-axis, 1280 panels",
        source="README.md (Milestone 3); docs/classical_benchmarks.md section 3 table",
        group="ellipsoid",
        quantity="added_mass_relative_error_z",
        parameters={"subdivisions": 3},
        declared=4.9359e-3,
        tier="heavy",
    ),
    # --- Ursell reference solution (independent analytical construction) ---
    DeclaredNumber(
        key="ursell_reference_added_mass",
        description="Ursell multipole added mass, a=1, f=2, ka=0.6, rho=1025",
        source="docs/marine_bem_benchmarks.md section 4.6 table (reference row)",
        group="ursell_reference",
        quantity="reference_added_mass",
        declared=2545.17,
        tolerance=1.0e-5,
    ),
    DeclaredNumber(
        key="ursell_reference_damping",
        description="Ursell multipole damping, a=1, f=2, ka=0.6, rho=1025",
        source="docs/marine_bem_benchmarks.md section 4.6 table (reference row)",
        group="ursell_reference",
        quantity="reference_damping",
        declared=2718.15,
        tolerance=1.0e-5,
    ),
    # --- Ursell panel solution (indirect formulation, constant panels) -----
    DeclaredNumber(
        key="ursell_added_mass_64",
        description="panel added-mass error at 64 panels",
        source="docs/marine_bem_benchmarks.md section 4.6 table",
        group="ursell",
        quantity="panel_added_mass_relative_error",
        parameters={"panels": 64},
        declared=1.6736e-2,
    ),
    DeclaredNumber(
        key="ursell_damping_64",
        description="panel damping error at 64 panels",
        source="docs/marine_bem_benchmarks.md section 4.6 table",
        group="ursell",
        quantity="panel_damping_relative_error",
        parameters={"panels": 64},
        declared=2.5415e-2,
    ),
    DeclaredNumber(
        key="ursell_added_mass_128",
        description="panel added-mass error at 128 panels",
        source="README.md (Milestone 4); docs/marine_bem_benchmarks.md section 4.6",
        group="ursell",
        quantity="panel_added_mass_relative_error",
        parameters={"panels": 128},
        declared=8.5130e-3,
        tier="standard",
    ),
    DeclaredNumber(
        key="ursell_damping_128",
        description="panel damping error at 128 panels",
        source="README.md (Milestone 4); docs/marine_bem_benchmarks.md section 4.6",
        group="ursell",
        quantity="panel_damping_relative_error",
        parameters={"panels": 128},
        declared=1.3361e-2,
        tier="standard",
    ),
    DeclaredNumber(
        key="ursell_near_far_128",
        description="near-field/far-field damping closure at 128 panels (upper bound)",
        source="README.md (Milestone 4): 'closing to 6e-3'",
        group="ursell",
        quantity="near_far_field_relative_difference",
        parameters={"panels": 128},
        declared=6.0e-3,
        tier="standard",
        kind="upper",
    ),
    DeclaredNumber(
        key="ursell_mfs_added_mass_64",
        description=(
            "method-of-fundamental-solutions added-mass error, 64 panels "
            "(the pre-2026-11-03 table claimed 1.9e-4, which no source count "
            "reproduces; the source curve is converged from 16 sources up)"
        ),
        source="docs/marine_bem_benchmarks.md section 4.6 table (MFS row)",
        group="ursell",
        quantity="reference_solution_added_mass_relative_error",
        parameters={"panels": 64},
        declared=7.570e-4,
        tolerance=5.0e-2,
    ),
    DeclaredNumber(
        key="ursell_mfs_damping_64",
        description=(
            "method-of-fundamental-solutions damping error, 64 panels "
            "(the pre-2026-11-03 table claimed 8.8e-4; measured 3.526e-3)"
        ),
        source="docs/marine_bem_benchmarks.md section 4.6 table (MFS row)",
        group="ursell",
        quantity="reference_solution_damping_relative_error",
        parameters={"panels": 64},
        declared=3.526e-3,
        tolerance=5.0e-2,
    ),
    DeclaredNumber(
        key="ursell_added_mass_256",
        description="panel added-mass error at 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.6 table",
        group="ursell",
        quantity="panel_added_mass_relative_error",
        parameters={"panels": 256},
        declared=4.2930e-3,
        tier="heavy",
    ),
    DeclaredNumber(
        key="ursell_damping_256",
        description="panel damping error at 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.6 table",
        group="ursell",
        quantity="panel_damping_relative_error",
        parameters={"panels": 256},
        declared=6.8412e-3,
        tier="heavy",
    ),
    # --- 3-D interior solid harmonics (Laplace, exact polynomial truth) -----
    DeclaredNumber(
        key="interior_harmonic_deg1_flux_1280",
        description="flux relative L2 error, degree 1 (x), 1280 panels",
        source="docs/classical_benchmarks.md section 5.2 table 3",
        group="interior_harmonics",
        quantity="normal_derivative_relative_l2_error",
        parameters={"harmonic": "x", "subdivisions": 3},
        declared=3.7640e-3,
        tier="heavy",
    ),
    DeclaredNumber(
        key="interior_harmonic_deg2_flux_1280",
        description="flux relative L2 error, degree 2 (x2-y2), 1280 panels",
        source="docs/classical_benchmarks.md section 5.2 table 3",
        group="interior_harmonics",
        quantity="normal_derivative_relative_l2_error",
        parameters={"harmonic": "x2-y2", "subdivisions": 3},
        declared=7.3437e-3,
        tier="heavy",
    ),
    DeclaredNumber(
        key="interior_harmonic_deg3_flux_1280",
        description="flux relative L2 error, degree 3 (x3-3xy2), 1280 panels",
        source="docs/classical_benchmarks.md section 5.2 table 3",
        group="interior_harmonics",
        quantity="normal_derivative_relative_l2_error",
        parameters={"harmonic": "x3-3xy2", "subdivisions": 3},
        declared=1.1908e-2,
        tier="heavy",
    ),
    DeclaredNumber(
        key="interior_harmonic_deg1_field_rmse_1280",
        description="interior probe RMSE, degree 1 (x), 1280 panels",
        source="docs/classical_benchmarks.md section 5.2 table 3",
        group="interior_harmonics",
        quantity="field_rmse",
        parameters={"harmonic": "x", "subdivisions": 3},
        declared=2.293e-4,
        tier="heavy",
    ),
    DeclaredNumber(
        key="interior_harmonic_deg2_field_rmse_1280",
        description="interior probe RMSE, degree 2 (x2-y2), 1280 panels",
        source="docs/classical_benchmarks.md section 5.2 table 3",
        group="interior_harmonics",
        quantity="field_rmse",
        parameters={"harmonic": "x2-y2", "subdivisions": 3},
        declared=2.308e-4,
        tier="heavy",
    ),
    DeclaredNumber(
        key="interior_harmonic_deg3_field_rmse_1280",
        description="interior probe RMSE, degree 3 (x3-3xy2), 1280 panels",
        source="docs/classical_benchmarks.md section 5.2 table 3",
        group="interior_harmonics",
        quantity="field_rmse",
        parameters={"harmonic": "x3-3xy2", "subdivisions": 3},
        declared=1.824e-4,
        tier="standard",
    ),
) + _free_surface_3d_rows()

_URSELL_RADIUS = 1.0
_URSELL_SUBMERGENCE = 2.0
_URSELL_KA = 0.6
_URSELL_DENSITY = 1025.0
_URSELL_GRAVITY = 9.80665


def _run_group(group: str, parameters: dict) -> dict[str, float]:
    """Run one benchmark and return the quantities the registry may quote."""
    if group == "cylinder":
        row = cylinder_benchmark.run(int(parameters["panels"]))
        return {
            "added_mass_relative_error": float(row["added_mass_relative_error"]),
            "panels": float(row["panels"]),
        }
    if group == "ellipsoid":
        row = ellipsoid_benchmark.run(int(parameters["subdivisions"]))
        values = {"panels": float(row["panels"])}
        for axis_row in row["axes"]:
            values[
                f"added_mass_relative_error_{axis_row['motion_axis']}"
            ] = float(axis_row["added_mass_relative_error"])
        return values
    if group == "ursell":
        row = ursell_benchmark.run(panel_count=int(parameters["panels"]))
        keys = (
            "panel_added_mass_relative_error",
            "panel_damping_relative_error",
            "near_far_field_relative_difference",
            "reference_solution_added_mass_relative_error",
            "reference_solution_damping_relative_error",
            "panel_added_mass",
            "panel_damping",
        )
        return {key: float(row[key]) for key in keys}
    if group == "interior_harmonics":
        row = interior_harmonics_benchmark.run_case(
            str(parameters["harmonic"]), int(parameters["subdivisions"])
        )
        return {
            "normal_derivative_relative_l2_error": float(
                row["normal_derivative_relative_l2_error"]
            ),
            "field_rmse": float(row["field_rmse"]),
            "panels": float(row["panels"]),
        }
    if group == "ursell_reference":
        omega = math.sqrt(_URSELL_KA * _URSELL_GRAVITY / _URSELL_RADIUS)
        added_mass, damping = ursell_cylinder_added_mass(
            _URSELL_RADIUS,
            _URSELL_SUBMERGENCE,
            omega,
            _URSELL_DENSITY,
            gravity=_URSELL_GRAVITY,
        )
        return {
            "reference_added_mass": float(added_mass),
            "reference_damping": float(damping),
        }
    if group == "hulme3d":
        levels = hulme_benchmark.run_refinement_study(
            [int(value) for value in parameters["subdivisions"]],
            ka=float(parameters.get("ka", 1.0)),
        )["levels"]
        values: dict[str, float] = {}
        previous = None
        for level in levels:
            tag = f"level{int(level['subdivisions'])}"
            for key, value in level.items():
                if value is None or isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    values[f"{tag}_{key}"] = float(value)
            if previous is not None:
                # B11 is quoted in table 10 but not emitted by the benchmark.
                values[f"{tag}_observed_order_b11"] = math.log(
                    previous["surge_damping_rel_error"]
                    / level["surge_damping_rel_error"]
                ) / math.log(2.0)
            previous = level
        return values
    if group == "hulme3d_chief":
        rows = hulme_benchmark.run_irregular_frequency_study(
            subdivision=int(parameters["subdivision"]),
            ka_values=[float(ka) for ka in parameters["ka_values"]],
        )
        values = {}
        for row in rows:
            tag = f"ka{float(row['ka']):.2f}".replace(".", "p")
            values[f"{tag}_exact_a33_dimless"] = float(row["exact_a33"])
            values[f"{tag}_exact_b33_dimless"] = float(row["exact_b33"])
            for block in ("unregularized", "chief_regularized"):
                for key, value in row[block].items():
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    values[f"{tag}_{block}_{key}"] = float(value)
        return values
    if group == "wang_linton3d":
        levels = wang_linton_benchmark.run_refinement_study(
            [int(value) for value in parameters["subdivisions"]],
            ka=float(parameters.get("ka", 0.8)),
        )["levels"]
        values = {}
        for level in levels:
            tag = f"level{int(level['subdivisions'])}"
            for key, value in level.items():
                if value is None or isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    values[f"{tag}_{key}"] = float(value)
        return values
    raise ValueError(f"unknown benchmark group: {group!r}")


def _judge(entry: DeclaredNumber, measured: float) -> tuple[float, str]:
    if not math.isfinite(measured):
        return math.inf, "DRIFT"
    if entry.declared == 0.0:
        deviation = abs(measured)
        return deviation, "PASS" if deviation == 0.0 else "DRIFT"
    deviation = (measured - entry.declared) / abs(entry.declared)
    if entry.kind == "upper":
        return deviation, "PASS" if measured <= entry.declared else "DRIFT"
    if entry.kind != "value":
        raise ValueError(f"unknown comparison kind: {entry.kind!r}")
    return deviation, "PASS" if abs(deviation) <= entry.tolerance else "DRIFT"


def check(
    tiers: Sequence[str] | Iterable[str] = ("fast", "standard"),
    *,
    entries: Sequence[DeclaredNumber] | None = None,
    runner: Callable[[str, dict], dict[str, float]] | None = None,
) -> dict:
    """Recompute the registered numbers and judge them.

    Args:
        tiers: which cost classes to include, a subset of :data:`TIERS`.
        entries: override the registry (used by the tests).
        runner: override the benchmark dispatch (used by the tests).

    Returns:
        A JSON-serialisable report with one row per checked number.
    """
    selected_tiers = set(tiers)
    unknown = selected_tiers.difference(TIERS)
    if unknown:
        raise ValueError(f"unknown tiers: {sorted(unknown)}")
    registry = DECLARED_NUMBERS if entries is None else tuple(entries)
    dispatch = _run_group if runner is None else runner

    cache: dict[tuple[str, str], dict[str, float]] = {}
    rows: list[dict] = []
    for entry in registry:
        if entry.tier not in selected_tiers:
            continue
        cache_key = (entry.group, json.dumps(entry.parameters, sort_keys=True))
        if cache_key not in cache:
            cache[cache_key] = dispatch(entry.group, entry.parameters)
        measured = cache[cache_key][entry.quantity]
        deviation, verdict = _judge(entry, measured)
        rows.append(
            {
                "key": entry.key,
                "description": entry.description,
                "source": entry.source,
                "group": entry.group,
                "parameters": entry.parameters,
                "tier": entry.tier,
                "comparison": entry.kind,
                "declared": entry.declared,
                "measured": measured,
                "relative_deviation": deviation,
                "tolerance": entry.tolerance,
                "verdict": verdict,
            }
        )

    drifted = [row for row in rows if row["verdict"] != "PASS"]
    worst = max((abs(row["relative_deviation"]) for row in rows), default=0.0)
    return {
        "benchmark": "declared_numbers_recomputation",
        "tiers": sorted(selected_tiers),
        "checks": rows,
        "summary": {
            "checked": len(rows),
            "passed": len(rows) - len(drifted),
            "drifted": len(drifted),
            "drifted_keys": [row["key"] for row in drifted],
            "worst_relative_deviation": worst,
            "verdict": "PASS" if not drifted else "DRIFT",
        },
        "environment": _environment(),
    }


def _environment() -> dict[str, str]:
    """Identify the environment, because a drift claim must be reproducible."""
    import numpy

    info = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "platform": platform.platform(),
    }
    try:  # best effort: a checkout without git history must not fail the check
        info["commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - environment dependent
        info["commit"] = "unknown"
    return info


def richardson_limit(values: Sequence[float], panels: Sequence[int]) -> tuple[float, float]:
    """Extrapolate a first-order family to zero mesh size.

    The constant-panel indirect formulation converges like ``C / N`` on this
    problem, so the limit of ``v(N)`` for ``N -> infinity`` follows from the two
    finest members. The observed order is measured, not assumed: it comes from
    the two finest successive differences, which is why three panel counts are
    needed to get a trustworthy pair.

    Returns:
        ``(limit, observed_order)``.
    """
    if len(values) != len(panels) or len(values) < 3:
        raise ValueError("at least three matched (panel count, value) pairs are required")
    finest, middle = values[-1], values[-2]
    coarser = values[-3]
    step = middle - coarser
    if step == 0.0 or finest == middle:
        raise ValueError("the family is not converging; extrapolation is meaningless")
    refinement = panels[-1] / panels[-2]
    order = math.log(abs(step) / abs(finest - middle)) / math.log(refinement)
    if order <= 0.0:
        raise ValueError(f"observed order {order:.3f} is not positive")
    denominator = refinement**order - 1.0
    return finest + (finest - middle) / denominator, order


def ursell_refinement(panel_counts: Sequence[int] = (64, 128, 256)) -> dict:
    """Refinement study of the Ursell panel solution, with extrapolation.

    This is the evidence behind the numbers quoted for the Ursell benchmark and
    behind the claim that the reference solution is correct: a boundary-element
    family that shares no code with the multipole solution, pushed to zero mesh
    size, must reproduce it. The difference between the extrapolated limit and
    the reference is the accuracy the reference can be trusted to.
    """
    counts = tuple(int(count) for count in panel_counts)
    if len(counts) < 3:
        raise ValueError("at least three panel counts are required")
    rows = []
    for count in counts:
        row = ursell_benchmark.run(panel_count=count)
        rows.append(
            {
                "panels": count,
                "panel_added_mass": float(row["panel_added_mass"]),
                "panel_damping": float(row["panel_damping"]),
                "added_mass_relative_error": float(row["panel_added_mass_relative_error"]),
                "damping_relative_error": float(row["panel_damping_relative_error"]),
                "near_far_field_relative_difference": float(
                    row["near_far_field_relative_difference"]
                ),
            }
        )
    reference = _run_group("ursell_reference", {})
    added_mass_limit, added_mass_order = richardson_limit(
        [row["panel_added_mass"] for row in rows], counts
    )
    damping_limit, damping_order = richardson_limit(
        [row["panel_damping"] for row in rows], counts
    )
    return {
        "benchmark": "ursell_refinement_and_extrapolation",
        "panel_counts": list(counts),
        "rows": rows,
        "reference_added_mass": reference["reference_added_mass"],
        "reference_damping": reference["reference_damping"],
        "extrapolated_added_mass": added_mass_limit,
        "extrapolated_damping": damping_limit,
        "observed_order_added_mass": added_mass_order,
        "observed_order_damping": damping_order,
        "extrapolation_vs_reference_added_mass": abs(
            added_mass_limit - reference["reference_added_mass"]
        )
        / abs(reference["reference_added_mass"]),
        "extrapolation_vs_reference_damping": abs(
            damping_limit - reference["reference_damping"]
        )
        / abs(reference["reference_damping"]),
        "note": (
            "The extrapolated limit comes from the three finest panel solutions; "
            "the reference is Ursell's multipole solution, which shares no code "
            "with the boundary-element assembly. Their agreement measures how "
            "far the reference can be trusted."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiers",
        default="fast,standard",
        help=f"comma-separated subset of {', '.join(TIERS)} (default: fast,standard)",
    )
    parser.add_argument(
        "--refinement",
        action="store_true",
        help="also run the Ursell refinement study and its extrapolation",
    )
    parser.add_argument("--output", type=Path, help="write the JSON report here")
    parser.add_argument(
        "--strict", action="store_true", help="exit non-zero when a number drifted"
    )
    args = parser.parse_args(argv)

    tiers = [tier.strip() for tier in args.tiers.split(",") if tier.strip()]
    report = check(tiers)
    if args.refinement:
        report["refinement"] = ursell_refinement()
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.strict and report["summary"]["verdict"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())