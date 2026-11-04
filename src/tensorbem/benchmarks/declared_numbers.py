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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from tensorbem.analytic import ursell_cylinder_added_mass
from tensorbem.benchmarks import (
    cylinder_added_mass as cylinder_benchmark,
    ellipsoid_added_mass as ellipsoid_benchmark,
    floating_hemisphere as hulme_benchmark,
    interior_harmonics as interior_harmonics_benchmark,
    submerged_sphere_wave as linton_benchmark,
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
    """Which benchmark to run: ``cylinder``, ``ellipsoid``, ``ursell``,
    ``ursell_reference``, ``interior_harmonics``, ``hulme``, ``hulme_chief``
    or ``submerged_sphere``."""
    quantity: str
    """Key into the dictionary returned by the group's runner."""
    declared: float
    tier: str = "fast"
    parameters: dict = field(default_factory=dict)
    tolerance: float = DEFAULT_TOLERANCE
    kind: str = "value"
    """``value`` compares for equality, ``upper`` only checks ``measured <=
    declared`` (for sentences of the form "within 6e-3")."""


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
    # --- 3-D deep-water free surface: Hulme (1982) floating hemisphere ------
    # Table 10 of docs/marine_bem_benchmarks.md.  One refinement ladder
    # (subdivisions 1, 2, 3 = 16, 64, 256 panels) supplies every row below, so
    # the whole table costs a single benchmark run.
    DeclaredNumber(
        key="hulme_condition_number_16",
        description="condition number of the exterior BIE, 16 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="condition_number@1",
        tier="standard",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=3.893,
    ),
    DeclaredNumber(
        key="hulme_heave_added_mass_16",
        description="heave added-mass relative error, 16 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="heave_added_mass_rel_error@1",
        tier="standard",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=1.20e-1,
    ),
    DeclaredNumber(
        key="hulme_heave_damping_16",
        description="heave damping relative error, 16 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="heave_damping_rel_error@1",
        tier="standard",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=1.22e-1,
    ),
    DeclaredNumber(
        key="hulme_surge_added_mass_16",
        description="surge added-mass relative error, 16 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="surge_added_mass_rel_error@1",
        tier="standard",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=1.80e-1,
    ),
    DeclaredNumber(
        key="hulme_surge_damping_16",
        description="surge damping relative error, 16 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="surge_damping_rel_error@1",
        tier="standard",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=4.02e-1,
    ),
    DeclaredNumber(
        key="hulme_excitation_z_16",
        description="vertical wave-force relative error, 16 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="excitation_force_z_rel_error@1",
        tier="standard",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=6.67e-2,
    ),
    DeclaredNumber(
        key="hulme_condition_number_64",
        description="condition number of the exterior BIE, 64 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="condition_number@2",
        tier="standard",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=3.052,
    ),
    DeclaredNumber(
        key="hulme_heave_added_mass_64",
        description="heave added-mass relative error, 64 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="heave_added_mass_rel_error@2",
        tier="standard",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=2.97e-2,
    ),
    DeclaredNumber(
        key="hulme_condition_number_256",
        description="condition number of the exterior BIE, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="condition_number@3",
        tier="standard",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=2.783,
    ),
    DeclaredNumber(
        key="hulme_heave_added_mass_256",
        description="heave added-mass relative error, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="heave_added_mass_rel_error@3",
        tier="standard",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=7.41e-3,
    ),
    DeclaredNumber(
        key="hulme_heave_damping_256",
        description="heave damping relative error, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="heave_damping_rel_error@3",
        tier="standard",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=1.18e-2,
    ),
    DeclaredNumber(
        key="hulme_surge_added_mass_256",
        description="surge added-mass relative error, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="surge_added_mass_rel_error@3",
        tier="standard",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=1.30e-2,
    ),
    DeclaredNumber(
        key="hulme_surge_damping_256",
        description="surge damping relative error, 256 panels (slowest quantity)",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="surge_damping_rel_error@3",
        tier="standard",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=3.53e-2,
    ),
    DeclaredNumber(
        key="hulme_excitation_z_256",
        description="vertical wave-force relative error, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="excitation_force_z_rel_error@3",
        tier="standard",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=5.67e-3,
    ),
    DeclaredNumber(
        key="hulme_order_heave_added_mass_256",
        description="observed order in h, heave added mass, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="observed_order_a33@3",
        tier="standard",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=2.06,
    ),
    DeclaredNumber(
        key="hulme_order_heave_damping_256",
        description="observed order in h, heave damping, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="observed_order_b33@3",
        tier="standard",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=1.98,
    ),
    DeclaredNumber(
        key="hulme_order_surge_added_mass_256",
        description="observed order in h, surge added mass, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="observed_order_a11@3",
        tier="standard",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=2.01,
    ),
    DeclaredNumber(
        key="hulme_order_excitation_z_256",
        description="observed order in h, vertical wave force, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="observed_order_fz@3",
        tier="standard",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=2.02,
    ),
    DeclaredNumber(
        key="hulme_far_field_damping_256",
        description="near-field vs far-field (Kochin) heave damping, 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 prose",
        group="hulme",
        quantity="far_field_damping_rel_diff_heave@3",
        tier="standard",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=5.45e-4,
    ),
    DeclaredNumber(
        key="hulme_haskind_surge_256",
        description="direct diffraction vs boundary Haskind, surge, 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 prose",
        group="hulme",
        quantity="haskind_reciprocity_rel_diff_x@3",
        tier="standard",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=2.09e-4,
    ),
    DeclaredNumber(
        key="hulme_haskind_heave_256",
        description="direct diffraction vs boundary Haskind, heave, 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 prose",
        group="hulme",
        quantity="haskind_reciprocity_rel_diff_z@3",
        tier="standard",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=8.09e-4,
    ),
    # --- Table 11: 3-D CHIEF regularisation at the interior waterline mode --
    DeclaredNumber(
        key="hulme_chief_condition_unregularized",
        description="condition number without interior CHIEF points, Ka=2.58",
        source="docs/marine_bem_benchmarks.md table 11",
        group="hulme_chief",
        quantity="unregularized_condition_number",
        tier="standard",
        parameters={"subdivision": 3, "ka": 2.58},
        declared=145.66,
    ),
    DeclaredNumber(
        key="hulme_chief_heave_added_mass_unregularized",
        description="A33' without CHIEF, Ka=2.58 (exact 0.4007)",
        source="docs/marine_bem_benchmarks.md table 11",
        group="hulme_chief",
        quantity="unregularized_a33_dimless",
        tier="standard",
        parameters={"subdivision": 3, "ka": 2.58},
        declared=0.3481,
    ),
    DeclaredNumber(
        key="hulme_chief_heave_damping_unregularized",
        description="B33' without CHIEF, Ka=2.58 (negative: passivity broken)",
        source="docs/marine_bem_benchmarks.md table 11",
        group="hulme_chief",
        quantity="unregularized_b33_dimless",
        tier="standard",
        parameters={"subdivision": 3, "ka": 2.58},
        declared=-0.1042,
    ),
    DeclaredNumber(
        key="hulme_chief_condition_augmented",
        description="condition number with 3-D interior CHIEF points, Ka=2.58",
        source="docs/marine_bem_benchmarks.md table 11",
        group="hulme_chief",
        quantity="chief_regularized_condition_number",
        tier="standard",
        parameters={"subdivision": 3, "ka": 2.58},
        declared=4.24,
    ),
    DeclaredNumber(
        key="hulme_chief_heave_added_mass_augmented",
        description="A33' with CHIEF, Ka=2.58 (exact 0.4007)",
        source="docs/marine_bem_benchmarks.md table 11",
        group="hulme_chief",
        quantity="chief_regularized_a33_dimless",
        tier="standard",
        parameters={"subdivision": 3, "ka": 2.58},
        declared=0.3919,
    ),
    DeclaredNumber(
        key="hulme_chief_heave_damping_augmented",
        description="B33' with CHIEF, Ka=2.58 (positive: passivity restored)",
        source="docs/marine_bem_benchmarks.md table 11",
        group="hulme_chief",
        quantity="chief_regularized_b33_dimless",
        tier="standard",
        parameters={"subdivision": 3, "ka": 2.58},
        declared=0.0687,
    ),
    DeclaredNumber(
        key="hulme_chief_point_count",
        description="number of automatic interior CHIEF points",
        source="docs/marine_bem_benchmarks.md table 11",
        group="hulme_chief",
        quantity="chief_regularized_chief_point_count",
        tier="standard",
        parameters={"subdivision": 3, "ka": 2.58},
        declared=5.0,
        tolerance=0.0,
    ),
    # --- Table 12: Wang/Linton submerged sphere (zero-quadrature multipoles) -
    DeclaredNumber(
        key="linton_condition_number_20",
        description="condition number of the exterior BIE, 20 panels",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="condition_number@0",
        tier="standard",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=2.407,
    ),
    DeclaredNumber(
        key="linton_heave_added_mass_20",
        description="heave added-mass relative error, 20 panels",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="heave_added_mass_rel_error@0",
        tier="standard",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=2.55e-1,
    ),
    DeclaredNumber(
        key="linton_surge_added_mass_20",
        description="surge added-mass relative error, 20 panels",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="surge_added_mass_rel_error@0",
        tier="standard",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=2.68e-1,
    ),
    DeclaredNumber(
        key="linton_excitation_x_20",
        description="horizontal wave-force relative error, 20 panels",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="excitation_force_x_rel_error@0",
        tier="standard",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=3.60e-1,
    ),
    DeclaredNumber(
        key="linton_excitation_z_20",
        description="vertical wave-force relative error, 20 panels",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="excitation_force_z_rel_error@0",
        tier="standard",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=3.70e-1,
    ),
    DeclaredNumber(
        key="linton_rotational_coupling_80",
        description="row/column-max rotational added-mass ratio, 80 panels "
        "(m=0/m=1 decoupling)",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="rotational_added_mass_max_ratio@1",
        tier="standard",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=1.0e-3,
        kind="upper",
    ),
    DeclaredNumber(
        key="linton_condition_number_320",
        description="condition number of the exterior BIE, 320 panels",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="condition_number@2",
        tier="standard",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=2.188,
    ),
    DeclaredNumber(
        key="linton_heave_added_mass_320",
        description="heave added-mass relative error, 320 panels",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="heave_added_mass_rel_error@2",
        tier="standard",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=1.85e-2,
    ),
    DeclaredNumber(
        key="linton_surge_added_mass_320",
        description="surge added-mass relative error, 320 panels",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="surge_added_mass_rel_error@2",
        tier="standard",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=2.08e-2,
    ),
    DeclaredNumber(
        key="linton_excitation_x_320",
        description="horizontal wave-force relative error, 320 panels",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="excitation_force_x_rel_error@2",
        tier="standard",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=3.10e-2,
    ),
    DeclaredNumber(
        key="linton_excitation_z_320",
        description="vertical wave-force relative error, 320 panels",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="excitation_force_z_rel_error@2",
        tier="standard",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=3.20e-2,
    ),
    DeclaredNumber(
        key="linton_rotational_coupling_320",
        description="row/column-max rotational added-mass ratio, 320 panels "
        "(m=0/m=1 decoupling)",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="rotational_added_mass_max_ratio@2",
        tier="standard",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=1.0e-4,
        kind="upper",
    ),
    DeclaredNumber(
        key="linton_order_heave_added_mass_320",
        description="observed order in h, heave added mass, 320 panels",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="observed_order_a33@2",
        tier="standard",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=2.06,
    ),
    DeclaredNumber(
        key="linton_order_heave_damping_320",
        description="observed order in h, heave damping, 320 panels",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="observed_order_b33@2",
        tier="standard",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=1.93,
    ),
    DeclaredNumber(
        key="linton_order_surge_added_mass_320",
        description="observed order in h, surge added mass, 320 panels",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="observed_order_a11@2",
        tier="standard",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=2.03,
    ),
    DeclaredNumber(
        key="linton_order_excitation_z_320",
        description="observed order in h, vertical wave force, 320 panels",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="observed_order_fz@2",
        tier="standard",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=1.98,
    ),
)

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
    if group in {"hulme", "submerged_sphere"}:
        module = hulme_benchmark if group == "hulme" else linton_benchmark
        study = module.run_refinement_study(
            subdivisions=[int(level) for level in parameters["subdivisions"]],
            ka=float(parameters["ka"]),
        )
        values: dict[str, float] = {}
        for row in study["levels"]:
            tag = int(row["subdivisions"])
            values[f"panel_count@{tag}"] = float(row["panel_count"])
            for name, value in row.items():
                if isinstance(value, bool) or value is None:
                    continue
                if not isinstance(value, (int, float)):
                    continue
                values[f"{name}@{tag}"] = float(value)
        return values
    if group == "hulme_chief":
        rows = hulme_benchmark.run_irregular_frequency_study(
            subdivision=int(parameters["subdivision"]),
            ka_values=(float(parameters["ka"]),),
        )
        row = rows[0]
        values = {
            "exact_a33": float(row["exact_a33"]),
            "exact_b33": float(row["exact_b33"]),
        }
        for label in ("unregularized", "chief_regularized"):
            block = row[label]
            for name in ("condition_number", "a33_dimless", "b33_dimless"):
                values[f"{label}_{name}"] = float(block[name])
            values[f"{label}_chief_point_count"] = float(
                block.get("chief_point_count", 0)
            )
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