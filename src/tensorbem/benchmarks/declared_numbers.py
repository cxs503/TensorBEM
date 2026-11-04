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
    interior_harmonics as interior_harmonics_benchmark,
    floating_hemisphere as floating_hemisphere_benchmark,
    submerged_sphere_wave as submerged_sphere_benchmark,
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
    # --- 3-D free-surface: Hulme (1982) floating hemisphere, table 10 -----
    DeclaredNumber(
        key="hulme_heave_added_mass_error_256",
        description="A33' relative error of the floating hemisphere, Ka=1, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10 (section 4.10)",
        group="hulme_hemisphere",
        quantity="heave_added_mass_rel_error",
        parameters={"subdivisions": [2, 3], "ka": 1.0},
        declared=7.41e-3,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_heave_damping_error_256",
        description="B33' relative error of the floating hemisphere, Ka=1, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10 (section 4.10)",
        group="hulme_hemisphere",
        quantity="heave_damping_rel_error",
        parameters={"subdivisions": [2, 3], "ka": 1.0},
        declared=1.18e-2,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_surge_added_mass_error_256",
        description="A11' relative error of the floating hemisphere, Ka=1, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10 (section 4.10)",
        group="hulme_hemisphere",
        quantity="surge_added_mass_rel_error",
        parameters={"subdivisions": [2, 3], "ka": 1.0},
        declared=1.30e-2,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_surge_damping_error_256",
        description="B11' relative error of the floating hemisphere (slowest quantity), Ka=1, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10 (section 4.10)",
        group="hulme_hemisphere",
        quantity="surge_damping_rel_error",
        parameters={"subdivisions": [2, 3], "ka": 1.0},
        declared=3.53e-2,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_vertical_force_error_256",
        description="vertical excitation force relative error of the floating hemisphere, Ka=1, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10 (section 4.10)",
        group="hulme_hemisphere",
        quantity="excitation_force_z_rel_error",
        parameters={"subdivisions": [2, 3], "ka": 1.0},
        declared=5.67e-3,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_condition_number_256",
        description="condition number of the CHIEF-augmented 256-panel hemisphere system",
        source="docs/marine_bem_benchmarks.md table 10 (section 4.10)",
        group="hulme_hemisphere",
        quantity="condition_number",
        parameters={"subdivisions": [2, 3], "ka": 1.0},
        declared=2.783,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_order_a33_256",
        description="observed order of A33' between 64 and 256 panels",
        source="docs/marine_bem_benchmarks.md table 10 (section 4.10)",
        group="hulme_hemisphere",
        quantity="observed_order_a33",
        parameters={"subdivisions": [2, 3], "ka": 1.0},
        declared=2.06,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_order_b33_256",
        description="observed order of B33' between 64 and 256 panels",
        source="docs/marine_bem_benchmarks.md table 10 (section 4.10)",
        group="hulme_hemisphere",
        quantity="observed_order_b33",
        parameters={"subdivisions": [2, 3], "ka": 1.0},
        declared=1.98,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_order_a11_256",
        description="observed order of A11' between 64 and 256 panels",
        source="docs/marine_bem_benchmarks.md table 10 (section 4.10)",
        group="hulme_hemisphere",
        quantity="observed_order_a11",
        parameters={"subdivisions": [2, 3], "ka": 1.0},
        declared=2.01,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_order_b11_256",
        description="observed order of B11' between 64 and 256 panels",
        source="docs/marine_bem_benchmarks.md table 10 (section 4.10)",
        group="hulme_hemisphere",
        quantity="observed_order_b11",
        parameters={"subdivisions": [2, 3], "ka": 1.0},
        # Table 10 used to quote 1.88 for this cell: that is the same ratio
        # evaluated with the nominal refinement factor 2 instead of the mesh's
        # actual h_mean ratio 1.9642. The errors it is derived from (1.30e-1 and
        # 3.53e-2) have always been reproduced; only the derived order was off.
        declared=1.94,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_order_fz_256",
        description="observed order of the vertical excitation force between 64 and 256 panels",
        source="docs/marine_bem_benchmarks.md table 10 (section 4.10)",
        group="hulme_hemisphere",
        quantity="observed_order_fz",
        parameters={"subdivisions": [2, 3], "ka": 1.0},
        declared=2.02,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_near_far_damping_256",
        description="relative difference between near-field and 3-D Kochin far-field heave damping, 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 (text after table 10)",
        group="hulme_hemisphere",
        quantity="far_field_damping_rel_diff_heave",
        parameters={"subdivisions": [2, 3], "ka": 1.0},
        declared=5.45e-4,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_haskind_surge_256",
        description="direct-diffraction versus boundary Haskind discrepancy, surge, 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 (text after table 10)",
        group="hulme_hemisphere",
        quantity="haskind_reciprocity_rel_diff_x",
        parameters={"subdivisions": [2, 3], "ka": 1.0},
        declared=2.09e-4,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_haskind_heave_256",
        description="direct-diffraction versus boundary Haskind discrepancy, heave, 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 (text after table 10)",
        group="hulme_hemisphere",
        quantity="haskind_reciprocity_rel_diff_z",
        parameters={"subdivisions": [2, 3], "ka": 1.0},
        declared=8.09e-4,
        tier="standard",
    ),
    # --- 3-D free-surface: CHIEF suppression at the irregular frequency ---
    # table 11, Ka=2.58 (first axisymmetric interior waterline eigenvalue)
    DeclaredNumber(
        key="hulme_irregular_unregularized_condition",
        description="condition number of the un-augmented 256-panel BIE at Ka=2.58",
        source="docs/marine_bem_benchmarks.md table 11 (section 4.10)",
        group="hulme_irregular",
        quantity="condition_number",
        parameters={"subdivisions": 3, "ka": 2.58},
        declared=145.66,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_irregular_unregularized_added_mass",
        description="unregularized A33' at Ka=2.58 (exact 0.4007)",
        source="docs/marine_bem_benchmarks.md table 11 (section 4.10)",
        group="hulme_irregular",
        quantity="added_mass_dimless",
        parameters={"subdivisions": 3, "ka": 2.58},
        declared=0.3481,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_irregular_unregularized_damping_magnitude",
        description="magnitude of the non-physical negative B33' at Ka=2.58 (sign is -1)",
        source="docs/marine_bem_benchmarks.md table 11 (section 4.10)",
        group="hulme_irregular",
        quantity="damping_magnitude",
        parameters={"subdivisions": 3, "ka": 2.58},
        declared=0.1042,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_irregular_passivity_violated",
        description="1 when the un-augmented BIE returns B33' < 0 at Ka=2.58 (non-physical damping)",
        source="docs/marine_bem_benchmarks.md table 11 (section 4.10)",
        group="hulme_irregular",
        quantity="passivity_violated",
        parameters={"subdivisions": 3, "ka": 2.58},
        declared=1.0,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_irregular_chief_condition",
        description="condition number after 5 interior CHIEF points at Ka=2.58",
        source="docs/marine_bem_benchmarks.md table 11 (section 4.10)",
        group="hulme_irregular",
        quantity="chief_condition_number",
        parameters={"subdivisions": 3, "ka": 2.58},
        declared=4.24,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_irregular_chief_added_mass",
        description="CHIEF-regularized A33' at Ka=2.58 (exact 0.4007)",
        source="docs/marine_bem_benchmarks.md table 11 (section 4.10)",
        group="hulme_irregular",
        quantity="chief_added_mass_dimless",
        parameters={"subdivisions": 3, "ka": 2.58},
        declared=0.3919,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_irregular_chief_damping",
        description="CHIEF-regularized B33' at Ka=2.58 (exact 0.0631); must stay positive",
        source="docs/marine_bem_benchmarks.md table 11 (section 4.10)",
        group="hulme_irregular",
        quantity="chief_damping_dimless",
        parameters={"subdivisions": 3, "ka": 2.58},
        declared=0.0687,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_irregular_exact_added_mass",
        description="Hulme semi-analytical A33' at Ka=2.58 quoted in table 11",
        source="docs/marine_bem_benchmarks.md table 11 (section 4.10)",
        group="hulme_irregular",
        quantity="exact_added_mass_dimless",
        parameters={"subdivisions": 3, "ka": 2.58},
        declared=0.4007,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_irregular_exact_damping",
        description="Hulme semi-analytical B33' at Ka=2.58 quoted in table 11",
        source="docs/marine_bem_benchmarks.md table 11 (section 4.10)",
        group="hulme_irregular",
        quantity="exact_damping_dimless",
        parameters={"subdivisions": 3, "ka": 2.58},
        declared=0.0631,
        tier="standard",
    ),
    # --- 3-D free-surface: Wang/Linton submerged sphere, table 12 ----------
    # f/a=1.5, Ka=0.8, 320 panels
    DeclaredNumber(
        key="submerged_sphere_heave_added_mass_error_320",
        description="A33' relative error of the submerged sphere, f/a=1.5, Ka=0.8, 320 panels",
        source="docs/marine_bem_benchmarks.md table 12 (section 4.11)",
        group="submerged_sphere",
        quantity="heave_added_mass_rel_error",
        parameters={"subdivisions": [1, 2], "ka": 0.8},
        declared=1.85e-2,
        tier="standard",
    ),
    DeclaredNumber(
        key="submerged_sphere_surge_added_mass_error_320",
        description="A11' relative error of the submerged sphere, f/a=1.5, Ka=0.8, 320 panels",
        source="docs/marine_bem_benchmarks.md table 12 (section 4.11)",
        group="submerged_sphere",
        quantity="surge_added_mass_rel_error",
        parameters={"subdivisions": [1, 2], "ka": 0.8},
        declared=2.08e-2,
        tier="standard",
    ),
    DeclaredNumber(
        key="submerged_sphere_horizontal_force_error_320",
        description="horizontal excitation force relative error, submerged sphere, 320 panels",
        source="docs/marine_bem_benchmarks.md table 12 (section 4.11)",
        group="submerged_sphere",
        quantity="excitation_force_x_rel_error",
        parameters={"subdivisions": [1, 2], "ka": 0.8},
        declared=3.10e-2,
        tier="standard",
    ),
    DeclaredNumber(
        key="submerged_sphere_vertical_force_error_320",
        description="vertical excitation force relative error, submerged sphere, 320 panels",
        source="docs/marine_bem_benchmarks.md table 12 (section 4.11)",
        group="submerged_sphere",
        quantity="excitation_force_z_rel_error",
        parameters={"subdivisions": [1, 2], "ka": 0.8},
        declared=3.20e-2,
        tier="standard",
    ),
    DeclaredNumber(
        key="submerged_sphere_condition_number_320",
        description="condition number of the 320-panel submerged-sphere system",
        source="docs/marine_bem_benchmarks.md table 12 (section 4.11)",
        group="submerged_sphere",
        quantity="condition_number",
        parameters={"subdivisions": [1, 2], "ka": 0.8},
        declared=2.188,
        tier="standard",
    ),
    DeclaredNumber(
        key="submerged_sphere_rotational_coupling_320",
        description="spherical-centre rotational coupling ratio (quoted 2.6e-5, 2 significant digits)",
        source="docs/marine_bem_benchmarks.md table 12 (section 4.11)",
        group="submerged_sphere",
        quantity="rotational_added_mass_max_ratio",
        parameters={"subdivisions": [1, 2], "ka": 0.8},
        declared=2.6e-5,
        tolerance=5.0e-2,
        tier="standard",
    ),
    DeclaredNumber(
        key="submerged_sphere_order_a33_320",
        description="observed order of A33' between 80 and 320 panels",
        source="docs/marine_bem_benchmarks.md table 12 (section 4.11)",
        group="submerged_sphere",
        quantity="observed_order_a33",
        parameters={"subdivisions": [1, 2], "ka": 0.8},
        declared=2.06,
        tier="standard",
    ),
    DeclaredNumber(
        key="submerged_sphere_order_b33_320",
        description="observed order of B33' between 80 and 320 panels",
        source="docs/marine_bem_benchmarks.md table 12 (section 4.11)",
        group="submerged_sphere",
        quantity="observed_order_b33",
        parameters={"subdivisions": [1, 2], "ka": 0.8},
        declared=1.93,
        tier="standard",
    ),
    DeclaredNumber(
        key="submerged_sphere_order_a11_320",
        description="observed order of A11' between 80 and 320 panels",
        source="docs/marine_bem_benchmarks.md table 12 (section 4.11)",
        group="submerged_sphere",
        quantity="observed_order_a11",
        parameters={"subdivisions": [1, 2], "ka": 0.8},
        declared=2.03,
        tier="standard",
    ),
    DeclaredNumber(
        key="submerged_sphere_order_fz_320",
        description="observed order of the vertical excitation force between 80 and 320 panels",
        source="docs/marine_bem_benchmarks.md table 12 (section 4.11)",
        group="submerged_sphere",
        quantity="observed_order_fz",
        parameters={"subdivisions": [1, 2], "ka": 0.8},
        declared=1.98,
        tier="standard",
    ),
)

_URSELL_RADIUS = 1.0
_URSELL_SUBMERGENCE = 2.0
_URSELL_KA = 0.6
_URSELL_DENSITY = 1025.0
_URSELL_GRAVITY = 9.80665


def _finest_level(
    study: dict, extra_orders: dict[str, str] | None = None
) -> dict[str, float]:
    """Flatten the finest level of a refinement study into scalar quantities.

    The three-dimensional free-surface studies return a list of levels; the
    documentation quotes the finest one plus its observed order of convergence.
    Orders are recomputed here for quantities the study itself does not order
    (``extra_orders`` maps a registry name to the error key it is derived from),
    using the same ``log(e0/e1)/log(h0/h1)`` estimator.
    """
    levels = study["levels"]
    finest = levels[-1]
    values: dict[str, float] = {}
    for key, value in finest.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        values[key] = float(value)
    values["panel_count"] = float(finest["panel_count"])
    if extra_orders and len(levels) >= 2:
        previous = levels[-2]
        h0, h1 = float(previous["h_mean"]), float(finest["h_mean"])
        for name, error_key in extra_orders.items():
            e0, e1 = float(previous[error_key]), float(finest[error_key])
            if e0 > 0.0 and e1 > 0.0 and h0 > h1:
                values[name] = float(math.log(e0 / e1) / math.log(h0 / h1))
    return values


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
    if group == "hulme_hemisphere":
        study = floating_hemisphere_benchmark.run_refinement_study(
            subdivisions=[int(level) for level in parameters["subdivisions"]],
            ka=float(parameters.get("ka", 1.0)),
        )
        return _finest_level(study, extra_orders={"observed_order_b11": "surge_damping_rel_error"})
    if group == "submerged_sphere":
        study = submerged_sphere_benchmark.run_refinement_study(
            subdivisions=[int(level) for level in parameters["subdivisions"]],
            ka=float(parameters.get("ka", 0.8)),
            submergence=float(parameters.get("submergence", 1.5)),
        )
        return _finest_level(study)
    if group == "hulme_irregular":
        rows = floating_hemisphere_benchmark.run_irregular_frequency_study(
            subdivision=int(parameters["subdivisions"]),
            ka_values=(float(parameters["ka"]),),
        )
        row = rows[0]
        unreg = row["unregularized"]
        chief = row["chief_regularized"]
        return {
            "condition_number": float(unreg["condition_number"]),
            "added_mass_dimless": float(unreg["a33_dimless"]),
            "damping_dimless": float(unreg["b33_dimless"]),
            # Registered as a magnitude: the registry only quotes positive
            # numbers, and the sign is the whole point of the claim anyway.
            "damping_magnitude": abs(float(unreg["b33_dimless"])),
            "passivity_violated": 1.0 if float(unreg["b33_dimless"]) < 0.0 else 0.0,
            "chief_condition_number": float(chief["condition_number"]),
            "chief_added_mass_dimless": float(chief["a33_dimless"]),
            "chief_damping_dimless": float(chief["b33_dimless"]),
            "chief_point_count": float(chief["chief_point_count"]),
            "exact_added_mass_dimless": float(row["exact_a33"]),
            "exact_damping_dimless": float(row["exact_b33"]),
        }
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