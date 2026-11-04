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

Provenance note (2026-11-05).  The 3-D free-surface tables of
``docs/marine_bem_benchmarks.md`` (§4.10–§4.11) were the last documented
benchmark family without an executable registry: the 2026-11-04 heavy-tier
settlement (m3059) left them open because the *full* benchmark suites
(frequency sweeps included) cost minutes.  The tables themselves quote only
the refinement studies, which cost ~15 s in total, so the entries live in
the ``standard`` tier and run in the default check.  Every quoted number is
a pure discretisation error against the Hulme (1982) wave-free-potential
reference or the zero-quadrature Thorne multipole expansion, a condition
number, or a conservation residual.
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
    floating_hemisphere as floating_hemisphere_benchmark,
    interior_harmonics as interior_harmonics_benchmark,
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
    """Which benchmark to run: ``cylinder``, ``ellipsoid``, ``ursell``,
    ``ursell_reference``, ``interior_harmonics``, ``hulme_hemisphere``,
    ``hulme_chief`` or ``submerged_sphere``."""
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
# 3-D free-surface tables (Hulme 1982; Wang 1986 / Linton 1991)
# --------------------------------------------------------------------------
#
# The numbers below are the ones a reader of §4.10–§4.11 will find in tables
# 10–12 and the adjacent prose.  They were documentation-only until 2026-11-05
# (m3631): the 2026-11-04 heavy-tier settlement (m3059) closed the 2-D and
# unbounded-flow families and explicitly deferred this one.  The truth values
# are the Hulme (1982) wave-free-potential reference and the zero-quadrature
# Thorne multipole expansion, so every quoted error is a pure discretisation
# error and can be pinned at printed precision.

_HULME_TABLE10 = (
    # (subdivisions, panels, A33', B33', A11', B11', Fz, condition number)
    (1, 16, 1.20e-1, 1.22e-1, 1.80e-1, 4.02e-1, 6.67e-2, 3.893),
    (2, 64, 2.97e-2, 4.50e-2, 5.05e-2, 1.30e-1, 2.21e-2, 3.052),
    (3, 256, 7.41e-3, 1.18e-2, 1.30e-2, 3.53e-2, 5.67e-3, 2.783),
)

#: Observed orders printed in table 10, per subdivision level:
#: (A33', B33', A11', B11', Fz).
_HULME_TABLE10_ORDERS = {
    2: (2.23, 1.59, 2.04, 1.62, 1.77),
    3: (2.06, 1.98, 2.01, 1.88, 2.02),
}

_HULME_TABLE10_PARAMS = {"subdivisions": [1, 2, 3], "ka": 1.0}
_HULME_CHIEF_PARAMS = {"subdivision": 3, "ka": 2.58}

_SPHERE_TABLE12 = (
    # (subdivisions, panels, A33', A11', Fx, Fz, rotational coupling, cond)
    (0, 20, 2.55e-1, 2.68e-1, 3.60e-1, 3.70e-1, 2.5e-32, 2.407),
    (1, 80, 7.30e-2, 8.02e-2, 1.16e-1, 1.19e-1, 2.2e-4, 2.259),
    (2, 320, 1.85e-2, 2.08e-2, 3.10e-2, 3.20e-2, 2.6e-5, 2.188),
)

#: Observed orders printed in table 12, per subdivision level:
#: (A33', B33', A11', Fz).
_SPHERE_TABLE12_ORDERS = {
    1: (2.12, 1.72, 2.04, 1.91),
    2: (2.06, 1.93, 2.03, 1.98),
}

_SPHERE_TABLE12_PARAMS = {"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5}

#: Quantities of a refinement-study level row that the registry may quote.
_LEVEL_QUANTITIES = (
    "heave_added_mass_rel_error",
    "heave_damping_rel_error",
    "surge_added_mass_rel_error",
    "surge_damping_rel_error",
    "excitation_force_x_rel_error",
    "excitation_force_z_rel_error",
    "rotational_added_mass_max_ratio",
    "far_field_damping_rel_diff_heave",
    "haskind_reciprocity_rel_diff_x",
    "haskind_reciprocity_rel_diff_z",
    "condition_number",
    "observed_order_a33",
    "observed_order_b33",
    "observed_order_a11",
    "observed_order_b11",
    "observed_order_fz",
)


def _flatten_levels(levels: list[dict]) -> dict[str, float]:
    """Flatten per-level study rows into ``level<sub>_<quantity>`` keys."""
    values: dict[str, float] = {}
    for row in levels:
        prefix = f"level{int(row['subdivisions'])}_"
        for quantity in _LEVEL_QUANTITIES:
            value = row.get(quantity)
            if value is not None:
                values[prefix + quantity] = float(value)
    return values


def _free_surface_3d_entries() -> tuple[DeclaredNumber, ...]:
    """Build the registry rows for the 3-D free-surface tables."""
    entries: list[DeclaredNumber] = []
    doc = "docs/marine_bem_benchmarks.md"

    def add(
        key: str,
        description: str,
        source: str,
        group: str,
        quantity: str,
        declared: float,
        parameters: dict,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> None:
        entries.append(
            DeclaredNumber(
                key=key,
                description=description,
                source=f"{doc} {source}",
                group=group,
                quantity=quantity,
                declared=declared,
                tier="standard",
                parameters=parameters,
                tolerance=tolerance,
            )
        )

    # --- table 10: Hulme floating hemisphere, Ka = 1.0 --------------------
    for sub, panels, a33, b33, a11, b11, fz, cond in _HULME_TABLE10:
        row = f"table 10 row N={panels}"
        add(f"hulme_a33_err_{panels}",
            f"heave added-mass relative error at {panels} panels (Ka=1.0)",
            f"§4.10 {row}: A33' relative error", "hulme_hemisphere",
            f"level{sub}_heave_added_mass_rel_error", a33,
            _HULME_TABLE10_PARAMS, 5.0e-3)
        add(f"hulme_b33_err_{panels}",
            f"heave wave damping relative error at {panels} panels (Ka=1.0)",
            f"§4.10 {row}: B33' relative error", "hulme_hemisphere",
            f"level{sub}_heave_damping_rel_error", b33,
            _HULME_TABLE10_PARAMS, 5.0e-3)
        add(f"hulme_a11_err_{panels}",
            f"surge added-mass relative error at {panels} panels (Ka=1.0)",
            f"§4.10 {row}: A11' relative error", "hulme_hemisphere",
            f"level{sub}_surge_added_mass_rel_error", a11,
            _HULME_TABLE10_PARAMS, 5.0e-3)
        add(f"hulme_b11_err_{panels}",
            f"surge wave damping relative error at {panels} panels (Ka=1.0)",
            f"§4.10 {row}: B11' relative error", "hulme_hemisphere",
            f"level{sub}_surge_damping_rel_error", b11,
            _HULME_TABLE10_PARAMS, 5.0e-3)
        add(f"hulme_fz_err_{panels}",
            f"heave excitation force relative error at {panels} panels (Ka=1.0)",
            f"§4.10 {row}: Fz relative error", "hulme_hemisphere",
            f"level{sub}_excitation_force_z_rel_error", fz,
            _HULME_TABLE10_PARAMS, 5.0e-3)
        add(f"hulme_cond_{panels}",
            f"condition number of the 256-panel ladder at {panels} panels",
            f"§4.10 {row}: condition number", "hulme_hemisphere",
            f"level{sub}_condition_number", cond, _HULME_TABLE10_PARAMS)
    for sub, orders in _HULME_TABLE10_ORDERS.items():
        panels = {1: 16, 2: 64, 3: 256}[sub]
        for name, order in zip(("a33", "b33", "a11", "b11", "fz"), orders):
            quantity = (
                f"level{sub}_observed_order_b11_sqrtN"
                if name == "b11"
                else f"level{sub}_observed_order_{name}"
            )
            add(f"hulme_order_{name}_{panels}",
                f"observed order of the {name.upper()} error at {panels} panels"
                + (" (panel-count convention, h ~ N**-1/2)" if name == "b11" else ""),
                f"§4.10 table 10 row N={panels}: observed order {name.upper()}",
                "hulme_hemisphere", quantity,
                order, _HULME_TABLE10_PARAMS, 5.0e-3)
    # --- §4.10 prose: Kochin / Haskind residuals at 256 panels ------------
    add("hulme_kochin_heave_256",
        "near/far-field heave damping relative difference at 256 panels",
        "§4.10 prose: near/far-field damping difference 5.45e-4",
        "hulme_hemisphere", "level3_far_field_damping_rel_diff_heave",
        5.45e-4, _HULME_TABLE10_PARAMS, 5.0e-3)
    add("hulme_haskind_surge_256",
        "boundary Haskind reciprocity relative difference, surge, at 256 panels",
        "§4.10 prose: Haskind reciprocity difference 2.09e-4 (surge)",
        "hulme_hemisphere", "level3_haskind_reciprocity_rel_diff_x",
        2.09e-4, _HULME_TABLE10_PARAMS, 5.0e-3)
    add("hulme_haskind_heave_256",
        "boundary Haskind reciprocity relative difference, heave, at 256 panels",
        "§4.10 prose: Haskind reciprocity difference 8.09e-4 (heave)",
        "hulme_hemisphere", "level3_haskind_reciprocity_rel_diff_z",
        8.09e-4, _HULME_TABLE10_PARAMS, 5.0e-3)

    # --- table 11: CHIEF regularisation at the interior waterline eigenvalue
    add("chief_exact_a33",
        "exact heave added mass at Ka=2.58 (Hulme reference)",
        "§4.10 table 11 header: exact A33' 0.4007", "hulme_chief",
        "exact_a33", 0.4007, _HULME_CHIEF_PARAMS)
    add("chief_exact_b33",
        "exact heave damping at Ka=2.58 (Hulme reference)",
        "§4.10 table 11 header: exact B33' 0.0631", "hulme_chief",
        "exact_b33", 0.0631, _HULME_CHIEF_PARAMS)
    add("chief_cond_unregularized",
        "condition number of the unregularized BIE at Ka=2.58, N=256",
        "§4.10 table 11 row 1: condition number", "hulme_chief",
        "unregularized_condition_number", 145.66, _HULME_CHIEF_PARAMS)
    add("chief_a33_unregularized",
        "heave added mass of the unregularized BIE at Ka=2.58, N=256",
        "§4.10 table 11 row 1: computed A33'", "hulme_chief",
        "unregularized_a33", 0.3481, _HULME_CHIEF_PARAMS)
    add("chief_b33_unregularized_magnitude",
        "magnitude of the unphysical negative heave damping of the "
        "unregularized BIE (documented as -0.1042)",
        "§4.10 table 11 row 1: computed B33' -0.1042", "hulme_chief",
        "unregularized_b33_magnitude", 0.1042, _HULME_CHIEF_PARAMS)
    add("chief_cond_chief",
        "condition number with 5-point interior CHIEF at Ka=2.58, N=256",
        "§4.10 table 11 row 2: condition number", "hulme_chief",
        "chief_condition_number", 4.24, _HULME_CHIEF_PARAMS)
    add("chief_a33_chief",
        "heave added mass with 5-point interior CHIEF at Ka=2.58, N=256",
        "§4.10 table 11 row 2: computed A33'", "hulme_chief",
        "chief_a33", 0.3919, _HULME_CHIEF_PARAMS)
    add("chief_b33_chief",
        "heave damping with 5-point interior CHIEF at Ka=2.58, N=256",
        "§4.10 table 11 row 2: computed B33' +0.0687", "hulme_chief",
        "chief_b33", 0.0687, _HULME_CHIEF_PARAMS)
    add("chief_point_count",
        "number of interior CHIEF points used by chief_points='auto'",
        "§4.10 prose: 5 interior CHIEF zero-field constraints", "hulme_chief",
        "chief_point_count", 5.0, _HULME_CHIEF_PARAMS)

    # --- table 12: Wang/Linton submerged sphere, f/a = 1.5, Ka = 0.8 ------
    for sub, panels, a33, a11, fx, fz, rot, cond in _SPHERE_TABLE12:
        row = f"table 12 row N={panels}"
        add(f"sphere_a33_err_{panels}",
            f"heave added-mass relative error at {panels} panels (f/a=1.5, Ka=0.8)",
            f"§4.11 {row}: A33' relative error", "submerged_sphere",
            f"level{sub}_heave_added_mass_rel_error", a33,
            _SPHERE_TABLE12_PARAMS, 5.0e-3)
        add(f"sphere_a11_err_{panels}",
            f"surge added-mass relative error at {panels} panels (f/a=1.5, Ka=0.8)",
            f"§4.11 {row}: A11' relative error", "submerged_sphere",
            f"level{sub}_surge_added_mass_rel_error", a11,
            _SPHERE_TABLE12_PARAMS, 5.0e-3)
        add(f"sphere_fx_err_{panels}",
            f"surge excitation force relative error at {panels} panels",
            f"§4.11 {row}: horizontal wave force relative error",
            "submerged_sphere", f"level{sub}_excitation_force_x_rel_error",
            fx, _SPHERE_TABLE12_PARAMS, 5.0e-3)
        add(f"sphere_fz_err_{panels}",
            f"heave excitation force relative error at {panels} panels",
            f"§4.11 {row}: vertical wave force relative error",
            "submerged_sphere", f"level{sub}_excitation_force_z_rel_error",
            fz, _SPHERE_TABLE12_PARAMS, 5.0e-3)
        add(f"sphere_rot_coupling_{panels}",
            f"max rotational added-mass coupling ratio at {panels} panels",
            f"§4.11 {row}: rotational coupling ratio", "submerged_sphere",
            f"level{sub}_rotational_added_mass_max_ratio", rot,
            _SPHERE_TABLE12_PARAMS, 2.5e-2)
        add(f"sphere_cond_{panels}",
            f"condition number at {panels} panels (f/a=1.5, Ka=0.8)",
            f"§4.11 {row}: condition number", "submerged_sphere",
            f"level{sub}_condition_number", cond, _SPHERE_TABLE12_PARAMS)
    for sub, orders in _SPHERE_TABLE12_ORDERS.items():
        panels = {0: 20, 1: 80, 2: 320}[sub]
        for name, order in zip(("a33", "b33", "a11", "fz"), orders):
            add(f"sphere_order_{name}_{panels}",
                f"observed order of the {name.upper()} error at {panels} panels",
                f"§4.11 table 12 row N={panels}: observed order {name.upper()}",
                "submerged_sphere", f"level{sub}_observed_order_{name}",
                order, _SPHERE_TABLE12_PARAMS, 5.0e-3)
    return tuple(entries)


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
    *_free_surface_3d_entries(),
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
    if group == "hulme_hemisphere":
        study = floating_hemisphere_benchmark.run_refinement_study(
            subdivisions=tuple(int(s) for s in parameters["subdivisions"]),
            ka=float(parameters.get("ka", 1.0)),
        )
        values = _flatten_levels(study["levels"])
        # The B11' order column of table 10 follows the m2609 panel-count
        # convention (h ~ N**-1/2), not the h_mean convention of the other
        # order columns; the document says so explicitly.
        levels = study["levels"]
        for i in range(1, len(levels)):
            lo, hi = levels[i - 1], levels[i]
            e_lo = float(lo["surge_damping_rel_error"])
            e_hi = float(hi["surge_damping_rel_error"])
            if e_lo > 0.0 and e_hi > 0.0:
                values[
                    f"level{int(hi['subdivisions'])}_observed_order_b11_sqrtN"
                ] = math.log2(e_lo / e_hi) / math.log2(
                    math.sqrt(hi["panel_count"] / lo["panel_count"])
                )
        return values
    if group == "hulme_chief":
        rows = floating_hemisphere_benchmark.run_irregular_frequency_study(
            subdivision=int(parameters["subdivision"]),
            ka_values=(float(parameters["ka"]),),
        )
        row = rows[0]
        return {
            "exact_a33": float(row["exact_a33"]),
            "exact_b33": float(row["exact_b33"]),
            "unregularized_condition_number": float(
                row["unregularized"]["condition_number"]
            ),
            "unregularized_a33": float(row["unregularized"]["a33_dimless"]),
            "unregularized_b33_magnitude": abs(
                float(row["unregularized"]["b33_dimless"])
            ),
            "chief_condition_number": float(
                row["chief_regularized"]["condition_number"]
            ),
            "chief_a33": float(row["chief_regularized"]["a33_dimless"]),
            "chief_b33": float(row["chief_regularized"]["b33_dimless"]),
            "chief_point_count": float(
                row["chief_regularized"]["chief_point_count"]
            ),
        }
    if group == "submerged_sphere":
        study = submerged_sphere_benchmark.run_refinement_study(
            subdivisions=tuple(int(s) for s in parameters["subdivisions"]),
            ka=float(parameters.get("ka", 0.8)),
            submergence=float(parameters.get("submergence", 1.5)),
        )
        return _flatten_levels(study["levels"])
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