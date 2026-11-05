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

Provenance note (2026-11-04). The three-dimensional free-surface tables of
``docs/marine_bem_benchmarks.md`` (sections 4.10-4.11: the Hulme floating
hemisphere refinement, the interior-CHIEF irregular-frequency comparison and
the Wang/Linton submerged-sphere refinement) were documentation-only: the
numbers came from the ``floating_hemisphere`` and ``submerged_sphere_wave``
command-line studies, and the tests that did exist guard the *physics* at a
few-percent threshold, not the quoted convergence columns. Every column of
those three tables is now registered (``hemisphere3d``,
``hemisphere3d_chief``, ``submerged_sphere3d`` groups) and recomputed on the
``standard`` tier, which costs about a minute of three-dimensional solves.
"""

from __future__ import annotations

import argparse
import json
import math
import numbers
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
    submerged_sphere_wave as submerged_sphere_wave_benchmark,
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
    """Which benchmark to run: ``cylinder``, ``ellipsoid``,
    ``interior_harmonics``, ``ursell``, ``ursell_reference``, ``hemisphere3d``,
    ``hemisphere3d_chief`` or ``submerged_sphere3d``."""
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
# 3-D free-surface tables (docs/marine_bem_benchmarks.md sections 4.10-4.11)
# --------------------------------------------------------------------------
# The numbers below are the columns of tables 10, 11 and 12 of
# ``docs/marine_bem_benchmarks.md`` as printed. They are registered with the
# mesh level they belong to, so that a stale column is reported per cell rather
# than as one lumped "the table moved" failure.

_FS3D_RADIUS = 1.0
_FS3D_DENSITY = 1025.0
_FS3D_GRAVITY = 9.80665
_HEMISPHERE_KA = 1.0
_HEMISPHERE_CHIEF_KA = 2.58
_HEMISPHERE_CHIEF_SUBDIVISION = 3
_SPHERE_KA = 0.8
_SPHERE_SUBMERGENCE = 1.5

#: Columns are quoted to three significant digits; 1.5 % tolerance covers the
#: printing truncation with a wide margin while still catching a kernel change
#: (those move an error by tens of percent).
_FS3D_TOLERANCE = 1.5e-2

#: The rotational decoupling ratio is a difference of large numbers and the docs
#: quote it to three significant digits, so it keeps a wider band than the rest
#: of the 3-D columns (main's registry quoted ``2.1510e-04`` / ``2.6481e-05``;
#: those are the values below, i.e. the ones printed in table 12 after the
#: m3150 rounding fix, rather than the earlier two-digit ``2.2e-4`` / ``2.6e-5``).
_FS3D_COARSE_TOLERANCE = 5.0e-2

#: Table 10: floating hemisphere at ``Ka = 1``, keyed by the quantity name that
#: :func:`_flatten_refinement_levels` produces. ``order_*`` entries are the
#: observed convergence orders printed in the last column of the table.
_HEMISPHERE_TABLE: dict[int, dict[str, float]] = {
#: ``order_b11`` (1.80 / 1.94): the table printed 1.62 / 1.88, inherited from the
#: m2609 panel study whose mesh family gives an exactly halving ``h_mean``; the
#: table-10 meshes do not (h_mean ratio 1.87 between consecutive levels), so the
#: h_mean-based order of *this* run is 1.80 / 1.94. Reproducing the printed pair
#: required changing the mesh family, not the solver: recorded as a doc correction
#: (see dev_report_m3250_free_surface3d_pins.md section 5).
    16: {
        "condition_number": 3.893,
        "heave_added_mass_rel_error": 1.20e-1,
        "heave_damping_rel_error": 1.22e-1,
        "surge_added_mass_rel_error": 1.80e-1,
        "surge_damping_rel_error": 4.02e-1,
        "excitation_force_z_rel_error": 6.67e-2,
    },
    64: {
        "condition_number": 3.052,
        "heave_added_mass_rel_error": 2.97e-2,
        "heave_damping_rel_error": 4.50e-2,
        "surge_added_mass_rel_error": 5.05e-2,
        "surge_damping_rel_error": 1.30e-1,
        "excitation_force_z_rel_error": 2.21e-2,
        "order_a33": 2.23,
        "order_b33": 1.59,
        "order_a11": 2.04,
        "order_b11": 1.80,
        "order_fz": 1.77,
        # measured 1.8021 (this entry) / 1.9377 (the 256-panel entry below);
        # the table prints two decimals, so the pin is the printed pair, and the
        # 1.5 % band also accepts main's four-digit registry values.
    },
    256: {
        "condition_number": 2.783,
        "heave_added_mass_rel_error": 7.41e-3,
        "heave_damping_rel_error": 1.18e-2,
        "surge_added_mass_rel_error": 1.30e-2,
        "surge_damping_rel_error": 3.53e-2,
        "excitation_force_z_rel_error": 5.67e-3,
        "order_a33": 2.06,
        "order_b33": 1.98,
        "order_a11": 2.01,
        "order_b11": 1.94,
        "order_fz": 2.02,
    },
}

#: Closures quoted in the section 4.10 prose at the finest hemisphere mesh: the
#: near-field/far-field damping identity and the two Haskind residuals.
_HEMISPHERE_CLOSURES: dict[str, float] = {
    "far_field_damping_rel_diff_heave": 5.45e-4,
    "haskind_reciprocity_rel_diff_x": 2.09e-4,
    "haskind_reciprocity_rel_diff_z": 8.09e-4,
}

#: Table 12: submerged sphere at ``Ka = 0.8``, ``f/a = 1.5``.
_SPHERE_TABLE: dict[int, dict[str, float]] = {
    20: {
        "condition_number": 2.407,
        "heave_added_mass_rel_error": 2.55e-1,
        "surge_added_mass_rel_error": 2.68e-1,
        "excitation_force_x_rel_error": 3.60e-1,
        "excitation_force_z_rel_error": 3.70e-1,
    },
    80: {
        "condition_number": 2.259,
        "heave_added_mass_rel_error": 7.30e-2,
        "surge_added_mass_rel_error": 8.02e-2,
        "excitation_force_x_rel_error": 1.16e-1,
        "excitation_force_z_rel_error": 1.19e-1,
        "order_a33": 2.12,
        "order_b33": 1.72,
        "order_a11": 2.04,
        "order_fz": 1.91,
    },
    320: {
        "condition_number": 2.188,
        "heave_added_mass_rel_error": 1.85e-2,
        "surge_added_mass_rel_error": 2.08e-2,
        "excitation_force_x_rel_error": 3.10e-2,
        "excitation_force_z_rel_error": 3.20e-2,
        "order_a33": 2.06,
        "order_b33": 1.93,
        "order_a11": 2.03,
        "order_fz": 1.98,
    },
}

#: Rotational decoupling about the sphere centre, ``max |A_ij| / A_33`` over the
#: rotation block. The table prints ``2.5e-32`` at 20 panels, a roundoff
#: artefact of an exactly symmetric mesh; the reproducible statement is "below
#: 1e-6", registered as an upper-bound claim.
_SPHERE_ROTATION: dict[int, tuple[float, str]] = {
    20: (1.0e-6, "upper"),
    80: (2.151e-4, "value"),
    320: (2.6481e-5, "value"),
}

#: Table 11: interior-CHIEF regularisation at the first interior waterline
#: resonance (``Ka = 2.58``, 256 panels). Damping *magnitudes* are registered
#: because the claim of the table is about the sign of ``B_33``; the sign
#: itself is asserted in ``tests/test_declared_numbers.py``.
_HEMISPHERE_CHIEF: dict[str, dict[str, float]] = {
    "unregularized": {
        "condition_number": 145.66,
        "a33_dimless": 0.3481,
        "b33_dimless_abs": 0.1042,
    },
    "chief_regularized": {
        "condition_number": 4.24,
        "a33_dimless": 0.3919,
        "b33_dimless_abs": 0.0687,
    },
}

#: Which level column drives which observed-order label.
_HEMISPHERE_ORDER_LABELS = {
    "heave_added_mass_rel_error": "a33",
    "heave_damping_rel_error": "b33",
    "surge_added_mass_rel_error": "a11",
    "surge_damping_rel_error": "b11",
    "excitation_force_z_rel_error": "fz",
}

_SPHERE_ORDER_LABELS = {
    "heave_added_mass_rel_error": "a33",
    "heave_damping_rel_error": "b33",
    "surge_added_mass_rel_error": "a11",
    "excitation_force_z_rel_error": "fz",
}

_LEVEL_META = frozenset({"subdivisions", "panel_count", "h_mean"})


def _flatten_refinement_levels(
    levels: Sequence[dict], order_labels: dict[str, str]
) -> dict[str, float]:
    """Flatten refinement rows to ``<panels>.<quantity>`` and add orders.

    The benchmarks already report four observed orders, but the registry also
    needs the surge-damping order (the slowest converging column of table 10),
    so the order is recomputed here from the same rows instead of duplicating
    the arithmetic in the benchmark.
    """
    values: dict[str, float] = {}
    h_values = [float(row["h_mean"]) for row in levels]
    for row in levels:
        panels = int(row["panel_count"])
        for name, value in row.items():
            if name in _LEVEL_META or name.startswith("observed_order"):
                continue
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, numbers.Real):
                values[f"{panels}.{name}"] = float(value)
    for source, label in order_labels.items():
        if not all(row.get(source) is not None for row in levels):
            continue
        for idx in range(1, len(levels)):
            h0, h1 = h_values[idx - 1], h_values[idx]
            e0 = float(levels[idx - 1][source])
            e1 = float(levels[idx][source])
            if e0 <= 0.0 or e1 <= 0.0 or h0 <= h1:
                continue
            values[f"{int(levels[idx]['panel_count'])}.order_{label}"] = float(
                math.log(e0 / e1) / math.log(h0 / h1)
            )
    return values


def _reference_values(reference: dict) -> dict[str, float]:
    """Flatten the exact-reference block (complex forces become re/im pairs)."""
    values: dict[str, float] = {}
    for name, value in reference.items():
        if isinstance(value, (list, tuple)):
            values[f"reference_{name}_re"] = float(value[0])
            values[f"reference_{name}_im"] = float(value[1])
        elif isinstance(value, numbers.Real) and not isinstance(value, bool):
            values[f"reference_{name}"] = float(value)
    return values


def _free_surface_three_dimensional_numbers() -> tuple[DeclaredNumber, ...]:
    """Registry rows for the 3-D free-surface tables 10, 11 and 12."""
    rows: list[DeclaredNumber] = []

    hemisphere_levels = tuple(sorted(_HEMISPHERE_TABLE))
    hemisphere_parameters = {
        "subdivisions": list(hemisphere_levels),
        "ka": _HEMISPHERE_KA,
        "radius": _FS3D_RADIUS,
        "density": _FS3D_DENSITY,
        "gravity": _FS3D_GRAVITY,
    }
    for panels, columns in _HEMISPHERE_TABLE.items():
        for quantity, declared in columns.items():
            rows.append(
                DeclaredNumber(
                    key=f"hemisphere3d_{quantity}_{panels}",
                    description=f"{quantity} at {panels} panels, Ka={_HEMISPHERE_KA}",
                    source=(
                        "docs/marine_bem_benchmarks.md table 10 "
                        "(Hulme floating hemisphere)"
                    ),
                    group="hemisphere3d",
                    quantity=f"{panels}.{quantity}",
                    declared=declared,
                    parameters=dict(hemisphere_parameters),
                tier="standard",
                )
            )
    finest_hemisphere = max(hemisphere_levels)
    for quantity, declared in _HEMISPHERE_CLOSURES.items():
        rows.append(
            DeclaredNumber(
                key=f"hemisphere3d_{quantity}_{finest_hemisphere}",
                description=(
                    f"{quantity} at {finest_hemisphere} panels (prose closure)"
                ),
                source="docs/marine_bem_benchmarks.md section 4.10 prose",
                group="hemisphere3d",
                quantity=f"{finest_hemisphere}.{quantity}",
                declared=declared,
                parameters=dict(hemisphere_parameters),
                tier="standard",
            )
        )

    chief_parameters = {
        "subdivisions": _HEMISPHERE_CHIEF_SUBDIVISION,
        "ka": [_HEMISPHERE_CHIEF_KA],
        "radius": _FS3D_RADIUS,
        "density": _FS3D_DENSITY,
        "gravity": _FS3D_GRAVITY,
    }
    for branch, columns in _HEMISPHERE_CHIEF.items():
        for quantity, declared in columns.items():
            rows.append(
                DeclaredNumber(
                    key=f"chief3d_{branch}_{quantity}",
                    description=(
                        f"{branch} {quantity} at Ka={_HEMISPHERE_CHIEF_KA}, "
                        f"subdivisions={_HEMISPHERE_CHIEF_SUBDIVISION}"
                    ),
                    source=(
                        "docs/marine_bem_benchmarks.md table 11 "
                        "(3-D interior CHIEF at the waterline resonance)"
                    ),
                    group="hemisphere3d_chief",
                    quantity=f"{branch}.{quantity}",
                    declared=declared,
                    parameters=dict(chief_parameters),
                tier="standard",
                )
            )

    sphere_levels = tuple(sorted(_SPHERE_TABLE))
    sphere_parameters = {
        "subdivisions": list(sphere_levels),
        "ka": _SPHERE_KA,
        "radius": _FS3D_RADIUS,
        "submergence": _SPHERE_SUBMERGENCE,
        "density": _FS3D_DENSITY,
        "gravity": _FS3D_GRAVITY,
    }
    for panels, columns in _SPHERE_TABLE.items():
        for quantity, declared in columns.items():
            rows.append(
                DeclaredNumber(
                    key=f"sphere3d_{quantity}_{panels}",
                    description=f"{quantity} at {panels} panels, Ka={_SPHERE_KA}",
                    source=(
                        "docs/marine_bem_benchmarks.md table 12 "
                        "(Wang/Linton submerged sphere)"
                    ),
                    group="submerged_sphere3d",
                    quantity=f"{panels}.{quantity}",
                    declared=declared,
                    parameters=dict(sphere_parameters),
                tier="standard",
                )
            )
    for panels, (declared, kind) in _SPHERE_ROTATION.items():
        rows.append(
            DeclaredNumber(
                key=f"sphere3d_rotational_coupling_{panels}",
                description=(
                    "max rotational/translational added-mass ratio at "
                    f"{panels} panels"
                ),
                source=(
                    "docs/marine_bem_benchmarks.md table 12 "
                    "(rotational decoupling column)"
                ),
                group="submerged_sphere3d",
                quantity=f"{panels}.rotational_added_mass_max_ratio",
                declared=declared,
                parameters=dict(sphere_parameters),
                tier="standard",
                kind=kind,
                tolerance=_FS3D_COARSE_TOLERANCE,
            )
        )
    return tuple(rows)

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
    *_free_surface_three_dimensional_numbers(),
)

_URSELL_RADIUS = 1.0
_URSELL_SUBMERGENCE = 2.0
_URSELL_KA = 0.6
_URSELL_DENSITY = 1025.0
_URSELL_GRAVITY = 9.80665


def _study_rows(study: dict) -> dict[str, float]:
    """Flatten a mesh-refinement study into ``sub<level>_<quantity>`` scalars.

    Kept from main while merging PR #136: the m3250 registry flattens the same
    studies with :func:`_flatten_refinement_levels` (``<panels>.<quantity>``
    keys), so this helper and :func:`_chief_rows` below are no longer on the
    registry path.  They are retained rather than deleted so that the merge did
    not drop a side's code without a trace; nothing imports them.

    One registered refinement study produces many quoted numbers (one row per
    panel count, plus the observed orders of the finest level). Flattening them
    into a single dict lets the registry cache one solve per parametric setting
    instead of one per quoted number.
    """
    rows: dict[str, float] = {}
    h_values: list[float] = []
    b11_errors: list[float] = []
    for level in study["levels"]:
        prefix = f"sub{int(level['subdivisions'])}_"
        for key, value in level.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            rows[prefix + key] = float(value)
        if "surge_damping_rel_error" in level:
            h_values.append(float(level["h_mean"]))
            b11_errors.append(float(level["surge_damping_rel_error"]))
    # The documented table also quotes the observed order of the surge damping,
    # which the benchmark itself does not emit; it follows from the same
    # h-based estimate the module uses for the other quantities.
    for idx in range(1, len(b11_errors)):
        h0, h1 = h_values[idx - 1], h_values[idx]
        e0, e1 = b11_errors[idx - 1], b11_errors[idx]
        if e0 <= 0.0 or e1 <= 0.0 or h0 <= h1:
            continue
        level = study["levels"][idx]
        rows[f"sub{int(level['subdivisions'])}_observed_order_b11"] = math.log(e0 / e1) / math.log(h0 / h1)
    return rows


def _chief_rows(rows) -> dict[str, float]:
    """Flatten the irregular-frequency study into ``ka<value>_<scheme>_<quantity>``."""
    values: dict[str, float] = {}
    for row in rows:
        prefix = f"ka{float(row['ka']):.2f}_"
        for scheme in ("unregularized", "chief_regularized"):
            for key, value in row[scheme].items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                values[prefix + scheme + "_" + key] = float(value)
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
    if group == "hemisphere3d":
        study = floating_hemisphere_benchmark.run_refinement_study(
            subdivisions=tuple(int(s) for s in parameters["subdivisions"]),
            ka=float(parameters.get("ka", _HEMISPHERE_KA)),
            radius=float(parameters.get("radius", _FS3D_RADIUS)),
            density=float(parameters.get("density", _FS3D_DENSITY)),
            gravity=float(parameters.get("gravity", _FS3D_GRAVITY)),
        )
        values = _flatten_refinement_levels(study["levels"], _HEMISPHERE_ORDER_LABELS)
        values.update(_reference_values(study["exact_reference"]))
        return values
    if group == "hemisphere3d_chief":
        ka = parameters["ka"]
        ka_values = tuple(
            float(value) for value in (ka if isinstance(ka, (list, tuple)) else (ka,))
        )
        rows = floating_hemisphere_benchmark.run_irregular_frequency_study(
            subdivision=int(parameters["subdivisions"]),
            ka_values=ka_values,
            radius=float(parameters.get("radius", _FS3D_RADIUS)),
            density=float(parameters.get("density", _FS3D_DENSITY)),
            gravity=float(parameters.get("gravity", _FS3D_GRAVITY)),
        )
        if len(rows) != 1:
            raise ValueError("the chief group runs exactly one Ka per registry row")
        row = rows[0]
        values = {
            "exact_a33": float(row["exact_a33"]),
            "exact_b33": float(row["exact_b33"]),
        }
        for branch in ("unregularized", "chief_regularized"):
            branch_values = row[branch]
            for name, value in branch_values.items():
                if isinstance(value, bool) or not isinstance(value, numbers.Real):
                    continue
                values[f"{branch}.{name}"] = float(value)
            values[f"{branch}.b33_dimless_abs"] = abs(
                float(branch_values["b33_dimless"])
            )
        return values
    if group == "submerged_sphere3d":
        study = submerged_sphere_wave_benchmark.run_refinement_study(
            subdivisions=tuple(int(s) for s in parameters["subdivisions"]),
            ka=float(parameters.get("ka", _SPHERE_KA)),
            radius=float(parameters.get("radius", _FS3D_RADIUS)),
            submergence=float(parameters.get("submergence", _SPHERE_SUBMERGENCE)),
            density=float(parameters.get("density", _FS3D_DENSITY)),
            gravity=float(parameters.get("gravity", _FS3D_GRAVITY)),
        )
        values = _flatten_refinement_levels(study["levels"], _SPHERE_ORDER_LABELS)
        values.update(_reference_values(study["exact_reference"]))
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