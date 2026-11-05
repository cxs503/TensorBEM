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
    """Which benchmark to run: ``cylinder``, ``ellipsoid``, ``ursell``,
    ``ursell_reference``, ``interior_harmonics``, ``hulme_hemisphere``,
    ``hulme_chief`` or ``submerged_sphere_wave``."""
    quantity: str
    """Key into the dictionary returned by the group's runner."""
    declared: float
    tier: str = "fast"
    parameters: dict = field(default_factory=dict)
    tolerance: float = DEFAULT_TOLERANCE
    kind: str = "value"
    """``value`` compares for equality, ``upper`` only checks ``measured <=
    declared`` (for sentences of the form "within 6e-3")."""



#: Quantities of one refinement level that the 3-D free-surface tables quote.
#: ``None`` entries (an observed order is undefined at the coarsest level) are
#: dropped during flattening instead of being smuggled in as NaN.
_FREE_SURFACE_3D_QUANTITIES = (
    "heave_added_mass_rel_error",
    "heave_damping_rel_error",
    "surge_added_mass_rel_error",
    "surge_damping_rel_error",
    "excitation_force_x_rel_error",
    "excitation_force_z_rel_error",
    "observed_order_a33",
    "observed_order_b33",
    "observed_order_a11",
    "observed_order_fz",
    "observed_order_b11",
    "condition_number",
    "relative_residual",
    "haskind_reciprocity_rel_diff_x",
    "haskind_reciprocity_rel_diff_z",
    "far_field_damping_rel_diff_heave",
    "rotational_added_mass_max_ratio",
)


def _flatten_levels(levels: Sequence[dict]) -> dict[str, float]:
    """Key one refinement row per panel count: ``panels256.heave_damping...``.

    The observed order of the surge damping is the one quoted order the
    benchmark itself does not emit (it emits A33'/B33'/A11'/Fz); it follows from
    the measured mean-edge-ratio estimate ``h0/h1`` the other orders use, so it
    is derived here rather than left unregistered.
    """
    values: dict[str, float] = {}
    for row in levels:
        prefix = f"panels{int(row['panel_count'])}"
        for quantity in _FREE_SURFACE_3D_QUANTITIES:
            value = row.get(quantity)
            if value is None:
                continue
            values[f"{prefix}.{quantity}"] = float(value)
    for coarse, fine in zip(levels, levels[1:]):
        h0, h1 = float(coarse["h_mean"]), float(fine["h_mean"])
        e0 = float(coarse.get("surge_damping_rel_error", 0.0))
        e1 = float(fine.get("surge_damping_rel_error", 0.0))
        if e0 <= 0.0 or e1 <= 0.0 or h0 <= h1:
            continue
        values[f"panels{int(fine['panel_count'])}.observed_order_b11"] = math.log(
            e0 / e1
        ) / math.log(h0 / h1)
    return values


def _flatten_reference(reference: dict) -> dict[str, float]:
    """Expose the semi-analytical reference values as ``reference.<name>``."""
    values: dict[str, float] = {}
    for key, value in reference.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            values[f"reference.{key}"] = float(value)
        elif isinstance(value, list) and len(value) == 2:
            values[f"reference.{key}_abs"] = math.hypot(float(value[0]), float(value[1]))
    return values


def _flatten_chief_rows(rows: Sequence[dict]) -> dict[str, float]:
    """Flatten the unregularized/CHIEF comparison of ``hulme_chief``."""
    values: dict[str, float] = {}
    for row in rows:
        tag = f"ka{float(row['ka']):.2f}"
        values[f"{tag}.exact_a33"] = float(row["exact_a33"])
        values[f"{tag}.exact_b33"] = float(row["exact_b33"])
        for scheme in ("unregularized", "chief_regularized"):
            for key, value in row[scheme].items():
                if isinstance(value, bool):
                    continue
                values[f"{tag}.{scheme}.{key}"] = float(value)
        # The unregularized heave damping is *negative* (it violates passivity),
        # and the well-formedness gate only admits positive ``declared`` values,
        # so the quoted magnitude is registered and the sign is asserted in
        # tests/test_declared_numbers.py instead of being lost.
        values[f"{tag}.unregularized.b33_magnitude"] = abs(
            float(row["unregularized"]["b33_dimless"])
        )
    return values



#: Parametric settings shared by every row of one documented table, so that the
#: ``(group, parameters)`` cache in :func:`check` runs each study once.
_HULME_REFINEMENT = {
    "subdivisions": [1, 2, 3],
    "ka": 1.0,
    "radius": 1.0,
    "density": 1025.0,
    "gravity": 9.80665,
}
_HULME_COARSE = {
    "subdivisions": [2],
    "ka": 1.0,
    "radius": 1.0,
    "density": 1025.0,
    "gravity": 9.80665,
}
_SPHERE_REFINEMENT = {
    "subdivisions": [0, 1, 2],
    "ka": 0.8,
    "radius": 1.0,
    "submergence": 1.5,
    "density": 1025.0,
    "gravity": 9.80665,
}
_SPHERE_COARSE = {
    "subdivisions": [1],
    "ka": 0.8,
    "radius": 1.0,
    "submergence": 1.5,
    "density": 1025.0,
    "gravity": 9.80665,
}
_HULME_CHIEF = {
    "subdivision": 3,
    "ka_values": [2.58],
    "radius": 1.0,
    "density": 1025.0,
    "gravity": 9.80665,
}

_HULME_SOURCE = (
    "docs/marine_bem_benchmarks.md section 4.10 table 10 "
    "(Hulme 1982 floating hemisphere, Ka=1.0)"
)
_HULME_CHIEF_SOURCE = (
    "docs/marine_bem_benchmarks.md section 4.10 table 11 "
    "(3-D CHIEF at the interior waterline resonance Ka=2.58, N=256)"
)
_SPHERE_SOURCE = (
    "docs/marine_bem_benchmarks.md section 4.11 table 12 "
    "(Wang/Linton submerged sphere, f/a=1.5, Ka=0.8)"
)

_HULME_TABLE_10 = {
    16: {
        "heave_added_mass_rel_error": 1.20e-1,
        "heave_damping_rel_error": 1.22e-1,
        "surge_added_mass_rel_error": 1.80e-1,
        "surge_damping_rel_error": 4.02e-1,
        "excitation_force_z_rel_error": 6.67e-2,
    },
    64: {
        "heave_added_mass_rel_error": 2.97e-2,
        "heave_damping_rel_error": 4.50e-2,
        "surge_added_mass_rel_error": 5.05e-2,
        "surge_damping_rel_error": 1.30e-1,
        "excitation_force_z_rel_error": 2.21e-2,
    },
    256: {
        "heave_added_mass_rel_error": 7.41e-3,
        "heave_damping_rel_error": 1.18e-2,
        "surge_added_mass_rel_error": 1.30e-2,
        "surge_damping_rel_error": 3.53e-2,
        "excitation_force_z_rel_error": 5.67e-3,
    },
}
_HULME_TABLE_10_ORDERS = {
    64: {
        "observed_order_a33": 2.23,
        "observed_order_b33": 1.59,
        "observed_order_a11": 2.04,
        "observed_order_b11": 1.80,
        "observed_order_fz": 1.77,
    },
    256: {
        "observed_order_a33": 2.06,
        "observed_order_b33": 1.98,
        "observed_order_a11": 2.01,
        "observed_order_b11": 1.94,
        "observed_order_fz": 2.02,
    },
}
_HULME_TABLE_10_CONDITION = {
    16: {"condition_number": 3.893},
    64: {"condition_number": 3.052},
    256: {"condition_number": 2.783},
}
_SPHERE_TABLE_12 = {
    20: {
        "heave_added_mass_rel_error": 2.55e-1,
        "surge_added_mass_rel_error": 2.68e-1,
        "excitation_force_x_rel_error": 3.60e-1,
        "excitation_force_z_rel_error": 3.70e-1,
    },
    80: {
        "heave_added_mass_rel_error": 7.30e-2,
        "surge_added_mass_rel_error": 8.02e-2,
        "excitation_force_x_rel_error": 1.16e-1,
        "excitation_force_z_rel_error": 1.19e-1,
    },
    320: {
        "heave_added_mass_rel_error": 1.85e-2,
        "surge_added_mass_rel_error": 2.08e-2,
        "excitation_force_x_rel_error": 3.10e-2,
        "excitation_force_z_rel_error": 3.20e-2,
    },
}
_SPHERE_TABLE_12_ORDERS = {
    80: {
        "observed_order_a33": 2.12,
        "observed_order_b33": 1.72,
        "observed_order_a11": 2.04,
        "observed_order_fz": 1.91,
    },
    320: {
        "observed_order_a33": 2.06,
        "observed_order_b33": 1.93,
        "observed_order_a11": 2.03,
        "observed_order_fz": 1.98,
    },
}
_SPHERE_TABLE_12_CONDITION = {
    20: {"condition_number": 2.407},
    80: {"condition_number": 2.259},
    320: {"condition_number": 2.188},
}
_SPHERE_TABLE_12_ROTATION = {
    80: {"rotational_added_mass_max_ratio": 2.15e-4},
}
#: Human-readable description of each flattened suffix, used to build the
#: registry rows without repeating the same sentence forty times.
_QUANTITY_LABEL = {
    "heave_added_mass_rel_error": "heave added mass A33' relative error",
    "heave_damping_rel_error": "heave damping B33' relative error",
    "surge_added_mass_rel_error": "surge added mass A11' relative error",
    "surge_damping_rel_error": "surge damping B11' relative error",
    "excitation_force_x_rel_error": "horizontal wave force relative error",
    "excitation_force_z_rel_error": "vertical wave force relative error",
    "observed_order_a33": "observed order of A33'",
    "observed_order_b33": "observed order of B33'",
    "observed_order_a11": "observed order of A11'",
    "observed_order_fz": "observed order of Fz",
    "observed_order_b11": "observed order of B11'",
    "condition_number": "condition number of the assembled system",
    "rotational_added_mass_max_ratio": "spurious heave-rotation coupling ratio",
}


def _table_rows(
    group: str,
    parameters: dict,
    source: str,
    table: dict,
    tier: str,
    key_suffix: str = "",
) -> tuple:
    """One registry row per quoted table cell, with the doc value as declared."""
    return tuple(
        DeclaredNumber(
            key=f"{group}_{panels}_{quantity}{key_suffix}",
            description=f"{_QUANTITY_LABEL[quantity]} at {panels} panels",
            source=source,
            group=group,
            quantity=f"panels{panels}.{quantity}",
            parameters=dict(parameters),
            declared=float(declared),
            tier=tier,
        )
        for panels, row in table.items()
        for quantity, declared in row.items()
    )


def _free_surface_3d_rows() -> tuple:
    """The 3-D free-surface tables, which quoted numbers no gate recomputed.

    Before this block the Hulme hemisphere, the interior-resonance CHIEF
    comparison and the Wang/Linton submerged sphere existed only as prose and
    tables in ``docs/marine_bem_benchmarks.md``: every number could rot without
    a single test noticing. The registration cost is the reason they were left
    out (a 256-panel free-surface solve is seconds, not milliseconds), so the
    mesh-refinement tables live in the ``heavy`` tier while one coarse level per
    family stays in ``standard`` so that the default test run still executes the
    three-dimensional free-surface path.
    """
    return (
        *_table_rows(
            "hulme_hemisphere",
            _HULME_REFINEMENT,
            _HULME_SOURCE,
            _HULME_TABLE_10,
            "heavy",
        ),
        *_table_rows(
            "hulme_hemisphere",
            _HULME_REFINEMENT,
            _HULME_SOURCE,
            _HULME_TABLE_10_ORDERS,
            "heavy",
        ),
        # The same documented 64-panel cell is recomputed a second time from a
        # single-level run in the cheap tier, so the default ``pytest`` run
        # still exercises the three-dimensional free-surface path. The key
        # carries ``_standard_guard`` because it is a different run setting
        # (one level instead of the three-level ladder), not a second document.
        *_table_rows(
            "hulme_hemisphere",
            _HULME_REFINEMENT,
            _HULME_SOURCE,
            _HULME_TABLE_10_CONDITION,
            "heavy",
        ),
        *_table_rows(
            "hulme_hemisphere",
            _HULME_COARSE,
            _HULME_SOURCE,
            {64: _HULME_TABLE_10[64]},
            "standard",
            key_suffix="_standard_guard",
        ),
        DeclaredNumber(
            key="hulme_hemisphere_256_far_field_damping_rel_diff_heave",
            description="near-field vs Kochin far-field heave damping gap at 256 panels",
            source=_HULME_SOURCE,
            group="hulme_hemisphere",
            quantity="panels256.far_field_damping_rel_diff_heave",
            parameters=dict(_HULME_REFINEMENT),
            declared=1.0e-3,
            tier="heavy",
            kind="upper",
        ),
        DeclaredNumber(
            key="hulme_hemisphere_256_haskind_reciprocity_rel_diff_z",
            description="boundary-Haskind vs diffraction vertical force gap at 256 panels",
            source=_HULME_SOURCE,
            group="hulme_hemisphere",
            quantity="panels256.haskind_reciprocity_rel_diff_z",
            parameters=dict(_HULME_REFINEMENT),
            declared=1.0e-3,
            tier="heavy",
            kind="upper",
        ),
        DeclaredNumber(
            key="hulme_hemisphere_256_haskind_reciprocity_rel_diff_x",
            description="boundary-Haskind vs diffraction horizontal force gap at 256 panels",
            source=_HULME_SOURCE,
            group="hulme_hemisphere",
            quantity="panels256.haskind_reciprocity_rel_diff_x",
            parameters=dict(_HULME_REFINEMENT),
            declared=1.0e-3,
            tier="heavy",
            kind="upper",
        ),
        DeclaredNumber(
            key="hulme_chief_ka2_58_unregularized_condition_number",
            description="condition number of the unregularized BIE at Ka=2.58, N=256",
            source=_HULME_CHIEF_SOURCE,
            group="hulme_chief",
            quantity="ka2.58.unregularized.condition_number",
            parameters=dict(_HULME_CHIEF),
            declared=145.66,
            tier="heavy",
        ),
        DeclaredNumber(
            key="hulme_chief_ka2_58_unregularized_a33",
            description="heave added mass from the unregularized BIE at Ka=2.58 (exact 0.4007)",
            source=_HULME_CHIEF_SOURCE,
            group="hulme_chief",
            quantity="ka2.58.unregularized.a33_dimless",
            parameters=dict(_HULME_CHIEF),
            declared=0.3481,
            tier="heavy",
        ),
        DeclaredNumber(
            key="hulme_chief_ka2_58_unregularized_b33_magnitude",
            description=(
                "magnitude of the unphysical negative heave damping of the "
                "unregularized BIE at Ka=2.58 (sign asserted in the test suite)"
            ),
            source=_HULME_CHIEF_SOURCE,
            group="hulme_chief",
            quantity="ka2.58.unregularized.b33_magnitude",
            parameters=dict(_HULME_CHIEF),
            declared=0.1042,
            tier="heavy",
        ),
        DeclaredNumber(
            key="hulme_chief_ka2_58_chief_condition_number",
            description="condition number with 5 interior 3-D CHIEF points at Ka=2.58",
            source=_HULME_CHIEF_SOURCE,
            group="hulme_chief",
            quantity="ka2.58.chief_regularized.condition_number",
            parameters=dict(_HULME_CHIEF),
            declared=4.24,
            tier="heavy",
        ),
        DeclaredNumber(
            key="hulme_chief_ka2_58_chief_a33",
            description="heave added mass with 5 CHIEF points at Ka=2.58 (exact 0.4007)",
            source=_HULME_CHIEF_SOURCE,
            group="hulme_chief",
            quantity="ka2.58.chief_regularized.a33_dimless",
            parameters=dict(_HULME_CHIEF),
            declared=0.3919,
            tier="heavy",
        ),
        DeclaredNumber(
            key="hulme_chief_ka2_58_chief_b33",
            description="heave damping with 5 CHIEF points at Ka=2.58 (exact 0.0631, positive)",
            source=_HULME_CHIEF_SOURCE,
            group="hulme_chief",
            quantity="ka2.58.chief_regularized.b33_dimless",
            parameters=dict(_HULME_CHIEF),
            declared=0.0687,
            tier="heavy",
        ),
        *_table_rows(
            "submerged_sphere_wave",
            _SPHERE_REFINEMENT,
            _SPHERE_SOURCE,
            _SPHERE_TABLE_12,
            "heavy",
        ),
        *_table_rows(
            "submerged_sphere_wave",
            _SPHERE_REFINEMENT,
            _SPHERE_SOURCE,
            _SPHERE_TABLE_12_ORDERS,
            "heavy",
        ),
        *_table_rows(
            "submerged_sphere_wave",
            _SPHERE_REFINEMENT,
            _SPHERE_SOURCE,
            _SPHERE_TABLE_12_CONDITION,
            "heavy",
        ),
        *_table_rows(
            "submerged_sphere_wave",
            _SPHERE_REFINEMENT,
            _SPHERE_SOURCE,
            _SPHERE_TABLE_12_ROTATION,
            "heavy",
        ),
        DeclaredNumber(
            key="submerged_sphere_wave_20_rotational_coupling_ratio",
            description=(
                "spurious heave-rotation coupling at 20 panels; table 12 quotes "
                "2.5e-32, which is the reference's own machine zero, so the "
                "documented claim is gated as an upper bound instead"
            ),
            source=_SPHERE_SOURCE,
            group="submerged_sphere_wave",
            quantity="panels20.rotational_added_mass_max_ratio",
            parameters=dict(_SPHERE_REFINEMENT),
            declared=1.0e-6,
            tier="heavy",
            kind="upper",
        ),
        *_table_rows(
            "submerged_sphere_wave",
            _SPHERE_COARSE,
            _SPHERE_SOURCE,
            {80: _SPHERE_TABLE_12[80]},
            "standard",
            key_suffix="_standard_guard",
        ),
        DeclaredNumber(
            key="submerged_sphere_wave_320_haskind_reciprocity_rel_diff_z",
            description=(
                "Newman Haskind reciprocity residual for the vertical force at "
                "320 panels, an upper bound because the reference satisfies it "
                "to machine precision"
            ),
            source=_SPHERE_SOURCE,
            group="submerged_sphere_wave",
            quantity="panels320.haskind_reciprocity_rel_diff_z",
            parameters=dict(_SPHERE_REFINEMENT),
            declared=4.0e-3,
            tier="heavy",
            kind="upper",
        ),
        DeclaredNumber(
            key="submerged_sphere_wave_320_haskind_reciprocity_rel_diff_x",
            description=(
                "Haskind reciprocity residual for the horizontal force at 320 "
                "panels, registered as an upper bound"
            ),
            source=_SPHERE_SOURCE,
            group="submerged_sphere_wave",
            quantity="panels320.haskind_reciprocity_rel_diff_x",
            parameters=dict(_SPHERE_REFINEMENT),
            declared=4.0e-3,
            tier="heavy",
            kind="upper",
        ),
        DeclaredNumber(
            key="submerged_sphere_wave_320_rotational_coupling_ratio",
            description=(
                "spurious heave-rotation coupling at 320 panels; zero by the "
                "sphere's symmetry, so the quoted 2.6e-5 is registered as a bound"
            ),
            source=_SPHERE_SOURCE,
            group="submerged_sphere_wave",
            quantity="panels320.rotational_added_mass_max_ratio",
            parameters=dict(_SPHERE_REFINEMENT),
            declared=1.0e-4,
            tier="heavy",
            kind="upper",
        ),
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
    *_free_surface_3d_rows(),
)

_URSELL_RADIUS = 1.0
_URSELL_SUBMERGENCE = 2.0
_URSELL_KA = 0.6
_URSELL_DENSITY = 1025.0
_URSELL_GRAVITY = 9.80665


#: main flattened the same studies with ``sub<level>_<quantity>`` and
#: ``ka<value>_<scheme>_<quantity>`` keys (``_study_rows`` / ``_chief_rows``).
#: m3245 replaced both with the ``panels<N>.<quantity>`` /
#: ``ka<value>.<scheme>.<quantity>`` flatteners above, which address the same
#: quantities; the B11' order derivation that lived in ``_study_rows`` moved
#: into :func:`_flatten_levels`, so the helpers had no remaining caller.


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
            tuple(int(sub) for sub in parameters.get("subdivisions", (1, 2, 3))),
            ka=float(parameters.get("ka", 1.0)),
            radius=float(parameters.get("radius", 1.0)),
            density=float(parameters.get("density", 1025.0)),
            gravity=float(parameters.get("gravity", 9.80665)),
        )
        values = _flatten_levels(study["levels"])
        values.update(_flatten_reference(study["exact_reference"]))
        return values
    if group in {"submerged_sphere_wave", "submerged_sphere"}:
        # ``submerged_sphere`` is the group name the registry used before the
        # family was renamed after its benchmark module; both spellings run the
        # same study so that a stored registry dump stays readable across the
        # rename.
        study = submerged_sphere_wave_benchmark.run_refinement_study(
            tuple(int(sub) for sub in parameters.get("subdivisions", (0, 1, 2))),
            ka=float(parameters.get("ka", 0.8)),
            radius=float(parameters.get("radius", 1.0)),
            submergence=float(parameters.get("submergence", 1.5)),
            density=float(parameters.get("density", 1025.0)),
            gravity=float(parameters.get("gravity", 9.80665)),
        )
        values = _flatten_levels(study["levels"])
        values.update(_flatten_reference(study["exact_reference"]))
        return values
    if group == "hulme_chief":
        rows = floating_hemisphere_benchmark.run_irregular_frequency_study(
            subdivision=int(parameters.get("subdivision", 3)),
            ka_values=tuple(float(ka) for ka in parameters.get("ka_values", (2.58,))),
            radius=float(parameters.get("radius", 1.0)),
            density=float(parameters.get("density", 1025.0)),
            gravity=float(parameters.get("gravity", 9.80665)),
        )
        return _flatten_chief_rows(rows)
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