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
    floating_hemisphere as floating_hemisphere_benchmark,
    floating_hemisphere as hemisphere_benchmark,
    interior_harmonics as interior_harmonics_benchmark,
    submerged_sphere_wave as sphere_benchmark,
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
    ``hulme_chief`` and ``submerged_sphere`` -- plus, for the cell-by-cell
    reading of tables 10-12, ``hulme_hemisphere_cells``, ``hulme_chief_cells``
    and ``submerged_sphere_cells``."""
    quantity: str
    """Key into the dictionary returned by the group's runner."""
    declared: float
    tier: str = "fast"
    parameters: dict = field(default_factory=dict)
    tolerance: float = DEFAULT_TOLERANCE
    kind: str = "value"
    """``value`` compares for equality, ``upper`` only checks ``measured <=
    declared`` (for sentences of the form "within 6e-3")."""


#: Settings of the three-dimensional free-surface families. These are the
#: settings the documentation quotes and the ones the benchmarks default to, so
#: the registry reproduces a reader's copy-paste of the ``--help`` command.
_HULME_KA = 1.0
_HULME_RADIUS = 1.0
_HULME_DENSITY = 1025.0
_HULME_GRAVITY = 9.80665
_HULME_CHIEF_KA = 2.58
_HULME_CHIEF_SUBDIVISION = 3
_SPHERE_KA = 0.8
_SPHERE_RADIUS = 1.0
_SPHERE_SUBMERGENCE = 1.5

#: Short aliases: a registry key then reads like the column header of the table
#: it guards (``hulme_b11_sub3`` is the \(B_{11}'\) error of table 10, 256
#: panels).
_ALIASES = {
    "condition_number": "kappa",
    "heave_added_mass_rel_error": "a33",
    "heave_damping_rel_error": "b33",
    "surge_added_mass_rel_error": "a11",
    "surge_damping_rel_error": "b11",
    "excitation_force_x_rel_error": "fx",
    "excitation_force_z_rel_error": "fz",
    "observed_order_a33": "order_a33",
    "observed_order_b33": "order_b33",
    "observed_order_a11": "order_a11",
    "observed_order_fz": "order_fz",
    "rotational_added_mass_max_ratio": "rot_coupling",
    "far_field_damping_rel_diff_heave": "kochin_heave",
    "haskind_reciprocity_rel_diff_x": "haskind_x",
    "haskind_reciprocity_rel_diff_z": "haskind_z",
}

_HULME_SOURCE = "docs/marine_bem_benchmarks.md section 4.10 table 10"
_SPHERE_SOURCE = "docs/marine_bem_benchmarks.md section 4.11 table 12"
_CHIEF_SOURCE = "docs/marine_bem_benchmarks.md section 4.10 table 11"

#: Table 10 (Hulme floating hemisphere, ``Ka = 1``, ``a = 1``, ``rho = 1025``):
#: relative error of the radiation/diffraction quantities per icosphere
#: subdivision. Panel counts are 16 / 64 / 256.
_HULME_TABLE = {
    1: {
        "condition_number": 3.893e0,
        "heave_added_mass_rel_error": 1.20e-1,
        "heave_damping_rel_error": 1.22e-1,
        "surge_added_mass_rel_error": 1.80e-1,
        "surge_damping_rel_error": 4.02e-1,
        "excitation_force_z_rel_error": 6.67e-2,
    },
    2: {
        "condition_number": 3.052e0,
        "heave_added_mass_rel_error": 2.97e-2,
        "heave_damping_rel_error": 4.50e-2,
        "surge_added_mass_rel_error": 5.05e-2,
        "surge_damping_rel_error": 1.30e-1,
        "excitation_force_z_rel_error": 2.21e-2,
    },
    3: {
        "condition_number": 2.783e0,
        "heave_added_mass_rel_error": 7.41e-3,
        "heave_damping_rel_error": 1.18e-2,
        "surge_added_mass_rel_error": 1.30e-2,
        "surge_damping_rel_error": 3.53e-2,
        "excitation_force_z_rel_error": 5.67e-3,
    },
}

#: Observed orders of table 10. The \(B_{11}'\) order is quoted in the prose
#: ("1.62 / 1.88", measured against \(h\) with \(h \propto N^{-1/2}\)); the
#: benchmark module does not emit it, so it is guarded by
#: ``tests/test_kochin_hulme_convergence.py`` instead of by a registry row.
_HULME_ORDERS = {
    2: {
        "observed_order_a33": 2.23,
        "observed_order_b33": 1.59,
        "observed_order_a11": 2.04,
        "observed_order_fz": 1.77,
    },
    3: {
        "observed_order_a33": 2.06,
        "observed_order_b33": 1.98,
        "observed_order_a11": 2.01,
        "observed_order_fz": 2.02,
    },
}

#: Table 12 (submerged sphere, ``f/a = 1.5``, ``Ka = 0.8``, ``rho = 1025``).
#: Panel counts are 20 / 80 / 320. ``("upper", x)`` marks a cell printed as a
#: bound rather than as a value.
_SPHERE_TABLE = {
    0: {
        "condition_number": 2.407e0,
        "heave_added_mass_rel_error": 2.55e-1,
        "surge_added_mass_rel_error": 2.68e-1,
        "excitation_force_x_rel_error": 3.60e-1,
        "excitation_force_z_rel_error": 3.70e-1,
        # The table prints 2.5e-32, i.e. round-off. Guarding a round-off digit
        # is not a claim about the method (see the 2026-11-03 audit of the
        # 1.6e-15 row-sum bounds), so the guarded statement is the one that
        # survives a BLAS change: the rotational coupling of a sphere is zero
        # to 1e-30. Measured 2.49e-32.
        "rotational_added_mass_max_ratio": ("upper", 1.0e-30),
    },
    1: {
        "condition_number": 2.259e0,
        "heave_added_mass_rel_error": 7.30e-2,
        "surge_added_mass_rel_error": 8.02e-2,
        "excitation_force_x_rel_error": 1.16e-1,
        "excitation_force_z_rel_error": 1.19e-1,
        # The document prints 2.15e-4 (three figures) since the m3150 audit;
        # the two-figure 2.2e-4 it replaced is a +2.3 % rounding of 2.151e-4.
        "rotational_added_mass_max_ratio": 2.15e-4,
    },
    2: {
        "condition_number": 2.188e0,
        "heave_added_mass_rel_error": 1.85e-2,
        "surge_added_mass_rel_error": 2.08e-2,
        "excitation_force_x_rel_error": 3.10e-2,
        "excitation_force_z_rel_error": 3.20e-2,
        # 2.65e-5 as printed after the m3150 audit (measured 2.6481e-5).
        "rotational_added_mass_max_ratio": 2.65e-5,
    },
}

_SPHERE_ORDERS = {
    1: {
        "observed_order_a33": 2.12,
        "observed_order_b33": 1.72,
        "observed_order_a11": 2.04,
        "observed_order_fz": 1.91,
    },
    2: {
        "observed_order_a33": 2.06,
        "observed_order_b33": 1.93,
        "observed_order_a11": 2.03,
        "observed_order_fz": 1.98,
    },
}

#: Condition numbers are a property of the assembled dense system: they move in
#: the third digit when the BLAS/LAPACK build changes, so they are guarded with
#: a looser band than the discretisation errors.
_KAPPA_TOLERANCE = 2.0e-2

#: Cells the tables print with two significant figures. Rounding to two figures
#: is a \(\pm2.5\,\%\) band on its own, which is wider than
#: :data:`DEFAULT_TOLERANCE`; these are the printed values, so the band has to
#: accommodate the printing rather than the solver.
_TWO_FIGURE_CELLS = frozenset({"rotational_added_mass_max_ratio"})
_TWO_FIGURE_TOLERANCE = 5.0e-2


def _family_levels(prefix: str) -> tuple[int, ...]:
    """Every mesh level a table reports, so one solve feeds all its rows.

    ``check`` caches a group run by its parameters; quoting one level at a time
    would re-run the same refinement study once per cell and leave the observed
    orders (which need two meshes) undefined.
    """
    if prefix == "hulme":
        return tuple(sorted(set(_HULME_TABLE) | set(_HULME_ORDERS)))
    if prefix == "sphere":
        return tuple(sorted(set(_SPHERE_TABLE) | set(_SPHERE_ORDERS)))
    raise ValueError(f"unknown free-surface family: {prefix!r}")


def _free_surface_3d_rows() -> tuple[DeclaredNumber, ...]:
    """The 3-D free-surface declarations (tables 10-12), one row per cell.

    These were the last block of quoted results with no test reading them. The
    2-D families have been registered since m2136 and the Laplace 3-D tables
    since m3085, but the Kelvin-Green-function solver, its two convergence
    tables and the CHIEF regularisation comparison of table 11 were pinned only
    by spot assertions inside ``tests/test_floating_hemisphere_3d.py``: change
    the free-surface kernel, the mesh generator or the CHIEF point heuristic and
    the documents would have kept their numbers while the solver moved.

    The cost is seconds, not minutes -- a 256-panel 3-D assembly is about a
    second -- so the rows live on the ``standard`` tier and a plain ``pytest``
    run guards them.
    """
    rows: list[DeclaredNumber] = []

    def emit(
        group: str,
        source: str,
        cells_by_level: dict[int, dict[str, float]],
        prefix: str,
    ) -> None:
        for level, cells in cells_by_level.items():
            for quantity, spec in cells.items():
                if isinstance(spec, tuple):
                    kind, declared = spec[0], float(spec[1])
                else:
                    kind, declared = "value", float(spec)
                alias = _ALIASES[quantity]
                rows.append(
                    DeclaredNumber(
                        key=f"{prefix}_{alias}_{level}",
                        description=(
                            f"{prefix} {alias} ({quantity}) at subdivision {level}"
                        ),
                        source=source,
                        group=group,
                        quantity=f"sub{level}.{quantity}",
                        declared=declared,
                        tier="standard",
                        parameters={"subdivisions": _family_levels(prefix)},
                        tolerance=(
                            _KAPPA_TOLERANCE
                            if quantity == "condition_number"
                            else _TWO_FIGURE_TOLERANCE
                            if quantity in _TWO_FIGURE_CELLS
                            else DEFAULT_TOLERANCE
                        ),
                        kind=kind,
                    )
                )

    def merged(table, orders):
        """Cells and observed orders of one table, level by level."""
        levels = sorted(set(table) | set(orders))
        combined = {}
        for level in levels:
            cells = dict(table.get(level, {}))
            cells.update(orders.get(level, {}))
            combined[level] = cells
        return combined

    emit("hulme_hemisphere_cells", _HULME_SOURCE, merged(_HULME_TABLE, _HULME_ORDERS), "hulme")
    emit("submerged_sphere_cells", _SPHERE_SOURCE, merged(_SPHERE_TABLE, _SPHERE_ORDERS), "sphere")

    # --- table 11: CHIEF regularisation at the interior waterline resonance --
    chief_parameters = {
        "subdivisions": (int(_HULME_CHIEF_SUBDIVISION),),
        "ka": _HULME_CHIEF_KA,
    }
    chief_cells = {
        "exact_a33": 4.007e-1,
        "exact_b33": 6.31e-2,
        "unregularized.condition_number": 1.4566e2,
        "unregularized.a33_dimless": 3.481e-1,
        # The table prints a negative damping here: without the interior CHIEF
        # constraints the unregularised BIE radiates energy at the waterline
        # resonance. The number is stored as a magnitude plus a one-sided sign
        # row, because a signed value cannot be guarded against zero.
        "unregularized.b33_magnitude": 1.042e-1,
        "unregularized.passivity_violated": 1.0,
        "chief_regularized.condition_number": 4.24e0,
        "chief_regularized.a33_dimless": 3.919e-1,
        "chief_regularized.b33_dimless": 6.87e-2,
        # (No sign row for the regularised scheme: its damping is a positive
        # value row already, and the registry cannot carry a declared zero.)
    }
    for quantity, declared in chief_cells.items():
        rows.append(
            DeclaredNumber(
                key="hulme_chief_" + quantity.replace(".", "_"),
                description=(
                    f"Hulme hemisphere, Ka={_HULME_CHIEF_KA}, 256 panels: {quantity}"
                ),
                source=_CHIEF_SOURCE,
                group="hulme_chief_cells",
                quantity=quantity,
                declared=declared,
                tier="standard",
                parameters=chief_parameters,
                tolerance=(
                    _KAPPA_TOLERANCE
                    if quantity.endswith("condition_number")
                    else 1.0e-12
                    if quantity.endswith("passivity_violated")
                    else DEFAULT_TOLERANCE
                ),
            )
        )

    # --- closure statements quoted in the prose of sections 4.10 and 4.11 ---
    closure = (
        # (group, level, quantity, bound, where the sentence lives)
        ("hulme_hemisphere_cells", 3, "far_field_damping_rel_diff_heave", 5.45e-4,
         "section 4.10 prose: near-field vs Kochin damping difference at 256 panels"),
        ("hulme_hemisphere_cells", 3, "haskind_reciprocity_rel_diff_x", 2.09e-4,
         "section 4.10 prose: direct diffraction vs boundary Haskind, surge"),
        ("hulme_hemisphere_cells", 3, "haskind_reciprocity_rel_diff_z", 8.09e-4,
         "section 4.10 prose: direct diffraction vs boundary Haskind, heave"),
        ("submerged_sphere_cells", 2, "far_field_damping_rel_diff_heave", 4.0e-3,
         "section 4.11 prose: Kochin far-field damping conservation"),
        ("submerged_sphere_cells", 2, "haskind_reciprocity_rel_diff_x", 4.0e-3,
         "section 4.11 prose: Haskind reciprocity, surge"),
        ("submerged_sphere_cells", 2, "haskind_reciprocity_rel_diff_z", 4.0e-3,
         "section 4.11 prose: Haskind reciprocity, heave"),
    )
    for group, level, quantity, bound, sentence in closure:
        prefix = "hulme" if group == "hulme_hemisphere_cells" else "sphere"
        rows.append(
            DeclaredNumber(
                key=f"{prefix}_{_ALIASES[quantity]}_{level}_bound",
                description=sentence,
                source=(_HULME_SOURCE if group == "hulme_hemisphere_cells" else _SPHERE_SOURCE)
                + " (prose)",
                group=group,
                quantity=f"sub{level}.{quantity}",
                declared=bound,
                tier="standard",
                parameters={"subdivisions": _family_levels(prefix)},
                kind="upper",
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
    # --- 3-D floating hemisphere, Hulme (1982) reference (free surface) -----
    # Source: docs/marine_bem_benchmarks.md table 10 (Ka=1.0, a=1, rho=1025).
    DeclaredNumber(
        key="hulme_hemisphere_16_heave_added_mass_error",
        description="A33' relative error at 16 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub1_heave_added_mass_rel_error",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=1.20e-1,
    ),
    DeclaredNumber(
        key="hulme_hemisphere_16_heave_damping_error",
        description="B33' relative error at 16 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub1_heave_damping_rel_error",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=1.22e-1,
    ),
    DeclaredNumber(
        key="hulme_hemisphere_16_surge_added_mass_error",
        description="A11' relative error at 16 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub1_surge_added_mass_rel_error",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=1.80e-1,
    ),
    DeclaredNumber(
        key="hulme_hemisphere_16_surge_damping_error",
        description="B11' relative error at 16 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub1_surge_damping_rel_error",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=4.02e-1,
    ),
    DeclaredNumber(
        key="hulme_hemisphere_16_vertical_force_error",
        description="Fz relative error at 16 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub1_excitation_force_z_rel_error",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=6.67e-2,
    ),
    DeclaredNumber(
        key="hulme_hemisphere_16_condition_number",
        description="condition number at 16 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub1_condition_number",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=3.893,
    ),
    DeclaredNumber(
        key="hulme_hemisphere_64_heave_added_mass_error",
        description="A33' relative error at 64 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub2_heave_added_mass_rel_error",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=2.97e-2,
    ),
    DeclaredNumber(
        key="hulme_hemisphere_64_heave_damping_error",
        description="B33' relative error at 64 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub2_heave_damping_rel_error",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=4.50e-2,
    ),
    DeclaredNumber(
        key="hulme_hemisphere_64_surge_added_mass_error",
        description="A11' relative error at 64 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub2_surge_added_mass_rel_error",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=5.05e-2,
    ),
    DeclaredNumber(
        key="hulme_hemisphere_64_surge_damping_error",
        description="B11' relative error at 64 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub2_surge_damping_rel_error",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=1.30e-1,
    ),
    DeclaredNumber(
        key="hulme_hemisphere_64_vertical_force_error",
        description="Fz relative error at 64 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub2_excitation_force_z_rel_error",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=2.21e-2,
    ),
    DeclaredNumber(
        key="hulme_hemisphere_64_condition_number",
        description="condition number at 64 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub2_condition_number",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=3.052,
    ),
    DeclaredNumber(
        key="hulme_hemisphere_64_order_a33",
        description="observed order of A33', 16 -> 64 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub2_observed_order_a33",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=2.23,
    ),
    DeclaredNumber(
        key="hulme_hemisphere_64_order_b33",
        description="observed order of B33', 16 -> 64 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub2_observed_order_b33",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=1.59,
    ),
    DeclaredNumber(
        key="hulme_hemisphere_64_order_a11",
        description="observed order of A11', 16 -> 64 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub2_observed_order_a11",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=2.04,
    ),
    DeclaredNumber(
        key="hulme_hemisphere_64_order_b11",
        description="observed order of B11', 16 -> 64 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub2_observed_order_b11",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=1.8021,
    ),
    DeclaredNumber(
        key="hulme_hemisphere_64_order_fz",
        description="observed order of Fz, 16 -> 64 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub2_observed_order_fz",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=1.77,
    ),
    DeclaredNumber(
        key="hulme_hemisphere_256_heave_added_mass_error",
        description="A33' relative error at 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub3_heave_added_mass_rel_error",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=7.41e-3,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_hemisphere_256_heave_damping_error",
        description="B33' relative error at 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub3_heave_damping_rel_error",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=1.18e-2,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_hemisphere_256_surge_added_mass_error",
        description="A11' relative error at 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub3_surge_added_mass_rel_error",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=1.30e-2,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_hemisphere_256_surge_damping_error",
        description="B11' relative error at 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub3_surge_damping_rel_error",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=3.53e-2,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_hemisphere_256_vertical_force_error",
        description="Fz relative error at 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub3_excitation_force_z_rel_error",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=5.67e-3,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_hemisphere_256_condition_number",
        description="condition number at 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub3_condition_number",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=2.783,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_hemisphere_256_order_a33",
        description="observed order of A33', 64 -> 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub3_observed_order_a33",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=2.06,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_hemisphere_256_order_b33",
        description="observed order of B33', 64 -> 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub3_observed_order_b33",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=1.98,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_hemisphere_256_order_a11",
        description="observed order of A11', 64 -> 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub3_observed_order_a11",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=2.01,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_hemisphere_256_order_b11",
        description="observed order of B11', 64 -> 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub3_observed_order_b11",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=1.9377,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_hemisphere_256_order_fz",
        description="observed order of Fz, 64 -> 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 10",
        group="hulme_hemisphere",
        quantity="sub3_observed_order_fz",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=2.02,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_hemisphere_256_near_far_damping_closure",
        description="near-field vs Kochin far-field heave damping difference at 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 running text",
        group="hulme_hemisphere",
        quantity="sub3_far_field_damping_rel_diff_heave",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=5.45e-4,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_hemisphere_256_haskind_surge",
        description="Haskind reciprocity difference, surge, 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 running text",
        group="hulme_hemisphere",
        quantity="sub3_haskind_reciprocity_rel_diff_x",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=2.09e-4,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_hemisphere_256_haskind_heave",
        description="Haskind reciprocity difference, heave, 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 running text",
        group="hulme_hemisphere",
        quantity="sub3_haskind_reciprocity_rel_diff_z",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=8.09e-4,
        tier="standard",
    ),
    # --- 3-D irregular-frequency (Ka ~ 2.58) CHIEF comparison --------------
    # Source: docs/marine_bem_benchmarks.md table 11 (Ka=2.58, N=256).
    DeclaredNumber(
        key="hulme_chief_2_58_unregularized_condition_number",
        description="unregularized BIE condition number at Ka=2.58, 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 11",
        group="hulme_chief",
        quantity="ka2.58_unregularized_condition_number",
        parameters={"subdivision": 3, "ka_values": [2.58]},
        declared=145.66,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_chief_2_58_unregularized_a33",
        description="unregularized A33' at Ka=2.58, 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 11",
        group="hulme_chief",
        quantity="ka2.58_unregularized_a33_dimless",
        parameters={"subdivision": 3, "ka_values": [2.58]},
        declared=0.3481,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_chief_2_58_unregularized_b33",
        description="unregularized B33' at Ka=2.58, 256 panels (non-physical negative)",
        source="docs/marine_bem_benchmarks.md section 4.10 table 11",
        group="hulme_chief",
        quantity="ka2.58_unregularized_b33_dimless",
        parameters={"subdivision": 3, "ka_values": [2.58]},
        declared=-0.1042,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_chief_2_58_chief_condition_number",
        description="CHIEF-regularized condition number at Ka=2.58, 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 11",
        group="hulme_chief",
        quantity="ka2.58_chief_regularized_condition_number",
        parameters={"subdivision": 3, "ka_values": [2.58]},
        declared=4.24,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_chief_2_58_chief_a33",
        description="CHIEF-regularized A33' at Ka=2.58, 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 11",
        group="hulme_chief",
        quantity="ka2.58_chief_regularized_a33_dimless",
        parameters={"subdivision": 3, "ka_values": [2.58]},
        declared=0.3919,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_chief_2_58_chief_b33",
        description="CHIEF-regularized B33' at Ka=2.58, 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 table 11",
        group="hulme_chief",
        quantity="ka2.58_chief_regularized_b33_dimless",
        parameters={"subdivision": 3, "ka_values": [2.58]},
        declared=0.0687,
        tier="standard",
    ),
    # --- 3-D submerged sphere, Wang (1986) / Linton (1991) reference -------
    # Source: docs/marine_bem_benchmarks.md table 12 (f/a=1.5, Ka=0.8).
    DeclaredNumber(
        key="submerged_sphere_20_heave_added_mass_error",
        description="A33' relative error at 20 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 table 12",
        group="submerged_sphere",
        quantity="sub0_heave_added_mass_rel_error",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5},
        declared=2.55e-1,
    ),
    DeclaredNumber(
        key="submerged_sphere_20_surge_added_mass_error",
        description="A11' relative error at 20 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 table 12",
        group="submerged_sphere",
        quantity="sub0_surge_added_mass_rel_error",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5},
        declared=2.68e-1,
    ),
    DeclaredNumber(
        key="submerged_sphere_20_force_x_error",
        description="horizontal excitation force relative error at 20 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 table 12",
        group="submerged_sphere",
        quantity="sub0_excitation_force_x_rel_error",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5},
        declared=3.60e-1,
    ),
    DeclaredNumber(
        key="submerged_sphere_20_force_z_error",
        description="vertical excitation force relative error at 20 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 table 12",
        group="submerged_sphere",
        quantity="sub0_excitation_force_z_rel_error",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5},
        declared=3.70e-1,
    ),
    DeclaredNumber(
        key="submerged_sphere_20_condition_number",
        description="condition number at 20 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 table 12",
        group="submerged_sphere",
        quantity="sub0_condition_number",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5},
        declared=2.407,
    ),
    DeclaredNumber(
        key="submerged_sphere_80_heave_added_mass_error",
        description="A33' relative error at 80 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 table 12",
        group="submerged_sphere",
        quantity="sub1_heave_added_mass_rel_error",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5},
        declared=7.30e-2,
    ),
    DeclaredNumber(
        key="submerged_sphere_80_surge_added_mass_error",
        description="A11' relative error at 80 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 table 12",
        group="submerged_sphere",
        quantity="sub1_surge_added_mass_rel_error",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5},
        declared=8.02e-2,
    ),
    DeclaredNumber(
        key="submerged_sphere_80_force_x_error",
        description="horizontal excitation force relative error at 80 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 table 12",
        group="submerged_sphere",
        quantity="sub1_excitation_force_x_rel_error",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5},
        declared=1.16e-1,
    ),
    DeclaredNumber(
        key="submerged_sphere_80_force_z_error",
        description="vertical excitation force relative error at 80 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 table 12",
        group="submerged_sphere",
        quantity="sub1_excitation_force_z_rel_error",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5},
        declared=1.19e-1,
    ),
    DeclaredNumber(
        key="submerged_sphere_80_condition_number",
        description="condition number at 80 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 table 12",
        group="submerged_sphere",
        quantity="sub1_condition_number",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5},
        declared=2.259,
    ),
    DeclaredNumber(
        key="submerged_sphere_80_rotational_coupling",
        description="sphere-centre rotational coupling ratio at 80 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 table 12",
        group="submerged_sphere",
        quantity="sub1_rotational_added_mass_max_ratio",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5},
        declared=2.1510e-04,
    ),
    DeclaredNumber(
        key="submerged_sphere_80_order_a33",
        description="observed order of A33', 20 -> 80 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 table 12",
        group="submerged_sphere",
        quantity="sub1_observed_order_a33",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5},
        declared=2.12,
    ),
    DeclaredNumber(
        key="submerged_sphere_80_order_b33",
        description="observed order of B33', 20 -> 80 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 table 12",
        group="submerged_sphere",
        quantity="sub1_observed_order_b33",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5},
        declared=1.72,
    ),
    DeclaredNumber(
        key="submerged_sphere_80_order_a11",
        description="observed order of A11', 20 -> 80 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 table 12",
        group="submerged_sphere",
        quantity="sub1_observed_order_a11",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5},
        declared=2.04,
    ),
    DeclaredNumber(
        key="submerged_sphere_80_order_fz",
        description="observed order of Fz, 20 -> 80 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 table 12",
        group="submerged_sphere",
        quantity="sub1_observed_order_fz",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5},
        declared=1.91,
    ),
    DeclaredNumber(
        key="submerged_sphere_320_heave_added_mass_error",
        description="A33' relative error at 320 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 table 12",
        group="submerged_sphere",
        quantity="sub2_heave_added_mass_rel_error",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5},
        declared=1.85e-2,
        tier="standard",
    ),
    DeclaredNumber(
        key="submerged_sphere_320_surge_added_mass_error",
        description="A11' relative error at 320 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 table 12",
        group="submerged_sphere",
        quantity="sub2_surge_added_mass_rel_error",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5},
        declared=2.08e-2,
        tier="standard",
    ),
    DeclaredNumber(
        key="submerged_sphere_320_force_x_error",
        description="horizontal excitation force relative error at 320 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 table 12",
        group="submerged_sphere",
        quantity="sub2_excitation_force_x_rel_error",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5},
        declared=3.10e-2,
        tier="standard",
    ),
    DeclaredNumber(
        key="submerged_sphere_320_force_z_error",
        description="vertical excitation force relative error at 320 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 table 12",
        group="submerged_sphere",
        quantity="sub2_excitation_force_z_rel_error",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5},
        declared=3.20e-2,
        tier="standard",
    ),
    DeclaredNumber(
        key="submerged_sphere_320_condition_number",
        description="condition number at 320 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 table 12",
        group="submerged_sphere",
        quantity="sub2_condition_number",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5},
        declared=2.188,
        tier="standard",
    ),
    DeclaredNumber(
        key="submerged_sphere_320_rotational_coupling",
        description="sphere-centre rotational coupling ratio at 320 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 table 12",
        group="submerged_sphere",
        quantity="sub2_rotational_added_mass_max_ratio",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5},
        declared=2.6481e-05,
        tier="standard",
    ),
    DeclaredNumber(
        key="submerged_sphere_320_order_a33",
        description="observed order of A33', 80 -> 320 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 table 12",
        group="submerged_sphere",
        quantity="sub2_observed_order_a33",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5},
        declared=2.06,
        tier="standard",
    ),
    DeclaredNumber(
        key="submerged_sphere_320_order_b33",
        description="observed order of B33', 80 -> 320 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 table 12",
        group="submerged_sphere",
        quantity="sub2_observed_order_b33",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5},
        declared=1.93,
        tier="standard",
    ),
    DeclaredNumber(
        key="submerged_sphere_320_order_a11",
        description="observed order of A11', 80 -> 320 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 table 12",
        group="submerged_sphere",
        quantity="sub2_observed_order_a11",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5},
        declared=2.03,
        tier="standard",
    ),
    DeclaredNumber(
        key="submerged_sphere_320_order_fz",
        description="observed order of Fz, 80 -> 320 panels",
        source="docs/marine_bem_benchmarks.md section 4.11 table 12",
        group="submerged_sphere",
        quantity="sub2_observed_order_fz",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5},
        declared=1.98,
        tier="standard",
    ),
) + _free_surface_3d_rows()

_URSELL_RADIUS = 1.0
_URSELL_SUBMERGENCE = 2.0
_URSELL_KA = 0.6
_URSELL_DENSITY = 1025.0
_URSELL_GRAVITY = 9.80665


def _study_rows(study: dict) -> dict[str, float]:
    """Flatten a mesh-refinement study into ``sub<level>_<quantity>`` scalars.

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


def _level_quantities(levels: Sequence[dict]) -> dict[str, float]:
    """Flatten a refinement study into ``sub<N>.<quantity>`` keys.

    The 3-D free-surface benchmarks return one row per mesh level; the registry
    quotes individual cells, so every numeric field of every level is exposed
    under a prefix. Fields the benchmark leaves undefined at the coarsest level
    (observed orders need two meshes) are simply absent.
    """
    values: dict[str, float] = {}
    for row in levels:
        level = int(row["subdivisions"])
        for key, value in row.items():
            if key in ("subdivisions",) or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                values[f"sub{level}.{key}"] = float(value)
    return values

#: One refinement study per parametric setting. The registry reads the three
#: 3-D free-surface tables twice -- keyed by panel count (m3150) and keyed by the
#: printed cell (m3264) -- so the second reading must not pay for a second
#: solve. The studies are plain dictionaries and every reader treats them as
#: read-only, which is what makes sharing them safe.
_STUDY_CACHE: dict[tuple, object] = {}


def _shared_study(key: tuple, run: Callable[[], object]) -> object:
    """Return the study for ``key``, running it on first use."""
    if key not in _STUDY_CACHE:
        _STUDY_CACHE[key] = run()
    return _STUDY_CACHE[key]


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
        subdivisions = tuple(
            int(n) for n in parameters.get("subdivisions", (1, 2, 3))
        )
        ka = float(parameters.get("ka", _HULME_KA))
        study = _shared_study(
            ("hulme", subdivisions, ka),
            lambda: hemisphere_benchmark.run_refinement_study(
                subdivisions=subdivisions, ka=ka
            ),
        )
        return _study_rows(study)
    if group == "submerged_sphere":
        subdivisions = tuple(
            int(n) for n in parameters.get("subdivisions", (0, 1, 2))
        )
        ka = float(parameters.get("ka", _SPHERE_KA))
        submergence = float(parameters.get("submergence", _SPHERE_SUBMERGENCE))
        study = _shared_study(
            ("sphere", subdivisions, ka, submergence),
            lambda: sphere_benchmark.run_refinement_study(
                subdivisions=subdivisions, ka=ka, submergence=submergence
            ),
        )
        return _study_rows(study)
    if group == "hulme_chief":
        subdivision = int(parameters.get("subdivision", _HULME_CHIEF_SUBDIVISION))
        ka_values = tuple(
            float(v)
            for v in parameters.get("ka_values", (_HULME_CHIEF_KA,))
        )
        rows = _shared_study(
            ("hulme_chief", subdivision, ka_values),
            lambda: hemisphere_benchmark.run_irregular_frequency_study(
                subdivision=subdivision, ka_values=ka_values
            ),
        )
        return _chief_rows(rows)
    if group == "hulme_hemisphere_cells":
        subdivisions = tuple(int(n) for n in parameters["subdivisions"])
        study = _shared_study(
            ("hulme", subdivisions, _HULME_KA),
            lambda: floating_hemisphere_benchmark.run_refinement_study(
                subdivisions=subdivisions,
                ka=_HULME_KA,
                radius=_HULME_RADIUS,
                density=_HULME_DENSITY,
                gravity=_HULME_GRAVITY,
            ),
        )
        return _level_quantities(study["levels"])
    if group == "hulme_chief_cells":
        subdivision = int(parameters["subdivisions"][0])
        ka_values = (float(parameters.get("ka", _HULME_CHIEF_KA)),)
        rows = _shared_study(
            ("hulme_chief", subdivision, ka_values),
            lambda: floating_hemisphere_benchmark.run_irregular_frequency_study(
                subdivision=subdivision,
                ka_values=ka_values,
                radius=_HULME_RADIUS,
                density=_HULME_DENSITY,
                gravity=_HULME_GRAVITY,
            ),
        )
        row = rows[0]
        values = {
            "exact_a33": float(row["exact_a33"]),
            "exact_b33": float(row["exact_b33"]),
        }
        for scheme in ("unregularized", "chief_regularized"):
            for key in ("condition_number", "a33_dimless", "b33_dimless"):
                values[f"{scheme}.{key}"] = float(row[scheme][key])
            # The sign of the damping is the physical claim of table 11, and the
            # registry stores magnitudes, so it is carried as its own quantity:
            # 1.0 means "this scheme radiates energy into the fluid".
            values[f"{scheme}.passivity_violated"] = float(
                bool(row[scheme]["passivity_violated"])
            )
            values[f"{scheme}.b33_magnitude"] = abs(float(row[scheme]["b33_dimless"]))
        return values
    if group == "submerged_sphere_cells":
        subdivisions = tuple(int(n) for n in parameters["subdivisions"])
        study = _shared_study(
            ("sphere", subdivisions, _SPHERE_KA, _SPHERE_SUBMERGENCE),
            lambda: submerged_sphere_benchmark.run_refinement_study(
                subdivisions=subdivisions,
                ka=_SPHERE_KA,
                radius=_SPHERE_RADIUS,
                submergence=_SPHERE_SUBMERGENCE,
                density=_HULME_DENSITY,
                gravity=_HULME_GRAVITY,
            ),
        )
        return _level_quantities(study["levels"])
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
    # Upper-bound rows are judged on one side only, so their relative deviation
    # is a distance from the bound, not an error; the headline accuracy figure
    # is therefore reported over the equality rows.
    worst_value = max(
        (
            abs(row["relative_deviation"])
            for row in rows
            if row["comparison"] == "value"
        ),
        default=0.0,
    )
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
            "worst_value_relative_deviation": worst_value,
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