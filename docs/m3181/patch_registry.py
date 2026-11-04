"""Insert the 3-D free-surface tables into the declared-number registry.

Exact-anchor text replacement (no regex), followed by a syntax/logic check.
"""
from __future__ import annotations

from pathlib import Path

SRC = Path("src/tensorbem/benchmarks/declared_numbers.py")
text = SRC.read_text(encoding="utf-8")

# ---------------------------------------------------------------- 1. imports
anchor = """from tensorbem.benchmarks import (
    cylinder_added_mass as cylinder_benchmark,
    ellipsoid_added_mass as ellipsoid_benchmark,
    interior_harmonics as interior_harmonics_benchmark,
    ursell_cylinder as ursell_benchmark,
)"""
assert text.count(anchor) == 1
new_imports = """from tensorbem.benchmarks import (
    cylinder_added_mass as cylinder_benchmark,
    ellipsoid_added_mass as ellipsoid_benchmark,
    floating_hemisphere as hulme_benchmark,
    interior_harmonics as interior_harmonics_benchmark,
    submerged_sphere_wave as wang_linton_benchmark,
    ursell_cylinder as ursell_benchmark,
)"""
text = text.replace(anchor, new_imports)

# ------------------------------------------------- 2. row factory + the tables
helper = '''
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
#: ``rotational_added_mass_max_ratio`` is quoted as machine zero, so it is
#: registered as an upper bound rather than an equality claim.
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


def _refinement_rows(prefix, table, *, group, source, parameters, upper=()):
    """One :class:`DeclaredNumber` per quoted table cell."""
    rows = []
    for subdivisions, panels, quantity, declared in table:
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
                tolerance=_REFINEMENT_TOLERANCE,
                kind="upper" if quantity in upper else "value",
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
        )
    )


#: The registry.'''

anchor = "#: The registry."
assert text.count(anchor) == 1
text = text.replace(anchor, helper.lstrip("\n"), 1)

# ------------------------------------------- 3. append the new rows to the tuple
anchor = """        declared=1.824e-4,
        tier="standard",
    ),
)"""
assert text.count(anchor) == 1
text = text.replace(
    anchor,
    """        declared=1.824e-4,
        tier="standard",
    ),
) + _free_surface_3d_rows()""",
)

# ------------------------------------------------------- 4. the group runners
anchor = '''    raise ValueError(f"unknown benchmark group: {group!r}")'''
assert text.count(anchor) == 1
runners = '''    if group == "hulme3d":
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
            values[f"{tag}_exact_a33"] = float(row["exact_a33"])
            values[f"{tag}_exact_b33"] = float(row["exact_b33"])
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
    raise ValueError(f"unknown benchmark group: {group!r}")'''
text = text.replace(anchor, runners, 1)

SRC.write_text(text, encoding="utf-8")
print("patched", SRC)