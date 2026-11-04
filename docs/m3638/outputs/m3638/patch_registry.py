"""Idempotent patch: register the 3-D free-surface doc tables (m3638).

Anchored exact-string insertion into src/tensorbem/benchmarks/declared_numbers.py.
Every anchor must occur exactly once; the script is a no-op if already applied.
"""

from pathlib import Path

PATH = Path("src/tensorbem/benchmarks/declared_numbers.py")
text = PATH.read_text()

# ---------------------------------------------------------------- imports
IMPORT_ANCHOR = """from tensorbem.benchmarks import (
    cylinder_added_mass as cylinder_benchmark,
    ellipsoid_added_mass as ellipsoid_benchmark,
    interior_harmonics as interior_harmonics_benchmark,
    ursell_cylinder as ursell_benchmark,
)"""
IMPORT_NEW = """from tensorbem.benchmarks import (
    cylinder_added_mass as cylinder_benchmark,
    ellipsoid_added_mass as ellipsoid_benchmark,
    floating_hemisphere as hulme_benchmark,
    interior_harmonics as interior_harmonics_benchmark,
    submerged_sphere_wave as linton_benchmark,
    ursell_cylinder as ursell_benchmark,
)"""

# ---------------------------------------------------------------- group docstring
GROUP_ANCHOR = '''    """Which benchmark to run: ``cylinder``, ``ellipsoid``, ``ursell`` or
    ``ursell_reference``."""'''
GROUP_NEW = '''    """Which benchmark to run: ``cylinder``, ``ellipsoid``, ``ursell``,
    ``ursell_reference``, ``interior_harmonics``, ``hulme``, ``hulme_chief``
    or ``submerged_sphere``."""'''

# ---------------------------------------------------------------- registry body
REG_ANCHOR = """        declared=1.824e-4,
        tier="standard",
    ),
)"""
REG_NEW = '''        declared=1.824e-4,
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
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=3.893,
    ),
    DeclaredNumber(
        key="hulme_heave_added_mass_16",
        description="heave added-mass relative error, 16 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="heave_added_mass_rel_error@1",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=1.20e-1,
    ),
    DeclaredNumber(
        key="hulme_heave_damping_16",
        description="heave damping relative error, 16 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="heave_damping_rel_error@1",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=1.22e-1,
    ),
    DeclaredNumber(
        key="hulme_surge_added_mass_16",
        description="surge added-mass relative error, 16 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="surge_added_mass_rel_error@1",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=1.80e-1,
    ),
    DeclaredNumber(
        key="hulme_surge_damping_16",
        description="surge damping relative error, 16 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="surge_damping_rel_error@1",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=4.02e-1,
    ),
    DeclaredNumber(
        key="hulme_excitation_z_16",
        description="vertical wave-force relative error, 16 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="excitation_force_z_rel_error@1",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=6.67e-2,
    ),
    DeclaredNumber(
        key="hulme_condition_number_64",
        description="condition number of the exterior BIE, 64 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="condition_number@2",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=3.052,
    ),
    DeclaredNumber(
        key="hulme_heave_added_mass_64",
        description="heave added-mass relative error, 64 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="heave_added_mass_rel_error@2",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=2.97e-2,
    ),
    DeclaredNumber(
        key="hulme_condition_number_256",
        description="condition number of the exterior BIE, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="condition_number@3",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=2.783,
    ),
    DeclaredNumber(
        key="hulme_heave_added_mass_256",
        description="heave added-mass relative error, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="heave_added_mass_rel_error@3",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=7.41e-3,
    ),
    DeclaredNumber(
        key="hulme_heave_damping_256",
        description="heave damping relative error, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="heave_damping_rel_error@3",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=1.18e-2,
    ),
    DeclaredNumber(
        key="hulme_surge_added_mass_256",
        description="surge added-mass relative error, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="surge_added_mass_rel_error@3",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=1.30e-2,
    ),
    DeclaredNumber(
        key="hulme_surge_damping_256",
        description="surge damping relative error, 256 panels (slowest quantity)",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="surge_damping_rel_error@3",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=3.53e-2,
    ),
    DeclaredNumber(
        key="hulme_excitation_z_256",
        description="vertical wave-force relative error, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="excitation_force_z_rel_error@3",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=5.67e-3,
    ),
    DeclaredNumber(
        key="hulme_order_heave_added_mass_256",
        description="observed order in h, heave added mass, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="observed_order_a33@3",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=2.06,
    ),
    DeclaredNumber(
        key="hulme_order_heave_damping_256",
        description="observed order in h, heave damping, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="observed_order_b33@3",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=1.98,
    ),
    DeclaredNumber(
        key="hulme_order_surge_added_mass_256",
        description="observed order in h, surge added mass, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="observed_order_a11@3",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=2.01,
    ),
    DeclaredNumber(
        key="hulme_order_excitation_z_256",
        description="observed order in h, vertical wave force, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10",
        group="hulme",
        quantity="observed_order_fz@3",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=2.02,
    ),
    DeclaredNumber(
        key="hulme_far_field_damping_256",
        description="near-field vs far-field (Kochin) heave damping, 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 prose",
        group="hulme",
        quantity="far_field_damping_rel_diff_heave@3",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=5.45e-4,
    ),
    DeclaredNumber(
        key="hulme_haskind_surge_256",
        description="direct diffraction vs boundary Haskind, surge, 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 prose",
        group="hulme",
        quantity="haskind_reciprocity_rel_diff_x@3",
        parameters={"subdivisions": [1, 2, 3], "ka": 1.0},
        declared=2.09e-4,
    ),
    DeclaredNumber(
        key="hulme_haskind_heave_256",
        description="direct diffraction vs boundary Haskind, heave, 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 prose",
        group="hulme",
        quantity="haskind_reciprocity_rel_diff_z@3",
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
        parameters={"subdivision": 3, "ka": 2.58},
        declared=145.66,
    ),
    DeclaredNumber(
        key="hulme_chief_heave_added_mass_unregularized",
        description="A33' without CHIEF, Ka=2.58 (exact 0.4007)",
        source="docs/marine_bem_benchmarks.md table 11",
        group="hulme_chief",
        quantity="unregularized_a33_dimless",
        parameters={"subdivision": 3, "ka": 2.58},
        declared=0.3481,
    ),
    DeclaredNumber(
        key="hulme_chief_heave_damping_unregularized",
        description="B33' without CHIEF, Ka=2.58 (negative: passivity broken)",
        source="docs/marine_bem_benchmarks.md table 11",
        group="hulme_chief",
        quantity="unregularized_b33_dimless",
        parameters={"subdivision": 3, "ka": 2.58},
        declared=-0.1042,
    ),
    DeclaredNumber(
        key="hulme_chief_condition_augmented",
        description="condition number with 3-D interior CHIEF points, Ka=2.58",
        source="docs/marine_bem_benchmarks.md table 11",
        group="hulme_chief",
        quantity="chief_regularized_condition_number",
        parameters={"subdivision": 3, "ka": 2.58},
        declared=4.24,
    ),
    DeclaredNumber(
        key="hulme_chief_heave_added_mass_augmented",
        description="A33' with CHIEF, Ka=2.58 (exact 0.4007)",
        source="docs/marine_bem_benchmarks.md table 11",
        group="hulme_chief",
        quantity="chief_regularized_a33_dimless",
        parameters={"subdivision": 3, "ka": 2.58},
        declared=0.3919,
    ),
    DeclaredNumber(
        key="hulme_chief_heave_damping_augmented",
        description="B33' with CHIEF, Ka=2.58 (positive: passivity restored)",
        source="docs/marine_bem_benchmarks.md table 11",
        group="hulme_chief",
        quantity="chief_regularized_b33_dimless",
        parameters={"subdivision": 3, "ka": 2.58},
        declared=0.0687,
    ),
    DeclaredNumber(
        key="hulme_chief_point_count",
        description="number of automatic interior CHIEF points",
        source="docs/marine_bem_benchmarks.md table 11",
        group="hulme_chief",
        quantity="chief_regularized_chief_point_count",
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
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=2.407,
    ),
    DeclaredNumber(
        key="linton_heave_added_mass_20",
        description="heave added-mass relative error, 20 panels",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="heave_added_mass_rel_error@0",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=2.55e-1,
    ),
    DeclaredNumber(
        key="linton_surge_added_mass_20",
        description="surge added-mass relative error, 20 panels",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="surge_added_mass_rel_error@0",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=2.68e-1,
    ),
    DeclaredNumber(
        key="linton_excitation_x_20",
        description="horizontal wave-force relative error, 20 panels",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="excitation_force_x_rel_error@0",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=3.60e-1,
    ),
    DeclaredNumber(
        key="linton_excitation_z_20",
        description="vertical wave-force relative error, 20 panels",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="excitation_force_z_rel_error@0",
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
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=2.188,
    ),
    DeclaredNumber(
        key="linton_heave_added_mass_320",
        description="heave added-mass relative error, 320 panels",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="heave_added_mass_rel_error@2",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=1.85e-2,
    ),
    DeclaredNumber(
        key="linton_surge_added_mass_320",
        description="surge added-mass relative error, 320 panels",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="surge_added_mass_rel_error@2",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=2.08e-2,
    ),
    DeclaredNumber(
        key="linton_excitation_x_320",
        description="horizontal wave-force relative error, 320 panels",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="excitation_force_x_rel_error@2",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=3.10e-2,
    ),
    DeclaredNumber(
        key="linton_excitation_z_320",
        description="vertical wave-force relative error, 320 panels",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="excitation_force_z_rel_error@2",
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
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=2.06,
    ),
    DeclaredNumber(
        key="linton_order_heave_damping_320",
        description="observed order in h, heave damping, 320 panels",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="observed_order_b33@2",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=1.93,
    ),
    DeclaredNumber(
        key="linton_order_surge_added_mass_320",
        description="observed order in h, surge added mass, 320 panels",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="observed_order_a11@2",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=2.03,
    ),
    DeclaredNumber(
        key="linton_order_excitation_z_320",
        description="observed order in h, vertical wave force, 320 panels",
        source="docs/marine_bem_benchmarks.md table 12",
        group="submerged_sphere",
        quantity="observed_order_fz@2",
        parameters={"subdivisions": [0, 1, 2], "ka": 0.8},
        declared=1.98,
    ),
)'''

# ---------------------------------------------------------------- runners
RUN_ANCHOR = '''    raise ValueError(f"unknown benchmark group: {group!r}")'''
RUN_NEW = '''    if group in {"hulme", "submerged_sphere"}:
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
    raise ValueError(f"unknown benchmark group: {group!r}")'''

for anchor, new, label in (
    (IMPORT_ANCHOR, IMPORT_NEW, "imports"),
    (GROUP_ANCHOR, GROUP_NEW, "group docstring"),
    (REG_ANCHOR, REG_NEW, "registry"),
    (RUN_ANCHOR, RUN_NEW, "runners"),
):
    count = text.count(anchor)
    if count != 1:
        raise SystemExit(f"anchor {label!r} occurs {count} times, aborting")
    text = text.replace(anchor, new)

PATH.write_text(text)
print("patched", PATH, "->", len(text), "bytes")