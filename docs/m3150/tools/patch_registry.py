"""Register the 3-D free-surface documented numbers (m3150).

One-shot, idempotent patch: inserts the ``hulme_hemisphere``, ``hulme_chief``
and ``submerged_sphere`` groups into ``tensorbem.benchmarks.declared_numbers``
together with their runner branches. Run once from the repository root; a second
run is a no-op because each anchor disappears after the insert.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / "src" / "tensorbem" / "benchmarks" / "declared_numbers.py"

IMPORT_OLD = """    interior_harmonics as interior_harmonics_benchmark,
    ursell_cylinder as ursell_benchmark,
)"""
IMPORT_NEW = """    floating_hemisphere as hemisphere_benchmark,
    interior_harmonics as interior_harmonics_benchmark,
    submerged_sphere_wave as sphere_benchmark,
    ursell_cylinder as ursell_benchmark,
)"""

SPEC_NOTE = '''
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


'''

RUNNER_OLD = """    raise ValueError(f"unknown benchmark group: {group!r}")"""
RUNNER_NEW = '''    if group == "hulme_hemisphere":
        study = hemisphere_benchmark.run_refinement_study(
            subdivisions=tuple(int(n) for n in parameters.get("subdivisions", (1, 2, 3))),
            ka=float(parameters.get("ka", 1.0)),
        )
        return _study_rows(study)
    if group == "submerged_sphere":
        study = sphere_benchmark.run_refinement_study(
            subdivisions=tuple(int(n) for n in parameters.get("subdivisions", (0, 1, 2))),
            ka=float(parameters.get("ka", 0.8)),
            submergence=float(parameters.get("submergence", 1.5)),
        )
        return _study_rows(study)
    if group == "hulme_chief":
        rows = hemisphere_benchmark.run_irregular_frequency_study(
            subdivision=int(parameters.get("subdivision", 3)),
            ka_values=tuple(float(v) for v in parameters.get("ka_values", (2.58,))),
        )
        return _chief_rows(rows)
    raise ValueError(f"unknown benchmark group: {group!r}")'''

REGISTRY_OLD = """        declared=1.824e-4,
        tier="standard",
    ),
)"""
REGISTRY_NEW = '''        declared=1.824e-4,
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
        declared=1.62,
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
        declared=1.88,
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
        declared=2.2e-4,
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
        declared=2.6e-5,
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
)'''


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    if "_study_rows" in text:
        print("already patched")
        return
    for old, new in (
        (IMPORT_OLD, IMPORT_NEW),
        (REGISTRY_OLD, REGISTRY_NEW),
        (RUNNER_OLD, RUNNER_NEW),
    ):
        assert text.count(old) == 1, f"anchor not unique: {old[:40]!r}"
        text = text.replace(old, new)
    anchor = "\ndef _run_group("
    assert text.count(anchor) == 1
    text = text.replace(anchor, SPEC_NOTE + "def _run_group(")
    TARGET.write_text(text, encoding="utf-8")
    print("patched", TARGET)


if __name__ == "__main__":
    main()