"""One-shot patch: register the 3-D free-surface documented numbers.

Idempotent guard: raises if the markers are already present.
"""
from pathlib import Path

path = Path("src/tensorbem/benchmarks/declared_numbers.py")
text = path.read_text()

assert "hulme_hemisphere" not in text, "already patched"

# ---------------------------------------------------------------- imports
old = """    ursell_cylinder as ursell_benchmark,
)"""
new = """    floating_hemisphere as floating_hemisphere_benchmark,
    submerged_sphere_wave as submerged_sphere_benchmark,
    ursell_cylinder as ursell_benchmark,
)"""
assert old in text
text = text.replace(old, new, 1)

# ------------------------------------------------------- group dispatch
old = """    raise ValueError(f"unknown benchmark group: {group!r}")"""
new = '''    if group == "hulme_hemisphere":
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
            "chief_condition_number": float(chief["condition_number"]),
            "chief_added_mass_dimless": float(chief["a33_dimless"]),
            "chief_damping_dimless": float(chief["b33_dimless"]),
            "chief_point_count": float(chief["chief_point_count"]),
            "exact_added_mass_dimless": float(row["exact_a33"]),
            "exact_damping_dimless": float(row["exact_b33"]),
        }
    raise ValueError(f"unknown benchmark group: {group!r}")'''
assert old in text
text = text.replace(old, new, 1)

# ----------------------------------------------------------- helper
old = """def _run_group(group: str, parameters: dict) -> dict[str, float]:"""
new = '''def _finest_level(
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


def _run_group(group: str, parameters: dict) -> dict[str, float]:'''
assert old in text
text = text.replace(old, new, 1)

# --------------------------------------------------------- registry rows
old = """)

_URSELL_RADIUS = 1.0"""
new = '''    # --- 3-D free-surface: Hulme (1982) floating hemisphere, table 10 -----
    DeclaredNumber(
        key="hulme_heave_added_mass_error_256",
        description="A33' relative error of the floating hemisphere, Ka=1, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10 (section 4.10)",
        group="hulme_hemisphere",
        quantity="heave_added_mass_rel_error",
        parameters={"subdivisions": [1, 3], "ka": 1.0},
        declared=7.41e-3,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_heave_damping_error_256",
        description="B33' relative error of the floating hemisphere, Ka=1, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10 (section 4.10)",
        group="hulme_hemisphere",
        quantity="heave_damping_rel_error",
        parameters={"subdivisions": [1, 3], "ka": 1.0},
        declared=1.18e-2,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_surge_added_mass_error_256",
        description="A11' relative error of the floating hemisphere, Ka=1, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10 (section 4.10)",
        group="hulme_hemisphere",
        quantity="surge_added_mass_rel_error",
        parameters={"subdivisions": [1, 3], "ka": 1.0},
        declared=1.30e-2,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_surge_damping_error_256",
        description="B11' relative error of the floating hemisphere (slowest quantity), Ka=1, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10 (section 4.10)",
        group="hulme_hemisphere",
        quantity="surge_damping_rel_error",
        parameters={"subdivisions": [1, 3], "ka": 1.0},
        declared=3.53e-2,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_vertical_force_error_256",
        description="vertical excitation force relative error of the floating hemisphere, Ka=1, 256 panels",
        source="docs/marine_bem_benchmarks.md table 10 (section 4.10)",
        group="hulme_hemisphere",
        quantity="excitation_force_z_rel_error",
        parameters={"subdivisions": [1, 3], "ka": 1.0},
        declared=5.67e-3,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_condition_number_256",
        description="condition number of the CHIEF-augmented 256-panel hemisphere system",
        source="docs/marine_bem_benchmarks.md table 10 (section 4.10)",
        group="hulme_hemisphere",
        quantity="condition_number",
        parameters={"subdivisions": [1, 3], "ka": 1.0},
        declared=2.783,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_order_a33_256",
        description="observed order of A33' between 64 and 256 panels",
        source="docs/marine_bem_benchmarks.md table 10 (section 4.10)",
        group="hulme_hemisphere",
        quantity="observed_order_a33",
        parameters={"subdivisions": [1, 3], "ka": 1.0},
        declared=2.06,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_order_b33_256",
        description="observed order of B33' between 64 and 256 panels",
        source="docs/marine_bem_benchmarks.md table 10 (section 4.10)",
        group="hulme_hemisphere",
        quantity="observed_order_b33",
        parameters={"subdivisions": [1, 3], "ka": 1.0},
        declared=1.98,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_order_a11_256",
        description="observed order of A11' between 64 and 256 panels",
        source="docs/marine_bem_benchmarks.md table 10 (section 4.10)",
        group="hulme_hemisphere",
        quantity="observed_order_a11",
        parameters={"subdivisions": [1, 3], "ka": 1.0},
        declared=2.01,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_order_b11_256",
        description="observed order of B11' between 64 and 256 panels",
        source="docs/marine_bem_benchmarks.md table 10 (section 4.10)",
        group="hulme_hemisphere",
        quantity="observed_order_b11",
        parameters={"subdivisions": [1, 3], "ka": 1.0},
        declared=1.88,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_order_fz_256",
        description="observed order of the vertical excitation force between 64 and 256 panels",
        source="docs/marine_bem_benchmarks.md table 10 (section 4.10)",
        group="hulme_hemisphere",
        quantity="observed_order_fz",
        parameters={"subdivisions": [1, 3], "ka": 1.0},
        declared=2.02,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_near_far_damping_256",
        description="relative difference between near-field and 3-D Kochin far-field heave damping, 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 (text after table 10)",
        group="hulme_hemisphere",
        quantity="far_field_damping_rel_diff_heave",
        parameters={"subdivisions": [1, 3], "ka": 1.0},
        declared=5.45e-4,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_haskind_surge_256",
        description="direct-diffraction versus boundary Haskind discrepancy, surge, 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 (text after table 10)",
        group="hulme_hemisphere",
        quantity="haskind_reciprocity_rel_diff_x",
        parameters={"subdivisions": [1, 3], "ka": 1.0},
        declared=2.09e-4,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_haskind_heave_256",
        description="direct-diffraction versus boundary Haskind discrepancy, heave, 256 panels",
        source="docs/marine_bem_benchmarks.md section 4.10 (text after table 10)",
        group="hulme_hemisphere",
        quantity="haskind_reciprocity_rel_diff_z",
        parameters={"subdivisions": [1, 3], "ka": 1.0},
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

_URSELL_RADIUS = 1.0'''
assert old in text
text = text.replace(old, new, 1)

path.write_text(text)
print("patched", path)