"""Register the 3-D free-surface tables in the declared-number registry.

Extends ``src/tensorbem/benchmarks/declared_numbers.py`` with the three
benchmark families quoted in docs/marine_bem_benchmarks.md §4.10–§4.11
(Hulme 1982 floating hemisphere, CHIEF regularisation, Wang/Linton 1991
submerged sphere).  All entries live in the ``standard`` tier: the tables
quote only the refinement studies, which cost ~15 s in total.

Run once from the repository root:
    python outputs/verif_m3631/patch_registry.py
"""
import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
p = REPO_ROOT / "src" / "tensorbem" / "benchmarks" / "declared_numbers.py"
text = p.read_text()

# --------------------------------------------------------------------------
# 1. imports
# --------------------------------------------------------------------------
old_imports = """from tensorbem.benchmarks import (
    cylinder_added_mass as cylinder_benchmark,
    ellipsoid_added_mass as ellipsoid_benchmark,
    interior_harmonics as interior_harmonics_benchmark,
    ursell_cylinder as ursell_benchmark,
)"""
new_imports = """from tensorbem.benchmarks import (
    cylinder_added_mass as cylinder_benchmark,
    ellipsoid_added_mass as ellipsoid_benchmark,
    floating_hemisphere as floating_hemisphere_benchmark,
    interior_harmonics as interior_harmonics_benchmark,
    submerged_sphere_wave as submerged_sphere_benchmark,
    ursell_cylinder as ursell_benchmark,
)"""
assert text.count(old_imports) == 1
text = text.replace(old_imports, new_imports)

# --------------------------------------------------------------------------
# 2. group docstring of the dataclass
# --------------------------------------------------------------------------
old_group_doc = '''    group: str
    """Which benchmark to run: ``cylinder``, ``ellipsoid``, ``ursell`` or
    ``ursell_reference``."""'''
new_group_doc = '''    group: str
    """Which benchmark to run: ``cylinder``, ``ellipsoid``, ``ursell``,
    ``ursell_reference``, ``interior_harmonics``, ``hulme_hemisphere``,
    ``hulme_chief`` or ``submerged_sphere``."""'''
assert text.count(old_group_doc) == 1
text = text.replace(old_group_doc, new_group_doc)

# --------------------------------------------------------------------------
# 3. provenance note in the module docstring
# --------------------------------------------------------------------------
old_prov = "``dev_report_tensorbem_declared_numbers.md`` rather than silently overwritten.\n\"\"\""
new_prov = """``dev_report_tensorbem_declared_numbers.md`` rather than silently overwritten.

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
\"\"\""""
assert text.count(old_prov) == 1
text = text.replace(old_prov, new_prov)

# --------------------------------------------------------------------------
# 4. registry tuple: append the new family
# --------------------------------------------------------------------------
old_tail = """        declared=1.824e-4,
        tier="standard",
    ),
)"""
new_tail = """        declared=1.824e-4,
        tier="standard",
    ),
    *_free_surface_3d_entries(),
)"""
assert text.count(old_tail) == 1
text = text.replace(old_tail, new_tail)

# --------------------------------------------------------------------------
# 5. runner dispatch
# --------------------------------------------------------------------------
old_dispatch = """    if group == "ursell_reference":
        omega = math.sqrt(_URSELL_KA * _URSELL_GRAVITY / _URSELL_RADIUS)"""
new_dispatch = """    if group == "hulme_hemisphere":
        study = floating_hemisphere_benchmark.run_refinement_study(
            subdivisions=tuple(int(s) for s in parameters["subdivisions"]),
            ka=float(parameters.get("ka", 1.0)),
        )
        return _flatten_levels(study["levels"])
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
        omega = math.sqrt(_URSELL_KA * _URSELL_GRAVITY / _URSELL_RADIUS)"""
assert text.count(old_dispatch) == 1
text = text.replace(old_dispatch, new_dispatch)

# --------------------------------------------------------------------------
# 6. the new section (data + builder + flattener), inserted before _run_group
# --------------------------------------------------------------------------
section = '''
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
            add(f"hulme_order_{name}_{panels}",
                f"observed order of the {name.upper()} error at {panels} panels",
                f"§4.10 table 10 row N={panels}: observed order {name.upper()}",
                "hulme_hemisphere", f"level{sub}_observed_order_{name}",
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
            _SPHERE_TABLE12_PARAMS)
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


'''
anchor = (
    "#: The registry. ``source`` cites the current wording; every value below is the\n"
    "#: number a reader of the docs will find there after the 2026-11-03 audit.\n"
    "DECLARED_NUMBERS: tuple[DeclaredNumber, ...] = ("
)
assert text.count(anchor) == 1
text = text.replace(anchor, section + anchor)

p.write_text(text)
ast.parse(text)
print("registry patched OK, AST valid")
