"""Patch declared_numbers.py: register docs tables 5 & 6 (rigid-body Neumann)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "src/tensorbem/benchmarks/declared_numbers.py"
text = TARGET.read_text(encoding="utf-8")

# --------------------------------------------------------------------------
# 1. import the benchmark module used by the new runner
# --------------------------------------------------------------------------
OLD_IMPORT = """    interior_harmonics as interior_harmonics_benchmark,
    ursell_cylinder as ursell_benchmark,
)"""
NEW_IMPORT = """    interior_harmonics as interior_harmonics_benchmark,
    rigid_body_added_mass as rigid_body_benchmark,
    ursell_cylinder as ursell_benchmark,
)"""
assert text.count(OLD_IMPORT) == 1
text = text.replace(OLD_IMPORT, NEW_IMPORT)

# --------------------------------------------------------------------------
# 2. the tables themselves + entry factory, inserted before the registry
# --------------------------------------------------------------------------
ANCHOR = "\n#: The registry. ``source`` cites the current wording"
BLOCK = '''
#: Table 5 of ``docs/marine_bem_benchmarks.md`` section 4.7: the elliptic
#: cylinder (``a=1.5``, ``b=0.75``, ``rho=1025``) rigid-body Neumann tensor
#: ladder. Columns are ``(panels, conditioning, surge_x, heave_y, roll_xy,
#: observed order of roll_xy)``; ``None`` marks the coarsest level, whose order
#: is undefined because it has no predecessor.
_RIGID_BODY_2D_TABLE: tuple[tuple, ...] = (
    (16, 3.078, 2.64e-3, 5.15e-3, 3.23e-2, None),
    (32, 3.019, 7.42e-4, 1.41e-3, 8.71e-3, 1.936),
    (64, 3.005, 1.94e-4, 3.75e-4, 2.25e-3, 1.966),
    (128, 3.001, 4.94e-5, 9.70e-5, 5.70e-4, 1.982),
)

#: Table 6 of the same section: the triaxial ellipsoid (``1.5 / 1.0 / 0.75``,
#: ``rho=1025``) ``6x6`` tensor ladder. Columns are ``(subdivisions, panels,
#: conditioning, max translation error, roll_x, pitch_y, yaw_y, (orders of
#: roll_x, pitch_y, yaw_z))``.
_RIGID_BODY_3D_TABLE: tuple[tuple, ...] = (
    (0, 20, 2.447, 2.63e-1, 4.90e-1, 5.19e-1, 5.11e-1, (None, None, None)),
    (1, 80, 2.246, 8.11e-2, 1.63e-1, 1.78e-1, 1.74e-1, (2.068, 2.012, 2.022)),
    (2, 320, 2.130, 2.14e-2, 4.60e-2, 4.84e-2, 4.85e-2, (1.972, 2.028, 1.992)),
)

#: The tables quote the antisymmetry residual of the assembled tensor with two
#: significant digits (``6.4e-13``, ``9.3e-13``, ...). Those digits are round-off
#: noise of the order of the machine epsilon times the matrix norm: the same
#: release returns ``5.98e-13`` .. ``1.14e-12`` on a loongarch64 host instead of
#: the ``6.4e-13`` .. ``8.0e-13`` recorded on x86-64, i.e. the quoted pair is not
#: reproducible and must not be pinned as an equality. What *is* meaningful --
#: and what a symmetry regression would break by orders of magnitude -- is the
#: statement that the residual stays below ``1e-11``.
_RIGID_BODY_SYMMETRY_BOUND = 1.0e-11


def _rigid_body_registry() -> tuple[DeclaredNumber, ...]:
    """Expand tables 5 and 6 of section 4.7 into registry rows."""
    source_2d = "docs/marine_bem_benchmarks.md section 4.7 tables 5 (2-D ellipse)"
    source_3d = "docs/marine_bem_benchmarks.md section 4.7 tables 6 (3-D ellipsoid)"
    rows: list[DeclaredNumber] = []
    for panels, kappa, surge, heave, roll, order in _RIGID_BODY_2D_TABLE:
        params = {"case": "ellipse"}
        rows.append(
            DeclaredNumber(
                key=f"rigid_body_2d_conditioning_{panels}",
                description=f"conditioning of (1/2 I - H), 2-D ellipse, {panels} panels",
                source=source_2d,
                group="rigid_body",
                quantity=f"condition_number_{panels}",
                parameters=params,
                declared=kappa,
                tier="standard",
            )
        )
        for mode, declared in (
            ("surge_x", surge),
            ("heave_y", heave),
            ("roll_xy", roll),
        ):
            rows.append(
                DeclaredNumber(
                    key=f"rigid_body_2d_{mode}_{panels}",
                    description=f"relative error of A[{mode}], 2-D ellipse, {panels} panels",
                    source=source_2d,
                    group="rigid_body",
                    quantity=f"{mode}_relative_error_{panels}",
                    parameters=params,
                    declared=declared,
                    tier="standard",
                )
            )
        rows.append(
            DeclaredNumber(
                key=f"rigid_body_2d_symmetry_{panels}",
                description=f"antisymmetry residual bound of A, 2-D ellipse, {panels} panels",
                source=source_2d,
                group="rigid_body",
                quantity=f"symmetry_error_{panels}",
                parameters=params,
                declared=_RIGID_BODY_SYMMETRY_BOUND,
                kind="upper",
                tier="standard",
            )
        )
        if order is not None:
            rows.append(
                DeclaredNumber(
                    key=f"rigid_body_2d_roll_order_{panels}",
                    description=f"observed order of roll_xy, 2-D ellipse, {panels} panels",
                    source=source_2d,
                    group="rigid_body",
                    quantity=f"roll_xy_observed_order_{panels}",
                    parameters=params,
                    declared=order,
                    tier="standard",
                )
            )
    for level, panels, kappa, trans, roll, pitch, yaw, orders in _RIGID_BODY_3D_TABLE:
        params = {"case": "ellipsoid"}
        rows.append(
            DeclaredNumber(
                key=f"rigid_body_3d_conditioning_{panels}",
                description=f"conditioning of (1/2 I - H), 3-D ellipsoid, {panels} panels",
                source=source_3d,
                group="rigid_body",
                quantity=f"condition_number_{level}",
                parameters=params,
                declared=kappa,
                tier="standard",
            )
        )
        rows.append(
            DeclaredNumber(
                key=f"rigid_body_3d_max_translation_{panels}",
                description=f"largest translational relative error, {panels} panels",
                source=source_3d,
                group="rigid_body",
                quantity=f"max_translation_relative_error_{level}",
                parameters=params,
                declared=trans,
                tier="standard",
            )
        )
        for mode, declared, order in (
            ("roll_x", roll, orders[0]),
            ("pitch_y", pitch, orders[1]),
            ("yaw_z", yaw, orders[2]),
        ):
            rows.append(
                DeclaredNumber(
                    key=f"rigid_body_3d_{mode}_{panels}",
                    description=f"relative error of A[{mode}], 3-D ellipsoid, {panels} panels",
                    source=source_3d,
                    group="rigid_body",
                    quantity=f"{mode}_relative_error_{level}",
                    parameters=params,
                    declared=declared,
                    tier="standard",
                )
            )
            if order is not None:
                rows.append(
                    DeclaredNumber(
                        key=f"rigid_body_3d_{mode}_order_{panels}",
                        description=f"observed order of A[{mode}], 3-D ellipsoid, {panels} panels",
                        source=source_3d,
                        group="rigid_body",
                        quantity=f"{mode}_observed_order_{level}",
                        parameters=params,
                        declared=order,
                        tier="standard",
                    )
                )
    return tuple(rows)


'''
assert text.count(ANCHOR) == 1
text = text.replace(ANCHOR, BLOCK + ANCHOR.lstrip("\n"), 1)

# --------------------------------------------------------------------------
# 3. splice the generated rows into the registry literal
# --------------------------------------------------------------------------
OLD_TAIL = """        declared=1.824e-4,
        tier="standard",
    ),
)
"""
NEW_TAIL = """        declared=1.824e-4,
        tier="standard",
    ),
    # --- Rigid-body Neumann tensors (docs table 5 and table 6) -------------
    *_rigid_body_registry(),
)
"""
assert text.count(OLD_TAIL) == 1
text = text.replace(OLD_TAIL, NEW_TAIL)

# --------------------------------------------------------------------------
# 4. the runner
# --------------------------------------------------------------------------
OLD_RUNNER = """    if group == "ursell_reference":"""
NEW_RUNNER = '''    if group == "rigid_body":
        case = str(parameters["case"])
        if case == "ellipse":
            report = rigid_body_benchmark.run_ellipse_neumann_study()
            levels = report["levels"]
            key_for = "panels"
        elif case == "ellipsoid":
            report = rigid_body_benchmark.run_ellipsoid_neumann_study()
            levels = report["levels"]
            key_for = "subdivisions"
        else:
            raise ValueError(f"unknown rigid-body case: {case!r}")
        values: dict[str, float] = {}
        for level in levels:
            index = int(level[key_for])
            errors = level["diagonal_relative_errors"]
            values[f"condition_number_{index}"] = float(level["condition_number"])
            values[f"symmetry_error_{index}"] = float(level["symmetry_error"])
            if case == "ellipsoid":
                values[f"max_translation_relative_error_{index}"] = max(
                    float(errors[mode]) for mode in ("surge_x", "sway_y", "heave_z")
                )
                modes = ("roll_x", "pitch_y", "yaw_z")
            else:
                modes = ("surge_x", "heave_y", "roll_xy")
            for mode in modes:
                values[f"{mode}_relative_error_{index}"] = float(errors[mode])
        for mode, orders in report["observed_orders"].items():
            for level, order in zip(levels, orders):
                if order is None:
                    continue
                index = int(level[key_for])
                values[f"{mode}_observed_order_{index}"] = float(order)
        return values
    if group == "ursell_reference":'''
assert text.count(OLD_RUNNER) == 1
text = text.replace(OLD_RUNNER, NEW_RUNNER)

# --------------------------------------------------------------------------
# 5. document the group in the dataclass docstring
# --------------------------------------------------------------------------
OLD_DOC = '''    """Which benchmark to run: ``cylinder``, ``ellipsoid``, ``ursell`` or
    ``ursell_reference``."""'''
NEW_DOC = '''    """Which benchmark to run: ``cylinder``, ``ellipsoid``, ``ursell``,
    ``ursell_reference``, ``interior_harmonics`` or ``rigid_body``."""'''
assert text.count(OLD_DOC) == 1
text = text.replace(OLD_DOC, NEW_DOC)

TARGET.write_text(text, encoding="utf-8")
print("patched", TARGET)