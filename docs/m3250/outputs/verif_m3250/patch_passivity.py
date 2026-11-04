"""Add the passivity/conditioning claims of table 11 as executable registry rows."""

from __future__ import annotations

import ast
import pathlib

TARGET = pathlib.Path(__file__).resolve().parents[2] / "src" / "tensorbem" / "benchmarks" / "declared_numbers.py"
text = TARGET.read_text(encoding="utf-8")

if "b33_is_negative" in text:
    raise SystemExit("already patched")

OLD_RUNNER = """            values[f"{branch}.b33_dimless_abs"] = abs(
                float(branch_values["b33_dimless"])
            )
        return values"""

NEW_RUNNER = """            b33 = float(branch_values["b33_dimless"])
            values[f"{branch}.b33_dimless_abs"] = abs(b33)
            values[f"{branch}.b33_is_negative"] = 1.0 if b33 < 0.0 else 0.0
            values[f"{branch}.b33_is_positive"] = 1.0 if b33 > 0.0 else 0.0
        values["state.condition_number_ratio"] = (
            float(row["chief_regularized"]["condition_number"])
            / float(row["unregularized"]["condition_number"])
        )
        return values"""

OLD_END = """                kind=kind,
                tolerance=_FS3D_COARSE_TOLERANCE,
            )
        )
    return tuple(rows)"""

NEW_END = """                kind=kind,
                tolerance=_FS3D_COARSE_TOLERANCE,
            )
        )

    # The message of table 11 is qualitative: the unregularised surface-piercing
    # BIE loses passivity at the waterline resonance and five interior CHIEF
    # points restore it. Pin the qualitative statements, not just the columns.
    qualitative = (
        (
            "chief3d_unregularized_b33_is_negative",
            "the unregularised BIE returns an unphysical negative heave damping",
            "unregularized.b33_is_negative",
            1.0,
            "value",
            1.0e-12,
        ),
        (
            "chief3d_regularized_b33_is_positive",
            "interior CHIEF restores strictly positive heave damping",
            "chief_regularized.b33_is_positive",
            1.0,
            "value",
            1.0e-12,
        ),
        (
            "chief3d_condition_number_ratio",
            "CHIEF must cut the condition number by more than 20x",
            "state.condition_number_ratio",
            5.0e-2,
            "upper",
            _FS3D_TOLERANCE,
        ),
    )
    for key, description, quantity, declared, kind, tolerance in qualitative:
        rows.append(
            DeclaredNumber(
                key=key,
                description=description,
                source=(
                    "docs/marine_bem_benchmarks.md table 11 "
                    "(lid-free CHIEF at the waterline resonance)"
                ),
                group="hemisphere3d_chief",
                quantity=quantity,
                declared=declared,
                parameters=dict(chief_parameters),
                tier="standard",
                kind=kind,
                tolerance=tolerance,
            )
        )
    return tuple(rows)"""

for old, new in ((OLD_RUNNER, NEW_RUNNER), (OLD_END, NEW_END)):
    assert text.count(old) == 1, (text.count(old), old[:60])
    text = text.replace(old, new)

ast.parse(text)
TARGET.write_text(text, encoding="utf-8")
print("patched passivity rows")