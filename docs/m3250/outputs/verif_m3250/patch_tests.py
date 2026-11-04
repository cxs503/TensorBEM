"""Add the 3-D free-surface regression test to tests/test_declared_numbers.py."""

from __future__ import annotations

import ast
import pathlib

TARGET = pathlib.Path(__file__).resolve().parents[2] / "tests" / "test_declared_numbers.py"
text = TARGET.read_text(encoding="utf-8")

if "test_free_surface_3d_tables" in text:
    raise SystemExit("already patched")

OLD_GROUPS = """    assert {
        "cylinder",
        "ellipsoid",
        "interior_harmonics",
        "ursell",
        "ursell_reference",
    } <= groups"""

NEW_GROUPS = """    assert {
        "cylinder",
        "ellipsoid",
        "hemisphere3d",
        "hemisphere3d_chief",
        "interior_harmonics",
        "submerged_sphere3d",
        "ursell",
        "ursell_reference",
    } <= groups"""

OLD_STANDARD = """    report = check(tiers=("standard",))
    keys = {row["key"] for row in report["checks"]}"""

NEW_STANDARD = """    report = check(
        tiers=("standard",),
        entries=tuple(
            entry for entry in DECLARED_NUMBERS if entry.group.startswith("ursell")
        ),
    )
    keys = {row["key"] for row in report["checks"]}"""

NEW_TEST = '''def test_free_surface_3d_tables_are_reproduced():
    """Tables 10-12 of docs/marine_bem_benchmarks.md were documentation-only.

    The 3-D free-surface studies print those tables, but nothing read them back:
    the long-standing tests guard the physics at a 2.5-5 % threshold, so a whole
    convergence column could move by an order of magnitude and stay green. Every
    cell (errors, observed orders, condition numbers, and the near/far-field and
    Haskind closures quoted in the surrounding prose) is now a registered number.
    """
    groups = ("hemisphere3d", "hemisphere3d_chief", "submerged_sphere3d")
    report = check(
        tiers=("standard",),
        entries=tuple(
            entry for entry in DECLARED_NUMBERS if entry.group in groups
        ),
    )
    assert report["summary"]["checked"] == 63
    assert report["summary"]["verdict"] == "PASS", [
        (row["key"], row["declared"], row["measured"])
        for row in report["checks"]
        if row["verdict"] != "PASS"
    ]
    measured = {row["key"]: row["measured"] for row in report["checks"]}

    # Refinement must stay refinement: the wave-exciting columns shrink with the
    # mesh in both families and land inside the 3 % acceptance band.
    for prefix, levels in (("hemisphere3d", (16, 64, 256)), ("sphere3d", (20, 80, 320))):
        a33 = [measured[f"{prefix}_heave_added_mass_rel_error_{n}"] for n in levels]
        assert a33[0] > a33[1] > a33[2]
        assert a33[-1] < 3.0e-2
        fz = [measured[f"{prefix}_excitation_force_z_rel_error_{n}"] for n in levels]
        assert fz[-1] < 3.0e-2
        kappa = [measured[f"{prefix}_condition_number_{n}"] for n in levels]
        assert kappa[-1] < 3.0  # well conditioned throughout

    # The surge wave damping is the one column the documentation itself flags as
    # discretisation-limited (3.53 % at 256 panels); it must not creep past 5 %.
    assert measured["hemisphere3d_surge_damping_rel_error_256"] < 5.0e-2
    assert measured["hemisphere3d_surge_added_mass_rel_error_256"] < 3.0e-2

    # Table 11's message: interior CHIEF cuts the condition number by >20x at the
    # waterline resonance and brings the heave damping back inside a factor of
    # three of the exact value (unregularised: |B33'| off by 265 %).
    assert (
        measured["chief3d_chief_regularized_condition_number"]
        / measured["chief3d_unregularized_condition_number"]
        < 5.0e-2
    )
    assert measured["chief3d_unregularized_b33_dimless_abs"] > 1.0e-1
    assert measured["chief3d_chief_regularized_b33_dimless_abs"] < 1.0e-1

'''

ANCHOR = "@HEAVY\ndef test_documented_1280_panel_harmonic_rows_are_reproduced():"

assert text.count(OLD_GROUPS) == 1
assert text.count(OLD_STANDARD) == 1
assert text.count(ANCHOR) == 1

text = text.replace(OLD_GROUPS, NEW_GROUPS)
text = text.replace(OLD_STANDARD, NEW_STANDARD)
text = text.replace(ANCHOR, NEW_TEST + ANCHOR)
ast.parse(text)
TARGET.write_text(text, encoding="utf-8")
print("tests patched")