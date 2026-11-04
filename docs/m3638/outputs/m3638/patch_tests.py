"""Patch: give the CHIEF rows the standard tier, and extend the registry tests."""

from pathlib import Path

# ---------------------------------------------------------------- tier fix
REG = Path("src/tensorbem/benchmarks/declared_numbers.py")
text = REG.read_text()
blocks = text.split("    DeclaredNumber(")
fixed = 0
for i, block in enumerate(blocks):
    if 'group="hulme_chief"' in block and 'tier="standard"' not in block:
        blocks[i] = block.replace(
            "        parameters=", '        tier="standard",\n        parameters=', 1
        )
        fixed += 1
REG.write_text("    DeclaredNumber(".join(blocks))
print("chief rows retiered:", fixed)

# ---------------------------------------------------------------- tests
TESTS = Path("tests/test_declared_numbers.py")
src = TESTS.read_text()

COVER_OLD = """    assert {
        "cylinder",
        "ellipsoid",
        "interior_harmonics",
        "ursell",
        "ursell_reference",
    } <= groups"""
COVER_NEW = """    assert {
        "cylinder",
        "ellipsoid",
        "hulme",
        "hulme_chief",
        "interior_harmonics",
        "submerged_sphere",
        "ursell",
        "ursell_reference",
    } <= groups"""

STD_OLD = '''def test_standard_tier_reproduces_the_documented_ursell_numbers():
    """The 128-panel Ursell row is the one a verification audit disputed."""
    report = check(tiers=("standard",))
    keys = {row["key"] for row in report["checks"]}
    assert {
        "ursell_added_mass_128",
        "ursell_damping_128",
        "ursell_near_far_128",
    } <= keys
    assert report["summary"]["verdict"] == "PASS", [
        (row["key"], row["declared"], row["measured"])
        for row in report["checks"]
        if row["verdict"] != "PASS"
    ]'''

STD_NEW = '''def test_standard_tier_reproduces_the_documented_ursell_numbers():
    """The 128-panel Ursell row is the one a verification audit disputed."""
    report = check(tiers=("standard",))
    keys = {row["key"] for row in report["checks"]}
    assert {
        "ursell_added_mass_128",
        "ursell_damping_128",
        "ursell_near_far_128",
    } <= keys
    assert report["summary"]["verdict"] == "PASS", [
        (row["key"], row["declared"], row["measured"])
        for row in report["checks"]
        if row["verdict"] != "PASS"
    ]


def test_standard_tier_guards_the_three_dimensional_free_surface_tables():
    """Tables 10-12 of docs/marine_bem_benchmarks.md, the 3-D Kelvin layer.

    Those tables were documentation-only until now: the 3-D free-surface rows
    of the registry stopped at the unbounded-flow ellipsoid, so a change in the
    Kelvin/Havelock kernel or in the CHIEF augmentation could move the quoted
    numbers without any test noticing.  The three groups are asserted here as a
    block because they share one cost class (~20-45 s per group).
    """
    report = check(tiers=("standard",))
    rows = {row["key"]: row for row in report["checks"]}
    expected = {
        # Hulme (1982) floating hemisphere, table 10
        "hulme_condition_number_256",
        "hulme_heave_added_mass_256",
        "hulme_surge_damping_256",
        "hulme_excitation_z_256",
        "hulme_order_heave_added_mass_256",
        "hulme_far_field_damping_256",
        "hulme_haskind_surge_256",
        "hulme_haskind_heave_256",
        # 3-D CHIEF interior null-field augmentation, table 11
        "hulme_chief_condition_unregularized",
        "hulme_chief_condition_augmented",
        "hulme_chief_heave_damping_unregularized",
        "hulme_chief_heave_damping_augmented",
        # Wang / Linton submerged sphere, table 12
        "linton_condition_number_320",
        "linton_heave_added_mass_320",
        "linton_excitation_z_320",
        "linton_order_heave_added_mass_320",
    }
    assert expected <= set(rows)
    assert report["summary"]["verdict"] == "PASS", [
        (row["key"], row["declared"], row["measured"])
        for row in report["checks"]
        if row["verdict"] != "PASS"
    ]


def test_chief_regularisation_restores_passivity():
    """Table 11 is a claim about a *sign*, not only about magnitudes.

    The unregularised exterior BIE at the first interior waterline eigenvalue
    returns a negative heave damping (non-physical energy production); the
    interior null-field points must flip it positive.  A magnitude-only pin
    would survive a sign error in the augmentation, so the sign is asserted.
    """
    report = check(tiers=("standard",))
    rows = {row["key"]: row for row in report["checks"]}
    assert rows["hulme_chief_heave_damping_unregularized"]["measured"] < 0.0
    assert rows["hulme_chief_heave_damping_augmented"]["measured"] > 0.0
    exact = 0.06312197075596611
    assert abs(rows["hulme_chief_heave_damping_augmented"]["measured"] - exact) < 1.0e-2
    # the conditioning claim: augmentation must reduce the condition number
    assert (
        rows["hulme_chief_condition_augmented"]["measured"]
        < 0.1 * rows["hulme_chief_condition_unregularized"]["measured"]
    )'''

for old, new, label in (
    (COVER_OLD, COVER_NEW, "coverage"),
    (STD_OLD, STD_NEW, "standard tier"),
):
    assert src.count(old) == 1, f"anchor {label} occurs {src.count(old)} times"
    src = src.replace(old, new)

TESTS.write_text(src)
print("tests patched")