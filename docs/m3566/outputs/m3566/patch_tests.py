"""Patch tests/test_declared_numbers.py with the rigid-body registry guards."""

from pathlib import Path

TARGET = Path(__file__).resolve().parents[2] / "tests/test_declared_numbers.py"
text = TARGET.read_text(encoding="utf-8")

# 1. the registry must cover the rigid-body family too
OLD = '''    assert {
        "cylinder",
        "ellipsoid",
        "interior_harmonics",
        "ursell",
        "ursell_reference",
    } <= groups'''
NEW = '''    assert {
        "cylinder",
        "ellipsoid",
        "interior_harmonics",
        "rigid_body",
        "ursell",
        "ursell_reference",
    } <= groups'''
assert text.count(OLD) == 1
text = text.replace(OLD, NEW)

# 2. two new guards, inserted after the standard-tier Ursell test
ANCHOR = '''@HEAVY
def test_documented_1280_panel_harmonic_rows_are_reproduced():'''
NEW_TESTS = '''def test_standard_tier_reproduces_the_documented_rigid_body_rows():
    """Tables 5 and 6 of section 4.7 quote 44 numbers that nothing recomputed.

    The rigid-body Neumann ladder was documentation-only until this guard
    existed: the tensors, the conditioning of the second-kind operator and the
    observed convergence orders were transcribed by hand from a run whose
    setting was not executable. Every quoted row is now registered and
    re-derived by the released code.
    """
    report = check(tiers=("standard",))
    rows = [row for row in report["checks"] if row["group"] == "rigid_body"]
    assert len(rows) >= 40
    assert report["summary"]["verdict"] == "PASS", [
        (row["key"], row["declared"], row["measured"])
        for row in rows
        if row["verdict"] != "PASS"
    ]
    keys = {row["key"] for row in rows}
    assert {
        "rigid_body_2d_roll_xy_128",
        "rigid_body_3d_max_translation_320",
        "rigid_body_3d_yaw_z_order_320",
    } <= keys


def test_rigid_body_claims_hold_where_the_documents_make_them():
    """The prose of section 4.7 makes three claims beyond the table cells.

    * the second-kind operator stays well conditioned under refinement
      (``O(1)``, not the ``O(h^-1)`` growth of the first-kind operator);
    * the measured orders are second order, not merely "decreasing";
    * the assembled tensor is symmetric to round-off, which is why the
      tables quote a bound rather than the noisy digits themselves.
    """
    measurements = check(
        tiers=("standard",),
        entries=tuple(
            entry for entry in DECLARED_NUMBERS if entry.group == "rigid_body"
        ),
    )
    values = {row["key"]: row["measured"] for row in measurements["checks"]}

    for panels in (16, 32, 64, 128):
        assert values[f"rigid_body_2d_conditioning_{panels}"] < 3.1
        assert values[f"rigid_body_2d_symmetry_{panels}"] < 1.0e-11
    for panels in (20, 80, 320):
        assert values[f"rigid_body_3d_conditioning_{panels}"] < 2.5

    orders = [
        values[key]
        for key in values
        if key.endswith("_order_32")
        or key.endswith("_order_64")
        or key.endswith("_order_128")
        or key.endswith("_order_80")
        or key.endswith("_order_320")
    ]
    assert len(orders) >= 9
    assert min(orders) > 1.85 and max(orders) < 2.15, orders


@HEAVY
def test_documented_1280_panel_harmonic_rows_are_reproduced():'''
assert text.count(ANCHOR) == 1
text = text.replace(ANCHOR, NEW_TESTS)

TARGET.write_text(text, encoding="utf-8")
print("patched", TARGET)