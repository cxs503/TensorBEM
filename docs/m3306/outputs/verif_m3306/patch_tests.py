"""Third patch: tests for the 3-D free-surface registry rows."""
from pathlib import Path

path = Path("tests/test_declared_numbers.py")
text = path.read_text()
assert "standard_report" not in text, "already patched"

old = """    assert {
        "cylinder",
        "ellipsoid",
        "interior_harmonics",
        "ursell",
        "ursell_reference",
    } <= groups"""
new = """    assert {
        "cylinder",
        "ellipsoid",
        "hulme_hemisphere",
        "hulme_irregular",
        "interior_harmonics",
        "submerged_sphere",
        "ursell",
        "ursell_reference",
    } <= groups"""
assert old in text
text = text.replace(old, new, 1)

old = '''def test_standard_tier_reproduces_the_documented_ursell_numbers():
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
'''
new = '''@pytest.fixture(scope="module")
def standard_report():
    """One standard-tier sweep, shared by the tests below.

    It is a couple of minutes of three-dimensional free-surface assembling and
    solving (256-panel Hulme hemisphere, 320-panel submerged sphere, two
    256-panel solves at the irregular frequency), so it is computed once per
    module rather than once per test.
    """
    return check(tiers=("standard",))


def test_standard_tier_reproduces_the_documented_ursell_numbers(standard_report):
    """The 128-panel Ursell row is the one a verification audit disputed."""
    keys = {row["key"] for row in standard_report["checks"]}
    assert {
        "ursell_added_mass_128",
        "ursell_damping_128",
        "ursell_near_far_128",
    } <= keys
    assert standard_report["summary"]["verdict"] == "PASS", [
        (row["key"], row["declared"], row["measured"])
        for row in standard_report["checks"]
        if row["verdict"] != "PASS"
    ]


def test_three_dimensional_free_surface_rows_are_pinned(standard_report):
    """Tables 10-12 of docs/marine_bem_benchmarks.md are documentation only.

    The three-dimensional free-surface families (Hulme 1982 hemisphere, the
    interior-CHIEF suppression of the Ka=2.58 irregular frequency, and the
    Wang/Linton submerged sphere) were verified by standalone test modules, but
    none of their quoted numbers was recomputed by the registry: a kernel or mesh
    change could move a documented table cell without any gate noticing.
    """
    rows = {row["key"]: row for row in standard_report["checks"]}
    assert {
        "hulme_heave_added_mass_error_256",
        "hulme_heave_damping_error_256",
        "hulme_surge_added_mass_error_256",
        "hulme_surge_damping_error_256",
        "hulme_vertical_force_error_256",
        "hulme_condition_number_256",
        "hulme_near_far_damping_256",
        "hulme_haskind_surge_256",
        "hulme_haskind_heave_256",
        "hulme_irregular_passivity_violated",
        "hulme_irregular_chief_condition",
        "hulme_irregular_chief_damping",
        "submerged_sphere_heave_added_mass_error_320",
        "submerged_sphere_surge_added_mass_error_320",
        "submerged_sphere_horizontal_force_error_320",
        "submerged_sphere_vertical_force_error_320",
        "submerged_sphere_condition_number_320",
        "submerged_sphere_rotational_coupling_320",
    } <= set(rows)
    drifted = {
        key: (row["declared"], row["measured"])
        for key, row in rows.items()
        if row["verdict"] != "PASS"
        and row["group"] in {"hulme_hemisphere", "hulme_irregular", "submerged_sphere"}
    }
    assert not drifted


def test_hulme_orders_are_second_order_in_the_asymptotic_range(standard_report):
    """The convergence story is the claim, not just the finest-mesh errors.

    Table 10's order column was derived with two different mesh-size
    normalisations over the years; the registry now pins the column that comes
    from the mesh's own ``h_mean`` ratio, so a future re-quote that mixes the
    conventions again fails here.
    """
    rows = {row["key"]: row for row in standard_report["checks"]}
    orders = {
        key: rows[f"hulme_order_{key}_256"]["measured"]
        for key in ("a33", "b33", "a11", "b11", "fz")
    }
    assert all(1.8 <= value <= 2.2 for value in orders.values()), orders
    # B11' is the slowest quantity: its error at 256 panels is the largest cell
    # of table 10, and its order is the smallest of the five.
    errors = {
        key: rows[f"hulme_" + name + "_256"]["measured"]
        for key, name in (
            ("a33", "heave_added_mass_error"),
            ("b33", "heave_damping_error"),
            ("a11", "surge_added_mass_error"),
            ("b11", "surge_damping_error"),
            ("fz", "vertical_force_error"),
        )
    }
    assert max(errors, key=errors.get) == "b11"
    assert min(orders, key=orders.get) == "b11"


def test_chief_regularization_restores_passivity_at_the_irregular_frequency(standard_report):
    """Ka=2.58 is the first axisymmetric interior waterline eigenvalue.

    The claim in table 11 is physical, not numerical: the un-augmented BIE turns
    damped motion into a negative damping coefficient (energy creation), and
    five interior CHIEF points remove it while keeping A33' near the exact
    value. Both halves are asserted here.
    """
    rows = {row["key"]: row for row in standard_report["checks"]}
    assert rows["hulme_irregular_passivity_violated"]["measured"] == 1.0
    chief_damping = rows["hulme_irregular_chief_damping"]["measured"]
    exact_damping = rows["hulme_irregular_exact_damping"]["measured"]
    unregularized_damping = rows["hulme_irregular_unregularized_damping_magnitude"]["measured"]
    chief_added_mass = rows["hulme_irregular_chief_added_mass"]["measured"]
    exact_added_mass = rows["hulme_irregular_exact_added_mass"]["measured"]
    unregularized_added_mass = rows["hulme_irregular_unregularized_added_mass"]["measured"]

    assert chief_damping > 0.0
    assert abs(chief_damping - exact_damping) < 0.25 * abs(
        unregularized_damping - exact_damping
    )
    assert abs(chief_added_mass - exact_added_mass) < 0.5 * abs(
        unregularized_added_mass - exact_added_mass
    )
    # The conditioning collapses, not merely improves.
    assert rows["hulme_irregular_chief_condition"]["measured"] < 0.05 * rows[
        "hulme_irregular_unregularized_condition"
    ]["measured"]
'''
assert old in text
text = text.replace(old, new, 1)
path.write_text(text)
print("patched", path)