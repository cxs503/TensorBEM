"""Idempotent patch: pin the heavy ellipsoid row and the 1280-panel harmonic rows."""
from pathlib import Path

path = Path("tests/test_declared_numbers.py")
text = path.read_text()

old_cov = '''def test_registry_covers_every_documented_benchmark_family():
    groups = {entry.group for entry in DECLARED_NUMBERS}
    assert {"cylinder", "ellipsoid", "ursell", "ursell_reference"} <= groups'''
new_cov = '''def test_registry_covers_every_documented_benchmark_family():
    groups = {entry.group for entry in DECLARED_NUMBERS}
    assert {
        "cylinder",
        "ellipsoid",
        "interior_harmonics",
        "ursell",
        "ursell_reference",
    } <= groups'''
if "interior_harmonics" not in text:
    assert old_cov in text
    text = text.replace(old_cov, new_cov)

anchor = '''def test_report_is_strict_json():'''
added = '''def test_documented_1280_panel_harmonic_rows_are_reproduced():
    """Table 3 of docs/classical_benchmarks.md quotes 1280-panel numbers.

    These rows were documentation-only until now: nothing recomputed them, so
    the degree-2 and degree-3 quotes could rot silently. The truth is the exact
    solid-harmonic polynomial, so the quoted quantity is a pure discretisation
    error and can be pinned tightly.
    """
    report = check(
        tiers=("standard",),
        entries=tuple(
            entry for entry in DECLARED_NUMBERS if entry.group == "interior_harmonics"
        ),
    )
    assert report["summary"]["checked"] == 6
    assert report["summary"]["verdict"] == "PASS", [
        (row["key"], row["declared"], row["measured"])
        for row in report["checks"]
        if row["verdict"] != "PASS"
    ]
    # The documented degree trending (higher degree -> larger flux error at the
    # same mesh) must survive any future re-quote.
    flux = {
        row["key"]: row["measured"]
        for row in report["checks"]
        if row["key"].endswith("_flux_1280")
    }
    assert (
        flux["interior_harmonic_deg1_flux_1280"]
        < flux["interior_harmonic_deg2_flux_1280"]
        < flux["interior_harmonic_deg3_flux_1280"]
    )


''' + anchor
if "test_documented_1280_panel_harmonic_rows_are_reproduced" not in text:
    assert text.count(anchor) == 1
    text = text.replace(anchor, added)

# Heavy-tier pin for the row that the 2026-10-24 audit could not reproduce.
anchor2 = '''HEAVY = pytest.mark.skipif(
    not os.environ.get("TENSORBEM_HEAVY_TESTS"),
    reason="set TENSORBEM_HEAVY_TESTS=1 to run the multi-minute refinement study",
)'''
heavy_test = anchor2 + '''


@HEAVY
def test_heavy_tier_reproduces_the_documented_1280_panel_ellipsoid_row():
    """The 1280-panel ellipsoid row is the one the 2026-10-24 audit left open.

    ``subdivisions=3`` was never run at the time (the audit only had 320 and
    5120 panels), so the documented 4.17e-3/4.46e-3/4.94e-3 trio had no direct
    recomputation. This runs the three-dimensional assembly the documentation
    asks for and judges it against the quoted numbers.
    """
    report = check(
        tiers=("heavy",),
        entries=tuple(
            entry for entry in DECLARED_NUMBERS if entry.group == "ellipsoid"
        ),
    )
    assert report["summary"]["checked"] == 3
    assert report["summary"]["verdict"] == "PASS", [
        (row["key"], row["declared"], row["measured"])
        for row in report["checks"]
        if row["verdict"] != "PASS"
    ]'''
if "test_heavy_tier_reproduces_the_documented_1280_panel_ellipsoid_row" not in text:
    assert text.count(anchor2) == 1
    text = text.replace(anchor2, heavy_test)

path.write_text(text)
print("patched tests")