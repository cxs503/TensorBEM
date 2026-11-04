"""Extend the registry tests and document the B11' order convention.

1. tests/test_declared_numbers.py: the coverage test must now include the
   three 3-D free-surface groups, and a new standard-tier test judges the
   whole family (66 entries) against the benchmarks.
2. docs/marine_bem_benchmarks.md: state explicitly that the B11' order
   column of table 10 uses the panel-count convention while the other four
   order columns use h_mean (the m3280 precedent for mixed conventions).
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------
# 1. tests
# --------------------------------------------------------------------------
tp = REPO_ROOT / "tests" / "test_declared_numbers.py"
t = tp.read_text()

old_cov = """    assert {
        "cylinder",
        "ellipsoid",
        "interior_harmonics",
        "ursell",
        "ursell_reference",
    } <= groups"""
new_cov = """    assert {
        "cylinder",
        "ellipsoid",
        "interior_harmonics",
        "ursell",
        "ursell_reference",
        "hulme_hemisphere",
        "hulme_chief",
        "submerged_sphere",
    } <= groups"""
assert t.count(old_cov) == 1
t = t.replace(old_cov, new_cov)

old_tail = """def test_report_is_strict_json():"""
new_tail = '''def test_standard_tier_reproduces_the_documented_3d_free_surface_numbers():
    """Tables 10-12 of docs/marine_bem_benchmarks.md were documentation-only.

    The 2026-11-04 heavy-tier settlement (m3059) closed the 2-D and
    unbounded-flow families and explicitly deferred the 3-D free-surface
    family because the *full* suites (frequency sweeps) cost minutes.  The
    tables quote only the refinement studies (~15 s total), so the family
    is judged in the standard tier: every quoted error is a pure
    discretisation error against the Hulme (1982) wave-free-potential
    reference or the zero-quadrature Thorne multipole expansion.
    """
    report = check(
        tiers=("standard",),
        entries=tuple(
            entry
            for entry in DECLARED_NUMBERS
            if entry.group in {"hulme_hemisphere", "hulme_chief", "submerged_sphere"}
        ),
    )
    assert report["summary"]["checked"] == 66
    assert report["summary"]["verdict"] == "PASS", [
        (row["key"], row["declared"], row["measured"])
        for row in report["checks"]
        if row["verdict"] != "PASS"
    ]
    # The CHIEF row is the physical claim: without regularisation the
    # unregularized BIE must still produce the unphysical negative damping,
    # and the 5-point interior CHIEF must remove it.
    by_key = {row["key"]: row for row in report["checks"]}
    assert by_key["chief_b33_unregularized_magnitude"]["measured"] > 0.0
    assert by_key["chief_b33_chief"]["measured"] > 0.0
    assert (
        by_key["chief_cond_unregularized"]["measured"]
        > by_key["chief_cond_chief"]["measured"]
    )


def test_report_is_strict_json():'''
assert t.count(old_tail) == 1
t = t.replace(old_tail, new_tail)
tp.write_text(t)

# --------------------------------------------------------------------------
# 2. document
# --------------------------------------------------------------------------
dp = REPO_ROOT / "docs" / "marine_bem_benchmarks.md"
d = dp.read_text()
old_doc = "两者相差恰为因子 2，由回归测试钉死。）"
new_doc = ("两者相差恰为因子 2，由回归测试钉死。）"
           "**口径说明（m3631）**：本表 B11' 列的观测阶（1.62/1.88）按 m2609 的面元数口径 "
           "$h\\propto N^{-1/2}$ 计，其余四列按网格平均边长 $h_{\\text{mean}}$ 计；"
           "两种口径在 16→64 层相差约 11%（该层 $h_{\\text{mean}}$ 比 1.87 而非 2.0），"
           "注册表 `tensorbem.benchmarks.declared_numbers` 按各自口径分别钉桩，"
           "standard 层默认复算。")
assert d.count(old_doc) == 1
d = d.replace(old_doc, new_doc)
dp.write_text(d)
print("tests + doc patched")
