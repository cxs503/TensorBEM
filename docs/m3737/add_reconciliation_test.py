#!/usr/bin/env python3
"""Append the m3737 convention-reconciliation test to tests/test_declared_numbers.py.

The test is the guard the merge needs: after ``agent/m3181`` (which registered
table 10/12 under the ``log2`` observed-order convention) was unioned with
``main`` (which re-quoted the same cells against the measured mean edge length
ratio), the registry legitimately holds *two* registrations of the same four
cells.  A future merge, dedupe or "cleanup" that silently drops one of them -- or
that edits one convention without the other -- turns this test red.
"""

from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/test_declared_numbers.py"
DOC = ROOT / "docs/marine_bem_benchmarks.md"

ANCHOR = '''def test_fast_tier_reproduces_the_documented_numbers():'''

TEST = '''def test_three_dimensional_b11_order_conventions_are_reconciled():
    """The merge of agent/m3181 into main keeps *two* registrations of table
    10's ``B11'`` observed order and of table 12's rotational coupling.

    ``hulme3d`` / ``wang_linton3d`` (this branch) normalise the order by
    ``log2(e0/e1)``, i.e. ideal ``h`` halving; ``hulme_hemisphere`` /
    ``submerged_sphere`` (main, m3150) normalise by the measured ``h_mean``
    ratio and quote the rotational couplings with four significant digits.
    Both are reproducible, so both must survive; dropping either side would make
    the document silently self-contradictory.  This test pins the coexistence
    and the exact algebraic relation between the two conventions.
    """
    by_key = {entry.key: entry for entry in DECLARED_NUMBERS}
    # --- table 10, observed order of B11' (one registration per convention) --
    log2_order = {64: by_key["hulme_observed_order_b11_64"].declared,
                  256: by_key["hulme_observed_order_b11_256"].declared}
    measured_order = {64: by_key["hulme_hemisphere_64_order_b11"].declared,
                      256: by_key["hulme_hemisphere_256_order_b11"].declared}
    # both sides of the merge contributed their rows
    assert by_key["hulme_observed_order_b11_64"].group == "hulme3d"
    assert by_key["hulme_hemisphere_64_order_b11"].group == "hulme_hemisphere"
    # log2(e0/e1) <= log_h(e0/e1) whenever h_ratio < 2, i.e. the two are not
    # copies of each other; an accidental dedupe shows up as equality here.
    for panels in (64, 256):
        assert log2_order[panels] < measured_order[panels]
        # order_h = order_log2 * ln2 / ln(h_ratio)  ->  h_ratio = 2**(log2/h)
        h_ratio = 2.0 ** (log2_order[panels] / measured_order[panels])
        assert 1.80 < h_ratio < 2.00, (panels, h_ratio)
    # meshes refine towards h_ratio -> 2, so the implied ratio must grow.
    h_ratio_16_64 = 2.0 ** (log2_order[64] / measured_order[64])
    h_ratio_64_256 = 2.0 ** (log2_order[256] / measured_order[256])
    assert h_ratio_16_64 < h_ratio_64_256
    # The m3150 note quotes h0/h1 = 1.87 for the first pair; the identity must
    # reproduce the measured-h order from it to better than the registry band.
    assert abs(log2_order[64] * math.log(2.0) / math.log(1.87)
               - measured_order[64]) < 0.02

    # --- table 12, rotational coupling: 2 vs 4 significant digits ----------
    two_sig = {80: by_key["wang_linton_rotational_added_mass_max_ratio_80"],
               320: by_key["wang_linton_rotational_added_mass_max_ratio_320"]}
    four_sig = {80: by_key["submerged_sphere_80_rotational_coupling"],
                320: by_key["submerged_sphere_320_rotational_coupling"]}
    for panels in (80, 320):
        assert two_sig[panels].group == "wang_linton3d"
        assert four_sig[panels].group == "submerged_sphere"
        assert two_sig[panels].declared != four_sig[panels].declared
        # the two-significant-digit registration carries the rounding band as
        # its tolerance, and the four-significant-digit value sits inside it.
        rel = abs(two_sig[panels].declared / four_sig[panels].declared - 1.0)
        assert rel <= two_sig[panels].tolerance, (panels, rel)

    # --- the document states both conventions, explicitly reconciled -------
    text = (Path(__file__).resolve().parents[1] / "docs/marine_bem_benchmarks.md").read_text()
    assert "2.23 / 1.59 / 2.04 / 1.80 / 1.77" in text, "table 10's measured-h row"
    assert "2.06 / 1.98 / 2.01 / 1.94 / 2.02" in text, "table 10's measured-h row"
    assert "口径对齐" in text, "the reconciliation block must stay in section 4.12"
    assert "1.8021" in text and "1.9377" in text


'''


def main() -> int:
    text = TESTS.read_text()
    if "test_three_dimensional_b11_order_conventions_are_reconciled" in text:
        print("test already present")
        return 0
    assert text.count(ANCHOR) == 1
    text = text.replace(ANCHOR, TEST + ANCHOR)
    TESTS.write_text(text)
    print("appended reconciliation test to", TESTS.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())