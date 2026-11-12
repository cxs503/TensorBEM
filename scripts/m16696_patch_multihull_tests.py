"""m16696 patch: retarget the multihull frequency-dependence tests at the exact
multipole reference and pin what is now declared unsupported."""

from __future__ import annotations

import pathlib

PATH = pathlib.Path("tests/test_multihull_frequency_dependence.py")

NEW_GEOMETRIC = '''    def test_high_frequency_added_mass_is_the_exact_series_value(self):
        """A(ka->inf) is NOT rho*pi*a**2: the free-surface image lowers it 6.8%."""
        a_over_f = 0.3 / 0.8
        A_inf, B_inf = single_cylinder_heave_coefficients(100.0, a_over_f)
        # exact multipole value; the pre-16696 formula hand-patched A = 1 - O(1/ka)
        assert A_inf == pytest.approx(0.931755, rel=2.0e-4)
        assert B_inf < 1.0e-20
        asym = verify_asymptotics(0.3, 0.8)
        assert asym["infinite_frequency"]["pass"] is True
        # the retracted "geometric limit" claim is recorded, not silently dropped
        assert asym["infinite_frequency"]["retracted_claim_error"] > 0.05
'''

NEW_BEM = '''    def test_bem_agrees_with_the_exact_series(self):
        """BEM vs exact Ursell multipole: measured 2.5% in A, 3.5% in B at N=48."""
        results = bem_cross_validation(
            radius=0.3, depth=0.8, n_panels=48,
            ka_values=[0.3, 0.5, 1.0],
        )
        for r in results:
            assert "error" not in r, f"BEM failed at ka={r['ka']}: {r.get('error')}"
            # the pre-16696 gate was a factor-3 band on A and no gate at all on B,
            # which is why a 200x damping error in the reference went unnoticed.
            assert r["A_rel_error"] < 0.05, (r["ka"], r["A_rel_error"])
            assert r["B_rel_error"] < 0.05, (r["ka"], r["B_rel_error"])
'''

NEW_CLASS = '''

# ---------------------------------------------------------------------------
# Exact-reference provenance (mission 16696)
# ---------------------------------------------------------------------------

class TestExactReferenceProvenance:
    """The "analytic" side is the repo's exact multipole series, and the pieces
    that have no reference are declared unsupported instead of fitted."""

    def test_single_hull_reference_is_the_multipole_series(self):
        from tensorbem.analytic import ursell_cylinder_coefficients

        for ka in (0.01, 0.3, 1.0, 10.0):
            for a_over_f in (0.2, 0.375, 0.6):
                assert single_cylinder_heave_coefficients(ka, a_over_f) == pytest.approx(
                    ursell_cylinder_coefficients(ka, a_over_f), rel=1.0e-12, abs=0.0
                )

    def test_truncation_is_converged(self):
        for ka in (0.01, 0.5, 5.0):
            coarse = single_cylinder_heave_coefficients(ka, 0.375, 40)
            fine = single_cylinder_heave_coefficients(ka, 0.375, 120)
            assert coarse[0] == pytest.approx(fine[0], rel=1.0e-10, abs=0.0)
            assert coarse[1] == pytest.approx(fine[1], rel=1.0e-10, abs=1.0e-14)

    def test_mutual_damping_is_declared_unsupported(self):
        assert MUTUAL_DAMPING_SUPPORTED is False
        A_mut, B_mut = mutual_interaction_coefficients(0.5, 0.375, 5.0)
        assert A_mut > 0.0
        assert B_mut == 0.0  # not claimed, never a fitted number
        assert verify_asymptotics(0.3, 0.8)["mutual_damping_supported"] is False

    def test_mutual_added_mass_is_the_exact_unbounded_fluid_value(self):
        from tensorbem.analytic import two_cylinder_added_mass

        reference = two_cylinder_added_mass(1.0, 5.0, 1.0)
        expected = reference.mutual_added_mass_transverse / reference.displaced_mass_per_span
        assert mutual_interaction_coefficients(0.5, 0.375, 5.0)[0] == pytest.approx(
            expected, rel=1.0e-12
        )
        # an unbounded fluid has no length scale: the value cannot depend on ka
        low = mutual_interaction_coefficients(0.05, 0.375, 5.0)[0]
        high = mutual_interaction_coefficients(5.0, 0.375, 5.0)[0]
        assert low == pytest.approx(high, rel=1.0e-14)

    def test_overlapping_cylinders_are_nan_not_zero(self):
        assert math.isnan(mutual_interaction_coefficients(0.5, 0.375, 1.5)[0])
        assert math.isnan(mutual_interaction_coefficients(0.5, 0.375, 2.0)[1])
'''


def replace_block(text: str, start: str, end: str, new: str) -> str:
    i = text.index(start)
    j = text.index(end, i)
    return text[:i] + new + text[j:]


def main() -> None:
    text = PATH.read_text(encoding="utf-8")

    text = replace_block(
        text,
        "    def test_infinite_frequency_added_mass_geometric(self):",
        "    def test_damping_non_negative_all_frequencies(self):",
        NEW_GEOMETRIC + "\n",
    )
    text = replace_block(
        text,
        "    def test_bem_added_mass_same_order(self):",
        "\n\n# ---------------------------------------------------------------------------\n# Full verification",
        NEW_BEM + "\n",
    )
    text = text.replace(
        "from tensorbem.benchmarks.multihull_frequency_dependence import (\n    bem_cross_validation,",
        "from tensorbem.benchmarks.multihull_frequency_dependence import (\n    MUTUAL_DAMPING_SUPPORTED,\n    bem_cross_validation,",
    )
    text = text.rstrip("\n") + NEW_CLASS
    PATH.write_text(text, encoding="utf-8")
    print("patched", PATH)


if __name__ == "__main__":
    main()