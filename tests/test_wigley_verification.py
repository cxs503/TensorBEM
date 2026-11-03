"""Verification tests for the Wigley hullform and Michell wave-resistance module.

These tests are the executable form of ``benchmarks/wigley_verification.py``:
every number in ``docs/wigley_wave_resistance.md`` and every accuracy claim of
``src/tensorbem/wigley.py`` is either compared with an independent reference or
explicitly registered as unverifiable.  The references are, in order of
strength,

* an exact analytic integral (``int_1^inf dlam/(lam^2 sqrt(lam^2-1)) = 1``),
* a composite Gauss-Legendre solution of the same Michell integral on a grid
  much finer than the production call,
* a direct two-dimensional integral of ``dY/dx exp(kz + ikx)`` over the hull,
* the volume of the generated panel mesh (independent of the analytic
  properties helper).
"""

import math

import pytest

from tensorbem.benchmarks.wigley_verification import (
    G,
    HULL,
    converged_wave_resistance,
    doc_claims,
    geometry_audit,
    jacobian_integral,
    kochin_numeric,
    legacy_wave_resistance,
    main as cli_main,
    pressure_path_audit,
    verify,
)
from tensorbem.wigley import (
    michell_kochin_function,
    michell_wave_resistance,
    wigley_hull_mesh,
    wigley_hull_properties,
)

L, B, T = HULL


class TestWigleyGeometry:
    """Cb / volume: the closed-form integral against the meshed hull."""

    def test_volume_is_four_ninths_LBT(self):
        props = wigley_hull_properties(L, B, T)
        assert props["volume"] == pytest.approx(4.0 * L * B * T / 9.0, rel=1e-14)
        assert props["waterplane_area"] == pytest.approx(2.0 * L * B / 3.0, rel=1e-14)

    def test_block_coefficient_is_four_ninths(self):
        props = wigley_hull_properties(L, B, T)
        assert props["Cb"] == pytest.approx(4.0 / 9.0, rel=1e-14)
        # Cb = Cm * Cp with Cm = Cp = 2/3 for this hullform
        assert props["Cb"] == pytest.approx(props["Cm"] * props["Cp"], rel=1e-14)
        assert props["Cm"] == pytest.approx(2.0 / 3.0, rel=1e-14)

    def test_meshed_volume_agrees_with_closed_form(self):
        """Second, independent route: integrate the generated panel mesh."""
        audit = geometry_audit(L, B, T)
        last = audit["mesh_ladder"][-1]
        assert last["rel_error_vs_closed_form"] < 1e-3
        assert last["mesh_Cb"] == pytest.approx(4.0 / 9.0, rel=2e-3)
        # the volume integral is second-order accurate
        orders = [o for o in audit["observed_order"] if math.isfinite(o)]
        assert orders and min(orders) > 1.5

    def test_superseded_two_thirds_claim_is_rejected(self):
        """The old ``volume = 2LBT/3`` / ``Cb = 2/3`` pair must not come back."""
        superseded = geometry_audit(L, B, T)["closed_form"]["superseded_claim"]
        assert superseded["ratio_to_truth"] == pytest.approx(1.5, rel=1e-12)
        mesh = wigley_hull_mesh(L, B, T, 32, 16)
        assert abs(mesh.volume / (L * B * T) - 2.0 / 3.0) > 0.2


class TestMichellQuadrature:
    """Accuracy of the Kochin integral: exact yardstick + converged reference."""

    def test_exact_jacobian_is_reproduced_by_substitution(self):
        """``int_1^lam_max dlam/(lam^2 sqrt(lam^2-1)) -> 1`` for the production scheme."""
        for lam_max in (10.0, 1.0e2, 1.0e3):
            value = jacobian_integral(lam_max, n_quad=256, substituted=True)
            truncation = 0.5 / lam_max**2  # tail bound of the exact integral
            assert abs(value - 1.0) < 10.0 * truncation + 1e-6

    def test_legacy_scheme_fails_the_exact_yardstick(self):
        """The pre-fix scheme integrated straight across the lam = 1 cusp."""
        lam_max = 1.0e3
        legacy = jacobian_integral(lam_max, n_quad=256, substituted=False)
        fixed = jacobian_integral(lam_max, n_quad=256, substituted=True)
        assert abs(legacy - 1.0) > 5e-2
        assert abs(fixed - 1.0) < 1e-5

    @pytest.mark.parametrize("froude", [0.20, 0.30, 0.40])
    def test_production_matches_converged_reference(self, froude):
        U = froude * math.sqrt(G * L)
        ref = converged_wave_resistance(L, B, T, U)
        prod = michell_wave_resistance(L, B, T, U)
        assert abs(prod - ref) / ref < 1e-4

    @pytest.mark.parametrize("froude", [0.20, 0.30, 0.40])
    def test_legacy_scheme_is_worse_than_production(self, froude):
        U = froude * math.sqrt(G * L)
        ref = converged_wave_resistance(L, B, T, U)
        legacy_err = abs(legacy_wave_resistance(L, B, T, U) - ref) / ref
        prod_err = abs(michell_wave_resistance(L, B, T, U) - ref) / ref
        assert legacy_err > 1e-3
        assert prod_err * 100.0 < legacy_err

    def test_node_count_insensitivity(self):
        U = 0.3 * math.sqrt(G * L)
        a = michell_wave_resistance(L, B, T, U, n_quad=128)
        b = michell_wave_resistance(L, B, T, U, n_quad=256)
        assert abs(a - b) / b < 1e-5

    def test_wave_resistance_vanishes_at_zero_speed(self):
        assert michell_wave_resistance(L, B, T, 0.0) == 0.0


class TestKochinReduction:
    """The closed-form x/z integrals against a direct 2-D integral."""

    @pytest.mark.parametrize("k", [0.3, 1.0, 5.0])
    def test_closed_form_matches_numeric_surface_integral(self, k):
        closed = michell_kochin_function(L, B, T, k)
        numeric = kochin_numeric(L, B, T, k)
        assert abs(numeric - closed) / abs(closed) < 1e-9


class TestPressurePathAudit:
    """The pressure path is audited, not trusted."""

    def test_pressure_path_is_negative_and_non_convergent(self):
        audit = pressure_path_audit(L, B, T, froude=(0.30,))
        assert audit["all_negative"] is True
        assert audit["monotone_convergence"] is False
        # the discrepancy is far larger than any acceptance band
        assert audit["relative_difference_max"] > 0.5

    def test_documented_discrepancy_band_is_mesh_dependent(self):
        audit = pressure_path_audit(L, B, T, froude=(0.30,))
        ratios = [r["ratio_pressure_over_kochin"] for r in audit["rows"]]
        assert max(ratios) - min(ratios) > 0.1


class TestDocumentedClaims:
    """Every claim in ``docs/wigley_wave_resistance.md`` has a verdict."""

    def test_claim_registry_covers_the_document(self):
        claims = doc_claims(L, B, T)
        verdicts = [c["verdict"] for c in claims]
        assert len(claims) >= 7
        assert all(v in {"PASS", "DRIFT", "UNSUPPORTED"} for v in verdicts)

    def test_corrected_geometry_claims_pass(self):
        by_claim = {c["claim"]: c for c in doc_claims(L, B, T)}
        assert by_claim["Waterplane area: A_wp = 2LB/3"]["verdict"] == "PASS"
        # the two geometry DRIFTs are what this mission corrected
        assert "Cb" in "".join(by_claim)
        drift_claims = [c["claim"] for c in doc_claims(L, B, T) if c["verdict"] == "DRIFT"]
        assert any("Pressure path converges" in c for c in drift_claims)

    def test_verify_report_is_self_consistent(self):
        report = verify(L, B, T)
        summary = report["summary"]
        assert summary["max_rel_error_production_quadrature"] < 1e-4
        assert summary["quadrature_improvement_factor"] > 100.0
        assert summary["max_rel_error_kochin_reduction"] < 1e-9
        assert report["geometry"]["closed_form"]["Cb"] == pytest.approx(4.0 / 9.0, rel=1e-14)


def test_cli_writes_report(tmp_path):
    out = tmp_path / "wigley.json"
    assert cli_main(["--output", str(out), "--no-figures"]) == 0
    assert out.exists()
    assert '"doc_claims"' in out.read_text()