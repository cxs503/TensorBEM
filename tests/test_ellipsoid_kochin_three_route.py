"""Three-route cross-validation of the ellipsoid Kochin function and wave resistance.

The ellipsoid_kochin module provides three independent evaluations of the Kochin
function H(k) for a half-submerged ellipsoid:
  1. kochin_analytic   — Bessel closed form (one spectral integral in u)
  2. kochin_quadrature — definition quadrature on elliptical projection
  3. kochin_panels     — thin-ship Rankine-source discretisation

This test cross-validates all three routes at multiple wavenumbers and checks
wave resistance consistency across routes.

Key finding: the 'panels' route is a thin-ship approximation, so it carries both
discretisation error AND modelling error O((b/a)^2). The 'analytic' and 'quadrature'
routes compute the exact ellipsoid Kochin function and agree to < 0.1%.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest

from tensorbem.ellipsoid_kochin import (
    EllipsoidHull,
    kochin_analytic,
    kochin_quadrature,
    kochin_panels,
    wave_resistance_analytic,
    wave_resistance_panels,
)


@pytest.fixture
def ellipsoid():
    """Standard test ellipsoid: a=5, b=1, c=0.5 (b/a=0.2 thin-ship regime)."""
    return EllipsoidHull(a=5.0, b=1.0, c=0.5)


@pytest.fixture
def wavenumbers():
    """Range of wavenumbers covering long-wave to short-wave."""
    return [0.3, 0.5, 1.0, 2.0, 4.0]


class TestExactRoutesAgreement:
    """The two exact routes (analytic and quadrature) must agree to high precision."""

    def test_analytic_vs_quadrature_magnitudes(self, ellipsoid, wavenumbers):
        """|H_analytic - H_quadrature| / |H_analytic| < 0.1% for all k."""
        for k in wavenumbers:
            H_a = kochin_analytic(k, ellipsoid)
            H_q = kochin_quadrature(k, ellipsoid)
            rel_err = abs(H_a - H_q) / abs(H_a)
            assert rel_err < 0.001, f"k={k}: rel_err={rel_err:.4e}"

    def test_analytic_vs_quadrature_phases(self, ellipsoid, wavenumbers):
        """Phase difference < 0.1 degree for all k."""
        for k in wavenumbers:
            H_a = kochin_analytic(k, ellipsoid)
            H_q = kochin_quadrature(k, ellipsoid)
            phase_a = np.angle(H_a)
            phase_q = np.angle(H_q)
            phase_diff = abs(phase_a - phase_q)
            if phase_diff > math.pi:
                phase_diff = 2 * math.pi - phase_diff
            assert phase_diff < 0.001, f"k={k}: phase_diff={math.degrees(phase_diff):.4f} deg"


#: Panel-route validity envelope: ``k * a <= 10`` (ten first-order panels per
#: wavelength along the half length).  Measured on the ``n_r = n_phi = 32`` grid
#: the two well-conditioned schemes stay inside 4 % for ``k a <= 10`` but the
#: ``k a = 20`` row is discretisation-limited (22 % / 58 %, see the benchmark
#: artifact ``artifacts/m14562/ellipsoid_kochin_three_route.json``); that row is
#: pinned separately by ``test_out_of_envelope_is_recorded`` instead of being
#: asserted away.
K_A_ENVELOPE = 10.0


def _in_envelope(ellipsoid, wavenumbers):
    return [k for k in wavenumbers if k * ellipsoid.a <= K_A_ENVELOPE]


class TestThinShipPanelsBounded:
    """The thin-ship panels route has modelling error O((b/a)^2) = 4%.

    These tests verify that the panels route is *bounded inside its validity
    envelope* — not that it matches the exact solution.  The discrepancy is
    modelling error plus panel discretisation, not a code bug.
    """

    def test_panels_source_bounded(self, ellipsoid, wavenumbers):
        """|H_analytic - H_panels(source)| / |H_analytic| < 20% for k a <= 10."""
        for k in _in_envelope(ellipsoid, wavenumbers):
            H_a = kochin_analytic(k, ellipsoid)
            H_p = kochin_panels(k, ellipsoid, n_r=32, n_phi=32, scheme="source")
            rel_err = abs(H_a - H_p) / abs(H_a)
            assert rel_err < 0.20, f"k={k}: rel_err={rel_err:.4e}"

    def test_panels_source_integrated_bounded(self, ellipsoid, wavenumbers):
        """|H_analytic - H_panels(source_integrated)| / |H_analytic| < 20%."""
        for k in _in_envelope(ellipsoid, wavenumbers):
            H_a = kochin_analytic(k, ellipsoid)
            H_p = kochin_panels(k, ellipsoid, n_r=32, n_phi=32, scheme="source_integrated")
            rel_err = abs(H_a - H_p) / abs(H_a)
            assert rel_err < 0.20, f"k={k}: rel_err={rel_err:.4e}"

    def test_out_of_envelope_is_recorded(self, ellipsoid):
        """Beyond ``k a = 10`` the coarse panel grid is discretisation-limited.

        This is a *pinned limitation*, not an acceptance test: the 32 x 32 grid
        misses the 10 % budget at ``k a = 20``, and the benchmark artifact shows
        the same error falling to 4.8 % at 64 x 64 panels.
        """
        k = max(w for w in (0.3, 0.5, 1.0, 2.0, 4.0) if w * ellipsoid.a > K_A_ENVELOPE)
        H_a = kochin_analytic(k, ellipsoid)
        H_p = kochin_panels(k, ellipsoid, n_r=32, n_phi=32, scheme="source_integrated")
        coarse = abs(H_a - H_p) / abs(H_a)
        H_f = kochin_panels(k, ellipsoid, n_r=64, n_phi=64, scheme="source_integrated")
        fine = abs(H_a - H_f) / abs(H_a)
        assert coarse > 0.10, f"k={k}: expected the coarse grid to miss the budget, got {coarse:.3e}"
        assert fine < coarse / 2.0, f"k={k}: refinement did not converge ({coarse:.3e} -> {fine:.3e})"

    def test_panels_ordinate_bounded(self, ellipsoid, wavenumbers):
        """|H_analytic - H_panels(ordinate)| / |H_analytic| < 50% (coarse at high k)."""
        for k in wavenumbers:
            H_a = kochin_analytic(k, ellipsoid)
            H_p = kochin_panels(k, ellipsoid, n_r=32, n_phi=32, scheme="ordinate")
            rel_err = abs(H_a - H_p) / abs(H_a)
            assert rel_err < 0.50, f"k={k}: rel_err={rel_err:.4e}"


class TestWaveResistanceConsistency:
    """Cross-validate wave resistance from different routes."""

    def test_analytic_vs_panels_resistance(self, ellipsoid):
        """R_W from wave_resistance_analytic vs wave_resistance_panels (thin-ship)."""
        for Fn in [0.2, 0.3, 0.5, 0.7]:
            R_a = wave_resistance_analytic(ellipsoid, Fn)
            R_p = wave_resistance_panels(ellipsoid, Fn, n_r=32, n_phi=32)
            
            if abs(R_a) > 1e-6:
                rel_err = abs(R_a - R_p) / abs(R_a)
                # Thin-ship modelling error O((b/a)^2) = 4%, plus discretisation
                assert rel_err < 0.20, f"Fn={Fn}: rel_err={rel_err:.4e}"


class TestConvergenceStudy:
    """Verify convergence of discretisation routes."""

    def test_panels_converge_to_analytic(self, ellipsoid):
        """Error decreases with panel count (finest better than coarsest)."""
        k = 1.0
        H_ref = kochin_analytic(k, ellipsoid)
        
        errors = []
        for n in [8, 16, 32, 64]:
            H_p = kochin_panels(k, ellipsoid, n_r=n, n_phi=n, scheme="source")
            err = abs(H_ref - H_p) / abs(H_ref)
            errors.append(err)
        
        # Finest should be better than coarsest
        assert errors[-1] < errors[0], f"Finest not better: errors={errors}"

    def test_quadrature_converges_to_analytic(self, ellipsoid):
        """kochin_quadrature converges rapidly to kochin_analytic."""
        k = 2.0
        H_ref = kochin_analytic(k, ellipsoid, n_quad=256)
        
        errors = []
        for n_psi in [24, 48, 96]:
            H_q = kochin_quadrature(k, ellipsoid, n_psi=n_psi, n_phi=2 * n_psi)
            err = abs(H_ref - H_q) / abs(H_ref)
            errors.append(err)
        
        # m15140: the definition quadrature is already at the double-precision
        # floor at n_psi = 24 (errors ~1e-15), so a monotonicity assertion here
        # would only be sampling round-off noise -- 1.18e-15 -> 1.16e-15 passes
        # by luck, 1.18e-15 -> 1.42e-15 would fail.  The honest claim is that
        # *every* resolution tested is converged; the study cannot resolve an
        # order.  See dev_report_m13787_ellipsoid_kochin_verification.md §9.4.
        assert all(err < 1.0e-12 for err in errors), (
            f"Quadrature not converged at every n_psi: errors={errors}"
        )
        # High-order quadrature should be very accurate
        assert errors[-1] < 0.001, f"Quadrature error {errors[-1]:.4e} exceeds 0.1%"


# ---------------------------------------------------------------------------
# Benchmark driver contract + committed run artifact
# ---------------------------------------------------------------------------
#
# ``tensorbem.benchmarks.ellipsoid_kochin_three_route`` is the reproduction entry
# for the numbers in ``dev_report_m14562_ellipsoid_kochin_three_route_benchmark.md``:
# it scores every Kochin route against ``kochin_analytic`` (and against the
# independent ``tensorbem.michell`` spectral reference) and writes one JSON
# artifact.  The tests below pin the CLI contract, the artifact schema, the
# predeclared thresholds, and that the committed artifact still agrees with a
# fresh recomputation.

ARTIFACT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "docs"
    / "m14562"
    / "ellipsoid_kochin_three_route.json"
)


def _load_artifact() -> dict:
    assert ARTIFACT.exists(), f"run artifact missing: {ARTIFACT}"
    return json.loads(ARTIFACT.read_text())


class TestBenchmarkDriver:
    def test_driver_main_fast_path(self, tmp_path):
        """``main`` writes a well-formed artifact and reports PASS."""
        from tensorbem.benchmarks.ellipsoid_kochin_three_route import main

        rc = main(
            [
                "--k", "0.5", "1.0",
                "--froude", "0.3",
                "--no-michell-reference",
                "--outdir", str(tmp_path),
            ]
        )
        assert rc == 0
        art = json.loads((tmp_path / "ellipsoid_kochin_three_route.json").read_text())
        assert art["pass"] is True
        assert art["geometry_gate"]["pass"] is True
        assert set(art) >= {
            "benchmark", "settings", "geometry_gate", "kochin",
            "wave_resistance", "convergence", "summary", "pass",
        }
        # michell rows are absent on the fast path, and their criteria say SKIP
        assert not any(r["route"] == "michell_reference" for r in art["kochin"])
        skipped = {c["name"] for c in art["summary"]["criteria"] if c["skipped"]}
        assert "michell_vs_analytic_kochin" in skipped

    def test_driver_reports_geometry_gate(self):
        """A hull whose b >= a is rejected before any solve (thin-ship gate)."""
        from tensorbem.ellipsoid_kochin import EllipsoidHull

        with pytest.raises(ValueError):
            EllipsoidHull(a=1.0, b=1.0, c=0.5)


class TestCommittedArtifact:
    """The committed artifact must satisfy the predeclared verdicts."""

    def test_geometry_gate_passes(self):
        art = _load_artifact()
        g = art["geometry_gate"]
        assert g["pass"] is True
        assert g["Cb"] == pytest.approx(math.pi / 6.0, rel=1e-12)
        assert g["b_over_a"] == pytest.approx(0.2)

    def test_exact_routes_agree_to_roundoff(self):
        """analytic vs quadrature vs michell spectral reference: < 1e-10."""
        art = _load_artifact()
        rows = [r for r in art["kochin"]
                if r["route"] in ("quadrature", "michell_reference")]
        assert rows
        assert max(r["rel_err"] for r in rows) < 1e-10

    def test_all_criteria_pass(self):
        art = _load_artifact()
        scored = [c for c in art["summary"]["criteria"] if not c["skipped"]]
        assert scored, "no scored criteria in artifact"
        for c in scored:
            assert c["pass"] is True, f"{c['name']}: worst={c['worst']} vs {c['threshold']}"
        assert art["pass"] is True

    def test_panel_envelope_is_honest(self):
        """Out-of-envelope panel rows are recorded and excluded from the verdict."""
        art = _load_artifact()
        env = art["settings"]["envelope_ka"]
        a = art["geometry_gate"]["a"]
        ooe = art["summary"]["out_of_envelope_rows"]
        assert ooe, "expected k*a > envelope rows on the default k-grid"
        for r in ooe:
            assert r["k"] * a > env
        # the worst out-of-envelope panel error is recorded, not hidden
        assert max(r["rel_err"] for r in ooe) > 0.1
        # and the scored worst for the panel route stays inside the envelope
        scored = next(c for c in art["summary"]["criteria"]
                      if c["name"] == "panels_integrated_vs_analytic_kochin")
        in_env = [r["rel_err"] for r in art["kochin"]
                  if r["route"] == "panels_integrated" and r["in_envelope"]]
        assert scored["worst"] == pytest.approx(max(in_env))
        assert scored["pass"] is True

    def test_convergence_is_monotone_and_reaches_budget(self):
        art = _load_artifact()
        errs = [r["rel_err"] for r in art["convergence"]]
        assert len(errs) >= 4
        assert all(b <= a for a, b in zip(errs, errs[1:])), errs
        assert errs[-1] < 0.10

    def test_artifact_matches_fresh_recomputation(self):
        """Recompute the cheap routes; the artifact must not be stale."""
        from tensorbem.benchmarks.ellipsoid_kochin_three_route import run
        from tensorbem.ellipsoid_kochin import EllipsoidHull

        art = _load_artifact()
        g = art["geometry_gate"]
        hull = EllipsoidHull(a=g["a"], b=g["b"], c=g["c"])
        fresh = run(hull, ks=(1.0,), froudes=(0.3,), include_michell=False,
                    convergence_sizes=(16,))
        by_route = {r["route"]: r for r in art["kochin"] if r["k"] == 1.0}
        for row in fresh["kochin"]:
            old = by_route[row["route"]]
            assert old["rel_err"] == pytest.approx(row["rel_err"], rel=1e-9, abs=1e-12)
            assert old["got_re"] == pytest.approx(row["got_re"], rel=1e-9)
        old_r = next(r for r in art["wave_resistance"]
                     if r["route"] == "panels_integrated" and r["froude"] == 0.3)
        new_r = fresh["wave_resistance"][0]
        assert old_r["got_n"] == pytest.approx(new_r["got_n"], rel=1e-9)
