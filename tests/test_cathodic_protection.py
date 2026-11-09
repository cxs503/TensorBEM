"""Tests for cathodic protection current distribution verification.

Tests the 3-D Laplace BEM solver for ship hull cathodic protection,
verifying against DNV-RP-B401 criteria and analytical expectations.

KNOWN LIMITATIONS
-----------------
The current implementation uses a simplified Dirichlet-Dirichlet formulation:
- Anode panels: phi = -1.05V (Dirichlet)
- Damaged panels: phi = -0.80V (Dirichlet)
- Coated panels: q_n = 0 (Neumann, insulating)

This formulation does NOT capture the electrochemical polarization behavior
of the anode and cathode. In a real CP system, the current-potential relationship
is nonlinear (polarization curves), and the anode current demand increases with
coating damage area.

The simplified formulation is suitable for:
- Verifying the BEM infrastructure (Laplace solver, mesh generation)
- Testing the mixed boundary condition assembly
- Checking numerical convergence

For production CP design, a proper formulation would use:
- Polarization curves for anode and cathode
- Iterative solution for the nonlinear boundary conditions
- Validation against experimental data or DNV-RP-B401 simplified methods

WHAT IS VERIFIED HERE
---------------------
The residual freedom of the linear model (no polarization kinetics) does not
weaken the anchors that *are* available, and those are pinned by tests:

* **Physical current scale (C8)** -- an equipotential sphere is the exact
  closed form ``I = 4 pi sigma a phi_a`` / ``phi = phi_a a / r``.  The solver
  reproduces the current to < 3 % at >= 320 panels; the historical defect of
  the module (current reported without Ohm's law, i.e. low by ``1/sigma``)
  would show up here as a 71 % error.
* **Conductivity enters linearly (Ohm's law)** -- doubling ``sigma`` doubles
  the current density and the anode current, and
  ``current_density == sigma * single_layer_density``.
* **Wetted-area accounting** -- the artificial waterline lid (36 % of the
  closed-mesh area of the 200 m bulk carrier) carries no holiday and no anode
  and is excluded from the wetted-area statistics used for the DNV comparison.
* **Honest criteria** -- C4/C5 are reported with their real verdicts for the
  linear model instead of being asserted true by construction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.cathodic_protection import (
    BulkCarrierHull,
    generate_closed_hull_mesh,
    assign_coating_damage,
    solve_cathodic_protection,
    evaluate_exterior_potential,
    check_laplace_residual,
    compute_dnv_anode_demand,
    run_verification_study,
    verify_conducting_sphere,
    waterline_lid_mask,
    PROTECTION_POTENTIAL,
    ANODE_OPEN_CIRCUIT_POTENTIAL,
    SEAWATER_CONDUCTIVITY,
)


class TestHullMeshGeneration:
    """Test bulk carrier hull mesh generation."""

    def test_generate_closed_mesh(self):
        """Test that a closed triangular mesh can be generated."""
        hull = BulkCarrierHull()
        mesh = generate_closed_hull_mesh(hull, n_sections=9, n_side=3, n_bottom=2)
        assert mesh.panel_count > 0
        assert mesh.vertex_count > 0
        # Check that mesh is closed (TriangleMesh validates this)
        assert mesh.panel_count >= 4

    def test_hull_dimensions(self):
        """Test that hull has correct principal dimensions."""
        hull = BulkCarrierHull(length=200.0, breadth=32.0, draft=12.0)
        mesh = generate_closed_hull_mesh(hull, n_sections=9, n_side=3, n_bottom=2)
        verts = mesh.vertices
        length = float(np.ptp(verts[:, 0]))
        breadth = float(np.ptp(verts[:, 1]))
        draft = float(np.ptp(verts[:, 2]))
        # Allow 5% tolerance for parametric hull
        assert abs(length - 200.0) / 200.0 < 0.05
        assert abs(breadth - 32.0) / 32.0 < 0.05
        assert abs(draft - 12.0) / 12.0 < 0.05


class TestCoatingDamage:
    """Test coating damage assignment."""

    def test_partition_coverage(self):
        """Test that panel partition covers all panels."""
        hull = BulkCarrierHull()
        mesh = generate_closed_hull_mesh(hull, n_sections=9, n_side=3, n_bottom=2)
        coated, damaged, anodes = assign_coating_damage(mesh, 0.05, seed=42)
        all_panels = np.sort(np.concatenate([coated, damaged, anodes]))
        assert len(all_panels) == mesh.panel_count
        assert np.array_equal(all_panels, np.arange(mesh.panel_count))

    def test_damage_ratio(self):
        """Test that damage ratio is approximately correct."""
        hull = BulkCarrierHull()
        mesh = generate_closed_hull_mesh(hull, n_sections=9, n_side=3, n_bottom=2)
        damage_ratio = 0.10
        coated, damaged, anodes = assign_coating_damage(mesh, damage_ratio, seed=42)
        lid = waterline_lid_mask(mesh)
        n_wetted = int((~lid).sum())
        actual_ratio = len(damaged) / n_wetted
        # The draw is over wetted panels only; allow rounding of a few panels
        assert abs(actual_ratio - damage_ratio) < 0.01

    def test_lid_never_damaged_or_anode(self):
        """The artificial waterline lid is neither a holiday nor an anode."""
        hull = BulkCarrierHull()
        mesh = generate_closed_hull_mesh(hull, n_sections=9, n_side=3, n_bottom=2)
        lid = waterline_lid_mask(mesh)
        assert lid.sum() > 0, "closed hull must have a waterline lid"
        coated, damaged, anodes = assign_coating_damage(mesh, 0.10, seed=42)
        assert not np.any(lid[damaged])
        assert not np.any(lid[anodes])


class TestBEMSolve:
    """Test the BEM solver for cathodic protection."""

    def test_solve_convergence(self):
        """Test that BEM system converges (low residual)."""
        hull = BulkCarrierHull()
        mesh = generate_closed_hull_mesh(hull, n_sections=9, n_side=3, n_bottom=2)
        coated, damaged, anodes = assign_coating_damage(mesh, 0.05, seed=42)
        result = solve_cathodic_protection(mesh, coated, damaged, anodes)
        # C1: Relative residual should be very small
        assert result.relative_residual < 1e-10

    def test_coated_insulating(self):
        """C2: Coated panels have zero normal current density."""
        hull = BulkCarrierHull()
        mesh = generate_closed_hull_mesh(hull, n_sections=9, n_side=3, n_bottom=2)
        coated, damaged, anodes = assign_coating_damage(mesh, 0.05, seed=42)
        result = solve_cathodic_protection(mesh, coated, damaged, anodes)
        # Coated panels have q_n = 0 by construction
        max_coated_current = float(np.max(np.abs(result.current_density[coated])))
        assert max_coated_current < 1e-12

    def test_damaged_potential(self):
        """C3: Damaged panels have prescribed protection potential."""
        hull = BulkCarrierHull()
        mesh = generate_closed_hull_mesh(hull, n_sections=9, n_side=3, n_bottom=2)
        coated, damaged, anodes = assign_coating_damage(mesh, 0.05, seed=42)
        result = solve_cathodic_protection(mesh, coated, damaged, anodes)
        # Damaged panels should be at protection potential
        damaged_potentials = result.potential[damaged]
        assert np.allclose(damaged_potentials, PROTECTION_POTENTIAL, atol=1e-10)

    def test_anode_potential(self):
        """Anode panels have prescribed anode potential."""
        hull = BulkCarrierHull()
        mesh = generate_closed_hull_mesh(hull, n_sections=9, n_side=3, n_bottom=2)
        coated, damaged, anodes = assign_coating_damage(mesh, 0.05, seed=42)
        result = solve_cathodic_protection(mesh, coated, damaged, anodes)
        # Anode panels should be at anode potential
        anode_potentials = result.potential[anodes]
        assert np.allclose(anode_potentials, ANODE_OPEN_CIRCUIT_POTENTIAL, atol=1e-10)


class TestMeshConvergence:
    """C6: mesh convergence.

    The section-count hull family is *not* a systematic refinement (the
    number of anode panels changes discretely with ``n_sections``), so the
    order of accuracy is pinned on the nested-icosphere conducting sphere
    instead of on a fabricated GCI of the hull mesh.
    """

    def test_sphere_second_order(self):
        """Nested icosphere refinement shows O(h^2) current convergence."""
        report = verify_conducting_sphere(subdivisions=(1, 2, 3))
        order = report["observed_order_finest_pair"]
        assert order is not None
        assert 1.5 < order < 2.5, report["observed_orders"]
        errors = [lv["current_rel_error"] for lv in report["levels"]]
        assert errors[0] > errors[1] > errors[2]

    def test_hull_family_reports_no_fake_order(self):
        """The hull family reports no GCI instead of an invented one."""
        results = run_verification_study(
            damage_ratios=(0.05,),
            mesh_levels=(9, 11, 13),
            sphere_levels=(1, 2),
            output_dir="/tmp/cp_test_conv",
        )
        summary = results["mesh_convergence"]
        assert summary["quantity"].startswith("|anode current|")
        # Either a justified GCI (monotone decay) or an explicit refusal.
        if summary["gci_percent"] is None:
            assert summary["note"]
        else:
            assert summary["observed_order"] > 0


class TestDNVComparison:
    """Test comparison with DNV-RP-B401 simplified method."""

    def test_dnv_anode_demand(self):
        """Test DNV simplified anode demand calculation."""
        total_area = 10000.0  # m^2
        damage_ratio = 0.05
        dnv = compute_dnv_anode_demand(total_area, damage_ratio)
        assert dnv["damaged_area_m2"] == 500.0
        assert dnv["coated_area_m2"] == 9500.0
        assert dnv["total_current_A"] > 0


class TestVerificationStudy:
    """Test the full verification study."""

    def test_run_study(self, tmp_path):
        """Test that the verification study runs and produces output."""
        output_dir = tmp_path / "cathodic_protection"
        # Use smaller meshes for faster execution
        results = run_verification_study(
            damage_ratios=(0.02, 0.05, 0.10),
            mesh_levels=(9, 11, 13),
            sphere_levels=(1, 2),
            output_dir=output_dir,
        )
        # Check that JSON files were created
        assert (output_dir / "case.json").exists()
        assert (output_dir / "results.json").exists()
        # Check that results contain expected keys
        assert "cases" in results
        assert "verification" in results
        assert len(results["cases"]) == 3

    def test_verification_criteria(self, tmp_path):
        """Test that verification criteria are checked."""
        output_dir = tmp_path / "cathodic_protection"
        results = run_verification_study(
            damage_ratios=(0.02, 0.05, 0.10),
            mesh_levels=(9, 11, 13),
            sphere_levels=(1, 2),
            output_dir=output_dir,
        )
        # Check that verification criteria are present
        assert "C2_coated_insulating" in results["verification"]
        assert "C7_json_output" in results["verification"]
        # C2 should pass
        assert results["verification"]["C2_coated_insulating"]["pass"]
        # C7 should pass
        assert results["verification"]["C7_json_output"]["pass"]


class TestExteriorPotential:
    """Test exterior potential evaluation."""

    def test_evaluate_exterior(self):
        """Test that exterior potential can be evaluated."""
        hull = BulkCarrierHull()
        mesh = generate_closed_hull_mesh(hull, n_sections=9, n_side=3, n_bottom=2)
        coated, damaged, anodes = assign_coating_damage(mesh, 0.05, seed=42)
        result = solve_cathodic_protection(mesh, coated, damaged, anodes)
        # Evaluate at a point outside the hull
        exterior_points = np.array([[0.0, 0.0, 5.0]])  # 5m above waterplane
        potentials = evaluate_exterior_potential(mesh, result, exterior_points)
        assert len(potentials) == 1
        assert np.isfinite(potentials[0])


class TestNegativeControls:
    """C4: Negative control tests.
    
    NOTE: The simplified Dirichlet-Dirichlet formulation does not capture
    the correct physical behavior for these tests. The tests verify that
    the solver runs without errors and produces finite results.
    """

    def test_solver_runs_with_small_damage(self):
        """Verify solver runs with very small damage ratio."""
        hull = BulkCarrierHull()
        mesh = generate_closed_hull_mesh(hull, n_sections=9, n_side=3, n_bottom=2)
        coated, damaged, anodes = assign_coating_damage(mesh, 0.001, seed=42)
        result = solve_cathodic_protection(mesh, coated, damaged, anodes)
        # Anode current should be finite
        assert np.isfinite(result.anode_current_total)
        # Note: The magnitude is not physically meaningful with the simplified formulation

    def test_solver_runs_with_default_conductivity(self):
        """Verify solver runs with default seawater conductivity."""
        hull = BulkCarrierHull()
        mesh = generate_closed_hull_mesh(hull, n_sections=9, n_side=3, n_bottom=2)
        coated, damaged, anodes = assign_coating_damage(mesh, 0.05, seed=42)
        result = solve_cathodic_protection(mesh, coated, damaged, anodes)
        # Current should be finite
        assert np.isfinite(result.anode_current_total)


class TestPhysicalCurrentScale:
    """C8: exact conducting-sphere anchor for the physical current scale."""

    def test_conducting_sphere_spreading_resistance(self):
        """Total current matches 4 pi sigma a phi_a to < 3 % and refines."""
        report = verify_conducting_sphere(subdivisions=(1, 2))
        assert report["pass"], report
        finest = report["levels"][-1]
        assert finest["current_rel_error"] < 0.03
        assert finest["max_potential_rel_error"] < 0.03
        # Refinement must reduce the current error (icosphere family is nested)
        errors = [lv["current_rel_error"] for lv in report["levels"]]
        assert errors[-1] < errors[0]

    def test_sphere_spreading_resistance_value(self):
        """The closed-form spreading resistance is 1/(4 pi sigma a)."""
        report = verify_conducting_sphere(
            subdivisions=(1, 3), radius=2.0, potential=-0.5, conductivity=3.5,
            evaluation_radii=(2.5, 3.0, 5.0),
        )
        assert report["exact_current_A"] == pytest.approx(
            4.0 * math.pi * 3.5 * 2.0 * -0.5, rel=1e-12
        )
        assert report["exact_spreading_resistance_ohm"] == pytest.approx(
            1.0 / (4.0 * math.pi * 3.5 * 2.0), rel=1e-12
        )
        assert report["pass"], report["levels"]

    def test_evaluation_inside_sphere_rejected(self):
        """Evaluation radii inside the sphere are refused, not silently wrong."""
        with pytest.raises(ValueError):
            verify_conducting_sphere(subdivisions=(1,), radius=2.0,
                                     evaluation_radii=(1.5, 3.0))

    def test_conductivity_scales_current_linearly(self):
        """Ohm's law: current density and anode current scale with sigma."""
        hull = BulkCarrierHull()
        mesh = generate_closed_hull_mesh(hull, n_sections=9, n_side=3, n_bottom=2)
        coated, damaged, anodes = assign_coating_damage(mesh, 0.05, seed=42)
        sigma = SEAWATER_CONDUCTIVITY
        base = solve_cathodic_protection(mesh, coated, damaged, anodes,
                                         conductivity=sigma)
        doubled = solve_cathodic_protection(mesh, coated, damaged, anodes,
                                            conductivity=2.0 * sigma)
        assert doubled.anode_current_total == pytest.approx(
            2.0 * base.anode_current_total, rel=1e-12
        )
        # current_density is the conductivity-scaled single-layer density
        assert np.allclose(base.current_density, sigma * base.single_layer_density)
        assert base.conductivity == pytest.approx(sigma)

    def test_invalid_conductivity_rejected(self):
        hull = BulkCarrierHull()
        mesh = generate_closed_hull_mesh(hull, n_sections=5, n_side=2, n_bottom=1)
        coated, damaged, anodes = assign_coating_damage(mesh, 0.05, seed=42)
        with pytest.raises(ValueError):
            solve_cathodic_protection(mesh, coated, damaged, anodes, conductivity=0.0)


class TestWettedAreaAccounting:
    """The waterline lid must not be counted as wetted steel."""

    def test_lid_area_excluded(self):
        mesh = generate_closed_hull_mesh(BulkCarrierHull(), n_sections=11,
                                         n_side=3, n_bottom=2)
        coated, damaged, anodes = assign_coating_damage(mesh, 0.05, seed=42)
        result = solve_cathodic_protection(mesh, coated, damaged, anodes)
        closed_area = float(np.sum(result.panel_areas))
        assert result.lid_area > 0.0
        assert result.wetted_area == pytest.approx(closed_area - result.lid_area)
        # For the 200 m bulk carrier the lid is a large fraction of the
        # closed-mesh area; the old study used the closed area for the DNV
        # comparison, inflating the demand.
        assert 0.30 < result.lid_area / closed_area < 0.45

    def test_coverage_uses_requested_threshold(self):
        """The coverage statistic follows the requested protection potential."""
        mesh = generate_closed_hull_mesh(BulkCarrierHull(), n_sections=9,
                                         n_side=3, n_bottom=2)
        coated, damaged, anodes = assign_coating_damage(mesh, 0.10, seed=42)
        strict = solve_cathodic_protection(mesh, coated, damaged, anodes,
                                           protection_potential=-0.80)
        loose = solve_cathodic_protection(mesh, coated, damaged, anodes,
                                          protection_potential=-0.20)
        assert loose.protection_coverage > strict.protection_coverage
        assert 0.0 <= strict.protection_coverage <= 1.0
        assert loose.protection_coverage <= 1.0


class TestStudyHonesty:
    """The study must report measured verdicts, not assumed ones."""

    def test_criteria_are_measured(self, tmp_path):
        results = run_verification_study(
            damage_ratios=(0.02, 0.05, 0.10),
            mesh_levels=(9, 11, 13),
            sphere_levels=(1, 2),
            output_dir=tmp_path / "cp",
        )
        verification = results["verification"]
        # C8 sphere anchor must pass
        assert verification["C8_conducting_sphere"]["pass"]
        assert verification["C8_conducting_sphere"]["relative_current_error"] < 0.03
        # C5 verdict must follow the direction-checked criterion
        c5 = verification["C5_monotonic_anode_demand"]
        assert c5["pass"] == (c5["r_squared"] > 0.95 and c5["slope_A_per_unit_damage"] > 0.0)
        # C2 is measured, not asserted
        assert verification["C2_coated_insulating"]["max_coated_current_density_A_m2"] <= 1e-12
        # The DNV comparison uses the wetted area of the sweep mesh
        case = results["cases"][0]
        dnv = compute_dnv_anode_demand(case["wetted_area_m2"], case["damage_ratio"])
        assert case["anode_current_dnv_A"] == pytest.approx(dnv["total_current_A"])
        # Sphere anchor is persisted
        assert (tmp_path / "cp" / "results.json").exists()
        payload = json.loads((tmp_path / "cp" / "results.json").read_text())
        assert payload["sphere_verification"]["pass"]


REPORT_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs" / "cathodic_protection_linear" / "REPORT.md"
)
RESULTS_PATH = REPORT_PATH.parent / "results.json"
SPHERE_PATH = REPORT_PATH.parent / "sphere-spreading-resistance.json"


class TestReportIsPinnedToArtifacts:
    """The published report must quote the artefacts, not hand-copied numbers."""

    def test_report_numbers_come_from_the_artifacts(self):
        if not REPORT_PATH.exists():  # pragma: no cover - docs not shipped
            pytest.skip("report not present")
        text = REPORT_PATH.read_text(encoding="utf-8")
        sphere = json.loads(SPHERE_PATH.read_text(encoding="utf-8"))
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        finest = sphere["levels"][-1]
        quoted = [
            f"{sphere['exact_current_A']:.4f}".replace("-", "\u2212"),
            f"{finest['current_A']:.4f}".replace("-", "\u2212"),
            f"{100 * finest['current_rel_error']:.3f} %",
            f"{100 * finest['max_potential_rel_error']:.3f} %",
            f"{sphere['observed_order_finest_pair']:.3f}",
            f"{results['hull']['wetted_area_m2']:.1f}",
            f"{results['hull']['lid_area_m2']:.1f}",
        ]
        for value in quoted:
            assert value in text, f"report does not quote {value!r} from the artefacts"
        for case in results["cases"]:
            assert f"{case['anode_current_dnv_A']:.2f}" in text
            assert f"{case['bem_to_dnv_current_ratio']:.3f}" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
