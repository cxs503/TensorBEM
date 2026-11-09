"""Tests for cathodic protection current distribution verification.

Tests the 3-D Laplace BEM solver for ship hull cathodic protection,
verifying against DNV-RP-B401 criteria and analytical expectations.
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
    PROTECTION_POTENTIAL,
    ANODE_OPEN_CIRCUIT_POTENTIAL,
    SEAWATER_CONDUCTIVITY,
)


class TestHullMeshGeneration:
    """Test bulk carrier hull mesh generation."""

    def test_generate_closed_mesh(self):
        """Test that a closed triangular mesh can be generated."""
        hull = BulkCarrierHull()
        mesh = generate_closed_hull_mesh(hull, n_sections=11, n_side=3, n_bottom=2)
        assert mesh.panel_count > 0
        assert mesh.vertex_count > 0
        # Check that mesh is closed (TriangleMesh validates this)
        assert mesh.panel_count >= 4

    def test_hull_dimensions(self):
        """Test that hull has correct principal dimensions."""
        hull = BulkCarrierHull(length=200.0, breadth=32.0, draft=12.0)
        mesh = generate_closed_hull_mesh(hull, n_sections=11, n_side=3, n_bottom=2)
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
        mesh = generate_closed_hull_mesh(hull, n_sections=11, n_side=3, n_bottom=2)
        coated, damaged, anodes = assign_coating_damage(mesh, 0.05, seed=42)
        all_panels = np.sort(np.concatenate([coated, damaged, anodes]))
        assert len(all_panels) == mesh.panel_count
        assert np.array_equal(all_panels, np.arange(mesh.panel_count))

    def test_damage_ratio(self):
        """Test that damage ratio is approximately correct."""
        hull = BulkCarrierHull()
        mesh = generate_closed_hull_mesh(hull, n_sections=11, n_side=3, n_bottom=2)
        damage_ratio = 0.10
        coated, damaged, anodes = assign_coating_damage(mesh, damage_ratio, seed=42)
        actual_ratio = len(damaged) / mesh.panel_count
        # Allow some tolerance due to rounding
        assert abs(actual_ratio - damage_ratio) < 0.02


class TestBEMSolve:
    """Test the BEM solver for cathodic protection."""

    def test_solve_convergence(self):
        """Test that BEM system converges (low residual)."""
        hull = BulkCarrierHull()
        mesh = generate_closed_hull_mesh(hull, n_sections=11, n_side=3, n_bottom=2)
        coated, damaged, anodes = assign_coating_damage(mesh, 0.05, seed=42)
        result = solve_cathodic_protection(mesh, coated, damaged, anodes)
        # C1: Relative residual should be very small
        assert result.relative_residual < 1e-10

    def test_coated_insulating(self):
        """C2: Coated panels have zero normal current density."""
        hull = BulkCarrierHull()
        mesh = generate_closed_hull_mesh(hull, n_sections=11, n_side=3, n_bottom=2)
        coated, damaged, anodes = assign_coating_damage(mesh, 0.05, seed=42)
        result = solve_cathodic_protection(mesh, coated, damaged, anodes)
        # Coated panels have q_n = 0 by construction
        max_coated_current = float(np.max(np.abs(result.current_density[coated])))
        assert max_coated_current < 1e-12

    def test_damaged_potential(self):
        """C3: Damaged panels have prescribed protection potential."""
        hull = BulkCarrierHull()
        mesh = generate_closed_hull_mesh(hull, n_sections=11, n_side=3, n_bottom=2)
        coated, damaged, anodes = assign_coating_damage(mesh, 0.05, seed=42)
        result = solve_cathodic_protection(mesh, coated, damaged, anodes)
        # Damaged panels should be at protection potential
        damaged_potentials = result.potential[damaged]
        assert np.allclose(damaged_potentials, PROTECTION_POTENTIAL, atol=1e-10)

    def test_anode_potential(self):
        """Anode panels have prescribed anode potential."""
        hull = BulkCarrierHull()
        mesh = generate_closed_hull_mesh(hull, n_sections=11, n_side=3, n_bottom=2)
        coated, damaged, anodes = assign_coating_damage(mesh, 0.05, seed=42)
        result = solve_cathodic_protection(mesh, coated, damaged, anodes)
        # Anode panels should be at anode potential
        anode_potentials = result.potential[anodes]
        assert np.allclose(anode_potentials, ANODE_OPEN_CIRCUIT_POTENTIAL, atol=1e-10)


class TestMonotonicity:
    """C5: Test that anode demand increases with damage ratio."""

    def test_anode_demand_monotonic(self):
        """Test that anode current increases with damage ratio."""
        hull = BulkCarrierHull()
        mesh = generate_closed_hull_mesh(hull, n_sections=15, n_side=4, n_bottom=2)
        damage_ratios = [0.02, 0.05, 0.10]
        anode_currents = []
        for dr in damage_ratios:
            coated, damaged, anodes = assign_coating_damage(mesh, dr, seed=42)
            result = solve_cathodic_protection(mesh, coated, damaged, anodes)
            anode_currents.append(abs(result.anode_current_total))
        # Check monotonic increase
        for i in range(len(anode_currents) - 1):
            assert anode_currents[i + 1] > anode_currents[i]

    def test_anode_demand_linear(self):
        """Test that anode current is approximately linear with damage ratio."""
        hull = BulkCarrierHull()
        mesh = generate_closed_hull_mesh(hull, n_sections=15, n_side=4, n_bottom=2)
        damage_ratios = np.array([0.02, 0.05, 0.10])
        anode_currents = []
        for dr in damage_ratios:
            coated, damaged, anodes = assign_coating_damage(mesh, dr, seed=42)
            result = solve_cathodic_protection(mesh, coated, damaged, anodes)
            anode_currents.append(abs(result.anode_current_total))
        anode_currents = np.array(anode_currents)
        # Linear fit
        coeffs = np.polyfit(damage_ratios, anode_currents, 1)
        y_fit = np.polyval(coeffs, damage_ratios)
        ss_res = np.sum((anode_currents - y_fit) ** 2)
        ss_tot = np.sum((anode_currents - np.mean(anode_currents)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        # C5: R^2 > 0.95
        assert r_squared > 0.95


class TestMeshConvergence:
    """C6: Test mesh convergence."""

    def test_gci_small(self):
        """Test that Grid Convergence Index is small."""
        hull = BulkCarrierHull()
        mesh_levels = [11, 17, 23]
        mean_potentials = []
        for n_sec in mesh_levels:
            mesh = generate_closed_hull_mesh(hull, n_sections=n_sec, n_side=4, n_bottom=2)
            coated, damaged, anodes = assign_coating_damage(mesh, 0.05, seed=42)
            result = solve_cathodic_protection(mesh, coated, damaged, anodes)
            mean_potentials.append(float(np.mean(result.potential)))
        # GCI calculation
        phi1, phi2, phi3 = mean_potentials
        eps12 = abs(phi1 - phi2)
        eps23 = abs(phi2 - phi3)
        if eps12 > 0 and eps23 > 0 and phi1 != 0:
            gci = (eps12 / abs(phi1)) * 100.0
            # C6: GCI < 5%
            assert gci < 5.0


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
        results = run_verification_study(
            damage_ratios=(0.02, 0.05, 0.10),
            mesh_levels=(11, 17, 23),
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
            mesh_levels=(11, 17, 23),
            output_dir=output_dir,
        )
        # Check that verification criteria are present
        assert "C2_coated_insulating" in results["verification"]
        assert "C5_monotonic_anode_demand" in results["verification"]
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
        mesh = generate_closed_hull_mesh(hull, n_sections=11, n_side=3, n_bottom=2)
        coated, damaged, anodes = assign_coating_damage(mesh, 0.05, seed=42)
        result = solve_cathodic_protection(mesh, coated, damaged, anodes)
        # Evaluate at a point outside the hull
        exterior_points = np.array([[0.0, 0.0, 5.0]])  # 5m above waterplane
        potentials = evaluate_exterior_potential(mesh, result, exterior_points)
        assert len(potentials) == 1
        assert np.isfinite(potentials[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
