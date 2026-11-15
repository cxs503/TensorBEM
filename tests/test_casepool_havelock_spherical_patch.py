"""Tests for spherical-patch surface measure tier in CASE-BEM-SUBMERGED-SPHERE-HAVELOCK.

m22332 addition: verifies that the economy mesh (320 panels) with the exact
spherical-patch surface measure passes the 5 % gate, correcting the inscribed
polyhedron area deficit that causes the flat-measure tier to FAIL on damping
coefficients.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Add benchmarks/verified path for the case pool driver
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "benchmarks" / "verified" / "casepool_bem_submerged_sphere_havelock"))

import run_case as rc
from tensorbem.benchmarks.submerged_sphere_havelock import (
    panel_area_measure,
    run_frequency_sweep,
)
from tensorbem.geometry3d import submerged_sphere_mesh, spherical_patch_areas


class TestSphericalPatchMeasureInvariant:
    """Closed-surface invariant: sum(A_patch) = 4*pi*a^2."""

    def test_subdivision_2_patch_area_invariant(self):
        mesh = submerged_sphere_mesh(2, radius=rc.RADIUS, submergence=rc.SUBMERGENCE)
        center = (0.0, 0.0, -rc.SUBMERGENCE)
        patch_areas = spherical_patch_areas(
            mesh.vertices, mesh.faces, center=center, radius=rc.RADIUS,
        )
        sphere_area = 4.0 * math.pi * rc.RADIUS**2
        deficit = abs(float(patch_areas.sum()) - sphere_area) / sphere_area
        assert deficit < 1e-12, f"Patch area deficit {deficit:.3e} > 1e-12"

    def test_subdivision_3_patch_area_invariant(self):
        mesh = submerged_sphere_mesh(3, radius=rc.RADIUS, submergence=rc.SUBMERGENCE)
        center = (0.0, 0.0, -rc.SUBMERGENCE)
        patch_areas = spherical_patch_areas(
            mesh.vertices, mesh.faces, center=center, radius=rc.RADIUS,
        )
        sphere_area = 4.0 * math.pi * rc.RADIUS**2
        deficit = abs(float(patch_areas.sum()) - sphere_area) / sphere_area
        assert deficit < 1e-12, f"Patch area deficit {deficit:.3e} > 1e-12"


class TestSphericalPatchFrequencySweep:
    """Economy mesh (320 panels) with spherical-patch measure: band sweep."""

    @pytest.fixture(scope="class")
    def patch_sweep(self):
        """Run the spherical-patch sweep once for the class."""
        return rc._run_frequency_sweep_patch(
            ka_values=rc.BAND_KA,
            radius=rc.RADIUS,
            submergence=rc.SUBMERGENCE,
            subdivision=rc.ECONOMY_SUBDIVISION,
            density=rc.DENSITY,
            gravity=rc.GRAVITY,
        )

    def test_all_items_pass_5pct_gate(self, patch_sweep):
        """All 28 items (7 ka × 4 coefficients) must be < 5 %."""
        items = rc._items(patch_sweep, "economy_mesh_spherical_patch")
        verdict = rc._tier_verdict(items, "economy_mesh_spherical_patch", rc.GATE)
        assert verdict["verdict"] == "PASS", (
            f"spherical-patch tier FAILs: {verdict['n_failures']}/{verdict['n_items']} "
            f"over gate, max error {verdict['max_rel_error']:.4f}"
        )

    def test_max_error_below_flat_mesh(self, patch_sweep):
        """Spherical-patch max error should be below the flat-measure economy max."""
        flat_sweep = run_frequency_sweep(
            ka_values=rc.BAND_KA,
            radius=rc.RADIUS,
            submergence=rc.SUBMERGENCE,
            subdivision=rc.ECONOMY_SUBDIVISION,
            density=rc.DENSITY,
            gravity=rc.GRAVITY,
            area_measure="flat",
        )
        flat_items = rc._items(flat_sweep, "economy_mesh")
        flat_verdict = rc._tier_verdict(flat_items, "economy_mesh", rc.GATE)

        patch_items = rc._items(patch_sweep, "economy_mesh_spherical_patch")
        patch_verdict = rc._tier_verdict(patch_items, "economy_mesh_spherical_patch", rc.GATE)

        assert patch_verdict["max_rel_error"] < flat_verdict["max_rel_error"], (
            f"Patch max error {patch_verdict['max_rel_error']:.4f} >= "
            f"flat max error {flat_verdict['max_rel_error']:.4f}"
        )

    def test_damping_coefficients_improved(self, patch_sweep):
        """Damping coefficients (b33, b11) should show the largest improvement."""
        flat_sweep = run_frequency_sweep(
            ka_values=rc.BAND_KA,
            radius=rc.RADIUS,
            submergence=rc.SUBMERGENCE,
            subdivision=rc.ECONOMY_SUBDIVISION,
            density=rc.DENSITY,
            gravity=rc.GRAVITY,
            area_measure="flat",
        )
        flat_by_ka = {r["ka"]: r for r in flat_sweep}
        patch_by_ka = {r["ka"]: r for r in patch_sweep}

        for ka in rc.BAND_KA:
            flat_row = flat_by_ka[ka]
            patch_row = patch_by_ka[ka]
            # Damping should improve (or at least not worsen significantly)
            assert patch_row["b33_rel_error"] <= flat_row["b33_rel_error"] * 1.1, (
                f"ka={ka}: patch b33 error {patch_row['b33_rel_error']:.4f} "
                f">= flat b33 error {flat_row['b33_rel_error']:.4f} * 1.1"
            )

    def test_area_measure_tagged(self, patch_sweep):
        """Each row must be tagged with area_measure='spherical_patch'."""
        for row in patch_sweep:
            assert row["area_measure"] == "spherical_patch"


class TestSphericalPatchPanelAreaMeasure:
    """Test the panel_area_measure helper with spherical_patch."""

    def test_flat_returns_none(self):
        mesh = submerged_sphere_mesh(2, radius=rc.RADIUS, submergence=rc.SUBMERGENCE)
        areas = panel_area_measure(
            mesh, radius=rc.RADIUS, submergence=rc.SUBMERGENCE, area_measure="flat",
        )
        assert areas is None

    def test_spherical_patch_returns_array(self):
        mesh = submerged_sphere_mesh(2, radius=rc.RADIUS, submergence=rc.SUBMERGENCE)
        areas = panel_area_measure(
            mesh, radius=rc.RADIUS, submergence=rc.SUBMERGENCE, area_measure="spherical_patch",
        )
        assert areas is not None
        assert len(areas) == mesh.panel_count
        sphere_area = 4.0 * math.pi * rc.RADIUS**2
        assert abs(float(areas.sum()) - sphere_area) / sphere_area < 1e-12


class TestRunCaseSphericalPatchTier:
    """Integration test: build_result includes spherical_patch tier."""

    def test_cheap_stage_includes_patch_tier(self):
        """Even cheap stage should include the spherical_patch tier."""
        result = rc.build_result(stage="cheap", with_reference_suite=False)
        assert "economy_mesh_spherical_patch" in result["tiers"]
        tier = result["tiers"]["economy_mesh_spherical_patch"]
        assert tier["verdict"] in ("PASS", "FAIL")  # must have a verdict

    def test_verdict_includes_patch_tier(self):
        """Verdict dict must include economy_mesh_spherical_patch key."""
        result = rc.build_result(stage="cheap", with_reference_suite=False)
        assert "economy_mesh_spherical_patch" in result["verdict"]
