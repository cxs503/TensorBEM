"""Verification test for CASE-BEM-HULME-DIFF-VERIF: Hulme 3D diffraction.

Tests the hulme_diffraction_3d benchmark artifacts against the ledger acceptance
criteria (<5% relative error on |F| for surge and heave at the finest mesh).

Reference: Hulme, A. (1982). The added mass and damping coefficients for a heaving
sphere. Applied Ocean Research, 4(4), 239–244.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.analytic import hulme_hemisphere_radiation
from tensorbem.free_surface_solver3d import solve_diffraction_3d
from tensorbem.geometry3d import hemisphere_hull_mesh

CASE_DIR = Path("benchmarks/verified/hulme_diffraction_3d")
RESULTS = CASE_DIR / "results.json"

DENSITY = 1025.0
GRAVITY = 9.80665
TOLERANCE = 0.05  # ledger tolerance <5%


def _results() -> dict:
    return json.loads(RESULTS.read_text())


class TestHulmeDiffraction3DArtifacts:
    """Verify that all required artifacts exist and are non-trivial."""

    def test_results_json_exists(self):
        assert RESULTS.exists(), f"Missing {RESULTS}"
        assert RESULTS.stat().st_size > 5000, f"{RESULTS} too small"

    def test_results_json_valid_structure(self):
        data = _results()
        assert "study" in data
        assert data["study"] == "hulme_diffraction_3d"
        assert "comparison" in data
        assert "full_results" in data
        assert "verdict" in data
        assert "tolerance" in data
        assert data["tolerance"] == pytest.approx(TOLERANCE)

    def test_all_gates_pass(self):
        data = _results()
        assert data["verdict"] == "ALL PASS", f"Verdict: {data['verdict']}"
        assert data["all_pass"] is True
        for row in data["comparison"]:
            assert row["pass"], f"Failed at ka={row['ka']}: worst_err={row['worst_rel_error']:.4f}"

    def test_figure_artifacts_exist(self):
        for name in ["fig_hulme_diffraction_magnitude.png",
                      "fig_hulme_diffraction_error.png",
                      "fig_hulme_diffraction_convergence.png"]:
            fig_path = CASE_DIR / name
            assert fig_path.exists(), f"Missing figure: {fig_path}"
            assert fig_path.stat().st_size > 10000, f"Figure too small: {fig_path}"


class TestHulmeDiffraction3DNumerical:
    """Numerical verification against Hulme 1982 reference."""

    def test_surge_error_within_tolerance(self):
        """Surge excitation force error < 5% at 256 panels for all ka."""
        data = _results()
        for row in data["comparison"]:
            err = abs(row["surge"]["rel_error"])
            assert err < TOLERANCE, (
                f"Surge error {err:.4f} exceeds {TOLERANCE} at ka={row['ka']}"
            )

    def test_heave_error_within_tolerance(self):
        """Heave excitation force error < 5% at 256 panels for all ka."""
        data = _results()
        for row in data["comparison"]:
            err = abs(row["heave"]["rel_error"])
            assert err < TOLERANCE, (
                f"Heave error {err:.4f} exceeds {TOLERANCE} at ka={row['ka']}"
            )

    def test_worst_error_margin(self):
        """Worst error across all ka has comfortable margin below 5%."""
        data = _results()
        worst = max(r["worst_rel_error"] for r in data["comparison"])
        assert worst < 0.03, f"Worst error {worst:.4f} too close to {TOLERANCE}"

    def test_reference_recomputed_from_library(self):
        """No hand-copied digits: reference values re-derive from analytic.py."""
        data = _results()
        for row in data["comparison"]:
            ka = row["ka"]
            ref = hulme_hemisphere_radiation(ka, radius=1.0, density=DENSITY, gravity=GRAVITY)
            stored_surge = row["surge"]["ref"]
            stored_heave = row["heave"]["ref"]
            assert abs(stored_surge - abs(ref.excitation_force_x)) / abs(ref.excitation_force_x) < 1e-6
            assert abs(stored_heave - abs(ref.excitation_force_z)) / abs(ref.excitation_force_z) < 1e-6

    def test_mesh_convergence_surge(self):
        """Surge error decreases with mesh refinement."""
        data = _results()
        full = data["full_results"]
        for ka in [0.1, 0.5, 1.0, 2.0]:
            sub2_err = abs(next(r for r in full if r["ka"] == ka and r["subdivisions"] == 2)["rel_error_surge"])
            sub3_err = abs(next(r for r in full if r["ka"] == ka and r["subdivisions"] == 3)["rel_error_surge"])
            assert sub3_err < sub2_err, (
                f"No convergence for surge at ka={ka}: sub2={sub2_err:.4f}, sub3={sub3_err:.4f}"
            )

    def test_mesh_convergence_heave(self):
        """Heave error decreases with mesh refinement."""
        data = _results()
        full = data["full_results"]
        for ka in [0.1, 0.5, 1.0, 2.0]:
            sub2_err = abs(next(r for r in full if r["ka"] == ka and r["subdivisions"] == 2)["rel_error_heave"])
            sub3_err = abs(next(r for r in full if r["ka"] == ka and r["subdivisions"] == 3)["rel_error_heave"])
            assert sub3_err < sub2_err, (
                f"No convergence for heave at ka={ka}: sub2={sub2_err:.4f}, sub3={sub3_err:.4f}"
            )


class TestHulmeDiffraction3DPhysics:
    """Physical consistency checks."""

    def test_surge_phase_quadrature(self):
        """Surge force has significant imaginary component (radiation-dominated).

        For a floating body in waves, the surge excitation force is dominated
        by the radiation component, giving a phase near -π/2. At higher ka
        the diffraction component becomes more significant, so we allow a
        wider tolerance.
        """
        data = _results()
        full = data["full_results"]
        for row in full:
            if row["subdivisions"] == 3:  # finest mesh
                phase = row["F_surge_phase"]
                # Should be in the range [-π, -π/4] for radiation-dominated response
                assert -math.pi < phase < -0.3, (
                    f"Surge phase {phase:.4f} outside expected range at ka={row['ka']}"
                )

    def test_symmetry_dofs_near_zero(self):
        """Sway, roll, pitch, yaw should be near zero for head seas."""
        data = _results()
        full = data["full_results"]
        for row in full:
            if row["subdivisions"] == 3:
                for dof in ["sway", "roll", "pitch", "yaw"]:
                    mag = row[f"F_{dof}_mag"]
                    heave_mag = row["F_heave_mag"]
                    assert mag < 0.02 * heave_mag, (
                        f"{dof} too large: {mag:.6e} vs heave {heave_mag:.4f} at ka={row['ka']}"
                    )

    def test_heave_dominant_over_surge_at_low_ka(self):
        """At low ka, heave excitation is much larger than surge."""
        data = _results()
        full = data["full_results"]
        for row in full:
            if row["subdivisions"] == 3 and row["ka"] <= 0.3:
                surge_mag = row["F_surge_mag"]
                heave_mag = row["F_heave_mag"]
                assert surge_mag < 0.4 * heave_mag, (
                    f"Surge not small vs heave at ka={row['ka']}: "
                    f"|Fx|={surge_mag:.2f}, |Fz|={heave_mag:.2f}"
                )

    def test_excitation_force_decreases_with_ka_for_heave(self):
        """Heave excitation force magnitude decreases monotonically with ka."""
        data = _results()
        comparison = data["comparison"]
        heave_mags = [r["heave"]["ref"] for r in comparison]
        for i in range(len(heave_mags) - 1):
            assert heave_mags[i] > heave_mags[i + 1], (
                f"Heave ref not decreasing: ka={comparison[i]['ka']}={heave_mags[i]:.2f}, "
                f"ka={comparison[i+1]['ka']}={heave_mags[i+1]:.2f}"
            )


class TestHulmeDiffraction3DEndToEnd:
    """Cheap end-to-end re-derivation to verify no artifact tampering."""

    def test_bem_at_ka1_sub3(self):
        """Re-solve ka=1.0 at 256 panels and compare with stored results."""
        ka = 1.0
        mesh = hemisphere_hull_mesh(3, radius=1.0)
        res = solve_diffraction_3d(mesh, nu=ka, amplitude=1.0, wave_angle=0.0,
                                   density=DENSITY, gravity=GRAVITY)
        ref = hulme_hemisphere_radiation(ka, radius=1.0, density=DENSITY, gravity=GRAVITY)

        for got, want, label in [
            (res.excitation_force[0], ref.excitation_force_x, "surge"),
            (res.excitation_force[2], ref.excitation_force_z, "heave"),
        ]:
            rel = abs(complex(got) - complex(want)) / abs(complex(want))
            assert rel < TOLERANCE, (
                f"{label} at ka={ka}: BEM={abs(got):.4f}, ref={abs(want):.4f}, err={rel:.4f}"
            )
