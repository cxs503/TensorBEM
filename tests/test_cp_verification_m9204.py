"""Tests for cathodic protection verification (m9204).

Tests the 3-D Laplace BEM solver for ship hull cathodic protection with
Butler-Volmer kinetics, verifying against:
- Analytical solutions (infinite plate, infinite cylinder)
- DNV-RP-B401 criteria
- Mesh convergence (GCI)
- Charge conservation
- Parameter study (coating breakdown ratio × anode count)

Acceptance criteria:
(a) Infinite plate: BEM vs analytical error < 1% (L2 norm)
(b) Infinite cylinder: BEM vs analytical error < 2% (L2 norm)
(c) Simplified hull: DNV-RP-B401 window satisfied for design case
(d) Anode current magnitude: β=0.01 → I < 10A, β=0.2 → I < 200A
(e) Mesh convergence: GCI < 2%
(f) Charge conservation: error < 0.1%
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.cp_galvanic import (
    BoxHull,
    box_hull_mesh,
    partition_surface,
    solve_galvanic_protection,
    dnv_envelope,
    grid_convergence_index,
    plate_linear_benchmark,
    cylinder_log_benchmark,
    sphere_exterior_benchmark,
    SEAWATER_CONDUCTIVITY,
    PROTECTION_POTENTIAL,
    OVERPROTECTION_POTENTIAL,
    MESH_LEVELS,
)


class TestAnalyticalBenchmarks:
    """Test analytical benchmarks against exact solutions."""

    def test_plate_linear_benchmark(self):
        """(a) Infinite parallel plates: BEM vs linear profile, L2 < 1%."""
        result = plate_linear_benchmark(n_horizontal=8, n_vertical=2)
        assert result["potential_rel_l2"] < 0.01, (
            f"Plate L2 error {result['potential_rel_l2']:.6e} >= 1%"
        )
        assert result["relative_residual"] < 1e-10
        assert result["wall_flux_max_abs"] < 1e-12

    def test_cylinder_log_benchmark(self):
        """(b) Coaxial cylinders: BEM vs log profile, L2 < 2%."""
        result = cylinder_log_benchmark(n_theta=32, n_radial=6, n_axial=1)
        assert result["potential_rel_l2"] < 0.02, (
            f"Cylinder L2 error {result['potential_rel_l2']:.6e} >= 2%"
        )
        assert result["relative_residual"] < 1e-10
        assert result["cap_flux_max_abs"] < 1e-12

    def test_sphere_exterior_benchmark(self):
        """Exterior sphere: BEM vs 1/r profile."""
        result = sphere_exterior_benchmark(subdivisions=2)
        assert result["potential_rel_l2"] < 0.03
        assert result["flux_rel_error"] < 0.02
        assert result["relative_residual"] < 1e-10


class TestDNVCriteria:
    """Test DNV-RP-B401 protection criteria."""

    @pytest.fixture(scope="class")
    def design_case(self):
        """Design case: beta=0.05, N_anodes=4, medium mesh."""
        hull = BoxHull(length=100.0, breadth=20.0, draft=10.0)
        mesh, groups = box_hull_mesh(hull, *MESH_LEVELS["medium"])
        partition = partition_surface(mesh, groups, beta=0.05, n_anodes=4)
        result = solve_galvanic_protection(mesh, partition)
        return {"hull": hull, "mesh": mesh, "partition": partition, "result": result}

    def test_dnv_window_design_case(self, design_case):
        """(c) Design case satisfies DNV-RP-B401 protection window."""
        result = design_case["result"]
        envelope = dnv_envelope(result)
        assert envelope["window_ok"], (
            f"DNV window not satisfied: phi_min={envelope['phi_min']:.4f}V, "
            f"phi_max={envelope['phi_max']:.4f}V"
        )
        assert envelope["all_points_protected"]
        assert envelope["no_overprotection"]

    def test_metal_potential_range(self, design_case):
        """Metal potential is within expected range."""
        result = design_case["result"]
        # Metal potential should be between anode and protection potential
        assert -1.10 < result.metal_potential < -0.70, (
            f"Metal potential {result.metal_potential:.4f}V out of expected range"
        )


class TestAnodeCurrentScale:
    """Test anode current magnitude vs coating breakdown."""

    @pytest.fixture(scope="class")
    def current_study(self):
        """Run current study for beta=0.01 and beta=0.2."""
        hull = BoxHull(length=100.0, breadth=20.0, draft=10.0)
        mesh, groups = box_hull_mesh(hull, *MESH_LEVELS["medium"])

        results = {}
        for beta in [0.01, 0.2]:
            partition = partition_surface(mesh, groups, beta=beta, n_anodes=4)
            result = solve_galvanic_protection(mesh, partition)
            results[beta] = result
        return results

    def test_beta_001_current(self, current_study):
        """(d) β=0.01: anode current < 10A."""
        result = current_study[0.01]
        assert result.anode_current_total < 10.0, (
            f"Anode current {result.anode_current_total:.2f}A >= 10A for β=0.01"
        )

    def test_beta_02_current(self, current_study):
        """(d) β=0.2: anode current < 200A."""
        result = current_study[0.2]
        assert result.anode_current_total < 200.0, (
            f"Anode current {result.anode_current_total:.2f}A >= 200A for β=0.2"
        )


class TestMeshConvergence:
    """Test mesh convergence and GCI."""

    @pytest.fixture(scope="class")
    def convergence_data(self):
        """Run mesh convergence study."""
        hull = BoxHull(length=100.0, breadth=20.0, draft=10.0)
        results = {}
        for name, spec in MESH_LEVELS.items():
            mesh, groups = box_hull_mesh(hull, *spec)
            partition = partition_surface(mesh, groups, beta=0.05, n_anodes=4)
            result = solve_galvanic_protection(mesh, partition)
            results[name] = {
                "n_panels": mesh.panel_count,
                "cell_size_m": float(np.sqrt(np.mean(mesh.areas))),
                "anode_current_A": result.anode_current_total,
                "phi_mean_bare": float(np.mean(result.electrolyte_potential[partition.bare])),
            }
        return results

    def test_gci_anode_current(self, convergence_data):
        """(e) GCI of anode current < 2%."""
        gci = grid_convergence_index(
            convergence_data["fine"]["anode_current_A"],
            convergence_data["medium"]["anode_current_A"],
            convergence_data["coarse"]["anode_current_A"],
            convergence_data["fine"]["cell_size_m"],
            convergence_data["medium"]["cell_size_m"],
            convergence_data["coarse"]["cell_size_m"],
        )
        gci_pct = gci.get("gci_fine_percent", 100)
        assert gci_pct < 2.0, f"GCI {gci_pct:.4f}% >= 2%"


class TestChargeConservation:
    """Test charge conservation."""

    @pytest.fixture(scope="class")
    def charge_study(self):
        """Run charge conservation study."""
        hull = BoxHull(length=100.0, breadth=20.0, draft=10.0)
        mesh, groups = box_hull_mesh(hull, *MESH_LEVELS["medium"])

        imbalances = []
        for beta in [0.01, 0.05, 0.1, 0.2]:
            partition = partition_surface(mesh, groups, beta=beta, n_anodes=4)
            result = solve_galvanic_protection(mesh, partition)
            imbalances.append(result.charge_imbalance)
        return imbalances

    def test_charge_conservation(self, charge_study):
        """(f) Charge imbalance < 0.1%."""
        max_imbalance = max(abs(imb) for imb in charge_study)
        assert max_imbalance < 0.001, (
            f"Max charge imbalance {max_imbalance:.2e} >= 0.1%"
        )


class TestParameterStudy:
    """Test parameter study (beta × N_anodes)."""

    @pytest.fixture(scope="class")
    def param_study(self):
        """Run parameter study."""
        hull = BoxHull(length=100.0, breadth=20.0, draft=10.0)
        mesh, groups = box_hull_mesh(hull, *MESH_LEVELS["medium"])

        cases = []
        for beta in [0.01, 0.05, 0.1, 0.2]:
            for n_anodes in [2, 4, 8]:
                partition = partition_surface(mesh, groups, beta=beta, n_anodes=n_anodes)
                result = solve_galvanic_protection(mesh, partition)
                envelope = dnv_envelope(result)
                cases.append({
                    "beta": beta,
                    "n_anodes": n_anodes,
                    "anode_current_A": result.anode_current_total,
                    "phi_min": envelope["phi_min"],
                    "phi_max": envelope["phi_max"],
                    "window_ok": envelope["window_ok"],
                })
        return cases

    def test_current_increases_with_beta(self, param_study):
        """Anode current should increase with coating breakdown."""
        for n_anodes in [2, 4, 8]:
            currents = [
                c["anode_current_A"] for c in param_study
                if c["n_anodes"] == n_anodes
            ]
            betas = [c["beta"] for c in param_study if c["n_anodes"] == n_anodes]
            # Check monotonic increase (allowing small numerical noise)
            for i in range(len(currents) - 1):
                assert currents[i] <= currents[i + 1] * 1.01, (
                    f"Current not monotonic for N={n_anodes}: "
                    f"{currents[i]:.2f}A at β={betas[i]} vs "
                    f"{currents[i+1]:.2f}A at β={betas[i+1]}"
                )

    def test_current_decreases_with_more_anodes(self, param_study):
        """Anode current per anode should decrease with more anodes."""
        for beta in [0.01, 0.05, 0.1, 0.2]:
            currents = [
                c["anode_current_A"] / c["n_anodes"]
                for c in param_study
                if c["beta"] == beta
            ]
            # Per-anode current should decrease (or stay similar) with more anodes
            assert currents[0] >= currents[2] * 0.9, (
                f"Per-anode current not decreasing for β={beta}: "
                f"{currents[0]:.2f}A (N=2) vs {currents[2]:.2f}A (N=8)"
            )
