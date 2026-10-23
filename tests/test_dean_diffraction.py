"""Tests for 2-D deep-water wave diffraction, Dean's theorem and Haskind reciprocity."""

from __future__ import annotations

import json
import math
import subprocess
import sys

import numpy as np
import pytest

from tensorbem import (
    dean_cylinder_diffraction,
    diffraction_reference,
    haskind_excitation_force,
    solve_diffraction_indirect,
    solve_radiation_indirect,
    wave_amplitude,
)
from tensorbem.benchmarks.dean_diffraction import submerged_ellipse_mesh
from tensorbem.benchmarks.ursell_cylinder import submerged_circle_mesh
from tensorbem.geometry import PanelMesh


class TestDeanAnalyticalReference:
    def test_zero_reflection_unit_transmission_and_circular_polarisation(self) -> None:
        ref = dean_cylinder_diffraction(
            radius=1.0, submergence=1.6, ka=0.75, amplitude=1.2, density=1025.0
        )
        assert ref["reflection"] == 0.0j
        transmission = complex(ref["transmission"])
        assert math.isclose(abs(transmission), 1.0, rel_tol=1.0e-13, abs_tol=1.0e-13)
        fx = complex(ref["force_x"])
        fz = complex(ref["force_z"])
        assert abs(fz - (-1.0j * fx)) / abs(fx) < 1.0e-14

        # Multipole truncation is converged well before terms=24.
        ref_fine = dean_cylinder_diffraction(
            radius=1.0, submergence=1.6, ka=0.75, amplitude=1.2, density=1025.0, terms=36
        )
        assert abs(transmission - complex(ref_fine["transmission"])) < 1.0e-13
        assert abs(fx - complex(ref_fine["force_x"])) / abs(fx) < 1.0e-13

    def test_analytical_force_matches_radiation_wave_amplitude_by_haskind(self) -> None:
        radius, submergence, ka, rho, g = 1.0, 2.0, 0.5, 1025.0, 9.80665
        nu = ka / radius
        exact = dean_cylinder_diffraction(
            radius, submergence, ka, amplitude=1.0, density=rho, gravity=g
        )
        mesh = submerged_circle_mesh(96, radius, submergence)
        rad = solve_radiation_indirect(mesh, nu, "sway", density=rho)
        # By symmetry on a symmetric body |A^-| == |A^+| == |wave_amplitude|,
        # and Haskind gives |X_x| = rho * g * A * |A^-|.
        amp_plus = wave_amplitude(mesh, rad.potential, rad.normal_velocity, nu)
        expected_modulus = rho * g * abs(amp_plus)
        assert math.isclose(
            abs(complex(exact["force_x"])), expected_modulus, rel_tol=8.0e-3
        )


class TestFreeSurfaceDiffractionAndHaskind:
    def test_odd_panel_nodes_and_vertex_depth_validation(self) -> None:
        circle = submerged_circle_mesh(32, 1.0, 2.0)
        # Odd Gauss-Legendre order includes midpoint xi = 0 on the self panel;
        # the Rankine split must prevent 0/0 NaNs in both radiation & diffraction.
        rad = solve_radiation_indirect(circle, 0.5, "heave", panel_nodes=15)
        diff = solve_diffraction_indirect(circle, 0.5, panel_nodes=15)
        assert np.all(np.isfinite(rad.potential))
        assert np.all(np.isfinite(rad.added_mass)) and np.all(np.isfinite(rad.damping))
        assert np.all(np.isfinite(diff.total_potential))
        assert math.isfinite(diff.energy_balance_error)

        # A polygon whose panel midpoints lie below z = 0 while a vertex touches z = 0
        # must still be rejected.
        touching = PanelMesh(
            np.array([[0.0, 0.0], [-0.5, -0.5], [0.0, -1.0], [0.5, -0.5]])
        )
        assert np.all(touching.centers[:, 1] < 0.0)
        with pytest.raises(ValueError, match="below the free surface"):
            solve_diffraction_indirect(touching, 0.5)

    def test_circular_cylinder_verifies_deans_theorem_convergence_and_haskind(self) -> None:
        radius, submergence, ka, amp, rho = 1.0, 2.0, 0.6, 1.0, 1025.0
        nu = ka / radius
        exact = dean_cylinder_diffraction(radius, submergence, ka, amp, rho)
        exact_t = complex(exact["transmission"])
        exact_fx = complex(exact["force_x"])
        exact_fz = complex(exact["force_z"])

        force_errors = []
        trans_errors = []
        for panels in (32, 64, 128):
            mesh = submerged_circle_mesh(panels, radius, submergence)
            solved = solve_diffraction_indirect(
                mesh, nu, amplitude=amp, density=rho, panel_nodes=16
            )
            assert abs(solved.reflection_coefficient) < 1.0e-12
            assert solved.energy_balance_error < 1.0e-12
            fx, fz = solved.excitation_force
            assert abs(fz - (-1.0j * fx)) / abs(fx) < 1.0e-11
            force_errors.append(abs(fx - exact_fx) / abs(exact_fx))
            trans_errors.append(abs(solved.transmission_coefficient - exact_t))

        assert force_errors[0] > force_errors[1] > force_errors[2]
        assert trans_errors[0] > trans_errors[1] > trans_errors[2]
        assert force_errors[2] < 5.0e-3
        assert trans_errors[2] < 5.0e-3

        # Independent MFS diffraction check + Haskind reciprocity on N=64.
        mesh64 = submerged_circle_mesh(64, radius, submergence)
        diff64 = solve_diffraction_indirect(mesh64, nu, amplitude=amp, density=rho, panel_nodes=16)
        mfs = diffraction_reference(mesh64, nu, amplitude=amp, source_count=32, density=rho)
        assert abs(mfs.reflection_coefficient) < 1.0e-12
        assert abs(mfs.excitation_force[0] - exact_fx) / abs(exact_fx) < 2.0e-3

        rad_sway = solve_radiation_indirect(mesh64, nu, "sway", density=rho, panel_nodes=16)
        rad_heave = solve_radiation_indirect(mesh64, nu, "heave", density=rho, panel_nodes=16)
        hx = haskind_excitation_force(mesh64, rad_sway, nu, amplitude=amp, density=rho)
        hz = haskind_excitation_force(mesh64, rad_heave, nu, amplitude=amp, density=rho)
        assert abs(hx - diff64.excitation_force[0]) / abs(exact_fx) < 1.0e-4
        assert abs(hz - diff64.excitation_force[1]) / abs(exact_fz) < 1.0e-4

    def test_elliptic_cylinder_reflects_waves_and_conserves_energy(self) -> None:
        ellipse = submerged_ellipse_mesh(64, semi_major=1.4, semi_minor=0.7, submergence=2.0)
        nu = 0.6
        diff = solve_diffraction_indirect(ellipse, nu, panel_nodes=16)
        # Non-circular cylinder violates Dean's zero-reflection property,
        # yet still obeys |R|**2 + |T|**2 = 1 and Haskind reciprocity.
        assert abs(diff.reflection_coefficient) > 0.05
        assert diff.energy_balance_error < 1.0e-11

        rad_sway = solve_radiation_indirect(ellipse, nu, "sway", panel_nodes=16)
        rad_heave = solve_radiation_indirect(ellipse, nu, "heave", panel_nodes=16)
        hx = haskind_excitation_force(ellipse, rad_sway, nu)
        hz = haskind_excitation_force(ellipse, rad_heave, nu)
        assert abs(hx - diff.excitation_force[0]) / abs(diff.excitation_force[0]) < 2.0e-3
        assert abs(hz - diff.excitation_force[1]) / abs(diff.excitation_force[1]) < 3.0e-3

    def test_cli_writes_strict_json(self, tmp_path) -> None:
        out = tmp_path / "dean.json"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "tensorbem.benchmarks.dean_diffraction",
                "--panels",
                "48",
                "--panel-nodes",
                "12",
                "--output",
                str(out),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        parsed = json.loads(completed.stdout)
        assert parsed == json.loads(out.read_text(encoding="utf-8"))
        assert parsed["circular_cylinder"]["panel_reflection_modulus"] < 1.0e-12
