"""Tests for Bessel functions, 2-D Helmholtz CHIEF solver and MacCamy-Fuchs diffraction."""

from __future__ import annotations

import json
import math
import subprocess
import sys

import numpy as np
import pytest

from tensorbem import (
    BESSEL_J0_FIRST_ZERO,
    BESSEL_J1_FIRST_ZERO,
    bessel_j01_y01,
    bessel_jn_yn,
    maccamy_fuchs_diffraction,
    maccamy_fuchs_surface_potential,
    regular_polygon,
    solve_helmholtz_diffraction,
    solve_vertical_cylinder_diffraction,
)


class TestBesselAndMacCamyFuchsReference:
    def test_bessel_wronskian_recurrence_and_zeros(self) -> None:
        x = np.array([1.0e-6, 1.0e-3, 0.1, 1.0, 3.8, 7.99, 8.01, 15.0, 30.0])
        j_all, y_all = bessel_jn_yn(12, x)
        j0, j1, y0, y1 = bessel_j01_y01(x)
        np.testing.assert_allclose(j_all[0], j0, rtol=1.0e-13, atol=1.0e-15)
        np.testing.assert_allclose(j_all[1], j1, rtol=1.0e-13, atol=1.0e-15)
        np.testing.assert_allclose(y_all[0], y0, rtol=1.0e-13, atol=1.0e-15)
        np.testing.assert_allclose(y_all[1], y1, rtol=1.0e-13, atol=1.0e-15)

        # Exact Wronskian J_n(x)*Y_n'(x) - J_n'(x)*Y_n(x) = 2 / (pi * x).
        for order in (0, 1):
            jp = -j_all[1] if order == 0 else 0.5 * (j_all[0] - j_all[2])
            yp = -y_all[1] if order == 0 else 0.5 * (y_all[0] - y_all[2])
            wronskian = j_all[order] * yp - jp * y_all[order]
            np.testing.assert_allclose(
                wronskian, 2.0 / (math.pi * x), rtol=1.0e-12, atol=1.0e-14
            )
        # Higher orders on moderate-to-large arguments (away from x**(-n) pole).
        x_mod = x[3:]
        j_mod, y_mod = j_all[:, 3:], y_all[:, 3:]
        for order in (2, 5, 9):
            jp = 0.5 * (j_mod[order - 1] - j_mod[order + 1])
            yp = 0.5 * (y_mod[order - 1] - y_mod[order + 1])
            wronskian = j_mod[order] * yp - jp * y_mod[order]
            np.testing.assert_allclose(
                wronskian, 2.0 / (math.pi * x_mod), rtol=1.0e-11, atol=1.0e-14
            )

        # Three-term recurrence on interior orders.
        x_mid = x[2:]
        j_mid, y_mid = j_all[:, 2:], y_all[:, 2:]
        for order in (1, 4, 8):
            np.testing.assert_allclose(
                j_mid[order - 1] + j_mid[order + 1],
                (2.0 * order / x_mid) * j_mid[order],
                rtol=1.0e-12,
                atol=1.0e-14,
            )
            np.testing.assert_allclose(
                y_mid[order - 1] + y_mid[order + 1],
                (2.0 * order / x_mid) * y_mid[order],
                rtol=1.0e-12,
                atol=1.0e-14,
            )

        j0_zero, j1_zero, _, _ = bessel_j01_y01(
            np.array([BESSEL_J0_FIRST_ZERO, BESSEL_J1_FIRST_ZERO])
        )
        assert abs(j0_zero[0]) < 1.0e-14
        assert abs(j1_zero[1]) < 1.0e-14

    def test_maccamy_fuchs_morison_and_deep_water_limits(self) -> None:
        long_wave = maccamy_fuchs_diffraction(1.0e-3, radius=1.0, depth=3.0)
        assert math.isclose(
            float(long_wave["effective_inertia_coefficient"]), 2.0, rel_tol=1.0e-5
        )
        assert abs(float(long_wave["phase_lag_radians"])) < 1.0e-6

        deep_finite = maccamy_fuchs_diffraction(1.0, radius=1.0, depth=40.0)
        deep_inf = maccamy_fuchs_diffraction(1.0, radius=1.0, depth=math.inf)
        assert deep_inf["moment_about_seabed"] is None
        assert math.isclose(
            abs(complex(deep_finite["force_x"])),
            abs(complex(deep_inf["force_x"])),
            rel_tol=1.0e-12,
        )


class TestHelmholtzAndVerticalCylinderBEM:
    def test_quadratic_convergence_of_force_moment_and_surface_potential(self) -> None:
        ka, radius, depth, rho = 1.0, 1.0, 3.0, 1025.0
        exact = maccamy_fuchs_diffraction(ka, radius=radius, depth=depth, density=rho)
        exact_fx = complex(exact["force_x"])
        exact_my = complex(exact["moment_about_seabed"])

        f_errs = []
        m_errs = []
        p_errs = []
        for count in (16, 32, 64, 128):
            mesh = regular_polygon(count, radius=radius)
            solved = solve_vertical_cylinder_diffraction(
                mesh, ka / radius, depth=depth, density=rho, chief_points="auto"
            )
            assert solved.helmholtz.condition_number < 4.0
            assert abs(solved.force[1]) < 5.0e-4 * abs(solved.force[0])
            assert solved.moment_about_seabed is not None
            assert abs(solved.moment_about_seabed[0]) < 5.0e-4 * abs(
                solved.moment_about_seabed[1]
            )
            f_errs.append(abs(solved.force[0] - exact_fx) / abs(exact_fx))
            m_errs.append(
                abs(solved.moment_about_seabed[1] - exact_my) / abs(exact_my)
            )
            angles = np.arctan2(mesh.centers[:, 1], mesh.centers[:, 0])
            exact_phi = maccamy_fuchs_surface_potential(ka, angles)
            p_errs.append(
                np.linalg.norm(solved.helmholtz.total_potential - exact_phi)
                / np.linalg.norm(exact_phi)
            )

        for series in (f_errs, m_errs, p_errs):
            assert series[0] > series[1] > series[2] > series[3]
            ratio = series[-2] / series[-1]
            assert 3.8 < ratio < 4.2
        assert f_errs[-1] < 2.0e-4
        assert p_errs[-1] < 3.0e-4

    def test_oblique_wave_incidence_rotates_horizontal_load_and_seabed_moment(self) -> None:
        mesh = regular_polygon(64, radius=1.0)
        beta = 0.65
        zero_deg = solve_vertical_cylinder_diffraction(
            mesh, 1.1, depth=2.5, incident_angle=0.0
        )
        oblique = solve_vertical_cylinder_diffraction(
            mesh, 1.1, depth=2.5, incident_angle=beta
        )
        fx0 = zero_deg.force[0]
        my0 = zero_deg.moment_about_seabed[1]
        np.testing.assert_allclose(
            oblique.force,
            np.array([fx0 * math.cos(beta), fx0 * math.sin(beta)]),
            rtol=2.0e-3,
            atol=1.0e-6,
        )
        # M = integral (z + h) * (-dF_y, +dF_x): along angle beta, M = (-sin(beta), cos(beta)) * M_y0.
        np.testing.assert_allclose(
            oblique.moment_about_seabed,
            np.array([-my0 * math.sin(beta), my0 * math.cos(beta)]),
            rtol=2.0e-3,
            atol=1.0e-6,
        )

    def test_chief_suppresses_j01_and_j11_irregular_frequencies(self) -> None:
        mesh = regular_polygon(64, radius=1.0)
        r_eff = math.sqrt(mesh.area / math.pi)

        # J_0 resonance: condition number spikes without CHIEF and stays O(1) with CHIEF.
        k_j0 = BESSEL_J0_FIRST_ZERO / r_eff
        raw_j0 = solve_helmholtz_diffraction(mesh, k_j0, chief_points=None)
        chief_j0 = solve_helmholtz_diffraction(mesh, k_j0, chief_points="auto")
        assert raw_j0.condition_number > 500.0
        assert chief_j0.condition_number < 5.0

        # J_1 dipole resonance: directly corrupts F_x without CHIEF (>8% error)
        # while CHIEF keeps the error around ~0.3%.
        k_j1 = BESSEL_J1_FIRST_ZERO / r_eff
        exact_fx = complex(maccamy_fuchs_diffraction(k_j1, radius=1.0, depth=3.0)["force_x"])
        raw_j1 = solve_vertical_cylinder_diffraction(
            mesh, k_j1, depth=3.0, chief_points=None
        )
        chief_j1 = solve_vertical_cylinder_diffraction(
            mesh, k_j1, depth=3.0, chief_points="auto"
        )
        raw_err = abs(raw_j1.force[0] - exact_fx) / abs(exact_fx)
        chief_err = abs(chief_j1.force[0] - exact_fx) / abs(exact_fx)
        assert raw_j1.helmholtz.condition_number > 400.0
        assert chief_j1.helmholtz.condition_number < 5.0
        assert raw_err > 0.08
        assert chief_err < 0.005

    def test_exterior_chief_point_is_rejected_and_cli_runs(self, tmp_path) -> None:
        mesh = regular_polygon(32, radius=1.0)
        with pytest.raises(ValueError, match="strictly inside"):
            solve_helmholtz_diffraction(
                mesh, 1.0, chief_points=np.array([[2.0, 0.0]])
            )
        with pytest.raises(ValueError, match="wavenumber"):
            solve_helmholtz_diffraction(mesh, -1.0)

        out = tmp_path / "maccamy_fuchs.json"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "tensorbem.benchmarks.maccamy_fuchs",
                "--panels",
                "16",
                "32",
                "64",
                "--output",
                str(out),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        parsed = json.loads(completed.stdout)
        assert parsed == json.loads(out.read_text(encoding="utf-8"))
        assert parsed["benchmark"] == "maccamy_fuchs_vertical_cylinder_wave_diffraction"
