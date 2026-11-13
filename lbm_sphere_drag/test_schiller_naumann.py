"""Tests for the Schiller-Naumann and Clift-Grace-Weber correlations.

These tests verify the reference correlation implementations that will be
used to validate LBM sphere drag simulations against the GAP-LBM-SPHERE
anchor criteria (C_D vs correlation < 8%).
"""

import math
import pytest

from lbm_sphere_drag.schiller_naumann import (
    schiller_naumann_cd,
    schiller_naumann_valid_range,
    clift_grace_weber_cd,
    drag_force_from_cd,
    reynolds_number,
    benchmark_table,
)


class TestReynoldsNumber:
    def test_basic(self):
        # Water at 20°C: nu ≈ 1.004e-6 m²/s
        # U = 0.1 m/s, D = 0.01 m → Re ≈ 996
        re = reynolds_number(0.1, 0.01, 1.004e-6)
        assert re == pytest.approx(996.0, rel=0.01)

    def test_stokes_regime(self):
        # Very low Re: U = 0.001 m/s, D = 0.001 m, nu = 1e-4
        re = reynolds_number(0.001, 0.001, 1e-4)
        assert re == pytest.approx(0.01)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            reynolds_number(1.0, 0.01, 0)
        with pytest.raises(ValueError):
            reynolds_number(1.0, -0.01, 1e-6)
        with pytest.raises(ValueError):
            reynolds_number(-1.0, 0.01, 1e-6)


class TestSchillerNaumann:
    def test_stokes_limit(self):
        # At very low Re, Schiller-Naumann approaches Stokes: C_D ≈ 24/Re
        re = 0.1
        cd = schiller_naumann_cd(re)
        stokes_cd = 24.0 / re
        # Should be close but not exact (the correction term is small)
        assert cd == pytest.approx(stokes_cd * 1.015, rel=0.02)

    def test_re_100(self):
        # Standard reference point: Re = 100
        cd = schiller_naumann_cd(100.0)
        # Expected: (24/100) * (1 + 0.15 * 100^0.687)
        # 100^0.687 ≈ 23.44
        # C_D ≈ 0.24 * (1 + 0.15 * 23.44) ≈ 0.24 * 4.516 ≈ 1.084
        assert cd == pytest.approx(1.084, rel=0.01)

    def test_re_1000(self):
        # Upper limit of valid range
        cd = schiller_naumann_cd(1000.0)
        # Expected: (24/1000) * (1 + 0.15 * 1000^0.687)
        # 1000^0.687 ≈ 114.5
        # C_D ≈ 0.024 * (1 + 0.15 * 114.5) ≈ 0.024 * 18.175 ≈ 0.436
        assert cd == pytest.approx(0.436, rel=0.02)

    def test_monotonic_decrease(self):
        # C_D should decrease monotonically with Re in the valid range
        re_values = [0.1, 1.0, 10.0, 100.0, 1000.0]
        cd_values = [schiller_naumann_cd(re) for re in re_values]
        for i in range(len(cd_values) - 1):
            assert cd_values[i] > cd_values[i + 1]

    def test_valid_range(self):
        re_min, re_max = schiller_naumann_valid_range()
        assert re_min == 0.1
        assert re_max == 1000.0

    def test_invalid_reynolds(self):
        with pytest.raises(ValueError):
            schiller_naumann_cd(0)
        with pytest.raises(ValueError):
            schiller_naumann_cd(-1)


class TestCliftGraceWeber:
    def test_stokes_regime(self):
        # Re < 0.01: Stokes' law
        cd = clift_grace_weber_cd(0.001)
        assert cd == pytest.approx(24.0 / 0.001, rel=0.001)

    def test_continuity_at_boundaries(self):
        # Check continuity at regime boundaries (approximate)
        cd_20_low = clift_grace_weber_cd(19.999)
        cd_20_high = clift_grace_weber_cd(20.001)
        # Should be close (within 5%)
        assert abs(cd_20_low - cd_20_high) / cd_20_low < 0.06

        cd_260_low = clift_grace_weber_cd(259.999)
        cd_260_high = clift_grace_weber_cd(260.001)
        assert abs(cd_260_low - cd_260_high) / cd_260_low < 0.06

    def test_newton_regime(self):
        # Re > 1000: approximately constant
        cd = clift_grace_weber_cd(2000.0)
        assert cd == pytest.approx(0.44, rel=0.05)

    def test_invalid_reynolds(self):
        with pytest.raises(ValueError):
            clift_grace_weber_cd(0)


class TestDragForce:
    def test_basic(self):
        # Standard case: C_D = 0.44, U = 1 m/s, D = 0.01 m, rho = 1000 kg/m³
        # A = pi * 0.01² / 4 = 7.854e-5 m²
        # F_D = 0.5 * 1000 * 1² * 7.854e-5 * 0.44 = 0.01728 N
        fd = drag_force_from_cd(0.44, 1.0, 0.01, 1000.0)
        expected = 0.5 * 1000.0 * 1.0**2 * (math.pi * 0.01**2 / 4.0) * 0.44
        assert fd == pytest.approx(expected, rel=1e-10)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            drag_force_from_cd(-0.1, 1.0, 0.01, 1000.0)
        with pytest.raises(ValueError):
            drag_force_from_cd(0.44, -1.0, 0.01, 1000.0)
        with pytest.raises(ValueError):
            drag_force_from_cd(0.44, 1.0, 0, 1000.0)
        with pytest.raises(ValueError):
            drag_force_from_cd(0.44, 1.0, 0.01, 0)


class TestBenchmarkTable:
    def test_default_table(self):
        table = benchmark_table()
        assert len(table) == 9  # Default has 9 Re values
        for row in table:
            assert "reynolds" in row
            assert "cd_schiller_naumann" in row
            assert "cd_clift_grace_weber" in row
            assert row["cd_schiller_naumann"] > 0
            assert row["cd_clift_grace_weber"] > 0

    def test_custom_reynolds(self):
        table = benchmark_table([10.0, 100.0])
        assert len(table) == 2
        assert table[0]["reynolds"] == 10.0
        assert table[1]["reynolds"] == 100.0
