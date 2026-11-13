"""CASE-BEM-DEAN-DIFFRACTION (m17203): guard the case's acceptance criteria.

The case (``docs/dean_diffraction_case_m17203.md``) claims that TensorBEM's 2-D
deep-water diffraction route reproduces the Dean (1948) / Ursell (1950) /
Ogilvie (1963) reference within the mission tolerance of 5% for a submerged
circular cylinder (a = 1 m, f/a = 2, ka = 0.6, amplitude 1 m).  These checks pin
that claim to the code that produced it so a kernel or Kochin-integral change
cannot invalidate the reported table silently.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "m17203_dean_case", ROOT / "scripts" / "m17203_dean_case.py"
)
dean_case = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dean_case)

# acceptance thresholds declared before the case was run
TOLERANCE = 0.05          # mission tolerance on every force metric
TARGET = 0.01             # internal target at production resolution
ZERO_TOLERANCE = 1.0e-9   # Dean's theorem: |R| is exactly zero
HASKIND_TOLERANCE = 1.0e-3
PHASE_TOLERANCE_DEG = 1.0


@pytest.fixture(scope="module")
def case_row():
    return dean_case.one_run(panels=128, panel_nodes=24)


def test_force_matches_dean_ursell_reference_within_mission_tolerance(case_row):
    assert case_row["panel_force_x_relative_error"] < TOLERANCE
    assert case_row["panel_force_z_relative_error"] < TOLERANCE
    # production resolution already meets the internal target
    assert case_row["panel_force_x_relative_error"] < TARGET


def test_dean_zero_reflection_and_unit_transmission(case_row):
    assert case_row["panel_reflection_modulus"] < ZERO_TOLERANCE
    assert abs(case_row["panel_transmission_modulus"] - 1.0) < ZERO_TOLERANCE
    assert abs(case_row["panel_energy_balance_error"]) < 1.0e-9
    assert case_row["panel_transmission_phase_error_degrees"] < PHASE_TOLERANCE_DEG


def test_circular_polarisation_of_the_wave_load(case_row):
    # F_z = -i F_x in deep water: equal moduli, -90 deg phase difference
    assert abs(case_row["panel_phase_z_minus_x_degrees"] + 90.0) < 1.0e-6
    assert abs(
        abs(complex(*case_row["panel_force_z"])) / abs(complex(*case_row["panel_force_x"]))
        - 1.0
    ) < 1.0e-3


def test_haskind_reciprocity_reproduces_the_direct_diffraction_force(case_row):
    assert case_row["haskind_vs_direct_x"] < HASKIND_TOLERANCE
    assert case_row["haskind_vs_direct_z"] < HASKIND_TOLERANCE


def test_morison_inertia_limit_is_approached_from_above(case_row):
    # deeply submerged limit |F_x| -> 2*pi*rho*g*a*A*ka*exp(-k f) (C_M = 2)
    assert 1.0 < case_row["morison_over_exact"] < 1.3
    assert math.isfinite(case_row["panel_condition_number"])