"""Tests for CASE-BEM-CUMMINS-WIGLEY (6-DOF memory kernel, mission m17209).

Fast tests only: no 3-D BEM solve is performed here (the full case is run by
``scripts/m17209_cummins_wigley_kernel_6dof.py``).  What is tested is the
pipeline that turns a radiation sweep into the five gate numbers, and the
loading of the independent Capytaine reference.
"""
from __future__ import annotations

import math

import numpy as np

from tensorbem.benchmarks import cummins_wigley_kernel_6dof as case
from tensorbem.causality import kkr_added_mass

OMEGA = np.linspace(0.3, 8.0, 41)  # dense enough that the KK quadrature is honest


def _synthetic_sweep(a_scale: float = 1.0) -> dict:
    """Smooth damping sweep + added mass generated from it by Kramers-Kronig.

    ``B(w) = B0 w^3 / (w0^2 + w^2)^2`` vanishes at both ends, so the principal
    value integral in :func:`kkr_added_mass` converges; the resulting ``A`` is
    causal by construction.  ``a_scale`` lets a test break that consistency.
    """
    b0, w0 = 1.0e3, 2.0
    b = b0 * OMEGA ** 3 / (w0 ** 2 + OMEGA ** 2) ** 2
    a_inf = 10.0
    mid = 0.5 * (OMEGA[:-1] + OMEGA[1:])  # principal value needs a non-node sample
    a = np.interp(OMEGA, mid, [kkr_added_mass(OMEGA, b, float(w), added_mass_inf=a_inf)
                               for w in mid])
    a = a_inf + a_scale * (a - a_inf)
    added = np.zeros((len(OMEGA), 6, 6))
    damping = np.zeros((len(OMEGA), 6, 6))
    added[:, 2, 2] = a
    damping[:, 2, 2] = b
    return {
        "level": case.level_label(20, 10),
        "panels": 800,
        "omega": OMEGA.tolist(),
        "added_mass": added.tolist(),
        "damping": damping.tolist(),
        "relative_residual": [0.0] * len(OMEGA),
    }


def test_capytaine_reference_loads_three_levels_and_six_dof():
    ref = case.load_capytaine_reference()
    assert set(ref["levels"]) == {
        "n_sections=10,n_levels=5", "n_sections=20,n_levels=10", "n_sections=40,n_levels=20"
    }
    for level, entry in ref["levels"].items():
        assert set(entry["added_mass"]) == {"Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"}, level
        assert all(v > 0.0 for v in entry["added_mass"].values()), level
        assert all(v >= 0.0 for v in entry["damping"].values()), level


def test_kernel_identity_and_round_trip_on_a_kk_consistent_sweep():
    sweep = _synthetic_sweep()
    study = case.memory_kernel_study(sweep, dof=2, dt=0.01, tau_max=20.0,
                                     added_mass_inf=10.0)
    assert study["R2"]["K_at_zero_sampled"] > 0.0
    assert study["R2"]["rel_err"] < 0.02, study["R2"]
    assert study["R5"]["rel_err"] < 0.05, study["R5"]
    assert study["diagnostics"]["even_residual"] < 1e-6
    values = np.asarray(study["kernel"]["values"])
    assert abs(values[-1]) < 0.05 * abs(values[0]), "kernel must decay"


def test_kramers_kronig_gate_bites_on_inconsistent_added_mass():
    consistent = case.memory_kernel_study(_synthetic_sweep(1.0), dof=2, dt=0.01,
                                          tau_max=20.0, added_mass_inf=10.0)
    broken = case.memory_kernel_study(_synthetic_sweep(0.5), dof=2, dt=0.01,
                                      tau_max=20.0, added_mass_inf=10.0)
    assert consistent["R3"]["rel_err"] < 0.02, consistent["R3"]["rel_err"]
    assert broken["R3"]["rel_err"] > 0.10, broken["R3"]["rel_err"]


def test_gate_tolerances_match_the_case_specification():
    assert case.GATES["R1_cross_solver"]["tol"] == 0.10
    assert case.GATES["R3_kramers_kronig"]["tol"] == 0.10
    assert case.GATES["R4_free_decay"]["tol"] == 0.10
    # the two identities between numerical routes are held tighter than the case tol
    assert case.GATES["R2_kernel_at_zero"]["tol"] < 0.10
    assert case.GATES["R5_round_trip"]["tol"] < 0.10
    assert math.isclose(case.WIGLEY3_L / case.WIGLEY3_B, 10.0)
    assert math.isclose(case.WIGLEY3_B / case.WIGLEY3_T, 1.6)
    assert math.isclose(case.WIGLEY3_CB, 0.543, rel_tol=1e-3)