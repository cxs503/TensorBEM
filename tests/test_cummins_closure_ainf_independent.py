"""Regression pins for the independent-``A_inf`` judgement of the Wigley closure gates.

The evidence file ``benchmarks/verified/casepool_bem_cummins_wigley/ainf_independent_gates.json``
is produced by ``scripts/cummins_closure_ainf_independent.py`` (image solves, ~10 min,
so it is *not* regenerated here).  These tests assert the invariants that make it
interpretable; they read the frozen sweep, so they run in a second.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from tensorbem.added_mass_limit import added_mass_limit, shift_added_mass
from tensorbem.benchmarks import case_bem_cummins_wigley as anchor_case
from tensorbem.benchmarks import cummins_wigley_kk_causality as case
from tensorbem.causality import kkr_added_mass
from tensorbem.wigley3_freq import WIGLEY3_B, WIGLEY3_L, WIGLEY3_T, wigley3_wetted_hull_mesh

CASE_DIR = pathlib.Path("benchmarks/verified/casepool_bem_cummins_wigley")
SWEEP = CASE_DIR / "sweep_closure_medium800.npz"
EVIDENCE = CASE_DIR / "ainf_independent_gates.json"


def load_sweep() -> dict:
    with np.load(SWEEP) as data:
        return {
            "omega": data["omega"].tolist(),
            "added_mass": data["added_mass"].tolist(),
            "damping": data["damping"].tolist(),
            "rotation_center": data["rotation_center"].tolist(),
            "level_key": str(data["level_key"]),
            "level": str(data["level"]),
            "density": float(data["density"]),
            "gravity": float(data["gravity"]),
        }


def test_shift_added_mass_is_the_inverse_of_itself() -> None:
    rng = np.random.default_rng(20261116)
    matrix = rng.normal(size=(6, 6))
    matrix = matrix + matrix.T
    there = shift_added_mass(matrix, (0.0, 0.0, 0.0), (0.1, -0.2, -0.09375))
    back = shift_added_mass(there, (0.1, -0.2, -0.09375), (0.0, 0.0, 0.0))
    assert np.allclose(back, matrix, atol=1e-13)


def test_shift_added_mass_leaves_translations_invariant() -> None:
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(6, 6))
    matrix = matrix + matrix.T
    shifted = shift_added_mass(matrix, (0.0, 0.0, 0.0), (0.3, 0.0, -0.09))
    assert np.allclose(np.diag(shifted)[:3], np.diag(matrix)[:3], atol=1e-12)


def test_shift_added_mass_matches_direct_image_solve() -> None:
    """The re-reference must equal solving the image problem about the new centre.

    Coarse mesh (6x3 = 96 panels) so the direct solve is a couple of seconds; the
    full-mesh version of this check is in the frozen JSON's ``frame_check``.
    """
    mesh = wigley3_wetted_hull_mesh(WIGLEY3_L, WIGLEY3_B, WIGLEY3_T,
                                    n_sections=6, n_levels=3)
    about_waterline = np.asarray(
        added_mass_limit(mesh, "infinite_frequency", center=(0.0, 0.0, 0.0),
                         density=case.DENSITY).added_mass_matrix, dtype=float)
    about_centre = np.asarray(
        added_mass_limit(mesh, "infinite_frequency", center=case.ROTATION_CENTER,
                         density=case.DENSITY).added_mass_matrix, dtype=float)
    shifted = shift_added_mass(about_waterline, (0.0, 0.0, 0.0), case.ROTATION_CENTER)
    assert np.max(np.abs(shifted - about_centre)) / np.max(np.abs(about_centre)) < 1e-10
    # the shift is not a no-op for a moment mode
    assert abs(np.diag(shifted)[4] - np.diag(about_waterline)[4]) > 0.0


def test_analysis_gates_honours_added_mass_inf_override() -> None:
    sweep = load_sweep()
    tail = case.analysis_gates(sweep, 2)
    external = 1.10 * tail["added_mass_inf_tail_fit"]
    anchored = case.analysis_gates(sweep, 2, added_mass_inf=external)
    assert tail["added_mass_inf_source"] == "tail_fit"
    assert anchored["added_mass_inf_source"] == "external"
    assert anchored["added_mass_inf"] == pytest.approx(external)
    assert anchored["added_mass_inf_tail_fit"] == pytest.approx(tail["added_mass_inf_tail_fit"])
    # changing the constant moves both statistics (they are not A_inf-free)
    assert anchored["G3"]["rel_err"] != pytest.approx(tail["G3"]["rel_err"])
    assert anchored["G5"]["rel_err"] != pytest.approx(tail["G5"]["rel_err"])


def test_tail_fit_asymptote_is_below_the_band_top_for_heave() -> None:
    """A necessary (not sufficient) condition: heave ``A`` rises with ``omega``.

    The in-band profile is increasing and its slope decreases, so the true
    asymptote cannot be below the last solved value; the two-term tail fit
    returns 60.2 kg while ``A33(20 rad/s) = 62.5 kg``.  This is the witness that
    :func:`fit_added_mass_asymptote` is not a usable ``A_inf`` here -- and the
    reason the closure gates were judging the data with a biased constant.
    """
    sweep = load_sweep()
    omega, added, _ = case._as_diag(sweep, case.DOF_INDEX["heave"])
    fit = float(case.fit_added_mass_asymptote(np.asarray(omega), np.asarray(added))
                .added_mass_inf)
    assert added[-1] > added[-2] > added[-3]
    assert fit < added[-1]


@pytest.mark.skipif(not EVIDENCE.exists(), reason="frozen evidence not generated")
def test_frozen_evidence_is_internally_consistent() -> None:
    payload = json.loads(EVIDENCE.read_text())
    anchor = payload["a_inf_anchor"]
    rows = payload["evaluation"]["rows"]
    assert payload["sweep"]["n_points"] == 69
    assert [row["dof_label"] for row in rows] == list(case.DOF_LABELS)
    # the frame check is a claim about a 1e-10 identity, not a tolerance
    assert anchor["frame_check"]["max_relative_difference_direct_vs_transform"] < 1e-10
    # the ladder must bracket the value it is carried as
    for row in rows:
        low, high = row["a_inf_image_interval"]
        assert low - 1e-9 <= row["a_inf_image"] <= high + 1e-9
    # the baseline numbers must reproduce the committed closure run (m23130)
    heave = next(row for row in rows if row["dof_label"] == "heave")
    assert heave["G3_tail"] == pytest.approx(0.0746, abs=2e-3)
    assert heave["G5_tail"] == pytest.approx(0.0763, abs=2e-3)
    # the independent asymptote lies inside the KK-admissible window for heave
    window = heave["scan"]["G3_pass_window"]
    assert window, "heave G3 never passes on the scanned A_inf range"
    assert any(low <= heave["a_inf_image"] <= high for low, high in window)