"""Gates for the Wigley/KCS radiation-diffraction verification (m6656).

Every number used here is one that the report quotes through the generator
``python3 -m tensorbem.benchmarks.wigley_kcs_kochin_energy`` (artifact
``docs/m6656/study.json``).  The BEM solves are deliberately run on *coarse*
meshes: the gate proves that the physics identities hold, not that the
production-resolution numbers of the artifact are reproduced.

Three decisions are worth documenting:

* the panel-Kochin wave resistance is compared with the closed-form
  :func:`tensorbem.wigley.michell_wave_resistance` on the *same* quadrature, so
  the residual is a mesh-discretisation error and nothing else;
* the reciprocity gate is scored only on "significant" DOFs (exciting force
  above 1 % of the largest component of that case) because a head sea makes the
  antisymmetric forces vanish, and a relative error on round-off is meaningless;
* the KCS EFD comparison is asserted to be ``WITHHELD`` -- the measured Tokyo
  2003 table is not available offline, so the gate must fail if a future edit
  quietly turns it into a number.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks import wigley_kcs_kochin_energy as study
from tensorbem.free_surface_solver3d import far_field_damping_matrix_3d, solve_radiation_3d
from tensorbem.kcs_hull import (
    KCS_REFERENCE,
    kcs_class_hull,
    kcs_class_mesh,
    kcs_geometry_report,
    length_ratio_from_wavenumber,
    wave_number_from_length_ratio,
)
from tensorbem.symmetry3d import (
    far_field_damping_matrix_3d_symmetric,
    solve_radiation_3d_symmetric,
    split_patch,
)
from tensorbem.wigley import michell_kochin_function, michell_wave_resistance, wigley_hull_properties
from tensorbem.wigley_hydrostatics import wigley_wetted_hull_mesh

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = REPO_ROOT / "docs" / "m6656" / "study.json"

WIGLEY = (10.0, 1.0, 0.625)
GRAVITY = 9.80665
DENSITY = 1025.0


def _coarse_wigley(n_sections: int = 12, n_levels: int = 5):
    return wigley_wetted_hull_mesh(*WIGLEY, n_sections=n_sections, n_levels=n_levels)


# ---------------------------------------------------------------------------
# numeric bookkeeping
# ---------------------------------------------------------------------------

def test_fit_order_recovers_synthetic_exponent():
    h = [0.4, 0.2, 0.1]
    assert study.fit_order(h, [0.5 * x**2 for x in h]) == pytest.approx(2.0, abs=1e-6)
    assert study.fit_order(h, [3.0 * x for x in h]) == pytest.approx(1.0, abs=1e-6)


def test_fit_order_and_helpers_reject_bad_input():
    with pytest.raises(ValueError):
        study.fit_order([0.1], [1.0])
    with pytest.raises(ValueError):
        study.fit_order([0.1, 0.0], [1.0, 1.0])
    assert study.panel_size(4.0) == pytest.approx(0.5)
    assert study.relative(1.0, 0.0) == float("inf")


# ---------------------------------------------------------------------------
# geometry: Wigley closed form, KCS-class anchors
# ---------------------------------------------------------------------------

def test_wigley_mesh_is_a_valid_wetted_hull():
    mesh = _coarse_wigley()
    props = wigley_hull_properties(*WIGLEY)
    volume = float(np.sum(np.einsum("ij,ij->i", mesh.v0, np.cross(mesh.v1, mesh.v2))) / 6.0)
    assert mesh.panel_count == 4 * 12 * 5
    assert mesh.waterline_edges.shape[1] == 2
    assert volume > 0.0
    assert abs(volume / (WIGLEY[0] * WIGLEY[1] * WIGLEY[2]) - props["Cb"]) < 0.01


def test_kcs_class_hull_matches_kcs_principal_particulars():
    hull = kcs_class_hull()
    assert hull.reference_block_coefficient() == pytest.approx(
        KCS_REFERENCE["block_coefficient"], rel=0.01
    )
    assert hull.length == KCS_REFERENCE["length"]
    assert hull.breadth == KCS_REFERENCE["breadth"]
    assert hull.draft == KCS_REFERENCE["draft"]


def test_kcs_class_mesh_passes_geometry_gate_and_reports_waterplane_deficit():
    hull = kcs_class_hull()
    mesh = kcs_class_mesh(hull, n_sections=15, n_side=3, n_bottom=2)
    report = kcs_geometry_report(hull, mesh)
    assert report["mesh_length_relative_error"] < 1e-9
    assert report["mesh_breadth_relative_error"] < 1e-9
    assert report["mesh_draft_relative_error"] < 1e-9
    assert report["block_coefficient_vs_kcs"] < 0.01
    assert report["midship_coefficient_vs_kcs"] < 0.01
    # the family locks Cwp to Cp: the deficit must stay visible, never silent
    assert 0.10 < report["waterplane_coefficient_vs_kcs"] < 0.25


def test_wave_number_round_trip_and_validation():
    for ratio in (0.65, 1.0, 20.0):
        nu, omega = wave_number_from_length_ratio(ratio)
        assert length_ratio_from_wavenumber(nu) == pytest.approx(ratio, rel=1e-12)
        assert omega == pytest.approx(math.sqrt(GRAVITY * nu), rel=1e-12)
    with pytest.raises(ValueError):
        wave_number_from_length_ratio(0.0)
    with pytest.raises(ValueError):
        length_ratio_from_wavenumber(-1.0)


# ---------------------------------------------------------------------------
# criterion (a)/(d): panel Kochin vs Michell
# ---------------------------------------------------------------------------

def test_panel_kochin_converges_to_closed_form():
    fine = wigley_wetted_hull_mesh(*WIGLEY, n_sections=36, n_levels=14)
    k = 1.0
    h_panel = study.wigley_panel_kochin(fine, k)
    h_exact = michell_kochin_function(*WIGLEY, k)
    assert abs(h_panel - h_exact) / abs(h_exact) < 0.01


def test_wigley_cw_panel_vs_michell_within_five_percent():
    """Criterion (a): panel-path ``Cw`` vs Michell at ``Fn = 0.25``, ``N >= 2000``."""
    mesh = wigley_wetted_hull_mesh(*WIGLEY, n_sections=36, n_levels=14)
    assert mesh.panel_count >= 2000
    fn = 0.25
    U = fn * math.sqrt(GRAVITY * WIGLEY[0])
    panel = study.wigley_wave_resistance_panel(
        mesh, *WIGLEY, U, gravity=GRAVITY, density=DENSITY
    )
    michell = michell_wave_resistance(*WIGLEY, U, rho=DENSITY, g=GRAVITY)
    deviation = abs(panel - michell) / michell
    assert deviation < 0.05, f"panel Cw deviates {deviation:.3%} from Michell"


def test_wigley_cw_convergence_order_at_least_one():
    """Criterion (d): observed order in the panel size ``h`` is >= 1."""
    U = 0.25 * math.sqrt(GRAVITY * WIGLEY[0])
    errors, panels = [], []
    for n_sections, n_levels in ((18, 7), (36, 14)):
        mesh = wigley_wetted_hull_mesh(*WIGLEY, n_sections=n_sections, n_levels=n_levels)
        panel = study.wigley_wave_resistance_panel(mesh, *WIGLEY, U)
        michell = michell_wave_resistance(*WIGLEY, U)
        errors.append(abs(panel - michell) / michell)
        panels.append(mesh.panel_count)
    order = study.fit_order([study.panel_size(n) for n in panels], errors)
    assert order >= 1.0, f"observed order {order:.2f} < 1"


# ---------------------------------------------------------------------------
# criterion (c): Kochin far field vs near-field pressure integral
# ---------------------------------------------------------------------------

def test_wigley_kochin_energy_identity():
    """Criterion (c): ``B_33``/``B_55`` near field vs Kochin far field < 2 %."""
    patch = split_patch(_coarse_wigley())[0]
    omega = 3.0 * math.sqrt(GRAVITY / WIGLEY[0])
    nu = omega**2 / GRAVITY
    rad = solve_radiation_3d_symmetric(patch, nu, modes=(2, 4), gravity=GRAVITY)
    far = far_field_damping_matrix_3d_symmetric(patch, rad, gravity=GRAVITY, n_azimuth=72)
    near = np.diag(rad.damping)
    for near_value, far_value in zip(near, np.diag(far)):
        assert near_value > 0.0
        assert abs(far_value - near_value) / near_value < 0.02


def test_far_field_damping_matrix_is_symmetric_semidefinite():
    mesh = _coarse_wigley()
    omega = 2.0 * math.sqrt(GRAVITY / WIGLEY[0])
    nu = omega**2 / GRAVITY
    rad = solve_radiation_3d(mesh, nu, modes=(2, 4), gravity=GRAVITY)
    far = far_field_damping_matrix_3d(mesh, rad, gravity=GRAVITY, n_azimuth=48)
    scale = max(1.0, float(np.max(np.abs(far))))
    assert np.allclose(far, far.T, atol=1e-9 * scale)
    assert np.linalg.eigvalsh(0.5 * (far + far.T)).min() > -1e-8 * scale


# ---------------------------------------------------------------------------
# KCS-class hull: reciprocity and the Froude-Krylov limit
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def kcs_case():
    hull = kcs_class_hull()
    mesh = kcs_class_mesh(hull, n_sections=11, n_side=3, n_bottom=2)
    nu, _ = wave_number_from_length_ratio(20.0)
    return mesh, study.kcs_diffraction_case(mesh, nu)


def test_kcs_haskind_reciprocity_on_significant_dofs(kcs_case):
    mesh, case = kcs_case
    assert mesh.panel_count > 100
    significant = [i for i, keep in enumerate(case["significant_dof"]) if keep]
    assert significant, "no significant DOF in the long-wave case"
    for idx in significant:
        assert case["relative_error_pressure_vs_haskind"][idx] < 0.02
        assert case["relative_error_pressure_vs_kochin"][idx] < 0.05


def test_kcs_long_wave_froude_krylov_limit(kcs_case):
    _, case = kcs_case
    significant = [i for i, keep in enumerate(case["significant_dof"]) if keep]
    for idx in significant:
        assert case["relative_error_pressure_vs_fk"][idx] < 0.05


# ---------------------------------------------------------------------------
# the artefact itself
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not ARTIFACT.exists(), reason="study artifact not generated")
def test_study_artifact_is_self_consistent():
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    criteria = data["criteria"]
    for key in (
        "a_wigley_cw_vs_michell",
        "b_kcs_efd_excitation",
        "b_kcs_reciprocity",
        "c_kochin_energy_wigley",
        "c_kochin_energy_kcs",
        "d_panel_convergence_orders",
    ):
        assert key in criteria, f"missing criterion {key}"
    assert criteria["b_kcs_efd_excitation"].startswith("WITHHELD")
    for entry in criteria["a_wigley_cw_vs_michell"].values():
        assert entry["panel_count"] >= 2000
        assert entry["pass_5pct"] == (entry["cw_relative_error"] < 0.05)
    wigley_energy = criteria["c_kochin_energy_wigley"]
    assert wigley_energy["pass_2pct"] == (wigley_energy["max_damping_relative_error"] < 0.02)
    rows = data["wigley_radiation"]["rows"]
    assert rows
    worst = max(max(r["damping_relative_error"]) for r in rows)
    assert worst == pytest.approx(wigley_energy["max_damping_relative_error"])
