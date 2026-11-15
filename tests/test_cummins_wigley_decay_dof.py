"""C5 on every restoring mode of CASE-BEM-CUMMINS-WIGLEY (roll and pitch included).

Closes the debt recorded in ``cummins_wigley_causal_decay``: "heave is the only
DOF the case scores C5 on; roll/pitch would need C44/C55 restoring terms the case
module does not build".  The oscillator for the rotational modes is carried by
two independent routes (closed form vs panel hydrostatics for the stiffness,
closed form vs quadrature for the body inertia), so a C5 residual cannot be an
unchecked input any more.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks import case_bem_cummins_wigley as case
from tensorbem.hydrostatic import compute_hydrostatic_restoring_matrix
from tensorbem.wigley_hydrostatics import wigley_hydrostatic_reference, wigley_wetted_hull_mesh

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "benchmarks/verified/casepool_bem_cummins_wigley"
MEDIUM_SWEEP = ARTIFACT_DIR / "sweep_medium.npz"


def _quadrature_second_moments(nx: int = 800, nz: int = 400) -> dict[str, float]:
    """Body second moments about CENTER by quadrature of the parabolic hull."""
    L, B, T = case.LENGTH, case.BEAM, case.DRAFT
    rho = case.DENSITY
    x = np.linspace(-L / 2, L / 2, nx)
    z = np.linspace(-T, 0.0, nz)
    X, Z = np.meshgrid(x, z, indexing="ij")
    y = 0.5 * B * (1.0 - (2.0 * X / L) ** 2) * (1.0 - (Z / T) ** 2)
    dx, dz = x[1] - x[0], z[1] - z[0]
    return {
        "roll": rho * float(np.sum((2.0 / 3.0) * y**3 + 2.0 * y * Z**2) * dx * dz),
        "pitch": rho * float(np.sum(2.0 * y * X**2 + 2.0 * y * Z**2) * dx * dz),
        "volume": float(np.sum(2.0 * y) * dx * dz),
    }


def test_closed_form_second_moments_match_quadrature() -> None:
    """The declared body inertia is a closed form: check it against the hull."""
    moments = case._uniform_body_second_moments()
    quadrature = _quadrature_second_moments()
    assert quadrature["volume"] == pytest.approx(case.VOLUME_ANALYTIC, rel=1e-2)
    assert moments["roll"] == pytest.approx(quadrature["roll"], rel=1e-2)
    assert moments["pitch"] == pytest.approx(quadrature["pitch"], rel=1e-2)


def test_implied_radii_of_gyration_are_ship_like() -> None:
    """A declared inertia still has to be a plausible ship body, not any number."""
    moments = case._uniform_body_second_moments()
    mass = case.DENSITY * case.VOLUME_ANALYTIC
    k_xx = math.sqrt(moments["roll"] / mass) / case.BEAM
    k_yy = math.sqrt(moments["pitch"] / mass) / case.LENGTH
    assert 0.30 <= k_xx <= 0.40
    assert 0.20 <= k_yy <= 0.30


def test_restoring_matrix_routes_agree_and_converge() -> None:
    """Roll/pitch stiffness: closed form vs panel hydrostatics, on both meshes."""
    routes = case.restoring_matrix_routes()
    coarse = routes["meshes"]["coarse"]["relative_error_vs_closed_form"]
    medium = routes["meshes"]["medium"]["relative_error_vs_closed_form"]
    for mode in ("heave", "roll", "pitch"):
        assert medium[mode] < 1.0e-2, (mode, medium[mode])
        assert medium[mode] <= coarse[mode] + 1.0e-3, (mode, coarse[mode], medium[mode])
    # the case hull is symmetric: no heave-roll / heave-pitch / roll-pitch coupling
    for label in ("coarse", "medium"):
        coupling = routes["meshes"][label]["cross_coupling_max"]
        assert coupling < 1.0e-9 * routes["closed_form"]["heave"]


def test_rotational_stiffness_uses_the_metacentric_heights_the_right_way_round() -> None:
    """Guards the ``C44``/``C55`` key swap of ``wigley_hydrostatic_reference``."""
    reference = wigley_hydrostatic_reference(case.LENGTH, case.BEAM, case.DRAFT)
    roll = case.oscillator_properties(3)
    pitch = case.oscillator_properties(4)
    volume = case.DENSITY * case.GRAVITY * reference["volume"]
    assert roll["stiffness"] == pytest.approx(volume * reference["GM_T"], rel=1e-12)
    assert pitch["stiffness"] == pytest.approx(volume * reference["GM_L"], rel=1e-12)
    assert reference["GM_L"] > 50.0 * reference["GM_T"]  # roll is the stiff-in-appearance mode


def test_only_restoring_modes_can_start_a_free_decay() -> None:
    for dof in (0, 1, 5):
        with pytest.raises(ValueError, match="no hydrostatic stiffness"):
            case.oscillator_properties(dof)


def test_heave_oscillator_is_unchanged_by_the_extension() -> None:
    """The committed C5 number must not move when roll/pitch are added."""
    heave = case.oscillator_properties(2)
    legacy_restoring, legacy_mass, legacy_area = _legacy_heave_pair()
    assert heave["stiffness"] == pytest.approx(legacy_restoring, rel=1e-12)
    assert heave["inertia"] == pytest.approx(legacy_mass, rel=1e-12)
    assert legacy_area > 0.0


def _legacy_heave_pair() -> tuple[float, float, float]:
    from tensorbem.hydrostatic import compute_waterplane_properties

    n_sections, n_levels = case.MESHES["medium"]
    mesh = wigley_wetted_hull_mesh(
        case.LENGTH, case.BEAM, case.DRAFT, n_sections=n_sections, n_levels=n_levels
    )
    waterplane_area = float(compute_waterplane_properties(mesh)[0])
    return (
        case.DENSITY * case.GRAVITY * waterplane_area,
        case.DENSITY * case.VOLUME_ANALYTIC,
        waterplane_area,
    )


def test_mesh_hydrostatic_matrix_pairs_roll_with_Ixx() -> None:
    """Independent check that C[3,3] is the transverse (roll) stiffness."""
    n_sections, n_levels = case.MESHES["medium"]
    mesh = wigley_wetted_hull_mesh(
        case.LENGTH, case.BEAM, case.DRAFT, n_sections=n_sections, n_levels=n_levels
    )
    matrix = np.asarray(
        compute_hydrostatic_restoring_matrix(
            mesh,
            case.DENSITY * case.VOLUME_ANALYTIC,
            np.array([0.0, 0.0, -0.5 * case.DRAFT]),
        ).restoring_matrix
    )
    reference = wigley_hydrostatic_reference(case.LENGTH, case.BEAM, case.DRAFT)
    volume = case.DENSITY * case.GRAVITY * reference["volume"]
    assert matrix[3, 3] == pytest.approx(volume * reference["GM_T"], rel=2.0e-2)
    assert matrix[4, 4] == pytest.approx(volume * reference["GM_L"], rel=2.0e-2)


@pytest.mark.skipif(not MEDIUM_SWEEP.exists(), reason="committed sweep artifact not present")
def test_pitch_is_decided_within_the_declared_bar_and_roll_is_not_decidable() -> None:
    """The finding: pitch closes at 5 %, roll does not, heave is unchanged."""
    sweep = case.load_sweep(MEDIUM_SWEEP)
    for dof in (2, 3, 4):
        gate = case.free_decay_gate(sweep, dof=dof)
        values = gate["values"]
        assert values["status"] == "decided", (dof, values["status"])
        assert values["natural_frequency_in_band"] is True
        assert sweep.omega[0] <= values["natural_frequency_rad_s"] <= sweep.omega[-1]
        # the oscillator actually entered: inertia and stiffness are the new ones
        assert values["oscillator_inertia"] > 0.0
        assert values["restoring_n_per_m"] > 0.0
    heave = case.free_decay_gate(sweep, dof=2)["values"]["relative_error"]
    pitch = case.free_decay_gate(sweep, dof=4)["values"]["relative_error"]
    roll = case.free_decay_gate(sweep, dof=3)["values"]["relative_error"]
    assert heave == pytest.approx(0.038, abs=5.0e-3)   # committed C5, unchanged
    assert pitch < case.TOLERANCES["free_decay"]
    assert roll > case.TOLERANCES["free_decay"]
    assert roll == pytest.approx(0.239, abs=3.0e-2)


@pytest.mark.skipif(not MEDIUM_SWEEP.exists(), reason="committed sweep artifact not present")
def test_audit_grades_every_restoring_mode_and_flags_the_roll_residual() -> None:
    audit = case.dof_decay_audit(case.load_sweep(MEDIUM_SWEEP))
    assert set(audit["modes"]) == {"heave", "roll", "pitch"}
    assert audit["undecidable"] == []
    assert set(audit["within_bar"]) == {"heave", "pitch"}
    assert audit["beyond_bar"] == []
    assert audit["estimator_undecidable"] == ["roll"]
    roll = audit["modes"]["roll"]
    # the residual is *not* a stiffness/inertia error: both routes agree below 1 %
    medium = audit["stiffness_routes"]["meshes"]["medium"]["relative_error_vs_closed_form"]
    assert medium["roll"] < 0.01
    probes = roll["residual_probes"]
    # the residual is window-dominated: the same record fitted over a longer span
    # moves it by an order of magnitude, which is why it is not scored as a FAIL
    assert probes["ten_times_wider_window"] < 0.5 * probes["declared_window"]
    assert roll["residual_probe_spread"] > audit["tolerance"]
    assert roll["attribution"]["added_mass_effective_over_band_value"] > 1.05
    assert roll["attribution"]["added_mass_dispersion_across_band"] > 0.3


@pytest.mark.skipif(not MEDIUM_SWEEP.exists(), reason="committed sweep artifact not present")
def test_published_audit_artifact_is_reproducible_from_the_sweep() -> None:
    """The JSON in the artifact directory must match a fresh replay."""
    import json

    path = ARTIFACT_DIR / "decay_dof_audit.json"
    if not path.exists():
        pytest.skip("decay_dof_audit.json not published yet")
    published = json.loads(path.read_text())
    replay = case.dof_decay_audit(case.load_sweep(MEDIUM_SWEEP))
    for name in ("heave", "roll", "pitch"):
        assert replay["modes"][name]["grade"] == published["modes"][name]["grade"]
        assert replay["modes"][name]["values"]["relative_error"] == pytest.approx(
            published["modes"][name]["values"]["relative_error"], rel=1e-6, abs=1e-9
        )