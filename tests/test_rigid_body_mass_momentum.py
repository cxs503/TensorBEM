"""Route "P": the rigid-body mass matrix assembled from the momentum definitions.

Every other check in the repository compares ``M_RB`` with the closed block form
(Fossen 2011 eq. 3.44) or with an external code that implements the same block
convention.  Route "P" (:func:`tensorbem.rigid_body_mass.mass_matrix_from_momentum`)
instead builds the matrix from

    p   = m (v_O + w x r_g) ,
    h_O = I_g w + r_g x p ,

so the *signs* of the off-diagonal blocks and the *placement* of the
parallel-axis term are pinned by momentum conservation rather than by a second
transcription of the same equation.  These tests fix three properties:

1. agreement — route "P" equals the shipped assembly for a CG offset, for a
   shifted reference point, for a rotated body with a full inertia tensor and
   for the two-component body of the case;
2. discriminating power — the sign-flipped coupling convention (which is still a
   *symmetric* matrix, so a symmetry check passes) is rejected at O(1) relative
   error, and an antisymmetric corruption that the kinetic-energy identity is
   structurally blind to is rejected as well;
3. the case-pool anchor is machine-registered and its committed artefact
   reports PASS with the worst accepted entry inside the 3 % criterion.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.benchmarks import case_pool_registry as registry
from tensorbem.benchmarks.case_rigid_body_mass import case_components, compose_analytic
from tensorbem.benchmarks.rigid_body_mass_orientation import rotation_matrix
from tensorbem.rigid_body_mass import (
    energy_residual,
    mass_matrix_from_momentum,
    momentum_map,
    momentum_residual,
    rigid_body_mass_matrix,
    skew_matrix,
)

MASS = 2.4730617369e07
CG = np.array([5.0, 0.0, -3.0])
INERTIA_CG = np.diag([1.0287936826e09, 1.8122596408e10, 1.8518286286e10])


def _full_tensor() -> np.ndarray:
    """A ship-plausible full inertia tensor (products of inertia non-zero)."""
    tensor = np.array(INERTIA_CG, dtype=float)
    tensor[0, 1] = tensor[1, 0] = -5.171590e09
    tensor[0, 2] = tensor[2, 0] = 4.142154e09
    tensor[1, 2] = tensor[2, 1] = 1.351832e09
    return tensor


def test_route_p_matches_block_form_for_cg_offset():
    shipped = rigid_body_mass_matrix(MASS, CG, INERTIA_CG)
    route_p = mass_matrix_from_momentum(MASS, INERTIA_CG, CG)
    assert np.allclose(route_p, shipped, rtol=0.0, atol=1e-6)
    assert momentum_residual(shipped, MASS, INERTIA_CG, CG)["max_relative_error"] == 0.0


def test_route_p_matches_block_form_for_shifted_reference_point():
    point = (2.0, -1.0, 4.0)
    shipped = rigid_body_mass_matrix(MASS, CG, INERTIA_CG, reference_point=point)
    route_p = mass_matrix_from_momentum(MASS, INERTIA_CG, CG, reference_point=point)
    scale = np.max(np.abs(shipped))
    assert np.max(np.abs(route_p - shipped)) / scale < 1e-15


def test_route_p_matches_block_form_for_rotated_full_tensor():
    rotation = rotation_matrix(np.radians(20.0), np.radians(15.0), np.radians(25.0))
    tensor = _full_tensor()
    shipped = rigid_body_mass_matrix(MASS, rotation @ CG, rotation @ tensor @ rotation.T)
    route_p = mass_matrix_from_momentum(MASS, tensor, CG, rotation=rotation)
    scale = np.max(np.abs(shipped))
    assert np.max(np.abs(route_p - shipped)) / scale < 1e-15
    # the rotated configuration really is a full matrix: products of inertia
    # contribute a substantial fraction of the diagonal scale
    rotated = rotation @ tensor @ rotation.T
    assert np.max(np.abs(rotated - np.diag(np.diag(rotated)))) > 0.1 * np.max(np.abs(rotated))


def test_route_p_matches_block_form_for_two_component_body():
    composed = compose_analytic(case_components())
    shipped = rigid_body_mass_matrix(
        float(composed["mass"]),
        np.asarray(composed["center_of_gravity"], dtype=float),
        np.asarray(composed["inertia_about_cg"], dtype=float),
    )
    route_p = mass_matrix_from_momentum(
        float(composed["mass"]),
        np.asarray(composed["inertia_about_cg"], dtype=float),
        np.asarray(composed["center_of_gravity"], dtype=float),
    )
    assert np.max(np.abs(route_p - shipped)) / np.max(np.abs(shipped)) < 1e-15


def test_route_p_columns_are_generalized_momenta_of_unit_twists():
    """Definition-level check, independent of the closed block form."""
    route_p = mass_matrix_from_momentum(MASS, INERTIA_CG, CG)
    rng = np.random.default_rng(20261113)
    for _ in range(8):
        twist = rng.normal(size=6)
        momenta = momentum_map(MASS, INERTIA_CG, CG, twist[:3], twist[3:])
        expected = np.concatenate([momenta["linear"], momenta["angular"]])
        assert np.max(np.abs(route_p @ twist - expected)) / np.max(np.abs(expected)) < 1e-14


def test_flipped_coupling_is_detected_but_keeps_symmetry():
    """The pre-fix convention is symmetric, so only physics can reject it."""
    shipped = rigid_body_mass_matrix(MASS, CG, INERTIA_CG)
    s = skew_matrix(CG)
    flipped = shipped.copy()
    flipped[:3, 3:] = +MASS * s
    flipped[3:, :3] = -MASS * s
    assert np.allclose(flipped, flipped.T)  # a symmetry gate would accept it
    metric = momentum_residual(flipped, MASS, INERTIA_CG, CG)
    assert metric["max_relative_error"] == pytest.approx(2.0, abs=1e-9)
    assert metric["worst_entry"][0] == "surge" and metric["worst_entry"][1] == "pitch"


def test_energy_identity_is_blind_to_antisymmetric_corruption():
    """``0.5 x^T M x`` only sees sym(M); route "P" sees the whole matrix."""
    shipped = rigid_body_mass_matrix(MASS, CG, INERTIA_CG)
    corrupt = shipped.copy()
    corrupt[0, 1] = 1.0e6
    corrupt[1, 0] = -1.0e6
    assert energy_residual(corrupt, MASS, INERTIA_CG, CG) < 1e-12
    assert momentum_residual(corrupt, MASS, INERTIA_CG, CG)["max_relative_error"] > 0.1


def test_case_pool_anchor_is_registered_and_passes():
    entry = registry.resolve("CASE-BEM-RIGID-BODY-MASS")
    assert entry.pool_code == "BEM-RIGID-BODY-MASS"
    assert entry.missing() == []
    assert entry.entry.endswith(".py")
    payload = entry.artifact_payload()
    verdict = payload["verdict"]
    assert verdict["overall"] == "PASS"
    assert verdict["all_pass"] is True
    assert verdict["worst_accepted_entry_rel_error"] < verdict["tolerance"] <= 0.03
    assert payload["momentum_route"]["worst_relative_error"] < 1e-9
    assert payload["momentum_route"]["control_flipped_coupling"]["rejected"] is True
    # the registry's own verdict extractor must be able to read the artefact
    assert registry.artifact_verdict("CASE-BEM-RIGID-BODY-MASS") == "PASS"