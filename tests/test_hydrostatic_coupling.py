"""Regression tests for the hydrostatic restoring-matrix coupling terms.

The previous implementation returned a diagonal heave/roll/pitch block, asserting
that "all off-diagonal terms are zero" under small-angle linearisation. That is
only true in the frame whose origin is the waterplane centroid. These tests pin
the full small-angle block against closed-form hydrostatics of wall-sided bodies
and against the hydrostatic energy identity, so neither the coupling entries nor
the origin-frame second moments can silently regress.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.benchmarks.hydrostatic_coupling import (
    DENSITY,
    GRAVITY,
    KG,
    Box,
    build_payload,
    multi_box_hull_mesh,
    multi_box_reference,
)
from tensorbem.hydrostatic import compute_hydrostatic_restoring_matrix

TWIN = (
    Box(4.0, 2.0, 1.0, x_center=2.0, y_center=1.5),
    Box(2.5, 1.2, 1.0, x_center=-1.2, y_center=-2.0),
)
OFFSET_BOX = (Box(6.0, 3.0, 1.2, x_center=1.8),)
CENTRED_BOX = (Box(6.0, 3.0, 1.2),)
CATAMARAN = (Box(4.0, 1.0, 1.0, y_center=1.6), Box(4.0, 1.0, 1.0, y_center=-1.6))


def block(boxes, nx: int = 1, ny: int = 1) -> np.ndarray:
    """Heave/roll/pitch block of the restoring matrix of a box complex."""
    reference = multi_box_reference(boxes, DENSITY, GRAVITY, KG)
    hull = multi_box_hull_mesh(boxes, nx, ny)
    props = compute_hydrostatic_restoring_matrix(
        hull,
        DENSITY * reference["displaced_volume"],
        np.array([reference["x_b"], reference["y_b"], reference["z_g"]]),
        DENSITY,
        GRAVITY,
    )
    return np.asarray(props.restoring_matrix, dtype=float)[2:5, 2:5]


@pytest.fixture(scope="module")
def payload() -> dict:
    return build_payload()


class TestClosedForm:
    """Wall-sided bodies have exact hydrostatics, so there is no tolerance."""

    @pytest.mark.parametrize(
        "boxes",
        [CENTRED_BOX, OFFSET_BOX, TWIN, CATAMARAN],
        ids=["centred-box", "offset-box", "asymmetric-twin", "catamaran"],
    )
    def test_entries_match_closed_form(self, boxes):
        reference = multi_box_reference(boxes, DENSITY, GRAVITY, KG)
        matrix = block(boxes, 2, 2)
        scale = max(abs(reference[name]) for name in
                    ("C33", "C34", "C35", "C44", "C45", "C55"))
        computed = {
            "C33": matrix[0, 0], "C34": matrix[0, 1], "C35": matrix[0, 2],
            "C44": matrix[1, 1], "C45": matrix[1, 2], "C55": matrix[2, 2],
        }
        for name, value in computed.items():
            assert value == pytest.approx(reference[name], abs=1.0e-12 * scale), name

    def test_off_diagonal_entries_are_not_dropped(self):
        """The coupling is first order: it must be present and large enough to matter."""
        reference = multi_box_reference(OFFSET_BOX, DENSITY, GRAVITY, KG)
        matrix = block(OFFSET_BOX)
        assert matrix[0, 2] != 0.0
        assert matrix[0, 2] == pytest.approx(-DENSITY * GRAVITY
                                             * reference["waterplane_area"]
                                             * reference["x_f"], rel=1.0e-12)
        assert matrix[2, 0] == matrix[0, 2]
        # Coupling strength of a 1.8 m LCF offset on a 6 m barge.
        assert abs(matrix[0, 2]) / np.sqrt(matrix[0, 0] * matrix[2, 2]) > 0.5

    def test_twin_complex_has_all_three_couplings(self):
        matrix = block(TWIN)
        reference = multi_box_reference(TWIN, DENSITY, GRAVITY, KG)
        assert reference["x_f"] != 0.0 and reference["y_f"] != 0.0
        assert matrix[0, 1] != 0.0   # waterplane centroid off the centreline
        assert matrix[0, 2] != 0.0   # LCF not at the pitch axis
        assert matrix[1, 2] != 0.0   # product of inertia

    def test_symmetric_bodies_keep_a_diagonal_block(self):
        """Backward compatibility: no coupling without waterplane asymmetry."""
        for boxes in (CENTRED_BOX, CATAMARAN):
            matrix = block(boxes)
            assert matrix[0, 1] == pytest.approx(0.0, abs=1.0e-9)
            assert matrix[0, 2] == pytest.approx(0.0, abs=1.0e-9)
            assert matrix[1, 2] == pytest.approx(0.0, abs=1.0e-9)

    def test_block_is_symmetric(self):
        for boxes in (CENTRED_BOX, OFFSET_BOX, TWIN, CATAMARAN):
            matrix = block(boxes, 2, 2)
            assert np.max(np.abs(matrix - matrix.T)) <= 1.0e-9 * np.max(np.abs(matrix))

    def test_stable_body_has_a_positive_definite_block(self):
        """A catamaran is stable (GM_T = 1.74 m, no waterplane asymmetry).

        The block must then be positive definite. The barge cases carry
        KG = 1.4 m on a 1.2 m draft, i.e. GM_T = -0.175 m: those bodies are
        genuinely unstable in roll, so a negative eigenvalue there is physics,
        not an implementation defect -- which is why the consistency of the
        block is checked against the closed form in
        ``test_entries_match_closed_form`` instead of against a definiteness
        assumption.
        """
        reference = multi_box_reference(CATAMARAN, DENSITY, GRAVITY, KG)
        assert reference["gm_transverse"] > 0.0
        eigenvalues = np.linalg.eigvalsh(block(CATAMARAN, 2, 2))
        assert np.all(eigenvalues > 0.0), eigenvalues


class TestEnergyAndFrame:
    def test_energy_identity(self, payload):
        energy = payload["energy_identity"]
        assert energy["relative_difference"] < 1.0e-12
        assert energy["energy_from_matrix"] > 0.0

    def test_centroid_frame_recovers_the_classical_diagonal(self, payload):
        centroid = payload["frame_covariance"][-1]
        assert centroid["centroid_frame"] is True
        assert abs(centroid["C34"]) < 1.0e-6
        assert abs(centroid["C35"]) < 1.0e-6
        # C44 / (rho g grad) must equal GM_T = I_T^centroid/grad + (z_B - z_G).
        assert centroid["C44_over_rho_g_grad"] == pytest.approx(
            centroid["GM_transverse"], rel=1.0e-9
        )
        assert centroid["C55_over_rho_g_grad"] == pytest.approx(
            centroid["GM_longitudinal"], rel=1.0e-9
        )

    def test_frame_covariance_under_origin_shift(self, payload):
        rows = [row for row in payload["frame_covariance"] if "C35_after" in row]
        assert len(rows) >= 2
        for row in rows:
            assert row["max_relative_difference"] < 1.0e-12
            assert row["C35_after"] == pytest.approx(row["C35_predicted"], rel=1.0e-9,
                                                     abs=1.0e-6)


class TestCurvedWaterplane:
    def test_annulus_converges_at_second_order(self, payload):
        orders = payload["annulus"]["observed_order"]
        for name in ("C33", "C44", "C55"):
            assert min(orders[name]) > 1.8, (name, orders[name])
        assert payload["annulus"]["finest_relative_error"] < 5.0e-3

    def test_annulus_is_symmetric(self, payload):
        for level in payload["annulus"]["levels"]:
            assert level["symmetry_break"] < 1.0e-12


class TestDocumentedNumbers:
    """The numbers in ``docs/hydrostatic_coupling.md`` come from this payload."""

    def test_wall_sided_errors_are_at_round_off(self, payload):
        for case in payload["cases"]:
            for level in case["levels"]:
                assert level["max_relative_error"] < 1.0e-10, (case["name"], level)

    def test_coupling_matters_for_the_static_response(self, payload):
        for row in payload["response"]:
            # The diagonal-only matrix predicts no pitch from a pure heave force.
            assert row["static"]["unit_heave_force"]["response_diagonal_only"][2] == 0.0
            assert row["static"]["unit_heave_force"]["response_coupled"][2] != 0.0
            assert row["coupling_coefficients"]["C35"] > 0.5