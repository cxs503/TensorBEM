"""Verification of the Wigley hull hydrostatic mesh against closed forms.

The references are the elementary integrals of the parabolic Wigley surface
(``tensorbem.wigley_hydrostatics.wigley_hydrostatic_reference``); none of them
is produced by the hydrostatic solver under test.  Geometry is gated first
(displaced volume / ``Cb`` / principal dimensions), then volume, centre of
buoyancy, waterplane area and second moments, then the restoring matrix.
"""

import numpy as np
import pytest

from tensorbem.geometry3d import WettedHullMesh
from tensorbem.hydrostatic import (
    compute_displaced_volume_and_buoyancy,
    compute_hydrostatic_restoring_matrix,
    compute_waterplane_properties,
)
from tensorbem.wigley_hydrostatics import (
    wigley_hydrostatic_reference,
    wigley_wetted_area_reference,
    wigley_wetted_hull_mesh,
)

# Standard mathematical Wigley hull: L/B = 10, B/T = 1.6.
L, B, T = 10.0, 1.0, 0.625
RHO = 1025.0
G = 9.80665


@pytest.fixture(scope="module")
def coarse_mesh() -> WettedHullMesh:
    return wigley_wetted_hull_mesh(L, B, T, 16, 8)


@pytest.fixture(scope="module")
def fine_mesh() -> WettedHullMesh:
    return wigley_wetted_hull_mesh(L, B, T, 64, 32)


@pytest.fixture(scope="module")
def reference() -> dict:
    return wigley_hydrostatic_reference(L, B, T, density=RHO, gravity=G)


class TestMeshGeneration:
    """The mesh itself: validity, topology, and panel quality."""

    def test_is_wetted_hull_mesh(self, coarse_mesh: WettedHullMesh) -> None:
        assert isinstance(coarse_mesh, WettedHullMesh)
        assert np.all(coarse_mesh.vertices[:, 2] <= 1.0e-12)

    def test_single_closed_waterline_loop(self, coarse_mesh: WettedHullMesh) -> None:
        edges = coarse_mesh.waterline_edges
        assert edges.shape[0] > 0
        out = edges[:, 0]
        into = edges[:, 1]
        # every waterline vertex has one outgoing and one incoming edge
        u_out, c_out = np.unique(out, return_counts=True)
        u_in, c_in = np.unique(into, return_counts=True)
        assert np.array_equal(u_out, u_in)
        assert np.all(c_out == 1) and np.all(c_in == 1)

    def test_no_degenerate_panels(self, coarse_mesh: WettedHullMesh) -> None:
        assert np.all(coarse_mesh.areas > 0.0)
        # no panel may be flat inside the zero-width seam y = 0
        v0 = coarse_mesh.v0
        v1 = coarse_mesh.v1
        v2 = coarse_mesh.v2
        seam = (
            (np.abs(v0[:, 1]) < 1e-12)
            & (np.abs(v1[:, 1]) < 1e-12)
            & (np.abs(v2[:, 1]) < 1e-12)
        )
        assert not np.any(seam)

    def test_posts_and_keel_are_grid_lines(self, coarse_mesh: WettedHullMesh) -> None:
        """The hull has zero width at x = +-L/2 and at the keel; both are seams."""
        v = coarse_mesh.vertices
        post = np.isclose(np.abs(v[:, 0]), 0.5 * L, atol=1e-12)
        assert np.all(np.abs(v[post, 1]) < 1e-12)
        keel = np.isclose(v[:, 2], -T, atol=1e-12)
        assert np.all(np.abs(v[keel, 1]) < 1e-12)
        assert keel.any() and post.any()

    def test_rejects_bad_arguments(self) -> None:
        with pytest.raises(ValueError):
            wigley_wetted_hull_mesh(0.0, B, T)
        with pytest.raises(ValueError):
            wigley_wetted_hull_mesh(L, B, T, 1, 4)


class TestClosedFormReference:
    """The closed forms themselves, against the analytic hull properties."""

    def test_volume_and_coefficients(self, reference: dict) -> None:
        assert reference["volume"] == pytest.approx(4.0 * L * B * T / 9.0, rel=0)
        assert reference["Cb"] == pytest.approx(4.0 / 9.0, rel=1e-15)
        assert reference["Cp"] * reference["Cm"] == pytest.approx(
            reference["Cb"], rel=1e-15
        )

    def test_metacentric_radii(self, reference: dict) -> None:
        volume = 4.0 * L * B * T / 9.0
        assert reference["BM_T"] == pytest.approx(3.0 * B**2 / (35.0 * T), rel=1e-15)
        assert reference["BM_L"] == pytest.approx(3.0 * L**2 / (40.0 * T), rel=1e-15)
        assert reference["BM_T"] == pytest.approx(reference["I_xx"] / volume, rel=1e-15)
        assert reference["BM_L"] == pytest.approx(reference["I_yy"] / volume, rel=1e-15)


class TestMeshAgainstClosedForms:
    """Mesh-derived hydrostatics against the closed forms (the actual audit)."""

    def test_volume_and_cb_geometry_gate(
        self, coarse_mesh: WettedHullMesh, reference: dict
    ) -> None:
        """Geometry gate: displaced volume -> Cb must be right before anything else."""
        volume, centre = compute_displaced_volume_and_buoyancy(coarse_mesh)
        cb = volume / (L * B * T)
        assert cb == pytest.approx(reference["Cb"], rel=5.0e-2)
        assert volume == pytest.approx(reference["volume"], rel=5.0e-2)
        # LCB on the centreplane by symmetry, and fore-aft centred
        assert abs(centre[0]) < 1.0e-12
        assert abs(centre[1]) < 1.0e-12

    def test_centre_of_buoyancy_height(
        self, fine_mesh: WettedHullMesh, reference: dict
    ) -> None:
        """z_B = -3T/8 for the parabolic vertical area distribution."""
        _, centre = compute_displaced_volume_and_buoyancy(fine_mesh)
        assert centre[2] == pytest.approx(-3.0 * T / 8.0, rel=5.0e-4)
        assert centre[2] == pytest.approx(
            reference["center_of_buoyancy"][2], rel=5.0e-4
        )

    def test_waterplane_properties(
        self, fine_mesh: WettedHullMesh, reference: dict
    ) -> None:
        area, centre, moments = compute_waterplane_properties(fine_mesh)
        assert area == pytest.approx(2.0 * L * B / 3.0, rel=1.0e-3)
        assert abs(centre[0]) < 1.0e-3
        assert abs(centre[1]) < 1.0e-3
        assert moments[0] == pytest.approx(4.0 * B**3 * L / 105.0, rel=1.0e-3)
        assert moments[1] == pytest.approx(B * L**3 / 30.0, rel=1.0e-3)
        assert abs(moments[2]) < 1.0e-3 * moments[1]

    def test_wetted_area(self, coarse_mesh: WettedHullMesh) -> None:
        reference_area = wigley_wetted_area_reference(L, B, T)
        assert coarse_mesh.surface_area == pytest.approx(reference_area, rel=5.0e-3)

    def test_second_order_convergence(self) -> None:
        """Every hydrostatic quantity must converge at second order in h."""
        reference = wigley_hydrostatic_reference(L, B, T)
        levels = [(8, 4), (16, 8), (32, 16), (64, 32)]
        errors = {"volume": [], "z_B": [], "A_wp": [], "I_xx": [], "I_yy": []}
        for ns, nt in levels:
            mesh = wigley_wetted_hull_mesh(L, B, T, ns, nt)
            volume, centre = compute_displaced_volume_and_buoyancy(mesh)
            area, _, moments = compute_waterplane_properties(mesh)
            errors["volume"].append(abs(volume - reference["volume"]) / reference["volume"])
            errors["z_B"].append(
                abs(centre[2] - reference["center_of_buoyancy"][2])
                / abs(reference["center_of_buoyancy"][2])
            )
            errors["A_wp"].append(abs(area - reference["waterplane_area"]) / reference["waterplane_area"])
            errors["I_xx"].append(abs(moments[0] - reference["I_xx"]) / reference["I_xx"])
            errors["I_yy"].append(abs(moments[1] - reference["I_yy"]) / reference["I_yy"])

        for name, series in errors.items():
            # monotone decrease, and roughly x4 per refinement (order ~2)
            assert all(b < a for a, b in zip(series, series[1:])), name
            observed = [
                np.log2(a / b) for a, b in zip(series, series[1:])
            ]
            assert min(observed) > 1.5, (name, observed)
            assert series[-1] < 1.0e-3, (name, series)

    def test_restoring_matrix(self, fine_mesh: WettedHullMesh, reference: dict) -> None:
        """C33 = rho g A_wp, C55 = rho g V GM_T, C44 = rho g V GM_L."""
        mass = RHO * reference["volume"]
        kg_z = -T + 0.5 * T  # centre of gravity at mid-draft, above the keel
        props = compute_hydrostatic_restoring_matrix(
            fine_mesh, mass, np.array([0.0, 0.0, kg_z]), density=RHO, gravity=G
        )
        c = props.restoring_matrix
        assert c.shape == (6, 6)
        assert c[2, 2] == pytest.approx(reference["C33"], rel=1.0e-3)
        # roll (index 3, 1-based C44) uses I_xx; pitch (index 4, 1-based C55)
        # uses I_yy.  The reference keys are named with the 1-based convention.
        assert c[3, 3] == pytest.approx(reference["C55"], rel=2.0e-3)
        assert c[4, 4] == pytest.approx(reference["C44"], rel=2.0e-3)
        assert c[5, 5] == 0.0  # yaw: no hydrostatic restoring
        # symmetry of the waterplane: no coupling into/out of the other modes
        off = c.copy()
        for k in (2, 3, 4):
            off[k, k] = 0.0
        assert np.max(np.abs(off)) < 1.0e-6 * c[2, 2]
        # this hull form is stable: GM_T > 0 with KG at mid-draft
        assert reference["GM_T"] > 0.0
        assert props.displaced_volume == pytest.approx(reference["volume"], rel=5.0e-4)