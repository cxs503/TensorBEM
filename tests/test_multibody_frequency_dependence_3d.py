"""Tests for the 3-D multibody frequency-dependence machinery.

The load-bearing checks are:

* the Wigley hull generator produces a valid :class:`WettedHullMesh` whose
  displaced volume and waterplane area converge to the analytic values -- no
  hydrodynamics is believed before the geometry passes;
* the mirror closure doubles the panel count and the volume, and its first
  ``n`` faces are the original wetted panels in order (the index contract the
  asymptote solver relies on);
* the pressure-release added mass of a floating hemisphere equals half the
  analytic added mass of the full sphere, because the odd continuation of the
  hemisphere *is* the sphere's rigid heave mode -- an exact reference value;
* the low-frequency Kelvin solve approaches the rigid-wall potential solution of
  the same hemisphere (two independent solvers agreeing on one limit);
* the asymptotic-fit helpers recover known parameters from synthetic data.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.benchmarks.multibody_frequency_dependence_3d import (
    adaptive_mesh,
    fit_asymptote,
    fit_scale_law,
    panels_per_wavelength,
    richardson_gci,
)
from tensorbem.geometry3d import TriangleMesh, WettedHullMesh, floating_sphere_hull_mesh
from tensorbem.multibody3d import (
    HEAVE,
    combined_hull_mesh,
    generalized_normals,
    mesh_displacement,
    mirror_closure,
    radiation_row,
    side_by_side_multihull,
    static_added_mass,
    translated_hull,
    waterplane_area,
    wigley_hull_mesh,
)
from tensorbem.wigley3_hull import (
    WIGLEY3_B,
    WIGLEY3_L,
    wigley3_block_coefficient,
    wigley3_displaced_volume,
)

DENSITY = 1025.0
G = 9.80665


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_wigley_mesh_is_valid_wetted_hull() -> None:
    mesh = wigley_hull_mesh(nx=16, nz=8)
    assert isinstance(mesh, WettedHullMesh)
    assert mesh.faces.shape[0] == 4 * 8 * 15  # (nx - 1) stations x 2*nz ring panels
    assert mesh.vertices[:, 2].max() == pytest.approx(0.0)
    assert mesh.vertices[:, 2].min() == pytest.approx(-0.1875)
    assert mesh.waterline_edges.shape[0] > 0
    assert np.all(mesh.vertices[mesh.waterline_edges, 2] == 0.0)


def test_wigley_geometry_converges_to_analytic_values() -> None:
    analytic_volume = wigley3_displaced_volume()
    from tensorbem.wigley3_hull import WIGLEY3_ALPHA
    # longitudinal fullness: A_wp = B L (2/3 + 2 alpha/15) (m16935)
    analytic_waterplane = WIGLEY3_B * WIGLEY3_L * (
        2.0 / 3.0 + 2.0 * WIGLEY3_ALPHA / 15.0
    )
    errors = []
    waterplane_errors = []
    for nx in (12, 24, 40):
        mesh = wigley_hull_mesh(nx=nx, nz=8)
        volume = mesh_displacement(mesh)
        errors.append(abs(volume - analytic_volume) / analytic_volume)
        waterplane_errors.append(
            abs(waterplane_area(mesh) - analytic_waterplane) / analytic_waterplane
        )
    assert errors[-1] < 0.01
    assert errors[-1] < errors[0]
    assert waterplane_errors[-1] < 2.0e-3
    assert waterplane_errors[-1] < waterplane_errors[0]
    assert 0.5 < wigley3_block_coefficient() < 0.6


def test_side_by_side_layout_separates_hulls_by_the_beam_gap() -> None:
    single = wigley_hull_mesh(nx=12, nz=6)
    pair = side_by_side_multihull(n_hulls=2, gap_over_beam=0.5, nx=12, nz=6)
    assert pair.faces.shape[0] == 2 * single.faces.shape[0]
    assert mesh_displacement(pair) == pytest.approx(2.0 * mesh_displacement(single), rel=1e-12)
    # the two hull sides must leave exactly the requested clear gap between them
    half_beam = float(np.max(np.abs(single.vertices[:, 1])))
    centre = 0.5 * WIGLEY3_B * (1.0 + 0.5)  # half the centre-to-centre pitch
    ys = np.abs(pair.vertices[:, 1])
    assert ys.max() == pytest.approx(centre + half_beam, rel=1e-9)
    # nothing lives strictly inside the gap band
    assert not np.any((ys > 1.0e-12) & (ys < centre - half_beam - 1.0e-12))
    # moving the reference gap moves the hulls
    wider = side_by_side_multihull(n_hulls=2, gap_over_beam=1.0, nx=12, nz=6)
    assert np.max(np.abs(wider.vertices[:, 1])) > ys.max()


def test_translated_hull_keeps_waterline_on_the_plane() -> None:
    shifted = translated_hull(wigley_hull_mesh(nx=12, nz=6), dy=1.25)
    assert shifted.vertices[:, 1].min() > 0.0
    assert shifted.vertices[:, 2].max() == pytest.approx(0.0)
    with pytest.raises(ValueError):
        translated_hull(wigley_hull_mesh(nx=12, nz=6), dx=float("nan"))


def test_mirror_closure_is_watertight_and_doubles_everything() -> None:
    mesh = wigley_hull_mesh(nx=12, nz=6)
    closure = mirror_closure(mesh)
    assert isinstance(closure, TriangleMesh)
    assert closure.panel_count == 2 * mesh.panel_count
    upper = mesh.panel_count
    # the first `upper` faces are the wetted hull panels, in the original order
    assert np.allclose(
        closure.vertices[closure.faces[:upper]], mesh.vertices[mesh.faces]
    )
    volume = float(
        np.sum(np.einsum("ij,ij->i", closure.centroids, closure.normals) * closure.areas) / 3.0
    )
    assert volume == pytest.approx(2.0 * mesh_displacement(mesh), rel=1e-9)


def test_generalized_normals_are_orthogonal_to_their_mode_axis() -> None:
    mesh = wigley_hull_mesh(nx=12, nz=6)
    normals = generalized_normals(mesh, modes=(1, 2, 4))
    assert normals.shape == (mesh.panel_count, 3)
    # heave mode carries no horizontal component
    assert np.allclose(normals[:, 1], mesh.normals[:, 2])
    # every generalized normal is bounded by the rigid-body lever arm
    lever = np.linalg.norm(mesh.centroids, axis=1) + 1.0
    assert np.all(np.abs(normals) <= lever[:, None])


# ---------------------------------------------------------------------------
# asymptotic potential solves
# ---------------------------------------------------------------------------


def test_pressure_release_hemisphere_matches_half_the_sphere_added_mass() -> None:
    """Odd continuation of a hemisphere = the sphere's rigid heave mode.

    The added mass of a sphere in an unbounded fluid is half the displaced mass,
    ``A33 = (1/2)(4/3) pi rho a**3``, and the hemisphere owns one half of the
    fluid, so its pressure-release value must be ``(1/3) pi rho a**3``.  The
    tolerance is set by the icosphere: the library's own
    :func:`tensorbem.solver3d.compute_added_mass_matrix_3d` lands 2.0 % below the
    analytic sphere value at this subdivision, and this solve reproduces that
    same discrete value to 0.05 %.
    """
    radius = 1.0
    # a cap of the sphere centred on the waterplane: centre z = -1e-6 keeps the
    # wetted surface a true hemisphere while satisfying the strict z0 < 0 rule
    hemisphere = floating_sphere_hull_mesh(
        subdivisions=2, radius=radius, center=(0.0, 0.0, -1.0e-6)
    )
    result = static_added_mass(
        hemisphere, modes=(HEAVE,), continuation="pressure_release", density=DENSITY
    )
    analytic = (1.0 / 3.0) * math.pi * DENSITY * radius**3
    assert result.parity_error < 5.0e-3
    assert result.closure_volume == pytest.approx(2.0 * result.mesh_displacement, rel=1e-9)
    assert result.added_mass[0, 0] == pytest.approx(analytic, rel=3.0e-2)


def test_rigid_wall_added_mass_exceeds_pressure_release_for_a_floating_hull() -> None:
    mesh = wigley_hull_mesh(nx=12, nz=6)
    rigid = static_added_mass(mesh, modes=(1, 2, 4), continuation="rigid_wall")
    released = static_added_mass(mesh, modes=(1, 2, 4), continuation="pressure_release")
    assert rigid.added_mass[1, 1] > released.added_mass[1, 1] > 0.0
    # the continuation parity is enforced to discretisation accuracy, not to
    # machine precision: the mirrored collocation rows differ by the quadrature
    # error on near-singular panel pairs
    assert rigid.parity_error < 5.0e-2
    assert released.parity_error < 5.0e-2
    assert rigid.condition_number < 1.0e4
    assert rigid.relative_residual < 1.0e-10


def test_low_frequency_radiation_approaches_the_rigid_wall_limit() -> None:
    """Two independent solvers must agree on the ``omega -> 0`` limit."""
    hemisphere = floating_sphere_hull_mesh(subdivisions=3, radius=1.0)
    reference = static_added_mass(
        hemisphere, modes=(HEAVE,), continuation="rigid_wall", density=DENSITY
    ).added_mass[0, 0]
    row = radiation_row(hemisphere, 0.02, label="hemisphere", modes=(HEAVE,), density=DENSITY)
    assert row.added_mass[0, 0] == pytest.approx(reference, rel=0.05)
    # no waves at zero frequency: the damping must be negligible
    assert row.damping[0, 0] < 0.05 * reference * math.sqrt(G * 0.02)


def test_static_added_mass_rejects_unknown_continuation() -> None:
    mesh = wigley_hull_mesh(nx=12, nz=6)
    with pytest.raises(ValueError):
        static_added_mass(mesh, continuation="dirichlet")


def test_combined_hull_mesh_keeps_both_waterline_loops() -> None:
    left = translated_hull(wigley_hull_mesh(nx=12, nz=6), dy=-0.45)
    right = translated_hull(wigley_hull_mesh(nx=12, nz=6), dy=0.45)
    combined = combined_hull_mesh([left, right])
    loops = combined.waterline_edges.shape[0]
    assert loops == 2 * left.waterline_edges.shape[0]


# ---------------------------------------------------------------------------
# analysis helpers
# ---------------------------------------------------------------------------


def test_adaptive_mesh_resolves_short_waves_better() -> None:
    slow = adaptive_mesh(2.0, cap=40)
    fast = adaptive_mesh(8.0, cap=40)
    assert fast["nx"] > slow["nx"]
    assert panels_per_wavelength(fast["nx"], 8.0) >= 7.0
    assert panels_per_wavelength(slow["nx"], 2.0) >= 7.0
    capped = adaptive_mesh(60.0, cap=20)
    assert capped["nx"] == 20
    assert panels_per_wavelength(capped["nx"], 60.0) < 7.0


def test_fit_asymptote_recovers_a_synthetic_limit() -> None:
    omegas = [4.0, 5.0, 6.5, 8.0]
    expected = 61.0
    coefficient = -1500.0
    values = [expected + coefficient / omega**2 for omega in omegas]
    fit = fit_asymptote(omegas, values)
    assert fit["a_inf"] == pytest.approx(expected, rel=1.0e-9)
    assert fit["c"] == pytest.approx(coefficient, rel=1.0e-9)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1.0e-12)


def test_fit_scale_law_recovers_a_synthetic_exponent() -> None:
    entries = []
    for gap_over_L in (0.05, 0.1, 0.2, 0.4, 0.8, 1.6):
        entries.append(
            {
                "gap_over_L": gap_over_L,
                "sigma_A": [0.2 * gap_over_L**-3.0 + 1.0, 2.0 * gap_over_L**-3.0 + 1.0, 0.0],
            }
        )
    fit = fit_scale_law(entries, key="sigma_A", mode_index=1)
    assert fit["n"] == pytest.approx(3.0, rel=1.0e-6)
    assert fit["C"] == pytest.approx(2.0, rel=1.0e-6)


def test_richardson_gci_recovers_a_second_order_sequence() -> None:
    exact = 10.0
    values = [exact - 4.0, exact - 4.0 / 2.0**2, exact - 4.0 / 2.0**4]
    result = richardson_gci(values, refinement=2.0)
    assert result["order"] == pytest.approx(2.0, rel=1.0e-9)
    assert result["extrapolated"] == pytest.approx(exact, rel=1.0e-9)
    assert 0.0 < result["gci"] < 0.1

# ---------------------------------------------------------------------------
# documentation pin
# ---------------------------------------------------------------------------


def test_design_note_points_at_the_code_and_the_record() -> None:
    """The docs page for this node must exist, name the modules and the record.

    ``docs/multibody_frequency_dependence_3d.md`` is the prose half of this
    study; this pin keeps it from drifting away from the code it describes or
    from the generated report it defers the numbers to.  It also registers the
    page as read by a test, so the documentation audit
    (``tensorbem.benchmarks.doc_claim_audit``) cannot lose it quietly.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    page = root / "docs" / "multibody_frequency_dependence_3d.md"
    text = page.read_text()
    assert "tensorbem.multibody3d" in text
    assert "tensorbem/benchmarks/multibody_frequency_dependence_3d.py" in text
    assert "benchmarks/multibody_frequency_dependence_3d/report.md" in text
    assert "rigid-wall" in text and "pressure-release" in text.lower()
