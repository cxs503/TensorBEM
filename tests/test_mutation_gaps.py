"""Tests backfilled from the mutation audit (``tensorbem.benchmarks.mutation_audit``).

Every test here is the direct answer to a *survivor*: a one-line defect that the
pre-existing suite accepted.  Each case is written against an independent
reference -- a closed-form integral identity, a quadrature, a finite difference,
or an exact algebraic identity -- not against the implementation it polices, so
it cannot be satisfied by copying the formula it is checking.

The mapping from survivor to test is recorded in
``docs/mutation_audit_tensorbem.md``.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from tensorbem.finite_depth import (
    finite_depth_wavenumber,
    green_function,
    propagating_mode_norm,
)
from tensorbem.free_surface import green_depth_gradient
from tensorbem.free_surface import green_function as deep_green_function
from tensorbem.free_surface import green_gradient as deep_green_gradient
from tensorbem.free_surface import green_gradient_field as deep_green_gradient_field
from tensorbem.geometry import regular_polygon
from tensorbem.geometry3d import WettedHullMesh, box_hull_mesh
from tensorbem.hydrostatic import (
    compute_displaced_volume_and_buoyancy,
    compute_mass_matrix,
    compute_waterplane_properties,
)
from tensorbem.kernels import (
    double_layer_kernel,
    integrate_double_layer_panel,
    integrate_single_layer_panel,
)
from tensorbem.solver import compute_added_mass_matrix_2d


def test_double_layer_kernel_reproduces_the_circle_solid_angle() -> None:
    """∮ ∂G/∂n ds over a circle about the target is exactly -1.

    This is the 2-D jump identity for a double layer: the interior solid angle of
    a closed curve is 2*pi, so the integral of the double-layer kernel is -1
    regardless of the radius.  A kernel that is off by any power of ``r`` -- the
    mutation ``_TWO_PI * r2`` -> ``_TWO_PI / r2`` is one -- breaks the
    radius-independence, which is what this test measures.
    """
    target = np.array([0.0, 0.0])
    for radius in (0.7, 3.7):
        angles = np.linspace(0.0, 2.0 * math.pi, 512, endpoint=False)
        sources = radius * np.column_stack((np.cos(angles), np.sin(angles)))
        normals = np.column_stack((np.cos(angles), np.sin(angles)))  # outward
        values = np.array(
            [double_layer_kernel(target, s, n) for s, n in zip(sources, normals)]
        )
        integral = float(np.sum(values) * (2.0 * math.pi * radius / angles.size))
        assert integral == pytest.approx(-1.0, rel=1e-6), (
            f"double-layer solid angle at radius {radius}: {integral}"
        )


def test_panel_integrals_reject_a_zero_length_panel() -> None:
    """A zero-length panel has no area; it must be refused, not silently divided.

    Both panel integrals are checked: the audit mutated the guard in the
    double-layer routine (``length <= 0`` -> ``length < 0``) and the single-layer
    guard was the only one covered.
    """
    target = np.array([0.0, 1.0])
    start = np.array([0.0, 0.0])
    tangent = np.array([1.0, 0.0])
    normal = np.array([0.0, 1.0])
    with pytest.raises(ValueError):
        integrate_single_layer_panel(target, start, tangent, 0.0)
    with pytest.raises(ValueError):
        integrate_double_layer_panel(target, start, tangent, normal, 0.0)


def test_propagating_mode_norm_matches_its_own_quadrature() -> None:
    """N_0 = ∫_{-h}^{0} cosh^2(k0 (z+h)) dz = (h/2)(1 + sinh(2 k0 h)/(2 k0 h)).

    The right-hand side is evaluated independently by composite trapezoid on the
    mode shape, so the closed form cannot be "verified" by itself.  The mutation
    ``sinh(2 k0 h)`` -> ``sinh(-2 k0 h)`` flips the sign of the wave term and is
    caught here.
    """
    nu, depth = 0.6, 2.0
    k0 = finite_depth_wavenumber(nu, depth)
    assert math.isclose(k0 * math.tanh(k0 * depth), nu, rel_tol=1e-11)

    z = np.linspace(-depth, 0.0, 40001)
    quadrature = float(np.trapezoid(np.cosh(k0 * (z + depth)) ** 2, z))
    assert propagating_mode_norm(k0, depth) == pytest.approx(quadrature, rel=1e-9)


def test_deep_water_source_gradient_matches_a_finite_difference() -> None:
    """dg/ds_z from ``green_gradient`` must equal a central difference of G.

    The wave (residue) term dominates this derivative at moderate depth, so a
    defect in it -- the audit flipped ``frequency * residue`` to
    ``1.0 / (frequency * residue)`` -- is visible immediately.
    """
    nu = 0.6
    x, z, s_x, s_z = 1.0, -1.0, 0.0, -1.2
    step = 1.0e-6
    _, dg_d_source_z = deep_green_gradient(x, z, s_x, s_z, nu)
    forward = deep_green_function(x, z, s_x, s_z, nu)
    plus = deep_green_function(x, z, s_x, s_z + step, nu)
    minus = deep_green_function(x, z, s_x, s_z - step, nu)
    assert forward == pytest.approx(deep_green_function(x, z, s_x, s_z, nu))
    finite_difference = (plus - minus) / (2.0 * step)
    assert complex(dg_d_source_z) == pytest.approx(finite_difference, rel=1e-6, abs=1e-9)


def test_mass_matrix_obeys_the_rigid_body_block_identity() -> None:
    """M = [[m I, -m S^T], [-m S, J + m S S^T]] with S the CG skew matrix.

    Built from an explicitly supplied inertia tensor, so the default
    radius-of-gyration approximation cannot mask a defect.  The audit mutated the
    ``(0, 0)`` entry of ``S`` from 0 to 1, which puts a spurious ``-m`` into the
    coupling block.
    """
    mass = 1200.0
    cg = np.array([1.2, -0.4, 0.9])
    inertia = np.array([[3.1e4, 2.0e2, -1.0e2], [2.0e2, 5.2e4, 3.0e2], [-1.0e2, 3.0e2, 6.7e4]])

    x, y, z = cg
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    expected = np.zeros((6, 6))
    expected[0:3, 0:3] = mass * np.eye(3)
    expected[0:3, 3:6] = -mass * skew.T
    expected[3:6, 0:3] = -mass * skew
    expected[3:6, 3:6] = inertia + mass * (skew @ skew.T)

    measured = compute_mass_matrix(mass, cg, inertia)
    assert np.allclose(measured, expected, rtol=1e-12, atol=1e-9)
    assert np.allclose(measured, measured.T, rtol=1e-12, atol=1e-9)
    # A rigid-body translation [v; 0] must produce momentum [m v; -m S v].
    velocity = np.array([0.3, -0.2, 0.1, 0.0, 0.0, 0.0])
    momentum = measured @ velocity
    assert np.allclose(momentum[0:3], mass * velocity[0:3], rtol=1e-12, atol=1e-9)
    assert np.allclose(momentum[3:6], -mass * (skew @ velocity[0:3]), rtol=1e-12, atol=1e-9)


def test_added_mass_defaults_to_unit_density() -> None:
    """ρ has a documented default of 1: a circle's added mass is then π a².

    Called without the ``density`` argument on purpose -- the audit showed the
    default value itself was unpinned.
    """
    radius = 1.0
    mesh = regular_polygon(96, radius)
    result = compute_added_mass_matrix_2d(mesh)
    matrix = result.added_mass_matrix
    assert matrix.shape == (3, 3)  # (surge_x, heave_y, roll_xy)
    expected = math.pi * radius**2
    assert matrix[0, 0] == pytest.approx(expected, rel=2e-2)
    assert matrix[1, 1] == pytest.approx(expected, rel=2e-2)
    assert abs(matrix[0, 1]) < 2e-2 * expected
    assert abs(matrix[1, 0]) < 2e-2 * expected
    # A circle centred on the expansion point has no surge-heave coupling and no
    # roll added mass.
    assert abs(matrix[0, 2]) < 2e-2 * expected
    assert abs(matrix[2, 2]) < 2e-2 * expected


def test_zero_modes_means_automatic_not_invalid() -> None:
    """``modes=0`` selects the automatic floor; only negative counts are errors."""
    nu, depth = 0.6, 2.0
    automatic = green_function(1.0, -1.0, 0.0, -1.0, nu, depth, modes=None)
    zero = green_function(1.0, -1.0, 0.0, -1.0, nu, depth, modes=0)
    assert complex(zero) == complex(automatic)
    with pytest.raises(ValueError):
        green_function(1.0, -1.0, 0.0, -1.0, nu, depth, modes=-1)# --------------------------------------------------------------------------
# Third backfill (mission m2829): the four gaps left open in
# ``docs/mutation_audit_tensorbem.md`` §5 ("仍然存活的变异体").
# --------------------------------------------------------------------------
def test_eccentric_barge_waterplane_centroid_and_second_moments() -> None:
    """An off-centre waterplane pins the waterplane *centroid*, not just its area.

    The audit survivor ``hydrostatic.py: x_moment ... / 6.0 -> * 6.0`` stayed
    green because every earlier case (hemisphere, centred barge, spheroid) has
    a waterplane whose first moment about the origin is zero, so both the right
    and the wrong formula give a centroid at (0, 0). Translating a 2 m x 1 m
    barge so that its waterplane is exactly ``x in [0, 2]``, ``y in [0, 1]``
    makes the centroid (1, 1/2) and the second moments about that centroid

        I_xx = L B^3 / 12 = 1/6,  I_yy = B L^3 / 12 = 2/3,  I_xy = 0,

    all closed-form polygon integrals, so the assertion cannot be satisfied by
    reproducing the implementation formula.
    """
    length, beam, draft = 2.0, 1.0, 1.0
    offset = np.array([1.0, 0.5, 0.0])
    base = box_hull_mesh(length=length, beam=beam, draft=draft)
    hull = WettedHullMesh(base.vertices + offset, base.faces)

    area, centroid, moments = compute_waterplane_properties(hull)
    assert area == pytest.approx(length * beam, rel=1e-13)
    assert centroid[0] == pytest.approx(1.0, rel=1e-12)
    assert centroid[1] == pytest.approx(0.5, rel=1e-12)
    assert moments[0] == pytest.approx(length * beam**3 / 12.0, rel=1e-12)
    assert moments[1] == pytest.approx(beam * length**3 / 12.0, rel=1e-12)
    assert abs(moments[2]) < 1e-12

    # The same translation must leave the displaced volume exact and move the
    # centre of buoyancy with the hull: V = L B T, CoB = (1, 1/2, -T/2).
    volume, centre = compute_displaced_volume_and_buoyancy(hull)
    assert volume == pytest.approx(length * beam * draft, rel=1e-12)
    assert centre[0] == pytest.approx(1.0, rel=1e-12)
    assert centre[1] == pytest.approx(0.5, rel=1e-12)
    assert centre[2] == pytest.approx(-draft / 2.0, rel=1e-12)

    # Translate again: area and centroid-relative second moments are invariant,
    # the centroid follows, and the survivor's factor-of-36 blunder cannot hide.
    shift = np.array([-2.5, 3.25, 0.0])
    moved = WettedHullMesh(base.vertices + offset + shift, base.faces)
    area2, centroid2, moments2 = compute_waterplane_properties(moved)
    assert area2 == pytest.approx(area, rel=1e-13)
    assert centroid2[0] == pytest.approx(1.0 + shift[0], rel=1e-12)
    assert centroid2[1] == pytest.approx(0.5 + shift[1], rel=1e-12)
    assert np.allclose(moments2, moments, rtol=1e-12, atol=1e-12)


def test_fully_submerged_hull_reports_an_exactly_empty_waterplane() -> None:
    """A hull with no waterline loop has zero waterplane area, not a unit area.

    The survivor ``hydrostatic.py`` ``return 0.0`` -> ``return 1.0`` in the
    ``len(waterline_edges) == 0`` branch lived because no case ever reached it:
    every benchmark hull pierces z = 0. A closed tetrahedron wholly below the
    free surface passes ``WettedHullMesh`` validation (a closed mesh has no
    unmatched edges, hence no waterline) and reaches the branch; its reference
    values are exact zeros, so any non-zero placeholder is caught.
    """
    vertices = np.array(
        [[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, -2.0]]
    )
    faces = np.array([[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]])
    hull = WettedHullMesh(vertices, faces)
    assert hull.waterline_edges.shape == (0, 2)

    area, centroid, moments = compute_waterplane_properties(hull)
    assert area == 0.0
    assert centroid.shape == (2,)
    assert moments.shape == (3,)
    assert np.all(centroid == 0.0)
    assert np.all(moments == 0.0)

    # Sanity: this hull is not degenerate in volume. V = 1/6, CoB = mean vertex.
    volume, centre = compute_displaced_volume_and_buoyancy(hull)
    assert volume == pytest.approx(1.0 / 6.0, rel=1e-12)
    assert np.allclose(centre, vertices.mean(axis=0), rtol=1e-12)


def test_degenerate_displaced_volume_returns_a_three_vector() -> None:
    """Both zero-volume exits must return a length-3 centre, exactly zero.

    The survivor ``compute_displaced_volume_and_buoyancy: np.zeros(3) ->
    np.zeros(4)`` survived because a length-4 centre is only produced on the
    two early exits, which valid hulls never reach. Both exits are exercised
    here on the documented contract (empty face list, cancelling signed
    volumes) -- the function only reads ``vertices`` and ``faces``, so a
    stand-in object is the honest way to reach them.
    """
    empty = SimpleNamespace(
        vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64)
    )
    volume, centre = compute_displaced_volume_and_buoyancy(empty)
    assert volume == 0.0
    assert centre.shape == (3,)
    assert np.all(centre == 0.0)

    # A closed surface plus its own reversed copy encloses zero signed volume
    # exactly, which reaches the ``abs(total_volume) < 1e-14`` exit.
    vertices = np.array(
        [[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, -2.0]]
    )
    faces = np.array([[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]])
    cancelling = SimpleNamespace(
        vertices=vertices,
        faces=np.vstack((faces, faces[:, ::-1])),
    )
    volume, centre = compute_displaced_volume_and_buoyancy(cancelling)
    assert volume == 0.0
    assert centre.shape == (3,)
    assert np.all(centre == 0.0)


def test_depth_derivative_matches_finite_differences_and_the_field_gradient() -> None:
    """``dG/dA`` (A = -(z + z0)) must agree with two independent evaluations.

    The survivor ``free_surface.py:243`` mutates the ``cos`` term of
    ``df_d_depth`` and lived because that derivative was computed and then
    discarded: nothing in the package consumed it. It is now the public
    :func:`green_depth_gradient`, and both checks below are independent of its
    formula:

    * a central finite difference of :func:`green_function` in which field and
      source point move together (offset fixed, ``A`` swept by ``-2h``), which
      only touches the kernel value routine;
    * the identity ``dG/dA = -dG_fs/dz`` against
      :func:`green_gradient_field`, whose vertical derivative is coded
      separately.

    Both are checked for the outgoing and the incoming (conjugated) kernel,
    which also pins the conjugation of this term.
    """
    nu, step = 0.6, 1.0e-6
    cases = (((0.7, -0.5, 0.0, -2.0)), ((0.3, -0.2, 0.0, -1.0)))
    for x, z, x0, z0 in cases:
        for outgoing in (True, False):
            analytic = green_depth_gradient(x, z, x0, z0, nu, outgoing=outgoing)
            field_vertical = deep_green_gradient_field(
                x, z, x0, z0, nu, outgoing=outgoing, include_rankine=False
            )[1]
            assert analytic == pytest.approx(-field_vertical, rel=1e-12)

            forward = deep_green_function(
                x, z + step, x0, z0 + step, nu, outgoing=outgoing,
                include_rankine=False,
            )
            backward = deep_green_function(
                x, z - step, x0, z0 - step, nu, outgoing=outgoing,
                include_rankine=False,
            )
            # d f/dh at h = 0 is dG/dz + dG/dz0 = 2 dG/dz = -2 dG/dA.
            finite_difference = -(forward - backward) / (2.0 * step) / 2.0
            assert abs(analytic) > 1.0e-3
            assert analytic == pytest.approx(finite_difference, rel=1e-8)

    # The incoming kernel's depth derivative must be the complex conjugate of
    # the outgoing one -- the sign of the residue term is the discriminating
    # part, so a conjugation slip shows up as a real-part mismatch.
    outgoing_value = green_depth_gradient(0.7, -0.5, 0.0, -2.0, nu)
    incoming_value = green_depth_gradient(0.7, -0.5, 0.0, -2.0, nu, outgoing=False)
    assert incoming_value == pytest.approx(complex(outgoing_value).conjugate(), rel=1e-14)
    assert abs(complex(outgoing_value).imag) > 1.0e-3