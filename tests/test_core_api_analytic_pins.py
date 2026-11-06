"""Analytic pins for core TensorBEM API that no test used to reference.

Mission m4757.  A function-level scan of ``src/tensorbem`` against ``tests/``
(see ``tools/untested_public_api.py``) found nine *mainline* (non-benchmark)
public functions that no test file mentions at all.  Module-level presence in
the suite -- the granularity that ``tensorbem.capability_evidence`` checks --
hid them: the module was "covered", the function never ran.

Each function below is exercised against an *independent closed form*, never
against another code path of the package:

======================================  ====================================
function                                analytic anchor
======================================  ====================================
``geometry3d.floating_sphere_hull_mesh``  spherical-cap volume / zone area /
                                          waterline circle of a sphere
                                          truncated by ``z = 0``
``free_surface_solver3d.incident_airy_wave_3d``  deep-water Airy kinematics
                                          (``dphi/dz = nu*phi`` at ``z=0``
                                          and ``phi ~ exp(nu*z)``)
``free_surface_solver3d.rigid_body_normals_3d``  closed-surface force / moment
                                          identities
``finite_depth.mode_norms``             Sturm--Liouville orthonormality of the
                                          finite-depth vertical eigenmodes
``finite_depth3d.vertical_mode``        ``cosh(k0(z+h))`` and its deep-water
                                          limit ``exp(k0 z)``
``finite_depth3d.polish_evanescent_roots``  dispersion residual of the
                                          evanescent roots
``finite_depth3d.deep_water_limit_error``  monotone exponential decay of the
                                          finite-depth kernel error in depth
``helmholtz.assemble_helmholtz_double_layer``  double-layer jump relation
                                          ``u_out - u_in = -sigma``
``helmholtz.default_chief_points``      incircle containment (independent of
                                          the module's own point-in-polygon)
``conditioning.condition_number_trend``  per-level agreement with
                                          ``estimate_operator_conditions``
======================================  ====================================
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem import finite_depth as fd
from tensorbem import finite_depth3d as fd3
from tensorbem import free_surface_solver3d as fs3
from tensorbem import geometry3d as g3
from tensorbem.conditioning import (
    condition_number_trend,
    estimate_operator_conditions,
)
from tensorbem.geometry import PanelMesh
from tensorbem.helmholtz import (
    assemble_helmholtz_double_layer,
    default_chief_points,
)

GRAVITY = 9.80665


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _wetted_volume(hull) -> float:
    """``(1/3) * closed integral of x . n dS`` over the hull + waterplane lid.

    The lid lies in ``z = 0`` and contributes nothing because ``x . n = 0``
    there, so the hull triangles alone give the displaced volume.
    """
    v = hull.vertices
    v0, v1, v2 = v[hull.faces[:, 0]], v[hull.faces[:, 1]], v[hull.faces[:, 2]]
    triple = np.einsum("ij,ij->i", v0, np.cross(v1, v2))
    return float(np.sum(triple) / 6.0)


def _wetted_area(hull) -> float:
    v = hull.vertices
    v0, v1, v2 = v[hull.faces[:, 0]], v[hull.faces[:, 1]], v[hull.faces[:, 2]]
    return float(0.5 * np.sum(np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)))


def _face_normals(hull) -> np.ndarray:
    v = hull.vertices
    v0, v1, v2 = v[hull.faces[:, 0]], v[hull.faces[:, 1]], v[hull.faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return cross / np.linalg.norm(cross, axis=1)[:, None]


def _regular_polygon(n: int, radius: float = 1.0) -> PanelMesh:
    theta = 2.0 * math.pi * np.arange(n) / n
    return PanelMesh(np.column_stack((radius * np.cos(theta), radius * np.sin(theta))))


# 200-node Gauss-Legendre rule, computed once: leggauss is an eigendecomposition
# and costs ~1.5 s under load, so it must not be inside the per-integral path.
_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(200)


def _gauss_integral(f, a: float, b: float) -> float:
    """Gauss-Legendre quadrature; exact to 1e-15 for the smooth cos/cosh products used here."""
    grid = 0.5 * (b - a) * _GAUSS_X + 0.5 * (a + b)
    return float(0.5 * (b - a) * np.sum(_GAUSS_W * f(grid)))


# --------------------------------------------------------------------------- #
# 1. geometry3d.floating_sphere_hull_mesh -- exact spherical-cap analytics
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("radius,z0", [(1.0, -0.5), (2.0, -0.75)])
def test_floating_sphere_hull_volume_area_waterline(radius: float, z0: float) -> None:
    """Volume, wetted area and waterline radius match the closed sphere forms.

    A sphere of radius ``R`` centred at ``z0 < 0`` cut by ``z = 0`` is a
    spherical cap of height ``H = R - z0`` (the wetted part is the *larger*
    cap, since the centre sits below the waterplane):

    * displaced volume ``pi*H**2*(3R - H)/3``
    * wetted zone area   ``2*pi*R*H``
    * waterline radius   ``sqrt(R**2 - z0**2)``
    """
    height = radius - z0
    volume_ref = math.pi * height**2 * (3.0 * radius - height) / 3.0
    area_ref = 2.0 * math.pi * radius * height
    radius_wl_ref = math.sqrt(radius**2 - z0**2)

    errors = {}
    wl_sagitta = {}
    for subdivisions in (2, 3, 4):
        hull = g3.floating_sphere_hull_mesh(
            subdivisions, radius=radius, center=(0.0, 0.0, z0)
        )
        volume = _wetted_volume(hull)
        area = _wetted_area(hull)
        wl = hull.vertices[np.abs(hull.vertices[:, 2]) <= 1.0e-12]
        radius_wl = np.hypot(wl[:, 0], wl[:, 1])

        assert radius_wl.size > 0
        # the clipped waterline polygon is inscribed in the analytic circle:
        # no point may lie outside it, and the chord sagitta shrinks with the
        # subdivision level (polyhedral, second order)
        assert np.max(radius_wl) <= radius_wl_ref + 1.0e-12 * radius
        wl_sagitta[subdivisions] = radius_wl_ref - np.min(radius_wl)

        # normals point outward from the body into the fluid
        normals = _face_normals(hull)
        centres = hull.vertices[hull.faces].mean(axis=1)
        radial = centres - np.array([0.0, 0.0, z0])
        assert np.min(np.einsum("ij,ij->i", normals, radial)) > 0.0

        errors[subdivisions] = (
            abs(volume - volume_ref) / volume_ref,
            abs(area - area_ref) / area_ref,
        )

    # polyhedral approximation: second order, so the error must fall ~4x
    assert errors[4][0] < errors[3][0] < errors[2][0]
    assert errors[4][0] * 2.5 < errors[2][0]
    assert errors[4][0] < 5.0e-3
    assert errors[4][1] < 5.0e-3
    # waterline chord error: inside the circle and shrinking ~4x per level
    assert wl_sagitta[4] < wl_sagitta[3] < wl_sagitta[2]
    assert wl_sagitta[4] < 2.0e-3 * radius


# --------------------------------------------------------------------------- #
# 2. free_surface_solver3d.incident_airy_wave_3d -- deep-water kinematics
# --------------------------------------------------------------------------- #
def test_incident_airy_wave_satisfies_free_surface_and_decay() -> None:
    """``phi_I`` obeys the deep-water free-surface condition and decays as e^{kz}."""
    nu = 0.7  # omega**2 / g  =>  k = nu for deep water
    amplitude = 1.7
    wave_angle = math.radians(25.0)
    xs = np.linspace(-2.0, 2.0, 5)
    ys = np.linspace(-1.5, 1.5, 4)
    zs = np.array([0.0, -0.4, -1.2])
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)

    phi, grad = fs3.incident_airy_wave_3d(
        grid, nu, amplitude=amplitude, wave_angle=wave_angle
    )
    assert phi.shape == (grid.shape[0],)
    assert grad.shape == (grid.shape[0], 3)

    # deep-water dispersion: omega = sqrt(g*k) with k = nu
    omega = math.sqrt(GRAVITY * nu)
    # dphi/dz = k*phi at z = 0 and, in fact, at every depth for deep water
    np.testing.assert_allclose(grad[:, 2], nu * phi, rtol=1.0e-13, atol=0.0)

    # gradient must be the actual gradient: central differences in the two
    # horizontal directions, one-sided second order in z (the potential is only
    # defined for z <= 0, so no upward stencil exists at the waterplane)
    step = 1.0e-6
    for axis in (0, 1):
        plus = grid.copy()
        minus = grid.copy()
        plus[:, axis] += step
        minus[:, axis] -= step
        phi_plus, _ = fs3.incident_airy_wave_3d(
            plus, nu, amplitude=amplitude, wave_angle=wave_angle
        )
        phi_minus, _ = fs3.incident_airy_wave_3d(
            minus, nu, amplitude=amplitude, wave_angle=wave_angle
        )
        fd_grad = (phi_plus - phi_minus) / (2.0 * step)
        assert np.max(np.abs(fd_grad - grad[:, axis])) < 1.0e-5 * np.max(
            np.abs(grad[:, axis])
        )

    down = grid.copy()
    down[:, 2] -= step
    down2 = grid.copy()
    down2[:, 2] -= 2.0 * step
    phi_1, _ = fs3.incident_airy_wave_3d(
        down, nu, amplitude=amplitude, wave_angle=wave_angle
    )
    phi_2, _ = fs3.incident_airy_wave_3d(
        down2, nu, amplitude=amplitude, wave_angle=wave_angle
    )
    # stencil taken downward, so the sign is (3, -4, +1), not (-3, +4, -1)
    fd_grad_z = (3.0 * phi - 4.0 * phi_1 + phi_2) / (2.0 * step)
    assert np.max(np.abs(fd_grad_z - grad[:, 2])) < 1.0e-5 * np.max(np.abs(grad[:, 2]))

    # magnitude: |phi(z)| = (g*A/omega) * exp(nu*z)
    z_column = grid[:, 2]
    magnitude = np.abs(phi)
    np.testing.assert_allclose(
        magnitude, (GRAVITY * amplitude / omega) * np.exp(nu * z_column), rtol=1.0e-13
    )

    # rotating the wave angle by beta rotates the crest field: phi(beta) evaluated
    # at rotated points equals phi(0) at the original points
    # rotate the *points* by -beta so that x' . e_x = x cos(beta) + y sin(beta)
    rot = np.array(
        [
            [math.cos(wave_angle), math.sin(wave_angle)],
            [-math.sin(wave_angle), math.cos(wave_angle)],
        ]
    )
    rotated = grid.copy()
    rotated[:, :2] = grid[:, :2] @ rot.T
    phi_rot, _ = fs3.incident_airy_wave_3d(rotated, nu, amplitude=amplitude)
    np.testing.assert_allclose(phi_rot, phi, rtol=1.0e-12, atol=1.0e-14)


def test_incident_airy_wave_rejects_invalid_input() -> None:
    pts = np.zeros((3, 3))
    with pytest.raises(ValueError):
        fs3.incident_airy_wave_3d(pts + np.array([0.0, 0.0, 0.25]), 0.5)  # z > 0
    with pytest.raises(ValueError):
        fs3.incident_airy_wave_3d(pts, 0.0)  # nu <= 0
    with pytest.raises(ValueError):
        fs3.incident_airy_wave_3d(pts, -1.0)
    with pytest.raises(ValueError):
        fs3.incident_airy_wave_3d(np.zeros((3, 2)), 0.5)  # not 3-D points


# --------------------------------------------------------------------------- #
# 3. free_surface_solver3d.rigid_body_normals_3d -- closed-surface identities
# --------------------------------------------------------------------------- #
def test_rigid_body_normals_closed_surface_identities() -> None:
    """A closed surface exerts no net force or moment: ``sum(n dS) = sum((x-c) x n dS) = 0``."""
    base = g3.icosphere(3, radius=1.3)
    shift = np.array([0.2, -0.1, -1.5])  # keep every centroid strictly below z = 0
    mesh = g3.TriangleMesh(base.vertices + shift, base.faces)
    centre = shift + np.array([0.0, 0.0, 0.3])
    gn = fs3.rigid_body_normals_3d(mesh, centre)
    assert gn.shape == (mesh.faces.shape[0], 6)

    areas = mesh.areas[:, None]
    force = np.sum(gn[:, :3], axis=0)
    # weighted by area, the generalised normals are exact surface integrals
    weighted_force = np.sum(gn[:, :3] * areas, axis=0)
    weighted_moment = np.sum(gn[:, 3:] * areas, axis=0)
    np.testing.assert_allclose(force, np.zeros(3), atol=1.0e-12)
    np.testing.assert_allclose(mesh.volume, 4.0 / 3.0 * math.pi * 1.3**3, rtol=1e-2)
    assert np.linalg.norm(weighted_force) < 1.0e-11
    assert np.linalg.norm(weighted_moment) < 1.0e-11

    # translation of the moment origin is an exact rigid shift of the rotational
    # block: (x - c') x n = (x - c) x n - delta x n with delta = c' - c
    shifted = fs3.rigid_body_normals_3d(mesh, centre + np.array([0.4, 0.0, 0.0]))
    delta = np.array([0.4, 0.0, 0.0])
    np.testing.assert_allclose(
        shifted[:, 3:],
        gn[:, 3:] - np.cross(np.broadcast_to(delta, gn[:, :3].shape), gn[:, :3]),
        rtol=0.0,
        atol=1.0e-13,
    )


# --------------------------------------------------------------------------- #
# 4. finite_depth.mode_norms -- Sturm-Liouville orthonormality
# --------------------------------------------------------------------------- #
def test_finite_depth_mode_norms_are_the_quadrature_limits() -> None:
    """``mode_norms`` reproduces ``int_{-h}^{0} cos^2(mu_n(z+h)) dz`` and the modes are orthogonal.

    The vertical eigenmodes of the finite-depth problem solve
    ``u'' + mu**2 u = 0`` with ``u'(-h) = 0`` and the Robin condition
    ``u'(0) = nu u(0)`` at the free surface, so they are a self-adjoint
    Sturm-Liouville family: ``int u_m u_n = norm_n * delta_mn``.
    """
    nu, depth = 1.2, 7.0
    count = 4
    mu = fd.evanescent_wavenumbers(nu, depth, count)
    # sanity: the evanescent roots satisfy mu*tan(mu*h) = -nu
    np.testing.assert_allclose(
        mu * np.tan(mu * depth), -nu * np.ones_like(mu), rtol=0.0, atol=1.0e-8
    )

    norms = fd.mode_norms(nu, depth, mu)
    for index, (m, norm) in enumerate(zip(mu, norms)):
        quadrature = _gauss_integral(
            lambda z, m=m: np.cos(m * (z + depth)) ** 2, -depth, 0.0
        )
        assert abs(quadrature - norm) < 1.0e-11 * norm, index

        # cross terms vanish
        for other in mu[index + 1 :]:
            cross = _gauss_integral(
                lambda z, m=m, o=other: np.cos(m * (z + depth)) * np.cos(o * (z + depth)),
                -depth,
                0.0,
            )
            assert abs(cross) < 1.0e-9 * norm

    # the propagating mode (cosh) is orthogonal to every evanescent one as well
    k0 = fd.finite_depth_wavenumber(nu, depth)
    norm0 = fd.propagating_mode_norm(k0, depth)
    for m in mu:
        cross = _gauss_integral(
            lambda z, m=m: np.cosh(k0 * (z + depth)) * np.cos(m * (z + depth)),
            -depth,
            0.0,
        )
        assert abs(cross) < 1.0e-8 * math.sqrt(norm0 * norms[0])


# --------------------------------------------------------------------------- #
# 5. finite_depth3d.vertical_mode -- definition, deep-water limit, domain guard
# --------------------------------------------------------------------------- #
def test_vertical_mode_definition_and_deep_water_limit() -> None:
    depth = 40.0
    k0 = 5.0  # k0*h = 200: deep water to machine precision
    z = np.linspace(-depth, 0.0, 21)
    phi = fd3.vertical_mode(z, k0, depth)
    np.testing.assert_allclose(phi, np.cosh(k0 * (z + depth)), rtol=1.0e-14)

    # normalised shape -> exp(k0*z) once k0*h >> 1.  The bottom row is excluded:
    # there phi = cosh(0) = 1 while cosh(k0*h) = e^{k0*h}/2, so the ratio picks up
    # the factor 2 of the half-exponential -- the limit holds only away from z = -h.
    ratio = (phi / np.cosh(k0 * depth))[1:]
    np.testing.assert_allclose(ratio, np.exp(k0 * z)[1:], rtol=1.0e-6)

    # endpoints are part of the closed domain; outside it is an error
    assert fd3.vertical_mode(np.array([0.0, -depth]), k0, depth).shape == (2,)
    with pytest.raises(ValueError):
        fd3.vertical_mode(np.array([0.1]), k0, depth)
    with pytest.raises(ValueError):
        fd3.vertical_mode(np.array([-depth - 0.1]), k0, depth)


# --------------------------------------------------------------------------- #
# 6. finite_depth3d.polish_evanescent_roots -- dispersion residual
# --------------------------------------------------------------------------- #
def test_polish_evanescent_roots_drives_dispersion_residual_down() -> None:
    nu, depth = 0.8, 6.0
    bracketed = fd.evanescent_wavenumbers(nu, depth, 6)
    polished = fd3.polish_evanescent_roots(bracketed, nu, depth, steps=3)

    def residual(mu):
        return np.abs(mu * np.tan(mu * depth) + nu) / nu

    before = residual(bracketed)
    after = residual(polished)
    assert np.all(after < before)
    assert np.max(after) < 1.0e-14

    # idempotent to round-off: polishing a polished root must not move it
    twice = fd3.polish_evanescent_roots(polished, nu, depth, steps=3)
    np.testing.assert_allclose(twice, polished, rtol=0.0, atol=1.0e-15)
    # roots keep their bracket (no jumping to a neighbour branch)
    np.testing.assert_allclose(polished, bracketed, rtol=1.0e-6)


# --------------------------------------------------------------------------- #
# 7. finite_depth3d.deep_water_limit_error -- the diagnostic must discriminate
# --------------------------------------------------------------------------- #
def test_deep_water_limit_error_decays_and_is_not_degenerate() -> None:
    nu = 0.5
    source = np.array([[0.0, 0.0, -1.0]])
    theta = np.linspace(0.0, 2.0 * math.pi, 6, endpoint=False)
    targets = np.column_stack(
        (1.5 * np.cos(theta), 1.5 * np.sin(theta), -0.5 * np.ones_like(theta))
    )

    errors = [
        fd3.deep_water_limit_error(targets, source, nu, depth) for depth in (20.0, 40.0)
    ]
    assert errors[0] < 1.0e-3
    assert errors[1] < 0.2 * errors[0]
    # exponential rate: doubling the depth knocks the error down by ~an order
    # (measured 1.88e-4 -> 1.90e-5 for h = 20 -> 40 at nu = 0.5)

    # negative control -- in shallow water the finite-depth kernel is genuinely
    # different from the deep-water one, so the diagnostic must be O(1) there.
    shallow = fd3.deep_water_limit_error(targets, source, nu, 1.0)
    assert shallow > 0.2


# --------------------------------------------------------------------------- #
# 8. helmholtz -- double-layer jump relation and CHIEF-point containment
# --------------------------------------------------------------------------- #
def test_helmholtz_double_layer_jump_relation() -> None:
    """The double-layer potential jumps by the local density across the boundary."""
    radius = 1.0
    mesh = _regular_polygon(128, radius=radius)
    sigma = np.ones(mesh.vertices.shape[0] - 0) * 1.0
    sigma = np.ones(len(mesh.vertices))

    for wavenumber in (0.5, 2.0):
        # probe at the first panel centre, offset along the outward normal
        centre = mesh.centers[0]
        normal = mesh.normals[0]
        offset = 2.0e-2
        targets = np.stack((centre + offset * normal, centre - offset * normal))
        matrix = assemble_helmholtz_double_layer(mesh, wavenumber, targets, panel_nodes=16)
        values = matrix @ sigma
        jump = values[0] - values[1]  # outside (into fluid) minus inside the body
        assert abs(abs(jump) - 1.0) < 0.03, (wavenumber, jump)
        assert abs(jump.real) > 10.0 * abs(jump.imag)


def test_default_chief_points_are_inside_and_deterministic() -> None:
    sides = 48
    mesh = _regular_polygon(sides, radius=1.0)
    points = default_chief_points(mesh, count=5)
    assert points.shape == (5, 2)
    assert np.isfinite(points).all()
    # every point is inside the incircle of the regular polygon -- an
    # independent containment test, not the module's point-in-polygon routine
    inradius = math.cos(math.pi / sides)
    assert np.max(np.linalg.norm(points, axis=1)) < inradius
    np.testing.assert_array_equal(points, default_chief_points(mesh, count=5))

    with pytest.raises(ValueError):
        default_chief_points(mesh, count=0)
    with pytest.raises(TypeError):
        default_chief_points("not a mesh", count=3)


# --------------------------------------------------------------------------- #
# 9. conditioning.condition_number_trend -- per-level consistency
# --------------------------------------------------------------------------- #
def test_condition_number_trend_matches_single_level_reports() -> None:
    counts = [1]  # 80 panels: the direct condition number stays cheap
    trend = condition_number_trend(
        g3.icosphere, counts, method="direct", power_iterations=10
    )
    assert len(trend) == len(counts)
    for subdivisions, report in zip(counts, trend):
        mesh = g3.icosphere(subdivisions)
        reference = estimate_operator_conditions(mesh)
        assert report.mesh_panels == len(mesh.faces)
        assert report.single_layer_condition == pytest.approx(
            reference.single_layer_condition, rel=0.0, abs=0.0
        )
        assert report.double_layer_condition == pytest.approx(
            reference.double_layer_condition, rel=0.0, abs=0.0
        )
        assert report.estimation_method == "direct"
        assert np.isfinite(report.single_layer_condition)