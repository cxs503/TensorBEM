"""Second-order mean wave drift force: near-field (Pinkster) and far-field (Maruo).

Two independent estimators of the time-mean, second-order horizontal force on a
*body fixed in regular waves* are provided, together with a closed-form analytic
anchor (full reflection at a two-dimensional vertical wall) used to pin the sign
and the magnitude of the near-field assembly:

Near field
----------
Integration of the second-order mean pressure over the mean wetted surface plus
the waterline correction (Pinkster's formulation, fixed body, no forward speed):

    F_j = -(1/2) rho * Int_S < |grad phi|^2 > n_j dS
          +(1/2) rho g * Int_WL < zeta^2 > n_j dl

with ``n`` the outward normal of the body (pointing into the fluid), ``phi`` the
first-order potential and ``zeta`` the first-order free-surface elevation.  For a
fixed body the two remaining terms of the general Pinkster formula vanish: the
``X . grad(dphi/dt)`` term because the body does not move, and the mean of
``-rho dphi^(2)/dt`` integrates to zero over the closed wetted surface.  With the
time convention ``exp(-i omega t)`` used throughout TensorBEM,

    < |grad phi|^2 > = (1/2) |grad phi_hat|^2,   < zeta^2 > = (1/2) |zeta_hat|^2,

and on the body surface of a fixed body the normal derivative of the *total*
potential vanishes identically, so ``grad phi_hat`` is purely tangential.

Far field
---------
Maruo's momentum-flux formula in deep water, written with the Kochin function
``K(theta)`` already provided by :func:`tensorbem.free_surface_solver3d.kochin_function_3d`:

    F_j = -(rho k^2 / 8 pi) * Int_0^{2 pi} |K(theta)|^2 n_j(theta) dtheta

The two routes share no numerical machinery: the near field needs the potential
and its surface gradient on the body panels, the far field only the far-field
asymptotics of the *scattered* potential through the Kochin integral over a full
azimuth circle.

Analytic anchors
----------------
* A two-dimensional vertical wall in a fully reflected (standing) wave carries a
  mean drift force per unit length ``rho g a^2 / 2``.  The same near-field
  assembly applied to the exact standing-wave field reproduces it to machine
  precision: the surface term ``-rho g a^2/2`` and the waterline term
  ``+rho g a^2`` cancel to one half of the waterline term.
* For a bottom-mounted vertical cylinder the mean drift force scales as
  ``(k a)^3`` as ``k a -> 0`` (the leading dipole contribution cancels in the
  azimuthal integral, the surviving term is a dipole x quadrupole product).

All quantities are SI: metres, seconds, kilograms, newtons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .free_surface_solver3d import (
    Diffraction3DResult,
    kochin_function_3d,
    solve_diffraction_3d,
)
from .geometry3d import SurfaceMesh3D, TriangleMesh, WettedHullMesh

FloatArray = np.ndarray

_DEFAULT_DENSITY = 1025.0
_DEFAULT_GRAVITY = 9.81


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _signed_volume_normals(mesh: SurfaceMesh3D) -> tuple[FloatArray, float]:
    """Return unit normals guaranteed to point out of the body, plus the volume.

    ``TriangleMesh.normals`` follows the face winding; to be robust against a
    mesh whose winding is globally reversed we orient with the divergence
    theorem, ``6 V = sum_faces v0 . (v1 x v2)`` with outward normals.
    """
    v0, v1, v2 = mesh.v0, mesh.v1, mesh.v2
    six_volume = float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))))
    normals = np.asarray(mesh.normals, dtype=float)
    if six_volume < 0.0:
        normals = -normals
        six_volume = -six_volume
    return normals, six_volume / 6.0


def _nodal_values(mesh: SurfaceMesh3D, panel_values: FloatArray) -> FloatArray:
    """Area-weighted vertex values from a piecewise-constant panel field."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    weights = np.asarray(mesh.areas, dtype=float)
    values = np.asarray(panel_values)
    acc = np.zeros((mesh.vertex_count,) + values.shape[1:], dtype=values.dtype)
    wsum = np.zeros(mesh.vertex_count, dtype=float)
    for corner in range(3):
        idx = faces[:, corner]
        np.add.at(acc, idx, values * weights.reshape((weights.size,) + (1,) * (values.ndim - 1)))
        np.add.at(wsum, idx, weights)
    return acc / wsum.reshape((wsum.size,) + (1,) * (values.ndim - 1))


def panel_surface_gradient(
    mesh: SurfaceMesh3D,
    panel_values: FloatArray,
    *,
    method: Literal["mls", "nodal"] = "mls",
    stencil_factor: float = 1.3,
) -> FloatArray:
    """Tangential gradient of a panel-wise field, one vector per panel.

    Two consistent recoveries are available:

    ``nodal``
        Area-weighted vertex values followed by the exact linear-triangle basis
        gradients ``grad N_i = (n x (v_{i+2} - v_{i+1})) / (2 A)``.  This is
        second-order accurate for the vertex values but its gradient is
        diffusive on coarse meshes, so it *under*-predicts ``|grad phi|^2``.
    ``mls`` (default)
        Moving least squares: a weighted fit of a *full three-dimensional linear*
        model over the panel centroids within ``stencil_factor * h_mean``, then
        projection of the fitted gradient on the tangent plane.  The 3-D model
        (rather than a 2-D tangent-plane model) is essential: it reproduces a
        linear field *exactly* on a curved surface, so the patch test is clean and
        the recovery is not biased by curvature.  A quadratic model was tried and
        rejected: its design matrix is nearly rank deficient because a
        surface patch is nearly planar, which silently destroyed the linear
        coefficients.
    """
    values = np.asarray(panel_values)
    if method == "nodal":
        nodal = _nodal_values(mesh, values)
        v0, v1, v2 = mesh.v0, mesh.v1, mesh.v2
        edge = np.cross(v1 - v0, v2 - v0)
        area2 = np.linalg.norm(edge, axis=1)
        n_plane = edge / area2[:, None]
        grad = np.zeros((mesh.panel_count, 3), dtype=nodal.dtype)
        for (a, b), corner in zip(((v2, v1), (v0, v2), (v1, v0)), (0, 1, 2)):
            grad += (
                np.cross(n_plane, a - b) / area2[:, None]
            ) * nodal[np.asarray(mesh.faces)[:, corner]][:, None]
        return grad
    if method != "mls":
        raise ValueError(f"unknown gradient method {method!r}")

    normals, _ = _signed_volume_normals(mesh)
    centres = np.asarray(mesh.centroids, dtype=float)
    radius = float(stencil_factor) * float(mesh.h_mean)
    if radius <= 0.0:
        raise ValueError("mesh has zero panel size")
    count = mesh.panel_count
    grad = np.zeros((count, 3), dtype=values.dtype)
    for i in range(count):
        offset = centres - centres[i]
        dist = np.linalg.norm(offset, axis=1)
        near = np.flatnonzero(dist <= radius)
        if near.size < 4:
            near = np.argsort(dist)[:4]
        d = offset[near]
        r = np.linalg.norm(d, axis=1)
        w = 1.0 / (1.0 + (r / radius) ** 2)
        du = values[near] - values[i]
        design = np.concatenate((np.ones((near.size, 1)), d), axis=1)
        sw = np.sqrt(w)
        try:
            coeffs, *_ = np.linalg.lstsq(design * sw[:, None], du * sw, rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            continue
        full = coeffs[1:4]
        grad[i] = full - (full @ normals[i]) * normals[i]
    return grad


def waterline_segments(mesh: SurfaceMesh3D, *, z_tolerance: float = 1.0e-9) -> dict:
    """Boundary edges lying on the mean free surface ``z = 0``.

    Returns the midpoints, lengths and horizontal outward normals of every
    boundary edge whose two vertices are at ``z = 0`` within ``z_tolerance``
    times the mesh size, together with the adjacent panel indices.
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    scale = max(float(mesh.h_max), 1.0e-12)
    tol = z_tolerance * scale
    edges = np.vstack(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])
    )
    keys = np.sort(edges, axis=1)
    uniq, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    boundary = np.flatnonzero(counts == 1)
    vertices = np.asarray(mesh.vertices, dtype=float)
    if boundary.size == 0:
        return {
            "lengths": np.zeros(0),
            "normals": np.zeros((0, 3)),
            "midpoints": np.zeros((0, 3)),
            "categories": 0,
        }
    va, vb = uniq[boundary, 0], uniq[boundary, 1]
    za, zb = vertices[va, 2], vertices[vb, 2]
    on_surface = (np.abs(za) <= tol) & (np.abs(zb) <= tol)
    sel = boundary[on_surface]
    if sel.size == 0:
        return {
            "lengths": np.zeros(0),
            "normals": np.zeros((0, 3)),
            "midpoints": np.zeros((0, 3)),
            "categories": 0,
        }
    pair = uniq[sel]
    lengths = np.linalg.norm(vertices[pair[:, 1]] - vertices[pair[:, 0]], axis=1)
    midpoints = 0.5 * (vertices[pair[:, 0]] + vertices[pair[:, 1]])
    # Adjacent face for each boundary edge: first face containing both vertices.
    face_of_edge = {}
    for face_idx, face in enumerate(faces):
        for i in range(3):
            a = int(face[i])
            b = int(face[(i + 1) % 3])
            if a > b:
                a, b = b, a
            face_of_edge.setdefault((a, b), face_idx)
    adjacent = np.array(
        [face_of_edge[(int(a), int(b))] for a, b in pair], dtype=np.int64
    )
    normals_all, _ = _signed_volume_normals(mesh)
    normals = normals_all[adjacent].copy()
    normals[:, 2] = 0.0
    norm = np.linalg.norm(normals, axis=1)
    normals /= np.where(norm > 0.0, norm, 1.0)[:, None]
    return {
        "lengths": lengths,
        "normals": normals,
        "midpoints": midpoints,
        "adjacent_faces": adjacent,
    }


# ---------------------------------------------------------------------------
# near field
# ---------------------------------------------------------------------------
def near_field_pressure_term(
    areas: FloatArray,
    normals: FloatArray,
    gradient_amplitudes: FloatArray,
    *,
    density: float = _DEFAULT_DENSITY,
) -> FloatArray:
    """Mean second-order pressure integral ``+(1/2) rho Int <|grad phi|^2> n dS``.

    The second-order mean pressure is ``<p^(2)> = -(1/2) rho <|grad phi|^2>``, and
    the force on the body is ``-Int <p^(2)> n dS`` with ``n`` pointing *out of the
    body*, hence the plus sign.  ``gradient_amplitudes`` holds the complex
    amplitude ``grad phi_hat`` per panel; the time average is
    ``<|grad phi|^2> = |grad phi_hat|^2 / 2``, so the factor is ``1/4``.
    """
    grad_amp = np.asarray(gradient_amplitudes)
    sq = np.sum(np.abs(grad_amp) ** 2, axis=1)
    return 0.25 * density * (np.asarray(areas) * sq) @ np.asarray(normals)


def near_field_waterline_term(
    lengths: FloatArray,
    normals: FloatArray,
    elevation_amplitudes: FloatArray,
    *,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> FloatArray:
    """Waterline correction ``-(1/2) rho g Int <zeta^2> n dl``.

    The extra wetted strip ``0 < z < zeta(t)`` on a surface-piercing body feels an
    (almost hydrostatic) pressure that pushes the body *against* the outward
    normal, hence the minus sign.  For a fully reflecting wall this term is
    ``-rho g a^2`` while the pressure term is ``+rho g a^2 / 2``: their sum is
    ``-rho g a^2/2``, i.e. the wall is pushed along the direction of the incident
    wave.
    """
    zeta = np.asarray(elevation_amplitudes)
    sq = np.abs(zeta) ** 2
    return -0.25 * density * gravity * (np.asarray(lengths) * sq) @ np.asarray(normals)


@dataclass
class DriftForce3DResult:
    """Near-field second-order mean drift force on a fixed body."""

    mean_force: FloatArray
    pressure_term: FloatArray
    waterline_term: FloatArray
    wavenumber: float
    amplitude: float
    density: float
    gravity: float
    wave_angle: float
    total_potential_amplitude: FloatArray = field(repr=False, default_factory=lambda: np.zeros(0))
    surface_gradient_amplitude: FloatArray = field(repr=False, default_factory=lambda: np.zeros((0, 3)))
    waterline_elevation_amplitude: FloatArray = field(repr=False, default_factory=lambda: np.zeros(0))
    waterline_count: int = 0
    relative_residual: float = float("nan")

    @property
    def horizontal_magnitude(self) -> float:
        return float(np.hypot(*self.mean_force))


def near_field_mean_drift_3d(
    mesh: SurfaceMesh3D,
    diffraction: Diffraction3DResult,
    *,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> DriftForce3DResult:
    """Pinkster near-field mean drift force for a fixed body in regular waves."""
    if not isinstance(mesh, (TriangleMesh, WettedHullMesh)):
        raise TypeError("mesh must be a TriangleMesh or WettedHullMesh")
    if not isinstance(diffraction, Diffraction3DResult):
        raise TypeError("diffraction must be a Diffraction3DResult")
    if density <= 0.0 or gravity <= 0.0:
        raise ValueError("density and gravity must be positive")
    k = float(diffraction.wavenumber)
    omega = math.sqrt(gravity * k)
    normals, _ = _signed_volume_normals(mesh)

    phi_tot = np.asarray(diffraction.total_potential, dtype=np.complex128)
    grad_t = panel_surface_gradient(mesh, phi_tot)
    # On a fixed body the normal derivative of the total potential vanishes; any
    # residual numerical component along n is removed so it cannot inflate |grad|^2.
    grad_t -= np.einsum("pi,pi->p", grad_t, normals)[:, None] * normals

    pressure = near_field_pressure_term(
        mesh.areas, normals, grad_t, density=density
    )

    wl = waterline_segments(mesh)
    if wl["lengths"].size:
        nodal_phi = _nodal_values(mesh, phi_tot)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        # Map boundary-edge endpoints to vertex indices through the adjacent face.
        adj = wl["adjacent_faces"]
        edge_vertices = []
        for face_idx in adj:
            face = faces[face_idx]
            zs = np.abs(np.asarray(mesh.vertices)[face, 2])
            edge_vertices.append(face[zs <= 1.0e-9 * max(float(mesh.h_max), 1e-12)])
        phi_edge = np.array(
            [
                np.mean(nodal_phi[v]) if v.size else 0.0 + 0.0j
                for v in edge_vertices
            ],
            dtype=np.complex128,
        )
        zeta_edge = 1.0j * omega * phi_edge / gravity
        waterline = near_field_waterline_term(
            wl["lengths"], wl["normals"], zeta_edge, density=density, gravity=gravity
        )
        waterline_amp = zeta_edge
    else:
        waterline = np.zeros(2)
        waterline_amp = np.zeros(0)

    return DriftForce3DResult(
        mean_force=np.asarray(pressure + waterline, dtype=float),
        pressure_term=np.asarray(pressure, dtype=float),
        waterline_term=np.asarray(waterline, dtype=float),
        wavenumber=k,
        amplitude=float(diffraction.amplitude),
        density=float(density),
        gravity=float(gravity),
        wave_angle=float(diffraction.wave_angle),
        total_potential_amplitude=phi_tot,
        surface_gradient_amplitude=grad_t,
        waterline_elevation_amplitude=waterline_amp,
        waterline_count=int(wl["lengths"].size),
        relative_residual=float(diffraction.relative_residual),
    )


# ---------------------------------------------------------------------------
# far field
# ---------------------------------------------------------------------------
def far_field_mean_drift_3d(
    mesh: SurfaceMesh3D,
    diffraction: Diffraction3DResult,
    *,
    density: float = _DEFAULT_DENSITY,
    n_azimuth: int = 360,
    weight: Literal["maruo", "cos_projection"] = "maruo",
) -> dict:
    """Maruo far-field mean drift force from the scattered Kochin function.

    Deep water, fixed body.  With ``K(theta)`` the Kochin function of the
    *scattered* potential (tangent-circle radiation condition already built into
    :func:`~tensorbem.free_surface_solver3d.kochin_function_3d`) the momentum flux
    through a large control cylinder gives

        F_j = (rho k^2 / 8 pi) Int_0^{2 pi} |K(theta)|^2
              [ e_beta_j - (cos theta, sin theta)_j ] dtheta

    i.e. the isotropic part along the direction of wave propagation minus the
    radiated momentum-flux moment.  The weight is frame covariant: rotating the
    incidence angle rotates the force with it (checked by a symmetry test), and a
    body that scatters nothing feels no force.  ``cos_projection`` keeps only the
    second term and is exposed as a diagnostic -- it annihilates any fore-aft
    symmetric scattering pattern, which is the fingerprint of a missing isotropic
    term; the near/far comparison in the verification driver uses it to
    demonstrate that the isotropic term is *required*.
    """
    if not isinstance(mesh, (TriangleMesh, WettedHullMesh)):
        raise TypeError("mesh must be a TriangleMesh or WettedHullMesh")
    if not isinstance(diffraction, Diffraction3DResult):
        raise TypeError("diffraction must be a Diffraction3DResult")
    if density <= 0.0:
        raise ValueError("density must be positive")
    if int(n_azimuth) < 16:
        raise ValueError("n_azimuth must be >= 16")
    k = float(diffraction.wavenumber)
    n_az = int(n_azimuth)
    theta = (2.0 * math.pi / n_az) * np.arange(n_az, dtype=float)
    dtheta = 2.0 * math.pi / n_az
    # Scattered potential; its own normal derivative is dphi_s/dn = -dphi_I/dn
    # because the total normal velocity vanishes on a fixed body.
    kochin = kochin_function_3d(
        mesh,
        diffraction.scattered_potential,
        -np.asarray(diffraction.incident_normal_velocity, dtype=np.complex128),
        k,
        theta,
    )
    pattern = np.abs(kochin) ** 2
    beta = float(diffraction.wave_angle)
    e_beta = np.array([math.cos(beta), math.sin(beta)])
    prefactor = density * k * k / (8.0 * math.pi) * dtheta
    isotropic = prefactor * float(np.sum(pattern)) * e_beta
    projection = prefactor * np.array(
        [
            float(np.sum(pattern * np.cos(theta))),
            float(np.sum(pattern * np.sin(theta))),
        ]
    )
    if weight == "maruo":
        force = isotropic - projection
    elif weight == "cos_projection":
        force = -projection
    else:  # pragma: no cover - guarded by the Literal annotation
        raise ValueError(f"unknown weight {weight!r}")
    return {
        "force": np.asarray(force, dtype=float),
        "kochin": kochin,
        "azimuth": theta,
        "weight": weight,
        "isotropic_term": np.asarray(isotropic, dtype=float),
        "projection_term": np.asarray(projection, dtype=float),
        "wavenumber": k,
        "amplitude": float(diffraction.amplitude),
    }


# ---------------------------------------------------------------------------
# analytic anchors
# ---------------------------------------------------------------------------
def vertical_wall_mean_drift(
    amplitude: float, *, density: float = _DEFAULT_DENSITY, gravity: float = _DEFAULT_GRAVITY
) -> float:
    """Closed-form mean drift force per unit length on a fully reflecting wall."""
    if amplitude <= 0.0:
        raise ValueError("amplitude must be positive")
    return 0.5 * density * gravity * amplitude * amplitude


def vertical_wall_near_field(
    amplitude: float,
    wavenumber: float,
    *,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    n_panels: int = 4001,
    depth_factor: float = 25.0,
) -> dict:
    """Near-field assembly applied to the exact 2-D standing-wave field.

    The wall is at ``x = 0`` with the fluid on ``x > 0``, so the outward body
    normal is ``+x``.  The exact first-order field of a fully reflected Airy wave
    of amplitude ``a`` is ``phi = -(i g a / omega) e^{kz} 2 cos(kx)`` (time
    convention ``exp(-i omega t)``), hence ``dphi/dx = 0`` on the wall and
    ``|dphi/dz| = 2 a omega e^{kz}``.  The free-surface elevation at the wall is
    ``zeta = 2 a`` (antinode).  The two contributions are integrated with the
    very same functions used for the 3-D assembly, so this is a genuine test of
    the coefficient bookkeeping and not a symbolic identity.
    """
    if amplitude <= 0.0 or wavenumber <= 0.0:
        raise ValueError("amplitude and wavenumber must be positive")
    k = float(wavenumber)
    omega = math.sqrt(gravity * k)
    # Panel edges along the wall, graded geometrically towards the free surface.
    edges = np.linspace(0.0, 1.0, n_panels + 1) ** 2  # 0 at the surface
    z_edges = -depth_factor / k * edges
    z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])
    lengths = np.abs(np.diff(z_edges))
    grad_z = 2.0 * amplitude * omega * np.exp(k * z_mid)
    grad = np.zeros((n_panels, 3), dtype=np.complex128)
    grad[:, 2] = grad_z
    normals = np.tile(np.array([[1.0, 0.0, 0.0]]), (n_panels, 1))
    pressure = near_field_pressure_term(
        lengths, normals, grad, density=density
    )
    zeta = np.array([2.0 * amplitude], dtype=np.complex128)
    wl_normals = np.array([[1.0, 0.0, 0.0]])
    waterline = near_field_waterline_term(
        np.array([1.0]), wl_normals, zeta, density=density, gravity=gravity
    )
    signed_exact = np.array(
        [-vertical_wall_mean_drift(amplitude, density=density, gravity=gravity), 0.0]
    )
    return {
        "pressure_term": pressure,
        "waterline_term": waterline,
        "total": pressure + waterline,
        "exact": signed_exact,
        "amplitude": float(amplitude),
        "wavenumber": float(k),
    }


def bottom_mounted_cylinder_mesh(
    radius: float,
    depth: float,
    *,
    n_theta: int = 32,
    n_z: int = 20,
    growth: float = 1.35,
) -> SurfaceMesh3D:
    """Waterline-truncated cylinder of ``radius`` reaching ``z = -depth``.

    The mesh is a :class:`~tensorbem.geometry3d.WettedHullMesh`: side panels with
    geometrically graded vertical spacing (dense near the free surface, where the
    incident potential and its normal velocity ``~ exp(k z)`` are largest), a flat
    bottom cap and an open, outward-oriented waterline loop at ``z = 0``.
    """
    if radius <= 0.0 or depth <= 0.0:
        raise ValueError("radius and depth must be positive")
    if n_theta < 6 or n_z < 2:
        raise ValueError("n_theta >= 6 and n_z >= 2 are required")
    growth = float(growth)
    if growth <= 1.0:
        raise ValueError("growth must exceed 1")
    dz0 = depth * (growth - 1.0) / (growth**n_z - 1.0)
    dz = dz0 * growth ** np.arange(n_z, dtype=float)
    levels = np.concatenate(([0.0], -np.cumsum(dz)))
    theta = 2.0 * math.pi * np.arange(n_theta, dtype=float) / n_theta
    ring = np.stack(
        (radius * np.cos(theta), radius * np.sin(theta)), axis=1
    )
    side = np.array(
        [[x, y, z] for z in levels for (x, y) in ring], dtype=float
    )
    centre = np.array([[0.0, 0.0, levels[-1]]], dtype=float)
    vertices = np.vstack((side, centre))
    centre_index = side.shape[0]
    faces = []
    for i in range(n_z):
        for j in range(n_theta):
            jn = (j + 1) % n_theta
            a = i * n_theta + j
            b = i * n_theta + jn
            c = (i + 1) * n_theta + jn
            d = (i + 1) * n_theta + j
            # Alternate the quad diagonal with (i + j) parity so that the
            # triangulation itself is mirror symmetric about y = 0; an
            # asymmetric pattern would leak a spurious sway force through the
            # gradient recovery.
            if (i + j) % 2 == 0:
                faces.append((a, c, b))
                faces.append((a, d, c))
            else:
                faces.append((a, d, b))
                faces.append((b, d, c))
    base = n_z * n_theta
    for j in range(n_theta):
        jn = (j + 1) % n_theta
        faces.append((centre_index, base + jn, base + j))
    faces = np.asarray(faces, dtype=np.int64)
    return WettedHullMesh(vertices, faces)


def drift_force_3d(
    mesh: SurfaceMesh3D,
    wavenumber: float,
    *,
    amplitude: float = 1.0,
    wave_angle: float = 0.0,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    n_azimuth: int = 360,
    **solver_kwargs,
) -> dict:
    """Convenience wrapper: solve the diffraction problem and return both estimates."""
    diffraction = solve_diffraction_3d(
        mesh,
        wavenumber,
        amplitude=amplitude,
        wave_angle=wave_angle,
        density=density,
        gravity=gravity,
        **solver_kwargs,
    )
    near = near_field_mean_drift_3d(
        mesh, diffraction, density=density, gravity=gravity
    )
    far = far_field_mean_drift_3d(
        mesh, diffraction, density=density, n_azimuth=n_azimuth
    )
    return {"diffraction": diffraction, "near": near, "far": far}


__all__ = [
    "DriftForce3DResult",
    "bottom_mounted_cylinder_mesh",
    "drift_force_3d",
    "far_field_mean_drift_3d",
    "near_field_mean_drift_3d",
    "near_field_pressure_term",
    "near_field_waterline_term",
    "panel_surface_gradient",
    "vertical_wall_mean_drift",
    "vertical_wall_near_field",
    "waterline_segments",
]