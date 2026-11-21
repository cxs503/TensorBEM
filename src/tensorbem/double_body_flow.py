"""Steady hull-tangent flow and rigid-mode boundary forcing on a mirrored hull."""

from dataclasses import dataclass

import numpy as np

from .geometry3d import WettedHullMesh
from .multibody3d import mirror_closure, static_added_mass


@dataclass(frozen=True)
class DoubleBodyFlow:
    """Unit-speed solution for the supplied wetted mesh advancing along +x.

    Translation and rotation m-terms use moments about the coordinate origin.
    Scale velocity and m-terms linearly by the ship speed.
    """

    mesh: WettedHullMesh
    velocity: np.ndarray
    translation_m: np.ndarray
    rotation_m: np.ndarray
    relative_residual: float
    condition_number: float
    max_patch_condition: float


def _surface_patch_neighbors(mesh):
    edges = mesh.faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    keys = np.sort(edges, axis=1)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    first, second = order[::2], order[1::2]
    if not np.array_equal(edges[first], edges[second, ::-1]):
        raise ValueError("expected a closed two-manifold surface")
    adjacent = np.empty((mesh.panel_count, 3), dtype=int)
    adjacent.flat[first] = second // 3
    adjacent.flat[second] = first // 3
    # Use one-sided patches on either side of a sharp chine or stem.
    current = np.arange(mesh.panel_count)[:, None]
    cos_angle = np.einsum("ik,ijk->ij", mesh.normals, mesh.normals[adjacent])
    return np.where(cos_angle >= np.cos(np.pi / 4), adjacent, current)


def _surface_jet(offsets, values, basis):
    """First and second derivatives in a locally scaled tangent-plane chart."""
    uv = offsets @ basis
    scale = np.max(np.abs(uv), axis=0)
    if np.any(scale == 0):
        raise ValueError("section of hull has a degenerate surface patch")
    u, v = (uv / scale).T
    design = np.column_stack((np.ones_like(u), u, v, u * u / 2, u * v, v * v / 2))
    weights = 1 / (0.35 + u * u + v * v)
    coef, _, rank, sv = np.linalg.lstsq(
        weights[:, None] * design, weights * values, rcond=None
    )
    if rank < 6:
        raise ValueError(
            "refine the hull to resolve a quadratic patch on each smooth surface"
        )
    curvature = np.array([[coef[3], coef[4]], [coef[4], coef[5]]]) / np.outer(
        scale, scale
    )
    return coef[1:3] / scale, curvature, sv[0] / sv[-1]


def _surface_kinematics(mesh, disturbance, wetted_normals):
    """Recover div_s[(d·n)V] from potential and geometry on local surface charts.

    For rigid translations the m-vector is (V·grad_s)n + n div_s(V).
    Rotations add the derivative of the moment arm along V. For tangential
    potential flow div_s(V) is the surface Laplacian of the potential.
    """
    count = len(wetted_normals)
    adjacent = _surface_patch_neighbors(mesh)
    velocity, translation, rotation = (np.empty((count, 3)) for _ in range(3))
    max_condition = 0.0
    for i in range(count):
        patch = np.array([i])
        while patch.size < 24:
            larger = np.unique(np.r_[patch, adjacent[patch].ravel()])
            if larger.size == patch.size:
                break
            patch = larger
        n = mesh.normals[i]
        first = np.cross(n, np.eye(3)[np.argmin(np.abs(n))])
        first /= np.linalg.norm(first)
        basis = np.column_stack((first, np.cross(n, first)))
        center = mesh.centroids[i]
        offsets = mesh.vertices[np.unique(mesh.faces[patch])] - center
        dz, H_z, cond_z = _surface_jet(offsets, offsets @ n, basis)
        dp, H_p, cond_p = _surface_jet(
            mesh.centroids[patch] - center, disturbance[patch] - disturbance[i], basis
        )
        max_condition = max(max_condition, cond_z, cond_p)
        # Total potential is the exact linear incident field -x plus disturbance.
        dp -= basis[0] + n[0] * dz
        H_p -= n[0] * H_z
        inverse_metric = np.eye(2) - np.outer(dz, dz) / (1 + dz @ dz)
        rates = inverse_metric @ dp
        v = basis @ rates + n * (dz @ rates)
        normal = n - basis @ dz
        length = np.linalg.norm(normal)
        normal /= length
        dn = -(basis @ (H_z @ rates)) / length
        dn -= normal * (normal @ dn)
        laplacian = np.sum(inverse_metric * (H_p - (rates @ dz) * H_z))
        translation[i] = dn + normal * laplacian
        rotation[i] = np.cross(center, translation[i]) + np.cross(v, normal)
        # Mirror welding rounds vertex positions. Project onto the actual BVP panel.
        panel_normal = wetted_normals[i]
        velocity[i] = v - panel_normal * (panel_normal @ v)
    return velocity, translation, rotation, max_condition


def prepare_double_body_flow(
    mesh: WettedHullMesh, *, quadrature_order=5
) -> DoubleBodyFlow:
    """Solve the steady rigid-lid Neumann problem once, before a frequency sweep.

    An even image of the hull makes the z=0 plane impermeable. The disturbance
    cancels the incident velocity -e_x through the surface. All derivatives
    are recovered on the closed double body, including at its waterline.
    """
    solution = static_added_mass(mesh, modes=[0], quadrature_order=quadrature_order)
    closed = mirror_closure(mesh)
    trace = np.tile(solution.potentials[:, 0], 2)
    velocity, trans, rot, condition = _surface_kinematics(
        closed, trace, mesh.normals
    )
    return DoubleBodyFlow(
        mesh,
        velocity,
        trans,
        rot,
        solution.relative_residual,
        solution.condition_number,
        condition,
    )
