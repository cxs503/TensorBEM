"""Superconvergent recovery of a surface gradient from cell-centred data.

The three-dimensional solver stores its unknowns at panel centroids.  For a
Dirichlet problem the potential on the boundary is known as one value per
panel, but the surface velocity needs the *tangential* derivative of that
potential, and cell-centred samples do not determine a derivative directly.

The recovery fits, over a patch of panels that share a vertex with the panel
of interest, the affine function

    u(x) ~ c + g . (x - x_c),      g in the tangent plane of the panel

to the cell-centred samples by least squares, and returns ``g``.  The panel
centroid is a superconvergence point of that fit for a smooth field, so the
recovered gradient is second-order accurate in the panel size.

The recovery is used by
:func:`tensorbem.solver3d.evaluate_surface_velocity_at_panels` to build the
tangential part of the surface velocity.  It is a post-processing step on
the discretised data; it does not change the boundary integral equations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .geometry3d import TriangleMesh

FloatArray = NDArray[np.float64]


def vertex_incident_panels(mesh: TriangleMesh) -> list[list[int]]:
    """Return, for every vertex, the panels that use it (ascending vertex order)."""
    incident: list[list[int]] = [[] for _ in range(mesh.vertex_count)]
    for panel, face in enumerate(mesh.faces):
        for vertex in face:
            incident[int(vertex)].append(panel)
    return incident


def panel_patches(mesh: TriangleMesh) -> list[list[int]]:
    """Return, for every panel, the panels sharing at least one of its vertices."""
    incident = vertex_incident_panels(mesh)
    patches: list[list[int]] = []
    for face in mesh.faces:
        neighbours: set[int] = set()
        for vertex in face:
            neighbours.update(incident[int(vertex)])
        patches.append(sorted(neighbours))
    return patches


def tangent_basis(normals: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return two orthonormal vectors spanning the tangent plane of each panel.

    The helper axis is chosen per panel as the coordinate axis least aligned
    with the normal, so the cross product is never degenerate.
    """
    normals = np.asarray(normals, dtype=float)
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError("normals must have shape (F, 3)")
    helper = np.zeros_like(normals)
    helper[np.arange(normals.shape[0]), np.argmin(np.abs(normals), axis=1)] = 1.0
    first = np.cross(helper, normals)
    norms = np.linalg.norm(first, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("normals must be non-zero")
    first = first / norms[:, None]
    second = np.cross(normals, first)
    return first, second


def recover_tangent_gradient(
    mesh: TriangleMesh, values: FloatArray, minimum_patch_size: int = 4
) -> FloatArray:
    """Recover the tangential gradient of a cell-centred field at the centroids.

    Parameters
    ----------
    mesh:
        The closed triangulation the samples belong to.
    values:
        One value per panel, in the panel order of ``mesh``.
    minimum_patch_size:
        Smallest accepted patch.  A patch with fewer samples than this cannot
        support the least-squares fit and is rejected.

    Returns
    -------
    numpy.ndarray
        Tangential gradient at each panel centroid, shape ``(F, 3)``.  The
        component along the panel normal is zero by construction.
    """
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    samples = np.asarray(values, dtype=float)
    if samples.shape != (mesh.panel_count,):
        raise ValueError("values must have one entry per panel")
    if not np.isfinite(samples).all():
        raise ValueError("values must be finite")
    if minimum_patch_size < 3:
        raise ValueError("minimum_patch_size must be at least 3")

    patches = panel_patches(mesh)
    width = max((len(patch) for patch in patches), default=0)
    if width < minimum_patch_size:
        raise ValueError(
            "the mesh is too coarse for a patch recovery: the largest patch has "
            f"{width} panels, at least {minimum_patch_size} are required"
        )
    for index, patch in enumerate(patches):
        if len(patch) < minimum_patch_size:
            raise ValueError(
                f"panel {index} has a patch of {len(patch)} panels, at least "
                f"{minimum_patch_size} are required"
            )

    centroids = mesh.centroids
    first, second = tangent_basis(mesh.normals)
    scale = max(mesh.h_mean, np.finfo(float).tiny)

    index = np.zeros((mesh.panel_count, width), dtype=np.int64)
    mask = np.zeros((mesh.panel_count, width), dtype=float)
    for row, patch in enumerate(patches):
        index[row, : len(patch)] = patch
        mask[row, : len(patch)] = 1.0

    offsets = centroids[index] - centroids[:, None, :]
    # The fit is done in three dimensions, not in the tangent plane: the
    # samples of a curved surface leave the panel plane by O(h**2), and
    # projecting them onto the tangent plane before the fit biases the
    # recovered gradient by an amount that does not vanish with the panel
    # size.  The recovered gradient is projected afterwards.
    constant = np.ones((mesh.panel_count, width))
    design = np.stack(
        [constant * mask, (offsets[..., 0] / scale) * mask,
         (offsets[..., 1] / scale) * mask, (offsets[..., 2] / scale) * mask],
        axis=2,
    )
    rhs = samples[index] * mask

    gram = np.einsum("fpi,fpq->fiq", design, design)
    moment = np.einsum("fpi,fp->fi", design, rhs)
    try:
        solution = np.linalg.solve(gram, moment[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate patch
        solution = np.empty((mesh.panel_count, 4))
        for row in range(mesh.panel_count):
            solution[row] = np.linalg.lstsq(
                design[row], rhs[row], rcond=None
            )[0]

    gradient = solution[:, 1:4] / scale
    normal = mesh.normals
    gradient = gradient - np.sum(gradient * normal, axis=1)[:, None] * normal
    return gradient
