"""Recovery of surface derivatives from cell-centred boundary data.

The boundary element solution stores the boundary potential as one value per
panel, but the surface velocity needs the *tangential* derivative of that
potential, and cell-centred samples do not determine a derivative directly.

The recovery fits, over a patch of panels that share a vertex with the panel
of interest, the affine function

    u(x) ~ c + g . (x - x_c),      g in the tangent plane of the panel

to the cell-centred samples by least squares, and returns ``g``.

Measured order: the recovered gradient is **first order** in the panel size for
a smooth field on an unstructured (geodesic) mesh -- observed orders 1.046,
0.987 and 1.009 over 80/320/1280/5120 panels, reproduced on an ellipsoid and
for a second exact field, see ``docs/recovery_robustness.md``.  An earlier
version of this docstring claimed second-order superconvergence; that claim was
never measured in isolation (the 1.67-1.78 order quoted in
``docs/three_dimensional_bem.md`` mixes the recovery error with the flux error
and the O(h**2) geometric error) and is withdrawn.

The recovery is used by
:func:`tensorbem.solver3d.evaluate_surface_velocity_at_panels` to build the
tangential part of the surface velocity.  It is a post-processing step on
the discretised data; it does not change the boundary integral equations.

:func:`patch_fit_system` exposes the least-squares system itself so that
:mod:`tensorbem.recovery_uncertainty` can propagate the data error through the
same design matrix instead of duplicating it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .geometry3d import TriangleMesh

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PatchFitSystem:
    """The dense per-panel least-squares system behind the gradient recovery.

    ``design`` has shape ``(F, W, 4)``: one row per patch sample, columns
    ``(1, dx, dy, dz)`` with the offsets scaled by ``scale``.  ``mask`` is the
    matching 0/1 weight (patches have different sizes, so the ragged block is
    zero-padded), ``index`` the sampled panel ids, and ``scale`` the length
    used to non-dimensionalise the offsets.  ``gradient_block`` is the
    ``(F, 3, 3)`` block of ``(design^T design)^{-1}`` that maps sample noise to
    the slope coefficients.
    """

    index: NDArray[np.int64]
    mask: FloatArray
    design: FloatArray
    scale: float
    normal_derivative_block: FloatArray

    @property
    def panel_count(self) -> int:
        return int(self.design.shape[0])

    @property
    def patch_size(self) -> FloatArray:
        return self.mask.sum(axis=1)


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


def patch_fit_system(
    mesh: TriangleMesh, minimum_patch_size: int = 4
) -> PatchFitSystem:
    """Assemble the per-panel least-squares system used by the recovery.

    Rejects a mesh whose patches are too small to support the four-parameter
    fit, exactly as :func:`recover_tangent_gradient` does.
    """
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
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
    gram = np.einsum("fpi,fpq->fiq", design, design)
    gradient_block = np.linalg.pinv(gram)[:, 1:4, 1:4]
    return PatchFitSystem(
        index=index,
        mask=mask,
        design=design,
        scale=scale,
        normal_derivative_block=gradient_block,
    )


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

    system = patch_fit_system(mesh, minimum_patch_size=minimum_patch_size)
    rhs = samples[system.index] * system.mask

    gram = np.einsum("fpi,fpq->fiq", system.design, system.design)
    moment = np.einsum("fpi,fp->fi", system.design, rhs)
    try:
        solution = np.linalg.solve(gram, moment[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate patch
        solution = np.empty((mesh.panel_count, 4))
        for row in range(mesh.panel_count):
            solution[row] = np.linalg.lstsq(
                system.design[row], rhs[row], rcond=None
            )[0]

    gradient = solution[:, 1:4] / system.scale
    normal = mesh.normals
    gradient = gradient - np.sum(gradient * normal, axis=1)[:, None] * normal
    return gradient