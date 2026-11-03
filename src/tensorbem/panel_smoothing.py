"""Smoothing of cell-centred panel data before tangential-gradient recovery.

``docs/three_dimensional_bem.md`` limitation 9 records one open prerequisite
for the surface-velocity recovery path: the recovery assumes the boundary
trace is smooth, while a real solve returns one value per panel.  This module
implements the preprocessing that limitation asks for -- an area-weighted
panel -> vertex -> panel averaging -- and :mod:`tensorbem.recovery_uncertainty`
quantifies what it buys and what it cannot buy.

The operator is the standard mass-lumped Galerkin smoother on a triangle mesh:
a vertex value is the area-weighted mean of its incident panels, a panel value
is the mean of its three vertex values.  One iteration is the identity when
the input is already the trace of a smooth field, and it attenuates
panel-to-panel (high-frequency) components of the input.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .geometry3d import TriangleMesh
from .recovery import recover_tangent_gradient

FloatArray = NDArray[np.float64]

_WEIGHTINGS = ("area", "uniform")


def _check_values(mesh: TriangleMesh, values: FloatArray) -> FloatArray:
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    samples = np.asarray(values, dtype=float)
    if samples.shape != (mesh.panel_count,):
        raise ValueError("values must have one entry per panel")
    if not np.isfinite(samples).all():
        raise ValueError("values must be finite")
    return samples


def panel_to_vertex_field(
    mesh: TriangleMesh, values: FloatArray, weighting: str = "area"
) -> FloatArray:
    """Average the panel values onto the vertices, incident-panel weighted.

    Returns an array of shape ``(V,)``.  Every vertex of a closed mesh is
    incident to at least three panels, so the average is always defined.
    """
    samples = _check_values(mesh, values)
    if weighting not in _WEIGHTINGS:
        raise ValueError(f"weighting must be one of {_WEIGHTINGS}")
    if weighting == "area":
        weights = np.repeat(mesh.areas, 3)
    else:
        weights = np.ones(3 * mesh.panel_count)
    flat = mesh.faces.reshape(-1)
    numerator = np.zeros(mesh.vertex_count)
    denominator = np.zeros(mesh.vertex_count)
    np.add.at(numerator, flat, np.repeat(samples, 3) * weights)
    np.add.at(denominator, flat, weights)
    if np.any(denominator <= 0.0):  # pragma: no cover - closed mesh guarantees this
        raise ValueError("a vertex without incident panels cannot be averaged")
    return numerator / denominator


def vertex_to_panel_field(mesh: TriangleMesh, vertex_values: FloatArray) -> FloatArray:
    """Sample a vertex field at the panel centroids (mean of the three corners)."""
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    nodes = np.asarray(vertex_values, dtype=float)
    if nodes.shape != (mesh.vertex_count,):
        raise ValueError("vertex_values must have one entry per vertex")
    if not np.isfinite(nodes).all():
        raise ValueError("vertex_values must be finite")
    return nodes[mesh.faces].mean(axis=1)


def smooth_panel_field(
    mesh: TriangleMesh,
    values: FloatArray,
    iterations: int = 1,
    weighting: str = "area",
) -> FloatArray:
    """Apply ``iterations`` rounds of panel -> vertex -> panel averaging."""
    if isinstance(iterations, (bool, np.bool_)) or not isinstance(
        iterations, (int, np.integer)
    ):
        raise ValueError("iterations must be a non-negative integer")
    if iterations < 0:
        raise ValueError("iterations must be a non-negative integer")
    field = _check_values(mesh, values).copy()
    for _ in range(int(iterations)):
        field = vertex_to_panel_field(
            mesh, panel_to_vertex_field(mesh, field, weighting=weighting)
        )
    return field


def smooth_and_recover_tangent_gradient(
    mesh: TriangleMesh,
    values: FloatArray,
    iterations: int = 1,
    weighting: str = "area",
) -> FloatArray:
    """Smooth the panel data, then recover the tangential gradient."""
    return recover_tangent_gradient(
        mesh, smooth_panel_field(mesh, values, iterations, weighting)
    )