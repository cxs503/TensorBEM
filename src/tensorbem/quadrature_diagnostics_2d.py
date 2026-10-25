"""2D near-singular quadrature diagnostics for boundary integrals.

This module detects integration pairs where the source point is close to the
target panel, which can cause numerical inaccuracies in boundary element
computations. For 2D polygon meshes (PanelMesh).
"""

from __future__ import annotations

import numpy as np

from tensorbem.geometry import PanelMesh


def diagnose_near_singular_quadrature_2d(
    mesh: PanelMesh,
    threshold: float = 4.0,
) -> dict[str, float | int]:
    """Diagnose near-singular integration pairs in 2D boundary mesh.

    For each panel pair (i, j), compute the distance from panel j's centroid
    to panel i. If this distance is less than ``threshold * panel_i_length``,
    the integration is considered near-singular.

    Parameters
    ----------
    mesh : PanelMesh
        2D boundary mesh (polygon with straight panels).
    threshold : float
        Distance threshold as a multiple of panel length. Default 4.0.
        For 2D Laplace problems, the Green's function decays as log(r),
        so interactions within 4 panel lengths require special quadrature
        treatment to maintain accuracy.

    Returns
    -------
    dict
        Diagnostic metrics including:
        - near_singular_count: number of near-singular pairs
        - total_pairs: total number of panel pairs (excluding diagonal)
        - near_singular_ratio: fraction of near-singular pairs
        - max_panel_length: maximum panel length
        - min_panel_length: minimum panel length
        - mean_panel_length: mean panel length
    """
    centers = mesh.centers
    lengths = mesh.lengths
    n_panels = mesh.panel_count

    # Compute pairwise distances between panel centers
    # distances[i, j] = distance from center j to center i
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)

    # Threshold for each panel pair: threshold * length of target panel i
    # For panel i, near-singular if distance to panel j < threshold * lengths[i]
    thresholds = threshold * lengths[:, np.newaxis]

    # Count near-singular pairs (exclude diagonal: i == j)
    near_singular_mask = (distances < thresholds) & ~np.eye(n_panels, dtype=bool)
    near_singular_count = int(np.sum(near_singular_mask))
    total_pairs = n_panels * (n_panels - 1)  # exclude diagonal

    return {
        "near_singular_count": near_singular_count,
        "total_pairs": total_pairs,
        "near_singular_ratio": near_singular_count / total_pairs if total_pairs > 0 else 0.0,
        "max_panel_length": float(np.max(lengths)),
        "min_panel_length": float(np.min(lengths)),
        "mean_panel_length": float(np.mean(lengths)),
        "threshold": threshold,
    }
