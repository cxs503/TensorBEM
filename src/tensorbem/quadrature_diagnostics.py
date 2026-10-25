"""Near-singular quadrature diagnostics for BEM integrals.

Detects source-field point pairs where the integration distance is small
relative to the panel size, indicating potentially inaccurate quadrature.
Reports the fraction of near-singular pairs and estimates the maximum error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .geometry3d import TriangleMesh

FloatArray = NDArray[np.float64]

# Default threshold factor: distance < factor * sqrt(panel_area) is near-singular
_DEFAULT_THRESHOLD_FACTOR = 0.5


@dataclass(frozen=True)
class NearSingularReport:
    """Diagnostics for near-singular quadrature pairs.

    Attributes
    ----------
    mesh_panels : int
        Number of panels in the mesh.
    total_pairs : int
        Total number of source-field point pairs (N*N for self-interaction).
    near_singular_pairs : int
        Number of pairs classified as near-singular.
    near_singular_fraction : float
        Fraction of pairs that are near-singular (0 to 1).
    threshold_factor : float
        The threshold factor used for classification.
    min_distance_ratio : float
        Minimum distance / panel_size ratio across all pairs.
    max_distance_ratio : float
        Maximum distance / panel_size ratio across all pairs.
    estimated_max_error : float
        Estimated maximum quadrature error for near-singular pairs.
    """

    mesh_panels: int
    total_pairs: int
    near_singular_pairs: int
    near_singular_fraction: float
    threshold_factor: float
    min_distance_ratio: float
    max_distance_ratio: float
    estimated_max_error: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary."""
        return {
            "mesh_panels": int(self.mesh_panels),
            "total_pairs": int(self.total_pairs),
            "near_singular_pairs": int(self.near_singular_pairs),
            "near_singular_fraction": float(self.near_singular_fraction),
            "threshold_factor": float(self.threshold_factor),
            "min_distance_ratio": float(self.min_distance_ratio),
            "max_distance_ratio": float(self.max_distance_ratio),
            "estimated_max_error": float(self.estimated_max_error),
        }


def _compute_panel_sizes(mesh: TriangleMesh) -> FloatArray:
    """Compute characteristic size for each panel (sqrt of area)."""
    return np.sqrt(mesh.areas)


def _compute_distance_matrix(
    mesh: TriangleMesh, targets: FloatArray | None = None
) -> FloatArray:
    """Compute pairwise distances between panel centroids (and optional targets).

    Returns matrix of shape (M, N) where M = len(targets) and N = mesh.panel_count.
    If targets is None, computes self-distance matrix (N, N).
    """
    sources = mesh.centroids
    if targets is None:
        targets = sources

    # Compute pairwise distances
    diff = targets[:, np.newaxis, :] - sources[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    return distances


def diagnose_near_singular_quadrature(
    mesh: TriangleMesh,
    targets: FloatArray | None = None,
    threshold_factor: float = _DEFAULT_THRESHOLD_FACTOR,
) -> NearSingularReport:
    """Detect near-singular source-field point pairs.

    A pair is classified as near-singular if the distance between source and
    field points is less than threshold_factor * sqrt(source_panel_area).

    Parameters
    ----------
    mesh : TriangleMesh
        The surface mesh.
    targets : ndarray, optional
        Field points of shape (M, 3). If None, analyzes self-interaction.
    threshold_factor : float
        Threshold factor for near-singular classification.

    Returns
    -------
    NearSingularReport
        Diagnostics for near-singular quadrature.
    """
    n = mesh.panel_count
    panel_sizes = _compute_panel_sizes(mesh)

    if targets is None:
        # Self-interaction analysis
        distances = _compute_distance_matrix(mesh)
        total_pairs = n * n
    else:
        targets = np.asarray(targets, dtype=float)
        if targets.ndim != 2 or targets.shape[1] != 3:
            raise ValueError("targets must have shape (M, 3)")
        distances = _compute_distance_matrix(mesh, targets)
        total_pairs = targets.shape[0] * n

    # Threshold for each source panel
    thresholds = threshold_factor * panel_sizes  # shape (N,)

    # Broadcast to compare: distances[i, j] < thresholds[j]
    is_near_singular = distances < thresholds[np.newaxis, :]

    near_singular_count = int(np.sum(is_near_singular))
    near_singular_fraction = near_singular_count / total_pairs

    # Distance ratios (distance / panel_size)
    distance_ratios = distances / panel_sizes[np.newaxis, :]

    # Exclude self-interaction diagonal for min ratio (it's zero by definition)
    if targets is None:
        # For self-interaction, exclude diagonal
        mask = ~np.eye(n, dtype=bool)
        min_ratio = float(np.min(distance_ratios[mask]))
        max_ratio = float(np.max(distance_ratios))
    else:
        min_ratio = float(np.min(distance_ratios))
        max_ratio = float(np.max(distance_ratios))

    # Estimate maximum error for near-singular pairs
    # Error scales as O(h^2 / d^2) for smooth kernels, where h is panel size
    # and d is distance. For near-singular pairs, this can be large.
    # Exclude self-interaction (diagonal) from error estimation.
    if targets is None:
        # Self-interaction: exclude diagonal
        mask = ~np.eye(n, dtype=bool)
        near_singular_mask = is_near_singular & mask
        if np.any(near_singular_mask):
            near_ratios = distance_ratios[near_singular_mask]
            # Avoid division by zero
            near_ratios = np.maximum(near_ratios, 1e-10)
            # Error estimate: 1/ratio^2 (rough scaling for gradient of 1/r kernel)
            error_estimates = 1.0 / (near_ratios**2)
            estimated_max_error = float(np.max(error_estimates))
        else:
            estimated_max_error = 0.0
    else:
        # External targets: no self-interaction
        if near_singular_count > 0:
            near_ratios = distance_ratios[is_near_singular]
            near_ratios = np.maximum(near_ratios, 1e-10)
            error_estimates = 1.0 / (near_ratios**2)
            estimated_max_error = float(np.max(error_estimates))
        else:
            estimated_max_error = 0.0

    return NearSingularReport(
        mesh_panels=n,
        total_pairs=total_pairs,
        near_singular_pairs=near_singular_count,
        near_singular_fraction=near_singular_fraction,
        threshold_factor=threshold_factor,
        min_distance_ratio=min_ratio,
        max_distance_ratio=max_ratio,
        estimated_max_error=estimated_max_error,
    )


def near_singular_by_panel(
    mesh: TriangleMesh,
    threshold_factor: float = _DEFAULT_THRESHOLD_FACTOR,
) -> tuple[FloatArray, FloatArray]:
    """Count near-singular pairs for each source panel.

    Returns
    -------
    counts : ndarray
        Number of near-singular field points for each source panel.
    fractions : ndarray
        Fraction of field points that are near-singular for each source panel.
    """
    n = mesh.panel_count
    panel_sizes = _compute_panel_sizes(mesh)
    distances = _compute_distance_matrix(mesh)
    thresholds = threshold_factor * panel_sizes

    is_near_singular = distances < thresholds[np.newaxis, :]

    # Count per source panel (column-wise)
    counts = np.sum(is_near_singular, axis=0)
    fractions = counts / n

    return counts, fractions


def identify_problematic_panels(
    mesh: TriangleMesh,
    threshold_factor: float = _DEFAULT_THRESHOLD_FACTOR,
    min_near_singular_fraction: float = 0.1,
) -> list[int]:
    """Identify panels with many near-singular interactions.

    Returns indices of panels where more than min_near_singular_fraction
    of all other panels are classified as near-singular.
    """
    _, fractions = near_singular_by_panel(mesh, threshold_factor)
    problematic = np.where(fractions > min_near_singular_fraction)[0]
    return problematic.tolist()
