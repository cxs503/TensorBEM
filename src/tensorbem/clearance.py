"""Geometry-aware clearance diagnostics for 3-D surface evaluations.

The fixed-order field quadrature has a hard minimum clearance. Velocity
accuracy can require a more conservative O(h) exclusion layer, so callers
can diagnose and record point-to-surface distances before trusting results.
This module does not claim that a point outside the warning layer is accurate.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from numpy.typing import NDArray

from .geometry3d import TriangleMesh
from .solver3d import _minimum_distance_to_mesh

FloatArray = NDArray[np.float64]
_POLICIES = ("warn", "raise", "ignore")


@dataclass(frozen=True)
class SurfaceClearanceReport:
    """Per-point clearance diagnostics, suitable for JSON serialization."""

    minimum_distance: FloatArray
    clearance_ratio: FloatArray
    threshold_distance: float
    h_max: float
    warning_factor: float
    unsafe_indices: tuple[int, ...]
    policy: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly report without NumPy scalar types."""
        return {
            "minimum_distance": self.minimum_distance.tolist(),
            "clearance_ratio": self.clearance_ratio.tolist(),
            "threshold_distance": float(self.threshold_distance),
            "h_max": float(self.h_max),
            "warning_factor": float(self.warning_factor),
            "unsafe_indices": list(self.unsafe_indices),
            "unsafe_count": len(self.unsafe_indices),
            "point_count": int(self.minimum_distance.size),
            "policy": self.policy,
        }


def diagnose_surface_clearance(
    mesh: TriangleMesh,
    points: FloatArray,
    *,
    warning_factor: float = 1.0,
    policy: str = "warn",
) -> SurfaceClearanceReport:
    """Measure each point's distance from a closed triangular surface.

    clearance_ratio is distance divided by mesh.h_max. Points whose ratio is
    less than warning_factor are considered inside the caution layer. Policy
    may be warn, raise, or ignore. This is a trust diagnostic, not an accuracy
    certificate. Field evaluators still reject points inside their own
    minimum-clearance threshold for fixed-order quadrature.
    """
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    targets = np.asarray(points, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("points must have shape (M, 3)")
    if not np.isfinite(targets).all():
        raise ValueError("points must all be finite")
    if not np.isfinite(warning_factor) or warning_factor < 0.0:
        raise ValueError("warning_factor must be finite and non-negative")
    if policy not in _POLICIES:
        raise ValueError(f"policy must be one of {_POLICIES}")

    distances = np.asarray(
        [_minimum_distance_to_mesh(mesh, point) for point in targets], dtype=float
    )
    ratios = distances / mesh.h_max
    unsafe = tuple(int(index) for index in np.flatnonzero(ratios < warning_factor))
    report = SurfaceClearanceReport(
        minimum_distance=distances,
        clearance_ratio=ratios,
        threshold_distance=float(warning_factor * mesh.h_max),
        h_max=float(mesh.h_max),
        warning_factor=float(warning_factor),
        unsafe_indices=unsafe,
        policy=policy,
    )
    if unsafe and policy == "raise":
        raise ValueError(
            f"{len(unsafe)} evaluation point(s) lie inside the configured "
            f"clearance layer (distance < {report.threshold_distance:.6g}, "
            f"warning_factor={warning_factor:g} * h_max)"
        )
    if unsafe and policy == "warn":
        warnings.warn(
            f"{len(unsafe)} evaluation point(s) lie inside the configured "
            "O(h) clearance layer; velocity accuracy is not guaranteed there",
            RuntimeWarning,
            stacklevel=2,
        )
    return report
