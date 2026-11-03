"""Data-driven uncertainty estimate for the recovered tangential gradient.

The recovery in :mod:`tensorbem.recovery` is a four-parameter least-squares
plane fit over a patch of panels, so a panel-to-panel error of size ``sigma``
in the boundary values becomes a gradient error of size ``sigma / h`` times a
purely geometric factor.  That amplification is the reason
``docs/three_dimensional_bem.md`` limitation 9 asks for smoothing, but the
amplification factor was never quantified and the documentation gave no way
to tell, from a computed solution, whether the surface velocity can be
trusted.

This module closes both gaps.  :func:`estimate_recovery_uncertainty` reuses
the exact design matrix of the recovery (:func:`tensorbem.recovery.patch_fit_system`)
and returns

* ``sigma_hat`` -- the per-panel residual estimate of the non-smooth part of
  the data, ``sqrt(RSS / (m - 4))`` for a patch of ``m`` panels;
* ``predicted_error`` -- the per-panel standard deviation of the recovered
  tangential gradient that this ``sigma_hat`` implies, propagated through
  ``(X^T X)^{-1}`` restricted to the slope block and projected on the tangent
  plane;
* the aggregate ``predicted_rms``, which is directly comparable to an L2 error
  measured against an exact solution.

Nothing here needs the exact solution: the estimate is computable from the
computed boundary values alone, which is what makes it usable as a trust
gate before the surface velocity is fed into a force integration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .geometry3d import TriangleMesh
from .recovery import patch_fit_system, tangent_basis

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RecoveryUncertainty:
    """Per-panel and aggregate uncertainty of a recovered tangential gradient."""

    sigma_hat: FloatArray
    predicted_error: FloatArray
    patch_size: FloatArray

    @property
    def predicted_rms(self) -> float:
        """Root-mean-square predicted gradient error over the panels."""
        return float(np.sqrt(np.mean(self.predicted_error**2)))

    @property
    def sigma_rms(self) -> float:
        """Root-mean-square of the residual-based data error estimate."""
        return float(np.sqrt(np.mean(self.sigma_hat**2)))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly summary (no per-panel arrays)."""
        return {
            "predicted_rms": self.predicted_rms,
            "sigma_rms": self.sigma_rms,
            "sigma_max": float(np.max(self.sigma_hat)),
            "predicted_max": float(np.max(self.predicted_error)),
            "panels": int(self.sigma_hat.size),
            "mean_patch_size": float(np.mean(self.patch_size)),
            "min_patch_size": float(np.min(self.patch_size)),
        }


def estimate_recovery_uncertainty(
    mesh: TriangleMesh, values: FloatArray, minimum_patch_size: int = 4
) -> RecoveryUncertainty:
    """Estimate the gradient error the data can support, without an exact solution.

    Parameters
    ----------
    mesh:
        The closed triangulation the samples belong to.
    values:
        One value per panel, as returned by a boundary element solve.
    minimum_patch_size:
        Forwarded to :func:`tensorbem.recovery.patch_fit_system`.

    Returns
    -------
    RecoveryUncertainty
        Per-panel ``sigma_hat`` and ``predicted_error`` plus their aggregates.

    Notes
    -----
    The residual is inflated by the *smooth* part of the discretisation error
    (a curved surface is fitted by a plane, so even an exact trace leaves an
    ``O(h**2)`` residual), therefore the estimate is conservative for perfect
    data and asymptotically tight for data whose error is panel-to-panel.
    """
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    samples = np.asarray(values, dtype=float)
    if samples.shape != (mesh.panel_count,):
        raise ValueError("values must have one entry per panel")
    if not np.isfinite(samples).all():
        raise ValueError("values must be finite")

    system = patch_fit_system(mesh, minimum_patch_size=minimum_patch_size)
    design = system.design
    rhs = samples[system.index] * system.mask

    gram = np.einsum("fpi,fpq->fiq", design, design)
    moment = np.einsum("fpi,fp->fi", design, rhs)
    solution = np.linalg.solve(gram, moment[:, :, None])[:, :, 0]

    fitted = np.einsum("fpi,fi->fp", design, solution)
    residual = (rhs - fitted) * system.mask
    patch = system.patch_size
    degrees = np.maximum(patch - 4.0, 1.0)
    sigma_hat = np.sqrt(np.einsum("fp,fp->f", residual, residual) / degrees)

    first, second = tangent_basis(mesh.normals)
    basis = np.stack([first, second], axis=2)  # (F, 3, 2)
    block = system.normal_derivative_block  # (F, 3, 3)
    # trace(T^T B T) with T = [t1 t2], written as the Frobenius product of B
    # with the tangent-plane projector T T^T (avoids a 3-operand einsum).
    projector = np.einsum("fai,fbi->fab", basis, basis)
    trace = np.einsum("fab,fab->f", block, projector)
    predicted = sigma_hat / system.scale * np.sqrt(np.maximum(trace, 0.0))
    return RecoveryUncertainty(
        sigma_hat=sigma_hat,
        predicted_error=predicted,
        patch_size=patch,
    )