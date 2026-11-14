"""Multipole (Fourier) decomposition of a 2-D exterior harmonic field.

Why this module exists
----------------------
:func:`tensorbem.solver.evaluate_exterior_field` returns point values of the
exterior potential and its gradient.  Several far-field routes want the
*multipole content* of that field instead of point values:

* far-field added mass / damped-radiation energy, where the outgoing wave is
  characterised by its dipole and higher harmonics;
* the drift force of a small body in a slowly varying ambient flow, which
  depends on the dipole (and quadrupole) of the disturbance potential;
* any convergence study in which "the field is right" has to be stated as
  "the harmonic coefficients are right and the higher ones vanish", which is a
  far sharper statement than a pointwise L2 error.

On a circle of fixed radius the functions ``r**(-n) cos(n theta)`` and
``r**(-n) sin(n theta)`` are orthogonal in ``theta``, so one ring already
separates every decaying harmonic.  The single term a ring cannot separate from
the mean potential is the 2-D source ``m log(r)``, because it is constant along
the ring: :func:`decompose_rings` therefore takes fits on two or more rings and
reports the source strength as the slope of the constant term against
``log r``.  For a solenoidal (net-flux-free) exterior field that slope is zero,
so the same number doubles as a conservation-leak detector.

Convention
----------
::

    phi(r, theta) = c + m log(r) + sum_{n>=1} (A_n cos(n theta)
                                               + B_n sin(n theta)) / r**n

with ``r`` and the coefficients carrying whatever units the caller uses; the
decomposition itself is dimensionless.  For the cylinder of
``CASE-BEM-LAPLACE-CYLINDER`` (``W(z) = U (z + a**2/z)``) the coefficients are
``A_1 = U a**2`` and zero for every other ``n``; the case reports them scaled by
``U a**(n+1)``.

Both functions validate their input: a silent ``lstsq`` on a rank-deficient
basis returns a plausible-looking split between harmonics that a caller has no
way of auditing afterwards, so the basis conditioning and the ring count are
checked instead of trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# A ring fit is a small linear least-squares problem; conditioning above this
# means the caller asked for more harmonics than the sampling can support.
_MAX_FIT_CONDITION = 1.0e10


def ring_angles(count: int, offset: float = 0.5) -> FloatArray:
    """``count`` equally spaced angles, rotated by ``offset`` cell widths.

    The default half-cell offset keeps the samples off the vertices of a
    regular polygonal body, where the boundary data of a panel discretisation
    is least smooth.
    """
    if not isinstance(count, (int, np.integer)):
        raise TypeError("count must be an integer")
    if count < 2:
        raise ValueError("at least two angles are required to fit a harmonic")
    if not np.isfinite(offset):
        raise ValueError("offset must be finite")
    return 2.0 * math.pi * (np.arange(count) + offset) / count


@dataclass(frozen=True)
class RingFit:
    """Harmonic amplitudes of a scalar field sampled on one circle.

    ``cos_coefficients[n]`` / ``sin_coefficients[n]`` are ``A_n`` / ``B_n`` in
    the convention of the module docstring; index 0 is padding kept at zero so
    that an index is the harmonic number.  ``constant`` is the ring mean (``c``).
    """

    radius: float
    order: int
    angles: FloatArray
    values: FloatArray
    constant: float
    cos_coefficients: FloatArray
    sin_coefficients: FloatArray
    rms_residual: float
    condition_number: float

    def amplitude(self, n: int) -> float:
        """Magnitude ``sqrt(A_n**2 + B_n**2) / radius**n`` sampled on this ring.

        The fit returns the coefficients of ``cos(n theta)`` itself, so the
        ``r**(-n)`` factor of the multipole is already folded in; use
        :meth:`strength` for the radius-independent coefficient ``A_n``.
        """
        self._validate_order_index(n)
        return float(math.hypot(self.cos_coefficients[n], self.sin_coefficients[n]))

    def strength(self, n: int) -> float:
        """Radius-independent multipole strength ``sqrt(A_n**2 + B_n**2)``."""
        return self.amplitude(n) * self.radius**n

    def cos_strength(self, n: int) -> float:
        """Signed radius-independent cosine coefficient ``A_n`` of harmonic ``n``.

        :meth:`amplitude` and :meth:`strength` are magnitudes and therefore hide
        the sign: a dipole and its image have the same amplitude.  Any check that
        has to tell the two apart (a negative control, say) must use the signed
        coefficient instead.
        """
        self._validate_order_index(n)
        return float(self.cos_coefficients[n] * self.radius**n)

    def sin_strength(self, n: int) -> float:
        """Signed radius-independent sine coefficient ``B_n`` of harmonic ``n``."""
        self._validate_order_index(n)
        return float(self.sin_coefficients[n] * self.radius**n)

    def phase(self, n: int) -> float:
        """Azimuth ``theta_n`` such that harmonic ``n`` is ``amplitude*cos(n(theta-theta_n))``."""
        self._validate_order_index(n)
        return float(math.atan2(self.sin_coefficients[n], self.cos_coefficients[n]) / n)

    @property
    def relative_residual(self) -> float:
        """``rms_residual`` divided by the largest sampled magnitude."""
        scale = float(np.max(np.abs(self.values)))
        if scale == 0.0:
            return 0.0 if self.rms_residual == 0.0 else float("inf")
        return self.rms_residual / scale

    def _validate_order_index(self, n: int) -> None:
        if not isinstance(n, (int, np.integer)):
            raise TypeError("harmonic number must be an integer")
        if n < 1 or n > self.order:
            raise ValueError(f"harmonic number must lie in [1, {self.order}]")


def fit_ring(
    values: FloatArray,
    radius: float,
    angles: FloatArray,
    order: int = 4,
) -> RingFit:
    """Least-squares harmonic fit of ``values`` sampled at ``angles`` on ``radius``."""
    if not isinstance(order, (int, np.integer)):
        raise TypeError("order must be an integer")
    if order < 1:
        raise ValueError("order must be at least 1 (a dipole term is required)")
    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    samples = np.asarray(values, dtype=float)
    theta = np.asarray(angles, dtype=float)
    if samples.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if theta.ndim != 1:
        raise ValueError("angles must be one-dimensional")
    if samples.shape != theta.shape:
        raise ValueError("values and angles must have the same length")
    if samples.size < 2 * order + 1:
        raise ValueError(
            f"a fit of order {order} needs at least {2 * order + 1} samples, "
            f"got {samples.size}"
        )
    if not np.isfinite(samples).all():
        raise ValueError("values must all be finite")
    if not np.isfinite(theta).all():
        raise ValueError("angles must all be finite")

    columns = [np.ones_like(theta)]
    for n in range(1, order + 1):
        columns.append(np.cos(n * theta))
    for n in range(1, order + 1):
        columns.append(np.sin(n * theta))
    basis = np.column_stack(columns)
    condition_number = float(np.linalg.cond(basis))
    if not np.isfinite(condition_number) or condition_number > _MAX_FIT_CONDITION:
        raise ValueError(
            "the harmonic basis is numerically rank deficient for this sampling "
            f"(condition number {condition_number:.3e} > {_MAX_FIT_CONDITION:.1e}); "
            "use more samples or a lower order"
        )
    coefficients, *_ = np.linalg.lstsq(basis, samples, rcond=None)
    model = basis @ coefficients
    rms_residual = float(np.sqrt(np.mean((samples - model) ** 2)))
    cos_coefficients = np.zeros(order + 1)
    sin_coefficients = np.zeros(order + 1)
    cos_coefficients[1:] = coefficients[1 : order + 1]
    sin_coefficients[1:] = coefficients[order + 1 :]
    return RingFit(
        radius=radius,
        order=int(order),
        angles=theta,
        values=samples,
        constant=float(coefficients[0]),
        cos_coefficients=cos_coefficients,
        sin_coefficients=sin_coefficients,
        rms_residual=rms_residual,
        condition_number=condition_number,
    )


def decompose_rings(fits: "list[RingFit] | tuple[RingFit, ...]") -> dict[str, object]:
    """Combine ring fits into a multipole spectrum plus a source-leak estimate.

    The harmonics of a valid exterior field do not depend on the radius at which
    they are measured; the spread across rings is reported because it is exactly
    what a mesh-induced non-harmonic contamination shows up as.  The source
    strength is the least-squares slope of the ring mean against ``log r``.
    """
    fits = list(fits)
    if len(fits) < 2:
        raise ValueError("at least two rings are required to separate a source term")
    order = fits[0].order
    for fit in fits:
        if not isinstance(fit, RingFit):
            raise TypeError("decompose_rings expects RingFit instances")
        if fit.order != order:
            raise ValueError("all rings must use the same harmonic order")
    radii = np.array([fit.radius for fit in fits], dtype=float)
    if len(np.unique(radii)) != len(radii):
        raise ValueError("ring radii must be distinct to separate a source term")

    constants = np.array([fit.constant for fit in fits], dtype=float)
    log_radius = np.log(radii)
    slope, intercept = np.polyfit(log_radius, constants, 1)
    trend = slope * log_radius + intercept
    residual = float(np.max(np.abs(constants - trend)))

    spectrum: dict[int, dict[str, object]] = {}
    for n in range(1, order + 1):
        # ``strength`` re-applies r**n so the numbers from different rings are
        # comparable: a valid exterior field gives the same value on every ring.
        values = np.array([fit.strength(n) for fit in fits], dtype=float)
        mean = float(np.mean(values))
        spread = float(np.max(values) - np.min(values))
        spectrum[n] = {
            "strengths": [float(value) for value in values],
            "sampled_amplitudes": [fit.amplitude(n) for fit in fits],
            "mean": mean,
            "absolute_spread": spread,
            "relative_spread": (
                spread / abs(mean) if mean != 0.0 else float("inf") if spread else 0.0
            ),
        }
    return {
        "order": order,
        "radii": [float(value) for value in radii],
        "constants": [float(value) for value in constants],
        "source_strength": float(slope),
        "source_trend_residual": residual,
        "spectrum": spectrum,
        "rms_residual": [fit.rms_residual for fit in fits],
        "relative_residual": [fit.relative_residual for fit in fits],
        "condition_number": [fit.condition_number for fit in fits],
    }