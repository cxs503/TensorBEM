"""Closed-form references for the classical Laplace benchmarks.

The functions here provide *analytical* targets for the verification cases
that do not involve a free surface:

* **Solid harmonics.** Real harmonic polynomials of degree 1, 2 and 3. On a
  polyhedral domain every polynomial is represented exactly, so the exact
  boundary flux ``grad(u) . n`` and the exact interior field are available at
  any point. This is the 3-D analogue of the 2-D analytical suite.
* **Ellipse added mass.** For an elliptic cylinder of semi-axes ``a`` (along
  ``x``) and ``b`` (along ``y``) in an unbounded fluid, the added mass per unit
  span is ``rho*pi*b**2`` for motion along ``x`` and ``rho*pi*a**2`` for motion
  along ``y``.
* **Ellipsoid added mass.** For an ellipsoid with semi-axes ``a1 >= a2 >= a3``
  and volume ``V``, the added mass for translation along axis ``i`` is
  ``rho*V*alpha_i/(2 - alpha_i)`` with Lamb's shape factors

      alpha_i = a1*a2*a3 * integral_0^inf d(lambda) /
                ((a_i^2 + lambda) * sqrt(prod_j (a_j^2 + lambda))).

  The factors satisfy ``alpha_1 + alpha_2 + alpha_3 = 2``, which is checked on
  every evaluation. The sphere limit ``alpha_i = 2/3`` returns the classical
  ``rho*V/2``, the thin-disc limit returns ``8/3*rho*b**3`` for motion normal
  to the disc, and a slender spheroid returns ``rho*V`` for transverse motion
  and zero for axial motion.

* **Ursell cylinder.** For a circular cylinder submerged in deep water, the
  multipole solution reduces the body condition to an infinite system whose
  coefficients have a closed form involving an exponential integral. See
  :func:`ursell_cylinder_coefficients`. This is the first reference here that
  involves a free surface.

Every reference here is independent of the boundary-element discretisation:
the numbers come from one-dimensional quadrature of a classical integral or
from a truncated analytical system, not from another BEM solve.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# Sum of Lamb's shape factors; a constant of the ellipsoid theory.
_SHAPE_FACTOR_SUM = 2.0
_SHAPE_FACTOR_SUM_TOLERANCE = 1.0e-8


def _as_points(points: FloatArray) -> FloatArray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape (M, 3)")
    if not np.isfinite(values).all():
        raise ValueError("points must all be finite")
    return values


# --------------------------------------------------------------------------
# Solid harmonics
# --------------------------------------------------------------------------
def _harmonic_x(points: FloatArray) -> tuple[FloatArray, FloatArray]:
    x = points[:, 0]
    return x, np.column_stack((np.ones_like(x), np.zeros_like(x), np.zeros_like(x)))


def _harmonic_y(points: FloatArray) -> tuple[FloatArray, FloatArray]:
    x = points[:, 0]
    return points[:, 1], np.column_stack((np.zeros_like(x), np.ones_like(x), np.zeros_like(x)))


def _harmonic_z(points: FloatArray) -> tuple[FloatArray, FloatArray]:
    x = points[:, 0]
    return points[:, 2], np.column_stack((np.zeros_like(x), np.zeros_like(x), np.ones_like(x)))


def _harmonic_x2_minus_y2(points: FloatArray) -> tuple[FloatArray, FloatArray]:
    x, y, z = points.T
    return x**2 - y**2, np.column_stack((2.0 * x, -2.0 * y, np.zeros_like(z)))


def _harmonic_2xy(points: FloatArray) -> tuple[FloatArray, FloatArray]:
    x, y, z = points.T
    return 2.0 * x * y, np.column_stack((2.0 * y, 2.0 * x, np.zeros_like(z)))


def _harmonic_2z2_minus_x2_minus_y2(
    points: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    x, y, z = points.T
    return (
        2.0 * z**2 - x**2 - y**2,
        np.column_stack((-2.0 * x, -2.0 * y, 4.0 * z)),
    )


def _harmonic_x3_minus_3xy2(points: FloatArray) -> tuple[FloatArray, FloatArray]:
    x, y, z = points.T
    return (
        x**3 - 3.0 * x * y**2,
        np.column_stack((3.0 * (x**2 - y**2), -6.0 * x * y, np.zeros_like(z))),
    )


def _harmonic_3x2y_minus_y3(points: FloatArray) -> tuple[FloatArray, FloatArray]:
    x, y, z = points.T
    return (
        3.0 * x**2 * y - y**3,
        np.column_stack((6.0 * x * y, 3.0 * (x**2 - y**2), np.zeros_like(z))),
    )


def _harmonic_x_4z2_minus_r2(points: FloatArray) -> tuple[FloatArray, FloatArray]:
    x, y, z = points.T
    return (
        4.0 * x * z**2 - x**3 - x * y**2,
        np.column_stack(
            (4.0 * z**2 - 3.0 * x**2 - y**2, -2.0 * x * y, 8.0 * x * z)
        ),
    )


def _harmonic_z_2z2_minus_3r2(points: FloatArray) -> tuple[FloatArray, FloatArray]:
    x, y, z = points.T
    return (
        2.0 * z**3 - 3.0 * x**2 * z - 3.0 * y**2 * z,
        np.column_stack(
            (-6.0 * x * z, -6.0 * y * z, 6.0 * z**2 - 3.0 * x**2 - 3.0 * y**2)
        ),
    )


#: Real solid harmonics: ``name -> (degree, value_and_gradient)``.
SOLID_HARMONICS: dict[str, tuple[int, object]] = {
    "x": (1, _harmonic_x),
    "y": (1, _harmonic_y),
    "z": (1, _harmonic_z),
    "x2-y2": (2, _harmonic_x2_minus_y2),
    "2xy": (2, _harmonic_2xy),
    "2z2-x2-y2": (2, _harmonic_2z2_minus_x2_minus_y2),
    "x3-3xy2": (3, _harmonic_x3_minus_3xy2),
    "3x2y-y3": (3, _harmonic_3x2y_minus_y3),
    "x(4z2-x2-y2)": (3, _harmonic_x_4z2_minus_r2),
    "z(2z2-3x2-3y2)": (3, _harmonic_z_2z2_minus_3r2),
}


def solid_harmonic_degree(name: str) -> int:
    """Return the polynomial degree of a registered solid harmonic."""
    if name not in SOLID_HARMONICS:
        raise ValueError(
            f"unknown solid harmonic {name!r}; available: {sorted(SOLID_HARMONICS)}"
        )
    return int(SOLID_HARMONICS[name][0])


def solid_harmonic_names(degree: int | None = None) -> tuple[str, ...]:
    """Return the harmonic names, optionally restricted to one degree."""
    if degree is None:
        return tuple(SOLID_HARMONICS)
    if isinstance(degree, (bool, np.bool_)) or not isinstance(degree, (int, np.integer)):
        raise ValueError("degree must be a positive integer")
    if degree < 1:
        raise ValueError("degree must be a positive integer")
    names = tuple(name for name, (value, _) in SOLID_HARMONICS.items() if value == degree)
    if not names:
        raise ValueError(f"no solid harmonic of degree {degree} is registered")
    return names


def solid_harmonic(name: str, points: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Evaluate one real solid harmonic and its gradient.

    Returns ``(values, gradients)`` with shapes ``(M,)`` and ``(M, 3)``.
    """
    if name not in SOLID_HARMONICS:
        raise ValueError(
            f"unknown solid harmonic {name!r}; available: {sorted(SOLID_HARMONICS)}"
        )
    coordinates = _as_points(points)
    _, evaluator = SOLID_HARMONICS[name]
    values, gradients = evaluator(coordinates)
    return values, gradients


# --------------------------------------------------------------------------
# Ellipse (2-D)
# --------------------------------------------------------------------------
def _positive(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def ellipse_added_mass(
    semi_major: float, semi_minor: float, density: float = 1.0
) -> tuple[float, float]:
    """Added mass per unit span of an elliptic cylinder: ``(x, y)`` motion.

    The classical results are ``rho*pi*b**2`` for translation along ``x`` and
    ``rho*pi*a**2`` along ``y``, where ``a`` is the ``x`` semi-axis and ``b``
    the ``y`` semi-axis. Note that each value uses the semi-axis *perpendicular*
    to the motion.
    """
    major = _positive("semi_major", semi_major)
    minor = _positive("semi_minor", semi_minor)
    rho = _positive("density", density)
    return rho * math.pi * minor**2, rho * math.pi * major**2


def ellipse_translation_trace_scale(
    semi_major: float, semi_minor: float, axis: str = "x"
) -> float:
    """Return ``k`` with ``u = -U*k*x_axis`` on the ellipse boundary.

    This is the exact Dirichlet trace of the exterior radiation potential of an
    elliptic cylinder translating with speed ``U``. With ``a`` and ``b`` the
    semi-axes along ``x`` and ``y``, the scale is ``b/a`` for motion along
    ``x`` and ``a/b`` along ``y``. It follows from ``m_A = rho*k*V`` with
    ``V = pi*a*b`` and the classical added masses above, and is consistent with
    the elliptic-coordinate solution.
    """
    major = _positive("semi_major", semi_major)
    minor = _positive("semi_minor", semi_minor)
    if axis == "x":
        return minor / major
    if axis == "y":
        return major / minor
    raise ValueError("axis must be 'x' or 'y'")


# --------------------------------------------------------------------------
# Ellipsoid (3-D)
# --------------------------------------------------------------------------
def _as_semi_axes(semi_axes: FloatArray) -> FloatArray:
    axes = np.asarray(semi_axes, dtype=float)
    if axes.shape != (3,):
        raise ValueError("semi_axes must have shape (3,)")
    if not np.isfinite(axes).all():
        raise ValueError("semi_axes must all be finite")
    if np.any(axes <= 0.0):
        raise ValueError("semi_axes must all be positive")
    return axes


def ellipsoid_volume(semi_axes: FloatArray) -> float:
    """Return the volume ``4*pi*a1*a2*a3/3`` of an ellipsoid."""
    axes = _as_semi_axes(semi_axes)
    return 4.0 / 3.0 * math.pi * float(np.prod(axes))


def ellipsoid_shape_factors(
    semi_axes: FloatArray, nodes: int = 600, dynamic_range: float = 1.0e12
) -> FloatArray:
    r"""Return Lamb's shape factors ``alpha`` of shape ``(3,)``.

    The factors are the one-dimensional integrals

    .. math::

        \alpha_i = a_1 a_2 a_3
        \int_0^\infty \frac{d\lambda}
        {(a_i^2+\lambda)\sqrt{\prod_j (a_j^2+\lambda)}},

    evaluated numerically. The substitution :math:`\lambda=e^{s}` turns the
    integration domain into the real line and the node range is anchored on
    the smallest and largest squared semi-axis, so neither a very flat nor a
    very slender ellipsoid loses resolution. The computed factors are checked
    against the identity :math:`\alpha_1+\alpha_2+\alpha_3=2`; a failure means
    the quadrature could not resolve the requested aspect ratio, and is
    reported instead of returning a plausible-looking wrong number. With the
    default settings the identity holds to ``1e-12`` for axis ratios from
    ``1e-5`` to ``1e4``.

    Parameters
    ----------
    semi_axes:
        Positive semi-axes of shape ``(3,)``.
    nodes:
        Number of Gauss-Legendre nodes on the logarithmic axis.
    dynamic_range:
        The integration is truncated where ``lambda`` is ``dynamic_range``
        times smaller than the smallest and larger than the largest squared
        semi-axis. The integrand decays like ``lambda**-2``, so the truncation
        must be generous.
    """
    axes = _as_semi_axes(semi_axes)
    if isinstance(nodes, (bool, np.bool_)) or not isinstance(nodes, (int, np.integer)):
        raise ValueError("nodes must be an integer >= 16")
    if int(nodes) < 16:
        raise ValueError("nodes must be an integer >= 16")
    range_value = _positive("dynamic_range", dynamic_range)
    a, b, c = (float(value) for value in axes)
    squared = np.array([a * a, b * b, c * c])
    lower = math.log(float(np.min(squared))) - math.log(range_value)
    upper = math.log(float(np.max(squared))) + math.log(range_value)

    nodes_array, weights = leggauss(int(nodes))
    exponent = 0.5 * (upper - lower) * nodes_array + 0.5 * (upper + lower)
    node_weights = 0.5 * (upper - lower) * weights
    lam = np.exp(exponent)
    root = np.sqrt((a * a + lam) * (b * b + lam) * (c * c + lam))
    factors = np.empty(3, dtype=float)
    for index, axis in enumerate((a, b, c)):
        integrand = lam / ((axis * axis + lam) * root)
        factors[index] = a * b * c * float(np.dot(node_weights, integrand))

    residual = float(np.sum(factors)) - _SHAPE_FACTOR_SUM
    if abs(residual) > _SHAPE_FACTOR_SUM_TOLERANCE:
        raise ValueError(
            "the ellipsoid shape-factor quadrature did not converge "
            f"(alpha sum error {residual:.3e}); the aspect ratio is probably "
            f"outside the range the {int(nodes)}-node rule can resolve"
        )
    return factors


def prolate_spheroid_shape_factors(
    long_semi_axis: float, transverse_semi_axis: float
) -> FloatArray:
    r"""Closed-form Lamb factors for a prolate spheroid ``(a > b = c)``.

    With :math:`e=\sqrt{1-b^2/a^2}`,

    .. math::

        \alpha_{a}=2\frac{1-e^2}{e^3}\left(\tanh^{-1}e-e\right),
        \qquad \alpha_b=\alpha_c=\frac{2-\alpha_a}{2}.

    This is an independent code path from
    :func:`ellipsoid_shape_factors`; the two agree to quadrature accuracy, and
    both satisfy the sphere limit ``e -> 0``, ``alpha = 2/3``.
    """
    major = _positive("long_semi_axis", long_semi_axis)
    minor = _positive("transverse_semi_axis", transverse_semi_axis)
    if minor >= major:
        raise ValueError("a prolate spheroid requires long_semi_axis > transverse_semi_axis")
    eccentricity = math.sqrt(1.0 - (minor / major) ** 2)
    alpha_axial = (
        2.0
        * (1.0 - eccentricity**2)
        / eccentricity**3
        * (math.atanh(eccentricity) - eccentricity)
    )
    alpha_transverse = 0.5 * (_SHAPE_FACTOR_SUM - alpha_axial)
    return np.array([alpha_axial, alpha_transverse, alpha_transverse])


def ellipsoid_added_mass(
    semi_axes: FloatArray, density: float = 1.0
) -> FloatArray:
    """Added mass of an ellipsoid for translation along each principal axis.

    Returns ``rho*V*alpha_i/(2-alpha_i)`` with ``V`` the ellipsoid volume and
    ``alpha`` from :func:`ellipsoid_shape_factors`.
    """
    axes = _as_semi_axes(semi_axes)
    rho = _positive("density", density)
    factors = ellipsoid_shape_factors(axes)
    volume = ellipsoid_volume(axes)
    return rho * volume * factors / (_SHAPE_FACTOR_SUM - factors)


def ellipsoid_translation_trace_scale(
    semi_axes: FloatArray, axis: int = 0
) -> float:
    """Return ``k`` with ``u = -U*k*x_axis`` on the ellipsoid boundary.

    The exterior radiation potential of a translating ellipsoid has a
    *linear* Dirichlet trace on the surface; this is what makes the ellipsoid
    a clean benchmark, because the boundary data need no pointwise exterior
    solution. With ``m_A = rho*k*V`` and the added mass above, ``k`` is
    ``alpha_axis/(2-alpha_axis)``.
    """
    axes = _as_semi_axes(semi_axes)
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise ValueError("axis must be 0, 1 or 2")
    if int(axis) not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    factor = ellipsoid_shape_factors(axes)[int(axis)]
    return float(factor / (_SHAPE_FACTOR_SUM - factor))


# --------------------------------------------------------------------------
# Ursell's circular cylinder in deep water
# --------------------------------------------------------------------------

_EULER_GAMMA = 0.5772156649015328606065120900824024310422


def exponential_integral(x: float) -> float:
    """The exponential integral ``Ei(x)`` for real ``x > 0``.

    Power series below ``x = 60``, then the divergent asymptotic series
    ``e**x/x * sum_k k!/x**k`` truncated at its smallest term. Truncating at the
    minimum matters: continuing past it adds terms that grow without bound and
    destroys the answer. The break is placed where the asymptotic truncation
    error, about ``x!/x**x ~ sqrt(2*pi*x)*exp(-x)``, is already ``1e-25``, so
    both branches reach full double precision.
    """
    value = float(x)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("x must be a positive finite number")
    if value > 700.0:
        raise ValueError("Ei overflows double precision beyond x = 700")
    if value <= 60.0:
        total = _EULER_GAMMA + math.log(value)
        term = 1.0
        for k in range(1, 600):
            term *= value / k
            increment = term / k
            total += increment
            if abs(increment) < 1e-17 * abs(total):
                return total
        return total
    return math.exp(value) * _scaled_exponential_integral(value)


def _scaled_exponential_integral(x: float) -> float:
    """Return ``exp(-x)*Ei(x)`` without forming ``exp(x)``.

    Used for the free-surface term, where ``exp(-2Kf)*Ei(2Kf)`` stays ``O(1/x)``
    while both factors are individually enormous. Computing the product from
    the asymptotic series avoids the overflow of ``Ei(x)`` for large ``x`` and
    keeps full relative accuracy in the cancellation that follows.
    """
    total = 0.0
    term = 1.0
    previous = math.inf
    for k in range(0, 400):
        if k > 0:
            term *= k / x
        if term > previous:
            break
        previous = term
        total += term
    return total / x


def ursell_cylinder_coefficients(
    ka: float, a_over_f: float, terms: int = 40
) -> tuple[float, float]:
    """Dimensionless added mass and damping of Ursell's circular cylinder.

    A circular cylinder of radius ``a`` whose axis lies a distance ``f`` below
    the free surface of deep water, oscillating in heave or sway. In infinite
    depth the heave and sway coefficients are *identical* (Ursell 1949; the
    matrix below does not depend on the mode); this function therefore returns
    a single pair, valid for either motion.

    The method of multipoles expands the potential in wave-free harmonics and
    reduces the body condition to the infinite system

        -a_m/m + sum_n a_n A_mn = -delta_{m1},      m = 1, 2, ...

    with, for deep water,

        A_mn = (-ka)**(m+n)/(m! n!) * [
                   (m+n-1)!/(2kf)**(m+n)
                 + 2 sum_{r=1}^{m+n-1} (r-1)!/(2kf)**r
                 + 2 exp(-2kf) * (i*pi - Ei(2kf)) ].

    The principal-value integral of the underlying representation collapses to
    that finite sum plus an exponential integral (Evans et al. 1979), so nothing
    here is quadrature: it is a closed-form matrix whose only numerically
    special ingredient is :func:`exponential_integral`. The hydrodynamic
    coefficients follow from the first coefficient,

        mu + i*nu = -1 + 2*a_1,

    where ``mu = A/(rho*pi*a**2)`` is the added mass and ``nu = B/(omega*rho*pi*a**2)``
    the damping, both normalised by the displaced mass per unit span. Note the
    ``-1`` applies to the real part only.

    Parameters
    ----------
    ka:
        Wavenumber times radius, ``k*a`` with ``k = omega**2/g``.
    a_over_f:
        Radius over submergence of the *axis*. Must lie strictly between zero
        and one for a fully submerged cylinder.
    terms:
        Truncation order of the infinite system.

    Returns
    -------
    ``(mu, nu)``, the dimensionless added mass and damping.
    """
    k_a = float(ka)
    ratio = float(a_over_f)
    if not math.isfinite(k_a) or k_a <= 0.0:
        raise ValueError("ka must be a positive finite number")
    if not math.isfinite(ratio) or not 0.0 < ratio < 1.0:
        raise ValueError("a_over_f must lie strictly between 0 and 1")
    if isinstance(terms, bool) or not isinstance(terms, (int, np.integer)):
        raise ValueError("terms must be a positive integer")
    if int(terms) < 1:
        raise ValueError("terms must be a positive integer")

    count = int(terms)
    k_f = k_a / ratio
    x = 2.0 * k_f
    if x > 700.0:
        raise ValueError("2*k*f is too large for double precision")
    log_x = math.log(x)
    # log(n!) up to 2*count, so that (m+n-1)! never overflows.
    log_factorial = [math.lgamma(n + 1) for n in range(2 * count + 2)]
    scaled_ei = _scaled_exponential_integral(x) if x > 60.0 else (
        math.exp(-x) * exponential_integral(x)
    )

    matrix = np.zeros((count, count), dtype=np.complex128)
    for m in range(1, count + 1):
        for n in range(1, count + 1):
            order = m + n
            bracket = math.exp(log_factorial[order - 1] - order * log_x)
            for r in range(1, order):
                bracket += 2.0 * math.exp(log_factorial[r - 1] - r * log_x)
            bracket += 2.0 * math.exp(-x) * 1j * math.pi - 2.0 * scaled_ei
            matrix[m - 1, n - 1] = (
                (-k_a) ** order
                * math.exp(-log_factorial[m] - log_factorial[n])
                * bracket
            )
    matrix[np.diag_indices(count)] -= 1.0 / np.arange(1, count + 1)
    right_hand_side = np.zeros(count, dtype=np.complex128)
    right_hand_side[0] = -1.0
    coefficients = np.linalg.solve(matrix, right_hand_side)
    first = coefficients[0]
    return float(-1.0 + 2.0 * first.real), float(2.0 * first.imag)


def ursell_cylinder_added_mass(
    radius: float,
    submergence: float,
    omega: float,
    density: float = 1025.0,
    *,
    gravity: float = 9.80665,
    terms: int = 40,
) -> tuple[float, float]:
    """Dimensional added mass and damping of a deeply submerged cylinder.

    Returns ``(A, B)`` in the convention

        F = -(A * acceleration) - (B * velocity)

    for a unit-length section oscillating in heave or sway, so ``A`` has units
    of mass per unit span and ``B`` of mass per unit span per unit time. The
    wave frequency is ``omega``, the axis sits ``submergence`` below the still
    water level, and the wavenumber is ``k = omega**2/g``.
    """
    a = _positive("radius", radius)
    f = _positive("submergence", submergence)
    w = _positive("omega", omega)
    rho = _positive("density", density)
    g = _positive("gravity", gravity)
    if f <= a:
        raise ValueError("the cylinder must be fully submerged: submergence > radius")
    mu, nu = ursell_cylinder_coefficients(w * w * a / g, a / f, terms)
    displaced = rho * math.pi * a * a
    return mu * displaced, nu * displaced * w
