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
* **Rigid-body rotational added inertia.** Closed-form rotational added
  moment of inertia for an elliptic cylinder ``(1/8)*rho*pi*(a^2-b^2)^2`` and
  for a triaxial ellipsoid in terms of Lamb's shape factors (Lamb 1932,
  sections 71 and 115), together with full ``3x3`` and ``6x6`` rigid-body
  added-mass tensors for verifying exterior Neumann boundary-element solves.
* **Dean-Ursell-Ogilvie cylinder diffraction.** Multipole solution for regular
  deep-water waves scattered by a submerged circular cylinder: exact zero
  reflection ``R = 0`` (Dean 1948), unit transmission ``T = exp(i*delta)``
  (Ursell 1950), and circularly polarised wave excitation forces ``X_z = -i*X_x``
  (Ogilvie 1963).
* **MacCamy-Fuchs vertical cylinder diffraction.** Exact cylindrical-harmonic
  diffraction potential, horizontal wave load, phase lag, and seabed
  overturning moment for a bottom-mounted surface-piercing circular pile in
  finite or infinite water depth (MacCamy & Fuchs 1954), evaluated from a
  pure-NumPy Bessel/Hankel implementation.
* **3-D deep-water floating hemisphere and submerged sphere.** Hulme's (1982)
  semi-analytical spherical wave-free multipole solution for heave and surge
  radiation and regular-wave excitation of a floating hemisphere, alongside
  the closed-form Thorne/Wang/Linton multipole solution for radiation and
  diffraction of a submerged sphere in deep water.

Every reference here is independent of the boundary-element discretisation:
the numbers come from one-dimensional quadrature of a classical integral or
from a truncated analytical system, not from another BEM solve.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

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


def ellipse_added_moment_of_inertia(
    semi_major: float, semi_minor: float, density: float = 1.0
) -> float:
    """Rotational added moment of inertia per unit span of an elliptic cylinder.

    For rotation in the ``(x, y)`` plane about the origin, Lamb (1932, sec. 71)
    and Newman (1977, table 4.3) give

        I_A = (1/8) * rho * pi * (a**2 - b**2)**2.

    For a circular cylinder ``a = b`` the normal velocity ``(r x n)_z`` vanishes
    identically on the boundary and ``I_A = 0``.
    """
    major = _positive("semi_major", semi_major)
    minor = _positive("semi_minor", semi_minor)
    rho = _positive("density", density)
    return 0.125 * rho * math.pi * (major * major - minor * minor) ** 2


def ellipse_added_mass_matrix(
    semi_major: float, semi_minor: float, density: float = 1.0
) -> FloatArray:
    """Full ``(3, 3)`` rigid-body added-mass matrix of a centred elliptic cylinder.

    Modes are ``(surge_x, heave_y, roll_xy)`` about the ellipse centre; symmetry
    planes decouple all off-diagonal entries.
    """
    m_x, m_y = ellipse_added_mass(semi_major, semi_minor, density)
    i_zz = ellipse_added_moment_of_inertia(semi_major, semi_minor, density)
    return np.diag([m_x, m_y, i_zz])


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


def ellipsoid_added_moment_of_inertia(
    semi_axes: FloatArray, density: float = 1.0
) -> FloatArray:
    """Rotational added moments of inertia ``(A_44, A_55, A_66)`` of an ellipsoid.

    For rotation about principal axis ``i`` (with ``(i, j, k)`` cyclic), Lamb
    (1932, sec. 115) and Newman (1977, eq. 4.132) give

        A_{4+i,4+i} = (1/5) * rho * V
                      * (a_j**2 - a_k**2)**2 * (alpha_k - alpha_j)
                      / (2*(a_j**2 - a_k**2) + (a_j**2 + a_k**2)*(alpha_j - alpha_k)).

    When ``a_j == a_k`` the cross-section in the ``(j, k)`` plane is circular,
    ``(r x n)_i == 0`` everywhere on the surface, and the rotational added
    inertia vanishes identically; that degenerate branch is handled explicitly
    rather than dividing zero by zero.
    """
    axes = _as_semi_axes(semi_axes)
    rho = _positive("density", density)
    factors = ellipsoid_shape_factors(axes)
    volume = ellipsoid_volume(axes)
    moments = np.zeros(3, dtype=float)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        aj, ak = float(axes[j]), float(axes[k])
        if abs(aj - ak) <= 1.0e-12 * max(aj, ak):
            moments[i] = 0.0
            continue
        aj2, ak2 = aj * aj, ak * ak
        diff = aj2 - ak2
        alpha_diff = float(factors[k] - factors[j])
        numerator = diff * diff * alpha_diff
        denominator = 2.0 * diff - (aj2 + ak2) * alpha_diff
        moments[i] = 0.2 * rho * volume * (numerator / denominator)
    return moments


def ellipsoid_added_mass_matrix(
    semi_axes: FloatArray, density: float = 1.0
) -> FloatArray:
    """Full ``(6, 6)`` rigid-body added-mass matrix of a centred ellipsoid.

    Modes are ``(surge, sway, heave, roll, pitch, yaw)`` about the origin. The
    three coordinate symmetry planes make all off-diagonal couplings vanish.
    """
    translational = ellipsoid_added_mass(semi_axes, density)
    rotational = ellipsoid_added_moment_of_inertia(semi_axes, density)
    return np.diag(np.concatenate((translational, rotational)))


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


def _ursell_multipole_system(
    ka: float, a_over_f: float, terms: int = 40
) -> tuple[np.ndarray, list[float], float, float]:
    """Assemble Ursell's deep-water multipole operator ``M = -diag(1/m) + A``."""
    if isinstance(ka, (bool, np.bool_)) or isinstance(a_over_f, (bool, np.bool_)):
        raise ValueError("ka and a_over_f must be positive finite numbers")
    k_a = float(ka)
    ratio = float(a_over_f)
    if not math.isfinite(k_a) or k_a <= 0.0:
        raise ValueError("ka must be a positive finite number")
    if not math.isfinite(ratio) or not 0.0 < ratio < 1.0:
        raise ValueError("a_over_f must lie strictly between 0 and 1")
    if isinstance(terms, (bool, np.bool_)) or not isinstance(terms, (int, np.integer)):
        raise ValueError("terms must be a positive integer")
    if int(terms) < 1:
        raise ValueError("terms must be a positive integer")

    count = int(terms)
    k_f = k_a / ratio
    x = 2.0 * k_f
    if x > 700.0:
        raise ValueError("2*k*f is too large for double precision")
    log_x = math.log(x)
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
    return matrix, log_factorial, k_a, x


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
    """
    matrix, _, _, _ = _ursell_multipole_system(ka, a_over_f, terms)
    right_hand_side = np.zeros(matrix.shape[0], dtype=np.complex128)
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


def dean_cylinder_diffraction(
    radius: float,
    submergence: float,
    ka: float,
    amplitude: float = 1.0,
    density: float = 1025.0,
    *,
    gravity: float = 9.80665,
    terms: int = 40,
) -> dict[str, complex | float]:
    """Closed-form multipole diffraction solution for a submerged circular cylinder.

    A regular deep-water Airy wave of elevation amplitude ``amplitude`` and
    wavenumber ``k = ka / radius`` propagates in ``+x`` (time factor
    ``exp(-i*omega*t)``, incident potential
    ``phi_I = -(i*g*amplitude/omega)*exp(k*z + i*k*x)``) past a fixed circular
    cylinder of radius ``a`` whose axis lies at ``(0, -f)`` with ``f > a``.

    Three classical analytical results hold simultaneously (Dean 1948;
    Ursell 1950; Ogilvie 1963):

    1. **Zero reflection.** Because ``exp(k*(z + i*x))`` is holomorphic in
       ``(z + f) + i*x``, its Taylor expansion about the cylinder axis excites
       only the chiral multipoles that radiate toward ``+infinity``. The linear
       reflection coefficient vanishes identically: ``R = 0``.
    2. **Unit transmission with pure phase shift.** Energy conservation forces
       ``|T| = 1``, with

           T = 1 + 4*i*pi*exp(-2*k*f) * sum_{m>=1} c_m * (-ka)**m / m!,

       where ``(-diag(1/m) + A) c = -[(-ka)**m / m!]``.
    3. **Circularly polarised wave excitation force.** In deep water the
       horizontal and vertical exciting forces per unit span have equal
       magnitude and a 90-degree phase lag,

           F_x = 2*i*pi*rho*g*a*amplitude*exp(-k*f)*c_1,    F_z = -i*F_x,

       recovering Morison's inertia limit ``|F_x| -> 2*rho*pi*a**2*(g*k*A*exp(-kf))``
       (``C_M = 2``) as ``f / a -> infinity``.
    """
    a = _positive("radius", radius)
    f = _positive("submergence", submergence)
    k_a = _positive("ka", ka)
    amp = _positive("amplitude", amplitude)
    rho = _positive("density", density)
    g = _positive("gravity", gravity)
    if f <= a:
        raise ValueError("the cylinder must be fully submerged: submergence > radius")

    matrix, log_factorial, _, x = _ursell_multipole_system(k_a, a / f, terms)
    count = matrix.shape[0]
    incident_modes = np.array(
        [((-k_a) ** m) * math.exp(-log_factorial[m]) for m in range(1, count + 1)],
        dtype=float,
    )
    coefficients = np.linalg.solve(matrix, -incident_modes.astype(np.complex128))
    transmission = complex(
        1.0 + 4.0j * math.pi * math.exp(-x) * np.dot(coefficients, incident_modes)
    )
    force_x = complex(
        2.0j * math.pi * rho * g * a * amp * math.exp(-0.5 * x) * coefficients[0]
    )
    force_z = -1.0j * force_x
    return {
        "reflection": 0.0j,
        "transmission": transmission,
        "transmission_modulus": abs(transmission),
        "transmission_phase_radians": math.atan2(transmission.imag, transmission.real),
        "force_x": force_x,
        "force_z": force_z,
        "force_magnitude": abs(force_x),
    }


# --------------------------------------------------------------------------
# Cylindrical Bessel functions and MacCamy-Fuchs pile diffraction
# --------------------------------------------------------------------------

#: First positive zero of ``J_0(x) = 0`` (first axisymmetric interior resonance).
BESSEL_J0_FIRST_ZERO = 2.4048255576957728
#: First positive zero of ``J_1(x) = 0`` (first dipole interior resonance).
BESSEL_J1_FIRST_ZERO = 3.8317059702075123

_BESSEL_SWITCH = 25.0
_MILLER_TOP = 56


def bessel_j01_y01(
    x: FloatArray | float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Evaluate ``(J_0(x), J_1(x), Y_0(x), Y_1(x))`` for positive real ``x``.

    Uses Miller's backward recurrence normalised by Neumann's sum
    ``J_0 + 2*sum_{k>=1} J_{2k} = 1`` together with Neumann's series (and its
    term-by-term derivative) for ``Y_0`` and ``Y_1`` when ``x <= 25``, and
    15-term Hankel asymptotic series when ``x > 25``. Both branches avoid
    alternating power-series cancellation and agree with the Wronskian
    ``J_0*Y_1 - J_1*Y_0 = -2/(pi*x)`` to ``~2e-15`` across ``[1e-8, 1e4]``.
    """
    if isinstance(x, (bool, np.bool_)):
        raise ValueError("x must be a positive finite real array or number")
    values = np.asarray(x, dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("x must be positive and finite")

    original_shape = values.shape
    flat = values.ravel()
    j0 = np.empty_like(flat)
    j1 = np.empty_like(flat)
    y0 = np.empty_like(flat)
    y1 = np.empty_like(flat)

    small = flat <= _BESSEL_SWITCH
    if np.any(small):
        xs = flat[small]
        top = _MILLER_TOP
        f_prev2 = np.zeros_like(xs)
        f_prev1 = np.full_like(xs, 1.0e-200)
        table = np.empty((top + 1, xs.size), dtype=float)
        table[top] = f_prev1
        inv_x = 2.0 / xs
        for order in range(top, 0, -1):
            f_curr = (order * inv_x) * f_prev1 - f_prev2
            overflow = np.abs(f_curr) > 1.0e100
            if np.any(overflow):
                f_curr[overflow] *= 1.0e-100
                f_prev1[overflow] *= 1.0e-100
                table[order:, overflow] *= 1.0e-100
            table[order - 1] = f_curr
            f_prev2 = f_prev1
            f_prev1 = f_curr
        scale = table[0] + 2.0 * np.sum(table[2 : top + 1 : 2], axis=0)
        table /= scale[None, :]
        j0_s, j1_s = table[0], table[1]
        ks = np.arange(1, top // 2 + 1, dtype=float)[:, None]
        signs = np.where(ks % 2 == 1, 1.0, -1.0)
        ln_term = np.log(0.5 * xs) + _EULER_GAMMA
        y0_s = (2.0 / math.pi) * (
            ln_term * j0_s + 2.0 * np.sum(signs * table[2 : top + 1 : 2] / ks, axis=0)
        )
        shifted = np.vstack((table[3 : top + 1 : 2], np.zeros((1, xs.size))))
        y1_s = (2.0 / math.pi) * (
            -j0_s / xs
            + ln_term * j1_s
            - np.sum(signs * (table[1:top:2] - shifted) / ks, axis=0)
        )
        j0[small], j1[small], y0[small], y1[small] = j0_s, j1_s, y0_s, y1_s

    if np.any(~small):
        xl = flat[~small]

        def _hankel_pq(nu: float, z: FloatArray) -> tuple[FloatArray, FloatArray]:
            mu = 4.0 * nu * nu
            p = np.ones_like(z)
            q = np.zeros_like(z)
            term = np.ones_like(z)
            for m in range(1, 16):
                term = term * ((mu - (2 * m - 1) ** 2) / (m * 8.0 * z))
                if m % 2 == 1:
                    q = q + ((-1.0) ** ((m - 1) // 2)) * term
                else:
                    p = p + ((-1.0) ** (m // 2)) * term
            return p, q

        amp = np.sqrt(2.0 / (math.pi * xl))
        chi0 = xl - 0.25 * math.pi
        p0, q0 = _hankel_pq(0.0, xl)
        j0[~small] = amp * (p0 * np.cos(chi0) - q0 * np.sin(chi0))
        y0[~small] = amp * (p0 * np.sin(chi0) + q0 * np.cos(chi0))
        chi1 = xl - 0.75 * math.pi
        p1, q1 = _hankel_pq(1.0, xl)
        j1[~small] = amp * (p1 * np.cos(chi1) - q1 * np.sin(chi1))
        y1[~small] = amp * (p1 * np.sin(chi1) + q1 * np.cos(chi1))

    return (
        j0.reshape(original_shape),
        j1.reshape(original_shape),
        y0.reshape(original_shape),
        y1.reshape(original_shape),
    )


def bessel_jn_yn(
    max_order: int, x: FloatArray | float
) -> tuple[FloatArray, FloatArray]:
    """Return ``(J_0..J_N, Y_0..Y_N)`` at scalar or array ``x > 0`` for ``N = max_order``."""
    if isinstance(max_order, (bool, np.bool_)) or not isinstance(
        max_order, (int, np.integer)
    ):
        raise ValueError("max_order must be a non-negative integer")
    order_max = int(max_order)
    if order_max < 0:
        raise ValueError("max_order must be a non-negative integer")
    if isinstance(x, (bool, np.bool_)):
        raise ValueError("x must be positive and finite")
    arr = np.asarray(x, dtype=float)
    if arr.size > 0 and (not np.isfinite(arr).all() or np.any(arr <= 0.0)):
        raise ValueError("x must be positive and finite")
    out_shape = (order_max + 1,) + arr.shape
    if arr.size == 0:
        empty = np.empty(out_shape, dtype=float)
        return empty, empty.copy()

    flat = arr.ravel()
    count = flat.size
    top = max(order_max + 40, int(1.5 * float(np.max(flat))) + 40)
    if top % 2 == 1:
        top += 1
    f_prev2 = np.zeros(count, dtype=float)
    f_prev1 = np.full(count, 1.0e-200, dtype=float)
    table = np.empty((top + 1, count), dtype=float)
    table[top] = f_prev1
    for order in range(top, 0, -1):
        f_curr = (2.0 * order / flat) * f_prev1 - f_prev2
        big = np.abs(f_curr) > 1.0e100
        if np.any(big):
            f_curr = np.where(big, f_curr * 1.0e-100, f_curr)
            f_prev1 = np.where(big, f_prev1 * 1.0e-100, f_prev1)
            table[order:, big] *= 1.0e-100
        table[order - 1] = f_curr
        f_prev2 = f_prev1
        f_prev1 = f_curr
    scale = table[0] + 2.0 * np.sum(table[2 : top + 1 : 2], axis=0)
    jn = table[: order_max + 1] / scale[None, :]

    j0_v, j1_v, y0_v, y1_v = bessel_j01_y01(flat)
    jn[0] = j0_v
    if order_max >= 1:
        jn[1] = j1_v
    yn = np.empty((order_max + 1, count), dtype=float)
    yn[0] = y0_v
    if order_max >= 1:
        yn[1] = y1_v
    for order in range(1, order_max):
        overflow = (~np.isfinite(yn[order])) | (np.abs(yn[order]) > 1.0e280)
        next_val = np.where(
            overflow,
            -np.inf,
            (2.0 * order / flat) * yn[order] - yn[order - 1],
        )
        yn[order + 1] = next_val
    return jn.reshape(out_shape), yn.reshape(out_shape)


def maccamy_fuchs_surface_potential(
    ka: float, angles: FloatArray, terms: int = 40
) -> np.ndarray:
    """Dimensionless horizontal total potential ``varphi(a, theta)`` on the pile.

    For an incident plane wave ``varphi_I = exp(i*k*x) = exp(i*ka*cos(theta))``
    scattered by a rigid cylinder ``r = a``, MacCamy & Fuchs (1954) give

        varphi(a, theta) = (2*i / (pi*ka))
                           * sum_{m=0}^inf epsilon_m * i**m * cos(m*theta) / H_m^{(1)'}(ka)

    with ``epsilon_0 = 1`` and ``epsilon_m = 2`` for ``m >= 1``.
    """
    k_a = _positive("ka", ka)
    if isinstance(terms, (bool, np.bool_)) or not isinstance(terms, (int, np.integer)):
        raise ValueError("terms must be a positive integer")
    count = int(terms)
    if count < 1:
        raise ValueError("terms must be a positive integer")
    theta = np.asarray(angles, dtype=float)
    if not np.isfinite(theta).all():
        raise ValueError("angles must all be finite")

    jn, yn = bessel_jn_yn(count + 1, k_a)
    total = np.zeros(theta.shape, dtype=np.complex128)
    prefactor = 2.0j / (math.pi * k_a)
    for m in range(count + 1):
        jp = -jn[1] if m == 0 else 0.5 * (jn[m - 1] - jn[m + 1])
        yp = -yn[1] if m == 0 else 0.5 * (yn[m - 1] - yn[m + 1])
        if not math.isfinite(yp) or abs(yp) > 1.0e140:
            break
        hp = jp + 1.0j * yp
        epsilon = 1.0 if m == 0 else 2.0
        total = total + epsilon * (1.0j**m) * np.cos(m * theta) / hp
    return prefactor * total


def maccamy_fuchs_diffraction(
    ka: float,
    radius: float = 1.0,
    depth: float = math.inf,
    amplitude: float = 1.0,
    density: float = 1025.0,
    *,
    gravity: float = 9.80665,
) -> dict[str, complex | float | None]:
    """Closed-form wave load on a bottom-mounted vertical circular cylinder.

    With time factor ``exp(-i*omega*t)`` and incident wave elevation
    ``eta_I = Re(A * exp(i*(k*x - omega*t)))``, the complex horizontal force
    phasor on a pile of radius ``a`` in water depth ``h`` is (MacCamy & Fuchs
    1954; Newman 1977, sec. 7.5)

        F_x = (4 * rho * g * A / k**2) * tanh(k*h) / H_1^{(1)'}(k*a).

    In the long-wave limit ``ka -> 0`` this tends to
    ``-2*i * rho * g * A * pi * a**2 * tanh(k*h)``, i.e. Morison's inertia force
    with ``C_M = 2``. For finite ``depth``, the seabed overturning moment about
    ``(0, 0, -h)`` is also returned in closed form.
    """
    k_a = _positive("ka", ka)
    a = _positive("radius", radius)
    amp = _positive("amplitude", amplitude)
    rho = _positive("density", density)
    g = _positive("gravity", gravity)
    if isinstance(depth, (bool, np.bool_)):
        raise ValueError("depth must be a positive number or inf")
    h_val = float(depth)
    if math.isnan(h_val) or h_val <= 0.0:
        raise ValueError("depth must be a positive number or inf")

    k = k_a / a
    j0, j1, y0, y1 = (float(item) for item in bessel_j01_y01(k_a))
    j1_prime = j0 - j1 / k_a
    y1_prime = y0 - y1 / k_a
    h1_prime = complex(j1_prime, y1_prime)

    if math.isinf(h_val):
        tanh_kh = 1.0
        omega = math.sqrt(g * k)
        moment_seabed: complex | None = None
    else:
        kh = k * h_val
        tanh_kh = math.tanh(kh) if kh < 50.0 else 1.0
        sech_kh = (1.0 / math.cosh(kh)) if kh < 50.0 else 0.0
        omega = math.sqrt(g * k * tanh_kh)
        moment_factor = (kh * tanh_kh + sech_kh - 1.0) / (k**3)
        moment_seabed = (4.0 * rho * g * amp * moment_factor) / h1_prime

    horizontal_integral = -4.0 / (k * h1_prime)
    force_x = (4.0 * rho * g * amp * tanh_kh / (k * k)) / h1_prime
    effective_cm = 4.0 / (math.pi * k_a * k_a * abs(h1_prime))
    phase_lag = -math.atan2(j1_prime, y1_prime)
    return {
        "ka": k_a,
        "wavenumber": k,
        "omega": omega,
        "horizontal_integral": horizontal_integral,
        "force_x": force_x,
        "force_magnitude": abs(force_x),
        "phase_lag_radians": phase_lag,
        "effective_inertia_coefficient": effective_cm,
        "moment_about_seabed": moment_seabed,
    }


# --------------------------------------------------------------------------
# Struve functions and 3-D deep-water spherical hydrodynamics
# --------------------------------------------------------------------------
_STRUVE_SWITCH = 8.0
_STRUVE_LAG_U, _STRUVE_LAG_W = laggauss(32)


def struve_h01(x: FloatArray | float) -> tuple[FloatArray, FloatArray]:
    """Return Struve functions ``(H_0(x), H_1(x))`` for finite ``x >= 0``.

    Uses the convergent power series on ``[0, 8]`` and the non-oscillatory
    Laplace integral representation combined with :func:`bessel_j01_y01` on
    ``(8, inf)``, achieving ``~1.5e-13`` relative continuity across the seam.
    """
    if isinstance(x, (bool, np.bool_)):
        raise ValueError("x must be a non-negative finite real array or number")
    values = np.asarray(x, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("x must be non-negative and finite")

    original_shape = values.shape
    flat = values.ravel()
    h0 = np.empty_like(flat)
    h1 = np.empty_like(flat)

    small = flat <= _STRUVE_SWITCH
    if np.any(small):
        xs = flat[small]
        x2 = xs * xs
        term0 = (2.0 / math.pi) * xs
        sum0 = term0.copy()
        term1 = (2.0 / (3.0 * math.pi)) * x2
        sum1 = term1.copy()
        for m in range(36):
            odd = 2 * m + 3
            term0 = term0 * (-x2 / (odd * odd))
            sum0 = sum0 + term0
            term1 = term1 * (-x2 / (odd * (odd + 2)))
            sum1 = sum1 + term1
        h0[small] = sum0
        h1[small] = sum1

    large = ~small
    if np.any(large):
        xl = flat[large]
        h0_large = np.empty_like(xl)
        h1_large = np.empty_like(xl)
        for start in range(0, xl.size, 4096):
            stop = min(start + 4096, xl.size)
            xb = xl[start:stop]
            _, _, y0, y1 = bessel_j01_y01(xb)
            ratio = _STRUVE_LAG_U[None, :] / xb[:, None]
            inv_sqrt = 1.0 / np.sqrt(1.0 + ratio * ratio)
            i0 = inv_sqrt @ _STRUVE_LAG_W
            i1 = inv_sqrt @ (_STRUVE_LAG_U * _STRUVE_LAG_W)
            h0_large[start:stop] = y0 + (2.0 / (math.pi * xb)) * i0
            h1_large[start:stop] = y1 + (2.0 / math.pi) * (1.0 + i1 / (xb * xb))
        h0[large] = h0_large
        h1[large] = h1_large

    return h0.reshape(original_shape), h1.reshape(original_shape)


@dataclass(frozen=True)
class HulmeHemisphereResult:
    """Semi-analytical reference for Hulme's (1982) floating hemisphere."""

    ka: float
    wavenumber: float
    omega: float
    displaced_volume: float
    surge_added_mass_dimless: float
    surge_damping_dimless: float
    surge_far_field_damping_dimless: float
    heave_added_mass_dimless: float
    heave_damping_dimless: float
    heave_far_field_damping_dimless: float
    surge_added_mass: float
    surge_damping: float
    heave_added_mass: float
    heave_damping: float
    excitation_force_x: complex
    excitation_force_z: complex


def hulme_hemisphere_radiation(
    ka: float,
    radius: float = 1.0,
    amplitude: float = 1.0,
    density: float = 1025.0,
    *,
    n_terms: int = 48,
    quad_order: int = 128,
    gravity: float = 9.80665,
) -> HulmeHemisphereResult:
    """Hulme (1982) floating-hemisphere radiation and excitation reference.

    Expands the heave (``m = 0``) and surge (``m = 1``) radiation potentials as
    a Kelvin wave source/horizontal dipole at the origin plus ``n_terms``
    spherical wave-free potentials satisfying ``d(psi)/dz - K*psi = 0``
    identically on ``z = 0``, projected onto odd Legendre harmonics on the
    hemisphere ``r = a, z <= 0``. Complex regular-wave excitation forces follow
    from the exact origin Kochin amplitude via Haskind reciprocity.
    """
    from .free_surface3d import wave_function_3d

    k_a = _positive("ka", ka)
    a = _positive("radius", radius)
    amp = _positive("amplitude", amplitude)
    rho = _positive("density", density)
    g = _positive("gravity", gravity)
    if isinstance(n_terms, (bool, np.bool_)) or not isinstance(
        n_terms, (int, np.integer)
    ) or int(n_terms) < 4:
        raise ValueError("n_terms must be an integer >= 4")
    if isinstance(quad_order, (bool, np.bool_)) or not isinstance(
        quad_order, (int, np.integer)
    ) or int(quad_order) < 2 * int(n_terms):
        raise ValueError("quad_order must be an integer >= 2 * n_terms")

    m_max = int(n_terms)
    q_order = int(quad_order)
    nu = k_a / a
    omega = math.sqrt(g * nu)
    volume = (2.0 / 3.0) * math.pi * (a**3)

    nodes, weights = leggauss(q_order)
    mu = 0.5 * (nodes + 1.0)
    w = 0.5 * weights
    s = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))

    x_dim = k_a * s
    y_dim = k_a * mu
    d_dim = np.hypot(x_dim, y_dim)
    f_val, fx_val, fy_val = wave_function_3d(x_dim, y_dim)

    j0 = np.ones_like(x_dim)
    j1 = np.zeros_like(x_dim)
    nz = x_dim > 1.0e-15
    if np.any(nz):
        j0[nz], j1[nz], _, _ = bessel_j01_y01(x_dim[nz])
    exp_my = np.exp(-y_dim)
    w_val = f_val + 1.0j * math.pi * exp_my * j0
    wx_val = fx_val - 1.0j * math.pi * exp_my * j1
    wy_val = fy_val - 1.0j * math.pi * exp_my * j0

    # --- Heave (axisymmetric m = 0, body normal n_z = -mu) ---
    p0 = np.zeros((2 * m_max + 4, q_order), dtype=float)
    p0[0] = 1.0
    p0[1] = mu
    for deg in range(2, 2 * m_max + 4):
        p0[deg] = ((2 * deg - 1) * mu * p0[deg - 1] - (deg - 1) * p0[deg - 2]) / deg

    basis_phi0 = np.zeros((m_max + 1, q_order), dtype=np.complex128)
    basis_dphi0 = np.zeros((m_max + 1, q_order), dtype=np.complex128)
    basis_phi0[0] = 2.0 + 2.0 * k_a * w_val
    basis_dphi0[0] = -2.0 + 2.0 * (k_a * k_a) * (s * wx_val + mu * wy_val)
    for n in range(1, m_max + 1):
        basis_phi0[n] = p0[2 * n] + (k_a / (2.0 * n)) * p0[2 * n - 1]
        basis_dphi0[n] = -(2 * n + 1) * p0[2 * n] - k_a * p0[2 * n - 1]

    test0 = p0[1 : 2 * m_max + 2 : 2]
    coeffs_heave = np.linalg.solve(
        (test0 * w[None, :]) @ basis_dphi0.T,
        (test0 * w[None, :]) @ (-mu),
    )
    phi_heave = coeffs_heave @ basis_phi0
    heave_dimless = -3.0 * np.sum(w * phi_heave * (-mu))
    a33_dimless = float(heave_dimless.real)
    b33_dimless = float(heave_dimless.imag)
    b33_ff_dimless = float(12.0 * math.pi * k_a * (abs(coeffs_heave[0]) ** 2))

    # --- Surge (m = 1, body normal n_x = +sin(theta)*cos(varphi)) ---
    w_yy = w_val + 1.0 / d_dim + y_dim / (d_dim**3)
    w_xx = -w_yy - wx_val / np.maximum(x_dim, 1.0e-30)
    w_xy = -wx_val + x_dim / (d_dim**3)

    p1 = np.zeros((2 * m_max + 4, q_order), dtype=float)
    p1[1] = s
    p1[2] = 3.0 * mu * s
    for deg in range(3, 2 * m_max + 4):
        p1[deg] = ((2 * deg - 1) * mu * p1[deg - 1] - deg * p1[deg - 2]) / (deg - 1)

    basis_phi1 = np.zeros((m_max + 1, q_order), dtype=np.complex128)
    basis_dphi1 = np.zeros((m_max + 1, q_order), dtype=np.complex128)
    basis_phi1[0] = 2.0 * s - 2.0 * (k_a * k_a) * wx_val
    basis_dphi1[0] = -4.0 * s - 2.0 * (k_a**3) * (s * w_xx + mu * w_xy)
    for n in range(1, m_max + 1):
        basis_phi1[n] = p1[2 * n + 1] + (k_a / (2.0 * n)) * p1[2 * n]
        basis_dphi1[n] = (
            -(2 * n + 2) * p1[2 * n + 1]
            - (k_a * (2 * n + 1) / (2.0 * n)) * p1[2 * n]
        )

    test1 = p1[1 : 2 * m_max + 2 : 2]
    coeffs_surge = np.linalg.solve(
        (test1 * w[None, :]) @ basis_dphi1.T,
        (test1 * w[None, :]) @ s,
    )
    phi_surge = coeffs_surge @ basis_phi1
    surge_dimless = -1.5 * np.sum(w * phi_surge * s)
    a11_dimless = float(surge_dimless.real)
    b11_dimless = float(surge_dimless.imag)
    b11_ff_dimless = float(6.0 * math.pi * (k_a**3) * (abs(coeffs_surge[0]) ** 2))

    force_z = complex(4.0 * math.pi * rho * g * amp * a * a * coeffs_heave[0])
    force_x = complex(4.0j * math.pi * rho * g * amp * k_a * a * a * coeffs_surge[0])

    return HulmeHemisphereResult(
        ka=k_a,
        wavenumber=nu,
        omega=omega,
        displaced_volume=volume,
        surge_added_mass_dimless=a11_dimless,
        surge_damping_dimless=b11_dimless,
        surge_far_field_damping_dimless=b11_ff_dimless,
        heave_added_mass_dimless=a33_dimless,
        heave_damping_dimless=b33_dimless,
        heave_far_field_damping_dimless=b33_ff_dimless,
        surge_added_mass=rho * volume * a11_dimless,
        surge_damping=rho * omega * volume * b11_dimless,
        heave_added_mass=rho * volume * a33_dimless,
        heave_damping=rho * omega * volume * b33_dimless,
        excitation_force_x=force_x,
        excitation_force_z=force_z,
    )


@dataclass(frozen=True)
class SubmergedSphereWaveResult:
    """Closed-form multipole reference for a submerged 3-D sphere in waves."""

    ka: float
    submergence_ratio: float
    wavenumber: float
    omega: float
    displaced_volume: float
    surge_added_mass_dimless: float
    surge_damping_dimless: float
    surge_far_field_damping_dimless: float
    heave_added_mass_dimless: float
    heave_damping_dimless: float
    heave_far_field_damping_dimless: float
    surge_added_mass: float
    surge_damping: float
    heave_added_mass: float
    heave_damping: float
    excitation_force_x: complex
    excitation_force_z: complex
    haskind_error_x: float
    haskind_error_z: float


def submerged_sphere_wave_multipole(
    ka: float,
    radius: float = 1.0,
    submergence: float = 1.5,
    amplitude: float = 1.0,
    density: float = 1025.0,
    *,
    n_terms: int = 20,
    gravity: float = 9.80665,
) -> SubmergedSphereWaveResult:
    """Exact Thorne/Wang/Linton multipole solution for a submerged sphere.

    Solves the truncated spherical-multipole linear systems for axisymmetric
    heave/diffraction (``m = 0``) and antisymmetric surge/diffraction
    (``m = 1``) of a sphere of radius ``a`` centered at ``(0, 0, -submergence)``
    in deep water. Every coupling matrix entry reduces algebraically to
    factorials and :func:`exponential_integral`, requiring zero numerical
    quadrature and completely independent of :mod:`tensorbem.free_surface3d`.
    """
    k_a = _positive("ka", ka)
    a = _positive("radius", radius)
    f = _positive("submergence", submergence)
    if f <= a:
        raise ValueError("submergence must be strictly greater than radius")
    amp = _positive("amplitude", amplitude)
    rho = _positive("density", density)
    g = _positive("gravity", gravity)
    if isinstance(n_terms, (bool, np.bool_)) or not isinstance(
        n_terms, (int, np.integer)
    ) or int(n_terms) < 4:
        raise ValueError("n_terms must be an integer >= 4")

    max_deg = int(n_terms)
    nu = k_a / a
    omega = math.sqrt(g * nu)
    volume = (4.0 / 3.0) * math.pi * (a**3)
    sub_ratio = f / a

    beta = 2.0 * nu * f
    l_seq = np.empty(2 * max_deg + 4, dtype=np.complex128)
    l_seq[0] = (
        -math.exp(-beta) * exponential_integral(beta)
        + 1.0j * math.pi * math.exp(-beta)
    )
    fact_run = 1.0
    beta_pow = 1.0
    for idx in range(1, 2 * max_deg + 4):
        if idx > 1:
            fact_run *= idx - 1
        beta_pow *= beta
        l_seq[idx] = l_seq[idx - 1] + fact_run / beta_pow

    factorials = [math.factorial(k) for k in range(2 * max_deg + 4)]
    base_inc = -(1.0j * g * amp / omega) * math.exp(-nu * f)

    def _solve_azimuthal(m: int) -> tuple[complex, complex]:
        degrees = list(range(m, max_deg + 1))
        size = len(degrees)
        g_mat = np.zeros((size, size), dtype=np.complex128)
        sys_mat = np.eye(size, dtype=np.complex128)
        for row_idx, p in enumerate(degrees):
            for col_idx, n in enumerate(degrees):
                sgn = -1.0 if (n + p) % 2 == 1 else 1.0
                denom = factorials[n - m] * factorials[p + m]
                g_pn = (sgn / denom) * (
                    factorials[n + p] / ((2.0 * sub_ratio) ** (n + p + 1))
                    + 2.0 * (k_a ** (n + p + 1)) * l_seq[n + p]
                )
                g_mat[row_idx, col_idx] = g_pn
                sys_mat[row_idx, col_idx] -= (p / (p + 1.0)) * g_pn

        idx_dipole = degrees.index(1)
        rhs_rad = np.zeros(size, dtype=np.complex128)
        rhs_rad[idx_dipole] = 0.5 * a if m == 0 else -0.5 * a
        c_rad = np.linalg.solve(sys_mat, rhs_rad)
        phi_rad_1 = c_rad[idx_dipole] + (g_mat @ c_rad)[idx_dipole]
        rad_dimless = (phi_rad_1 / a) if m == 0 else (-phi_rad_1 / a)

        eps_m = 1.0 if m == 0 else 2.0
        b_inc = np.array(
            [
                base_inc
                * eps_m
                * (1.0j**m)
                * ((-1.0) ** (p + m))
                * (k_a**p)
                / factorials[p + m]
                for p in degrees
            ],
            dtype=np.complex128,
        )
        rhs_diff = np.array(
            [(p / (p + 1.0)) * b_inc[i] for i, p in enumerate(degrees)],
            dtype=np.complex128,
        )
        d_diff = np.linalg.solve(sys_mat, rhs_diff)
        phi_tot_1 = b_inc[idx_dipole] + d_diff[idx_dipole] + (g_mat @ d_diff)[idx_dipole]
        area_factor = (-4.0 * math.pi * a * a / 3.0) if m == 0 else (4.0 * math.pi * a * a / 3.0)
        force = -1.0j * omega * rho * area_factor * phi_tot_1
        return complex(rad_dimless), complex(force)

    heave_rad, force_z = _solve_azimuthal(0)
    surge_rad, force_x = _solve_azimuthal(1)

    a33_dim = float(heave_rad.real)
    b33_dim = float(heave_rad.imag)
    a11_dim = float(surge_rad.real)
    b11_dim = float(surge_rad.imag)

    b33_phys = rho * omega * volume * b33_dim
    b11_phys = rho * omega * volume * b11_dim
    b33_ff_phys = (nu * nu / (2.0 * rho * g * omega * amp * amp)) * (abs(force_z) ** 2)
    b11_ff_phys = (nu * nu / (4.0 * rho * g * omega * amp * amp)) * (abs(force_x) ** 2)
    b33_ff_dim = b33_ff_phys / (rho * omega * volume)
    b11_ff_dim = b11_ff_phys / (rho * omega * volume)

    haskind_err_z = abs(b33_phys - b33_ff_phys) / max(1.0e-30, abs(b33_phys), 1.0)
    haskind_err_x = abs(b11_phys - b11_ff_phys) / max(1.0e-30, abs(b11_phys), 1.0)

    return SubmergedSphereWaveResult(
        ka=k_a,
        submergence_ratio=sub_ratio,
        wavenumber=nu,
        omega=omega,
        displaced_volume=volume,
        surge_added_mass_dimless=a11_dim,
        surge_damping_dimless=b11_dim,
        surge_far_field_damping_dimless=float(b11_ff_dim),
        heave_added_mass_dimless=a33_dim,
        heave_damping_dimless=b33_dim,
        heave_far_field_damping_dimless=float(b33_ff_dim),
        surge_added_mass=rho * volume * a11_dim,
        surge_damping=b11_phys,
        heave_added_mass=rho * volume * a33_dim,
        heave_damping=b33_phys,
        excitation_force_x=force_x,
        excitation_force_z=force_z,
        haskind_error_x=float(haskind_err_x),
        haskind_error_z=float(haskind_err_z),
    )


