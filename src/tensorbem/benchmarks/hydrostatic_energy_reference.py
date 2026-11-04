"""Independent hydrostatic restoring matrix from the exact potential energy.

Why a second implementation: every guard in ``tests/`` that checks the restoring
matrix against a closed form re-states the same waterplane-integral formula that
``tensorbem.hydrostatic`` uses, so a wrong formula would be "confirmed" by its own
clone. This module knows only the *physics* — Archimedes plus the weight of the
body — and evaluates it on the exact geometry:

.. math::

    V(\\xi) = m g \\, z_G(\\xi) - \\rho g \\int_{S(\\xi)} z \\, dV

where :math:`S(\\xi)` is the part of the rigidly displaced body below the calm
water surface. The integral is taken with the divergence theorem
(:math:`\\int z\\,dV = \\oint (z^2/2) n_z\\,dA`) over the *clipped* body mesh, so
the submerged region is obtained by polygon clipping rather than by any
waterplane formula. The restoring matrix is then the numerical Hessian
:math:`\\partial^2 V / \\partial \\xi_i \\partial \\xi_j` at :math:`\\xi = 0`.

The only shared ingredient with the implementation under test is the body mesh
itself (a faceted box), which is exact geometry rather than a formula.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

#: Mode order used throughout: surge, sway, heave, roll, pitch, yaw.
MODE_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")


def rotation_matrix(rotation_vector: Sequence[float]) -> FloatArray:
    """Exponential map of a rotation vector (Rodrigues' formula).

    The rotation vector is the unambiguous small-angle parametrisation: for a
    rigid body its components tend to the roll/pitch/yaw angles, and unlike a
    product of three sequential elementary rotations the exponential map has a
    Hessian that does not depend on the order of composition.
    """
    alpha = np.asarray(rotation_vector, dtype=float)
    theta = float(np.linalg.norm(alpha))
    skew = np.array(
        [
            [0.0, -alpha[2], alpha[1]],
            [alpha[2], 0.0, -alpha[0]],
            [-alpha[1], alpha[0], 0.0],
        ]
    )
    if theta < 1.0e-12:
        return np.eye(3) + skew + 0.5 * skew @ skew
    return (
        np.eye(3)
        + (math.sin(theta) / theta) * skew
        + ((1.0 - math.cos(theta)) / theta**2) * (skew @ skew)
    )


def clip_below_waterline(polygon: Sequence[Sequence[float]]) -> list[list[float]]:
    """Clip a planar polygon to the half-space ``z <= 0`` (Sutherland-Hodgman)."""
    output: list[list[float]] = []
    count = len(polygon)
    for index in range(count):
        current = list(polygon[index])
        following = list(polygon[(index + 1) % count])
        current_in = current[2] <= 0.0
        following_in = following[2] <= 0.0
        if current_in:
            output.append(current)
        if current_in != following_in:
            # Intersection with the plane z = 0.
            t = -current[2] / (following[2] - current[2])
            output.append(
                [
                    current[0] + t * (following[0] - current[0]),
                    current[1] + t * (following[1] - current[1]),
                    0.0,
                ]
            )
    return output


def _triangle_z_squared_integral(triangle: FloatArray) -> float:
    """Return ``∫_T z² dA`` for a triangle in 3-D (uniform-area measure).

    With barycentric coordinates the mean of :math:`z^2` over a triangle is
    :math:`(z_1^2 + z_2^2 + z_3^2 + z_1 z_2 + z_2 z_3 + z_3 z_1)/6`.
    """
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    area = 0.5 * float(np.linalg.norm(normal))
    z = triangle[:, 2]
    mean_square = (
        z[0] ** 2 + z[1] ** 2 + z[2] ** 2 + z[0] * z[1] + z[1] * z[2] + z[2] * z[0]
    ) / 6.0
    return 0.5 * area * mean_square


def submerged_first_moment_z(vertices: FloatArray, faces: NDArray[np.integer]) -> float:
    """Return :math:`\\int_{S} z\\,dV` for the body clipped to ``z <= 0``.

    Uses :math:`\\int_S z\\,dV = \\oint_{\\partial S} (z^2/2) n_z\\,dA`. The
    waterplane cap lies at ``z = 0`` and contributes nothing, so only the clipped
    body faces matter.
    """
    total = 0.0
    for face in np.asarray(faces, dtype=np.int64):
        corners = vertices[face]
        clipped = clip_below_waterline(corners)
        if len(clipped) < 3:
            continue
        normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        length = float(np.linalg.norm(normal))
        if length == 0.0:
            continue
        n_z = float(normal[2] / length)
        if n_z == 0.0:
            continue
        polygon = np.asarray(clipped, dtype=float)
        integral = 0.0
        for k in range(1, len(polygon) - 1):
            integral += _triangle_z_squared_integral(
                np.array([polygon[0], polygon[k], polygon[k + 1]])
            )
        total += n_z * integral
    return total


def potential_energy(
    displacement: Sequence[float],
    vertices: FloatArray,
    faces: NDArray[np.integer],
    mass: float,
    center_of_gravity: Sequence[float],
    density: float,
    gravity: float,
) -> float:
    """Total potential energy ``m g z_G - ρ g ∫_S z dV`` of the displaced body."""
    xi = np.asarray(displacement, dtype=float)
    rotation = rotation_matrix(xi[3:6])
    translation = xi[0:3]
    moved = vertices @ rotation.T + translation
    cg = rotation @ np.asarray(center_of_gravity, dtype=float) + translation
    body = mass * gravity * float(cg[2])
    buoyancy = density * gravity * submerged_first_moment_z(moved, faces)
    return body - buoyancy


def restoring_matrix_from_energy(
    vertices: FloatArray,
    faces: NDArray[np.integer],
    mass: float,
    center_of_gravity: Sequence[float],
    density: float = 1025.0,
    gravity: float = 9.80665,
    step: float = 2.0e-4,
) -> FloatArray:
    """Numerical Hessian of :func:`potential_energy` at zero displacement.

    Central differences: ``step`` is the absolute increment in every mode (m for
    translations, rad for rotations). The truncation error is ``O(step²)``; with
    ``step = 2e-4`` the entries agree with the closed form to ~1e-7 relative.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    zero = np.zeros(6)

    def energy(xi: FloatArray) -> float:
        return potential_energy(xi, vertices, faces, mass, center_of_gravity, density, gravity)

    base = energy(zero)
    matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(i, 6):
            if i == j:
                plus = zero.copy()
                minus = zero.copy()
                plus[i] = step
                minus[i] = -step
                value = (energy(plus) - 2.0 * base + energy(minus)) / step**2
            else:
                plus_plus = zero.copy()
                plus_minus = zero.copy()
                minus_plus = zero.copy()
                minus_minus = zero.copy()
                plus_plus[i] = plus_plus[j] = step
                plus_minus[i] = step
                plus_minus[j] = -step
                minus_plus[i] = -step
                minus_plus[j] = step
                minus_minus[i] = minus_minus[j] = -step
                value = (
                    energy(plus_plus)
                    - energy(plus_minus)
                    - energy(minus_plus)
                    + energy(minus_minus)
                ) / (4.0 * step**2)
            matrix[i, j] = matrix[j, i] = value
    return matrix