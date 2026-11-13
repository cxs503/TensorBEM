"""Six-degree-of-freedom rigid-body mass matrix in closed block form.

The 6-DOF equation of motion used everywhere in TensorBEM

    (M_RB + A) xdd + B xd + C x = F

needs the rigid-body inertia matrix ``M_RB`` referred to the **reference point**
of the body-fixed frame (the point about which the generalized normals
``(n, r x n)`` are taken).  With the centre of gravity at ``r_g`` relative to
that point, the standard closed form is the block matrix

    M_RB = [[ m I3 ,          -m S(r_g)                 ],
            [ m S(r_g) ,   I_b + m S(r_g)^T S(r_g)       ]]                (1)

where ``S(a) b = a x b`` is the skew-symmetric cross-product matrix, ``I_b`` is
the inertia tensor about the *centre of gravity*, and the lower-right block is
the parallel-axis transfer

    I_O = I_b + m S(r_g)^T S(r_g) = I_b + m (|r_g|^2 I3 - r_g r_g^T).      (2)

Form (1) is the rigid-body inertia matrix of Fossen, *Handbook of Marine Craft
Hydrodynamics and Motion Control* (Wiley, 2011), Sec. 3.3 ("Rigid-Body
Dynamics"), eq. (3.44), written for a body-fixed frame whose origin is not the
CG; it is the same block structure used by Newman, *Marine Hydrodynamics* (MIT
Press, 1977), Ch. 4, and by the Cummins time-domain formulation.  Two properties
are used below as machine-checkable invariants, and neither depends on the
sign convention of any particular code:

* **Symmetry.**  ``M_RB = M_RB^T`` because ``S^T = -S``: the transpose of the
  lower-left block ``m S`` is ``-m S``, i.e. exactly the upper-right block.
* **Kinetic-energy identity.**  For a twist ``x = (v, w)`` of the reference
  point (``v`` velocity, ``w`` angular velocity),

      0.5 x^T M_RB x == 0.5 m |v + w x r_g|^2 + 0.5 w . I_b . w ,          (3)

  which is the rigid-body kinetic energy written at the CG.  This identity is
  frame independent and is therefore the strongest available check of the
  assembly: a sign error in either off-diagonal block, or a double-counted
  parallel-axis term in the lower-right block, breaks it at O(1) relative
  error (see :func:`energy_residual`).

The module also provides a mass-properties route that does *not* use the closed
form at all: :func:`mesh_mass_properties` integrates volume, centroid and
second moments over a closed BEM surface mesh by decomposing it into signed
tetrahedra.  Comparing that quadrature with the analytic values of
:func:`solid_ellipsoid_mass_properties` is an independent (non-tautological)
verification of both.

Mode ordering is TensorBEM's ``(surge, sway, heave, roll, pitch, yaw)`` about
``(x, y, z)``, so ``r x n`` follows the right-hand rule with the same axes.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

__all__ = [
    "MODES",
    "skew_matrix",
    "inertia_about_point",
    "inertia_about_centroid",
    "rigid_body_mass_matrix",
    "solid_ellipsoid_mass_properties",
    "mesh_mass_properties",
    "kinetic_energy",
    "energy_residual",
    "block_form_residual",
    "mass_at_equilibrium",
]

#: Generalized-coordinate order used by every 6-DOF matrix in TensorBEM.
MODES: tuple[str, ...] = ("surge", "sway", "heave", "roll", "pitch", "yaw")

#: Water density used when a caller does not supply one (ITTC sea water).
DEFAULT_DENSITY = 1025.0


def skew_matrix(vector: ArrayLike) -> FloatArray:
    """Return the cross-product matrix ``S(a)`` with ``S(a) b = a x b``."""
    a = np.asarray(vector, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -a[2], a[1]],
            [a[2], 0.0, -a[0]],
            [-a[1], a[0], 0.0],
        ]
    )


def inertia_about_point(
    inertia_cg: ArrayLike,
    mass: float,
    offset: ArrayLike,
    *,
    density: float | None = None,
) -> FloatArray:
    """Transfer an inertia tensor from the CG to a point ``offset`` away.

    ``I_O = I_cg + m (|r|^2 I3 - r r^T)``  (parallel-axis theorem, eq. (2)).
    """
    i_cg = np.asarray(inertia_cg, dtype=float)
    if i_cg.shape != (3, 3):
        raise ValueError("inertia_cg must be a (3, 3) matrix")
    if not np.allclose(i_cg, i_cg.T, rtol=0.0, atol=1e-9 * max(np.max(np.abs(i_cg)), 1.0)):
        raise ValueError("inertia_cg must be symmetric")
    m = float(mass)
    if not math.isfinite(m) or m <= 0.0:
        raise ValueError("mass must be finite and positive")
    r = np.asarray(offset, dtype=float).reshape(3)
    return i_cg + m * (float(r @ r) * np.eye(3) - np.outer(r, r))


def inertia_about_centroid(
    inertia_point: ArrayLike, mass: float, offset: ArrayLike
) -> FloatArray:
    """Inverse of :func:`inertia_about_point` (subtract the transfer term)."""
    i_o = np.asarray(inertia_point, dtype=float)
    if i_o.shape != (3, 3):
        raise ValueError("inertia_point must be a (3, 3) matrix")
    m = float(mass)
    if not math.isfinite(m) or m <= 0.0:
        raise ValueError("mass must be finite and positive")
    r = np.asarray(offset, dtype=float).reshape(3)
    return i_o - m * (float(r @ r) * np.eye(3) - np.outer(r, r))


def rigid_body_mass_matrix(
    mass: float,
    center_of_gravity: ArrayLike,
    inertia: ArrayLike | None = None,
    *,
    inertia_about: str = "cg",
    reference_point: ArrayLike = (0.0, 0.0, 0.0),
    radius_of_gyration: float | None = None,
) -> FloatArray:
    """Assemble the 6x6 rigid-body mass matrix in the closed block form (1).

    Parameters
    ----------
    mass:
        Body mass ``m`` (kg).
    center_of_gravity:
        CG position relative to :func:`reference_point`'s frame (m).  The
        reference point itself is subtracted internally, so
        ``center_of_gravity`` may be given in any frame as long as
        ``reference_point`` is given in the same frame.
    inertia:
        ``(3, 3)`` symmetric inertia tensor.  Interpreted according to
        ``inertia_about``; when omitted the diagonal approximation
        ``I_cg = m k^2 I3`` is used with ``k = 0.4 * |r_g|`` (radius of
        gyration default inherited from
        :func:`tensorbem.hydrostatic.compute_mass_matrix`).
    inertia_about:
        ``"cg"`` (``I_b`` in eq. (1)) or ``"origin"``/``"reference"`` (the
        tensor already referred to the reference point, eq. (2) result).
    reference_point:
        Point about which the 6-DOF generalized coordinates are taken
        (default: the body-fixed origin).
    radius_of_gyration:
        Optional explicit ``k`` for the default (``inertia is None``) path.

    Returns
    -------
    M_RB :
        ``(6, 6)`` symmetric matrix, mode order :data:`MODES`.

    Notes
    -----
    The matrix is not positive definite when ``mass`` is positive and the
    inertia tensor is physically realisable it always is; ``M_RB`` is the
    inertia of a rigid body and must satisfy ``x^T M_RB x > 0`` for every
    non-zero twist (asserted by :func:`energy_residual` callers and by
    ``tests/test_rigid_body_mass.py``).
    """
    m = float(mass)
    if not math.isfinite(m) or m <= 0.0:
        raise ValueError("mass must be finite and positive")

    r_g = np.asarray(center_of_gravity, dtype=float).reshape(3)
    r_ref = np.asarray(reference_point, dtype=float).reshape(3)
    if not np.all(np.isfinite(r_g)) or not np.all(np.isfinite(r_ref)):
        raise ValueError("center_of_gravity and reference_point must be finite")
    r = r_g - r_ref

    if inertia is None:
        k = 0.4 * float(np.linalg.norm(r)) if radius_of_gyration is None else float(
            radius_of_gyration
        )
        if k <= 0.0:
            # CG on the reference point: keep a documented default of 0.4 m so
            # the matrix stays positive definite instead of degenerating.
            k = 0.4 if radius_of_gyration is None else k
        inertia_cg = m * k * k * np.eye(3)
    else:
        tensor = np.asarray(inertia, dtype=float)
        if tensor.shape != (3, 3):
            raise ValueError("inertia must be a (3, 3) matrix")
        if not np.allclose(tensor, tensor.T, rtol=0.0, atol=1e-9 * max(float(np.max(np.abs(tensor))), 1.0)):
            raise ValueError("inertia must be symmetric")
        if inertia_about == "cg":
            inertia_cg = tensor
        elif inertia_about in ("origin", "reference", "reference_point"):
            inertia_cg = inertia_about_centroid(tensor, m, r)
        else:
            raise ValueError("inertia_about must be 'cg' or 'origin'")

    s = skew_matrix(r)
    matrix = np.zeros((6, 6), dtype=float)
    matrix[:3, :3] = m * np.eye(3)
    matrix[:3, 3:] = -m * s
    matrix[3:, :3] = m * s
    matrix[3:, 3:] = inertia_cg + m * (s.T @ s)
    return matrix


def solid_ellipsoid_mass_properties(
    semi_axes: ArrayLike, density: float = DEFAULT_DENSITY
) -> dict[str, object]:
    """Closed-form mass, volume, centroid and CG inertia of a solid ellipsoid.

    For semi-axes ``(a, b, c)`` and density ``rho`` the classical results are

        nabla = 4/3 pi a b c,     m = rho nabla,
        I_cg  = diag( m (b^2+c^2)/5, m (a^2+c^2)/5, m (a^2+b^2)/5 ).

    These are the standard solid-ellipsoid second moments (Lamb, *Hydrodynamics*,
    6th ed., 1932, Ch. VI; Fossen 2011 App. B, table of rigid-body moments of
    inertia), evaluated in machine arithmetic here -- nothing is transcribed.
    """
    axes = np.asarray(semi_axes, dtype=float).reshape(3)
    if np.any(~np.isfinite(axes)) or np.any(axes <= 0.0):
        raise ValueError("semi_axes must be three positive finite values")
    rho = float(density)
    if not math.isfinite(rho) or rho <= 0.0:
        raise ValueError("density must be finite and positive")
    a, b, c = (float(value) for value in axes)
    volume = 4.0 / 3.0 * math.pi * a * b * c
    mass = rho * volume
    inertia_cg = mass / 5.0 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])
    return {
        "semi_axes": axes.tolist(),
        "density": rho,
        "volume": float(volume),
        "mass": float(mass),
        "centroid": [0.0, 0.0, 0.0],
        "inertia_cg": inertia_cg,
        "radii_of_gyration": [
            float(math.sqrt(inertia_cg[i, i] / mass)) for i in range(3)
        ],
    }


def mesh_mass_properties(
    vertices: ArrayLike,
    faces: ArrayLike,
    density: float = DEFAULT_DENSITY,
    *,
    reference_point: ArrayLike = (0.0, 0.0, 0.0),
) -> dict[str, object]:
    """Mass properties of a closed surface mesh by signed-tetrahedron integration.

    Every triangle together with the reference point forms a signed tetrahedron,
    so volume, first moments and second moments are exact sums of polynomial
    integrals over those tetrahedra -- no panel-centroid approximation and no
    use of the closed-form ellipsoid results.  For a tetrahedron with vertices
    ``v0..v3`` (``v0`` = reference point) and signed volume ``V``:

        int x dV     = V * (v0 + v1 + v2 + v3) / 4
        int xi xj dV = V/20 * (sum_k vk_i vk_j + (sum_k vk_i)(sum_k vk_j))

    and the inertia tensor about the reference point is
    ``I = tr(int xx^T dV) I3 - int xx^T dV``.

    Parameters
    ----------
    vertices, faces:
        Closed, outward-oriented triangulation (``(V,3)``, ``(F,3)``); this is
        exactly the ``(mesh.vertices, mesh.faces)`` pair of a TensorBEM
        :class:`~tensorbem.geometry3d.TriangleMesh`.
    density:
        Uniform density (kg/m^3).
    reference_point:
        Point about which the moments are taken (default: the coordinate
        origin).

    Returns
    -------
    dict with ``volume``, ``mass``, ``centroid``, ``inertia_reference``,
    ``inertia_cg`` (transferred with the parallel-axis theorem),
    ``radii_of_gyration`` and ``face_count``.
    """
    verts = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("vertices must have shape (V, 3)")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    rho = float(density)
    if not math.isfinite(rho) or rho <= 0.0:
        raise ValueError("density must be finite and positive")
    p0 = np.asarray(reference_point, dtype=float).reshape(3)

    a = verts[tris[:, 0]]
    b = verts[tris[:, 1]]
    c = verts[tris[:, 2]]
    # 6 * signed tetra volume = det[a - p0, b - p0, c - p0]
    rel_a, rel_b, rel_c = a - p0, b - p0, c - p0
    six_vol = np.einsum("ij,ij->i", rel_a, np.cross(rel_b, rel_c))
    tet_vol = six_vol / 6.0

    volume = float(np.sum(tet_vol))
    if volume <= 0.0:
        raise ValueError(
            "mesh volume is non-positive: the triangulation is not outward oriented "
            "or not closed"
        )

    vertex_sum = a + b + c + p0  # (F, 3)
    first_moment = np.einsum("i,ij->j", tet_vol, vertex_sum) / 4.0
    centroid = first_moment / volume

    # int xi xj dV per tetra, then summed
    sum_sq = np.einsum("fi,fj->fij", a, a)
    sum_sq += np.einsum("fi,fj->fij", b, b)
    sum_sq += np.einsum("fi,fj->fij", c, c)
    sum_sq += np.outer(p0, p0)[None, :, :]
    sum_first = np.einsum("fi,fj->fij", vertex_sum, vertex_sum)
    # Second moments of volume, scaled by density so every tensor below is in
    # SI mass units (kg m^2) rather than geometric units (m^5).
    second = rho * np.einsum("f,fij->ij", tet_vol, sum_sq + sum_first) / 20.0

    second = 0.5 * (second + second.T)
    inertia_reference = np.trace(second) * np.eye(3) - second
    mass = rho * volume
    inertia_cg = inertia_about_centroid(inertia_reference, mass, centroid - p0)

    return {
        "volume": volume,
        "mass": float(mass),
        "centroid": centroid.tolist(),
        "inertia_reference": inertia_reference,
        "inertia_cg": inertia_cg,
        "radii_of_gyration": [
            float(math.sqrt(max(inertia_cg[i, i], 0.0) / mass)) for i in range(3)
        ],
        "face_count": int(tris.shape[0]),
        "density": rho,
    }


def kinetic_energy(
    mass: float,
    inertia_cg: ArrayLike,
    center_of_gravity: ArrayLike,
    velocity: ArrayLike,
    omega: ArrayLike,
    *,
    reference_point: ArrayLike = (0.0, 0.0, 0.0),
) -> float:
    """Rigid-body kinetic energy written at the CG (right-hand side of eq. (3))."""
    m = float(mass)
    r = np.asarray(center_of_gravity, dtype=float).reshape(3) - np.asarray(
        reference_point, dtype=float
    ).reshape(3)
    v = np.asarray(velocity, dtype=float).reshape(3)
    w = np.asarray(omega, dtype=float).reshape(3)
    i_cg = np.asarray(inertia_cg, dtype=float)
    v_g = v + np.cross(w, r)
    return 0.5 * m * float(v_g @ v_g) + 0.5 * float(w @ i_cg @ w)


def energy_residual(
    matrix: ArrayLike,
    mass: float,
    inertia_cg: ArrayLike,
    center_of_gravity: ArrayLike,
    *,
    reference_point: ArrayLike = (0.0, 0.0, 0.0),
    samples: int = 64,
    seed: int = 20261113,
) -> float:
    """Worst relative error of ``0.5 x^T M x`` against the kinetic energy (3).

    Returns ``max |quadratic - energy| / energy`` over ``samples`` random
    twists.  This is the frame-independent physical arbiter for the assembly;
    a wrong off-diagonal sign or a double-counted parallel-axis term shows up at
    O(1) relative error (measured 7.7e-1 for the legacy TensorBEM assembly, see
    ``docs/cases/case_bem_rigid_body_mass/``).
    """
    m = np.asarray(matrix, dtype=float)
    if m.shape != (6, 6):
        raise ValueError("matrix must be (6, 6)")
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(int(samples)):
        v = rng.normal(size=3)
        w = rng.normal(size=3)
        twist = np.concatenate((v, w))
        quadratic = 0.5 * float(twist @ m @ twist)
        exact = kinetic_energy(
            mass,
            inertia_cg,
            center_of_gravity,
            v,
            w,
            reference_point=reference_point,
        )
        worst = max(worst, abs(quadratic - exact) / max(abs(exact), 1e-300))
    return worst


def block_form_residual(
    matrix: ArrayLike,
    mass: float,
    inertia_cg: ArrayLike,
    center_of_gravity: ArrayLike,
    *,
    reference_point: ArrayLike = (0.0, 0.0, 0.0),
) -> dict[str, float]:
    """Entrywise comparison of ``matrix`` with the closed block form (1).

    Returns ``max_relative_error`` (over all 36 entries, normalised by the
    largest entry), ``block_relative_error`` for each of the four 3x3 blocks,
    ``symmetry_error`` and ``scale``.
    """
    exact = rigid_body_mass_matrix(
        mass,
        center_of_gravity,
        inertia_cg,
        inertia_about="cg",
        reference_point=reference_point,
    )
    got = np.asarray(matrix, dtype=float)
    scale = float(np.max(np.abs(exact)))
    deviation = np.abs(got - exact)
    blocks = {
        "translation_translation": deviation[:3, :3],
        "translation_rotation": deviation[:3, 3:],
        "rotation_translation": deviation[3:, :3],
        "rotation_rotation": deviation[3:, 3:],
    }
    block_scale = {
        "translation_translation": float(np.max(np.abs(exact[:3, :3]))),
        "translation_rotation": float(np.max(np.abs(exact[:3, 3:]))),
        "rotation_translation": float(np.max(np.abs(exact[3:, :3]))),
        "rotation_rotation": float(np.max(np.abs(exact[3:, 3:]))),
    }
    return {
        "max_relative_error": float(np.max(deviation) / scale),
        "max_absolute_error": float(np.max(deviation)),
        "scale": scale,
        "symmetry_error": float(np.max(np.abs(got - got.T)) / scale),
        "block_relative_error": {
            name: float(np.max(block) / block_scale[name]) if block_scale[name] > 0 else 0.0
            for name, block in blocks.items()
        },
        "exact": exact,
    }


def mass_at_equilibrium(
    displaced_volume: float, density: float = DEFAULT_DENSITY
) -> float:
    """Mass that puts a body of volume ``nabla`` in hydrostatic equilibrium."""
    volume = float(displaced_volume)
    if not math.isfinite(volume) or volume <= 0.0:
        raise ValueError("displaced_volume must be finite and positive")
    return float(density) * volume


def _as_list(values: Sequence[float]) -> list[float]:
    """Small helper so callers can serialise tuples without numpy scalars."""
    return [float(value) for value in values]