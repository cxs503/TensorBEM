"""Analytic verification of the 6x6 rigid-body mass matrix with off-origin CG.

The unit under test is :func:`tensorbem.hydrostatic.compute_mass_matrix`.

Reference construction (deliberately independent of the implementation)
----------------------------------------------------------------------
The implementation assembles ``M`` from a skew-symmetric matrix ``S(r_g)``.
To avoid restating that algebra, every reference value here is built from the
*rigid-body velocity field* of a discrete body,

.. math::  u(r) = v + \\omega \\times r ,

through the linear momentum map

.. math::

    p(v, \\omega)    = \\sum_i m_i u(r_i),
    \\quad h_O(v, \\omega) = \\sum_i m_i r_i \\times u(r_i),

where ``O`` is the reference point (the origin). Both are linear in
``(v, omega)``, so evaluating them on the six unit generalised velocities gives
the exact 6x6 generalised mass matrix of that body,

.. math:: M_{ref} = [\\,\\partial(p, h_O) / \\partial(v, \\omega)\\,] .

The body is then characterised by ``mass = sum_i m_i``,
``r_g = sum_i m_i r_i / mass`` and the inertia tensor about the CG

.. math::  J_{cg} = \\sum_i m_i \\big(|\\rho_i|^2 I - \\rho_i \\rho_i^T\\big),
           \\quad \\rho_i = r_i - r_g .

``compute_mass_matrix(mass, r_g, J_cg)`` must reproduce ``M_ref`` to machine
precision. ``r_g`` is deliberately non-zero in all three components: with a CG
at the origin the translation/rotation coupling blocks vanish and the
parallel-axis term is identically zero, so that degenerate case cannot detect
either of the two defects reported by the external audit.

Physical acceptance criteria used below
---------------------------------------
1. momentum map: ``M @ (v, omega)`` reproduces ``(p, h_O)`` (both signs of the
   coupling blocks are exercised);
2. pure CG translation: for ``omega = 0`` the angular momentum about the origin
   must be ``r_g x (m v)`` -- a sign error in the ``v``-``omega`` coupling flips
   it;
3. kinetic energy: ``0.5 * x^T M x == 0.5*m*|v + omega x r_g|^2
   + 0.5*omega.J_cg.omega`` for random ``(v, omega)``;
4. parallel-axis bookkeeping: the ``(3:6, 3:6)`` block must equal the inertia
   about the origin ``J_cg + m*(|r_g|^2 I - r_g r_g^T)`` exactly once, and the
   default-inertia path must transfer the parallel-axis term exactly once too.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.hydrostatic import compute_mass_matrix

# --------------------------------------------------------------------------- #
# Discrete reference body: four point masses, CG off the origin in x, y and z
# --------------------------------------------------------------------------- #
_MASSES = np.array([1200.0, 850.0, 2400.0, 640.0, 1310.0])
_POSITIONS = np.array(
    [
        [1.00, 0.20, -0.40],
        [-0.50, 1.10, 0.30],
        [0.30, -0.80, -1.50],
        [2.00, 0.50, 0.90],
        [-1.20, -0.30, 0.10],
    ]
)


def _reference_body() -> tuple[float, np.ndarray, np.ndarray]:
    """Return ``(mass, r_g, J_cg)`` of the discrete reference body."""
    mass = float(_MASSES.sum())
    r_g = (_MASSES[:, None] * _POSITIONS).sum(axis=0) / mass
    rho = _POSITIONS - r_g
    j_cg = np.zeros((3, 3))
    for m_i, rho_i in zip(_MASSES, rho):
        j_cg += m_i * (float(rho_i @ rho_i) * np.eye(3) - np.outer(rho_i, rho_i))
    return mass, r_g, j_cg


def _momentum_map() -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """6x6 mass matrix from the momentum map; no skew-matrix algebra used."""
    m_ref = np.zeros((6, 6))
    for column in range(6):
        x = np.zeros(6)
        x[column] = 1.0
        v, w = x[:3], x[3:]
        # Rigid-body velocity field u_i = v + w x r_i
        p = np.zeros(3)
        h_o = np.zeros(3)
        for m_i, r_i in zip(_MASSES, _POSITIONS):
            u_i = v + np.cross(w, r_i)
            p += m_i * u_i
            h_o += m_i * np.cross(r_i, u_i)
        m_ref[:3, column] = p
        m_ref[3:, column] = h_o
    mass, r_g, j_cg = _reference_body()
    return m_ref, mass, r_g, j_cg


def test_reference_cg_is_off_origin() -> None:
    """Guard: the analytic case must have a CG offset in all three axes."""
    _mass, r_g, _j_cg = _reference_body()
    assert np.all(np.abs(r_g) > 0.05), r_g


class TestMomentumMap:
    """M built from the momentum map versus compute_mass_matrix."""

    def test_matches_momentum_map_to_machine_precision(self) -> None:
        m_ref, mass, r_g, j_cg = _momentum_map()
        m = compute_mass_matrix(mass, r_g, inertia=j_cg)
        scale = float(np.max(np.abs(m_ref)))
        assert np.max(np.abs(m - m_ref)) / scale < 1.0e-13, (
            "compute_mass_matrix disagrees with the momentum map; "
            f"max_abs_diff/scale = {np.max(np.abs(m - m_ref)) / scale:.3e}"
        )

    def test_coupling_block_is_minus_m_skew_of_cg(self) -> None:
        m_ref, mass, r_g, j_cg = _momentum_map()
        m = compute_mass_matrix(mass, r_g, inertia=j_cg)
        s = np.array(
            [
                [0.0, -r_g[2], r_g[1]],
                [r_g[2], 0.0, -r_g[0]],
                [-r_g[1], r_g[0], 0.0],
            ]
        )
        # p = m*v - m*S(r_g)*w  and  h_O = m*S(r_g)*v + J_O*w
        assert np.allclose(m[:3, 3:], -mass * s, rtol=0.0, atol=1.0e-12 * abs(mass))
        assert np.allclose(m[3:, :3], mass * s, rtol=0.0, atol=1.0e-12 * abs(mass))
        assert np.allclose(m[:3, 3:], m_ref[:3, 3:], rtol=0.0, atol=1.0e-12 * abs(mass))
        assert np.allclose(m[3:, :3], m_ref[3:, :3], rtol=0.0, atol=1.0e-12 * abs(mass))

    def test_pure_cg_translation_carries_moment_rg_cross_mv(self) -> None:
        """omega = 0: p = m v and h_O = r_g x (m v), exactly."""
        _m_ref, mass, r_g, j_cg = _momentum_map()
        m = compute_mass_matrix(mass, r_g, inertia=j_cg)
        v = np.array([0.7, -1.3, 0.45])
        x = np.concatenate((v, np.zeros(3)))
        y = m @ x
        assert np.allclose(y[:3], mass * v, rtol=1.0e-14)
        assert np.allclose(y[3:], np.cross(r_g, mass * v), rtol=1.0e-13, atol=1.0e-13 * mass)

    def test_kinetic_energy_identity_for_random_velocities(self) -> None:
        _m_ref, mass, r_g, j_cg = _momentum_map()
        m = compute_mass_matrix(mass, r_g, inertia=j_cg)
        rng = np.random.default_rng(20261103)
        errors = []
        for _ in range(64):
            v = rng.normal(size=3)
            w = rng.normal(size=3)
            x = np.concatenate((v, w))
            quad = 0.5 * float(x @ m @ x)
            v_g = v + np.cross(w, r_g)
            energy = 0.5 * mass * float(v_g @ v_g) + 0.5 * float(w @ j_cg @ w)
            errors.append(abs(quad - energy) / abs(energy))
        assert max(errors) < 1.0e-13, f"max relative energy error {max(errors):.3e}"


class TestParallelAxisBookkeeping:
    """The parallel-axis transfer must be applied exactly once."""

    def test_rotational_block_equals_inertia_about_origin_once(self) -> None:
        _m_ref, mass, r_g, j_cg = _momentum_map()
        m = compute_mass_matrix(mass, r_g, inertia=j_cg)
        parallel = mass * (float(r_g @ r_g) * np.eye(3) - np.outer(r_g, r_g))
        j_origin = j_cg + parallel
        assert np.allclose(m[3:, 3:], j_origin, rtol=0.0, atol=1.0e-9 * float(np.max(np.abs(j_origin))))
        # explicit double-count guard
        double_counted = j_cg + 2.0 * parallel
        assert not np.allclose(m[3:, 3:], double_counted, rtol=0.0, atol=1.0e-9 * float(np.max(np.abs(double_counted))))

    def test_default_inertia_transfers_parallel_axis_once(self) -> None:
        """inertia=None: J_cg = m*(0.4*|r_g|)^2*I, transferred once to the origin."""
        mass = 1000.0
        r_g = np.array([0.3, -0.4, -0.6])
        m = compute_mass_matrix(mass, r_g)  # default heuristic inertia (about CG)
        k = 0.4 * float(np.linalg.norm(r_g))
        j_cg = mass * k**2 * np.eye(3)
        parallel = mass * (float(r_g @ r_g) * np.eye(3) - np.outer(r_g, r_g))
        j_origin = j_cg + parallel
        assert np.allclose(m[3:, 3:], j_origin, rtol=1.0e-13)
        # the same body seen through the energy identity
        v = np.array([0.4, 0.9, -0.25])
        w = np.array([-0.6, 0.35, 0.8])
        x = np.concatenate((v, w))
        quad = 0.5 * float(x @ m @ x)
        v_g = v + np.cross(w, r_g)
        energy = 0.5 * mass * float(v_g @ v_g) + 0.5 * float(w @ j_cg @ w)
        assert abs(quad - energy) / abs(energy) < 1.0e-13


class TestStructuralProperties:
    """Symmetry, positive definiteness and the CG-at-origin limit."""

    def test_matrix_is_symmetric(self) -> None:
        _m_ref, mass, r_g, j_cg = _momentum_map()
        m = compute_mass_matrix(mass, r_g, inertia=j_cg)
        assert np.max(np.abs(m - m.T)) <= 1.0e-9 * float(np.max(np.abs(m)))

    def test_matrix_is_positive_definite(self) -> None:
        _m_ref, mass, r_g, j_cg = _momentum_map()
        m = compute_mass_matrix(mass, r_g, inertia=j_cg)
        assert np.all(np.linalg.eigvalsh(m) > 0.0)

    def test_cg_at_origin_gives_block_diagonal_matrix(self) -> None:
        mass = 2000.0
        j_cg = np.diag([1.0e3, 2.0e3, 3.0e3])
        m = compute_mass_matrix(mass, np.zeros(3), inertia=j_cg)
        assert np.allclose(m[:3, :3], mass * np.eye(3))
        assert np.allclose(m[:3, 3:], 0.0)
        assert np.allclose(m[3:, :3], 0.0)
        assert np.allclose(m[3:, 3:], j_cg)

    def test_inertia_documented_as_about_the_cg(self) -> None:
        """Regression guard for the audited docstring defect."""
        doc = compute_mass_matrix.__doc__ or ""
        assert "centre of gravity" in doc.lower() or "center of gravity" in doc.lower()
        # the audited wording that caused the double parallel-axis transfer
        assert "inertia tensor about the origin" not in doc.lower()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))