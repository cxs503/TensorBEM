"""Gate: the two Wigley III builders must describe the *same* hull (m16935).

Mission m16892's geometry gate found that ``tensorbem.wigley3_hull`` placed the
fullness parameter on the vertical coordinate while
``tensorbem.wigley3_freq`` (the documented definition, and the builder behind
every published Wigley III number) places it on the longitudinal one.  Both
carry ``Cb = 0.543``, so the block coefficient -- the number everyone checks --
could not see the difference; heave added mass could (36 %).

This file is failure-loud in both directions:

* the builders must agree pointwise, on the mesh, and on the closed-form
  hydrostatics, and
* the retired recipe (fullness on ``zeta``, ``Cwp = 2/3``) must stay far
  outside the tolerances, so a silent revert shows up as a red test rather
  than as a 36 % physics drift.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem import wigley3_hull as wh
from tensorbem import wigley3_freq as wf

L, B, T, ALPHA = wh.WIGLEY3_L, wh.WIGLEY3_B, wh.WIGLEY3_T, wh.WIGLEY3_ALPHA
CWP_LONGITUDINAL = 2.0 / 3.0 + 2.0 * ALPHA / 15.0


def _grid(nx: int = 81, nz: int = 41):
    x = np.linspace(-0.5 * L, 0.5 * L, nx)
    z = np.linspace(-T, 0.0, nz)
    return np.meshgrid(x, z)


def test_the_two_builders_are_the_same_surface_pointwise() -> None:
    X, Z = _grid()
    hull = np.asarray(wh.wigley3_half_breadth(X, Z), dtype=float)
    freq = wf.wigley3_halfbreadth(X, Z, L, B, T)
    assert np.max(np.abs(hull - freq)) < 1e-15 * B


def test_waterplane_coefficient_is_the_documented_fuller_ends_value() -> None:
    X, Z = _grid(nx=2001, nz=3)
    y0 = np.asarray(wh.wigley3_half_breadth(X[0], np.zeros_like(X[0])), dtype=float)
    a_wp = 2.0 * np.trapezoid(y0, X[0]) if hasattr(np, "trapezoid") else 2.0 * np.trapz(y0, X[0])
    cwp = a_wp / (L * B)
    assert cwp == pytest.approx(CWP_LONGITUDINAL, rel=1e-6)
    # regression sentinel: the retired zeta-placed recipe gave Cwp = 2/3
    assert abs(cwp - 2.0 / 3.0) / (2.0 / 3.0) > 0.15


def test_Cb_Cm_Cwp_are_consistent_with_the_documented_dimensions() -> None:
    props = wh.compute_wigley3_properties()
    assert props.Cb == pytest.approx(0.543, abs=1e-12)
    # sections are the standard parabolic ones: the midship section area is
    # int_{-T}^{0} 2 y(0, z) dz = B T * 2/3, i.e. Cm = 2/3, so Cwp = Cb / Cm
    v, w = np.polynomial.legendre.leggauss(64)
    z = 0.5 * T * (v - 1.0)
    section = float(np.sum(0.5 * T * w * 2.0 * np.asarray(
        wh.wigley3_half_breadth(0.0, z), dtype=float)))
    cm = section / (B * T)
    assert cm == pytest.approx(2.0 / 3.0, rel=1e-12)
    assert props.waterplane_area / (L * B) == pytest.approx(props.Cb / cm, rel=1e-12)


def test_closed_form_hydrostatics_match_quadrature_of_the_same_surface() -> None:
    """Every closed form in ``compute_wigley3_properties`` vs Gauss-Legendre."""
    u, wu = np.polynomial.legendre.leggauss(200)
    for alpha in (0.0, ALPHA, 0.7):
        props = wh.compute_wigley3_properties(alpha=alpha)
        x = 0.5 * L * u
        z = 0.5 * T * (u - 1.0)
        X, Z = np.meshgrid(x, z, indexing="ij")
        W = np.outer(0.5 * L * wu, 0.5 * T * wu)
        y = np.asarray(wh.wigley3_half_breadth(X, Z, alpha=alpha), dtype=float)
        volume = float(np.sum(W * 2.0 * y))
        a_wp = float(np.sum(0.5 * L * wu * 2.0 * np.asarray(
            wh.wigley3_half_breadth(x, np.zeros_like(x), alpha=alpha))))
        y0 = np.asarray(wh.wigley3_half_breadth(x, np.zeros_like(x), alpha=alpha))
        i_xx = float(np.sum(0.5 * L * wu * (2.0 / 3.0) * y0**3))
        i_yy = float(np.sum(0.5 * L * wu * x**2 * 2.0 * y0))
        kb = T + float(np.sum(W * 2.0 * y * Z)) / volume
        assert props.volume == pytest.approx(volume, rel=1e-12)
        assert props.waterplane_area == pytest.approx(a_wp, rel=1e-12)
        assert props.Ixx == pytest.approx(i_xx, rel=1e-11)
        assert props.Iyy == pytest.approx(i_yy, rel=1e-11)
        assert props.KB == pytest.approx(kb, rel=1e-11)
    # the alpha = 0 limits are the classical parabolic Wigley values
    base = wh.compute_wigley3_properties(alpha=0.0)
    assert base.waterplane_area == pytest.approx(2.0 * B * L / 3.0, rel=1e-14)
    assert base.Ixx == pytest.approx(4.0 * B**3 * L / 105.0, rel=1e-13)
    assert base.Iyy == pytest.approx(B * L**3 / 30.0, rel=1e-13)
    assert base.KB == pytest.approx(5.0 * T / 8.0, rel=1e-14)


def test_mesh_volume_and_panel_count_are_the_documented_hull() -> None:
    mesh = wh.wigley3_wetted_hull_mesh(nx=20, nz=10)
    tri = np.asarray(mesh.vertices, dtype=float)[np.asarray(mesh.faces, dtype=np.int64)]
    volume = float(np.sum(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))) / 6.0)
    assert volume == pytest.approx(wh.wigley3_displaced_volume(), rel=2e-2)
    assert wh.wigley3_displaced_volume() == pytest.approx(L * B * T * 0.543, rel=1e-12)


def test_cross_builder_radiation_agrees_now_that_the_geometry_agrees() -> None:
    """m16892 recorded a 36 % heave added-mass split between the builders.

    Same surface, same solver, same panel count -> the split must collapse to
    discretisation (the two meshes differ in node layout, not in geometry).
    """
    from tensorbem.benchmarks.wigley3_recalled_ref import (
        DENSITY,
        GRAVITY,
        OMEGA_STATED,
        build_mesh,
    )
    from tensorbem.free_surface_solver3d import solve_radiation_3d

    # 180 (hull) / 200 (freq) panels: the two builders use different node
    # layouts, so this is a discretisation-level cross-check, not an identity.
    values = {}
    for builder in ("wigley3_hull", "wigley3_freq"):
        mesh = build_mesh(builder, 10)
        res = solve_radiation_3d(
            mesh, OMEGA_STATED**2 / GRAVITY, modes=(2,), density=DENSITY,
            gravity=GRAVITY, center=(0.0, 0.0, -T / 2.0),
        )
        values[builder] = (float(res.added_mass[0, 0]), float(res.damping[0, 0]))
    scale = DENSITY * wh.wigley3_displaced_volume()
    a33 = {k: v[0] / scale for k, v in values.items()}
    b33 = {k: v[1] / (scale * OMEGA_STATED) for k, v in values.items()}
    assert a33["wigley3_freq"] == pytest.approx(a33["wigley3_hull"], rel=0.03)
    assert b33["wigley3_freq"] == pytest.approx(b33["wigley3_hull"], rel=0.03)
    assert a33["wigley3_hull"] > 1.5
