"""Verification tests for the 3-D Kelvin free-surface Green function and hull meshes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.analytic import exponential_integral, struve_h01
from tensorbem.free_surface3d import (
    green_function_3d,
    green_gradient_field_3d,
    green_gradient_source_3d,
    pure_wave_value_and_source_gradient_3d,
    verify_free_surface_bc_3d,
    verify_laplace_3d,
    verify_radiation_condition_3d,
    wave_function_3d,
)
from tensorbem.geometry3d import (
    WettedHullMesh,
    hemisphere_hull_mesh,
    icosphere,
    submerged_sphere_mesh,
)


def test_struve_h01_identities_and_seam_continuity() -> None:
    h0_zero, h1_zero = struve_h01(0.0)
    assert float(h0_zero) == 0.0
    assert float(h1_zero) == 0.0

    # Direct seam continuity across x = 8.0 to ~1e-12
    seam_l0, seam_l1 = struve_h01(8.0 - 1.0e-12)
    seam_r0, seam_r1 = struve_h01(8.0 + 1.0e-12)
    assert abs(float(seam_r0) - float(seam_l0)) < 5.0e-12
    assert abs(float(seam_r1) - float(seam_l1)) < 5.0e-12

    eps = 1.0e-5
    left_0, left_1 = struve_h01(8.0 - eps)
    right_0, right_1 = struve_h01(8.0 + eps)
    mid_0, mid_1 = struve_h01(8.0)
    # Central difference derivative check across the x = 8 seam:
    # H_0'(x) = 2/pi - H_1(x),  d/dx(x * H_1(x)) = x * H_0(x) => H_1'(x) = H_0(x) - H_1(x)/x
    dh0_fd = (float(right_0) - float(left_0)) / (2.0 * eps)
    dh1_fd = (float(right_1) - float(left_1)) / (2.0 * eps)
    assert abs(dh0_fd - (2.0 / math.pi - float(mid_1))) < 5.0e-9
    assert abs(dh1_fd - (float(mid_0) - float(mid_1) / 8.0)) < 5.0e-9

    xs = np.array([0.4, 2.5, 7.999999, 8.000001, 15.0, 42.0])
    h0, h1 = struve_h01(xs)
    assert abs(h0[2] - h0[3]) < 5.0e-6
    assert abs(h1[2] - h1[3]) < 5.0e-6
    for xv in (0.7, 5.2, 14.0):
        step = 1.0e-6
        p0, p1 = struve_h01(xv + step)
        m0, m1 = struve_h01(xv - step)
        c0, c1 = struve_h01(xv)
        assert abs((float(p0 - m0) / (2.0 * step)) - (2.0 / math.pi - float(c1))) < 5.0e-10
        assert abs((float(p1 - m1) / (2.0 * step)) - (float(c0) - float(c1) / xv)) < 5.0e-10

    with pytest.raises(ValueError, match="non-negative"):
        struve_h01(-0.1)


def test_wave_function_3d_seams_axis_and_ode() -> None:
    # Exact vertical axis X = 0 across both shallow (Y <= 4.5) and deep (Y > 4.5) regimes:
    for y_val in (0.4, 2.5, 4.5, 7.0, 20.0):
        f_v, fx_v, fy_v = wave_function_3d(0.0, y_val)
        exact_f = -math.exp(-y_val) * exponential_integral(y_val)
        assert abs(float(f_v) - exact_f) < 1.0e-12
        assert abs(float(fx_v)) == 0.0
        assert abs(float(fy_v) + float(f_v) + 1.0 / y_val) < 1.0e-14

    # Seam 1: near-axis cone X = 0.35 * Y for Y in (0, 4.5]
    delta = 1.0e-7
    ys_cone = np.array([0.5, 2.0, 4.2])
    f_in, fx_in, _ = wave_function_3d((0.35 - delta) * ys_cone, ys_cone)
    f_out, fx_out, _ = wave_function_3d((0.35 + delta) * ys_cone, ys_cone)
    assert np.max(np.abs(f_in - f_out)) < 5.0e-7
    assert np.max(np.abs(fx_in - fx_out)) < 5.0e-7

    # Seam 2: horizontal depth switch Y = 4.5
    xs_depth = np.array([0.2, 1.5, 4.0, 12.0])
    f_lo, fx_lo, _ = wave_function_3d(xs_depth, np.full_like(xs_depth, 4.5 - delta))
    f_hi, fx_hi, _ = wave_function_3d(xs_depth, np.full_like(xs_depth, 4.5 + delta))
    assert np.max(np.abs(f_lo - f_hi)) < 5.0e-7
    assert np.max(np.abs(fx_lo - fx_hi)) < 5.0e-7

    # Seam 3: far-field switch D = 26.0 (Y > 4.5, X > 0.25)
    angles = np.array([0.22, 0.6, 1.1])
    r_in = 26.0 - delta
    r_out = 26.0 + delta
    f_near, fx_near, _ = wave_function_3d(r_in * np.sin(angles), r_in * np.cos(angles))
    f_far, fx_far, _ = wave_function_3d(r_out * np.sin(angles), r_out * np.cos(angles))
    assert np.max(np.abs(f_near - f_far)) < 5.0e-7
    assert np.max(np.abs(fx_near - fx_far)) < 5.0e-7

    with pytest.raises(ValueError, match="singular"):
        wave_function_3d(0.0, 0.0)


def test_green_function_3d_gradients_bc_laplace_and_sommerfeld() -> None:
    nu = 1.35
    pts = np.array([[0.35, -0.42, -0.25], [0.0, 0.0, -0.6], [2.4, -1.8, -1.7]])
    srcs = np.array([[-0.15, 0.28, -0.7], [0.0, 0.0, -1.9], [-0.8, 0.9, -2.5]])

    # Symmetry G(x, xi) == G(xi, x) and outgoing/incoming conjugation
    g_out = green_function_3d(pts, srcs, nu, outgoing=True)
    g_swap = green_function_3d(srcs, pts, nu, outgoing=True)
    g_in = green_function_3d(pts, srcs, nu, outgoing=False)
    assert np.allclose(g_out, g_swap, rtol=1.0e-13, atol=1.0e-14)
    assert np.allclose(g_in, np.conj(g_out), rtol=1.0e-14, atol=1.0e-15)

    # Fused pure-wave helper matches include_rankine=False, include_image=False
    gw_sep = green_function_3d(pts, srcs, nu, include_rankine=False, include_image=False)
    grad_sep = green_gradient_source_3d(
        pts, srcs, nu, include_rankine=False, include_image=False
    )
    gw_fused, grad_fused = pure_wave_value_and_source_gradient_3d(pts, srcs, nu)
    assert np.allclose(gw_sep, gw_fused, rtol=1.0e-14, atol=1.0e-15)
    assert np.allclose(grad_sep, grad_fused, rtol=1.0e-14, atol=1.0e-15)

    # Central finite-difference verification of both field and source gradients
    step = 1.0e-6
    for tgt, src in zip(pts, srcs):
        grad_f = green_gradient_field_3d(tgt, src, nu)
        grad_s = green_gradient_source_3d(tgt, src, nu)
        fd_f = np.empty(3, dtype=np.complex128)
        fd_s = np.empty(3, dtype=np.complex128)
        for axis in range(3):
            shift = np.zeros(3, dtype=float)
            shift[axis] = step
            fd_f[axis] = (
                green_function_3d(tgt + shift, src, nu)
                - green_function_3d(tgt - shift, src, nu)
            ) / (2.0 * step)
            fd_s[axis] = (
                green_function_3d(tgt, src + shift, nu)
                - green_function_3d(tgt, src - shift, nu)
            ) / (2.0 * step)
        assert np.max(np.abs(grad_f - fd_f)) < 1.0e-9
        assert np.max(np.abs(grad_s - fd_s)) < 1.0e-9
        # Horizontal translation invariance holds, vertical does NOT:
        assert abs(grad_f[0] + grad_s[0]) < 1.0e-13
        assert abs(grad_f[1] + grad_s[1]) < 1.0e-13
        assert abs(grad_f[2] + grad_s[2]) > 1.0e-3

    # Exact Kelvin free-surface boundary condition dG/dz - nu * G = 0 on z = 0
    xy = np.array([[0.0, 0.0], [0.25, -0.4], [2.5, 1.8], [22.0, -14.0]])
    bc_res = verify_free_surface_bc_3d(xy, [0.2, -0.3, -0.8], nu=1.4)
    assert np.max(np.abs(bc_res)) < 1.0e-14

    # Laplace PDE away from source
    lap_res = verify_laplace_3d([0.7, -0.4, -0.9], [-0.2, 0.3, -0.5], nu=1.2)
    assert abs(lap_res) < 1.0e-7

    # Sommerfeld outgoing cylindrical radiation decay O(R^{-1})
    radii = np.array([5.0, 20.0, 80.0, 320.0])
    sommerfeld = verify_radiation_condition_3d(radii, 0.6, -0.4, [0.1, -0.2, -0.7], 1.1)
    assert sommerfeld[-1] < 5.0e-4
    assert sommerfeld[0] / sommerfeld[-1] > 50.0

    with pytest.raises(ValueError, match="z <= 0"):
        green_function_3d([0.0, 0.0, 0.1], [0.0, 0.0, -1.0], nu=1.0)
    with pytest.raises(ValueError, match="singular"):
        green_function_3d([0.0, 0.0, -1.0], [0.0, 0.0, -1.0], nu=1.0)


def test_wetted_hull_and_submerged_sphere_mesh_validation() -> None:
    hemi2 = hemisphere_hull_mesh(2, radius=1.5)
    hemi3 = hemisphere_hull_mesh(3, radius=1.5)
    exact_vol = (2.0 / 3.0) * math.pi * (1.5**3)
    exact_wp = math.pi * (1.5**2)
    assert hemi2.panel_count == 64
    assert hemi3.panel_count == 256
    assert abs(hemi3.volume - exact_vol) < 0.27 * abs(hemi2.volume - exact_vol)
    assert abs(hemi3.waterplane_area - exact_wp) < 0.27 * abs(hemi2.waterplane_area - exact_wp)

    # Reversed orientation is rejected
    with pytest.raises(ValueError, match="positive displaced volume"):
        WettedHullMesh(hemi2.vertices, hemi2.faces[:, [0, 2, 1]])

    # Open boundary below z = 0 is rejected
    shifted = hemi2.vertices.copy()
    shifted[:, 2] -= 0.5
    with pytest.raises(ValueError, match="waterplane z = 0"):
        WettedHullMesh(shifted, hemi2.faces)

    # Submerged sphere validation
    sph = submerged_sphere_mesh(1, radius=1.0, submergence=1.5)
    assert sph.panel_count == 80
    assert np.all(sph.vertices[:, 2] <= -0.5 + 1.0e-12)
    with pytest.raises(ValueError, match="strictly greater than radius"):
        submerged_sphere_mesh(1, radius=1.0, submergence=1.0)
    assert icosphere(1).panel_count == 80
