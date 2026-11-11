"""Scratch: verify standard steady NK Green function + Havelock Cw vs Michell (small mesh)."""
import math
import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.laguerre import laggauss
from tensorbem.analytic import bessel_j01_y01

_LEG64_X, _LEG64_W = leggauss(64)
_LAG64_X, _LAG64_W = laggauss(64)
_TWO_PI = 2.0 * math.pi
_G = 9.80665
_RHO = 1025.0


def _j0j1(x):
    x = np.asarray(x, dtype=float)
    j0 = np.ones_like(x)
    j1 = np.zeros_like(x)
    pos = x > 1e-12
    if np.any(pos):
        j0p, j1p, _, _ = bessel_j01_y01(x[pos])
        j0[pos] = j0p
        j1[pos] = j1p
    return j0, j1


def nk_wave_kernel_std(targets, sources, kappa, n_leg=16, n_lag=16, chunk=64):
    """Standard steady NK wave kernel G_W and gradients (M,N).

    G_W = (kappa/2pi) * PV int_0^inf exp(-kappa u |z+zeta|) J_0(kappa u R) / (u-1) du
    """
    tgt = np.atleast_2d(np.asarray(targets, dtype=float))
    src = np.atleast_2d(np.asarray(sources, dtype=float))
    M, N = tgt.shape[0], src.shape[0]

    dx = tgt[:, 0][:, None] - src[None, :, 0]
    dy = tgt[:, 1][:, None] - src[None, :, 1]
    zsum = tgt[:, 2][:, None] + src[None, :, 2]
    abs_z = np.abs(zsum)
    sign_z = np.sign(zsum)
    R = np.hypot(dx, dy)
    R = np.maximum(R, 1e-300)

    kR = kappa * R
    kz = kappa * abs_z

    eps = 1e-8
    u_near = 0.5 * (1.0 - eps) * (_LEG64_X[:n_leg] + 1.0)
    w_near = 0.5 * (1.0 - eps) * _LEG64_W[:n_leg]
    u_lag = 1.0 + eps + _LAG64_X[:n_lag]
    w_lag = _LAG64_W[:n_lag]
    u = np.concatenate([u_near, u_lag])
    w = np.concatenate([w_near, w_lag])
    denom = u - 1.0
    Q = u.size

    G = np.zeros((M, N))
    Gx = np.zeros((M, N))
    Gy = np.zeros((M, N))
    Gz = np.zeros((M, N))
    for i0 in range(0, M, chunk):
        sl = slice(i0, min(i0 + chunk, M))
        kR_u = kR[sl][:, :, None] * u[None, None, :]      # (chunk,N,Q)
        kz_u = kz[sl][:, :, None] * u[None, None, :]
        exp_kz = np.exp(-kz_u)
        j0, j1 = _j0j1(kR_u.ravel())
        j0 = j0.reshape(kR_u.shape)
        j1 = j1.reshape(kR_u.shape)
        wgt = w[None, None, :] / denom[None, None, :]
        G[sl] = (kappa / _TWO_PI) * np.sum(exp_kz * j0 * wgt, axis=2)
        Gx[sl] = (kappa / _TWO_PI) * np.sum(
            exp_kz * (-kappa * u[None, None, :] * j1) * (dx[sl][:, :, None] / R[sl][:, :, None]) * wgt, axis=2)
        Gy[sl] = (kappa / _TWO_PI) * np.sum(
            exp_kz * (-kappa * u[None, None, :] * j1) * (dy[sl][:, :, None] / R[sl][:, :, None]) * wgt, axis=2)
        Gz[sl] = (kappa / _TWO_PI) * np.sum(
            exp_kz * (-kappa * u[None, None, :] * sign_z[sl][:, :, None]) * j0 * wgt, axis=2)
    return G, Gx, Gy, Gz


def rankine_kernel_std(field, src, mirror=True):
    dx = field[:, 0] - src[:, 0]
    dy = field[:, 1] - src[:, 1]
    dz = field[:, 2] - src[:, 2]
    r = np.maximum(np.sqrt(dx * dx + dy * dy + dz * dz), 1e-300)
    G = 1.0 / (4.0 * math.pi * r)
    inv3 = 1.0 / r**3
    Gx = -dx / (4.0 * math.pi) * inv3
    Gy = -dy / (4.0 * math.pi) * inv3
    Gz = -dz / (4.0 * math.pi) * inv3
    if mirror:
        dzi = field[:, 2] + src[:, 2]
        ri = np.maximum(np.sqrt(dx * dx + dy * dy + dzi * dzi), 1e-300)
        G = G + 1.0 / (4.0 * math.pi * ri)
        inv3i = 1.0 / ri**3
        Gx = Gx - dx / (4.0 * math.pi) * inv3i
        Gy = Gy - dy / (4.0 * math.pi) * inv3i
        Gz = Gz - dzi / (4.0 * math.pi) * inv3i
    return G, Gx, Gy, Gz


def total_kernel_std(field, src, kappa, n_leg=16, n_lag=16):
    G0, Gx0, Gy0, Gz0 = rankine_kernel_std(field, src, mirror=True)
    Gw, Gxw, Gyw, Gzw = nk_wave_kernel_std(field, src, kappa, n_leg, n_lag)
    return G0 + Gw, Gx0 + Gxw, Gy0 + Gyw, Gz0 + Gzw


def verify_fs_condition(kappa=1.0, n_leg=16, n_lag=16):
    src = np.array([[0.0, 0.0, -0.3 / kappa]])
    hs = 1e-3 / kappa
    eps = 1e-7 / kappa
    res = []
    for xr in (0.6, 1.2, 2.0):
        for yr in (0.0, 0.7):
            x = xr / kappa
            y = yr / kappa
            base = np.array([[x, y, 0.0]])
            G0 = total_kernel_std(base, src, kappa, n_leg, n_lag)[0].ravel()[0]
            Gp = total_kernel_std(np.array([[x + hs, y, 0.0]]), src, kappa, n_leg, n_lag)[0].ravel()[0]
            Gm = total_kernel_std(np.array([[x - hs, y, 0.0]]), src, kappa, n_leg, n_lag)[0].ravel()[0]
            Gd = total_kernel_std(np.array([[x, y, -eps]]), src, kappa, n_leg, n_lag)[0].ravel()[0]
            gxx = (Gp - 2.0 * G0 + Gm) / hs**2
            gz = (G0 - Gd) / eps
            scale = max(abs(gxx), abs(kappa * gz), 1e-30)
            res.append(abs(gxx + kappa * gz) / scale)
    return {"max_relative_residual": float(max(res)), "mean": float(np.mean(res))}


def wigley_half_panels(L, B, T, nx, nz):
    thx = np.linspace(0, np.pi, nx + 1)
    x_st = -L / 2 + L / 2 * (1 - np.cos(thx))
    z_lv = np.linspace(-T, 0.0, nz + 1)
    cen_list, nor_list, area_list = [], [], []
    for i in range(nx):
        for j in range(nz):
            x0, x1 = x_st[i], x_st[i + 1]
            z0, z1 = z_lv[j], z_lv[j + 1]
            def hb(x, z):
                return 0.5 * B * max(0.0, 1 - (2 * x / L) ** 2) * max(0.0, 1 - (z / T) ** 2)
            y00 = hb(x0, z0); y01 = hb(x0, z1); y10 = hb(x1, z0); y11 = hb(x1, z1)
            cx = 0.25 * (x0 + x0 + x1 + x1)
            cz = 0.25 * (z0 + z0 + z1 + z1)
            cy = 0.25 * (y00 + y01 + y10 + y11)
            dy_dx = -4.0 * B * cx / L**2 * (1 - (cz / T) ** 2)
            dy_dz = -B * (1 - (2 * cx / L) ** 2) * (cz / T**2)
            n = np.array([-dy_dx, 1.0, -dy_dz])
            n = n / np.linalg.norm(n)
            dx = x1 - x0
            dz = z1 - z0
            area = dx * dz * np.sqrt(1 + dy_dx**2 + dy_dz**2)
            cen_list.append([cx, cy, cz])
            nor_list.append(n)
            area_list.append(area)
    n0 = len(cen_list)
    for k in range(n0):
        cen_list.append([cen_list[k][0], -cen_list[k][1], cen_list[k][2]])
        nor_list.append([-nor_list[k][0], -nor_list[k][1], -nor_list[k][2]])
        area_list.append(area_list[k])
    return (np.array(cen_list), np.array(nor_list), np.array(area_list))


def solve_panel_std(cen, nor, area, U, kappa, n_leg=16, n_lag=16):
    F = cen.shape[0]
    G0, Gx0, Gy0, Gz0 = rankine_kernel_std(cen, cen, mirror=True)
    S = G0 * area[None, :]
    K = (Gx0 * nor[:, 0][:, None] + Gy0 * nor[:, 1][:, None] + Gz0 * nor[:, 2][:, None]) * area[None, :]
    Gw, Gxw, Gyw, Gzw = nk_wave_kernel_std(cen, cen, kappa, n_leg, n_lag)
    S = S + Gw * area[None, :]
    K = K + (Gxw * nor[:, 0][:, None] + Gyw * nor[:, 1][:, None] + Gzw * nor[:, 2][:, None]) * area[None, :]
    src_m = cen.copy()
    src_m[:, 1] *= -1.0
    Gwm, Gxwm, Gywm, Gzwm = nk_wave_kernel_std(cen, src_m, kappa, n_leg, n_lag)
    S = S + Gwm * area[None, :]
    K = K + (Gxwm * nor[:, 0][:, None] + Gywm * nor[:, 1][:, None] + Gzwm * nor[:, 2][:, None]) * area[None, :]
    A = -0.5 * np.eye(F) + K
    rhs = U * nor[:, 0]
    sigma = np.linalg.solve(A, rhs)
    bc = float(np.linalg.norm(A @ sigma - rhs) / max(np.linalg.norm(rhs), 1e-30))
    return sigma, bc


def kochin_std(sigma, cen, area, k0, k_grid):
    H = np.zeros(k_grid.size, dtype=complex)
    for i, k in enumerate(k_grid):
        H[i] = np.sum(sigma * area * np.exp(k * cen[:, 2] + 1j * k * cen[:, 0]))
    return H


def havelock_cw(sigma, cen, area, U, rho, g, L, T, n_quad=256):
    k0 = g / U**2
    lam_max = max(5.0, 10.0 / (k0 * T))
    nodes, weights = leggauss(n_quad)
    s_max = math.sqrt(lam_max - 1.0)
    s = 0.5 * s_max * (nodes + 1.0)
    w = 0.5 * s_max * weights
    lam = 1.0 + s * s
    H = kochin_std(sigma, cen, area, k0, lam * k0)
    integrand = np.abs(H) ** 2 * 2.0 / (lam**2 * np.sqrt(2.0 + s * s))
    Rw = (4.0 * rho / U**2) * float(np.sum(w * integrand))
    return Rw / (0.5 * rho * U**2 * L * T)


if __name__ == "__main__":
    print("=== free-surface condition ===")
    print(verify_fs_condition())

    print("=== Wigley Cw vs Michell ===")
    L, B, T = 10.0, 1.0, 0.625
    from tensorbem import wigley
    for fn in (0.2, 0.3, 0.4):
        U = fn * math.sqrt(_G * L)
        kappa = _G / U**2
        cen, nor, area = wigley_half_panels(L, B, T, 16, 8)
        print(f"  panels={cen.shape[0]}")
        sigma, bc = solve_panel_std(cen, nor, area, U, kappa)
        cw_panel = havelock_cw(sigma, cen, area, U, _RHO, _G, L, T)
        cw_m = wigley.michell_wave_resistance(L, B, T, U, _RHO, _G) / (0.5 * _RHO * U**2 * L * T)
        print(f"  Fr={fn}: bc={bc:.2e} Cw_panel={cw_panel:.3e} Cw_michell={cw_m:.3e} err={abs(cw_panel-cw_m)/cw_m*100:.1f}%")
