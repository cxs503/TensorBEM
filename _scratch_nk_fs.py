"""Scratch: is the tensorbem.neumann_kelvin wave kernel the steady NK kernel?

Checks (numerically, first principles):
  1. correct PV evaluation of  I(mu,q) = PV int_0^inf e^{-mu u} J0(q u)/(u-1) du
     via the convergent (log-divergence-subtracted) decomposition, anchored to mpmath;
  2. the free-surface condition of the *steady* NK problem for G = G_R + G_W:
        K * G_z + G_xx = 0  on z = 0^-     (K = g/U^2, i.e. phi_z = -(1/K) phi_xx)
     and for reference the frequency-domain deep-water condition
        G_z - k0 G = 0 ... (radial) is not needed; only the residual matters
  3. Kelvin far-field: transverse wavenumber should be K (wavelength 2 pi / K)
     and the wake half angle should be 19.47 deg.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import j0 as scipy_j0
from mpmath import mp, mpf, quad, inf, exp as mexp, besselj, euler

mp.dps = 30

GAMMA = float(euler)

# --- Gauss rules -----------------------------------------------------------
_L1_X, _L1_W = np.polynomial.legendre.leggauss(80)          # on [-1, 1] -> [0, 1]
# map [0, inf) with u = a*(1+s)/(1-s), s in [-1,1]
_S_X, _S_W = np.polynomial.legendre.leggauss(120)


def pv_integral(mu: float, q: float, a: float = 4.0) -> float:
    """PV int_0^inf e^{-mu u} J0(q u)/(u-1) du  (convergent decomposition).

    PV int_0^inf f/(u-1) du
      = -int_0^1 [f(u)-f(1)]/(1-u) du
        + int_0^inf [f(1+v)-f(1) e^{-v}]/v dv  - gamma f(1),      f(u)=e^{-mu u}J0(q u)
    """
    f1 = math.exp(-mu) * scipy_j0(q)

    # near part on [0,1]
    u = 0.5 * (_L1_X + 1.0)
    w = 0.5 * _L1_W
    f = np.exp(-mu * u) * scipy_j0(q * u)
    near = -float(np.sum(w * (f - f1) / (1.0 - u)))

    # far part on [0, inf) via u = a(1+s)/(1-s)
    s = _S_X
    v = a * (1.0 + s) / (1.0 - s)
    dvds = 2.0 * a / (1.0 - s) ** 2
    f = np.exp(-mu * (1.0 + v)) * scipy_j0(q * (1.0 + v))
    g = np.where(v < 1e-12, (f1 * (-mu) + f1), (f - f1 * np.exp(-v)) / np.maximum(v, 1e-300))
    far = float(np.sum(_S_W * dvds * g))

    return near + far - GAMMA * f1


def pv_mpmath(mu: float, q: float) -> float:
    """Independent high-precision PV: analytic subtraction of the log divergence."""
    mu_, q_ = mpf(mu), mpf(q)
    f = lambda u: mexp(-mu_ * u) * besselj(0, q_ * u)
    f1 = f(mpf(1))

    def near():
        # -int_0^1 (f(u)-f1)/(1-u) du ; integrand regular at u=1
        return -quad(lambda u: (f(u) - f1) / (1 - u) if u != 1 else -(mu_ * f1), [0, 1])

    def far():
        # int_0^inf (f(1+v) - f1 e^{-v})/v dv ; regular at v=0, decays at inf
        return quad(lambda v: (f(1 + v) - f1 * mexp(-v)) / v if v != 0 else (-mu_ * f1 + f1),
                    [0, 20, inf])

    return float(near() + far() - euler * f1)


def g_wave(R, z_sum, kappa):
    mu = kappa * abs(z_sum)
    q = kappa * R
    return (kappa / (2 * math.pi)) * pv_integral(mu, q)


def g_rankine(x, y, z, zeta):
    R = math.hypot(x, y)
    r = math.hypot(R, z - zeta)
    r1 = math.hypot(R, z + zeta)
    return 1.0 / (4 * math.pi * r) + 1.0 / (4 * math.pi * r1)


if __name__ == "__main__":
    print("== 1. PV integral vs mpmath high precision ==")
    for (mu, q) in [(0.5, 1.0), (2.0, 3.0), (0.1, 0.2), (5.0, 0.5)]:
        a, b = pv_integral(mu, q), pv_mpmath(mu, q)
        print(f"  mu={mu:4.2f} q={q:4.2f}  ours={a: .12e}  mp={b: .12e}  relerr={abs(a-b)/max(abs(b),1e-300):.2e}")

    print("== 2. steady NK free-surface residual  K*G_z + G_xx = 0 on z=0 ==")
    kappa = 5.0            # = g/U^2  [1/m]
    zeta = -0.6            # source depth
    hz = 1e-6
    G = lambda x, y, z: g_rankine(x, y, z, zeta) + g_wave(math.hypot(x, y), z + zeta, kappa)
    for (x, y) in [(1.0, 0.0), (2.0, 0.5), (-1.5, 0.3), (0.5, 1.2)]:
        Gz = (G(x, y, -hz) - G(x, y, -2 * hz)) / hz          # one-sided, z<0
        Gxx = (G(x * (1 + 1e-4), y, 0.0) - 2 * G(x, y, 0.0) + G(x * (1 - 1e-4), y, 0.0)) / (x * 1e-4) ** 2
        res = kappa * Gz + Gxx
        scale = abs(kappa * Gz) + abs(Gxx)
        print(f"  x={x:5.1f} y={y:4.1f}  K*Gz={kappa*Gz: .4e}  Gxx={Gxx: .4e}  resid={res: .4e}  rel={abs(res)/scale:.3e}")

    print("== 3. frequency-domain style radial condition: is G_w ~ J0(k0 R) exp(k0 z)? ==")
    for R in [1.0, 5.0]:
        print(f"  R={R}: G_w = {g_wave(R, -0.5, kappa): .6e}")