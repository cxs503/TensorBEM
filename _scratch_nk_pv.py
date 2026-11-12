"""Scratch 2: closed-form anchor for the PV integral and accuracy envelope."""
from __future__ import annotations
import math, time
import numpy as np
from scipy.special import j0 as spj0
from mpmath import mp, mpf, quad, inf, exp as mexp, besselj, euler

mp.dps = 20
GAMMA = 0.5772156649015328606


def pv(mu, q, n_near=160, n_far=160, a=None):
    """PV int_0^inf e^{-mu u} J0(q u)/(u-1) du via convergent decomposition."""
    f1 = math.exp(-mu) * float(spj0(q))
    x, w = np.polynomial.legendre.leggauss(n_near)
    u = 0.5 * (x + 1.0)
    f = np.exp(-mu * u) * spj0(q * u)
    near = -float(np.sum(0.5 * w * (f - f1) / (1.0 - u)))
    if a is None:
        a = 1.0 / max(mu, 1e-6)
    xs, ws = np.polynomial.legendre.leggauss(n_far)
    v = a * (1.0 + xs) / (1.0 - xs)
    dv = 2.0 * a / (1.0 - xs) ** 2
    fv = np.exp(-mu * (1.0 + v)) * spj0(q * (1.0 + v))
    g = np.where(v < 1e-14, -mu * f1 + f1, (fv - f1 * np.exp(-v)) / np.maximum(v, 1e-300))
    far = float(np.sum(ws * dv * g))
    return near + far - GAMMA * f1


def pv_mp(mu, q):
    mu_, q_ = mpf(mu), mpf(q)
    f = lambda u: mexp(-mu_ * u) * besselj(0, q_ * u)
    f1 = f(mpf(1))
    return float(-quad(lambda u: (f(u) - f1) / (1 - u), [-inf, 0, 1])
                 + quad(lambda v: (f(1 + v) - f1 * mexp(-v)) / v, [0, 1, 10, inf])
                 - euler * f1)


def ein_series(z, n=200):
    """Ein(z) = sum_{k>=1} (-1)^{k+1} z^k /(k k!)"""
    s, term = 0.0, z
    for k in range(1, n + 1):
        s += ((-1) ** (k + 1)) * term / k
        term *= z / (k + 1)
    return s


# closed form for f = e^{-mu u} (i.e. q = 0):
#   PV int_0^inf e^{-mu u}/(u-1) du = e^{-mu} [ Ein(-mu) - ln mu - gamma ]
def pv_closed(mu):
    return math.exp(-mu) * (ein_series(-mu) - math.log(mu) - GAMMA)


print("== closed form check (q=0) ==")
for mu in [0.05, 0.5, 1.0, 3.0, 10.0]:
    a, b = pv(mu, 0.0), pv_closed(mu)
    c = pv_mp(mu, 0.0)
    print(f"  mu={mu:5.2f} quad={a: .14e} closed={b: .14e} mp={c: .14e} "
          f"dq={abs(a-b)/max(abs(b),1e-300):.1e} dm={abs(a-c)/max(abs(c),1e-300):.1e}")

print("== accuracy envelope vs mpmath, a = 1/mu ==")
t0 = time.time()
worst = 0.0
for mu in [1e-2, 0.1, 1.0, 5.0]:
    for q in [0.0, 2.0, 10.0]:
        a, b = pv(mu, q), pv_mp(mu, q)
        rel = abs(a - b) / max(abs(b), 1e-12)
        worst = max(worst, rel)
        if rel > 1e-10:
            print(f"  BAD mu={mu:8.1e} q={q:6.2f} ours={a: .12e} mp={b: .12e} rel={rel:.1e}")
print(f"  worst rel over grid = {worst:.2e}  ({time.time()-t0:.1f}s)")

print("== large q behaviour (oscillation) ==")
for q in [100.0, 400.0, 1000.0]:
    for n in [160, 640, 2000]:
        a = pv(5.0, q, n_near=n, n_far=n)
        print(f"  q={q:7.1f} n={n:5d} val={a: .10e}")