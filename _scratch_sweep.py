"""Independent oscillatory-aware referee for PV int_0^inf e^{-s s} cos(X s)/(s-1) ds
(breakpoints forced at s = 1, 2 and every oscillation cell) + accuracy sweep."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import exp1
import sys
sys.path.insert(0, "/root/tp-agent/work/m16829/src")
from tensorbem import free_surface


def ref_pv(sigma, X, nper=8, n_q=40, decay=40.0):
    """Panelled Gauss-Legendre; panel edges forced at s = 1 and s = 2."""
    x = abs(X)
    f1 = np.exp(-sigma) * np.cos(x)
    df1 = -sigma * f1 - x * np.exp(-sigma) * np.sin(x)
    s_hi = 2.0 + decay / sigma
    period = 2 * np.pi / x if x > 1e-9 else np.inf
    width = min(period / nper, 0.25)
    n_pan = max(4, int(np.ceil((s_hi - 2.0) / width)) + 2)
    edges = list(np.linspace(0.0, s_hi, n_pan + 1))
    edges = sorted(set([round(e, 15) for e in edges] + [1.0, 2.0]))
    edges = np.array(edges)
    nodes, weights = leggauss(n_q)
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        half = 0.5 * (b - a)
        mid = 0.5 * (b + a)
        s = mid + half * nodes
        f = np.exp(-sigma * s) * np.cos(x * s)
        num = f - (f1 if b <= 2.0 else 0.0)
        den = s - 1.0
        safe = np.abs(den) > 1e-9
        integrand = np.where(safe, num / np.where(safe, den, 1.0), df1)
        total += half * float(np.sum(weights * integrand))
    return total


def head_lib(sigma, X, nleg):
    nodes, weights = leggauss(nleg)
    ref = np.exp(-sigma) * np.cos(X)
    k = 1.0 + nodes          # nu = 1, so k = nu*(1+t), t in [-1, 1]
    diff = np.exp(-k * sigma) * np.cos(k * X) - ref
    return float(np.sum(weights * diff / nodes))


def tail_closed(sigma, X):
    b = complex(sigma, -X)
    return float(np.real(np.exp(-b) * exp1(b)))


def tail_lag(sigma, X, n):
    from numpy.polynomial.laguerre import laggauss
    xw, ww = laggauss(n)
    t = xw / sigma
    integ = np.cos((2.0 + t) * X) / (1.0 + t)
    return float(np.exp(-2.0 * sigma) / sigma * np.sum(ww * integ))


if __name__ == "__main__":
    print(f"{'sigma':>6} {'X':>7} {'ref':>20} {'hyb80':>10} {'hyb320':>10} {'lib(now)':>10} {'lag160':>10}")
    for sigma in (0.05, 0.48, 1.0, 5.0, 50.0):
        for X in (0.0, 1.0, 3.0, 5.3, 9.4, 13.3, 30.0, 60.0, 120.0):
            r = ref_pv(sigma, X)
            v80 = head_lib(sigma, X, 80) + tail_closed(sigma, X)
            v320 = head_lib(sigma, X, 320) + tail_closed(sigma, X)
            lib_now = free_surface.principal_value_integral(1.0, sigma, X)
            lag = head_lib(sigma, X, 80) + tail_lag(sigma, X, 160)
            dev = lambda v: abs(v - r) / max(abs(r), 1e-300)
            print(f"{sigma:>6} {X:>7} {r:>20.12f} {dev(v80):>10.2e} {dev(v320):>10.2e} "
                  f"{dev(lib_now):>10.2e} {dev(lag):>10.2e}")