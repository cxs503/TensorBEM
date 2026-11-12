"""High-precision referee (mpmath, 30 digits) for the 2-D PV integral + tail closed form."""
import mpmath as mp

mp.mp.dps = 30


def f_pv(sigma, X):
    """PV int_0^inf e^{-sigma s} cos(X s)/(s-1) ds by pole-split + tanh-sinh."""
    f1 = mp.e ** (-sigma) * mp.cos(X)

    def reg(s):
        f = mp.e ** (-sigma * s) * mp.cos(X * s)
        return (f - f1) / (s - 1)

    head = mp.quad(reg, [0, 1, 2])
    # s = 2 + t/(1-t)
    def tail_fun(t):
        s = 2 + t / (1 - t)
        return mp.e ** (-sigma * s) * mp.cos(X * s) / (s - 1) / (1 - t) ** 2

    tail = mp.quad(tail_fun, [0, mp.mpf(1) / 2, 1 - mp.mpf(1) / 4, 1 - mp.mpf("1e-6"), 1 - mp.mpf("1e-12")])
    return head + tail


def tail_closed(sigma, X):
    """Candidate: int_2^inf ... = Re[e^{-b} E1(b)], b = sigma - i X."""
    b = mp.mpc(sigma, -X)
    return mp.re(mp.e ** (-b) * mp.e1(b))


def full_closed(sigma, X):
    """Candidate: full PV = -e^{-b} Ei(b)."""
    b = mp.mpc(sigma, -X)
    return mp.re(-mp.e ** (-b) * mp.ei(b))


print(f"{'X':>6} {'PV(mpmath)':>26} {'tail_closed':>24} {'full_closed':>24} {'|full-PV|':>10}")
for X in [0, 1, 3, 5.3, 9.4, 13.3, 30, 60]:
    sigma = mp.mpf("0.48")
    Xv = mp.mpf(str(X))
    pv = f_pv(sigma, Xv)
    tc = tail_closed(sigma, Xv)
    fc = full_closed(sigma, Xv)
    print(f"{X:>6} {mp.nstr(pv, 20):>26} {mp.nstr(tc, 16):>24} {mp.nstr(fc, 16):>24} "
          f"{mp.nstr(abs(fc - pv), 3):>10}")

# known closed form at X = 0:  XL = -e^{-sigma} Ei(sigma)
sigma = mp.mpf("0.48")
print("X=0 analytic -e^-s Ei(s) =", mp.nstr(-mp.e ** (-sigma) * mp.ei(sigma), 20))
print("X=0 tail closed e^-s E1(s) =", mp.nstr(mp.e ** (-sigma) * mp.e1(sigma), 20))
print("X=0 PV(-inf..) head pieces check:", mp.nstr(f_pv(sigma, mp.mpf(0)), 20))