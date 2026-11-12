import mpmath as mp
mp.mp.dps = 30


def phi_num(b):
    """PV int_{-1}^inf e^{-b u}/u du by pole-split regularisation."""
    def reg(u):
        return (mp.e ** (-b * u) - 1) / u

    inner = mp.quad(reg, [-1, 0])       # pole at 0 handled by (g-1)/u
    # int_0^inf (e^{-bu}-1)/u du + int_0^inf 1/u? no -> use: PV int_{-1}^{inf} =
    # int_{-1}^{0} (g-1)/u du + int_0^{inf} (g-1)/u du + [PV int_{-1}^{0} 1/u du = 0]
    #                                  + int_0^{inf} 1/u du  <- diverges, so instead:
    # int_{-1}^{inf} g/u du = int_{-1}^{inf} (g - 1_{u<1}) / u du  + int_0^1 du/u
    # keep it simple: use the b-derivative route instead (verify below)
    return inner


def tail_num(b):
    def t(u):
        return mp.e ** (-b * u) / u
    # int_1^inf e^{-bu}/u du with substitution  u = 1 + w/(1-w)
    def tw(w):
        u = 1 + w / (1 - w)
        return mp.e ** (-b * u) / u / (1 - w) ** 2
    return mp.quad(tw, [0, mp.mpf(1) / 2, 1 - mp.mpf(1) / 100, 1 - mp.mpf("1e-8"), 1 - mp.mpf("1e-14")])


for X in ["0", "1", "3", "5.3", "9.4", "13.3", "30", "60"]:
    b = mp.mpc(mp.mpf("0.48"), -mp.mpf(X))
    tn = tail_num(b)
    e1 = mp.e1(b)
    ei_b = mp.ei(b)
    print(f"X={X:>5}  tail_num={mp.nstr(mp.re(tn), 16)} {mp.nstr(mp.im(tn), 16)}"
          f"  e1={mp.nstr(mp.re(e1), 16)} {mp.nstr(mp.im(e1), 16)}"
          f"  |tail-e1|={mp.nstr(abs(tn - e1), 3)}"
          f"  |e1+ei(-b)|={mp.nstr(abs(e1 + mp.ei(-b)), 3)}")