"""scratch 2: fix the scale factor of the exterior SC map for a regular n-gon."""
import math
import numpy as np

def half_moment(alpha):
    return 0.5 * math.sqrt(math.pi) * math.gamma(0.5 * (1 + alpha)) / math.gamma(1 + 0.5 * alpha)

def half_moment_quad(alpha):
    x, w = np.polynomial.legendre.leggauss(40)
    psi = 0.25 * math.pi * (x + 1)
    return 0.25 * math.pi * np.sum(w * np.sin(psi) ** alpha)

def scale(n, a=1.0):
    alpha = 2.0 / n
    return a * math.sin(math.pi / n) / (alpha * 2.0**alpha * half_moment(alpha))

for n in (8, 16, 32, 64, 128):
    c = scale(n)
    alpha = 2.0 / n
    I = half_moment(alpha)
    print(n, "c=%.10f" % c, "I_gamma=%.15f" % I, "I_quad=%.15f" % half_moment_quad(alpha))
    L = 2 * alpha * 2.0**alpha * I * c
    print("   edge=", L, "exact=", 2 * math.sin(math.pi / n),
          "circum=", L / (2 * math.sin(math.pi / n)),
          "apothem=", 0.5 * L / math.tan(math.pi / n), "vs", math.cos(math.pi / n))

# exact velocity at arc midpoints on the polygon boundary
n = 32
c = scale(n)
alpha = 2.0 / n
w = np.exp(1j * (2 * np.pi * np.arange(n) + np.pi) / n)
v = (1 - w**-2) / (1 - w**-n) ** alpha
theta = np.angle(w)
print("speed mid vs 2|sin| * 2^-alpha: max diff",
      np.max(np.abs(np.abs(v) - 2 * np.abs(np.sin(theta)) / 2**alpha)))
print("normal component (should be 0):", np.max(np.abs((v * np.exp(-1j * theta)).real)))