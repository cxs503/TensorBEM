"""Independent spectral (Legendre-projection) solve of the rigid-lid heave problem.

Rigid-lid limit (omega -> 0) of the free-surface condition: dphi/dz = 0 on z = 0,
i.e. phi is *even* about the free surface.  Reflecting the heaving hemisphere
(r = a, z <= 0) about z = 0 turns the body into the *full* sphere whose two
halves slide in opposite directions: the outward normal-velocity data over the
full sphere is

    dphi/dn = -V |cos(theta)|

(a *sliding* pair), NOT V cos(theta) (which is the classical rigid translation of
the whole sphere, dphi/dn = V cos(theta), phi odd, single dipole, A = 2/3 rho pi a^3).

Both BCs are axisymmetric and admit a closed-form Legendre projection:
phi = sum_n c_n (a/r)^{n+1} P_n(cos theta) with n of matching parity, and on r=a
the BC gives, mode by mode,

    -(n+1) c_n * 2/(2n+1) = int_{-1}^{1} (BC)(mu) P_n(mu) dmu .

Heave added mass of the *hemisphere* from the pressure integral over z <= 0:
    A33 = -rho * int_lower phi n_z dS = rho * a^2 * int_{-1}^{0} ... (sign fixed below)
"""

import math

import numpy as np


def legendre_even_project(N):
    """Gauss-Legendre quadrature-ready nodes/weights for mu in [-1, 1]."""
    x, w = np.polynomial.legendre.leggauss(400)
    return x, w


def legendre_values(n_max, mu):
    """P_n(mu) for n = 0..n_max via Bonnet recursion."""
    out = np.zeros((n_max + 1, mu.size))
    out[0] = 1.0
    if n_max >= 1:
        out[1] = mu
    for n in range(1, n_max):
        out[n + 1] = ((2 * n + 1) * mu * out[n] - n * out[n - 1]) / (n + 1)
    return out


def solve(n_terms=60, mode="sliding"):
    mu, w = legendre_even_project(n_terms)
    P = legendre_values(n_terms, mu)
    if mode == "sliding":
        bc = -np.abs(mu)          # two halves sliding in opposite directions
        parity = 0                # even n only
    elif mode == "translation":
        bc = mu                   # whole sphere translating (self-test)
        parity = 1                # odd n only
    else:
        raise ValueError(mode)
    c = np.zeros(n_terms + 1)
    for n in range(parity, n_terms + 1, 2):
        rhs = np.sum(w * bc * P[n])
        c[n] = rhs / (-(n + 1) * 2.0 / (2 * n + 1))
    return c, mu, w, P


def heave_added_mass(n_terms=60, mode="sliding", radius=1.0, density=1025.0):
    """A33 / (rho * a^3) and A33 / (rho * grad_hemisphere), unit velocity."""
    c, mu, w, P = solve(n_terms, mode)
    # phi on the hemisphere (r = a, -1 <= mu <= 0)
    lower = mu <= 0.0
    wl, Pl = w[lower], P[:, lower]
    mul = mu[lower]
    phi = np.zeros_like(mul)
    for n in range(0, n_terms + 1, 2 if mode == "sliding" else 1):
        phi += c[n] * Pl[n]
    # n_z = mu on the sphere; dS = a^2 dmu dphi_int ; A33 = -rho int phi n_z dS
    a33 = -density * radius**2 * 2 * math.pi * np.sum(wl * phi * mul)
    rho_grad = density * (2.0 / 3.0) * math.pi * radius**3
    return a33, a33 / rho_grad, c


if __name__ == "__main__":
    for mode in ("translation", "sliding"):
        for n_terms in (10, 20, 40, 60, 100, 200):
            a33, ratio, _ = heave_added_mass(n_terms, mode)
            print(f"{mode:12s} n_terms={n_terms:4d}  A33={a33:12.6f}  A33/(rho grad)={ratio:.6f}")
        print()