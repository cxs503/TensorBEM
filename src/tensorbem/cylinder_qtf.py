"""Closed-form difference-frequency QTF of a fixed bottom-mounted cylinder.

Why this module exists
----------------------
:mod:`tensorbem.qtf_drift` computes the difference-frequency QTF ``T(omega_i,
omega_j)`` from a *panel* BEM solution, and its only external anchor so far is
the diagonal (which must collapse onto the near-field mean drift).  The
off-diagonal part had **no reference at all**: the module's own OC4 study
concluded that "the off-diagonal QTF magnitude computed here is a lower bound"
because the second-order-potential term is not implemented, and every
off-diagonal check in ``docs/qtf_drift_oc4.md`` is therefore internal
(self-consistency) rather than absolute.  This module supplies the missing
absolute reference for the quadratic-pressure part on a geometry where the
first-order diffraction problem has a closed solution, so that a BEM QTF can be
graded against an independent analytic value instead of against itself.

Geometry and first-order solution
---------------------------------
Deep water (finite depth is out of scope), fixed bottom-mounted vertical circular
cylinder of radius ``a`` reaching ``z = -inf``; head waves along ``+x``.  With
``Phi = Re{Phi_hat e^{-i omega t}}``, ``omega^2 = g k``,

    Phi_hat = (g A / omega) exp(k z) psi(x, y),
    psi = sum_m [ i^m J_m(kr) + A_m H_m^(1)(kr) ] e^{i m theta},
    A_m = - i^m J_m'(ka) / H_m^(1)'(ka)                (MacCamy & Fuchs, 1954),

which is exactly the field :mod:`tensorbem.cylinder_drift` already implements and
verifies against the reduced Maruo far field and the ``(ka)^3`` long-wave law.

Closed-form difference-frequency QTF
------------------------------------
The quadratic-pressure (Pinkster-type) second-order force at the difference
frequency ``mu = omega_i - omega_j``, per unit product of first-order amplitudes,
is the *same* assembly :mod:`tensorbem.qtf_drift` evaluates on panels,

    T_xx(i, j) = (1/4) rho Int_S (grad phi_i . grad phi_j*) n_x dS
                 - (1/4) rho g Int_WL zeta_i zeta_j* n_x dl ,

with ``phi`` the unit-amplitude first-order potential and ``zeta = i omega phi/g``.
On the cylinder wall ``n = (cos theta, sin theta, 0)``, ``dpsi/dr = 0`` (the
body boundary condition, satisfied to machine precision by the MF modes), and

    d/dz [exp(k z) psi] = k exp(k z) psi,   Int_{-inf}^{0} exp((k_i+k_j)z) dz = 1/(k_i+k_j),

so both integrals reduce to *azimuthal* quadratures of products of the two wall
fields:

    I_1 = Int_0^{2pi} [ (dpsi_i/dtheta)(dpsi_j/dtheta)* / a^2
                        + k_i k_j psi_i psi_j* ] cos theta d theta ,
    I_2 = Int_0^{2pi} psi_i psi_j* cos theta d theta ,
    T_xx(i, j) = (rho/4) (g^2/(omega_i omega_j)) (a/(k_i+k_j)) I_1
                 - (rho/4) g a I_2 .

The elevation used in ``I_2`` is ``zeta_i zeta_j* = psi_i psi_j*`` because
``i omega phi/g = i psi`` on ``z = 0``; the ``i x (-i)`` product is ``+1``.

The integrands are finite Fourier series (``m_max`` modes per field), so the
2*pi-periodic trapezoid rule at ``n_theta > 2 (2 m_max + 1)`` points is exact --
the closed form below is machine-precision, not "closed form plus quadrature
error".

Diagonal consistency (the regression pin)
-----------------------------------------
For ``i = j`` the formula collapses exactly onto
:func:`tensorbem.cylinder_drift.near_field_cylinder_drift` (``k_i = k_j = k``,
``g^2/omega^2 = 1/k``), which in turn is verified against the far-field Maruo
route and the classical ``F_x = (5 pi^2/16) rho g A^2 a (ka)^3`` long-wave law.
:func:`closed_form_cylinder_qtf` exposes both the matrix and the diagonal so that
this identity can be asserted, per frequency, in the tests.

Scope (declared, not hidden)
----------------------------
* Quadratic-pressure part only -- the same declared scope as
  :func:`tensorbem.qtf_drift.quadratic_qtf`, so the two are directly comparable.
  The force of the second-order potential is *not* included in either; it
  vanishes on the diagonal and is non-zero off the diagonal.
* Infinite draft (no bottom cap).  A truncated cylinder differs by the bottom-cap
  contribution and by ``O(exp(-k depth))`` wall truncation; the study quantifies
  that with a depth sweep instead of hiding it.
* Surge only (``F_x``); sway vanishes by mirror symmetry, heave/yaw are not
  second-order horizontal drift components.  Head seas only (``beta = 0``).

Author: TensorBEM node (m17277, case anchor BEM-QTF-DRIFT).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .cylinder_drift import cylinder_wall_field
from .qtf_drift import DriftQTF

DEFAULT_DENSITY = 1025.0
DEFAULT_GRAVITY = 9.81


def _positive(name: str, value: float) -> float:
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return out


@dataclass(frozen=True)
class PairIntegrals:
    """The two azimuthal integrals of one frequency pair (per unit radius)."""

    integral_surface: float          # I_1, units 1/L^2
    integral_waterline: float        # I_2, dimensionless
    qtf_surge: float                 # T_xx per unit density, per A_i A_j*: multiply by rho for N/m^2
    qtf_surge_imag: float            # Im T_xx, same normalisation


def pair_integrals(
    ka_i: float,
    ka_j: float,
    *,
    radius: float = 1.0,
    gravity: float = DEFAULT_GRAVITY,
    n_theta: int = 512,
    m_max: int = 14,
) -> PairIntegrals:
    """Closed-form azimuthal integrals and ``T_xx`` for one ``(ka_i, ka_j)`` pair.

    ``ka`` values are non-dimensional (``ka = k a``); the returned ``T_xx`` is per
    unit density and per unit ``A_i A_j*`` (multiply by ``rho`` for N/m^2), i.e.
    ``F_x(t) ~ Re{ rho A_i A_j* T_xx exp(-i mu t) }``.  Density is deliberately
    kept out of this function so that it stays a pure geometry/kinematics object;
    :func:`closed_form_cylinder_qtf` applies it.
    """
    x_i = _positive("ka_i", ka_i)
    x_j = _positive("ka_j", ka_j)
    a = _positive("radius", radius)
    n_th = int(n_theta)
    if n_th < 64:
        raise ValueError("n_theta must be >= 64 for an exact periodic quadrature")
    theta = (2.0 * math.pi / n_th) * np.arange(n_th, dtype=float)
    dtheta = 2.0 * math.pi / n_th

    f_i = cylinder_wall_field(x_i, theta, m_max=m_max)
    f_j = cylinder_wall_field(x_j, theta, m_max=m_max)
    k_i = x_i / a
    k_j = x_j / a

    psi_prod = f_i["psi"] * np.conj(f_j["psi"])
    dtheta_prod = f_i["dpsi_dtheta"] * np.conj(f_j["dpsi_dtheta"])
    integrand_surface = dtheta_prod / a**2 + k_i * k_j * psi_prod
    i1 = complex(np.sum(integrand_surface * np.cos(theta)) * dtheta)
    i2 = complex(np.sum(psi_prod * np.cos(theta)) * dtheta)

    g = _positive("gravity", gravity)
    omega_i = math.sqrt(g * k_i)
    omega_j = math.sqrt(g * k_j)
    t = (0.25 * (g * g) / (omega_i * omega_j) * (a / (k_i + k_j)) * i1
         - 0.25 * g * a * i2)
    return PairIntegrals(
        integral_surface=float(i1.real),
        integral_waterline=float(i2.real),
        qtf_surge=float(t.real),
        qtf_surge_imag=float(t.imag),
    )


def closed_form_cylinder_qtf(
    ka_values,
    *,
    radius: float = 1.0,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    n_theta: int = 512,
    m_max: int = 14,
) -> DriftQTF:
    """Full ``T_xx(omega_i, omega_j)`` matrix of the fixed bottom-mounted cylinder.

    Returns a :class:`tensorbem.qtf_drift.DriftQTF` so that the analytic reference
    and the BEM result are the *same object type* downstream (same
    ``matrix``/``diagonal`` semantics: ``matrix[i, i].real`` is the mean-drift
    coefficient ``F_bar / A^2``).
    """
    ks = np.asarray([_positive("ka", x) for x in ka_values], dtype=float)
    if ks.size == 0:
        raise ValueError("at least one ka value is required")
    a = _positive("radius", radius)
    rho = _positive("density", density)
    g = _positive("gravity", gravity)
    n = ks.size
    matrix = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(i, n):
            pair = pair_integrals(
                ks[i], ks[j], radius=a, gravity=g, n_theta=n_theta, m_max=m_max
            )
            value = complex(pair.qtf_surge, pair.qtf_surge_imag)
            matrix[i, j] = value
            matrix[j, i] = np.conj(value)
    omegas = np.sqrt(g * ks / a)
    diagonal = matrix.diagonal().real * rho
    return DriftQTF(
        omegas=omegas,
        wavenumbers=ks / a,
        matrix=matrix * rho,
        diagonal=diagonal,
        route="cylinder_closed_form",
        density=rho,
        gravity=g,
        wave_angle=0.0,
        metadata={
            "case": "bottom-mounted vertical cylinder, deep water, head seas",
            "part": "quadratic pressure (Pinkster near field), infinite draft",
            "radius": a,
            "ka": ks.tolist(),
            "reference": "MacCamy & Fuchs (1954) modes + second-order near-field assembly",
        },
    )


def qtf_surge_abs(
    ka_i: float, ka_j: float, *, radius: float = 1.0, **kwargs
) -> float:
    """``|T_xx|`` of one pair -- the scalar the BEM comparison is graded on."""
    pair = pair_integrals(ka_i, ka_j, radius=radius, **kwargs)
    return math.hypot(pair.qtf_surge, pair.qtf_surge_imag)