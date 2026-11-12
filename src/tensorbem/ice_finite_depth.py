"""3-D radiation in a mass-loading ice cover over water of *finite* depth.

Why this module exists
----------------------
:mod:`tensorbem.ice_rao` solves the ice-covered radiation and diffraction
problem with the deep-water Kelvin kernel and therefore refuses to run whenever
``k_i * H < 3`` (:data:`tensorbem.ice_rao.DEEP_WATER_MARGIN`): the deep-water
kernel has no bottom condition to impose and its ``tanh`` assumption for the
vertical mode is outside its validity envelope.  ``docs/ice_mass_loading_verification.md``
records exactly this as an open gap -- *"the ice_rao module requires k_i H >= 3;
finite-depth ice Green function is not implemented"* -- which leaves every
shallow and medium-depth ice field (the Arctic shelf, ``H`` = 20-200 m)
unreachable for the radiation path.

The observation that closes the gap
-----------------------------------
The mass-loading ice cover enters the problem *only* through the linearised
free-surface condition (Weitz & Keller 1950)

    dphi/dz = K(omega) phi,      K(omega) = omega**2 / (g - mu omega**2),
                                 mu = rho_i h_i / rho_w,

which is the same Robin condition ``dphi/dz = nu phi`` that
:mod:`tensorbem.finite_depth3d` implements for open water, with ``nu`` replaced
by ``K``.  Nothing else in the finite-depth kernel (the John modal series, the
propagating root ``k0 tanh(k0 H) = nu``, the evanescent roots
``mu_n tan(mu_n H) = -nu``) refers to gravity waves per se, so the ice-covered
finite-depth Green function is

    G_ice(x, xi) = G_finite_depth(x, xi; nu = K(omega), H).

The dispersion relation then follows for free, ``k_i tanh(k_i H) = K``, i.e.

    omega**2 = g k tanh(k H) / (1 + mu k tanh(k H)),

which is the relation :func:`tensorbem.ice.mass_loading_wavenumber` already
solves -- the dispersion shipped in :mod:`tensorbem.ice` is *already* the
finite-depth one, so only the kernel was missing.

Bookkeeping caveat (the reason this is not a one-liner)
-------------------------------------------------------
:func:`tensorbem.finite_depth_solver3d.solve_radiation_finite_depth_3d` infers
``omega = sqrt(g * nu)`` from ``nu``, because for open water ``nu = omega**2/g``.
Under ice ``K != omega**2/g``, so the solver's implied frequency is
``omega_implied = sqrt(g K) != omega``.  The impedance relation it uses is

    A_ij + (i/omega) B_ij = -rho * int_{Gamma_B} phi_j n_i dS,

so the real part (added mass) is unaffected, while the damping it returns is
``omega_implied * Im(Z)`` instead of the physical ``omega * Im(Z)``.  The
correction applied here is therefore

    A = Re(Z),      B = B_solver * omega / omega_implied,

the same rescaling :mod:`tensorbem.ice_rao` applies to the deep-water solver.
``FiniteDepthIceRadiation.implied_omega`` exposes the factor so a reader can
check it rather than trust it.

What is *not* claimed
---------------------
* Diffraction / excitation under ice in finite depth is **not** provided here:
  the finite-depth solver ships no incident-wave or Kochin machinery.  The
  diffraction row therefore stays a deep-water row.
* There is no experimental or third-party numerical ice RAO benchmark in this
  repository, so absolute ice RAO magnitudes remain WITHHELD, exactly as for
  :mod:`tensorbem.ice_rao`.  The evidence collected here is *consistency between
  independent routes*: the vanishing-ice limit reproduces the open-water
  finite-depth solver, and the deep-water limit reproduces the deep-water ice
  solver, whose kernels (John modal series vs Kelvin Green function) are
  different code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from .finite_depth_solver3d import (
    FiniteDepth3DOperators,
    solve_radiation_finite_depth_3d,
)
from .geometry3d import SurfaceMesh3D
from .ice import IceCover, mass_loading_wavenumber, robin_coefficient

FloatArray = NDArray[np.float64]

#: Reference density of sea water (kg/m^3), matching the deep-water ice path.
DEFAULT_DENSITY: float = 1025.0


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return value


@dataclass(frozen=True)
class FiniteDepthIceRadiation:
    """Six-degree-of-freedom radiation in a mass-loading ice cover, finite depth."""

    omega: float
    wavenumber_ice: float
    wavenumber_open: float
    robin_coefficient: float
    depth_parameter: float
    implied_omega: float
    added_mass: FloatArray
    damping: FloatArray
    condition_number: float
    relative_residual: float
    ice: IceCover


def ice_finite_depth_wavenumber(omega: float, ice: IceCover) -> float:
    """Propagating root ``k_i`` of ``k tanh(k H) = K(omega)``.

    Thin wrapper over :func:`tensorbem.ice.mass_loading_wavenumber` (the module
    that owns the ice dispersion) that refuses the evanescent branch: a radiation
    solve needs a real pole in the spectrum.
    """
    omega = _positive("omega", omega)
    wave = mass_loading_wavenumber(omega, ice)
    if not wave.propagating:
        raise ValueError(
            "omega exceeds the mass-loading cut-off sqrt(g/mu) = "
            f"{ice.cutoff_omega:.6g} rad/s; the ice waves are evanescent and the "
            "finite-depth kernel has no real pole"
        )
    return float(wave.wavenumber)


def solve_radiation_finite_depth_ice(
    mesh: SurfaceMesh3D,
    omega: float,
    ice: IceCover,
    *,
    center: ArrayLike | None = None,
    density: float = DEFAULT_DENSITY,
    rankine_order: int = 5,
    kernel: str = "eigenseries",
    kernel_nodes: int = 16,
    mode_count: int | None = None,
    operators: FiniteDepth3DOperators | None = None,
) -> FiniteDepthIceRadiation:
    """Radiation added mass and damping under ice over water of depth ``H``.

    ``operators`` (from :func:`tensorbem.finite_depth_solver3d.assemble_finite_depth_operators_3d`
    at ``nu = K(omega)``, ``depth = ice.water_depth``) may be supplied to reuse
    the assembled matrices across frequencies that share a coefficient; the
    reuse check of the underlying solver still applies.
    """
    omega = _positive("omega", omega)
    rho = _positive("density", density)
    g = _positive("gravity", ice.gravity)

    k_ice = ice_finite_depth_wavenumber(omega, ice)
    K = robin_coefficient(omega, ice)
    # the kernel is the open-water finite-depth kernel evaluated at nu = K
    implied_omega = math.sqrt(g * K)

    result = solve_radiation_finite_depth_3d(
        mesh,
        K,
        ice.water_depth,
        center=center,
        density=rho,
        gravity=g,
        rankine_order=rankine_order,
        kernel=kernel,
        kernel_nodes=kernel_nodes,
        mode_count=mode_count,
        operators=operators,
    )
    # restore the physical frequency: A = Re(Z) is unaffected, B is not
    damping = np.asarray(result.damping, dtype=float) * (omega / implied_omega)

    target = omega * omega
    depth = ice.water_depth

    def residual(k: float) -> float:
        return g * k * math.tanh(k * depth) / (1.0 + ice.mu * k * math.tanh(k * depth)) - target

    return FiniteDepthIceRadiation(
        omega=omega,
        wavenumber_ice=k_ice,
        wavenumber_open=float(_open_water_wavenumber(omega, ice)),
        robin_coefficient=float(K),
        depth_parameter=float(k_ice * ice.water_depth),
        implied_omega=float(implied_omega),
        added_mass=np.asarray(result.added_mass, dtype=float),
        damping=damping,
        condition_number=float(result.condition_number),
        relative_residual=max(float(result.relative_residual), abs(residual(k_ice))),
        ice=ice,
    )


def _open_water_wavenumber(omega: float, ice: IceCover) -> float:
    """Open-water finite-depth wavenumber at the same frequency (reference)."""
    from .ice import open_water_wavenumber

    return open_water_wavenumber(omega, ice.water_depth, ice.gravity)


def radiation_report(result: FiniteDepthIceRadiation) -> dict[str, float]:
    """Scalar summary of a result: heave/sway added mass and damping, ``k_i H``."""
    A = np.asarray(result.added_mass, dtype=float)
    B = np.asarray(result.damping, dtype=float)
    return {
        "omega": float(result.omega),
        "wavenumber_ice": float(result.wavenumber_ice),
        "wavenumber_open": float(result.wavenumber_open),
        "robin_coefficient": float(result.robin_coefficient),
        "depth_parameter": float(result.depth_parameter),
        "implied_omega": float(result.implied_omega),
        "added_mass_heave": float(A[2, 2]),
        "added_mass_surge": float(A[0, 0]),
        "damping_heave": float(B[2, 2]),
        "damping_surge": float(B[0, 0]),
        "condition_number": float(result.condition_number),
        "relative_residual": float(result.relative_residual),
    }


__all__ = [
    "DEFAULT_DENSITY",
    "FiniteDepthIceRadiation",
    "ice_finite_depth_wavenumber",
    "radiation_report",
    "solve_radiation_finite_depth_ice",
]