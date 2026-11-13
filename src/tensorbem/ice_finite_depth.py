"""Radiation in a mass-loading ice cover over water of finite depth.

The deep-water ice module (:mod:`tensorbem.ice_rao`) refuses to run below
``k_i H >= 3`` because its Kelvin kernel has no bottom.  This module closes
that guard band: the mass-loading free-surface condition

    dphi/dz = K(omega) phi,   K = omega**2 / (g - mu omega**2),

is the same Robin condition the open-water finite-depth kernel implements
with ``nu = omega**2/g``, so the ice-covered finite-depth Green function is
the finite-depth kernel of :mod:`tensorbem.finite_depth_solver3d` evaluated
at ``nu = K(omega)``.  The dispersion relation then reads
``k_i tanh(k_i H) = K``, i.e. the propagating branch already owned by
:func:`tensorbem.ice.mass_loading_wavenumber`.

The finite-depth solver infers ``omega = sqrt(g nu)`` from ``nu``; under ice
``K != omega**2/g`` so the inferred frequency is wrong.  The impedance
identity ``A + (i/omega) B = -rho int phi n dS`` is unaffected in its real
part, but the damping must be rescaled by ``omega / sqrt(g K)``.  The result
exposes ``implied_omega`` and a test pins the identity.

Verification anchors (see ``docs/ice_finite_depth_radiation.md`` and
``tests/test_ice_finite_depth.py``):

* the kernel at ``nu = K`` satisfies ``dG/dz = K G`` on ``z = 0`` and
  ``dG/dz = 0`` on ``z = -H``;
* the propagating root solves ``k_i tanh(k_i H) = K`` to the solver
  tolerance;
* ``mu -> 0`` reproduces the open-water finite-depth solver to round-off;
* in the overlap band ``k_i H >= 3`` the answer agrees with the deep-water
  ice solver (a different kernel: John modal series with a bottom vs the
  Kelvin deep-water kernel) to well under 1%.

The two limits are *independent routes*: the vanishing-ice limit reproduces
the open-water finite-depth solver, and the deep-water limit reproduces the
deep-water ice solver, whose kernels (John modal series vs Kelvin Green
function) are different code.
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
