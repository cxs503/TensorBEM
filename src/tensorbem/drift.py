"""Second-order mean wave-drift force on a two-dimensional deep-water body.

The rest of the package solves the *first-order* radiation/diffraction problem
and reports added mass, damping and excitation forces.  The mean (time-averaged)
second-order wave force -- the drift force, and for a ship advancing in waves the
added resistance -- is a genuinely different quantity that is routinely needed by
seakeeping tools and was listed as uncovered in
``docs/marine_bem_benchmarks.md`` (benchmark class 12).  This module supplies it
for the two-dimensional deep-water case, with two independent computational
routes so that the result can be checked against itself:

**Far field (Maruo / momentum theorem).**  For deep water in two dimensions the
mean horizontal force per unit crest length follows from a momentum balance on a
control volume that cuts the free surface far upstream and far downstream.  With
incident elevation amplitude ``A``, reflected amplitude ``A_R = R A`` and
transmitted amplitude ``A_T`` (energy conservation ``|A_T|^2 = |A|^2 - |A_R|^2``)::

    F_x = (rho*g/4) * (|A|^2 + |A_R|^2 - |A_T|^2) = (rho*g/2) * |A_R|^2

The reflected-wave amplitude is exactly the ``reflection_coefficient`` that
:func:`tensorbem.free_surface_solver.solve_diffraction_indirect` already returns
from its Kochin integrals, so the far-field route costs nothing beyond a solve
that the package performs anyway.

**Near field (pressure integration).**  To second order the mean pressure on a
fixed body is ``<p> = -(rho/2) <|grad phi|^2>`` (the ``-rho*dphi/dt`` term
averages to zero and the body is fully submerged, so there is no waterline term
and no second-order potential is required).  The force on the body is
``F_j = -oint <p> n_j ds``, so, with ``phi`` the total first-order potential and
``n`` the outward normal of the body::

    F_j = (rho/2) * oint <|grad phi|^2> n_j ds

The body boundary condition ``dphi/dn = 0`` leaves only the tangential
derivative: ``<|grad phi|^2> = (1/2) |dphi/ds|^2`` for the complex amplitude
``phi_hat`` with time factor ``exp(-i*omega*t)``.  Consequently the near-field
route needs the *surface tangential velocity*, which the first-order panel
solution does not return directly -- that is the one piece of new machinery in
this module, :func:`surface_tangential_velocity`.

Why the recovery step is not a trivial finite difference: the trace of the
indirect single-layer solution is pointwise ``O(h)`` accurate but carries a
panel-scale sawtooth error component.  Differencing it (or differentiating it
spectrally without filtering) amplifies that component to ``O(1)`` and destroys
the second-order force -- measured, see ``docs/drift_force_2d.md``.  The recovery
therefore fits the trace with a band-limited Fourier series in arclength
(``|m| <= mode_fraction * N``) and differentiates the series analytically.  The
truncation is a documented modelling choice, not a fit: the physical trace varies
on the wavelength scale, which corresponds to a handful of modes.

Verification (``tests/test_drift_force.py``, ``docs/drift_force_2d.md``):

* the recovery layer is checked against closed-form derivatives of the incident
  Airy potential on a circle;
* a submerged circular cylinder must produce ``F = 0`` exactly, because Dean's
  (1948) / Ursell's (1950) zero-reflection theorem applies at every frequency in
  deep water -- both routes return machine-precision zero;
* for a strongly reflecting submerged ellipse, where ``|R| ~ 0.25`` and the
  force is ``O(10^2)`` N/m, the near- and far-field routes agree to about 1% and
  converge to the same value under panel refinement.

Limitations (stated here so they cannot be mistaken for capabilities): deep water
only, two dimensions (per unit crest length), fixed body (no forward speed, no
radiation contribution), and second-order *mean* force only -- the difference-
frequency QTF, the waterline integral for surface-piercing bodies, and three
dimensions are out of scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .free_surface_solver import DiffractionResult
from .geometry import PanelMesh

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_GRAVITY = 9.80665
_DENSITY = 1025.0

#: Default band limit of the tangential-velocity recovery, as a fraction of the
#: panel count.  The physical trace is resolved by a handful of modes, so a
#: quarter-band fit removes the panel-scale sawtooth while leaving the solution
#: untouched; the study in ``docs/drift_force_2d.md`` shows the drift force
#: varies by less than 1% for ``mode_fraction`` between 0.10 and 0.40.
DEFAULT_MODE_FRACTION = 0.25


def _validate_scalar(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_positive(name: str, value: float) -> float:
    number = _validate_scalar(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def panel_arclength(mesh: PanelMesh) -> tuple[FloatArray, float]:
    """Return the arclength coordinate of every panel centre and the perimeter.

    Panel centres of a closed counter-clockwise polygon are ordered along the
    boundary, so their arclength coordinate is strictly increasing and spans
    one period of the perimeter.
    """
    if not isinstance(mesh, PanelMesh):
        raise TypeError("mesh must be a PanelMesh")
    vertices = mesh.vertices
    edges = np.roll(vertices, -1, axis=0) - vertices
    lengths = np.linalg.norm(edges, axis=1)
    s = np.cumsum(lengths) - 0.5 * lengths
    return s, float(np.sum(lengths))


def _validate_mode_fraction(panel_count: int, mode_fraction: float) -> int:
    fraction = _validate_scalar("mode_fraction", mode_fraction)
    if not 0.0 < fraction <= 0.5:
        raise ValueError("mode_fraction must lie in (0, 0.5]")
    mode_count = int(math.floor(fraction * panel_count))
    if mode_count < 1:
        raise ValueError(
            "mode_fraction * panel_count must be at least 1 to resolve the trace"
        )
    # A band-limited fit needs more data than unknowns.
    if 2 * mode_count + 1 > panel_count:
        raise ValueError(
            "mode_fraction is too large for this panel count; "
            "use mode_fraction <= 0.5 * (1 - 1/panel_count)"
        )
    return mode_count


def surface_tangential_velocity(
    mesh: PanelMesh,
    trace: ComplexArray,
    *,
    mode_fraction: float = DEFAULT_MODE_FRACTION,
) -> ComplexArray:
    """Recover ``dphi/ds`` along the body from the panel-centre trace of ``phi``.

    The complex amplitude ``trace`` (one value per panel centre) is fitted by the
    band-limited Fourier series ``sum_{|m|<=M} c_m exp(2*pi*i*m*s/L)`` in
    arclength ``s`` with ``L`` the perimeter and ``M`` the band limit, and the
    series is differentiated analytically.  Restricting the band removes the
    panel-scale sawtooth of the constant-panel trace, which a raw difference or
    an unfiltered spectral derivative would amplify into a spurious velocity.

    Parameters
    ----------
    mesh:
        Closed counter-clockwise polygon; ``len(trace)`` must equal its panel
        count.
    trace:
        Complex potential amplitude ``phi_hat`` at the panel centres, with time
        factor ``exp(-i*omega*t)``.
    mode_fraction:
        Band limit as a fraction of the panel count, in ``(0, 0.5]``.
    """
    s, perimeter = panel_arclength(mesh)
    values = np.asarray(trace, dtype=np.complex128)
    if values.ndim != 1 or values.shape[0] != mesh.panel_count:
        raise ValueError("trace must hold one value per panel centre")
    if not np.all(np.isfinite(values)):
        raise ValueError("trace must be finite")
    mode_count = _validate_mode_fraction(mesh.panel_count, mode_fraction)

    modes = np.arange(-mode_count, mode_count + 1)
    wavenumbers = 2.0 * math.pi * modes / perimeter
    basis = np.exp(1.0j * np.outer(s, wavenumbers))
    coefficients, *_ = np.linalg.lstsq(basis, values, rcond=None)
    derivative_basis = basis * (1.0j * wavenumbers)
    return derivative_basis @ coefficients


def mean_drift_force_far_field(
    reflection_coefficient: complex,
    *,
    amplitude: float = 1.0,
    density: float = _DENSITY,
    gravity: float = _GRAVITY,
) -> float:
    """Mean horizontal drift force per unit crest length from the reflection coefficient.

    ``F_x = (rho*g/2)*|R*A|^2`` for a deep-water two-dimensional wave train of
    incident elevation amplitude ``amplitude``.  This is the momentum-theorem
    (Maruo) form: the transmitted amplitude does not appear because energy
    conservation already fixes ``|A_T|^2 = |A|^2 - |A_R|^2``.
    """
    rho = _validate_positive("density", density)
    g = _validate_positive("gravity", gravity)
    amp = _validate_positive("amplitude", amplitude)
    reflection = complex(reflection_coefficient)
    if not math.isfinite(reflection.real) or not math.isfinite(reflection.imag):
        raise ValueError("reflection_coefficient must be finite")
    return 0.5 * rho * g * abs(reflection * amp) ** 2


def mean_drift_force_near_field(
    mesh: PanelMesh,
    potential: ComplexArray,
    *,
    density: float = _DENSITY,
    mode_fraction: float = DEFAULT_MODE_FRACTION,
) -> FloatArray:
    """Mean second-order force from direct pressure integration on the body.

    Returns ``(F_x, F_z)`` per unit crest length for a fixed fully submerged
    body whose surface potential amplitude is ``potential`` (one value per panel
    centre) and which satisfies ``dphi/dn = 0`` on the body.  The integrand is
    ``(rho/2)*<|grad phi|^2> = (rho/4)*|dphi/ds|^2``.
    """
    rho = _validate_positive("density", density)
    velocity = surface_tangential_velocity(mesh, potential, mode_fraction=mode_fraction)
    kinetic = 0.5 * np.abs(velocity) ** 2
    integrand = kinetic[:, None] * mesh.normals * mesh.lengths[:, None]
    return 0.5 * rho * np.sum(integrand, axis=0)


@dataclass(frozen=True)
class MeanDriftForce:
    """Mean second-order wave force from the two independent routes.

    ``far_field`` and ``near_field[0]`` are the same physical quantity computed
    by the momentum theorem and by pressure integration respectively;
    ``discrepancy_normalised`` is their difference divided by ``rho*g*A^2``,
    which stays meaningful when the force itself vanishes (Dean's zero-reflection
    cylinder), where a force-relative error would be 0/0.
    """

    far_field: float
    near_field: FloatArray
    amplitude: float
    density: float
    gravity: float
    mode_fraction: float
    reflection_coefficient: complex
    energy_scale: float

    @property
    def discrepancy(self) -> float:
        return float(self.near_field[0] - self.far_field)

    @property
    def discrepancy_normalised(self) -> float:
        return self.discrepancy / self.energy_scale

    @property
    def relative_discrepancy(self) -> float:
        if self.far_field == 0.0:
            return math.inf if self.discrepancy != 0.0 else 0.0
        return self.discrepancy / self.far_field

    @property
    def force_ratio(self) -> float:
        """``F_x / (rho*g*A^2)``: the dimensionless drift-force amplitude."""
        return float(self.near_field[0] / self.energy_scale)


def mean_drift_force(
    mesh: PanelMesh,
    result: DiffractionResult,
    *,
    amplitude: float = 1.0,
    density: float = _DENSITY,
    gravity: float = _GRAVITY,
    mode_fraction: float = DEFAULT_MODE_FRACTION,
) -> MeanDriftForce:
    """Both drift-force routes for a solved two-dimensional diffraction problem."""
    if not isinstance(result, DiffractionResult):
        raise TypeError("result must be a DiffractionResult")
    rho = _validate_positive("density", density)
    g = _validate_positive("gravity", gravity)
    amp = _validate_positive("amplitude", amplitude)
    far_field = mean_drift_force_far_field(
        result.reflection_coefficient, amplitude=amp, density=rho, gravity=g
    )
    near_field = mean_drift_force_near_field(
        mesh,
        result.total_potential,
        density=rho,
        mode_fraction=mode_fraction,
    )
    return MeanDriftForce(
        far_field=far_field,
        near_field=near_field,
        amplitude=amp,
        density=rho,
        gravity=g,
        mode_fraction=float(mode_fraction),
        reflection_coefficient=complex(result.reflection_coefficient),
        energy_scale=rho * g * amp**2,
    )