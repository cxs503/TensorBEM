"""Linear wavemaker transfer functions in a constant-depth flume.

A wave flume generates regular waves by moving one end wall.  Linear
(Biesel-Havelock) wavemaker theory gives the far-field wave height ``H`` for a
given stroke ``S`` in closed form: the propagating mode of the free-surface
Green function is projected onto the prescribed wall velocity, and the
evanescent modes are simply dropped because they decay within a fraction of a
wavelength.  That projection is an exact solution of the Neumann problem

    d(phi)/dn = u(z)  on the wavemaker,   free surface and bed conditions on the
    rest of the boundary, radiation condition downstream,

so it is a genuine analytic reference for a panel code, not a fitted curve::

    a      = (4 k sinh(kh) / (i omega (2kh + sinh 2kh))) * integral u(z) cosh k(z+h) dz
    H / S  = 2 |a| / S

with ``omega**2 = g k tanh(kh)``.  For the two classical wall motions

    piston  u(z) = U0                         -> H/S = 4 sinh^2(kh) / (2kh + sinh 2kh)
    flap    u(z) = U0 (z + h)/h  (hinge: bed) -> H/S = 4 sinh(kh) [sinh(kh) + (1-cosh(kh))/(kh)]
                                                    / (2kh + sinh 2kh)

Both limits are the textbook ones: piston ``H/S -> 2`` in deep water and
``H/S -> kh`` in the long-wave limit, the flap's long-wave value being half of
that (its depth-mean velocity is ``U0/2``).  The long-wave limits are also what
mass conservation demands, which is why they are asserted in the tests rather
than quoted only in prose.

Why this module assembles its own matrix
----------------------------------------
:mod:`tensorbem.finite_depth_solver` deliberately refuses bodies that touch
``z = 0`` or ``z = -depth`` ("panels touching z = 0 are outside the supported
formulation"), and a wavemaker touches both by definition.  The two pieces of
that assembly are reused here unchanged -- the exact Rankine row integral
:func:`tensorbem.free_surface_solver._rankine_normal_derivative_matrix` and the
Gauss quadrature of ``d(G_smooth)/dn`` from
:func:`tensorbem.finite_depth.green_gradient` (``include_rankine=False``) -- so
the operator is the same one; only the geometry container is new (an *open*
panel curve cannot be a :class:`tensorbem.geometry.PanelMesh`, which requires a
closed counter-clockwise polygon).  ``tests/test_wavemaker_transfer.py`` proves
the operator is identical by applying both assemblies to a submerged closed
body and requiring them to agree to round-off.

The Gauss nodes always sit strictly inside each panel, so no kernel evaluation
ever lands on the waterline or on the bed: the image term of the finite-depth
kernel is ``log``-singular when field and source are both at ``z = 0``, and the
closed-form Rankine part only covers the direct term.  That invariant is
asserted in :func:`_gauss_nodes`, not left implicit.

Conventions
-----------
Time factor ``exp(-i omega t)``, ``nu = omega**2 / g`` (the codebase's
frequency parameter), wall normal ``+x`` (fluid on ``x > 0``), unit stroke by
default.  The far-field elevation comes back as a complex amplitude ``eta_a``
with ``eta(x) -> eta_a exp(i k x)``; only its modulus enters ``H/S``.

References
----------
- Havelock, T.H. (1929).  *Forced surface waves on water.*  Phil. Mag. 8, 569.
- Biesel, F. & Suquet, F. (1951).  *Les appareils generateurs de houle.*
  La Houille Blanche 6, 147.
- Dean, R.G. & Dalrymple, R.A. (1991).  *Water Wave Mechanics for Engineers and
  Scientists*, World Scientific, ch. 6 (wavemaker theory).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from .finite_depth import (
    evanescent_wavenumbers,
    finite_depth_wavenumber,
    mode_norms,
    green_function,
    green_gradient,
    propagating_mode_norm,
)
from .free_surface_solver import _rankine_normal_derivative_matrix

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

GRAVITY = 9.80665
DEFAULT_DENSITY = 1025.0


# --------------------------------------------------------------------------
# Analytic (Biesel-Havelock) transfer functions
# --------------------------------------------------------------------------
def transfer_function(kind: str, kh: float) -> float:
    """Exact linear ``H/S`` for a piston or a bed-hinged flap wavemaker.

    ``kh`` is the dimensionless depth; ``kind`` is ``'piston'`` (uniform
    horizontal velocity) or ``'flap'`` (velocity growing linearly from the bed
    hinge to the waterline, so the stroke is the surface value).
    """
    kind = _validate_kind(kind)
    kh = _validate_kh(kh)
    sinh_kh = math.sinh(kh)
    denominator = 2.0 * kh + math.sinh(2.0 * kh)
    if kind == "piston":
        return 4.0 * sinh_kh * sinh_kh / denominator
    # flap: int_0^h (zeta/h) cosh(k zeta) dzeta with zeta = z + h
    return 4.0 * sinh_kh * (sinh_kh + (1.0 - math.cosh(kh)) / kh) / denominator


def _validate_kind(kind: str) -> str:
    if kind not in ("piston", "flap"):
        raise ValueError("kind must be 'piston' or 'flap'")
    return kind


def _validate_kh(kh: float) -> float:
    if isinstance(kh, (bool, np.bool_)):
        raise ValueError("kh must be a positive finite number")
    value = float(kh)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("kh must be a positive finite number")
    return value


def velocity_profile(kind: str, z, depth: float) -> FloatArray:
    """Shape function of the wall velocity, unity stroke velocity at the surface."""
    kind = _validate_kind(kind)
    depth = _validate_depth(depth)
    z = np.asarray(z, dtype=float)
    if np.any(z > 0.0) or np.any(z < -depth):
        raise ValueError("the wavemaker spans -depth <= z <= 0")
    if kind == "piston":
        return np.ones_like(z)
    return (z + depth) / depth


# --------------------------------------------------------------------------
# Geometry: an open panel curve from the bed to the waterline
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class WavemakerSurface:
    """Panelized wavemaker face: an *open* curve, bed first, waterline last.

    Only the attributes the shared kernel assembly needs are provided, so the
    object can be handed to the matrix routines directly.  Vertices run from
    ``z = -depth`` to ``z = 0`` at ``x = 0``, which makes every panel normal
    ``(+1, 0)`` -- pointing into the fluid on ``x > 0``.
    """

    segments: FloatArray

    def __post_init__(self) -> None:
        segments = np.asarray(self.segments, dtype=float)
        object.__setattr__(self, "segments", segments)
        if segments.ndim != 3 or segments.shape[1:] != (2, 2):
            raise ValueError("segments must have shape (N, 2, 2)")
        if segments.shape[0] < 2:
            raise ValueError("a wavemaker needs at least two panels")
        if not np.isfinite(segments).all():
            raise ValueError("panel coordinates must be finite")
        lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
        if np.any(lengths <= 0.0):
            raise ValueError("zero-length panels are not allowed")

    @property
    def panel_count(self) -> int:
        return int(len(self.segments))

    @property
    def vertices(self) -> FloatArray:
        return self.segments[:, 0]

    @property
    def ends(self) -> FloatArray:
        return self.segments[:, 1]

    @property
    def vectors(self) -> FloatArray:
        return self.ends - self.vertices

    @property
    def lengths(self) -> FloatArray:
        return np.linalg.norm(self.vectors, axis=1)

    @property
    def centers(self) -> FloatArray:
        return 0.5 * (self.vertices + self.ends)

    @property
    def normals(self) -> FloatArray:
        tangent = self.vectors / self.lengths[:, None]
        return np.column_stack((tangent[:, 1], -tangent[:, 0]))

    @property
    def perimeter(self) -> float:
        return float(np.sum(self.lengths))


def vertical_wavemaker(depth: float, panels: int, *, x: float = 0.0) -> WavemakerSurface:
    """Uniform panelization of the wall at ``x`` from the bed up to the waterline."""
    depth = _validate_depth(depth)
    if isinstance(panels, (bool, np.bool_)) or not isinstance(panels, (int, np.integer)):
        raise ValueError("panels must be an integer >= 2")
    if int(panels) < 2:
        raise ValueError("panels must be an integer >= 2")
    edges = -depth + depth * np.arange(int(panels) + 1) / int(panels)
    segments = np.zeros((int(panels), 2, 2), dtype=float)
    segments[:, 0, 0] = x
    segments[:, 1, 0] = x
    segments[:, 0, 1] = edges[:-1]
    segments[:, 1, 1] = edges[1:]
    return WavemakerSurface(segments)


def _validate_depth(depth: float) -> float:
    if isinstance(depth, (bool, np.bool_)):
        raise ValueError("depth must be a positive finite number")
    value = float(depth)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("depth must be a positive finite number")
    return value


# --------------------------------------------------------------------------
# Operator assembly (same pieces as finite_depth_solver, open geometry)
# --------------------------------------------------------------------------
def _gauss_nodes(surface: WavemakerSurface, panel_nodes: int, depth: float):
    """Panel quadrature nodes, asserted to stay strictly off both boundaries.

    ``leggauss`` nodes are interior to ``(-1, 1)``, so the endpoints -- where the
    finite-depth image term would be ``log``-singular -- are never touched.  The
    assertion is cheap and turns a silent NaN into a clear failure.
    """
    if panel_nodes < 2:
        raise ValueError("panel_nodes must be at least 2")
    nodes, weights = leggauss(panel_nodes)
    fraction = 0.5 * (nodes + 1.0)
    source_x = (
        surface.vertices[:, 0][:, None]
        + fraction[None, :] * surface.vectors[:, 0][:, None]
    ).ravel()
    source_z = (
        surface.vertices[:, 1][:, None]
        + fraction[None, :] * surface.vectors[:, 1][:, None]
    ).ravel()
    jacobian = np.tile(weights, surface.panel_count) * np.repeat(
        0.5 * surface.lengths, panel_nodes
    )
    if np.any(source_z >= 0.0) or np.any(source_z <= -depth):
        raise ValueError("quadrature nodes must stay strictly inside the fluid strip")
    return source_x, source_z, jacobian


def _smooth_normal_derivative(
    surface: WavemakerSurface,
    nu: float,
    depth: float,
    panel_nodes: int,
    modes: int | None,
) -> ComplexArray:
    source_x, source_z, jacobian = _gauss_nodes(surface, panel_nodes, depth)
    count = surface.panel_count
    matrix = np.zeros((count, count), dtype=complex)
    # Rows are batched: the kernel broadcasts over (batch, count * panel_nodes), and
    # the mode series is what makes a batch costly, so the block size keeps the
    # working set bounded instead of vectorizing all rows at once.
    block = max(1, 2048 // max(count * panel_nodes, 1))
    for start in range(0, count, block):
        stop = min(start + block, count)
        centers = surface.centers[start:stop]
        normals = surface.normals[start:stop]
        gradient_x, gradient_z = green_gradient(
            centers[:, 0][:, None],
            centers[:, 1][:, None],
            source_x[None, :],
            source_z[None, :],
            nu,
            depth,
            modes=modes,
            include_rankine=False,
        )
        values = (
            gradient_x * normals[:, 0][:, None] + gradient_z * normals[:, 1][:, None]
        ) * jacobian[None, :]
        matrix[start:stop] = values.reshape(stop - start, count, panel_nodes).sum(axis=2)
    return matrix


def assemble_normal_derivative(
    surface: WavemakerSurface,
    nu: float,
    depth: float,
    *,
    panel_nodes: int = 8,
    modes: int | None = None,
) -> ComplexArray:
    """``d(single-layer potential)/dn`` matrix, jump not included."""
    matrix = _rankine_normal_derivative_matrix(surface)
    matrix += _smooth_normal_derivative(surface, float(nu), float(depth), panel_nodes, modes)
    return matrix


def solve_sources(
    surface: WavemakerSurface,
    normal_velocity: ComplexArray,
    nu: float,
    depth: float,
    *,
    panel_nodes: int = 8,
    modes: int | None = None,
) -> tuple[ComplexArray, ComplexArray]:
    """Single-layer density for ``d(phi)/dn = normal_velocity`` on the wall.

    The jump of the normal derivative across a single layer is one half, so the
    system is ``A + I/2`` exactly as in the closed-body solvers.  Returns
    ``(sources, system)``.
    """
    data = np.asarray(normal_velocity, dtype=complex)
    if data.shape != (surface.panel_count,):
        raise ValueError("normal_velocity must have one value per panel")
    matrix = assemble_normal_derivative(
        surface, nu, depth, panel_nodes=panel_nodes, modes=modes
    )
    system = matrix + 0.5 * np.eye(surface.panel_count)
    return np.linalg.solve(system, data), system


def free_surface_potential(
    surface: WavemakerSurface,
    sources: ComplexArray,
    nu: float,
    depth: float,
    x,
    *,
    panel_nodes: int = 8,
    modes: int | None = None,
) -> ComplexArray:
    """Potential at ``(x, 0)`` from the wall source distribution.

    Straight Gauss quadrature of the *full* finite-depth kernel: the field points
    are downstream of the wall, so the ``log r`` part is smooth there and no
    closed form is needed.  Far downstream this must approach
    ``eta_a exp(i k x) * g / (i omega)``, i.e. the Kochin amplitude of
    :func:`tensorbem.finite_depth_solver.wave_amplitude` -- the second route.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    source_x, source_z, jacobian = _gauss_nodes(surface, panel_nodes, depth)
    values = np.zeros(x.shape[0], dtype=complex)
    for index, xi in enumerate(x):
        kernel = green_function(
            xi, 0.0, source_x, source_z, nu, depth, modes=modes
        )
        values[index] = np.sum(
            sources.repeat(panel_nodes) * kernel * jacobian
        )
    return values


def kochin_amplitude(
    surface: WavemakerSurface,
    sources: ComplexArray,
    nu: float,
    depth: float,
    *,
    gravity: float = GRAVITY,
    modes: int | None = None,
) -> complex:
    """Far-field elevation amplitude of a source distribution (propagating mode).

    Same normalisation as
    :func:`tensorbem.finite_depth_solver.wave_amplitude`, which cannot be called
    here because it enforces ``isinstance(mesh, PanelMesh)`` and a wavemaker is
    an open curve.  The two are required to agree to round-off on a submerged
    closed body in ``tests/test_wavemaker_transfer.py``, so this copy cannot
    drift from the graded routine unnoticed.

    The kernel's far field is ``G -> -i/(2 k0 N0) phi0(z) phi0(z0) exp(i k0|x|)``
    with ``phi0(z) = cosh(k0 (z + h))``, so a density ``sigma`` radiates

        eta_a = (i omega/g) phi0(0) * [-i/(2 k0 N0)] *
                integral exp(-i k0 x) phi0(z) sigma ds .
    """
    k0 = finite_depth_wavenumber(nu, depth)
    norm = propagating_mode_norm(k0, depth)
    phi0 = np.cosh(k0 * (surface.centers[:, 1] + depth))
    integral = np.sum(
        phi0 * np.exp(-1j * k0 * surface.centers[:, 0]) * sources * surface.lengths
    )
    coefficient = -1j / (2.0 * k0 * norm) * integral
    omega = math.sqrt(float(nu) * gravity)
    return 1j * omega / gravity * math.cosh(k0 * depth) * coefficient


# --------------------------------------------------------------------------
# Exact eigenfunction field (evanescent modes included)
# --------------------------------------------------------------------------
def _mode_integrals(kind: str, k: float, mu: FloatArray, depth: float):
    """``int u(z) * {cosh k(z+h), cos(mu (z+h))} dz`` for unit surface velocity."""
    kind = _validate_kind(kind)
    if kind == "piston":
        main = math.sinh(k * depth) / k
        secondary = np.sin(mu * depth) / mu
        return main, secondary
    # flap: u = zeta/h on zeta in [0, h]
    main = (depth * math.sinh(k * depth) / k - (math.cosh(k * depth) - 1.0) / (k * k)) / depth
    secondary = (
        depth * np.sin(mu * depth) / mu + (np.cos(mu * depth) - 1.0) / (mu * mu)
    ) / depth
    return main, secondary


def exact_elevation(
    kind: str,
    x,
    *,
    kh: float = 1.0,
    depth: float = 1.0,
    stroke: float = 1.0,
    modes: int = 8,
    gravity: float = GRAVITY,
) -> ComplexArray:
    """Exact linear wavemaker field on the free surface, evanescent modes included.

    The wall problem separates in the vertical eigenfunctions of the strip:

        phi = A cosh(k(z+h)) exp(i k x) + sum_n B_n cos(mu_n (z+h)) exp(-mu_n x)

    with ``mu_n tan(mu_n h) = -nu``, ``A = I_0/(i k N_0)``, ``B_n = -I_n/(mu_n N_n)``
    (orthogonality of the two families), ``N_0``, ``N_n`` the mode norms and
    ``I`` the wall-velocity projections.  At ``z = 0`` the elevation is
    ``eta = i omega phi/g``.  This is the same closed form the transfer functions
    are read off, but here all the modes are kept, so it is a reference for the
    near-wall field -- where the evanescent modes carry most of the signal -- and
    not only for the far-field height.
    """
    kind = _validate_kind(kind)
    kh = _validate_kh(kh)
    depth = _validate_depth(depth)
    if int(modes) < 0:
        raise ValueError("modes must be non-negative")
    k = kh / depth
    omega = math.sqrt(float(gravity) * k * math.tanh(kh))
    nu = omega * omega / gravity
    x = np.atleast_1d(np.asarray(x, dtype=float))
    velocity = -0.5j * omega * stroke  # phasor of the surface wall velocity

    mu = evanescent_wavenumbers(nu, depth, int(modes))
    main_integral, secondary_integrals = _mode_integrals(kind, k, mu, depth)
    norm0 = propagating_mode_norm(k, depth)
    amplitude0 = velocity * main_integral / (1j * k * norm0)
    phi = amplitude0 * math.cosh(k * depth) * np.exp(1j * k * x)
    if len(mu):
        norms = mode_norms(nu, depth, mu)
        coefficients = -velocity * secondary_integrals / (mu * norms)
        shape = np.cos(mu * depth)
        phi = phi + np.exp(-np.outer(x, mu)) @ (coefficients * shape)
    return 1j * omega / gravity * phi


# --------------------------------------------------------------------------
# End-to-end solve
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class WavemakerResult:
    """Transfer-function verification for one ``(kind, kh, mesh)`` combination."""

    kind: str
    kh: float
    depth: float
    panels: int
    stroke: float
    wavenumber: float
    omega: float
    #: complex far-field elevation from the propagating-mode (Kochin) route
    kochan_amplitude: complex
    #: complex elevation from the free-surface probe far downstream
    probe_amplitude: complex
    probe_x: float
    transfer_bem: float
    transfer_analytic: float
    relative_error: float
    route_disagreement: float
    condition_number: float
    relative_residual: float

    @property
    def passed(self) -> bool:
        return self.relative_error <= 0.10


def solve_wavemaker(
    kind: str = "piston",
    *,
    kh: float = 1.0,
    depth: float = 1.0,
    panels: int = 64,
    stroke: float = 1.0,
    probe_wavelengths: float = 2.0,
    panel_nodes: int = 8,
    modes: int | None = None,
    gravity: float = GRAVITY,
) -> WavemakerResult:
    """Solve the flume radiation problem and compare with Biesel-Havelock.

    ``probe_wavelengths`` places the free-surface probe ``2 pi n / k``
    downstream of the wall (default two wavelengths), where the evanescent modes
    of the kernel have decayed to ``O(1e-8)``.
    """
    kind = _validate_kind(kind)
    kh = _validate_kh(kh)
    depth = _validate_depth(depth)
    surface = vertical_wavemaker(depth, panels)
    k = kh / depth
    omega = math.sqrt(gravity * k * math.tanh(kh))
    nu = omega * omega / gravity

    # Imposed wall velocity for a stroke S: x_wall(t) = -(S/2) Re exp(-i omega t)
    # gives u = -i omega S/2; only the modulus of eta_a enters H/S.
    shape = velocity_profile(kind, surface.centers[:, 1], depth)
    data = -0.5j * omega * stroke * shape * surface.normals[:, 0]

    sources, system = solve_sources(
        surface, data, nu, depth, panel_nodes=panel_nodes, modes=modes
    )
    residual = np.linalg.norm(system @ sources - data) / max(
        np.linalg.norm(data), 1e-300
    )
    eta_kochin = complex(
        kochin_amplitude(surface, sources, nu, depth, gravity=gravity, modes=modes)
    )

    probe_x = probe_wavelengths * 2.0 * math.pi / k
    phi_probe = free_surface_potential(
        surface, sources, nu, depth, [probe_x], panel_nodes=panel_nodes, modes=modes
    )[0]
    eta_probe = 1j * omega / gravity * phi_probe

    amplitude = abs(eta_kochin)
    transfer_bem = 2.0 * amplitude / stroke
    transfer_analytic = transfer_function(kind, kh)
    scale = max(abs(eta_kochin), abs(eta_probe), 1e-300)
    return WavemakerResult(
        kind=kind,
        kh=kh,
        depth=depth,
        panels=int(surface.panel_count),
        stroke=float(stroke),
        wavenumber=k,
        omega=omega,
        kochan_amplitude=eta_kochin,
        probe_amplitude=complex(eta_probe),
        probe_x=float(probe_x),
        transfer_bem=transfer_bem,
        transfer_analytic=transfer_analytic,
        relative_error=abs(transfer_bem - transfer_analytic) / transfer_analytic,
        route_disagreement=abs(eta_kochin - eta_probe) / scale,
        condition_number=float(np.linalg.cond(system)),
        relative_residual=float(residual),
    )


def evanescent_decay(
    kind: str = "piston",
    *,
    kh: float = 1.0,
    depth: float = 1.0,
    panels: int = 64,
    stroke: float = 1.0,
    offsets=(0.25, 0.5, 1.0, 2.0, 3.0, 4.0),
    panel_nodes: int = 8,
    modes: int | None = None,
    gravity: float = GRAVITY,
) -> list[dict]:
    """Elevation versus downstream distance, showing the evanescent modes dying.

    Returns one row per ``x = offsets * wavelength`` with the elevation modulus
    and its deviation from the far-field (``4`` wavelength) value.  The
    deviation is the frequency-domain counterpart of the flume's "wave-height
    attenuation": past two wavelengths what is left is the propagating mode
    alone and the modulus is flat to the reference-spacing of a wavemaker test.
    """
    kind = _validate_kind(kind)
    kh = _validate_kh(kh)
    depth = _validate_depth(depth)
    surface = vertical_wavemaker(depth, panels)
    k = kh / depth
    omega = math.sqrt(gravity * k * math.tanh(kh))
    nu = omega * omega / gravity
    shape = velocity_profile(kind, surface.centers[:, 1], depth)
    data = -0.5j * omega * stroke * shape * surface.normals[:, 0]
    sources, _ = solve_sources(
        surface, data, nu, depth, panel_nodes=panel_nodes, modes=modes
    )
    wavelength = 2.0 * math.pi / k
    x = np.asarray(offsets, dtype=float) * wavelength
    phi = free_surface_potential(
        surface, sources, nu, depth, x, panel_nodes=panel_nodes, modes=modes
    )
    eta = 1j * omega / gravity * phi
    # the farthest station is the reference: past ~1 wavelength only the
    # propagating mode is left, so the modulus there is the far-field value
    reference = float(abs(eta[-1]))
    rows = []
    for offset, xi, value in zip(np.asarray(offsets, dtype=float), x, eta):
        rows.append(
            {
                "offset_wavelengths": float(offset),
                "x_over_depth": float(xi / depth),
                "elevation_abs": float(abs(value)),
                "deviation_from_flat": float(abs(abs(value) - reference) / reference),
            }
        )
    return rows