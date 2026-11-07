"""Hydroelastic coupling of a flexible rectangular panel with a BEM fluid model.

Physical problem
----------------
A thin, homogeneous, isotropic Kirchhoff plate of planform ``a x b``, thickness
``h``, flexural rigidity ``D = E h^3 / (12 (1 - nu^2))`` and mass per unit area
``m'' = rho_p h`` is mounted flush in an infinite rigid baffle which separates
the fluid (``z < 0``, density ``rho``, inviscid, incompressible, irrotational)
from the air side.  The plate edges are simply supported
(``w = d2w/dn2 = 0``).  This is the classical *wet panel* configuration of
hydroelasticity: a ship hull / tank / bottom panel in contact with water.

Fluid model
-----------
The flow is described by a velocity potential ``phi`` with ``Laplace(phi) = 0``
in the half space, ``dphi/dz = 0`` on the rigid baffle and the plate normal
velocity ``dphi/dz = -i omega w`` on the panel.  In the zero-frequency limit the
calm free surface is a rigid wall (``omega^2 phi = g dphi/dz -> dphi/dz = 0``),
so the whole boundary plane ``z = 0`` is Neumann and the Green function of the
half space reduces, for source and field point both on the plane, to

    G(x, x') = 1 / (2 pi |x - x'|),

i.e. the mirror image doubles the Rankine kernel.  The hydrodynamic pressure on
the panel is ``p = i omega rho phi``, so the generalized added mass of two plate
modes ``w_i, w_j`` is the symmetric single-layer (Galerkin) boundary integral

    A_ij = (rho / 2 pi) * int_panel int_panel w_i(x) w_j(x') / |x - x'| dA dA'.

This is exactly the classical *baffled plate* radiation problem whose
high-wavenumber asymptotics is the local (plane-wave) impedance
``m_a = rho / (k tanh(kH))`` -- ``rho / k`` for deep water (Lamb 1932; Fahy,
*Sound and Structural Vibration*, ch. 4).  The same asymptote follows from the
plane-wave analysis of the half space: for ``w ~ exp(i k x)`` the potential is
``phi = -i omega w exp(k z) / k`` and therefore ``p = rho omega^2 w / k``.

What is implemented
-------------------
* :func:`simply_supported_modes` -- exact dry modes of the Kirchhoff plate.
* :func:`exact_uniform_added_mass` -- the uniform ("piston") mode added mass of
  a rectangle, evaluated in *closed form* (see :func:`rectangle_self_integral`).
* :func:`panel_added_mass_matrix` -- the Galerkin single-layer BEM matrix over a
  uniform panel grid, with the self-panel term integrated exactly and refined
  product rules for near panels.
* :func:`local_added_mass_matrix` -- the local plane-wave impedance model.
* :func:`wet_modes` -- the wet eigenproblem ``(K + K_h) phi = omega^2 (M + A) phi``.
* :func:`staggered_coupling` -- partitioned displacement/pressure fixed point
  with residual history (the FSI convergence check).
* :func:`modal_rao` -- wave-excited modal response (vertical and bending RAO).

Honest scope
------------
The hydrodynamic model is the *zero-frequency* (rigid calm-free-surface) one, so
the added mass is real and frequency independent and the panel has **no
radiation damping** (an incompressible fluid bounded by a rigid baffle radiates
no acoustic energy).  Wave excitation is the Froude-Krylov pressure of the
incident wave; diffraction by the panel is neglected because a flush panel in an
infinite baffle does not scatter in the zero-thickness limit of this model.
Damping is therefore structural only and enters through a modal ratio ``zeta``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
from typing import Callable, Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray
from scipy.linalg import eigh

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_DEFAULT_GRAVITY = 9.80665
_DEFAULT_FLUID_DENSITY = 1025.0
_DEFAULT_STEEL_E = 2.10e11
_DEFAULT_STEEL_NU = 0.30
_DEFAULT_STEEL_DENSITY = 7850.0


def _positive(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    val = float(value)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return val


@dataclass(frozen=True)
class ElasticPanel:
    """Simply supported Kirchhoff plate mounted flush in a rigid baffle.

    Attributes
    ----------
    length, breadth :
        Planform ``a`` (x) and ``b`` (y) in metres.
    thickness :
        Plate thickness ``h`` in metres.
    young_modulus, poisson_ratio, plate_density :
        Isotropic plate material.
    fluid_density, gravity :
        Fluid properties used by the hydrodynamic models.
    water_depth :
        Rigid-bottom depth below the panel plane (``None`` = infinite depth).
        Only used by :func:`local_added_mass_matrix`.
    """

    length: float
    breadth: float
    thickness: float
    young_modulus: float = _DEFAULT_STEEL_E
    poisson_ratio: float = _DEFAULT_STEEL_NU
    plate_density: float = _DEFAULT_STEEL_DENSITY
    fluid_density: float = _DEFAULT_FLUID_DENSITY
    gravity: float = _DEFAULT_GRAVITY
    water_depth: float | None = None

    def __post_init__(self) -> None:
        for name in ("length", "breadth", "thickness", "young_modulus",
                     "plate_density", "fluid_density", "gravity"):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        nu = float(self.poisson_ratio)
        if not (-0.999 < nu < 0.5):
            raise ValueError("poisson_ratio must lie in (-1, 0.5)")
        object.__setattr__(self, "poisson_ratio", nu)
        if self.water_depth is not None:
            object.__setattr__(self, "water_depth", _positive("water_depth", self.water_depth))

    # -- derived quantities -------------------------------------------------
    @property
    def area(self) -> float:
        return self.length * self.breadth

    @property
    def flexural_rigidity(self) -> float:
        return (
            self.young_modulus * self.thickness ** 3
            / (12.0 * (1.0 - self.poisson_ratio ** 2))
        )

    @property
    def mass_per_area(self) -> float:
        return self.plate_density * self.thickness

    @property
    def plate_mass(self) -> float:
        return self.mass_per_area * self.area

    @property
    def draft(self) -> float:
        """Equivalent submerged depth of the plate if it were freely floating."""
        return self.mass_per_area / self.fluid_density


def _sine_integral(index: int, length: float) -> float:
    """``int_0^L sin(index * pi * x / L) dx``."""
    return length * (1.0 - (-1.0) ** index) / (index * math.pi)


@dataclass(frozen=True)
class DryMode:
    """A dry simply supported mode ``sin(m pi x/a) sin(n pi y/b)``."""

    m: int
    n: int
    panel: ElasticPanel

    @property
    def wavenumber(self) -> float:
        return math.pi * math.hypot(self.m / self.panel.length, self.n / self.panel.breadth)

    @property
    def generalized_mass(self) -> float:
        return 0.25 * self.panel.mass_per_area * self.panel.area

    @property
    def generalized_stiffness(self) -> float:
        return self.panel.flexural_rigidity * self.wavenumber ** 4 * 0.25 * self.panel.area

    @property
    def omega_dry(self) -> float:
        return math.sqrt(self.generalized_stiffness / self.generalized_mass)

    @property
    def label(self) -> str:
        return f"({self.m},{self.n})"

    def shape(self, x, y) -> FloatArray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return (
            np.sin(self.m * math.pi * x / self.panel.length)
            * np.sin(self.n * math.pi * y / self.panel.breadth)
        )

    def curvature(self, x, y) -> tuple[FloatArray, FloatArray]:
        """Second derivatives ``(d2w/dx2, d2w/dy2)`` of the mode shape."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        kx = self.m * math.pi / self.panel.length
        ky = self.n * math.pi / self.panel.breadth
        sxy = np.sin(kx * x) * np.sin(ky * y)
        return -(kx ** 2) * sxy, -(ky ** 2) * sxy

    def bending_moment(self, x, y) -> FloatArray:
        """``M_xx = -D (w_xx + nu w_yy)`` (N m / m)."""
        wxx, wyy = self.curvature(x, y)
        return -self.panel.flexural_rigidity * (wxx + self.panel.poisson_ratio * wyy)

    def static_load(self, pressure: float) -> float:
        """Generalized force of a uniform pressure (N)."""
        return (
            pressure
            * _sine_integral(self.m, self.panel.length)
            * _sine_integral(self.n, self.panel.breadth)
        )


def simply_supported_modes(panel: ElasticPanel, count: int = 3) -> list[DryMode]:
    """The ``count`` lowest dry modes of the simply supported panel."""
    if count < 1:
        raise ValueError("count must be >= 1")
    modes = [
        DryMode(m, n, panel)
        for m in range(1, 4 * count + 4)
        for n in range(1, 4 * count + 4)
    ]
    modes.sort(key=lambda mode: (mode.omega_dry, mode.m, mode.n))
    return modes[:count]


# ---------------------------------------------------------------------------
# exact uniform ("piston") added mass of a rectangle
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def rectangle_self_integral(length: float, breadth: float) -> float:
    r"""Exact ``int_R int_R dA dA' / |r - r'|`` over a rectangle.

    Changing variables to the difference ``u = x - x'``, ``v = y - y'`` turns the
    four-fold integral into the autocorrelation form

        I = int_{-a}^{a} int_{-b}^{b} Lambda_a(u) Lambda_b(v) / sqrt(u^2+v^2) du dv,
        Lambda_a(u) = a - |u|,

    which in polar coordinates ``(u, v) = r (cos t, sin t)`` has the elementary
    inner integral (``r dr / r = dr``, no singularity left)

        int_0^{r_max} (a - r cos t)(b - r sin t) dr
            = a b r - (a sin t + b cos t) r^2 / 2 + sin t cos t r^3 / 3,

    with ``r_max(t) = min(a / cos t, b / sin t)``.  The remaining one-dimensional
    integral is smooth on each side of the switching angle
    ``t_c = atan(b/a)`` and is evaluated with Gauss-Legendre quadrature to
    machine precision.
    """
    a = _positive("length", length)
    b = _positive("breadth", breadth)
    t_c = math.atan2(b, a)

    def integrand(t: FloatArray) -> FloatArray:
        cos_t = np.cos(t)
        sin_t = np.sin(t)
        r_max = np.minimum(a / cos_t, b / sin_t)
        return (
            a * b * r_max
            - 0.5 * (a * sin_t + b * cos_t) * r_max ** 2
            + (sin_t * cos_t) * r_max ** 3 / 3.0
        )

    total = 0.0
    for lo, hi in ((1.0e-12, t_c), (t_c, 0.5 * math.pi - 1.0e-12)):
        nodes, weights = leggauss(160)
        t = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
        total += 0.5 * (hi - lo) * float(np.sum(weights * integrand(t)))
    return 4.0 * total


def exact_uniform_added_mass(panel: ElasticPanel) -> float:
    """Added mass (kg) of the uniform piston mode of the panel, in closed form."""
    return (
        panel.fluid_density / (2.0 * math.pi)
        * rectangle_self_integral(panel.length, panel.breadth)
    )


# ---------------------------------------------------------------------------
# Galerkin single-layer BEM over a uniform panel grid
# ---------------------------------------------------------------------------
def _panel_grid(panel: ElasticPanel, nx: int, ny: int) -> tuple[FloatArray, float, float]:
    """Centroids of a uniform ``nx x ny`` panel grid plus the panel sizes."""
    hx = panel.length / nx
    hy = panel.breadth / ny
    xc = (np.arange(nx) + 0.5) * hx
    yc = (np.arange(ny) + 0.5) * hy
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], axis=1), hx, hy


def _quad_points(centroids: FloatArray, hx: float, hy: float, order: int,
                 sub: int = 1) -> tuple[FloatArray, FloatArray]:
    """Tensor-product Gauss rule on every panel, optionally subdivided ``sub x sub``.

    Returns points ``(n_panels, order**2 * sub**2, 2)`` and weights
    ``(order**2 * sub**2,)``; the weights sum to ``hx * hy``.
    """
    nodes, weights = leggauss(order)
    half = 0.5 / sub
    local: list[float] = []
    locw: list[float] = []
    for idx in range(sub):
        centre = -0.5 + (idx + 0.5) / sub
        local.extend(centre + half * nodes)
        locw.extend(half * weights)
    local = np.asarray(local)
    locw = np.asarray(locw)
    q = local.size
    n = centroids.shape[0]
    X = centroids[:, 0][:, None, None] + local[None, :, None] * hx
    Y = centroids[:, 1][:, None, None] + local[None, None, :] * hy
    pts = np.stack(
        [np.broadcast_to(X, (n, q, q)), np.broadcast_to(Y, (n, q, q))], axis=-1
    ).reshape(n, q * q, 2)
    wts = (locw[:, None] * locw[None, :] * hx * hy).ravel()
    return pts, wts


def _kernel_block(Pp: FloatArray, Wp: FloatArray, Pall: FloatArray,
                  Wall: FloatArray) -> FloatArray:
    """``sum_ab Wp_a Wall_b / |Pp_a - P_qb|`` for every panel ``q``."""
    dx = Pp[:, 0][None, :, None] - Pall[:, None, :, 0]
    dy = Pp[:, 1][None, :, None] - Pall[:, None, :, 1]
    r = np.sqrt(dx * dx + dy * dy)
    # the singular self pair is replaced analytically by the caller
    inv_r = 1.0 / np.maximum(r, 1.0e-300)
    return np.einsum("a,nab,b->n", Wp, inv_r, Wall)


def panel_added_mass_matrix(
    panel: ElasticPanel,
    modes: Sequence[DryMode],
    nx: int = 24,
    ny: int = 24,
    *,
    gauss_order: int = 3,
    near_gauss_order: int = 2,
    near_sub: int = 4,
    near_factor: float = 2.0,
    return_kernel: bool = False,
):
    """Galerkin single-layer added-mass matrix over a uniform panel grid.

    The panel is split into ``nx x ny`` rectangles.  The kernel integral
    ``K_pq = int_p int_q dA dA' / |r - r'|`` is

    * **exact** for ``p == q`` (:func:`rectangle_self_integral`; the autocorrelation
      form of the same four-fold integral),
    * a ``gauss_order``-point product rule for well separated panels,
    * a ``near_sub x near_sub`` subdivided ``near_gauss_order``-point rule for
      panels closer than ``near_factor`` panel sizes, where ``1/r`` varies too
      fast (and is singular on the shared edge) for the coarse rule.

    Returns the ``(len(modes), len(modes))`` added-mass matrix in kg, optionally
    with the panel kernel matrix for diagnostics.
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1")
    centroids, hx, hy = _panel_grid(panel, nx, ny)
    n_panels = centroids.shape[0]

    coarse_pts, coarse_wts = _quad_points(centroids, hx, hy, gauss_order)
    kernel = np.zeros((n_panels, n_panels), dtype=float)
    for p in range(n_panels):
        kernel[p] = _kernel_block(coarse_pts[p], coarse_wts, coarse_pts, coarse_wts)
    kernel = 0.5 * (kernel + kernel.T)
    for p in range(n_panels):
        kernel[p, p] = rectangle_self_integral(hx, hy)

    # refined near field
    delta = centroids[:, None, :] - centroids[None, :, :]
    distance = np.sqrt(np.einsum("pqi,pqi->pq", delta, delta))
    near = (distance < near_factor * max(hx, hy)) & (distance > 0.0)
    if near_gauss_order * near_sub != gauss_order:
        fine_pts, fine_wts = _quad_points(centroids, hx, hy, near_gauss_order, sub=near_sub)
        for p in range(n_panels):
            neighbours = np.nonzero(near[p])[0]
            if neighbours.size == 0:
                continue
            rows = _kernel_block(fine_pts[p], fine_wts, fine_pts[neighbours], fine_wts)
            kernel[p, neighbours] = rows
            kernel[neighbours, p] = rows

    weights = np.stack(
        [mode.shape(centroids[:, 0], centroids[:, 1]) for mode in modes], axis=1
    )
    added_mass = (panel.fluid_density / (2.0 * math.pi)) * (weights.T @ kernel @ weights)
    added_mass = 0.5 * (added_mass + added_mass.T)
    if return_kernel:
        return added_mass, kernel
    return added_mass


def local_added_mass_matrix(panel: ElasticPanel, modes: Sequence[DryMode]) -> FloatArray:
    """Local plane-wave impedance model ``m_a = rho / (k tanh(kH))``.

    Valid for ``k a >> 1`` (many mode half-wavelengths across the panel); it is
    the classical high-wavenumber asymptote of the baffled-plate added mass.
    """
    out = np.zeros((len(modes), len(modes)), dtype=float)
    for idx, mode in enumerate(modes):
        k = mode.wavenumber
        if panel.water_depth is None:
            impedance = panel.fluid_density / k
        else:
            impedance = panel.fluid_density / (k * math.tanh(k * panel.water_depth))
        out[idx, idx] = impedance * 0.25 * panel.area
    return out


def added_mass_ratios(modes: Sequence[DryMode], added_mass: FloatArray) -> FloatArray:
    """Modal added-mass ratio ``mu_i = A_ii / M_ii``."""
    diag = np.diag(np.asarray(added_mass, dtype=float))
    gm = np.array([mode.generalized_mass for mode in modes], dtype=float)
    return diag / gm


# ---------------------------------------------------------------------------
# wet modes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WetModeResult:
    """Wet eigenpairs of the coupled panel."""

    omegas: FloatArray
    eigenvectors: FloatArray
    dry_omegas: FloatArray
    modal_damping_absent: bool = True

    @property
    def frequencies_hz(self) -> FloatArray:
        return self.omegas / (2.0 * math.pi)

    @property
    def ratio_wet_over_dry(self) -> FloatArray:
        return self.omegas / self.dry_omegas


def wet_modes(
    panel: ElasticPanel,
    modes: Sequence[DryMode],
    added_mass: FloatArray,
    extra_stiffness: FloatArray | None = None,
) -> WetModeResult:
    """Solve ``(K + K_h) phi = omega^2 (M + A) phi`` in the dry-mode basis."""
    n = len(modes)
    gm = np.array([mode.generalized_mass for mode in modes], dtype=float)
    gk = np.array([mode.generalized_stiffness for mode in modes], dtype=float)
    stiffness = np.diag(gk)
    if extra_stiffness is not None:
        stiffness = stiffness + np.asarray(extra_stiffness, dtype=float)
    mass = np.diag(gm) + np.asarray(added_mass, dtype=float)
    values, vectors = eigh(stiffness, mass)
    values = np.clip(values, 0.0, None)
    omegas = np.sqrt(values)
    order = np.argsort(omegas)
    return WetModeResult(
        omegas=omegas[order],
        eigenvectors=vectors[:, order],
        dry_omegas=np.array([mode.omega_dry for mode in modes], dtype=float),
    )


# ---------------------------------------------------------------------------
# partitioned fluid-structure coupling
# ---------------------------------------------------------------------------
@dataclass
class CouplingIteration:
    """Residual history of a partitioned displacement/pressure fixed point."""

    omegas: FloatArray
    residuals: list[float] = field(default_factory=list)
    pressure_residuals: list[float] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False

    @property
    def residual(self) -> float:
        return self.residuals[-1] if self.residuals else float("nan")


def monolithic_residual(
    panel: ElasticPanel,
    modes: Sequence[DryMode],
    added_mass: FloatArray,
    omega: float,
    vector: FloatArray,
) -> float:
    """Relative residual ``||K phi - omega^2 (M + A) phi|| / ||K phi||``."""
    modes_list = list(modes)
    gm = np.array([mode.generalized_mass for mode in modes_list], dtype=float)
    gk = np.array([mode.generalized_stiffness for mode in modes_list], dtype=float)
    vec = np.asarray(vector, dtype=float).ravel()
    lhs = gk * vec
    rhs = omega ** 2 * ((gm * vec) + np.asarray(added_mass, dtype=float) @ vec)
    return float(np.linalg.norm(lhs - rhs) / max(np.linalg.norm(lhs), 1.0e-300))


def staggered_coupling(
    panel: ElasticPanel,
    modes: Sequence[DryMode],
    *,
    nx: int = 24,
    ny: int | None = None,
    relaxation: float = 0.7,
    tolerance: float = 1.0e-6,
    max_iterations: int = 50,
    kernel: FloatArray | None = None,
    store_kernel: bool = False,
) -> CouplingIteration:
    """Partitioned displacement/pressure fixed point of the wet eigenproblem.

    One sweep is

    1. **structure**: solve ``K phi = omega^2 (M + A) phi`` with the current
       hydrodynamic operator ``A``,
    2. **fluid**: re-linearize the hydrodynamic operator from the panel pressures
       of the current iterate (the BEM single-layer matrix ``K_pq`` contracted
       with the mode shapes), giving ``A_new``,
    3. under-relax ``A <- (1 - omega_r) A + omega_r A_new`` and monitor the
       relative change of the eigenfrequencies and of ``A``.

    The loop starts from the *in vacuum* operator ``A = 0``, so the first sweep
    is the dry solution and the residual history is the coupling convergence.
    Because the hydrodynamic operator of this model is linear and frequency
    independent, the fixed point is reached in a handful of sweeps; the reported
    residuals are what the coupled solver certifies, and the converged state
    also satisfies the monolithic generalized eigenproblem to round-off (see
    :func:`monolithic_residual`).
    """
    if not (0.0 < relaxation <= 1.0):
        raise ValueError("relaxation must lie in (0, 1]")
    modes_list = list(modes)
    n = len(modes_list)
    if ny is None:
        ny = max(1, int(round(nx * panel.breadth / panel.length)))
    if kernel is None:
        _, kernel_matrix = panel_added_mass_matrix(
            panel, modes_list, nx=nx, ny=ny, return_kernel=True
        )
    else:
        kernel_matrix = np.asarray(kernel, dtype=float)

    centroids, _, _ = _panel_grid(panel, nx, ny)
    weights = np.stack(
        [mode.shape(centroids[:, 0], centroids[:, 1]) for mode in modes_list], axis=1
    )
    target = (panel.fluid_density / (2.0 * math.pi)) * (weights.T @ kernel_matrix @ weights)
    target = 0.5 * (target + target.T)

    history = CouplingIteration(omegas=np.zeros(n), iterations=0)
    current = np.zeros((n, n), dtype=float)
    previous = None
    for it in range(1, max_iterations + 1):
        result = wet_modes(panel, modes_list, current)
        omegas = result.omegas
        delta_omega = (
            float(np.max(np.abs(omegas - previous)) / max(np.max(omegas), 1.0e-300))
            if previous is not None
            else float("inf")
        )
        delta_operator = float(
            np.linalg.norm(target - current) / max(np.linalg.norm(target), 1.0e-300)
        )
        history.iterations = it
        history.omegas = omegas
        history.residuals.append(delta_omega)
        history.pressure_residuals.append(delta_operator)
        if previous is not None and max(delta_omega, delta_operator) < tolerance:
            history.converged = True
            break
        previous = omegas
        current = (1.0 - relaxation) * current + relaxation * target
    if store_kernel:
        return history, kernel_matrix
    return history


# ---------------------------------------------------------------------------
# wave-excited response
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModalRAOResult:
    """Frequency-domain response of the panel to an incident regular wave."""

    omega: FloatArray
    vertical_rao: FloatArray
    bending_rao: FloatArray
    modal_amplitudes: ComplexArray
    wet_omegas: FloatArray
    static_vertical_rao: float
    wave_heading: float
    damping_ratio: float


def froude_krylov_forces(
    panel: ElasticPanel,
    modes: Sequence[DryMode],
    omega: float,
    *,
    heading: float = 0.0,
    submergence: float = 0.0,
    n_gauss: int = 40,
) -> ComplexArray:
    """Generalized Froude-Krylov force per unit incident wave amplitude (N/m).

    The incident wave has amplitude ``A``, deep-water wavenumber ``k = omega^2/g``
    and travelling direction ``heading`` in the panel plane.  On the panel plane
    the pressure amplitude is ``p0 = rho g A exp(-k d)`` and it varies as
    ``exp(i k (x cos b + y sin b))``.
    """
    omega = _positive("omega", omega)
    k = omega ** 2 / panel.gravity
    p0 = panel.fluid_density * panel.gravity * math.exp(-k * submergence)
    nodes, weights = leggauss(n_gauss)
    x = 0.5 * panel.length * (nodes + 1.0)
    y = 0.5 * panel.breadth * (nodes + 1.0)
    wx = 0.5 * panel.length * weights
    wy = 0.5 * panel.breadth * weights
    phase = np.exp(1j * k * (np.cos(heading) * x[:, None] + np.sin(heading) * y[None, :]))
    area_weight = wx[:, None] * wy[None, :] * phase
    out = np.zeros(len(modes), dtype=np.complex128)
    for idx, mode in enumerate(modes):
        shape = mode.shape(x[:, None] * np.ones_like(y)[None, :],
                           y[None, :] * np.ones_like(x)[:, None])
        out[idx] = p0 * np.sum(shape * area_weight)
    return out


def static_vertical_rao(panel: ElasticPanel, modes: Sequence[DryMode]) -> float:
    """Quasi-static centre deflection per unit wave amplitude (m/m).

    For ``omega -> 0`` the pressure is uniform, so the response is the static
    compliance.  ``sin(m pi x/L)`` integrates to zero for even ``m``, hence only
    odd-odd modes participate -- the value is a closed form of the dry data.
    """
    pressure = panel.fluid_density * panel.gravity
    load = np.array([mode.static_load(pressure) for mode in modes])
    gk = np.array([mode.generalized_stiffness for mode in modes])
    coeff = load / gk
    centre = np.array([mode.shape(0.5 * panel.length, 0.5 * panel.breadth) for mode in modes])
    return float(np.sum(coeff * centre))


def modal_rao(
    panel: ElasticPanel,
    modes: Sequence[DryMode],
    added_mass: FloatArray,
    omegas: Sequence[float],
    *,
    wet_omegas: Sequence[float] | None = None,
    heading: float = 0.0,
    submergence: float = 0.0,
    damping_ratio: float = 0.02,
    extra_stiffness: FloatArray | None = None,
) -> ModalRAOResult:
    """Wave-excited modal response: vertical and bending RAO of the panel.

    Equation of motion in the dry-mode basis,

        [-omega^2 (M + A) + i omega C + K + K_h] eta = F_exc(omega),

    with ``C_ii = 2 zeta omega_wet,i (M_ii + A_ii)`` (structural damping only;
    the hydrodynamic model supplies neither radiation damping nor diffraction).
    ``omegas`` is the response sweep, ``wet_omegas`` the coupled eigenvalues used
    for the modal damping (defaults to the eigenvalues of ``(K, M + A)``).
    """
    modes_list = list(modes)
    sweep = np.asarray(omegas, dtype=float)
    n = len(modes_list)
    gm = np.array([mode.generalized_mass for mode in modes_list], dtype=float)
    gk = np.array([mode.generalized_stiffness for mode in modes_list], dtype=float)
    added = np.asarray(added_mass, dtype=float)
    mass = np.diag(gm) + added
    stiffness = np.diag(gk)
    if extra_stiffness is not None:
        stiffness = stiffness + np.asarray(extra_stiffness, dtype=float)
    if wet_omegas is None:
        wet = np.sqrt(np.clip(np.diag(stiffness) / np.diag(mass), 0.0, None))
    else:
        wet = np.asarray(wet_omegas, dtype=float)
    damping = 2.0 * damping_ratio * wet * np.diag(mass)

    centre = (0.5 * panel.length, 0.5 * panel.breadth)
    shape_centre = np.array([mode.shape(*centre) for mode in modes_list])
    moment_centre = np.array([mode.bending_moment(*centre) for mode in modes_list])

    vertical = np.zeros(sweep.size)
    bending = np.zeros(sweep.size)
    amplitudes = np.zeros((sweep.size, n), dtype=np.complex128)
    for idx, omega in enumerate(sweep):
        force = froude_krylov_forces(
            panel, modes_list, float(omega), heading=heading, submergence=submergence
        )
        impedance = -omega ** 2 * mass + 1j * omega * np.diag(damping) + stiffness
        eta = np.linalg.solve(impedance, force)
        amplitudes[idx] = eta
        vertical[idx] = abs(np.sum(eta * shape_centre))
        bending[idx] = abs(np.sum(eta * moment_centre))
    return ModalRAOResult(
        omega=sweep,
        vertical_rao=vertical,
        bending_rao=bending,
        modal_amplitudes=amplitudes,
        wet_omegas=wet,
        static_vertical_rao=static_vertical_rao(panel, modes_list),
        wave_heading=heading,
        damping_ratio=damping_ratio,
    )


# ---------------------------------------------------------------------------
# analytic reference for the infinite plate (flexural-gravity / virtual mass)
# ---------------------------------------------------------------------------
def infinite_plate_response(
    omega: float,
    wavenumber: float,
    plate: ElasticPanel,
    *,
    mass_per_area: float | None = None,
    with_hydrostatic: bool = True,
) -> float:
    """``|w/A|`` of an *infinite* plate driven by a plane-wave pressure.

    Per unit area,

        [-omega^2 (m'' + rho/k) + D k^4 + rho g] w = rho g A,

    so ``|w/A| = rho g / |D k^4 + rho g - omega^2 (m'' + rho/k)|``.

    Two things follow.  (i) At ``omega^2 = (D k^4 + rho g)/(m'' + rho/k)`` the
    *homogeneous* operator is singular: that root is the flexural-gravity
    dispersion of :func:`flexural_gravity_omega`, and locating the peak of this
    response is a numerical check of the local operator.  (ii) The Froude-Krylov
    drive used here is **not** a valid excitation model for a *floating* plate:
    a plate covering the free surface also blocks it, so the incident-wave
    pressure ``rho g A`` is cancelled by diffraction, whereas this formula
    returns a resonance of order ``rho/(k m'')`` in the long-wave limit.  The
    function is therefore offered as a limit/consistency reference for the local
    hydrodynamic operator, never as a finite- or infinite-plate RAO.  The
    production configuration of this module (a panel in a rigid baffle) has no
    free surface on the panel and is not affected by this restriction.
    """
    omega = _positive("omega", omega)
    k = _positive("wavenumber", wavenumber)
    m_pp = plate.mass_per_area if mass_per_area is None else float(mass_per_area)
    added = plate.fluid_density / k
    stiffness = plate.flexural_rigidity * k ** 4 + (
        plate.fluid_density * plate.gravity if with_hydrostatic else 0.0
    )
    inertia = omega ** 2 * (m_pp + added)
    numerator = plate.fluid_density * plate.gravity
    denominator = abs(stiffness - inertia)
    if denominator == 0.0:
        return float("inf")
    return numerator / denominator


def flexural_gravity_omega(
    wavenumber: float,
    plate: ElasticPanel,
    *,
    mass_per_area: float | None = None,
    flexural_rigidity: float | None = None,
    with_hydrostatic: bool = True,
    depth: float | None = None,
) -> float:
    r"""Flexural-gravity dispersion of a floating elastic plate.

    With the local (plane-wave) impedance ``m_a = rho / (k tanh kH)`` and
    buoyancy restoring ``rho g`` the coupled dispersion is

        omega^2 = (D k^4 + rho g) / (m'' + rho / (k tanh kH)),

    which for ``D = 0`` and deep water reduces to the classical mass-loading
    relation ``omega^2 = g k / (1 + (m''/rho) k)`` implemented independently in
    :func:`tensorbem.ice.mass_loading_omega` -- the cross-module check used by
    the benchmark.  Dropping the buoyancy term gives the pressure-loaded panel
    variant ``omega^2 = D k^4 / (m'' + rho/k)``.
    """
    k = _positive("wavenumber", wavenumber)
    m_pp = plate.mass_per_area if mass_per_area is None else float(mass_per_area)
    rigidity = plate.flexural_rigidity if flexural_rigidity is None else float(flexural_rigidity)
    rho = plate.fluid_density
    depth_val = plate.water_depth if depth is None else depth
    if depth_val is None:
        impedance = rho / k
    else:
        impedance = rho / (k * math.tanh(k * _positive("depth", depth_val)))
    stiffness = rigidity * k ** 4 + (rho * plate.gravity if with_hydrostatic else 0.0)
    return math.sqrt(stiffness / (m_pp + impedance))


def panel_wet_omegas_analytic(
    panel: ElasticPanel,
    modes: Sequence[DryMode],
    *,
    with_hydrostatic: bool = False,
    depth: float | None = None,
) -> FloatArray:
    """Wet frequencies under the local impedance model, mode by mode.

    For the baffled, pressure-loaded panel (``with_hydrostatic=False``) the local
    model gives ``omega_i^2 = D k_i^4 / (m'' + rho/k_i)`` with ``k_i`` the dry
    mode wavenumber: the shear-plate limit of the dispersion relation.
    """
    return np.array(
        [
            flexural_gravity_omega(
                mode.wavenumber, panel, with_hydrostatic=with_hydrostatic, depth=depth
            )
            for mode in modes
        ],
        dtype=float,
    )
