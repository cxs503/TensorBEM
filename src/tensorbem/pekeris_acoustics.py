"""Pekeris two-layer shallow-water normal-mode propagation with an optional elastic ice sheet.

Range-independent, horizontally stratified two-layer fluid waveguide (water column
``0 <= z <= H`` over a fluid half-space ``z > H``) with a point source in the water.
Two top boundary conditions are supported: pressure release (Pekeris 1948 ideal
shallow-water problem) and a thin Kirchhoff-Love elastic ice sheet (``E``, ``h``,
``rho_i``, ``nu``, vacuum above).

Nothing here embeds a literature number: every quantity is computed.  Two independent
solution routes are provided so that one can act as the reference for the other.

* analytic route: closed-form vertical dispersion relation + bracketed root finding +
  modal amplitudes from *numerical residues* of the spectral depth Green function (no
  hand-derived normalisation is used, which is what makes the elastic-plate case
  tractable: the plate condition depends on ``k_r`` and destroys self-adjointness).
* finite-difference route: the same vertical problem discretised with 2nd-order finite
  differences, the half-space entering through its exact impedance relation
  ``dP/dz = -gamma P`` at ``z = H``.  It never uses the transcendental equation, the
  closed-form mode shapes or a normalisation integral, so it independently checks the
  analytic route.

Conventions
-----------
Time factor ``exp(-i omega t)``; ``k_z = sqrt(k1^2 - k_r^2)`` (branch ``Re >= 0``);
``gamma = sqrt(k_r^2 - k2^2)`` with ``Re gamma >= 0, Im gamma <= 0``.  Transmission loss
uses the standard reference ``|p| = 1`` at 1 m, ``TL = -20 log10(4 pi |p|)``, so that
free space gives ``TL = 20 log10(r)``.

Far field: ``p(r, z) = (i/4) sum_m R_m H_0^{(1)}(k_rm r)`` with
``R_m = Res_{k_r = k_rm} [(k_r / k_z) G(k_r; z, z_s)]`` and ``G`` the vertical Green
function satisfying ``G'' + k_z^2 G = 2 i k_z delta(z - z_s)`` (free-space limit
``exp(i k_z |z - z_s|)``).  The prefactor follows from the Sommerfeld identity
``exp(i k R)/R = int_0^inf (k_r/k_z) J_0(k_r r) exp(i k_z |z|) dk_r`` and closing the
wavenumber integral in the upper ``k_r`` half-plane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.special import hankel1

# ----------------------------------------------------------------------------------
# media
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class PekerisCase:
    """Two-layer fluid waveguide (water column over fluid half-space)."""

    depth_m: float = 100.0
    c_water_m_s: float = 1500.0
    c_bottom_m_s: float = 1600.0
    rho_water_kg_m3: float = 1000.0
    rho_bottom_kg_m3: float = 1800.0

    @property
    def b(self) -> float:
        """``rho_water / rho_bottom``, the density ratio of the interface conditions."""
        return self.rho_water_kg_m3 / self.rho_bottom_kg_m3

    def k1(self, f: float) -> float:
        return 2.0 * math.pi * f / self.c_water_m_s

    def k2(self, f: float) -> float:
        return 2.0 * math.pi * f / self.c_bottom_m_s

    def kz_max(self, f: float) -> float:
        """Upper bound of ``k_z`` for trapped modes (``k_r -> k2``)."""
        return math.sqrt(max(self.k1(f) ** 2 - self.k2(f) ** 2, 0.0))

    def to_dict(self) -> dict:
        return {
            "water_depth_m": self.depth_m,
            "c_water_m_s": self.c_water_m_s,
            "c_bottom_m_s": self.c_bottom_m_s,
            "rho_water_kg_m3": self.rho_water_kg_m3,
            "rho_bottom_kg_m3": self.rho_bottom_kg_m3,
            "density_ratio_rho_w_over_rho_b": self.b,
        }


@dataclass(frozen=True)
class IcePlate:
    """Thin elastic (Kirchhoff-Love) ice sheet floating on the water, vacuum above."""

    young_Pa: float
    thickness_m: float
    rho_kg_m3: float = 910.0
    nu: float = 0.3

    @property
    def flexural_rigidity(self) -> float:
        """``D = E h^3 / (12 (1 - nu^2))`` [N m]."""
        return self.young_Pa * self.thickness_m ** 3 / (12.0 * (1.0 - self.nu ** 2))

    @property
    def mass_per_area(self) -> float:
        return self.rho_kg_m3 * self.thickness_m

    @property
    def plate_wave_speed(self) -> float:
        """In-vacuo bending reference speed ``c_f = (D / (rho_i h))^{1/4}``."""
        return (self.flexural_rigidity / self.mass_per_area) ** 0.25

    def bending_wavenumber(self, f: float) -> float:
        """In-vacuo bending wavenumber, ``k_b^4 = rho_i h omega^2 / D``."""
        return (self.mass_per_area * (2.0 * math.pi * f) ** 2 / self.flexural_rigidity) ** 0.25

    def coincidence_frequency(self, c_water: float) -> float:
        """Frequency where the in-vacuo bending wavenumber equals ``k1``."""
        return (c_water / self.plate_wave_speed) ** 2 / (2.0 * math.pi)

    def top_admittance(self, omega: float, kr: complex, rho_water: float) -> complex:
        """Robin coefficient ``m`` of ``dP/dz = m P`` at ``z = 0``.

        Plate momentum balance with the water load and continuity of the normal
        velocity give ``m = rho_w omega^2 / (rho_i h omega^2 - D k_r^4)``.  The limits
        reproduce the idealised boundaries: ``h -> 0`` gives ``m -> inf`` (pressure
        release), ``h -> inf`` gives ``m -> 0`` (rigid wall), and
        ``k_r^4 = rho_i h omega^2 / D`` (in-vacuo plate resonance) again gives
        ``m -> inf``.
        """
        den = self.mass_per_area * omega ** 2 - self.flexural_rigidity * kr ** 4
        return rho_water * omega ** 2 / den

    def to_dict(self) -> dict:
        return {
            "young_modulus_Pa": self.young_Pa,
            "thickness_m": self.thickness_m,
            "rho_ice_kg_m3": self.rho_kg_m3,
            "poisson_ratio": self.nu,
            "flexural_rigidity_Nm": self.flexural_rigidity,
            "mass_per_area_kg_m2": self.mass_per_area,
            "plate_wave_speed_m_s": self.plate_wave_speed,
        }


# ----------------------------------------------------------------------------------
# bottom models (domain truncation control)
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class HalfSpaceBottom:
    """Exact fluid half-space (reference model, no truncation)."""

    kind: str = "half_space"
    thickness_m: float | None = None


@dataclass(frozen=True)
class FiniteLayerBottom:
    """Truncated bottom: homogeneous layer of thickness ``thickness_m``, free surface below.

    Used only by the *domain truncation* convergence study: doubling
    ``thickness_m`` must not change the transmission loss once the layer is thick
    enough (criterion "domain length doubling").
    """

    thickness_m: float
    kind: str = "finite_layer"


# ----------------------------------------------------------------------------------
# branches and vertical Green function (closed form, complex k_r)
# ----------------------------------------------------------------------------------


def _kz_of(k1: float, kr: complex) -> complex:
    return np.sqrt(k1 * k1 - kr * kr + 0j)


def _gamma_of(k2: float, kr: complex) -> complex:
    """Half-space decay, branch ``Re(gamma) > 0`` (and ``Im(gamma) <= 0`` when
    ``Re(gamma) == 0``).

    The branch cut runs along the real segment ``|kr| <= k2``.  Choosing the branch by
    continuity of ``Re(gamma) > 0`` in the trapped window (``Re kr > k2``) is essential
    for complex-contour work: a branch that flips on ``Im(gamma)`` is discontinuous on
    any circle that straddles the real axis, which silently halves the computed
    residue.
    """
    g = np.sqrt(kr * kr - k2 * k2 + 0j)
    if (g.real < 0.0) or (g.real == 0.0 and g.imag > 0.0):
        g = -g
    return g


def bottom_admittance(
    case: PekerisCase,
    f: float,
    kr: complex,
    bottom: HalfSpaceBottom | FiniteLayerBottom | None = None,
) -> complex:
    """Downward-looking admittance ``Y`` with ``P'(H) = -Y P(H)`` at the interface.

    Half-space: ``Y = b gamma``.  Finite layer of thickness ``d`` closed by a pressure
    release: ``Y = b k_z2 cot(k_z2 d)``, which tends to ``b gamma`` as ``d -> inf``.
    """
    b, k2 = case.b, case.k2(f)
    if bottom is None or bottom.kind == "half_space":
        return b * _gamma_of(k2, kr)
    kz2 = np.sqrt(k2 * k2 - kr * kr + 0j)
    return b * kz2 / np.tan(kz2 * bottom.thickness_m)


def depth_green(
    kr: complex,
    z: float,
    zs: float,
    case: PekerisCase,
    f: float,
    ice: IcePlate | None = None,
    bottom: HalfSpaceBottom | FiniteLayerBottom | None = None,
) -> complex:
    """Vertical Green function ``G(k_r; z, z_s)``, free-space limit ``exp(i k_z |z-z_s|)``."""
    k1 = case.k1(f)
    H = case.depth_m
    kz = _kz_of(k1, kr)
    Y = bottom_admittance(case, f, kr, bottom)

    if ice is None:

        def u(zz: complex) -> complex:
            return np.sin(kz * zz)

        def up(zz: complex) -> complex:
            return kz * np.cos(kz * zz)

    else:
        m = ice.top_admittance(2.0 * math.pi * f, kr, case.rho_water_kg_m3)

        def u(zz: complex) -> complex:
            return np.cos(kz * zz) + (m / kz) * np.sin(kz * zz)

        def up(zz: complex) -> complex:
            return -kz * np.sin(kz * zz) + m * np.cos(kz * zz)

    def v(zz: complex) -> complex:
        return np.cos(kz * (zz - H)) - (Y / kz) * np.sin(kz * (zz - H))

    def vp(zz: complex) -> complex:
        return -kz * np.sin(kz * (zz - H)) - Y * np.cos(kz * (zz - H))

    wronskian = u(0.0) * vp(0.0) - v(0.0) * up(0.0)
    src = 2j * kz  # delta strength: G'(zs+) - G'(zs-) = 2 i k_z
    lo, hi = (z, zs) if z < zs else (zs, z)
    return src * u(lo) * v(hi) / wronskian


def dispersion_function(
    kr: complex,
    case: PekerisCase,
    f: float,
    ice: IcePlate | None = None,
    bottom: HalfSpaceBottom | FiniteLayerBottom | None = None,
) -> complex:
    """Wronskian ``W(k_r)``; its zeros are the modal horizontal wavenumbers."""
    k1 = case.k1(f)
    H = case.depth_m
    kz = _kz_of(k1, kr)
    Y = bottom_admittance(case, f, kr, bottom)
    # u(0) = 0, u'(0) = kz (pressure release);  v(0) = cos(kz H) + (Y/kz) sin(kz H)
    # v'(0) = kz sin(kz H) - Y cos(kz H)
    if ice is None:
        return -kz * np.cos(kz * H) - Y * np.sin(kz * H)
    m = ice.top_admittance(2.0 * math.pi * f, kr, case.rho_water_kg_m3)
    # v(0) = cos(kz H) + (Y/kz) sin(kz H); multiply W by kz to keep it regular at kz -> 0
    v0_kz = kz * np.cos(kz * H) + Y * np.sin(kz * H)
    vp0 = kz * np.sin(kz * H) - Y * np.cos(kz * H)
    # W = u(0) v'(0) - v(0) u'(0) = vp0 - m v(0)   (u(0)=1 for the plate branch)
    return kz * vp0 - m * v0_kz


# ----------------------------------------------------------------------------------
# analytic route: roots + numerical residues
# ----------------------------------------------------------------------------------


def analytic_roots(
    case: PekerisCase,
    f: float,
    ice: IcePlate | None = None,
    n_scan: int = 4000,
    bottom: HalfSpaceBottom | FiniteLayerBottom | None = None,
) -> list[float]:
    """Trapped-mode wavenumbers ``k_rm`` from the transcendental dispersion relation.

    The residual is scanned in ``k_z`` over ``0 < k_z < sqrt(k1^2 - k2^2)``; every sign
    change is polished with Brent's method.  ``n_scan`` is the scan density; doubling it
    is the mode-truncation (self-convergence) test.
    """
    kz_max = case.kz_max(f)
    if kz_max <= 0.0:
        return []

    def residual_scaled(kz: float) -> float:
        kr = math.sqrt(max(case.k1(f) ** 2 - kz * kz, 0.0))
        val = dispersion_function(kr, case, f, ice, bottom)
        scale = 1.0
        if ice is not None:
            scale = max(1.0, abs(ice.top_admittance(2.0 * math.pi * f, kr,
                                                    case.rho_water_kg_m3)))
        return float(np.real(val)) / scale

    grid = np.linspace(kz_max * 1e-9, kz_max * (1.0 - 1e-9), n_scan + 1)
    vals = np.array([residual_scaled(kz) for kz in grid])
    kz_roots: list[float] = []
    for i in range(n_scan):
        a, bb = float(grid[i]), float(grid[i + 1])
        if vals[i] == 0.0:
            kz_roots.append(a)
        elif vals[i] * vals[i + 1] < 0.0:
            kz_roots.append(brentq(residual_scaled, a, bb, xtol=1e-16, rtol=8.9e-16,
                                   maxiter=200))
    uniq: list[float] = []
    for r in sorted(kz_roots):
        if r < 1e-7 * kz_max:      # kz -> 0 (k_r -> k1): no vertical structure, not a mode
            continue
        if not uniq or abs(r - uniq[-1]) > 1e-9 * kz_max:
            uniq.append(r)
    return [math.sqrt(max(case.k1(f) ** 2 - kz * kz, 0.0)) for kz in uniq]


def residue_numeric(
    case: PekerisCase,
    f: float,
    kr0: float,
    z: float,
    zs: float,
    ice: IcePlate | None = None,
    bottom: HalfSpaceBottom | FiniteLayerBottom | None = None,
    radius_factor: float = 1e-5,
    n_theta: int = 96,
) -> complex:
    """``Res_{k_r = kr0} [(k_r/k_z) G]`` on a small circle (no hand algebra)."""
    rho = radius_factor * case.k1(f)
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    vals = np.array(
        [
            (k / _kz_of(case.k1(f), k)) * depth_green(k, z, zs, case, f, ice, bottom)
            for k in kr0 + rho * np.exp(1j * theta)
        ]
    )
    # Res = (1/2 pi i) oint f dk = rho * mean(f * exp(i theta))
    return complex(np.mean(vals * np.exp(1j * theta)) * rho)


def tl_from_modes(krs: Sequence[float], residues: Sequence[complex], r: float) -> float:
    """``TL = -20 log10(4 pi |p|)`` for ``p = (i/4) sum_m R_m H_0^{(1)}(k_rm r)``."""
    p = 0.0 + 0.0j
    for kr, res in zip(krs, residues):
        p += res * hankel1(0, kr * r)
    p *= 0.25j
    return -20.0 * math.log10(max(abs(p) * 4.0 * math.pi, 1e-300))


def analytic_tl(
    case: PekerisCase,
    f: float,
    r: float,
    z: float,
    zs: float,
    ice: IcePlate | None = None,
    n_scan: int = 4000,
    max_modes: int | None = None,
    bottom: HalfSpaceBottom | FiniteLayerBottom | None = None,
) -> dict:
    """Analytic (closed-form dispersion + numerical residue) transmission loss."""
    krs = analytic_roots(case, f, ice, n_scan=n_scan, bottom=bottom)
    if max_modes is not None:
        krs = krs[:max_modes]
    residues = [residue_numeric(case, f, kr, z, zs, ice, bottom) for kr in krs]
    tl = tl_from_modes(krs, residues, r) if krs else float("inf")
    return {
        "tl_db": tl,
        "n_modes": len(krs),
        "k_r": krs,
        "residue_abs": [abs(c) for c in residues],
        "residue_phase": [float(np.angle(c)) for c in residues],
    }


def analytic_tl_normalised(
    case: PekerisCase, f: float, r: float, z: float, zs: float, n_scan: int = 4000
) -> dict:
    """Classical Pekeris mode sum with the textbook density normalisation (validation).

    Mode shape ``psi = sin(k_z z)`` in the water and ``sin(k_z H) exp(-gamma (z-H))`` in
    the half-space, normalised by ``int psi^2 / rho dz = 1``:

        p = (i / sqrt(8 pi r)) exp(-i pi/4) sum_m psi(zs) psi(z) exp(i k_rm r)/sqrt(k_rm)

    This is the density-weighted (velocity-potential) form of the two-layer mode sum: the
    normalisation integral carries the ``1/rho`` weight, so the quantity it returns is the
    pressure divided by ``rho_water``.  The two analytic routes are therefore related by an
    exact constant, not by an unknown "source-strength convention" (until m16708 this
    docstring claimed the comparison could only be done on the TL *shape*; the constant is
    now measured and identified, see :func:`cross_route_level_identity`):

        TL_residue - TL_normalised = 20 log10(rho_water / (4 pi)) + eps(r),

    with ``eps`` the error of replacing the exact Hankel function by its far-field
    asymptotic ``sqrt(2/(pi k r)) exp(i(k r - pi/4))`` in this route -- the only
    approximation either route makes.  ``eps`` is bounded by
    :func:`hankel_asymptotic_error_db` and vanishes like ``1/(k r)``.
    """
    krs = analytic_roots(case, f, None, n_scan=n_scan)
    k1 = case.k1(f)
    p = 0.0 + 0.0j
    for kr in krs:
        kz = math.sqrt(max(k1 * k1 - kr * kr, 0.0))
        gam = math.sqrt(max(kr * kr - case.k2(f) ** 2, 0.0))
        sinH = math.sin(kz * case.depth_m)
        norm2 = (case.depth_m / 2.0 - math.sin(2 * kz * case.depth_m) / (4 * kz))
        norm2 /= case.rho_water_kg_m3
        norm2 += sinH ** 2 / (2.0 * gam * case.rho_bottom_kg_m3)
        p += (math.sin(kz * zs) * math.sin(kz * z)) / norm2 * np.exp(1j * kr * r) / math.sqrt(kr)
    p *= 1j / math.sqrt(8.0 * math.pi * r) * np.exp(-0.25j * math.pi)
    return {"tl_db": -20.0 * math.log10(max(abs(p), 1e-300)), "n_modes": len(krs), "k_r": krs}


# ----------------------------------------------------------------------------------
# finite-difference route: independent discretisation of the vertical problem
# ----------------------------------------------------------------------------------


@dataclass
class FDVerticalSolver:
    """2nd-order finite differences for the vertical mode problem.

    Grid ``z_j = j h``, ``j = 0..N``.  Top: ``P_0 = 0`` (pressure release) or the plate
    Robin condition; bottom: the exact half-space impedance relation.  Both are
    eliminated through ghost nodes, keeping the operator tridiagonal.
    """

    case: PekerisCase
    f: float
    ice: IcePlate | None = None
    n_nodes: int = 400

    def __post_init__(self) -> None:
        self.omega = 2.0 * math.pi * self.f
        self.h = self.case.depth_m / self.n_nodes

    def _bands(self, kr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        N, h = self.n_nodes, self.h
        k1 = self.case.k1(self.f)
        kz2 = k1 * k1 - kr * kr
        Y = float(np.real(bottom_admittance(self.case, self.f, kr, None)))
        diag = np.full(N + 1, -2.0 / h ** 2 + kz2)
        sub = np.full(N + 1, 1.0 / h ** 2)
        sup = np.full(N + 1, 1.0 / h ** 2)
        if self.ice is None:
            diag[0], sup[0] = 1.0, 0.0
        else:
            m = float(np.real(self.ice.top_admittance(self.omega, kr,
                                                      self.case.rho_water_kg_m3)))
            diag[0] = -2.0 / h ** 2 - 2.0 * m / h + kz2
            sup[0] = 2.0 / h ** 2
        # bottom ghost: (P_{N+1} - P_{N-1})/(2h) = -Y P_N
        diag[N] = -2.0 / h ** 2 - 2.0 * Y / h + kz2
        sub[N] = 2.0 / h ** 2
        return sub, diag, sup

    def determinant_sign(self, kr: float) -> float:
        """Sign of the tridiagonal determinant from the LU (Thomas) pivots.

        ``det = prod_j denom_j`` with ``denom_0 = diag_0`` and
        ``denom_j = diag_j - sub_j * sup_{j-1} / denom_{j-1}``.  Only the sign is
        accumulated (the magnitude overflows for N >= 400); this is the quantity whose
        sign changes bracket the eigenvalues.  Using the pivots (rather than a two-term
        recurrence initialised with ``D_{-1} = 0``) is essential: the plate boundary
        condition only modifies ``diag_0`` and ``sup_0``, and it enters the determinant
        exactly through ``denom_1 = diag_1 - sub_1 sup_0 / diag_0``.
        """
        sub, diag, sup = self._bands(kr)
        d = diag[0]
        if d == 0.0:
            return 0.0
        sgn = 1.0 if d > 0.0 else -1.0
        cp = sup[0] / d
        for j in range(1, self.n_nodes + 1):
            d = diag[j] - sub[j] * cp
            if d == 0.0:
                return 0.0
            sgn = sgn if d > 0.0 else -sgn
            cp = sup[j] / d
        return sgn

    def roots(self, n_scan: int = 4000) -> list[float]:
        """Modal wavenumbers from sign changes of the discretised determinant."""
        k1, k2 = self.case.k1(self.f), self.case.k2(self.f)
        if k1 <= k2:
            return []
        grid = np.linspace(k2 * (1 + 1e-9), k1 * (1 - 1e-9), n_scan + 1)
        vals = np.array([self.determinant_sign(float(k)) for k in grid])
        out: list[float] = []
        for i in range(n_scan):
            if vals[i] * vals[i + 1] < 0.0:
                out.append(
                    float(
                        brentq(self.determinant_sign, float(grid[i]), float(grid[i + 1]),
                               xtol=1e-17, rtol=8.9e-16, maxiter=200)
                    )
                )
        return out

    def green_raw(self, kr: float, z: float, zs: float) -> float:
        """Discretised vertical Green function (the ``2 k_z`` delta factor is applied)."""
        N, h = self.n_nodes, self.h
        sub, diag, sup = self._bands(kr)
        k1 = self.case.k1(self.f)
        kz = math.sqrt(max(k1 * k1 - kr * kr, 0.0))
        js = max(0, min(N, int(round(zs / h))))
        rhs = np.zeros(N + 1)
        rhs[js] = 2.0 * kz / h
        cp = np.zeros(N + 1)
        dp = np.zeros(N + 1)
        cp[0] = sup[0] / diag[0]
        dp[0] = rhs[0] / diag[0]
        for j in range(1, N + 1):
            denom = diag[j] - sub[j] * cp[j - 1]
            if j < N:
                cp[j] = sup[j] / denom
            dp[j] = (rhs[j] - sub[j] * dp[j - 1]) / denom
        x = np.zeros(N + 1)
        x[N] = dp[N]
        for j in range(N - 1, -1, -1):
            x[j] = dp[j] - cp[j] * x[j + 1]
        jr = max(0, min(N, int(round(z / h))))
        return float(x[jr])


def fd_tl(
    case: PekerisCase,
    f: float,
    r: float,
    z: float,
    zs: float,
    ice: IcePlate | None = None,
    n_nodes: int = 400,
    n_scan: int = 4000,
    delta_factor: float = 1e-6,
    max_modes: int | None = None,
) -> dict:
    """Transmission loss from the finite-difference route."""
    solver = FDVerticalSolver(case, f, ice, n_nodes=n_nodes)
    krs = solver.roots(n_scan=n_scan)
    if max_modes is not None:
        krs = krs[:max_modes]
    krs_use: list[float] = []
    res: list[complex] = []
    delta = delta_factor * case.k1(f)
    for kr in krs:
        g = solver.green_raw(kr + delta, z, zs)
        # (k_r/k_z) G has a simple pole at kr; the k_z ratio is smooth there
        kz = math.sqrt(max(case.k1(f) ** 2 - kr * kr, 0.0))
        kz_p = math.sqrt(max(case.k1(f) ** 2 - (kr + delta) ** 2, 0.0))
        # G is purely imaginary for real k_r (delta strength 2 i k_z); the FD solve
        # returns G/i, hence the 1j.  (k_r/k_z) is smooth at the pole.
        res.append(1j * delta * ((kr + delta) / kz_p) * g)
        krs_use.append(kr)
    tl = tl_from_modes(krs_use, res, r) if krs_use else float("inf")
    return {"tl_db": tl, "n_modes": len(krs_use), "k_r": krs_use}


# ----------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------


def phase_velocity(f: float, kr: float) -> float:
    return 2.0 * math.pi * f / kr


def grazing_angle_deg(case: PekerisCase, f: float, kr: float) -> float:
    s = min(1.0, max(-1.0, kr / case.k1(f)))
    return math.degrees(math.acos(s))


def hankel_asymptotic_error_db(kr: float, r: float) -> float:
    """Error of ``sqrt(2/(pi k r)) exp(i(kr - pi/4))`` against the exact Hankel [dB]."""
    exact = abs(hankel1(0, kr * r))
    asym = math.sqrt(2.0 / (math.pi * kr * r))
    return abs(20.0 * math.log10(exact / asym))

# ----------------------------------------------------------------------------------
# absolute-level cross-calibration of the two analytic routes (m16708)
# ----------------------------------------------------------------------------------


def predicted_route_offset_db(case: PekerisCase) -> float:
    """Constant that separates the residue route from the density-normalised route [dB].

    ``20 log10(rho_water / (4 pi))``: the ``4 pi`` is the source-strength convention the
    residue route inherits from the free-space Green function ``exp(i k R)/(4 pi R)`` and
    the ``rho_water`` is the density weight of the textbook normalisation
    ``int psi^2/rho dz = 1`` (a velocity-potential expansion), i.e.
    ``p_residue = p_normalised / rho_water``.
    """
    return 20.0 * math.log10(case.rho_water_kg_m3 / (4.0 * math.pi))


def cross_route_level_identity(
    case: PekerisCase,
    f: float,
    r: float,
    z: float,
    zs: float,
    ice: IcePlate | None = None,
    bottom: HalfSpaceBottom | FiniteLayerBottom | None = None,
    n_scan: int = 4000,
) -> dict:
    """Compare the residue route with the density-normalised mode sum at one point.

    The two analytic routes are independent in everything that can go wrong in a BEM
    amplitude: one takes the residue of the spectral depth Green function (no hand-derived
    mode normalisation, works for the non-self-adjoint plate top), the other uses the
    textbook ``int psi^2/rho dz = 1`` normalisation and the far-field asymptotic.  They
    differ by exactly ``predicted_route_offset_db`` plus the asymptotic residual ``eps``.

    ``residue_minus_prediction_db`` is the measured deviation from the predicted constant;
    ``hankel_bound_db`` is the largest ``hankel_asymptotic_error_db`` over the modes and is
    the scale ``eps`` is expected to take (equal to it in the single-mode regime).
    """
    res = analytic_tl(case, f, r, z, zs, ice=ice, bottom=bottom, n_scan=n_scan)
    norm = analytic_tl_normalised(case, f, r, z, zs, n_scan=n_scan)
    offset = res["tl_db"] - norm["tl_db"]
    predicted = predicted_route_offset_db(case)
    krs = norm["k_r"]
    bound = max((hankel_asymptotic_error_db(kr, r) for kr in krs), default=0.0)
    return {
        "f_hz": f,
        "r_m": r,
        "z_m": z,
        "source_depth_m": zs,
        "n_modes": norm["n_modes"],
        "k_r": krs,
        "tl_residue_db": res["tl_db"],
        "tl_normalised_db": norm["tl_db"],
        "offset_db": offset,
        "predicted_offset_db": predicted,
        "residue_minus_prediction_db": offset - predicted,
        "hankel_bound_db": bound,
    }


def rigid_wall_roots(case: PekerisCase, f: float) -> list[float]:
    """Closed-form mode wavenumbers of the ``rho_bottom -> inf`` (rigid bottom) limit.

    With a rigid bottom the interface condition ``dP/dz = 0`` replaces the half-space
    impedance, ``tan(k_z H) -> inf``, so ``k_z,m H = (m + 1/2) pi`` and

        k_r,m = sqrt(k1^2 - ((m + 1/2) pi / H)^2),   m = 0, 1, ...

    The trapping cutoff is *not* ``k_z = k1``: the fluid half-space still exists in this
    limit (only its density becomes infinite, its sound speed does not), so a mode is
    trapped only while ``k_z,m < kz_max`` with ``kz_max = sqrt(k1^2 - k2^2)``.  Using ``k1``
    as the cutoff would invent eight spurious modes at 100 Hz for the default case.

    This needs no root finding and no transcendental equation, so it is an external
    reference for the dispersion solver: ``analytic_roots`` at a very large ``rho_bottom``
    must approach these values, and the deviation must vanish like ``rho_w/rho_bottom``
    (the ``1/rho_b`` weight of the interface term in ``tan(k_z H) = -(rho_b/rho_w) k_z/gamma``).
    """
    k1 = case.k1(f)
    kz_max = case.kz_max(f)
    out: list[float] = []
    m = 0
    while True:
        kz = (m + 0.5) * math.pi / case.depth_m
        if kz >= kz_max:
            break
        out.append(math.sqrt(k1 * k1 - kz * kz))
        m += 1
    return out
